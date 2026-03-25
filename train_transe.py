import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from utils.files import cache_array, save_tensor
from utils.pre_processed_data import data_loader, check_minimized_files
from utils.settings import settings
from TransE import TransEDataset, TransEModel 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
NUM_WORKERS = 4 if use_cuda else 0
BATCH_SIZE = 512 if use_cuda else 256 



MARGIN = 1.0
TRANSE_EMB_DIM = 100
LEARNING_RATE = 1e-3
NUM_EPOCHS = 90


def main():
    """Trains TransE and saves the resulting relation embeddings.

    Detects whether to use Distributed Data Parallel by checking for the
    ``LOCAL_RANK`` environment variable (set automatically by ``torchrun``).
    Falls back to single-GPU or CPU training when ``LOCAL_RANK`` is absent.

    Training procedure:
        1. Builds ``TransEDataset`` from the training triples.
        2. Initialises ``TransEModel`` and optionally wraps it with DDP.
        3. Runs ``NUM_EPOCHS`` epochs with Adam + CosineAnnealingLR.
        4. After each batch, L1-normalises entity and relation embeddings.
        5. On completion (rank 0 only), saves the relation embedding matrix
           to ``transe_rel_embs.npz`` inside the minimized or preprocessed
           data directory depending on ``MINIMIZED``.

    Output file shape: ``(n_rels, TRANSE_EMB_DIM)``, accessible via
    ``np.load(path)["arr"]``.
    """
    use_minimized = True
    if not check_minimized_files():
        return



    local_rank_str = os.environ.get("LOCAL_RANK")
    use_ddp = local_rank_str is not None

    if use_ddp:
        local_rank = int(local_rank_str)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        print(f"Using DDP. LOCAL RANK: {local_rank}")
    else:
        local_rank = 0
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Running on {device} (no DDP)")

    triples = data_loader.get_triples_train(minimized=True)
    dataset = TransEDataset(triples=triples)
    print(f"Dataset: {len(dataset):,} triples | {dataset.n_ents:,} entities | {dataset.n_rels:,} relations")

    if use_ddp:
        sampler = DistributedSampler(dataset)
        loader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")

    model = TransEModel(dataset.n_ents, dataset.n_rels, emb_dim=TRANSE_EMB_DIM).to(device)
    if use_ddp:
        model = torch.compile(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    if device.type == "cuda":
        autocast_ctx = lambda: autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = lambda: autocast(device_type="cpu", dtype=torch.bfloat16)

    def get_core_model():
        return model.module if use_ddp else model

    print(f"Starting training on {device}, use DDP: {use_ddp}, batch size: {BATCH_SIZE}, epochs: {NUM_EPOCHS}, learning rate: {LEARNING_RATE}")

    for epoch in tqdm(range(NUM_EPOCHS), total=NUM_EPOCHS, desc=f"Epochs [rank {local_rank}]"):
        model.train()
        if use_ddp:
            sampler.set_epoch(epoch)
        total_loss = 0.0
        for pos_batch, neg_batch in tqdm(loader, desc=f"Epoch {epoch + 1} [rank {local_rank}]", leave=False):
            pos_batch, neg_batch = pos_batch.to(device), neg_batch.to(device)

            optimizer.zero_grad()
            with autocast_ctx():
                pos_dist, neg_dist = model(pos_batch, neg_batch)
                loss = torch.clamp(MARGIN + pos_dist - neg_dist, min=0).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            core = get_core_model()
            core.ent_embs.weight.data = F.normalize(core.ent_embs.weight.data, p=2, dim=1)

            total_loss += loss.item()
        print(f"[rank {local_rank}] Epoch {epoch + 1} Loss: {total_loss:.4f}")
        scheduler.step()

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

    if local_rank == 0:
        out_path = settings.MINIMIZED_FILES.TRANSE_MODEL_RESULTS if use_minimized else settings.PREPROCESSED_FILES.TRANSE_MODEL_RESULTS
        rel2idx_out = settings.MINIMIZED_FILES.REL2IDX if use_minimized else settings.PREPROCESSED_FILES.REL2IDX
        save_tensor(get_core_model().rel_embs.weight.data, out_path)
        cache_array(dataset.rel2idx, rel2idx_out)
        print(f"Shape: {tuple(get_core_model().rel_embs.weight.data.shape)}  →  {out_path}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
