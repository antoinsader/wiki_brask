

import os
import torch
import sys
sys.path.insert(0, '..')
from utils.files import read_cached_array, read_tensor

from torch.utils.data import Dataset, DataLoader
from train import (
    BraskModel, collate_fn, set_stage, get_optimizer,
    build_gold_entity_labels, build_gold_tail_labels,
    build_sk_from_gold, stage1_loss, brask_loss,
    MODEL_OUTPUT_KEYS, BraskDataset
)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
NUM_WORKERS = 4 if use_cuda else 0


def make_dataset(ids):
    id_to_idx = {id_: i for i, id_ in enumerate(description_embs_ids)}
    idx = [id_to_idx[i] for i in ids]
    return BraskDataset(
        description_embs=description_embs_all[idx],
        description_embs_ids=ids,
        description_embs_masks=description_embs_masks[idx],
        description_mean_embs=description_embs_mean[idx],
        golden_triples=golden_triples
    )


batch_size      = 16
stage1_epochs   = 100
stage2_epochs   = 100
stage3_epochs   = 100   # teacher forcing decays over these
val_split       = 0.1

rel2idx = read_cached_array("../data/minimized/rel2idx.pkl")
description_embs_all = read_tensor("../data/minimized/description_embeddings_all.npy", mmap=True)
description_embs_ids = read_cached_array("../data/minimized/description_embeddings_ids.pkl")
description_embs_masks = read_tensor("../data/minimized/description_embeddings_all_masks.npy")
description_embs_mean = read_tensor("../data/minimized/description_embeddings_mean.npy")
golden_triples        = read_cached_array("../data/minimized/golden_triples.pkl")



semantic_rel_emb      = read_tensor("../data/minimized/relation_embeddings.npy").to(device)  # (R, H)
transe_rel_emb      = read_tensor("../data/minimized/transe_rel_embs.npy").to(device)  # (R, E)

assert semantic_rel_emb.shape[0] == transe_rel_emb.shape[0] == len(rel2idx), f"Number of relations mismatch between semantic and transe embeddings and rel2idx: {semantic_rel_emb.shape[0]} vs {transe_rel_emb.shape[0]} vs {len(rel2idx)}"

transe_rel_dim = transe_rel_emb.shape[1]
num_relations = semantic_rel_emb.shape[0]

max_length = description_embs_all.shape[1]
H          = description_embs_all.shape[2]
N     = len(description_embs_ids)

n_val = int(N * val_split)
ids_train = description_embs_ids[n_val:]
ids_val   = description_embs_ids[:n_val]



train_loader = DataLoader(make_dataset(ids_train), batch_size=batch_size,
                            shuffle=True,  collate_fn=collate_fn, num_workers=NUM_WORKERS)

model = BraskModel(hidden_dim=H, transe_rel_dim=transe_rel_dim).to(device)

# Get one real batch
batch = next(iter(train_loader))
X       = batch[BraskDataset.BATCH_KEYS["EMBS"]].to(device)
X_mean  = batch[BraskDataset.BATCH_KEYS["MEAN_EMBS"]].to(device)
mask    = batch[BraskDataset.BATCH_KEYS["EMBS_MASKS"]].to(device)
golden_triples = batch[BraskDataset.BATCH_KEYS["GOLDEN_TRIPLES"]]

set_stage(model, stage=1)
optimizer = get_optimizer(model, stage=1)

print("Overfitting single batch — stage 1...")
for epoch in range(100):
    fwd_start, fwd_end = model.fwd_head_predictor(X)
    bwd_start, bwd_end = model.bwd_tail_predictor(X)
    gold_fhs, gold_fhe, gold_bts, gold_bte = build_gold_entity_labels(golden_triples, mask)
    loss = stage1_loss(
        fwd_start.squeeze(-1), fwd_end.squeeze(-1),
        bwd_start.squeeze(-1), bwd_end.squeeze(-1),
        gold_fhs, gold_fhe, gold_bts, gold_bte, mask
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1}: loss={loss.item():.4f}")

# Expected: loss should drop from ~log(2)*pos_weight toward near 0
