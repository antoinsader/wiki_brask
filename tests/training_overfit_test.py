

import os
import torch
import sys
from tqdm import tqdm
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


batch_size      = 4
stage1_epochs   = 100
stage2_epochs   = 100
stage3_epochs   = 100   # teacher forcing decays over these
val_split       = 0.1

print(f"Reading pickles...");
rel2idx = read_cached_array("../data/minimized/rel2idx.pkl")
description_embs_ids = read_cached_array("../data/minimized/description_embeddings_ids.pkl")
golden_triples        = read_cached_array("../data/minimized/golden_triples.pkl")


description_embs_masks = read_tensor("../data/minimized/description_embeddings_all_masks.npy", mmap=True)
description_embs_mean  = read_tensor("../data/minimized/description_embeddings_mean.npy",      mmap=True)
semantic_rel_emb      = read_tensor("../data/minimized/relation_embeddings.npy").to(device)  # (R, H)
transe_rel_emb      = read_tensor("../data/minimized/transe_rel_embs.npy").to(device)  # (R, E)
description_embs_all = read_tensor("../data/minimized/description_embeddings_all.npy", mmap=True)



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
for epoch in range(200):
    fwd_start, fwd_end = model.fwd_head_predictor(X)
    bwd_start, bwd_end = model.bwd_tail_predictor(X)
    gold_fhs, gold_fhe, gold_bts, gold_bte = build_gold_entity_labels(golden_triples, mask)


    loss = stage1_loss(
        fwd_start.squeeze(-1), fwd_end.squeeze(-1),
        bwd_start.squeeze(-1), bwd_end.squeeze(-1),
        gold_fhs, gold_fhe, gold_bts, gold_bte, mask, pos_weight=30.0
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1}: loss={loss.item():.4f}")

# # Expected: loss should drop from ~log(2)*pos_weight toward near 0
print(f"Overfitting stage  2")
set_stage(model, stage=2)
optimizer = get_optimizer(model, stage=2)
teacher_forcing_ratio = 1.0
# best_val_loss = float("inf")



for epoch in range(200):
    model.train()
    total_loss, n_batches = 0.0, 0
    outputs = model(
        X,
        X_mean,
        mask,
        golden_triples,
        teacher_forcing_ratio,
        semantic_rel_emb,
        transe_rel_emb
    )
    unique_subjects_batch = outputs[MODEL_OUTPUT_KEYS["unique_subjects_batch"]]
    gold_fhs, gold_fhe, gold_bts, gold_bte = build_gold_entity_labels(golden_triples, mask)

    gold_fts, gold_fte, gold_bhs, gold_bhe = build_gold_tail_labels(
        triples_batch= golden_triples,
        unique_subjects_batch =unique_subjects_batch ,
        mask=mask,
        num_relations=num_relations,
        rel2idx=rel2idx
    )

    gold_labels = {
        "fwd_head_start": gold_fhs,
        "fwd_head_end": gold_fhe,
        "bwd_tail_start": gold_bts,
        "bwd_tail_end": gold_bte,
        "fwd_tail_start": gold_fts,
        "fwd_tail_end": gold_fte,
        "bwd_head_start": gold_bhs,
        "bwd_head_end": gold_bhe,
    }


    loss, components = brask_loss(
        outputs, 
        gold_labels, 
        mask, 
        pos_weight_entity=1.0, 
        pos_weight_obj=1.0)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()


    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1}: loss={loss.item():.4f}, components={components}")

# batch = next(iter(train_loader))
# mask  = batch[BraskDataset.BATCH_KEYS["EMBS_MASKS"]].to(device)
# gt    = batch[BraskDataset.BATCH_KEYS["GOLDEN_TRIPLES"]]
# gold_fhs, gold_fhe, gold_bts, gold_bte = build_gold_entity_labels(gt, mask)

# print(f"head start positives: {gold_fhs.sum().item()}, SHOULD BE BIGGER THAN 0 ")
# print(f"head end positives:   {gold_fhe.sum().item()}, SHOULD BE BIGGER THAN 0 ")
# print(f"tail start positives: {gold_bts.sum().item()}, SHOULD BE BIGGER THAN 0 ")
# print(f"tail end positives:   {gold_bte.sum().item()}, SHOULD BE BIGGER THAN 0 ")

