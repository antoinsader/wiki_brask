import argparse
import os
import random

import torch
from torch.utils.data import DataLoader

from models.BraskModel import BraskModel
from training.config import (
    BATCH_SIZE, CHECKPOINTS_DIR, EARLY_STOP_PATIENCE,
    NUM_WORKERS, STAGE1_EPOCHS, STAGE2_EPOCHS, STAGE3_EPOCHS, VAL_SPLIT, device, use_cuda,
)
from training.dataset import BraskDataset, collate_fn
from training.loops import evaluate, get_optimizer, run_epoch_stage1, run_epoch_stage_2, set_stage
from utils.pre_processed_data import data_loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size",          type=int,   default=BATCH_SIZE)
    p.add_argument("--stage1-epochs",       type=int,   default=STAGE1_EPOCHS)
    p.add_argument("--stage2-epochs",       type=int,   default=STAGE2_EPOCHS)
    p.add_argument("--stage3-epochs",       type=int,   default=STAGE3_EPOCHS)
    p.add_argument("--val-split",           type=float, default=VAL_SPLIT)
    p.add_argument("--early-stop-patience", type=int,   default=EARLY_STOP_PATIENCE)
    return p.parse_args()


def main():
    args = parse_args()
    batch_size          = args.batch_size
    stage1_epochs       = args.stage1_epochs
    stage2_epochs       = args.stage2_epochs
    stage3_epochs       = args.stage3_epochs
    val_split           = args.val_split
    early_stop_patience = args.early_stop_patience

    ckpt_stage1 = os.path.join(CHECKPOINTS_DIR, "brask_stage1_best.pt")
    ckpt_stage2 = os.path.join(CHECKPOINTS_DIR, "brask_stage2_best.pt")
    ckpt_stage3 = os.path.join(CHECKPOINTS_DIR, "brask_stage3_best.pt")

    rel2idx          = data_loader.get_rel2idx(minimized=True)
    embs_all, emb_ids, embs_masks = data_loader.get_description_embeddings_all()
    embs_mean        = data_loader.get_description_embeddings_mean()
    golden_triples   = data_loader.get_golden_triples()
    semantic_rel_emb = data_loader.get_semantic_relation_embeddings().to(device)
    transe_rel_emb   = data_loader.get_trane_relation_embeddings().to(device)

    assert semantic_rel_emb.shape[0] == transe_rel_emb.shape[0] == len(rel2idx), (
        f"Relations mismatch: semantic {semantic_rel_emb.shape[0]}, "
        f"transe {transe_rel_emb.shape[0]}, rel2idx {len(rel2idx)}"
    )

    transe_rel_dim = transe_rel_emb.shape[1]
    H              = embs_all.shape[2]
    all_rel_ids    = list(rel2idx.keys())

    # Shuffle before splitting to avoid ordering bias (embeddings sorted by description length)
    shuffled_ids = list(emb_ids)
    random.seed(42)
    random.shuffle(shuffled_ids)
    n_val     = int(len(shuffled_ids) * val_split)
    ids_val   = shuffled_ids[:n_val]
    ids_train = shuffled_ids[n_val:]

    id_to_idx = {id_: i for i, id_ in enumerate(emb_ids)}

    def make_dataset(ids):
        idx = [id_to_idx[i] for i in ids]
        return BraskDataset(
            description_embs=embs_all[idx],
            description_embs_ids=ids,
            description_embs_masks=embs_masks[idx],
            description_mean_embs=embs_mean[idx],
            golden_triples=golden_triples,
        )

    train_loader = DataLoader(
        make_dataset(ids_train), batch_size=batch_size,
        shuffle=True,  collate_fn=collate_fn, num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        make_dataset(ids_val),   batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS,
    )

    model  = BraskModel(hidden_dim=H, transe_rel_dim=transe_rel_dim).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, "brask_init.pt"))

    # ════════════════════════════════════════
    # STAGE 1 — Entity extractors only
    # ════════════════════════════════════════
    print("\n── Stage 1: Entity extractors ──")
    set_stage(model, stage=1)
    optimizer     = get_optimizer(model, stage=1)
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(stage1_epochs):
        train_loss = run_epoch_stage1(model, train_loader, optimizer, scaler)
        val_loss   = evaluate(model, val_loader, rel2idx, all_rel_ids, semantic_rel_emb, transe_rel_emb, stage=1)
        print(f"  [S1] Epoch {epoch+1}/{stage1_epochs} — train: {train_loss:.4f}  val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), ckpt_stage1)
            print(f"    ✓ Saved (val={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"  Early stopping after {early_stop_patience} epochs without improvement.")
                break

    # ════════════════════════════════════════
    # STAGE 2 — Full model, teacher forcing = 1.0
    # ════════════════════════════════════════
    print("\n── Stage 2: Full model (teacher forcing = 1.0) ──")
    model.load_state_dict(torch.load(ckpt_stage1, map_location=device))
    set_stage(model, stage=2)
    optimizer     = get_optimizer(model, stage=2)
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(stage2_epochs):
        train_loss = run_epoch_stage_2(
            model, train_loader, optimizer, scaler, rel2idx, all_rel_ids,
            teacher_forcing_ratio=1.0,
            semantic_rel_emb=semantic_rel_emb, transe_rel_emb=transe_rel_emb,
        )
        val_loss = evaluate(model, val_loader, rel2idx, all_rel_ids, semantic_rel_emb, transe_rel_emb, stage=2)
        print(f"  [S2] Epoch {epoch+1}/{stage2_epochs} — train: {train_loss:.4f}  val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), ckpt_stage2)
            print(f"    ✓ Saved (val={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"  Early stopping after {early_stop_patience} epochs without improvement.")
                break

    # ════════════════════════════════════════
    # STAGE 3 — Full model, teacher forcing decays 1.0 → 0.0
    # ════════════════════════════════════════
    print("\n── Stage 3: Full model (teacher forcing decay) ──")
    model.load_state_dict(torch.load(ckpt_stage2, map_location=device))
    set_stage(model, stage=3)
    optimizer     = get_optimizer(model, stage=3)
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(stage3_epochs):
        tf_ratio   = max(0.0, 1.0 - epoch / stage3_epochs)
        train_loss = run_epoch_stage_2(
            model, train_loader, optimizer, scaler, rel2idx, all_rel_ids,
            teacher_forcing_ratio=tf_ratio,
            semantic_rel_emb=semantic_rel_emb, transe_rel_emb=transe_rel_emb,
        )
        val_loss = evaluate(model, val_loader, rel2idx, all_rel_ids, semantic_rel_emb, transe_rel_emb, stage=3)
        print(f"  [S3] Epoch {epoch+1}/{stage3_epochs}  tf={tf_ratio:.2f} — "
              f"train: {train_loss:.4f}  val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), ckpt_stage3)
            print(f"    ✓ Saved (val={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"  Early stopping after {early_stop_patience} epochs without improvement.")
                break

    print("\nTraining complete.")
    print(f"Best checkpoints saved in {CHECKPOINTS_DIR}")


if __name__ == "__main__":
    main()
