import argparse
import os
import random

import torch
from torch.utils.data import DataLoader

from models.BraskModel import BraskModel
from training.config import (
    BATCH_SIZE, CHECKPOINTS_DIR, EARLY_STOP_PATIENCE_STAGES,
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
    p.add_argument("--early_stop_patience_stage_1", type=int,   default=EARLY_STOP_PATIENCE_STAGES[0])
    p.add_argument("--early_stop_patience_stage_2", type=int,   default=EARLY_STOP_PATIENCE_STAGES[1])
    p.add_argument("--early_stop_patience_stage_3", type=int,   default=EARLY_STOP_PATIENCE_STAGES[2])
    return p.parse_args()


def _save_resume(path, model, optimizer, scaler, epoch, best_val_loss, no_improve, done=False):
    torch.save({
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scaler":        scaler.state_dict(),
        "epoch":         epoch,
        "best_val_loss": best_val_loss,
        "no_improve":    no_improve,
        "done":          done,
    }, path)


def _load_resume(path, model, optimizer, scaler):
    """Load resume checkpoint in-place. Returns (done, start_epoch, best_val_loss, no_improve)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("done", False), ckpt["epoch"] + 1, ckpt["best_val_loss"], ckpt["no_improve"]


def main():
    args = parse_args()
    batch_size          = args.batch_size
    stage1_epochs       = args.stage1_epochs
    stage2_epochs       = args.stage2_epochs
    stage3_epochs       = args.stage3_epochs
    val_split           = args.val_split
    early_stop_patience_stage_1 = args.early_stop_patience_stage_1
    early_stop_patience_stage_2 = args.early_stop_patience_stage_2
    early_stop_patience_stage_3 = args.early_stop_patience_stage_3

    ckpt_best   = {s: os.path.join(CHECKPOINTS_DIR, f"brask_stage{s}_best.pt")   for s in (1, 2, 3)}
    ckpt_resume = {s: os.path.join(CHECKPOINTS_DIR, f"brask_stage{s}_resume.pt") for s in (1, 2, 3)}

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
    if not os.path.exists(os.path.join(CHECKPOINTS_DIR, "brask_init.pt")):
        torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, "brask_init.pt"))

    # ════════════════════════════════════════
    # STAGE 1 — Entity extractors only
    # ════════════════════════════════════════
    print("\n── Stage 1: Entity extractors ──")
    set_stage(model, stage=1)
    optimizer     = get_optimizer(model, stage=1)
    best_val_loss = float("inf")
    no_improve    = 0
    start_epoch   = 0

    if os.path.exists(ckpt_resume[1]):
        done, start_epoch, best_val_loss, no_improve = _load_resume(
            ckpt_resume[1], model, optimizer, scaler
        )
        if done:
            print(f"  Stage 1 already complete — loading best checkpoint.")
            model.load_state_dict(torch.load(ckpt_best[1], map_location=device))
        else:
            print(f"  Resuming Stage 1 from epoch {start_epoch} (best_val={best_val_loss:.4f})")
    else:
        done = False

    if not done:
        for epoch in range(start_epoch, stage1_epochs):
            train_loss = run_epoch_stage1(model, train_loader, optimizer, scaler)
            val_loss   = evaluate(model, val_loader, rel2idx, all_rel_ids, semantic_rel_emb, transe_rel_emb, stage=1)
            print(f"  [S1] Epoch {epoch+1}/{stage1_epochs} — train: {train_loss:.4f}  val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve    = 0
                torch.save(model.state_dict(), ckpt_best[1])
                print(f"    ✓ Saved best (val={best_val_loss:.4f})")
            else:
                no_improve += 1

            early_stopped = no_improve >= early_stop_patience_stage_1
            _save_resume(ckpt_resume[1], model, optimizer, scaler, epoch, best_val_loss, no_improve,
                         done=(epoch == stage1_epochs - 1 or early_stopped))
            if early_stopped:
                print(f"  Early stopping after {early_stop_patience_stage_1} epochs without improvement.")
                break

        model.load_state_dict(torch.load(ckpt_best[1], map_location=device))

    # ════════════════════════════════════════
    # STAGE 2 — Full model, teacher forcing = 1.0
    # ════════════════════════════════════════
    print("\n── Stage 2: Full model (teacher forcing = 1.0) ──")
    set_stage(model, stage=2)
    optimizer     = get_optimizer(model, stage=2)
    best_val_loss = float("inf")
    no_improve    = 0
    start_epoch   = 0

    if os.path.exists(ckpt_resume[2]):
        done, start_epoch, best_val_loss, no_improve = _load_resume(
            ckpt_resume[2], model, optimizer, scaler
        )
        if done:
            print(f"  Stage 2 already complete — loading best checkpoint.")
            model.load_state_dict(torch.load(ckpt_best[2], map_location=device))
        else:
            print(f"  Resuming Stage 2 from epoch {start_epoch} (best_val={best_val_loss:.4f})")
    else:
        done = False

    if not done:
        for epoch in range(start_epoch, stage2_epochs):
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
                torch.save(model.state_dict(), ckpt_best[2])
                print(f"    ✓ Saved best (val={best_val_loss:.4f})")
            else:
                no_improve += 1

            early_stopped = no_improve >= early_stop_patience_stage_2
            _save_resume(ckpt_resume[2], model, optimizer, scaler, epoch, best_val_loss, no_improve,
                         done=(epoch == stage2_epochs - 1 or early_stopped))
            if early_stopped:
                print(f"  Early stopping after {early_stop_patience_stage_2} epochs without improvement.")
                break

        model.load_state_dict(torch.load(ckpt_best[2], map_location=device))

    # ════════════════════════════════════════
    # STAGE 3 — Full model, teacher forcing decays 1.0 → 0.0
    # ════════════════════════════════════════
    print("\n── Stage 3: Full model (teacher forcing decay) ──")
    set_stage(model, stage=3)
    optimizer     = get_optimizer(model, stage=3)
    best_val_loss = float("inf")
    no_improve    = 0
    start_epoch   = 0

    if os.path.exists(ckpt_resume[3]):
        done, start_epoch, best_val_loss, no_improve = _load_resume(
            ckpt_resume[3], model, optimizer, scaler
        )
        if done:
            print(f"  Stage 3 already complete — loading best checkpoint.")
            model.load_state_dict(torch.load(ckpt_best[3], map_location=device))
        else:
            print(f"  Resuming Stage 3 from epoch {start_epoch} (best_val={best_val_loss:.4f})")
    else:
        done = False

    if not done:
        for epoch in range(start_epoch, stage3_epochs):
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
                torch.save(model.state_dict(), ckpt_best[3])
                print(f"    ✓ Saved best (val={best_val_loss:.4f})")
            else:
                no_improve += 1

            early_stopped = no_improve >= early_stop_patience_stage_3
            _save_resume(ckpt_resume[3], model, optimizer, scaler, epoch, best_val_loss, no_improve,
                         done=(epoch == stage3_epochs - 1 or early_stopped))
            if early_stopped:
                print(f"  Early stopping after {early_stop_patience_stage_3} epochs without improvement.")
                break

    print("\nTraining complete.")
    print(f"Best checkpoints saved in {CHECKPOINTS_DIR}")


if __name__ == "__main__":
    main()
