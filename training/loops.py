import os

import torch
from tqdm import tqdm

from models.BraskModel import BraskModel
from training.config import (
    CHECKPOINTS_DIR,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE_STAGE_1,
    LEARNING_RATE_STAGE_2,
    MODEL_OUTPUT_KEYS,
    device,
    use_cuda,
)
from training.dataset import BraskDataset
from training.labels import build_gold_entity_labels, build_gold_tail_labels, sample_active_relations
from training.loss import brask_loss, stage1_loss


def set_stage(model: BraskModel, stage: int) -> None:
    """Freeze/unfreeze model parameters per training stage."""
    for param in model.parameters():
        param.requires_grad = False
    if stage == 1:
        for param in model.fwd_head_predictor.parameters():
            param.requires_grad = True
        for param in model.bwd_tail_predictor.parameters():
            param.requires_grad = True
    elif stage in (2, 3):
        for param in model.parameters():
            param.requires_grad = True


def get_optimizer(model: BraskModel, stage: int) -> torch.optim.Optimizer:
    """Return an Adam optimizer with the stage-appropriate learning rate."""
    lr = LEARNING_RATE_STAGE_1 if stage == 1 else LEARNING_RATE_STAGE_2
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=lr)


def run_epoch_stage1(model: BraskModel, dataloader, optimizer, scaler,
                     grad_accum_steps: int = GRAD_ACCUM_STEPS) -> float:
    K = BraskDataset.BATCH_KEYS
    model.train()
    total_loss, n_batches = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(dataloader, desc="Stage 1 epoch")):
        X              = batch[K["EMBS"]].to(device)
        mask           = batch[K["EMBS_MASKS"]].to(device)
        golden_triples = batch[K["GOLDEN_TRIPLES"]]

        gold_fhs, gold_fhe, gold_bts, gold_bte = build_gold_entity_labels(golden_triples, mask)

        with torch.amp.autocast("cuda", enabled=use_cuda):
            fwd_start, fwd_end = model.fwd_head_predictor(X)
            bwd_start, bwd_end = model.bwd_tail_predictor(X)
            loss = stage1_loss(
                fwd_head_start_logits=fwd_start.squeeze(-1),
                fwd_head_end_logits=fwd_end.squeeze(-1),
                bwd_tail_start_logits=bwd_start.squeeze(-1),
                bwd_tail_end_logits=bwd_end.squeeze(-1),
                golden_head_start_labels=gold_fhs,
                golden_head_end_labels=gold_fhe,
                golden_tail_start_labels=gold_bts,
                golden_tail_end_labels=gold_bte,
                token_mask=mask,
            )
            loss = loss / grad_accum_steps

        if torch.isnan(loss):
            print("  NaN loss detected — stopping")
            break

        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum_steps
        n_batches  += 1

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    # flush any remaining accumulated gradients
    if n_batches % grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / (n_batches + 1e-8)


def run_epoch_stage_2(
    model: BraskModel,
    dataloader,
    optimizer,
    scaler,
    rel2idx: dict,
    all_rel_ids: list,
    teacher_forcing_ratio: float,
    semantic_rel_emb: torch.Tensor,
    transe_rel_emb: torch.Tensor,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
) -> float:
    K   = BraskDataset.BATCH_KEYS
    MOK = MODEL_OUTPUT_KEYS
    model.train()
    total_loss, n_batches = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(dataloader, desc=f"Stage 2 (tf={teacher_forcing_ratio:.2f})")):
        X              = batch[K["EMBS"]].to(device)
        X_mean         = batch[K["MEAN_EMBS"]].to(device)
        mask           = batch[K["EMBS_MASKS"]].to(device)
        golden_triples = batch[K["GOLDEN_TRIPLES"]]

        active_indices, rel_to_slot = sample_active_relations(golden_triples, rel2idx, all_rel_ids)
        active_semantic = semantic_rel_emb[active_indices.to(device)]
        active_transe   = transe_rel_emb[active_indices.to(device)]
        num_active      = len(active_indices)

        with torch.amp.autocast("cuda", enabled=use_cuda):
            outputs = model(X, X_mean, mask, golden_triples, teacher_forcing_ratio, active_semantic, active_transe)

            gold_fhs, gold_fhe, gold_bts, gold_bte = build_gold_entity_labels(golden_triples, mask)
            gold_fts, gold_fte, gold_bhs, gold_bhe = build_gold_tail_labels(
                triples_batch=golden_triples,
                unique_subjects_fwd=outputs[MOK["unique_subjects_batch"]],
                unique_subjects_bwd=outputs["unique_subjects_bwd"],
                mask=mask,
                num_relations=num_active,
                rel2idx=rel_to_slot,
            )
            gold_labels = {
                "fwd_head_start": gold_fhs, "fwd_head_end": gold_fhe,
                "bwd_tail_start": gold_bts, "bwd_tail_end": gold_bte,
                "fwd_tail_start": gold_fts, "fwd_tail_end": gold_fte,
                "bwd_head_start": gold_bhs, "bwd_head_end": gold_bhe,
            }
            loss, components = brask_loss(outputs, gold_labels, mask)
            loss = loss / grad_accum_steps

        if torch.isnan(loss):
            print("  NaN loss detected — stopping")
            break

        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum_steps
        n_batches  += 1

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if n_batches % 100 == 0:
            print(f"\t\t batch {n_batches}: {components}")

    # flush any remaining accumulated gradients
    if n_batches % grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / (n_batches + 1e-8)


@torch.no_grad()
def evaluate(
    model: BraskModel,
    dataloader,
    rel2idx: dict,
    all_rel_ids: list,
    semantic_rel_emb: torch.Tensor,
    transe_rel_emb: torch.Tensor,
    stage: int,
) -> float:
    K   = BraskDataset.BATCH_KEYS
    MOK = MODEL_OUTPUT_KEYS
    model.eval()
    total_loss, n_batches = 0.0, 0

    for batch in tqdm(dataloader, desc="Evaluation"):
        X              = batch[K["EMBS"]].to(device)
        X_mean         = batch[K["MEAN_EMBS"]].to(device)
        mask           = batch[K["EMBS_MASKS"]].to(device)
        golden_triples = batch[K["GOLDEN_TRIPLES"]]

        gold_fhs, gold_fhe, gold_bts, gold_bte = build_gold_entity_labels(golden_triples, mask)

        if stage == 1:
            fwd_start, fwd_end = model.fwd_head_predictor(X)
            bwd_start, bwd_end = model.bwd_tail_predictor(X)
            loss = stage1_loss(
                fwd_start.squeeze(-1), fwd_end.squeeze(-1),
                bwd_start.squeeze(-1), bwd_end.squeeze(-1),
                gold_fhs, gold_fhe, gold_bts, gold_bte,
                mask, pos_weight=1.0,
            )
        else:
            active_indices, rel_to_slot = sample_active_relations(
                golden_triples, rel2idx, all_rel_ids, n_neg_override=0
            )
            active_semantic = semantic_rel_emb[active_indices.to(device)]
            active_transe   = transe_rel_emb[active_indices.to(device)]
            num_active      = len(active_indices)

            outputs = model(X, X_mean, mask, golden_triples, 0.0, active_semantic, active_transe)

            gold_fts, gold_fte, gold_bhs, gold_bhe = build_gold_tail_labels(
                triples_batch=golden_triples,
                unique_subjects_fwd=outputs[MOK["unique_subjects_batch"]],
                unique_subjects_bwd=outputs["unique_subjects_bwd"],
                mask=mask,
                num_relations=num_active,
                rel2idx=rel_to_slot,
            )
            gold_labels = {
                "fwd_head_start": gold_fhs, "fwd_head_end": gold_fhe,
                "bwd_tail_start": gold_bts, "bwd_tail_end": gold_bte,
                "fwd_tail_start": gold_fts, "fwd_tail_end": gold_fte,
                "bwd_head_start": gold_bhs, "bwd_head_end": gold_bhe,
            }
            loss, _ = brask_loss(outputs, gold_labels, mask, pos_weight_entity=1.0, pos_weight_obj=1.0)

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)
