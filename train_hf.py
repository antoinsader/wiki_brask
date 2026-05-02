"""Equivalent of train.py using HuggingFace Trainer."""
import argparse
import os
import random

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments

from experiment_logging.experiment_logger import ExperimentLogger
from models.BraskModel import BraskModel
from training.config import (
    BATCH_SIZE, CHECKPOINTS_DIR, EARLY_STOP_PATIENCE_STAGES,
    GRAD_ACCUM_STEPS, LEARNING_RATE_STAGE_1, LEARNING_RATE_STAGE_2,
    MODEL_OUTPUT_KEYS, NUM_WORKERS, POS_WEIGHT_ENTITY, POS_WEIGHT_OBJ,
    STAGE1_EPOCHS, STAGE2_EPOCHS, STAGE3_EPOCHS, VAL_SPLIT, device, use_cuda,
)
from training.dataset import BraskDataset, collate_fn
from training.labels import build_gold_entity_labels, build_gold_tail_labels, sample_active_relations
from training.loss import brask_loss, stage1_loss
from training.loops import set_stage
from utils.pre_processed_data import data_loader


# ── Custom Trainer ────────────────────────────────────────────────────────────

class BraskTrainer(Trainer):
    def __init__(self, *args, stage, rel2idx, all_rel_ids, semantic_rel_emb, transe_rel_emb,
                 teacher_forcing_ratio=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage
        self.rel2idx = rel2idx
        self.all_rel_ids = all_rel_ids
        self.semantic_rel_emb = semantic_rel_emb
        self.transe_rel_emb = transe_rel_emb
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def create_optimizer(self):
        lr = LEARNING_RATE_STAGE_1 if self.stage == 1 else LEARNING_RATE_STAGE_2
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=lr)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        K   = BraskDataset.BATCH_KEYS
        MOK = MODEL_OUTPUT_KEYS
        is_eval   = not model.training
        pw_entity = 1.0 if is_eval else POS_WEIGHT_ENTITY
        pw_obj    = 1.0 if is_eval else POS_WEIGHT_OBJ
        tf        = 0.0 if is_eval else self.teacher_forcing_ratio

        X              = inputs[K["EMBS"]]
        mask           = inputs[K["EMBS_MASKS"]]
        golden_triples = inputs[K["GOLDEN_TRIPLES"]]

        gold_fhs, gold_fhe, gold_bts, gold_bte = build_gold_entity_labels(golden_triples, mask)

        if self.stage == 1:
            fwd_start, fwd_end = model.fwd_head_predictor(X)
            bwd_start, bwd_end = model.bwd_tail_predictor(X)
            loss = stage1_loss(
                fwd_start.squeeze(-1), fwd_end.squeeze(-1),
                bwd_start.squeeze(-1), bwd_end.squeeze(-1),
                gold_fhs, gold_fhe, gold_bts, gold_bte,
                mask, pos_weight=pw_entity,
            )
            outputs = None
        else:
            X_mean         = inputs[K["MEAN_EMBS"]]
            n_neg          = 0 if is_eval else -1
            active_indices, rel_to_slot = sample_active_relations(
                golden_triples, self.rel2idx, self.all_rel_ids, n_neg_override=n_neg
            )
            active_semantic = self.semantic_rel_emb[active_indices.to(self.semantic_rel_emb.device)]
            active_transe   = self.transe_rel_emb[active_indices.to(self.transe_rel_emb.device)]
            num_active      = len(active_indices)

            outputs = model(X, X_mean, mask, golden_triples, tf, active_semantic, active_transe)

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
            loss, _ = brask_loss(outputs, gold_labels, mask,
                                 pos_weight_entity=pw_entity, pos_weight_obj=pw_obj)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


# ── Callbacks ─────────────────────────────────────────────────────────────────

class TeacherForcingDecayCallback(TrainerCallback):
    """Decays teacher_forcing_ratio from 1.0 → 0.0 over total_epochs."""

    def __init__(self, trainer_ref: BraskTrainer, total_epochs: int):
        self.trainer_ref = trainer_ref
        self.total_epochs = total_epochs

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = state.epoch or 0
        self.trainer_ref.teacher_forcing_ratio = max(0.0, 1.0 - epoch / self.total_epochs)


class NaNStopCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and loss != loss:  # NaN
            print("  NaN loss detected — stopping")
            control.should_training_stop = True


class BraskLoggerCallback(TrainerCallback):
    def __init__(self, experiment_logger: ExperimentLogger, stage: int):
        self.logger = experiment_logger
        self.stage  = stage
        self._last_train_loss = None
        self._best_val_loss   = float("inf")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            self._last_train_loss = logs["loss"]
        if "eval_loss" in logs and self._last_train_loss is not None:
            val_loss = logs["eval_loss"]
            is_best  = val_loss < self._best_val_loss
            if is_best:
                self._best_val_loss = val_loss
            self.logger.log_epoch(
                self.stage, round(state.epoch),
                self._last_train_loss, val_loss, is_new_best=is_best,
            )


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _has_checkpoint(stage_dir: str) -> bool:
    if not os.path.isdir(stage_dir):
        return False
    return any(
        d.startswith("checkpoint-") and os.path.isdir(os.path.join(stage_dir, d))
        for d in os.listdir(stage_dir)
    )


def _is_done(stage_dir: str) -> bool:
    return os.path.exists(os.path.join(stage_dir, "stage_complete.txt"))


def _mark_done(stage_dir: str, model: BraskModel) -> None:
    """Save best model weights and mark stage as complete."""
    torch.save(model.state_dict(), os.path.join(stage_dir, "best_model.pt"))
    open(os.path.join(stage_dir, "stage_complete.txt"), "w").close()


def _load_best(stage_dir: str, model: BraskModel) -> None:
    model.load_state_dict(torch.load(
        os.path.join(stage_dir, "best_model.pt"), map_location=device,
    ))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size",                  type=int,   default=BATCH_SIZE)
    p.add_argument("--stage1-epochs",               type=int,   default=STAGE1_EPOCHS)
    p.add_argument("--stage2-epochs",               type=int,   default=STAGE2_EPOCHS)
    p.add_argument("--stage3-epochs",               type=int,   default=STAGE3_EPOCHS)
    p.add_argument("--val-split",                   type=float, default=VAL_SPLIT)
    p.add_argument("--early_stop_patience_stage_1", type=int,   default=EARLY_STOP_PATIENCE_STAGES[0])
    p.add_argument("--early_stop_patience_stage_2", type=int,   default=EARLY_STOP_PATIENCE_STAGES[1])
    p.add_argument("--early_stop_patience_stage_3", type=int,   default=EARLY_STOP_PATIENCE_STAGES[2])
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    stage_dirs = {s: os.path.join(CHECKPOINTS_DIR, f"hf_stage{s}") for s in (1, 2, 3)}
    for d in stage_dirs.values():
        os.makedirs(d, exist_ok=True)

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

    shuffled_ids = list(emb_ids)
    random.seed(42)
    random.shuffle(shuffled_ids)
    n_val     = int(len(shuffled_ids) * args.val_split)
    ids_val   = shuffled_ids[:n_val]
    ids_train = shuffled_ids[n_val:]

    id_to_idx = {id_: i for i, id_ in enumerate(emb_ids)}

    logger = ExperimentLogger(vars(args))
    logger.log_dataset_stats(
        n_train_descriptions=len(ids_train),
        n_val_descriptions=len(ids_val),
        n_train_triples=sum(len(golden_triples.get(i, [])) for i in ids_train),
        n_val_triples=sum(len(golden_triples.get(i, [])) for i in ids_val),
        n_relations=len(rel2idx),
    )

    def make_dataset(ids):
        idx = [id_to_idx[i] for i in ids]
        return BraskDataset(
            description_embs=embs_all[idx],
            description_embs_ids=ids,
            description_embs_masks=embs_masks[idx],
            description_mean_embs=embs_mean[idx],
            golden_triples=golden_triples,
        )

    train_dataset = make_dataset(ids_train)
    val_dataset   = make_dataset(ids_val)

    model = BraskModel(hidden_dim=H, transe_rel_dim=transe_rel_dim).to(device)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(CHECKPOINTS_DIR, "brask_init.pt")):
        torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, "brask_init.pt"))

    shared_trainer_kwargs = dict(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        rel2idx=rel2idx,
        all_rel_ids=all_rel_ids,
        semantic_rel_emb=semantic_rel_emb,
        transe_rel_emb=transe_rel_emb,
    )

    def make_training_args(stage_dir, num_epochs, max_grad_norm):
        return TrainingArguments(
            output_dir=stage_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            fp16=use_cuda,
            max_grad_norm=max_grad_norm,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=NUM_WORKERS,
            logging_strategy="epoch",
            report_to="none",
        )

    # ════════════════════════════════════════
    # STAGE 1 — Entity extractors only
    # ════════════════════════════════════════
    print("\n── Stage 1: Entity extractors ──")
    set_stage(model, stage=1)
    if _is_done(stage_dirs[1]):
        print("  Stage 1 already complete — loading best checkpoint.")
        _load_best(stage_dirs[1], model)
        logger.log_stage_end(1, "already_done")
    else:
        trainer1 = BraskTrainer(
            model=model,
            args=make_training_args(stage_dirs[1], args.stage1_epochs, max_grad_norm=1.0),
            stage=1,
            teacher_forcing_ratio=1.0,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience_stage_1),
                NaNStopCallback(),
                BraskLoggerCallback(logger, 1),
            ],
            **shared_trainer_kwargs,
        )
        resume = _has_checkpoint(stage_dirs[1])
        if resume:
            print(f"  Resuming Stage 1 from checkpoint.")
            logger.log_resume(1, 0, float("inf"))
        trainer1.train(resume_from_checkpoint=resume or None)
        _mark_done(stage_dirs[1], model)
        logger.log_stage_end(1, "completed")

    # ════════════════════════════════════════
    # STAGE 2 — Full model, teacher forcing = 1.0
    # ════════════════════════════════════════
    print("\n── Stage 2: Full model (teacher forcing = 1.0) ──")
    set_stage(model, stage=2)
    if _is_done(stage_dirs[2]):
        print("  Stage 2 already complete — loading best checkpoint.")
        _load_best(stage_dirs[2], model)
        logger.log_stage_end(2, "already_done")
    else:
        trainer2 = BraskTrainer(
            model=model,
            args=make_training_args(stage_dirs[2], args.stage2_epochs, max_grad_norm=5.0),
            stage=2,
            teacher_forcing_ratio=1.0,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience_stage_2),
                NaNStopCallback(),
                BraskLoggerCallback(logger, 2),
            ],
            **shared_trainer_kwargs,
        )
        resume = _has_checkpoint(stage_dirs[2])
        if resume:
            print(f"  Resuming Stage 2 from checkpoint.")
            logger.log_resume(2, 0, float("inf"))
        trainer2.train(resume_from_checkpoint=resume or None)
        _mark_done(stage_dirs[2], model)
        logger.log_stage_end(2, "completed")

    # ════════════════════════════════════════
    # STAGE 3 — Full model, teacher forcing decays 1.0 → 0.0
    # ════════════════════════════════════════
    print("\n── Stage 3: Full model (teacher forcing decay) ──")
    set_stage(model, stage=3)
    if _is_done(stage_dirs[3]):
        print("  Stage 3 already complete.")
        logger.log_stage_end(3, "already_done")
    else:
        trainer3 = BraskTrainer(
            model=model,
            args=make_training_args(stage_dirs[3], args.stage3_epochs, max_grad_norm=5.0),
            stage=3,
            teacher_forcing_ratio=1.0,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience_stage_3),
                NaNStopCallback(),
                BraskLoggerCallback(logger, 3),
            ],
            **shared_trainer_kwargs,
        )
        trainer3.add_callback(TeacherForcingDecayCallback(trainer3, args.stage3_epochs))
        resume = _has_checkpoint(stage_dirs[3])
        if resume:
            print(f"  Resuming Stage 3 from checkpoint.")
            logger.log_resume(3, 0, float("inf"))
        trainer3.train(resume_from_checkpoint=resume or None)
        _mark_done(stage_dirs[3], model)
        logger.log_stage_end(3, "completed")

    logger.finish()
    print("\nTraining complete.")
    print(f"Best checkpoints saved in {CHECKPOINTS_DIR}")


if __name__ == "__main__":
    main()
