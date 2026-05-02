# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

**Pipeline order:** `download_dataset.py` → `prepare.py` → `train_transe.py` → `prepare_gold_labels.py` → `train.py`

All intermediate data lives in `data/minimized/` (the active working subset). `data/preprocessed/` holds the full dataset caches (optional). Paths are centralised in `utils/settings.py` via dataclasses; `utils/pre_processed_data.py` (`data_loader` singleton) provides lazy-loaded getters for all files.

**Model** (`models/`): `BraskModel` runs a bidirectional pipeline — forward pass predicts head spans then tail spans; backward pass predicts tail spans then head spans. Key sub-modules: `EntityExtractor` (two linear heads for start/end logits), `RelationAttention` (relation-conditioned sentence representation, run twice: once with BERT semantic embeddings, once with TransE), `FuseExtractor` (additive fusion of subject, relation context, token embeddings → object span prediction).

**Staged training** (`train.py` + `training/loops.py`):
- Stage 1 — only `fwd_head_predictor` and `bwd_tail_predictor` are unfrozen; uses `stage1_loss` (4 BCE terms)
- Stage 2 — full model, teacher forcing ratio = 1.0 (gold spans always fed)
- Stage 3 — full model, teacher forcing decays 1.0 → 0.0 over epochs
- Each stage has its own `best` and `resume` checkpoint in `checkpoints/`; training resumes automatically if a resume checkpoint exists

**Loss** (`training/loss.py`): `masked_bce` is the primitive — all losses apply a token mask before BCE. `brask_loss` sums 4 components: forward subject (head start/end), backward subject (tail start/end), forward object (tail start/end per head×relation slot), backward object (head start/end per tail×relation slot).

**Labels** (`training/labels.py`): `sample_active_relations` selects all positive relations in a batch plus an equal number of sampled negatives. `build_gold_tail_labels` produces `(B, R, S, L)` tensors where R = active relations, S = unique subject spans per batch.

