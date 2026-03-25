import torch
import torch.nn.functional as F
from collections import defaultdict
import sys
sys.path.insert(0, '..')

from train import (
    BraskModel, set_stage, get_optimizer,
    build_gold_entity_labels, build_gold_tail_labels,
    build_sk_from_gold, stage1_loss, brask_loss,
    MODEL_OUTPUT_KEYS, BraskDataset
)

# ── Fake data — no need to load real files ──
B, L, H   = 2, 20, 768
R         = 5
transe_dim = 100
device    = torch.device("cpu")

X         = torch.randn(B, L, H)
X_mean    = X.mean(dim=1)                          # (B, H)
mask      = torch.ones(B, L)
mask[0, 15:] = 0.0                                 # simulate padding
mask[1, 18:] = 0.0

# Fake relation embeddings
semantic_rel_emb = torch.randn(R, H)
transe_rel_emb   = torch.randn(R, transe_dim)

# Fake rel2idx
rel2idx = {f"rel_{i}": i for i in range(R)}

# Fake golden triples — two triples per sentence
golden_triples = [
    [((1, 2), "rel_0", (5, 7)), ((3, 3), "rel_1", (10, 12))],
    [((0, 1), "rel_2", (4, 6))],
]

# ── Build a minimal batch dict ──
batch = {
    "embs":          X,
    "mean_embs":     X_mean,
    "embs_mask":     mask,
    "golden_triples": golden_triples,
    "entity_id":     ["e1", "e2"],
}

print("── Stage 1 smoke test ──")
model = BraskModel(hidden_dim=H, transe_rel_dim=transe_dim).to(device)
set_stage(model, stage=1)
optimizer = get_optimizer(model, stage=1)

fwd_start, fwd_end = model.fwd_head_predictor(X)
bwd_start, bwd_end = model.bwd_tail_predictor(X)

print(f"  fwd_start shape: {fwd_start.shape}")  # expect (B, L, 1)
assert fwd_start.shape == (B, L, 1), "Wrong shape"

# Gold labels
gold_fhs, gold_fhe, gold_bts, gold_bte = build_gold_entity_labels(golden_triples, mask)
print(f"  gold_fhs shape: {gold_fhs.shape}")    # expect (B, L)

# Loss
loss = stage1_loss(
    fwd_start.squeeze(-1), fwd_end.squeeze(-1),
    bwd_start.squeeze(-1), bwd_end.squeeze(-1),
    gold_fhs, gold_fhe, gold_bts, gold_bte,
    mask
)
print(f"  stage1 loss: {loss.item():.4f}")
assert not torch.isnan(loss), "Loss is nan"
assert not torch.isinf(loss), "Loss is inf"

# Backward
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Check gradients exist
for name, p in model.named_parameters():
    if p.requires_grad:
        assert p.grad is not None, f"No gradient on {name}"
        assert not torch.isnan(p.grad).any(), f"Nan gradient on {name}"
print("  ✓ Stage 1 backward OK")

print("\n── Stage 2 smoke test ──")
set_stage(model, stage=2)
optimizer = get_optimizer(model, stage=2)

# Full forward with teacher forcing
outputs = model(
    X, X_mean, mask, golden_triples,
    teacher_forcing_ratio=1.0,
    semantic_rel_emb=semantic_rel_emb,
    transe_rel_emb=transe_rel_emb
)

# Check all expected keys exist
for key in MODEL_OUTPUT_KEYS.values():
    assert key in outputs, f"Missing output key: {key}"

# Check output shapes
print(f"  fwd_head_start shape: {outputs['fwd_head_start'].shape}")  # (B, L)
print(f"  fwd_tail_start shape: {outputs['fwd_tail_start'].shape}")  # (B, R, S, L)

B_out, R_out, S_out, L_out = outputs['fwd_tail_start'].shape
assert B_out == B
assert R_out == R
assert L_out == L
print(f"  R={R_out}, S={S_out}")

# Gold labels
unique_subjects_batch = outputs[MODEL_OUTPUT_KEYS["unique_subjects_batch"]]
gold_fhs, gold_fhe, gold_bts, gold_bte = build_gold_entity_labels(golden_triples, mask)
gold_fts, gold_fte, gold_bhs, gold_bhe = build_gold_tail_labels(
    triples_batch=golden_triples,
    unique_subjects_batch=unique_subjects_batch,
    mask=mask,
    num_relations=R,
    rel2idx=rel2idx
)

# Check gold tail label shapes match model output shapes
assert gold_fts.shape == outputs['fwd_tail_start'].shape, \
    f"Shape mismatch: gold {gold_fts.shape} vs output {outputs['fwd_tail_start'].shape}"

gold_labels = {
    "fwd_head_start": gold_fhs, "fwd_head_end": gold_fhe,
    "bwd_tail_start": gold_bts, "bwd_tail_end": gold_bte,
    "fwd_tail_start": gold_fts, "fwd_tail_end": gold_fte,
    "bwd_head_start": gold_bhs, "bwd_head_end": gold_bhe,
}

loss, components = brask_loss(outputs, gold_labels, mask)
print(f"  brask loss: {loss.item():.4f}")
print(f"  components: {components}")
assert not torch.isnan(loss)
assert not torch.isinf(loss)
optimizer.zero_grad()
loss.backward()
optimizer.step()

for name, p in model.named_parameters():
    if p.requires_grad:
        assert p.grad is not None, f"No gradient on {name}"
        assert not torch.isnan(p.grad).any(), f"Nan gradient on {name}"

print("  ✓ Stage 2 backward OK")


print("\n── Stage 3 smoke test (teacher forcing = 0.0) ──")
outputs = model(
    X, X_mean, mask, golden_triples,
    teacher_forcing_ratio=0.0,    # uses predicted sk
    semantic_rel_emb=semantic_rel_emb,
    transe_rel_emb=transe_rel_emb
)

unique_subjects_s3 = outputs[MODEL_OUTPUT_KEYS["unique_subjects_batch"]]
gold_fts_s3, gold_fte_s3, gold_bhs_s3, gold_bhe_s3 = build_gold_tail_labels(
    triples_batch=golden_triples,
    unique_subjects_batch=unique_subjects_s3,
    mask=mask,
    num_relations=R,
    rel2idx=rel2idx
)
gold_labels_s3 = {
    "fwd_head_start": gold_fhs, "fwd_head_end": gold_fhe,
    "bwd_tail_start": gold_bts, "bwd_tail_end": gold_bte,
    "fwd_tail_start": gold_fts_s3, "fwd_tail_end": gold_fte_s3,
    "bwd_head_start": gold_bhs_s3, "bwd_head_end": gold_bhe_s3,
}


# o_fts = outputs[MODEL_OUTPUT_KEYS["FORWARD_TAIL_START"]]
# o_fte = outputs[MODEL_OUTPUT_KEYS["FORWARD_TAIL_END"]]
# o_bhs = outputs[MODEL_OUTPUT_KEYS["BACKWARD_HEAD_START"]]
# o_bhe = outputs[MODEL_OUTPUT_KEYS["BACKWARD_HEAD_END"]]




# assert o_fts.shape == gold_fts.shape, f"Shape mismatch: output {o_fts.shape} vs gold {gold_fts.shape}"
# assert o_fte.shape == gold_fte.shape, f"Shape mismatch: output {o_fte.shape} vs gold {gold_fte.shape}"
# assert o_bhs.shape == gold_bhs.shape, f"Shape mismatch: output {o_bhs.shape} vs gold {gold_bhs.shape}"
# assert o_bhe.shape == gold_bhe.shape, f"Shape mismatch: output {o_bhe.shape} vs gold {gold_bhe.shape}"
loss, _ = brask_loss(outputs, gold_labels_s3, mask)
print(f"  brask loss (no tf): {loss.item():.4f}")
assert not torch.isnan(loss)
assert not torch.isinf(loss)

optimizer.zero_grad()
loss.backward()
optimizer.step()

for name, p in model.named_parameters():
    if p.requires_grad:
        assert p.grad is not None, f"No gradient on {name}"
        assert not torch.isnan(p.grad).any(), f"Nan gradient on {name}"

print("  ✓ Stage 3 backward OK")

# print("\n✓ All smoke tests passed")