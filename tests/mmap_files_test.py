import numpy as np
import os
import gc

B, L, H = 100, 20, 8
dtype   = "float32"

original_path = "./test_data/test_embs.npy"
tmp_path      = original_path + ".tmp.npy"

os.makedirs("./test_data", exist_ok=True)

# ── Create original file ──
print(f"Creating original: {original_path}  shape=({B},{L},{H})")
mm_orig = np.lib.format.open_memmap(original_path, mode="w+", dtype=dtype, shape=(B, L, H))
mm_orig[:] = np.random.randn(B, L, H).astype(dtype)
mm_orig.flush()
del mm_orig
gc.collect()
print(f"  Saved. Checksum: {np.load(original_path, mmap_mode='r').sum():.4f}")

# ── Simulate filtering: keep first 60 rows ──
mask  = np.zeros(B, dtype=bool)
mask[:60] = True
N_new = int(mask.sum())

mm_src = np.load(original_path, mmap_mode='r')
original_checksum = float(mm_src[mask].sum())
print(f"\nFiltering {B} -> {N_new} rows")
print(f"  Expected checksum after filter: {original_checksum:.4f}")

mm_dst = np.lib.format.open_memmap(
    tmp_path, mode="w+", dtype=dtype, shape=(N_new, L, H)
)

# Write row by row to avoid fancy-index assignment corrupting header
src_indices = np.where(mask)[0]  # actual row indices to keep
for dst_i, src_i in enumerate(src_indices):
    mm_dst[dst_i] = mm_src[src_i]

mm_dst.flush()
del mm_dst
del mm_src
gc.collect()

# ── Verify tmp before replacing ──
result_tmp = np.load(tmp_path, mmap_mode='r')
print(f"\n  tmp shape:    {result_tmp.shape}   expected ({N_new}, {L}, {H})")
print(f"  tmp checksum: {result_tmp.sum():.4f}   expected {original_checksum:.4f}")
assert result_tmp.shape == (N_new, L, H), "Shape mismatch in tmp"
del result_tmp
gc.collect()

# ── Replace ──
os.replace(tmp_path, original_path)
print(f"\nReplaced original with filtered version")

# ── Verify final ──
result = np.load(original_path, mmap_mode='r')
print(f"  Final shape:    {result.shape}")
print(f"  Final checksum: {result.sum():.4f}")
assert result.shape == (N_new, L, H), "Shape mismatch"
assert abs(result.sum() - original_checksum) < 1e-2, "Checksum mismatch"
print("\n✓ mmap filter and replace test passed")