"""
Tests for the chunked mmap filtering logic used in prepare_gold_labels.py.

We don't test the full pipeline (which needs real files on disk), but we
verify the core invariants:
  - filtered mmap has the right shape
  - filtered rows contain the exact source values (no row swaps, no corruption)
  - the file is a valid .npy that can be round-tripped via np.load
  - tmp files are cleaned up after os.replace

Windows note: numpy memmap holds OS file handles until the object is GC-ed.
Every test explicitly `del`s mmap-backed tensors and calls gc.collect() before
the TemporaryDirectory context exits, otherwise cleanup raises PermissionError.
"""
import gc
import os
import tempfile

import numpy as np
import pytest
import torch

from utils.files import init_mmap, read_tensor


# ── helpers ────────────────────────────────────────────────────────────────────

def _write_mmap(path, array: np.ndarray) -> None:
    mm = init_mmap(path, array.shape, array.dtype.str.lstrip("<>=!"))
    mm[:] = array
    mm.flush()
    del mm


def _run_chunk_filter(src_mean, src_masks, keep_indices, chunk_size, tmp_mean_path, tmp_masks_path):
    """Replicates the chunked-mmap filtering loop from prepare_gold_labels."""
    N_new  = len(keep_indices)
    H_mean = src_mean.shape[1]
    L_mask = src_masks.shape[1]

    mean_new  = init_mmap(tmp_mean_path,  (N_new, H_mean), "float32")
    masks_new = init_mmap(tmp_masks_path, (N_new, L_mask), "int64")

    for start in range(0, len(keep_indices), chunk_size):
        end      = min(start + chunk_size, len(keep_indices))
        dst_rows = slice(start, end)
        src_rows = keep_indices[start:end]
        mean_new[dst_rows]  = src_mean[src_rows].numpy()
        masks_new[dst_rows] = src_masks[src_rows].numpy()

    mean_new.flush()
    masks_new.flush()
    del mean_new, masks_new


# ── fixtures ───────────────────────────────────────────────────────────────────

N, H, L = 50, 16, 32   # small synthetic sizes

@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    mean_arr  = rng.random((N, H), dtype=np.float32)
    masks_arr = rng.integers(0, 2, size=(N, L), dtype=np.int64)
    bool_mask    = rng.random(N) > 0.4
    keep_indices = np.where(bool_mask)[0]
    return mean_arr, masks_arr, bool_mask, keep_indices


# ── tests ──────────────────────────────────────────────────────────────────────

def test_filtered_shape_is_correct(synthetic_data):
    mean_arr, masks_arr, _, keep_indices = synthetic_data
    N_new = len(keep_indices)

    with tempfile.TemporaryDirectory() as tmp_dir:
        _write_mmap(os.path.join(tmp_dir, "mean.npy"),  mean_arr)
        _write_mmap(os.path.join(tmp_dir, "masks.npy"), masks_arr)

        src_mean  = read_tensor(os.path.join(tmp_dir, "mean.npy"),  mmap=True)
        src_masks = read_tensor(os.path.join(tmp_dir, "masks.npy"), mmap=True)
        tmp_mean  = os.path.join(tmp_dir, "mean_tmp.npy")
        tmp_masks = os.path.join(tmp_dir, "masks_tmp.npy")
        _run_chunk_filter(src_mean, src_masks, keep_indices, 10, tmp_mean, tmp_masks)

        result_mean_shape  = tuple(np.load(tmp_mean).shape)
        result_masks_shape = tuple(np.load(tmp_masks).shape)

        del src_mean, src_masks  # release OS file handles before TemporaryDirectory cleanup
        gc.collect()

    assert result_mean_shape  == (N_new, H), f"mean shape: {result_mean_shape}"
    assert result_masks_shape == (N_new, L), f"masks shape: {result_masks_shape}"


def test_filtered_values_match_source(synthetic_data):
    """Each row in the output must equal the corresponding kept row in the source."""
    mean_arr, masks_arr, _, keep_indices = synthetic_data

    with tempfile.TemporaryDirectory() as tmp_dir:
        _write_mmap(os.path.join(tmp_dir, "mean.npy"),  mean_arr)
        _write_mmap(os.path.join(tmp_dir, "masks.npy"), masks_arr)

        src_mean  = read_tensor(os.path.join(tmp_dir, "mean.npy"),  mmap=True)
        src_masks = read_tensor(os.path.join(tmp_dir, "masks.npy"), mmap=True)
        tmp_mean  = os.path.join(tmp_dir, "mean_tmp.npy")
        tmp_masks = os.path.join(tmp_dir, "masks_tmp.npy")
        _run_chunk_filter(src_mean, src_masks, keep_indices, 10, tmp_mean, tmp_masks)

        result_mean  = np.load(tmp_mean)
        result_masks = np.load(tmp_masks)

        del src_mean, src_masks
        gc.collect()

    np.testing.assert_array_equal(result_mean,  mean_arr[keep_indices],  err_msg="mean rows mismatch")
    np.testing.assert_array_equal(result_masks, masks_arr[keep_indices], err_msg="masks rows mismatch")


def test_chunk_boundary_does_not_corrupt(synthetic_data):
    """Results must be identical regardless of chunk_size (1, 7, or larger than N)."""
    mean_arr, masks_arr, _, keep_indices = synthetic_data

    results = {}
    for chunk_size in (1, 7, len(keep_indices) + 100):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_mmap(os.path.join(tmp_dir, "mean.npy"),  mean_arr)
            _write_mmap(os.path.join(tmp_dir, "masks.npy"), masks_arr)

            src_mean  = read_tensor(os.path.join(tmp_dir, "mean.npy"),  mmap=True)
            src_masks = read_tensor(os.path.join(tmp_dir, "masks.npy"), mmap=True)
            tmp_mean  = os.path.join(tmp_dir, "mean_tmp.npy")
            tmp_masks = os.path.join(tmp_dir, "masks_tmp.npy")
            _run_chunk_filter(src_mean, src_masks, keep_indices, chunk_size, tmp_mean, tmp_masks)

            results[chunk_size] = (np.load(tmp_mean), np.load(tmp_masks))

            del src_mean, src_masks
            gc.collect()

    ref_mean, ref_masks = results[7]
    for chunk_size, (m, k) in results.items():
        np.testing.assert_array_equal(m, ref_mean,  err_msg=f"mean differs at chunk_size={chunk_size}")
        np.testing.assert_array_equal(k, ref_masks, err_msg=f"masks differs at chunk_size={chunk_size}")


def test_output_is_valid_npy_readable_as_tensor(synthetic_data):
    """The written file must be loadable by read_tensor (both mmap and full load)."""
    mean_arr, masks_arr, _, keep_indices = synthetic_data

    with tempfile.TemporaryDirectory() as tmp_dir:
        _write_mmap(os.path.join(tmp_dir, "mean.npy"),  mean_arr)
        _write_mmap(os.path.join(tmp_dir, "masks.npy"), masks_arr)

        src_mean  = read_tensor(os.path.join(tmp_dir, "mean.npy"),  mmap=True)
        src_masks = read_tensor(os.path.join(tmp_dir, "masks.npy"), mmap=True)
        out_mean  = os.path.join(tmp_dir, "mean_out.npy")
        out_masks = os.path.join(tmp_dir, "masks_out.npy")
        _run_chunk_filter(src_mean, src_masks, keep_indices, 15, out_mean, out_masks)

        mean_mmap  = read_tensor(out_mean,  mmap=True)
        mean_full  = read_tensor(out_mean,  mmap=False)
        masks_mmap = read_tensor(out_masks, mmap=True)
        masks_full = read_tensor(out_masks, mmap=False)

        mean_mmap_shape  = tuple(mean_mmap.shape)
        masks_mmap_shape = tuple(masks_mmap.shape)
        all_are_tensors  = all(
            isinstance(t, torch.Tensor)
            for t in (mean_mmap, mean_full, masks_mmap, masks_full)
        )

        # all_tensors list would keep mmap refs alive — avoid it and del directly
        del src_mean, src_masks, mean_mmap, mean_full, masks_mmap, masks_full
        gc.collect()

    assert all_are_tensors
    assert mean_mmap_shape  == (len(keep_indices), H)
    assert masks_mmap_shape == (len(keep_indices), L)


def test_tmp_files_cleaned_up_after_replace(synthetic_data):
    """After os.replace, tmp files must not exist and originals must be updated."""
    mean_arr, masks_arr, _, keep_indices = synthetic_data

    with tempfile.TemporaryDirectory() as tmp_dir:
        orig_mean  = os.path.join(tmp_dir, "mean.npy")
        orig_masks = os.path.join(tmp_dir, "masks.npy")
        tmp_mean   = os.path.join(tmp_dir, "mean_tmp.npy")
        tmp_masks  = os.path.join(tmp_dir, "masks_tmp.npy")

        _write_mmap(orig_mean,  mean_arr)
        _write_mmap(orig_masks, masks_arr)

        src_mean  = read_tensor(orig_mean,  mmap=True)
        src_masks = read_tensor(orig_masks, mmap=True)
        _run_chunk_filter(src_mean, src_masks, keep_indices, 10, tmp_mean, tmp_masks)

        del src_mean, src_masks  # must close before os.replace on Windows
        gc.collect()

        os.replace(tmp_mean,  orig_mean)
        os.replace(tmp_masks, orig_masks)

        assert not os.path.exists(tmp_mean),  "tmp mean file was not removed"
        assert not os.path.exists(tmp_masks), "tmp masks file was not removed"

        final_mean_shape  = tuple(np.load(orig_mean).shape)
        final_masks_shape = tuple(np.load(orig_masks).shape)

    assert final_mean_shape  == (len(keep_indices), H)
    assert final_masks_shape == (len(keep_indices), L)
