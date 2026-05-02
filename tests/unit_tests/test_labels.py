import pytest
import torch

from training.labels import sample_active_relations


REL2IDX = {f"P{i}": i for i in range(20)}   # P0..P19 → 0..19
ALL_REL_IDS = list(REL2IDX.keys())


def make_batch(*rel_lists):
    """Build a triples_batch where each item has the given relation strings."""
    return [
        [((0, 0), r, (1, 1)) for r in rels]
        for rels in rel_lists
    ]


# ── return types ──────────────────────────────────────────────────────────────

def test_returns_tensor_and_dict():
    batch = make_batch(["P0", "P1"])
    indices, slot = sample_active_relations(batch, REL2IDX, ALL_REL_IDS)
    assert isinstance(indices, torch.Tensor)
    assert indices.dtype == torch.long
    assert isinstance(slot, dict)


# ── positives are always included ─────────────────────────────────────────────

def test_all_positives_present():
    batch = make_batch(["P0", "P1", "P2"])
    indices, slot = sample_active_relations(batch, REL2IDX, ALL_REL_IDS)
    active_set = set(slot.keys())
    assert {"P0", "P1", "P2"}.issubset(active_set)


def test_positives_from_multiple_batch_items():
    batch = make_batch(["P0"], ["P3"], ["P0", "P5"])   # P0 appears twice — deduplicated
    _, slot = sample_active_relations(batch, REL2IDX, ALL_REL_IDS)
    assert {"P0", "P3", "P5"}.issubset(set(slot.keys()))


def test_unknown_relation_skipped():
    batch = make_batch(["P0", "UNKNOWN_REL"])
    _, slot = sample_active_relations(batch, REL2IDX, ALL_REL_IDS)
    assert "UNKNOWN_REL" not in slot
    assert "P0" in slot


# ── 1:1 negative sampling (default) ──────────────────────────────────────────

def test_default_ratio_is_one_to_one():
    batch = make_batch(["P0", "P1", "P2"])   # 3 positives
    indices, slot = sample_active_relations(batch, REL2IDX, ALL_REL_IDS)
    assert len(slot) == 6   # 3 pos + 3 neg


def test_negatives_are_not_positives():
    batch = make_batch(["P0", "P1"])
    _, slot = sample_active_relations(batch, REL2IDX, ALL_REL_IDS)
    pos = {"P0", "P1"}
    negatives = set(slot.keys()) - pos
    assert negatives.isdisjoint(pos)


def test_active_indices_match_slot_keys():
    batch = make_batch(["P0", "P1"])
    indices, slot = sample_active_relations(batch, REL2IDX, ALL_REL_IDS)
    for rel, slot_idx in slot.items():
        assert indices[slot_idx].item() == REL2IDX[rel]


# ── n_neg_override=0 (evaluation mode) ───────────────────────────────────────

def test_n_neg_override_zero_returns_positives_only():
    batch = make_batch(["P0", "P1", "P2"])
    indices, slot = sample_active_relations(batch, REL2IDX, ALL_REL_IDS, n_neg_override=0)
    assert set(slot.keys()) == {"P0", "P1", "P2"}
    assert len(indices) == 3


def test_n_neg_override_custom_value():
    batch = make_batch(["P0", "P1"])   # 2 positives
    indices, slot = sample_active_relations(batch, REL2IDX, ALL_REL_IDS, n_neg_override=5)
    assert len(slot) == 7   # 2 pos + 5 neg


# ── slot indices are compact 0..K-1 ──────────────────────────────────────────

def test_slot_indices_are_contiguous():
    batch = make_batch(["P0", "P1", "P2"])
    _, slot = sample_active_relations(batch, REL2IDX, ALL_REL_IDS)
    assert set(slot.values()) == set(range(len(slot)))


# ── edge cases ────────────────────────────────────────────────────────────────

def test_empty_batch_returns_empty():
    batch = [[], []]
    indices, slot = sample_active_relations(batch, REL2IDX, ALL_REL_IDS)
    assert len(slot) == 0
    assert len(indices) == 0


def test_all_relations_positive_caps_negatives():
    # Use a tiny rel2idx so positives exhaust the whole set
    tiny_rel2idx  = {"P0": 0, "P1": 1, "P2": 2}
    tiny_all_ids  = list(tiny_rel2idx.keys())
    batch = make_batch(["P0", "P1", "P2"])   # all 3 are positive
    _, slot = sample_active_relations(batch, tiny_rel2idx, tiny_all_ids)
    # neg_pool is empty → no negatives can be sampled
    assert set(slot.keys()) == {"P0", "P1", "P2"}


def test_n_neg_override_capped_by_neg_pool():
    tiny_rel2idx = {"P0": 0, "P1": 1, "P2": 2}
    tiny_all_ids = list(tiny_rel2idx.keys())
    batch = make_batch(["P0", "P1"])   # 2 positives, only 1 negative available
    _, slot = sample_active_relations(batch, tiny_rel2idx, tiny_all_ids, n_neg_override=10)
    assert len(slot) == 3   # 2 pos + 1 neg (capped)
