import random

import torch

from training.config import device


def build_gold_entity_labels(
    triples_batch, mask
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Binary start/end labels for head and tail entity spans. Used for Stage-1 BCE loss."""
    B, L = mask.shape
    fwd_head_start = torch.zeros((B, L), dtype=torch.float32, device=device)
    fwd_head_end   = torch.zeros((B, L), dtype=torch.float32, device=device)
    bwd_tail_start = torch.zeros((B, L), dtype=torch.float32, device=device)
    bwd_tail_end   = torch.zeros((B, L), dtype=torch.float32, device=device)

    for b in range(B):
        for (hs, he), _, (ts, te) in triples_batch[b]:
            if hs < L: fwd_head_start[b, hs] = 1.0
            if he < L: fwd_head_end[b, he]   = 1.0
            if ts < L: bwd_tail_start[b, ts]  = 1.0
            if te < L: bwd_tail_end[b, te]    = 1.0

    return fwd_head_start, fwd_head_end, bwd_tail_start, bwd_tail_end


def build_gold_tail_labels(
    triples_batch,
    unique_subjects_fwd,
    unique_subjects_bwd,
    mask,
    num_relations: int,
    rel2idx: dict,
):
    """Build (B, R, S, L) gold labels for forward tails and backward heads.

    unique_subjects_fwd: head spans used as sk in the forward extractor.
    unique_subjects_bwd: tail spans used as sk in the backward extractor.
    rel2idx is the compact slot mapping returned by sample_active_relations (R == num_relations).
    """
    B, L = mask.shape
    R = num_relations

    unique_fwd = [s if s else [(0, 0)] for s in unique_subjects_fwd]
    unique_bwd = [s if s else [(0, 0)] for s in unique_subjects_bwd]
    S_fwd = max(len(s) for s in unique_fwd)
    S_bwd = max(len(s) for s in unique_bwd)

    gold_fwd_tail_start = torch.zeros(B, R, S_fwd, L, dtype=torch.float32, device=device)
    gold_fwd_tail_end   = torch.zeros(B, R, S_fwd, L, dtype=torch.float32, device=device)
    gold_bwd_head_start = torch.zeros(B, R, S_bwd, L, dtype=torch.float32, device=device)
    gold_bwd_head_end   = torch.zeros(B, R, S_bwd, L, dtype=torch.float32, device=device)

    for b, triples in enumerate(triples_batch):
        fwd_slot = {span: idx for idx, span in enumerate(unique_fwd[b])}
        bwd_slot = {span: idx for idx, span in enumerate(unique_bwd[b])}
        for (hs, he), r, (ts, te) in triples:
            if r not in rel2idx:
                continue
            r_idx = rel2idx[r]

            s_fwd = fwd_slot.get((hs, he))
            if s_fwd is not None and s_fwd < S_fwd:
                if ts < L: gold_fwd_tail_start[b, r_idx, s_fwd, ts] = 1.0
                if te < L: gold_fwd_tail_end[b,   r_idx, s_fwd, te] = 1.0

            s_bwd = bwd_slot.get((ts, te))
            if s_bwd is not None and s_bwd < S_bwd:
                if hs < L: gold_bwd_head_start[b, r_idx, s_bwd, hs] = 1.0
                if he < L: gold_bwd_head_end[b,   r_idx, s_bwd, he] = 1.0

    return gold_fwd_tail_start, gold_fwd_tail_end, gold_bwd_head_start, gold_bwd_head_end


def sample_active_relations(
    triples_batch: list,
    rel2idx: dict,
    all_rel_ids: list,
    n_neg_override: int = -1,
) -> tuple[torch.Tensor, dict]:
    """Sample a compact set of K relations for one batch: all positives + sampled negatives.

    Positive relations are those that appear in at least one triple in the batch. Negative
    relations are sampled uniformly from the remainder at a 1:1 ratio by default
    (``|negatives| == |positives|``). Pass ``n_neg_override=0`` to suppress negatives entirely
    (used during evaluation so that val loss reflects only the positives present in the batch).

    Passing the returned ``rel_to_slot`` dict as the ``rel2idx`` argument of
    ``build_gold_tail_labels`` correctly skips non-active relations and maps each active relation
    to its compact slot index 0..K-1.

    Parameters
    ----------
    triples_batch:
        Length-B list of triple lists for the current batch, where each triple is
        ``((hs, he), rel_str, (ts, te))``.
    rel2idx:
        Full relation-string → global-index mapping (used to look up embedding row indices and
        to validate that a relation string is known).
    all_rel_ids:
        ``list(rel2idx.keys())`` pre-computed once before the training loop to avoid rebuilding it
        on every batch call.
    n_neg_override:
        If >= 0, use this as the number of negatives instead of the default 1:1 ratio.
        Pass 0 for evaluation (positives only).

    Returns
    -------
    active_indices : torch.LongTensor, shape (K,)
        Row indices into the full relation-embedding matrices. Use as
        ``semantic_rel_emb[active_indices]`` and ``transe_rel_emb[active_indices]``.
    rel_to_slot : dict[str, int]
        Maps each active relation string to its position 0..K-1 in the sliced embedding.
        Pass directly as the ``rel2idx`` argument of ``build_gold_tail_labels``.
    """
    pos_rels = set()
    for triples in triples_batch:
        for _, r, _ in triples:
            if r in rel2idx:
                pos_rels.add(r)

    neg_pool = [r for r in all_rel_ids if r not in pos_rels]
    n_neg    = n_neg_override if n_neg_override >= 0 else len(pos_rels)
    n_neg    = min(n_neg, len(neg_pool))
    neg_rels = random.sample(neg_pool, n_neg) if n_neg > 0 else []

    active         = list(pos_rels) + neg_rels
    active_indices = torch.tensor([rel2idx[r] for r in active], dtype=torch.long)
    rel_to_slot    = {r: i for i, r in enumerate(active)}
    return active_indices, rel_to_slot
