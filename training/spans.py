import torch


def _span_repr(X: torch.Tensor, b: int, s: int, e: int) -> torch.Tensor:
    """Mean-pool token embeddings over [s, e] inclusive."""
    return X[b, s:e + 1].mean(dim=0)


def build_sk_from_gold(
    triples_batch, X: torch.Tensor, mask: torch.Tensor, use_tail: bool = False
):
    """Build subject-key tensors from gold spans.

    Parameters
    ----------
    triples_batch:
        Length-B list of gold triple lists ``((hs, he), rel, (ts, te))``.
    X : (B, L, H)
        Token embeddings.
    mask : (B, L)
        Padding mask.
    use_tail:
        False → sk from head spans (forward extraction).
        True  → sk from tail spans (backward extraction).

    Returns
    -------
    sk : (B, S, H)
    sk_mask : (B, S)
    unique_spans : list[list[tuple[int, int]]]
    """
    B, L, H = X.shape
    unique_spans = []
    for b, triples in enumerate(triples_batch):
        seen, ordered = {}, []
        for (hs, he), _r, (ts, te) in triples:
            span = (ts, te) if use_tail else (hs, he)
            s0, e0 = span
            if span not in seen:
                if s0 >= L or e0 >= L or mask[b, s0] == 0.0 or mask[b, e0] == 0.0:
                    continue
                seen[span] = len(ordered)
                ordered.append(span)
        unique_spans.append(ordered)

    unique_spans = [s if s else [(0, 0)] for s in unique_spans]
    S       = max(len(s) for s in unique_spans)
    sk      = torch.zeros(B, S, H, dtype=torch.float32, device=X.device)
    sk_mask = torch.zeros(B, S,    dtype=torch.float32, device=X.device)
    for b, spans in enumerate(unique_spans):
        for s_idx, (s0, e0) in enumerate(spans):
            sk[b, s_idx]      = _span_repr(X, b, s0, e0)
            sk_mask[b, s_idx] = 1.0
    return sk, sk_mask, unique_spans


def build_sk_prediction(
    X: torch.Tensor,
    mask: torch.Tensor,
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    threshold: float = 0.5,
    max_span_length: int = 10,
):
    """Extract spans from predicted logits and return sk embeddings.

    Parameters
    ----------
    X : (B, L, H)
    mask : (B, L)
    start_logits : (B, L)  head-start (forward) or tail-start (backward)
    end_logits   : (B, L)  head-end   (forward) or tail-end   (backward)

    Returns
    -------
    sk : (B, S, H)
    sk_mask : (B, S)
    unique_spans : list[list[tuple[int, int]]]
    """
    B, L, H = X.shape
    unique_spans = []
    for b in range(B):
        start_probs     = torch.sigmoid(start_logits[b])
        end_probs       = torch.sigmoid(end_logits[b])
        start_positions = (start_probs >= threshold).nonzero(as_tuple=False).squeeze(-1)
        end_positions   = (end_probs   >= threshold).nonzero(as_tuple=False).squeeze(-1)
        spans, consumed = [], set()
        for s_t in start_positions:
            s = s_t.item()
            valid_ends = [
                e.item() for e in end_positions
                if e.item() >= s
                and e.item() < s + max_span_length
                and e.item() not in consumed
                and mask[b, e.item()] == 1.0
            ]
            if valid_ends:
                e = min(valid_ends)
                spans.append((s, e))
                consumed.add(e)
        unique_spans.append(spans)

    unique_spans = [s if s else [(0, 0)] for s in unique_spans]
    S       = max((len(s) for s in unique_spans), default=1)
    sk      = torch.zeros(B, S, H, dtype=torch.float32, device=X.device)
    sk_mask = torch.zeros(B, S,    dtype=torch.float32, device=X.device)
    for b, spans in enumerate(unique_spans):
        for s_idx, (s0, e0) in enumerate(spans):
            s0 = min(s0, L - 1)
            e0 = min(e0, L - 1)
            sk[b, s_idx]      = _span_repr(X, b, s0, e0)
            sk_mask[b, s_idx] = 1.0
    return sk, sk_mask, unique_spans
