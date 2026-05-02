import pytest
import torch

from training.spans import _span_repr, build_sk_from_gold, build_sk_prediction


B, L, H = 2, 8, 4


@pytest.fixture
def X():
    torch.manual_seed(0)
    return torch.rand(B, L, H)


@pytest.fixture
def mask():
    # batch 0: all real tokens, batch 1: first 5 real, rest padding
    m = torch.ones(B, L)
    m[1, 5:] = 0.0
    return m


# ── _span_repr ────────────────────────────────────────────────────────────────

def test_span_repr_single_token(X):
    result = _span_repr(X, b=0, s=2, e=2)
    assert result.shape == (H,)
    assert torch.allclose(result, X[0, 2])


def test_span_repr_multi_token_is_mean(X):
    result = _span_repr(X, b=0, s=1, e=3)
    expected = X[0, 1:4].mean(dim=0)
    assert torch.allclose(result, expected)


def test_span_repr_full_sequence(X):
    result = _span_repr(X, b=1, s=0, e=L - 1)
    expected = X[1].mean(dim=0)
    assert torch.allclose(result, expected)


# ── build_sk_from_gold ────────────────────────────────────────────────────────

def make_triples(head_spans, tail_spans):
    """Pair head_spans[i] with tail_spans[i] using a dummy relation."""
    return [((hs, he), "P31", (ts, te)) for (hs, he), (ts, te) in zip(head_spans, tail_spans)]


def test_build_sk_from_gold_shape(X, mask):
    triples_batch = [
        make_triples([(0, 1), (3, 4)], [(2, 2), (5, 6)]),  # 2 head spans
        make_triples([(1, 2)],         [(3, 3)]),            # 1 head span
    ]
    sk, sk_mask, unique_spans = build_sk_from_gold(triples_batch, X, mask, use_tail=False)

    assert sk.shape == (B, 2, H)       # S = max(2, 1)
    assert sk_mask.shape == (B, 2)


def test_build_sk_from_gold_mask_values(X, mask):
    triples_batch = [
        make_triples([(0, 1), (3, 4)], [(2, 2), (5, 6)]),
        make_triples([(1, 2)],         [(3, 3)]),
    ]
    _, sk_mask, _ = build_sk_from_gold(triples_batch, X, mask, use_tail=False)

    assert sk_mask[0, 0] == 1.0
    assert sk_mask[0, 1] == 1.0
    assert sk_mask[1, 0] == 1.0
    assert sk_mask[1, 1] == 0.0   # padded slot


def test_build_sk_from_gold_embedding_is_span_mean(X, mask):
    triples_batch = [
        make_triples([(1, 3)], [(0, 0)]),
        make_triples([(0, 0)], [(1, 1)]),
    ]
    sk, _, _ = build_sk_from_gold(triples_batch, X, mask, use_tail=False)

    expected = X[0, 1:4].mean(dim=0)
    assert torch.allclose(sk[0, 0], expected)


def test_build_sk_from_gold_use_tail(X, mask):
    triples_batch = [
        make_triples([(0, 0)], [(2, 4)]),   # tail span (2,4) for batch 0
        make_triples([(0, 0)], [(1, 1)]),
    ]
    sk_head, _, _ = build_sk_from_gold(triples_batch, X, mask, use_tail=False)
    sk_tail, _, _ = build_sk_from_gold(triples_batch, X, mask, use_tail=True)

    # head sk is built from (0,0), tail sk from (2,4) — should differ
    assert not torch.allclose(sk_head[0, 0], sk_tail[0, 0])
    assert torch.allclose(sk_tail[0, 0], X[0, 2:5].mean(dim=0))


def test_build_sk_from_gold_deduplicates_spans(X, mask):
    triples_batch = [
        [((1, 2), "P31", (3, 3)), ((1, 2), "P21", (4, 4))],  # same head span twice
        make_triples([(0, 0)], [(1, 1)]),
    ]
    _, _, unique_spans = build_sk_from_gold(triples_batch, X, mask, use_tail=False)
    assert len(unique_spans[0]) == 1  # deduplicated


def test_build_sk_from_gold_empty_triples_fallback(X, mask):
    triples_batch = [[], []]
    sk, sk_mask, unique_spans = build_sk_from_gold(triples_batch, X, mask, use_tail=False)

    # fallback (0,0) sentinel is inserted and treated as a real slot so the model
    # always has something to attend to — sk_mask is 1.0 for each batch item
    assert sk.shape == (B, 1, H)
    assert all(spans == [(0, 0)] for spans in unique_spans)
    assert sk_mask.sum() == float(B)


def test_build_sk_from_gold_ignores_padding_tokens(X, mask):
    # batch 1 mask has tokens 5-7 as padding; span (5, 6) should be filtered out,
    # leaving no valid spans → fallback (0, 0) sentinel is inserted instead
    triples_batch = [
        make_triples([(0, 1)], [(2, 2)]),
        [((5, 6), "P31", (0, 0))],   # span in padding region
    ]
    _, sk_mask, unique_spans = build_sk_from_gold(triples_batch, X, mask, use_tail=False)
    assert (5, 6) not in unique_spans[1]
    assert unique_spans[1] == [(0, 0)]   # fallback sentinel


# ── build_sk_prediction ───────────────────────────────────────────────────────

def _logits_for_spans(spans, L):
    """Return start and end logit tensors (B=1, L) with high values at the given positions."""
    start = torch.full((1, L), -10.0)
    end   = torch.full((1, L), -10.0)
    for s, e in spans:
        start[0, s] = 10.0
        end[0, e]   = 10.0
    return start, end


def test_build_sk_prediction_shape(X, mask):
    start_logits = torch.zeros(B, L)
    end_logits   = torch.zeros(B, L)
    sk, sk_mask, _ = build_sk_prediction(X, mask, start_logits, end_logits, threshold=0.5)
    assert sk.shape[0] == B
    assert sk.shape[2] == H


def test_build_sk_prediction_detects_span(X, mask):
    # Force a clear span at (1, 3) for batch 0
    start_logits = torch.full((B, L), -10.0)
    end_logits   = torch.full((B, L), -10.0)
    start_logits[0, 1] = 10.0
    end_logits[0, 3]   = 10.0

    sk, sk_mask, unique_spans = build_sk_prediction(X, mask, start_logits, end_logits)

    assert (1, 3) in unique_spans[0]
    assert sk_mask[0, unique_spans[0].index((1, 3))] == 1.0


def test_build_sk_prediction_no_spans_fallback(X, mask):
    # All logits highly negative → no span detected → fallback (0, 0) sentinel inserted
    start_logits = torch.full((B, L), -10.0)
    end_logits   = torch.full((B, L), -10.0)

    sk, sk_mask, unique_spans = build_sk_prediction(X, mask, start_logits, end_logits)

    assert sk.shape == (B, 1, H)
    assert all(spans == [(0, 0)] for spans in unique_spans)
    assert sk_mask.sum() == float(B)


def test_build_sk_prediction_respects_max_span_length(X, mask):
    # start at 0, end at 7 — distance > max_span_length=5, should be ignored
    start_logits = torch.full((B, L), -10.0)
    end_logits   = torch.full((B, L), -10.0)
    start_logits[0, 0] = 10.0
    end_logits[0, 7]   = 10.0

    _, _, unique_spans = build_sk_prediction(X, mask, start_logits, end_logits, max_span_length=5)

    assert (0, 7) not in unique_spans[0]


def test_build_sk_prediction_embedding_matches_span_mean(X, mask):
    start_logits = torch.full((B, L), -10.0)
    end_logits   = torch.full((B, L), -10.0)
    start_logits[0, 2] = 10.0
    end_logits[0, 4]   = 10.0

    sk, _, unique_spans = build_sk_prediction(X, mask, start_logits, end_logits)

    slot = unique_spans[0].index((2, 4))
    expected = X[0, 2:5].mean(dim=0)
    assert torch.allclose(sk[0, slot], expected)
