import torch
import torch.nn as nn

from models.EntityExtractor import EntityExtractor
from training.config import MODEL_OUTPUT_KEYS
from training.spans import build_sk_from_gold, build_sk_prediction


class RelationAttention(nn.Module):
    """Semantic relation guidance (paper §3.3.2): fine-grained sentence representation."""

    def __init__(self, hidden_dim: int, rel_dim: int, attention_dim: int = 256):
        super().__init__()
        self.w_x = nn.Linear(hidden_dim, attention_dim)
        self.w_r = nn.Linear(rel_dim,    attention_dim)
        self.w_g = nn.Linear(hidden_dim, attention_dim)
        self.V   = nn.Linear(attention_dim, 1)

    def forward(self, X, relation_embedding, tokens_mean_embedding, mask):
        """Which tokens matter most for this relation? summing the embeddings weighted by how much they matter
        X                    : (B, L, H)
        relation_embedding   : (R, rel_dim)
        tokens_mean_embedding: (B, H)
        mask                 : (B, L)
        Returns c (B, R, H), a (B, R, L)
        """
        wx_xi = self.w_x(X)                         # (B, L, attn_dim)
        wr_rj = self.w_r(relation_embedding)         # (R, attn_dim)
        wg_hg = self.w_g(tokens_mean_embedding)      # (B, attn_dim)

        x_exp = wx_xi.unsqueeze(1)                   # (B, 1, L, attn_dim)
        r_exp = wr_rj.unsqueeze(0).unsqueeze(2)      # (1, R, 1, attn_dim)
        g_exp = wg_hg.unsqueeze(1).unsqueeze(2)      # (B, 1, 1, attn_dim)

        z = torch.tanh(x_exp + r_exp + g_exp)        # (B, R, L, attn_dim)
        e = self.V(z).squeeze(-1)                    # (B, R, L)
        e = e.masked_fill(~mask.unsqueeze(1).bool(), float("-inf"))
        a = torch.softmax(e, dim=-1)                 # (B, R, L)

        c = (a.unsqueeze(-1) * X.unsqueeze(1)).sum(dim=2)  # (B, R, H)
        return c, a


class FuseExtractor(nn.Module):
    """Object extraction via fused subject-key + relation context (paper §3.3.3, §3.4)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_x = nn.Linear(hidden_dim, hidden_dim)
        self.w_s = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        X: torch.Tensor,
        c: torch.Tensor,
        sk: torch.Tensor,
        sk_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        X       : (B, L, H)
        c       : (B, R, H)   relation context
        sk      : (B, S, H)   subject-key embeddings
        sk_mask : (B, S)

        Returns h_ijk : (B, R, S, L, H)
        """
        R = c.shape[1]

        w_x  = self.w_x(X)                              # (B, L, H)
        w_sk = self.w_s(sk)                             # (B, S, H)
        w_sk = sk_mask.unsqueeze(-1) * w_sk             # zero out padded subjects

        h_ik  = w_sk.unsqueeze(2) + w_x.unsqueeze(1)   # (B, S, L, H)
        h_ij  = c.unsqueeze(2) + X.unsqueeze(1).expand(-1, R, -1, -1)  # (B, R, L, H)

        h_ijk = h_ik.unsqueeze(1) + h_ij.unsqueeze(2)  # (B, R, S, L, H)
        return h_ijk


class BraskModel(nn.Module):
    """Full BRASK model with forward and backward triple extraction."""

    def __init__(self, hidden_dim: int, transe_rel_dim: int):
        super().__init__()
        self.fwd_head_predictor     = EntityExtractor(hidden_dim)
        self.bwd_tail_predictor     = EntityExtractor(hidden_dim)
        self.fwd_relation_attention = RelationAttention(hidden_dim, rel_dim=hidden_dim)
        self.bwd_relation_attention = RelationAttention(hidden_dim, rel_dim=transe_rel_dim)
        self.fwd_fuse_extractor     = FuseExtractor(hidden_dim)
        self.bwd_fuse_extractor     = FuseExtractor(hidden_dim)
        self.fwd_tail_predictor     = EntityExtractor(hidden_dim)
        self.bwd_head_predictor     = EntityExtractor(hidden_dim)
        self.inference_threshold    = 0.5

    def forward(
        self,
        X,
        X_mean,
        mask,
        golden_triples,
        teacher_forcing_ratio: float,
        semantic_rel_emb,
        transe_rel_emb,
    ) -> dict:
        """Run one forward pass of the full BRASK model.

        Parameters
        ----------
        X : (B, L, H)
            Token embeddings from the description encoder.
        X_mean : (B, H)
            Mean-pooled token embedding (global sentence representation).
        mask : (B, L)
            Padding mask — 1.0 for real tokens, 0.0 for padding.
        golden_triples : list[list[((int,int), str, (int,int))]]
            Length-B list of gold triples ``((hs, he), rel_str, (ts, te))`` per example.
            Used only when teacher forcing is active.
        teacher_forcing_ratio : float
            Probability in [0, 1] of using gold spans to build sk instead of predicted spans.
            1.0 = always gold (Stage 2), decays to 0.0 by end of Stage 3.
        semantic_rel_emb : (K, H)
            BERT-based relation embeddings for the K active relations in this batch.
        transe_rel_emb : (K, transe_dim)
            TransE relation embeddings for the K active relations in this batch.

        Returns
        -------
        dict with keys from MODEL_OUTPUT_KEYS plus ``"sk_bwd"``, ``"sk_bwd_mask"``,
        ``"unique_subjects_bwd"``:

        - ``fwd_head_start`` / ``fwd_head_end``   : (B, L)        forward subject span logits
        - ``bwd_tail_start`` / ``bwd_tail_end``   : (B, L)        backward object span logits
        - ``fwd_tail_start`` / ``fwd_tail_end``   : (B, K, S, L)  forward object span logits
        - ``bwd_head_start`` / ``bwd_head_end``   : (B, K, S, L)  backward subject span logits
        - ``sk`` / ``sk_mask``                    : (B, S_fwd, H) / (B, S_fwd)  forward subject keys
        - ``sk_bwd`` / ``sk_bwd_mask``            : (B, S_bwd, H) / (B, S_bwd) backward subject keys
        - ``unique_subjects_batch``               : list[list[tuple]] forward subject spans per example
        - ``unique_subjects_bwd``                 : list[list[tuple]] backward subject spans per example
        """
        MOK = MODEL_OUTPUT_KEYS

        fwd_head_start_logits, fwd_head_end_logits = self.fwd_head_predictor(X)   # (B, L, 1), (B, L, 1)
        bwd_tail_start_logits, bwd_tail_end_logits = self.bwd_tail_predictor(X)   # (B, L, 1), (B, L, 1)

        forward_c,  _ = self.fwd_relation_attention(X, semantic_rel_emb, X_mean, mask)  # (B, K, H)
        backward_c, _ = self.bwd_relation_attention(X, transe_rel_emb,   X_mean, mask)  # (B, K, H)

        use_gold = torch.rand(1).item() < teacher_forcing_ratio
        if use_gold:
            sk_fwd, sk_fwd_mask, unique_subjects_fwd = build_sk_from_gold(
                golden_triples, X, mask, use_tail=False)   # (B, S_fwd, H), (B, S_fwd)
            sk_bwd, sk_bwd_mask, unique_subjects_bwd = build_sk_from_gold(
                golden_triples, X, mask, use_tail=True)    # (B, S_bwd, H), (B, S_bwd)
        else:
            sk_fwd, sk_fwd_mask, unique_subjects_fwd = build_sk_prediction(
                X, mask,
                fwd_head_start_logits.squeeze(-1),
                fwd_head_end_logits.squeeze(-1),
                self.inference_threshold,
            )                                              # (B, S_fwd, H), (B, S_fwd)
            sk_bwd, sk_bwd_mask, unique_subjects_bwd = build_sk_prediction(
                X, mask,
                bwd_tail_start_logits.squeeze(-1),
                bwd_tail_end_logits.squeeze(-1),
                self.inference_threshold,
            )                                              # (B, S_bwd, H), (B, S_bwd)

        forward_hijk  = self.fwd_fuse_extractor(X, forward_c,  sk_fwd, sk_fwd_mask)   # (B, K, S_fwd, L, H)
        backward_hijk = self.bwd_fuse_extractor(X, backward_c, sk_bwd, sk_bwd_mask)   # (B, K, S_bwd, L, H)

        fwd_tail_start, fwd_tail_end = self.fwd_tail_predictor(forward_hijk)   # (B, K, S_fwd, L, 1)
        bwd_head_start, bwd_head_end = self.bwd_head_predictor(backward_hijk)  # (B, K, S_bwd, L, 1)

        return {
            MOK["FORWARD_HEAD_START"]:    fwd_head_start_logits.squeeze(-1),  # (B, L)
            MOK["FORWARD_HEAD_END"]:      fwd_head_end_logits.squeeze(-1),    # (B, L)
            MOK["BACKWARD_TAIL_START"]:   bwd_tail_start_logits.squeeze(-1),  # (B, L)
            MOK["BACKWARD_TAIL_END"]:     bwd_tail_end_logits.squeeze(-1),    # (B, L)
            MOK["FORWARD_TAIL_START"]:    fwd_tail_start.squeeze(-1),         # (B, K, S_fwd, L)
            MOK["FORWARD_TAIL_END"]:      fwd_tail_end.squeeze(-1),           # (B, K, S_fwd, L)
            MOK["BACKWARD_HEAD_START"]:   bwd_head_start.squeeze(-1),         # (B, K, S_bwd, L)
            MOK["BACKWARD_HEAD_END"]:     bwd_head_end.squeeze(-1),           # (B, K, S_bwd, L)
            MOK["SK"]:                    sk_fwd,                             # (B, S_fwd, H)
            MOK["SK_MASK"]:               sk_fwd_mask,                        # (B, S_fwd)
            "sk_bwd":                     sk_bwd,                             # (B, S_bwd, H)
            "sk_bwd_mask":                sk_bwd_mask,                        # (B, S_bwd)
            MOK["unique_subjects_batch"]: unique_subjects_fwd,                # list[list[tuple]]
            "unique_subjects_bwd":        unique_subjects_bwd,                # list[list[tuple]]
        }
