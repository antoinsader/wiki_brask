import torch
import torch.nn.functional as F

from training.config import MODEL_OUTPUT_KEYS, POS_WEIGHT_ENTITY, POS_WEIGHT_OBJ


def masked_bce(pred_logits, gold, mask, pos_weight: float) -> torch.Tensor:
    """Binary cross-entropy with masking. Works for any tensor shape."""
    pw   = torch.tensor([pos_weight], device=pred_logits.device)
    loss = F.binary_cross_entropy_with_logits(pred_logits, gold, pos_weight=pw, reduction="none")
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)


def stage1_loss(
    fwd_head_start_logits,
    fwd_head_end_logits,
    bwd_tail_start_logits,
    bwd_tail_end_logits,
    golden_head_start_labels,
    golden_head_end_labels,
    golden_tail_start_labels,
    golden_tail_end_labels,
    token_mask,
    pos_weight: float = POS_WEIGHT_ENTITY,
) -> torch.Tensor:
    """Stage-1 loss: entity extractors only (L_sub + L_obj from the paper)."""
    L_fwd = (
        masked_bce(fwd_head_start_logits, golden_head_start_labels, token_mask, pos_weight)
        + masked_bce(fwd_head_end_logits, golden_head_end_labels,   token_mask, pos_weight)
    )
    L_bwd = (
        masked_bce(bwd_tail_start_logits, golden_tail_start_labels, token_mask, pos_weight)
        + masked_bce(bwd_tail_end_logits, golden_tail_end_labels,   token_mask, pos_weight)
    )
    return L_fwd + L_bwd


def brask_loss(
    outputs: dict,
    gold_labels: dict,
    token_mask,
    pos_weight_entity: float = POS_WEIGHT_ENTITY,
    pos_weight_obj: float    = POS_WEIGHT_OBJ,
) -> tuple[torch.Tensor, dict]:
    """Full BRASK loss: entity extractor + object/subject predictor terms."""
    MOK = MODEL_OUTPUT_KEYS

    L_f_subject = (
        masked_bce(outputs[MOK["FORWARD_HEAD_START"]], gold_labels["fwd_head_start"], token_mask, pos_weight_entity)
        + masked_bce(outputs[MOK["FORWARD_HEAD_END"]], gold_labels["fwd_head_end"],   token_mask, pos_weight_entity)
    )
    L_b_subject = (
        masked_bce(outputs[MOK["BACKWARD_TAIL_START"]], gold_labels["bwd_tail_start"], token_mask, pos_weight_entity)
        + masked_bce(outputs[MOK["BACKWARD_TAIL_END"]], gold_labels["bwd_tail_end"],   token_mask, pos_weight_entity)
    )

    tok_mask_exp = token_mask.unsqueeze(1).unsqueeze(2)                                        # (B, 1, 1, L)
    fwd_mask_4d  = outputs[MOK["SK_MASK"]].unsqueeze(1).unsqueeze(-1) * tok_mask_exp          # (B, 1, S_fwd, L)
    bwd_mask_4d  = outputs["sk_bwd_mask"].unsqueeze(1).unsqueeze(-1)  * tok_mask_exp          # (B, 1, S_bwd, L)

    L_f_obj = (
        masked_bce(outputs[MOK["FORWARD_TAIL_START"]], gold_labels["fwd_tail_start"], fwd_mask_4d, pos_weight_obj)
        + masked_bce(outputs[MOK["FORWARD_TAIL_END"]], gold_labels["fwd_tail_end"],   fwd_mask_4d, pos_weight_obj)
    )
    L_b_obj = (
        masked_bce(outputs[MOK["BACKWARD_HEAD_START"]], gold_labels["bwd_head_start"], bwd_mask_4d, pos_weight_obj)
        + masked_bce(outputs[MOK["BACKWARD_HEAD_END"]], gold_labels["bwd_head_end"],   bwd_mask_4d, pos_weight_obj)
    )

    total = L_f_subject + L_b_subject + L_f_obj + L_b_obj
    return total, {
        "L_f_subject": L_f_subject.item(),
        "L_b_subject": L_b_subject.item(),
        "L_f_obj":     L_f_obj.item(),
        "L_b_obj":     L_b_obj.item(),
    }
