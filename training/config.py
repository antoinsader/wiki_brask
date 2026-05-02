import torch

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda  = torch.cuda.is_available()
NUM_WORKERS = 4 if use_cuda else 0

CHECKPOINTS_DIR = "checkpoints/"

LEARNING_RATE_STAGE_1 = 1e-4
LEARNING_RATE_STAGE_2 = 1e-5

BATCH_SIZE           = 4
GRAD_ACCUM_STEPS     = 2   # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
STAGE1_EPOCHS        = 100
STAGE2_EPOCHS        = 100
STAGE3_EPOCHS        = 128
VAL_SPLIT            = 0.1
#list[int] for each stage 1,2,3
EARLY_STOP_PATIENCE_STAGES = [10,40, 40]
# Tune if model predicts all zeros (increase) or all ones (decrease)
POS_WEIGHT_ENTITY = 7.9
POS_WEIGHT_OBJ    = 7.9

MODEL_OUTPUT_KEYS = {
    "FORWARD_HEAD_START":    "fwd_head_start",
    "FORWARD_HEAD_END":      "fwd_head_end",
    "FORWARD_TAIL_START":    "fwd_tail_start",
    "FORWARD_TAIL_END":      "fwd_tail_end",
    "BACKWARD_TAIL_START":   "bwd_tail_start",
    "BACKWARD_TAIL_END":     "bwd_tail_end",
    "BACKWARD_HEAD_START":   "bwd_head_start",
    "BACKWARD_HEAD_END":     "bwd_head_end",
    "SK":                    "sk",
    "SK_MASK":               "sk_mask",
    "unique_subjects_batch": "unique_subjects_batch",
}
