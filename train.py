
"""
Plan:
- BraskModel: EntityExtractor(head/ tail), RelationAttention (semantic/transe), FuseExtractor
- Training stages:
    1. Train EntityExtractor (fwd_head/ bwd_tail) only. BCE vs gold triples.
    2. Full model, teacher forcing ratio = 1.0 (gold sk passed in).
    3. Full model, teacher forcing ratio decay from 1.0 to 0.0 over epochs.

"""


import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm

from utils.files import read_cached_array, read_tensor
from utils.settings import settings
from utils.pre_processed_data import data_loader






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()


class BraskDataset(Dataset):
    """The item would represent one entity with its description embedding and golden triples"""


    BATCH_KEYS = {
        "EMBS": "embs",
        "EMBS_MASKS": "embs_mask",
        "MEAN_EMBS": "mean_embs",
        "GOLDEN_TRIPLES": "golden_triples",
        "ENTITY_ID": "entity_id",

    }

    def __init__(
        self,
        description_embs,
        description_embs_ids,
        description_embs_masks,
        description_mean_embs,
        golden_triples
    ):
        self.embs = description_embs
        self.ids = description_embs_ids
        self.embs_masks = description_embs_masks
        self.mean_embs = description_mean_embs
        self.golden_triples = golden_triples


    def __len__(self):
        return len(self.ids)

    def __getitem__(self,idx):
        entity_id = self.ids[idx]
        return {
            self.BATCH_KEYS["EMBS"]: torch.tensor(self.embs[idx], dtype=torch.float32),
            self.BATCH_KEYS["EMBS_MASKS"]: torch.tensor(self.embs_masks[idx], dtype=torch.float32),
            self.BATCH_KEYS["MEAN_EMBS"]: torch.tensor(self.mean_embs[idx], dtype=torch.float32),
            self.BATCH_KEYS["GOLDEN_TRIPLES"]: self.golden_triples[entity_id],
            self.BATCH_KEYS["ENTITY_ID"]: entity_id,
        }

def collate_fn(batch):
    return {
        BraskDataset.BATCH_KEYS["EMBS"]: torch.stack([b[BraskDataset.BATCH_KEYS["EMBS"]] for b in batch], dim=0), #(B, L, H)
        BraskDataset.BATCH_KEYS["MEAN_EMBS"]: torch.stack([b[BraskDataset.BATCH_KEYS["MEAN_EMBS"]] for b in batch], dim=0), #(B, L)
        BraskDataset.BATCH_KEYS["EMBS_MASKS"]: torch.stack([b[BraskDataset.BATCH_KEYS["EMBS_MASKS"]] for b in batch], dim=0), #(B, H)
        BraskDataset.BATCH_KEYS["GOLDEN_TRIPLES"]: [b[BraskDataset.BATCH_KEYS["GOLDEN_TRIPLES"]] for b in batch], # B list [list]
        BraskDataset.BATCH_KEYS["ENTITY_ID"]: [b[BraskDataset.BATCH_KEYS["ENTITY_ID"]] for b in batch], # B list[str]
    }


def build_gold_entity_labels(triples_batch, mask, max_length) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build binary labels for head and tail entity spans from golden triples. Used for computing BCE loss"""


    B, L = mask.shape
    fwd_head_start = torch.zeros((B, L), dtype=torch.float32, device=device)
    fwd_head_end = torch.zeros((B, L), dtype=torch.float32, device=device)
    bwd_tail_start = torch.zeros((B, L), dtype=torch.float32, device=device)
    bwd_tail_end = torch.zeros((B, L), dtype=torch.float32, device=device)


    for b, triples in enumerate(triples_batch):
        for (hs, he), _, (ts, te) in triples:
            if hs < L:
                fwd_head_start[b, hs] = 1.0
            if he < L:
                fwd_head_end[b, he] = 1.0
            if ts < L:
                bwd_tail_start[b, ts] = 1.0
            if te < L:
                bwd_tail_end[b, te] = 1.0
    return fwd_head_start, fwd_head_end, bwd_tail_start, bwd_tail_end