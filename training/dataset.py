import torch
from torch.utils.data import Dataset


class BraskDataset(Dataset):
    """One item = one entity with its description embeddings and gold triples."""

    BATCH_KEYS = {
        "EMBS":           "embs",
        "EMBS_MASKS":     "embs_mask",
        "MEAN_EMBS":      "mean_embs",
        "GOLDEN_TRIPLES": "golden_triples",
        "ENTITY_ID":      "entity_id",
    }

    def __init__(
        self,
        description_embs,
        description_embs_ids,
        description_embs_masks,
        description_mean_embs,
        golden_triples,
    ):
        self.embs        = description_embs
        self.ids         = description_embs_ids
        self.embs_masks  = description_embs_masks
        self.mean_embs   = description_mean_embs
        self.golden_triples = golden_triples

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        entity_id = self.ids[idx]
        return {
            self.BATCH_KEYS["EMBS"]:           torch.tensor(self.embs[idx],       dtype=torch.float32),
            self.BATCH_KEYS["EMBS_MASKS"]:     torch.tensor(self.embs_masks[idx], dtype=torch.float32),
            self.BATCH_KEYS["MEAN_EMBS"]:      torch.tensor(self.mean_embs[idx],  dtype=torch.float32),
            self.BATCH_KEYS["GOLDEN_TRIPLES"]: self.golden_triples.get(entity_id, []),
            self.BATCH_KEYS["ENTITY_ID"]:      entity_id,
        }


def collate_fn(batch):
    K = BraskDataset.BATCH_KEYS
    return {
        K["EMBS"]:           torch.stack([b[K["EMBS"]]       for b in batch], dim=0),  # (B, L, H)
        K["MEAN_EMBS"]:      torch.stack([b[K["MEAN_EMBS"]]  for b in batch], dim=0),  # (B, H)
        K["EMBS_MASKS"]:     torch.stack([b[K["EMBS_MASKS"]] for b in batch], dim=0),  # (B, L)
        K["GOLDEN_TRIPLES"]: [b[K["GOLDEN_TRIPLES"]] for b in batch],                  # list[list]
        K["ENTITY_ID"]:      [b[K["ENTITY_ID"]]      for b in batch],                  # list[str]
    }
