

import torch
import torch.nn as nn
import torch.nn.functional as F



class EntityExtractor(nn.Module):
    """Which tokens could be the start/end of entities, regardless of relation"""

    def __init__(self, hidden_dim):
        """Entity extraction module, predits start and end"""
        super().__init__()
        self.start_linear = nn.Linear(hidden_dim, 1)
        self.end_linear = nn.Linear(hidden_dim, 1)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            X can be any shape (..., L, H).
            Returns start_logits, end_logits each of shape (..., L, 1) — trailing H collapsed to scalar.
        """
        start_logits = self.start_linear(X)
        end_logits = self.end_linear(X)
        return start_logits, end_logits
