"""Client interface used by the one federated round loop (server.py).

Phase 1 (LoRA encoder), Phase 2 (HetGNN), B1/A8 (MLP) all implement it, so every
federated method shares the same broadcast / local-train / aggregate / select /
checkpoint logic and the same round_metrics.csv rows.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class EvalResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    loss: float
    probs: np.ndarray = None


class FederatedClient:
    cid: int
    n_train: int      # sample count used by FedAvg weights
    n_val: int

    def fit(self, global_state: Dict[str, torch.Tensor], round_idx: int) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Load the global state, train locally, return (new state, {'train_loss': ..., 'lr': ...})."""
        raise NotImplementedError

    def evaluate(self, global_state: Dict[str, torch.Tensor], role: str) -> EvalResult:
        """Predictions of the given global state on this client's `role` split (val | test)."""
        raise NotImplementedError
