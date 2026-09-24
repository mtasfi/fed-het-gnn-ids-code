"""Privacy-safe global feature scaler (plan §2.3).

Each client sends, per feature, only its TRAINING-only count n_k, mean mu_k and
M2_k (sum of squared deviations). The server combines them with the parallel
variance equations (Chan et al.) into the exact global mean/variance and
broadcasts one scaler, which every client applies to its own train, validation
and test rows. Validation/test rows never contribute; raw rows never move.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SufficientStats:
    n: int
    mean: np.ndarray
    m2: np.ndarray

    def nbytes(self) -> int:
        return 8 + self.mean.nbytes + self.m2.nbytes


def pre_transform(x: np.ndarray, log1p: bool) -> np.ndarray:
    """Deterministic per-value transform applied before scaling (needs no statistics).
    Signed log1p tames heavy-tailed counts (bytes, rates) and keeps -1 sentinels finite."""
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sign(x) * np.log1p(np.abs(x)) if log1p else x


def client_stats(x: np.ndarray) -> SufficientStats:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return SufficientStats(0, np.zeros(x.shape[1]), np.zeros(x.shape[1]))
    mean = x.mean(axis=0)
    return SufficientStats(n, mean, ((x - mean) ** 2).sum(axis=0))


def combine(stats: List[SufficientStats]) -> SufficientStats:
    total = None
    for s in stats:
        if s.n == 0:
            continue
        if total is None:
            total = SufficientStats(s.n, s.mean.copy(), s.m2.copy())
            continue
        n = total.n + s.n
        delta = s.mean - total.mean
        mean = total.mean + delta * s.n / n
        m2 = total.m2 + s.m2 + delta ** 2 * total.n * s.n / n
        total = SufficientStats(n, mean, m2)
    if total is None:
        raise ValueError("No client provided training statistics")
    return total


class FederatedScaler:
    def __init__(self, log1p: bool = True):
        self.log1p = log1p
        self.mean = None
        self.std = None
        self.n = 0
        self.bytes_uploaded = 0

    def fit_clients(self, client_train_matrices: List[np.ndarray]) -> 'FederatedScaler':
        stats = [client_stats(pre_transform(x, self.log1p)) for x in client_train_matrices]
        self.bytes_uploaded = sum(s.nbytes() for s in stats)
        g = combine(stats)
        self.n = g.n
        self.mean = g.mean
        var = g.m2 / max(g.n, 1)          # population variance, as StandardScaler
        self.std = np.where(var > 1e-12, np.sqrt(var), 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("FederatedScaler used before fit_clients")
        return ((pre_transform(x, self.log1p) - self.mean) / self.std).astype(np.float32)

    def state(self) -> dict:
        return {'log1p': self.log1p, 'n': self.n, 'mean': self.mean.tolist(), 'std': self.std.tolist()}
