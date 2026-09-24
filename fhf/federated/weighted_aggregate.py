"""A7: performance-weighted aggregation (Methodology Eq. 11)

    theta^{t+1} = sum_k n_k s_k / (sum_j n_j s_j) * theta_k^t

s_k = client k's validation macro-F1 from the PREVIOUS round (global model on its
validation graph). Zero / missing rule, fixed before running:
    * round 1 (no previous score) or a missing / NaN score: s_k = 1, i.e. plain FedAvg weight;
    * if every s_k is 0, fall back to FedAvg (sample counts only).
"""

import math
from typing import Dict, List, Optional, Sequence

import torch

from fhf.federated.fedavg import normalise, weighted_average


def performance_weights(sample_counts: Sequence[int], scores: Optional[Sequence[Optional[float]]]) -> List[float]:
    if scores is None:
        return normalise(sample_counts)
    s = [1.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v) for v in scores]
    raw = [n * v for n, v in zip(sample_counts, s)]
    if sum(raw) <= 0:
        return normalise(sample_counts)
    return normalise(raw)


def performance_weighted(states: List[Dict[str, torch.Tensor]], sample_counts: Sequence[int],
                         scores: Optional[Sequence[Optional[float]]] = None, **_) -> Dict[str, torch.Tensor]:
    return weighted_average(states, performance_weights(sample_counts, scores))


def get_aggregator(name: str):
    from fhf.federated.fedavg import fedavg
    if name == 'fedavg':
        return fedavg
    if name == 'performance':
        return performance_weighted
    raise ValueError(f"Unknown aggregation.method '{name}' (fedavg | performance)")
