"""Sample-count-weighted FedAvg: w = sum_k (n_k / n) w_k."""

from typing import Dict, List, Sequence

import torch


def normalise(values: Sequence[float]) -> List[float]:
    total = float(sum(values))
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [float(v) / total for v in values]


def weighted_average(states: List[Dict[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    keys = list(states[0])
    for s in states[1:]:
        if list(s) != keys:
            raise ValueError("Client states have different keys; clients must share one architecture")
    out = {}
    for key in keys:
        acc = torch.zeros_like(states[0][key], dtype=torch.float64)
        for state, w in zip(states, weights):
            acc += state[key].to(torch.float64) * w
        out[key] = acc.to(states[0][key].dtype)
    return out


def fedavg(states: List[Dict[str, torch.Tensor]], sample_counts: Sequence[int], **_) -> Dict[str, torch.Tensor]:
    return weighted_average(states, normalise(sample_counts))
