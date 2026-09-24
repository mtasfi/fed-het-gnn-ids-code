"""Small shared helpers: logging, seeding, device, timing, class weights, tensor sizes."""

import logging
import os
import random
import resource
import sys
import time
from contextlib import contextmanager
from typing import Dict, Optional, Sequence

import numpy as np


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level), handlers=handlers, force=True,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(requested: str = 'auto'):
    import torch
    if requested and requested != 'auto':
        return torch.device(requested)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Timer:
    """Accumulates named wall-clock durations: with timer('phase1'): ..."""

    def __init__(self):
        self.totals: Dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - start


def peak_rss_mb() -> float:
    """Peak resident memory of this process (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def peak_gpu_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2 ** 20
    except ImportError:
        pass
    return None


def class_weights(labels: Sequence[int], num_classes: int, max_weight: Optional[float] = None) -> np.ndarray:
    """The one shared local inverse-frequency rule (Methodology Eq. 3 / 10).

    Computed over the classes PRESENT on this client, with Laplace smoothing:
        w_c = (n + C_k) / (C_k * (n_c + 1))     for n_c > 0
        w_c = 0                                   for classes absent locally
    An absent class has no samples, so its weight never enters the loss; setting
    it to 0 rather than infinity keeps the vector finite. Weights are then
    normalised so the present classes average to 1 (keeps the LR scale stable).
    """
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    present = counts > 0
    n, c_k = counts.sum(), present.sum()
    w = np.zeros(num_classes)
    if c_k == 0:
        return w
    w[present] = (n + c_k) / (c_k * (counts[present] + 1.0))
    w[present] /= w[present].mean()
    if max_weight is not None:
        w = np.minimum(w, max_weight)
    return w


def state_bytes(state: Dict) -> int:
    """Serialized payload size of a state dict (what one client sends / receives)."""
    total = 0
    for value in state.values():
        if hasattr(value, 'numel'):
            total += value.numel() * value.element_size()
        elif isinstance(value, np.ndarray):
            total += value.nbytes
    return int(total)


def count_parameters(module, trainable_only: bool = False) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only))
