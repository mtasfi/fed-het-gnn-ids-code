"""File layout of the local work area and the loader every run uses.

work/<dataset>/
    flows/part-*.parquet          extractor output (raw segments, local only)
    labels.parquet                normalised label rows
    matched.parquet               every extracted flow + match status
    flows_labeled.parquet         matched flows + canonical label + has_payload decision
    e0/                           E0 reports (audit, rule, match, split, templates, kappa)
    splits/split.parquet          flow_uid, block_id, split
    splits/partition_alpha=<a>.parquet   flow_uid, client, role (train|val|test)
    cache/embeddings/<tag>/       frozen payload embeddings per encoder tag
    cache/phase1/<tag>/           Phase-1 checkpoints (LoRA + head)
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from fhf.common.config import resolve_path
from fhf.data.partition_clients import alpha_tag

logger = logging.getLogger(__name__)


class Store:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = os.path.join(resolve_path(cfg.paths.work_dir), cfg.dataset.name)

    def p(self, *parts) -> str:
        path = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    @property
    def flows_dir(self): return self.p('flows', '.keep').rsplit('/', 1)[0]
    @property
    def labels(self): return self.p('labels.parquet')
    @property
    def matched(self): return self.p('matched.parquet')
    @property
    def labeled(self): return self.p('flows_labeled.parquet')
    @property
    def split(self): return self.p('splits', 'split.parquet')

    def e0(self, name: str) -> str:
        return self.p('e0', name)

    def partition(self, alpha) -> str:
        return self.p('splits', f'partition_alpha={alpha_tag(alpha)}.parquet')

    def embeddings_dir(self, tag: str) -> str:
        return self.p('cache', 'embeddings', tag, '.keep').rsplit('/', 1)[0]

    def phase1_dir(self, tag: str) -> str:
        return self.p('cache', 'phase1', tag, '.keep').rsplit('/', 1)[0]


def load_run_frame(cfg, alpha=None, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Labelled flows of the fixed split joined with the stored client partition.
    Every run consumes these IDs directly (plan: no on-the-fly repartitioning)."""
    store = Store(cfg)
    alpha = cfg.partition.alpha if alpha is None else alpha
    part_path = store.partition(alpha)
    if not os.path.exists(part_path):
        raise FileNotFoundError(f"{part_path} missing: run `experiments/run_e0.py split` for this dataset first")
    part = pd.read_parquet(part_path, columns=['flow_uid', 'client', 'role'])
    flows = pd.read_parquet(store.labeled, columns=columns)
    df = part.merge(flows, on='flow_uid', how='left', validate='one_to_one')
    if df['label'].isna().any():
        raise RuntimeError("Partition references flows missing from flows_labeled.parquet; rerun E0 split")

    if cfg.get('subset_flows') or cfg.get('smoke'):
        # earliest flows of every client/role (probe: proportional share; smoke: fixed caps)
        df = df.sort_values('first_ts', kind='stable')
        keys = [df['client'], df['role']]
        rank = df.groupby(keys).cumcount()
        size = df.groupby(keys)['flow_uid'].transform('size')
        if cfg.get('subset_flows'):
            frac = min(1.0, int(cfg.subset_flows) / max(len(df), 1))
            cap = (size * frac).round().clip(lower=5)
            note = 'SUBSET for the timing probe'
        else:
            n = int(cfg.get('smoke_flows_per_client', 400))
            cap = np.where(df['role'] == 'train', n, max(n // 4, 20))
            note = 'SMOKE MODE'
        df = df[rank < cap]
        logger.warning(f"{note}: {len(df)} real flows; not a reported result")
    return df.reset_index(drop=True)


def class_names(cfg) -> List[str]:
    """Fixed label order for the run: sorted canonical classes present in the split."""
    store = Store(cfg)
    split = pd.read_parquet(store.split, columns=['label'])
    excluded = set(cfg.dataset.get('exclude_classes') or [])
    return sorted(c for c in split['label'].unique() if c not in excluded)


def encode_labels(labels: pd.Series, names: List[str]) -> np.ndarray:
    index: Dict[str, int] = {n: i for i, n in enumerate(names)}
    missing = set(labels.unique()) - set(index)
    if missing:
        raise ValueError(f"Labels outside the fixed label set: {sorted(missing)}")
    return labels.map(index).to_numpy(dtype=np.int64)
