"""Client partition (E0.4): class-conditional Dirichlet-alpha over whole blocks,
plus one IID reference, plus per-client grouped validation.

* Each block has a dominant class (its most frequent label).
* For every class c a client distribution p_c ~ Dir(alpha * 1_K) is drawn once;
  each block of dominant class c goes to a client drawn from p_c. The same p_c
  assigns the TEST blocks, so each client's held-out graph mirrors its training
  distribution and worst-client macro-F1 is meaningful. The union of test flows
  is identical for every alpha; only its grouping changes.
* IID reference: blocks go to clients uniformly at random.
* Validation: per client, its LATEST training blocks (by block start time) until
  >= val_fraction of that client's training flows; at least one block stays in
  training. Blocks are never cut.
Partitions are seed-stable and stored as files; runs read them, never recompute.
"""

import logging
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _dominant(split_df: pd.DataFrame) -> pd.Series:
    return split_df.groupby('block_id')['label'].agg(lambda s: s.value_counts().idxmax())


def partition(split_df: pd.DataFrame, num_clients: int, alpha: Union[float, str], val_fraction: float,
              seed: int = 0, max_tries: int = 200) -> Tuple[pd.DataFrame, Dict]:
    """split_df: flow_uid, block_id, split, label, first_ts. Returns (flow_uid, client, role) and a report."""
    dominant = _dominant(split_df)
    block_split = split_df.groupby('block_id')['split'].first()
    classes = sorted(split_df['label'].unique())

    for attempt in range(max_tries):
        rng = np.random.default_rng(seed + attempt * 7919)
        if alpha == 'iid':
            probs = {c: np.full(num_clients, 1.0 / num_clients) for c in classes}
        else:
            probs = {c: rng.dirichlet(np.full(num_clients, float(alpha))) for c in classes}
        block_client = {b: int(rng.choice(num_clients, p=probs[dominant[b]])) for b in dominant.index}
        train_blocks = [b for b in dominant.index if block_split[b] == 'train']
        per_client = pd.Series([block_client[b] for b in train_blocks]).value_counts()
        # every client needs >= 2 training blocks (one may become validation)
        if len(per_client) == num_clients and per_client.min() >= 2:
            break
    else:
        raise RuntimeError(f"Could not draw a partition with every client holding >= 2 training blocks "
                           f"(alpha={alpha}, {num_clients} clients). Use fewer clients or smaller blocks.")

    out = split_df[['flow_uid', 'block_id', 'split', 'label', 'first_ts']].copy()
    out['client'] = out['block_id'].map(block_client).astype(int)
    out['role'] = out['split']

    for client in range(num_clients):
        blocks = out[(out.client == client) & (out.split == 'train')].groupby('block_id').agg(
            start=('first_ts', 'min'), n=('flow_uid', 'size')).sort_values('start', ascending=False)
        need = val_fraction * blocks['n'].sum()
        taken = 0
        val_blocks = []
        for block, row in blocks.iloc[:-1].iterrows():      # keep the earliest block for training
            if taken >= need:
                break
            val_blocks.append(block)
            taken += int(row['n'])
        out.loc[out['block_id'].isin(val_blocks), 'role'] = 'val'

    dist = out.groupby(['client', 'role', 'label']).size().rename('flows').reset_index()
    report = {
        'alpha': alpha,
        'seed': seed,
        'attempt': attempt,
        'num_clients': num_clients,
        'flows_per_client_role': out.groupby(['client', 'role']).size().unstack(fill_value=0).to_dict(),
        'dirichlet_probs': {c: [round(float(p), 4) for p in probs[c]] for c in classes},
    }
    return out[['flow_uid', 'client', 'role', 'block_id', 'label']], {'report': report, 'distribution': dist}


def alpha_tag(alpha) -> str:
    return 'iid' if alpha == 'iid' else f'{float(alpha):g}'
