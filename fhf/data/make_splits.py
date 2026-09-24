"""Leakage-safe sprint split (E0.4).

Unit of assignment = block = (capture, contiguous time window of `block_seconds`).
A block is never cut: it lands entirely in train or entirely in test, which
also keeps each attacker session together (host-grouped splitting was rejected:
ToN-IoT has only a handful of attacker IPs, so it could leave a class with no
test support).

1. Selection (sprint scale): shuffle blocks with a fixed seed, add whole blocks
   until >= target_flows; then, for every class below rare_class_floor, add the
   unselected blocks richest in that class until the floor is met or blocks run
   out. The final count may exceed the target; rows are never sampled.
2. Train/test: classes are visited rarest first; blocks holding the class are
   moved to test until that class reaches test_fraction of its selected flows;
   remaining blocks fill the overall test_fraction; the rest is train.
3. Checks: every kept class needs >= min_test_per_class test flows and >= 1
   training flow; otherwise the split is marked not ok (E0 stop rule).
"""

import json
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def assign_blocks(df: pd.DataFrame, block_seconds: float) -> pd.Series:
    window = np.floor(df['first_ts'].to_numpy() / block_seconds).astype(np.int64)
    return df['capture_id'].astype(str) + '#' + pd.Series(window, index=df.index).astype(str)


def _block_class_counts(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(['block_id', 'label']).size().unstack(fill_value=0)


def select_blocks(df: pd.DataFrame, target: int, floor: int, rng: np.random.Generator) -> Tuple[set, Dict]:
    bc = _block_class_counts(df)
    sizes = bc.sum(axis=1)
    order = rng.permutation(bc.index.to_numpy())
    selected, total = [], 0
    for block in order:
        if total >= target:
            break
        selected.append(block)
        total += int(sizes[block])
    selected = set(selected)

    added_for_floor = {}
    for cls in bc.columns:
        have = int(bc.loc[list(selected), cls].sum()) if selected else 0
        if have >= floor:
            continue
        candidates = bc.index[(bc[cls] > 0) & ~bc.index.isin(list(selected))]
        ranked = bc.loc[candidates, cls].sort_values(ascending=False)
        added = 0
        for block, count in ranked.items():
            if have >= floor:
                break
            selected.add(block)
            have += int(count)
            added += 1
        added_for_floor[cls] = {'blocks_added': added, 'final_count': have, 'floor_met': have >= floor}
    return selected, added_for_floor


def train_test_blocks(df: pd.DataFrame, test_fraction: float, rng: np.random.Generator) -> Dict[str, str]:
    bc = _block_class_counts(df)
    sizes = bc.sum(axis=1)
    totals = bc.sum(axis=0)
    assignment: Dict[str, str] = {}
    test_counts = pd.Series(0, index=bc.columns)

    for cls in totals.sort_values().index:            # rarest class first
        target = test_fraction * totals[cls]
        blocks = bc.index[(bc[cls] > 0)].to_numpy()
        blocks = [b for b in rng.permutation(blocks) if b not in assignment]
        # leave at least one block of the class for training
        n_class_blocks = int((bc[cls] > 0).sum())
        assigned_test = sum(1 for b in bc.index[bc[cls] > 0] if assignment.get(b) == 'test')
        for block in blocks:
            if test_counts[cls] >= target or assigned_test >= n_class_blocks - 1:
                break
            assignment[block] = 'test'
            assigned_test += 1
            test_counts += bc.loc[block]

    test_total = sum(int(sizes[b]) for b, s in assignment.items() if s == 'test')
    goal = test_fraction * sizes.sum()
    for block in rng.permutation(bc.index.to_numpy()):
        if block in assignment:
            continue
        if test_total < goal:
            assignment[block] = 'test'
            test_total += int(sizes[block])
        else:
            assignment[block] = 'train'
    return assignment


def make_split(labeled: pd.DataFrame, cfg, seed: int = 0) -> Tuple[pd.DataFrame, Dict]:
    """labeled: matched flows with columns flow_uid, capture_id, first_ts, label.
    Returns (flow_uid, block_id, split) and a report dict."""
    s = cfg.split
    rng = np.random.default_rng(seed)
    df = labeled[['flow_uid', 'capture_id', 'first_ts', 'label']].copy()
    df['block_id'] = assign_blocks(df, float(s.block_seconds))

    selected, floor_report = select_blocks(df, int(s.target_flows), int(s.rare_class_floor), rng)
    df = df[df['block_id'].isin(selected)].copy()
    assignment = train_test_blocks(df, float(s.test_fraction), rng)
    df['split'] = df['block_id'].map(assignment)

    counts = df.groupby(['label', 'split']).size().unstack(fill_value=0)
    for col in ('train', 'test'):
        if col not in counts:
            counts[col] = 0
    problems = []
    for cls, row in counts.iterrows():
        if row['train'] < 1:
            problems.append(f"class '{cls}' has no training flows")
        if row['test'] < int(s.min_test_per_class):
            problems.append(f"class '{cls}' has {int(row['test'])} test flows (< {s.min_test_per_class})")

    report = {
        'seed': seed,
        'flows_selected': int(len(df)),
        'blocks_selected': int(df['block_id'].nunique()),
        'blocks_train': int(df.loc[df.split == 'train', 'block_id'].nunique()),
        'blocks_test': int(df.loc[df.split == 'test', 'block_id'].nunique()),
        'test_fraction_realised': float((df['split'] == 'test').mean()),
        'rare_class_floor': floor_report,
        'class_counts': {cls: {k: int(v) for k, v in row.items()} for cls, row in counts.iterrows()},
        'block_overlap_train_test': int(len(set(df.loc[df.split == 'train', 'block_id']) &
                                            set(df.loc[df.split == 'test', 'block_id']))),
        'problems': problems,
        'split_ok': not problems,
    }
    if problems:
        logger.warning("Split problems (E0 stop rule): " + '; '.join(problems))
    return df[['flow_uid', 'block_id', 'split', 'label', 'first_ts']], report


def save_report(report: Dict, path: str):
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
