"""Flow <-> label-row matching (E0.3, the highest data risk).

An extracted flow takes the label of the label-CSV row with the same canonical
bidirectional 5-tuple whose start time is closest to the flow's first packet
(after the clock offset), within `matching.time_tolerance_s`.

Outcomes per flow (match_status):
    matched            exactly one candidate in tolerance
    matched_consistent several candidates in tolerance, all with the same label; nearest taken
    ambiguous_conflict candidates in tolerance with different labels -> excluded
    no_label_row       no row with this 5-tuple within tolerance -> excluded
Rows matched by more than one flow are flagged `row_shared` (e.g. a long Zeek
connection our timeouts split, or duplicate rows); they are kept and counted.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from fhf.data.label_sources import canonical_key

logger = logging.getLogger(__name__)


def flow_keys(flows: pd.DataFrame) -> np.ndarray:
    return canonical_key(flows['proto'], flows['src_ip'], flows['src_port'], flows['dst_ip'], flows['dst_port_id'])


def _asof(left: pd.DataFrame, right: pd.DataFrame, direction: str, tol: float) -> pd.DataFrame:
    return pd.merge_asof(left, right, left_on='t', right_on='ts', by='key', direction=direction,
                         tolerance=tol, allow_exact_matches=True)


def estimate_clock_offset(flows: pd.DataFrame, labels: pd.DataFrame, max_search_s: float,
                          sample: int = 200000, seed: int = 0) -> Dict:
    """Mode of (label ts - flow start) over key-matched pairs; resolution 1 s.
    A clear single peak means a constant clock/timezone offset; set
    matching.clock_offset_s to it."""
    f = flows[['first_ts', 'key']].rename(columns={'first_ts': 't'})
    if len(f) > sample:
        f = f.sample(sample, random_state=seed)
    f = f.sort_values('t')
    lab = labels[['ts', 'key']].sort_values('ts')
    m = _asof(f, lab.assign(label_ts=lab['ts']), 'nearest', max_search_s).dropna(subset=['label_ts'])
    if m.empty:
        return {'pairs': 0, 'offset_s': None, 'peak_share': 0.0, 'top': []}
    diff = np.round(m['label_ts'].to_numpy() - m['t'].to_numpy())
    values, counts = np.unique(diff, return_counts=True)
    order = np.argsort(-counts)
    top = [(float(values[i]), int(counts[i])) for i in order[:10]]
    return {'pairs': int(len(diff)), 'offset_s': top[0][0], 'peak_share': top[0][1] / len(diff), 'top': top}


def match_flows(flows: pd.DataFrame, labels: pd.DataFrame, tolerance_s: float,
                offset_s: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (flows with match columns, per-flow failure table).

    Every label row with the same key and |row ts - (flow start + offset)| <= tolerance
    is a candidate, not only the nearest one on each side. Lookup is a single sorted
    array of composite values key_rank * span + ts, so each flow needs two binary
    searches and nothing is joined many-to-many (ports reused thousands of times by
    DoS / scanning traffic would explode a merge).
    """
    f = flows.copy()
    f['key'] = flow_keys(f)
    t = f['first_ts'].to_numpy(dtype=np.float64) + offset_s

    lab = labels[['ts', 'key', 'row_id', 'raw_label']].dropna(subset=['ts'])
    keys = np.concatenate([lab['key'].to_numpy(), f['key'].to_numpy()])
    codes, _ = pd.factorize(keys)
    lab_k, flow_k = codes[:len(lab)], codes[len(lab):]
    tmin = min(lab['ts'].min(), t.min()) if len(lab) else t.min()
    span = max(lab['ts'].max(), t.max()) - tmin + 10 * tolerance_s + 1.0
    lab_c = lab_k * span + (lab['ts'].to_numpy() - tmin)
    order = np.argsort(lab_c, kind='stable')
    comp = lab_c[order]
    row_id = lab['row_id'].to_numpy()[order]
    raw = lab['raw_label'].to_numpy()[order]
    row_ts = lab['ts'].to_numpy()[order]
    raw_codes, _ = pd.factorize(raw)
    change = np.concatenate([[0], np.cumsum(raw_codes[1:] != raw_codes[:-1])])

    fc = flow_k * span + (t - tmin)
    lo = np.searchsorted(comp, fc - tolerance_s, side='left')
    hi = np.searchsorted(comp, fc + tolerance_s, side='right')
    n_cand = hi - lo
    has = n_cand > 0
    conflict = np.zeros(len(f), dtype=bool)
    conflict[has] = change[hi[has] - 1] != change[lo[has]]

    # nearest candidate: one of the two neighbours of the insertion point, clipped to [lo, hi)
    p = np.searchsorted(comp, fc)
    a = np.clip(p - 1, lo, np.maximum(hi - 1, lo))
    b = np.clip(p, lo, np.maximum(hi - 1, lo))
    a = np.minimum(a, max(len(comp) - 1, 0))
    b = np.minimum(b, max(len(comp) - 1, 0))
    if len(comp):
        pick = np.where(np.abs(comp[b] - fc) < np.abs(comp[a] - fc), b, a)
    else:
        pick = np.zeros(len(f), dtype=int)

    status = np.full(len(f), 'no_label_row', dtype=object)
    status[has] = 'matched'
    status[has & (n_cand > 1)] = 'matched_consistent'
    status[conflict] = 'ambiguous_conflict'
    good = np.isin(status, ['matched', 'matched_consistent'])

    f['match_status'] = status
    f['match_candidates'] = n_cand
    f['label_row_id'] = np.where(good, row_id[pick] if len(comp) else None, None)
    f['raw_label'] = np.where(good, raw[pick] if len(comp) else None, None)
    f['match_dt_s'] = np.where(good, (row_ts[pick] - t) if len(comp) else np.nan, np.nan)

    shared = f['label_row_id'].dropna().value_counts()
    f['row_shared'] = f['label_row_id'].map(shared).fillna(0).astype(int) > 1

    failures = f.loc[~good, ['flow_uid', 'capture_id', 'first_ts', 'proto', 'match_status', 'match_candidates']].rename(
        columns={'match_status': 'reason'})
    return f, failures


def match_summary(matched: pd.DataFrame) -> Dict:
    counts = matched['match_status'].value_counts().to_dict()
    n = len(matched)
    ok = counts.get('matched', 0) + counts.get('matched_consistent', 0)
    return {
        'flows': n,
        'matched': ok,
        'match_rate': ok / n if n else 0.0,
        'ambiguous_rate': counts.get('ambiguous_conflict', 0) / n if n else 0.0,
        'no_label_rate': counts.get('no_label_row', 0) / n if n else 0.0,
        'row_shared_flows': int(matched['row_shared'].sum()),
        'status_counts': counts,
        'abs_dt_p50': float(np.nanmedian(np.abs(matched['match_dt_s']))) if ok else None,
        'abs_dt_p99': float(np.nanpercentile(np.abs(matched['match_dt_s']), 99)) if ok else None,
    }
