"""E0.3 checks on the flow <-> label matching, beyond the match rate itself.

    signature_crosstab   flows whose forward payload carries an unambiguous attack
                         signature (SQL injection, command injection, XSS) against the
                         label the matching gave them. A correct matching labels these
                         flows with the expected class; a shifted clock or a wrong key
                         labels them `normal` or leaves them unmatched.
    class_coverage       per raw label: share of label rows inside the capture time
                         spans that were claimed by at least one flow. Rows outside every
                         capture span are ignored, so a partial download is judged only
                         on the time it covers.
    failure_breakdown    unmatched / ambiguous flows by protocol, end reason, port class,
                         duration bucket and whether their 5-tuple appears in the labels
                         at all (time problem vs tuple problem).
    gate                 pass / fail against `matching.gate` thresholds.

Reports contain counts and rates only; no payload text, IP address or port value.
"""

import logging
import re
from typing import Dict, List
from urllib.parse import unquote_plus

import numpy as np
import pandas as pd

from fhf.data.payload_rules import to_text

logger = logging.getLogger(__name__)

MATCHED = ('matched', 'matched_consistent')

# name -> (pattern over URL-decoded forward payload text, expected raw label)
SIGNATURES = {
    'sqli': (r"union\s+(all\s+)?select|\bor\s+'?1'?\s*=\s*'?1\b|\bsleep\s*\(\s*\d|\bbenchmark\s*\("
             r"|information_schema|'\s*(--|#)", 'injection'),
    'cmd_injection': (r"(;|\|\|?|&&|`|\$\()\s*(cat|ls|id|whoami|uname|wget|curl|nc|ping)\b", 'injection'),
    'xss': (r"<\s*script|javascript:|\bon(error|load|mouseover)\s*=", 'xss'),
}
_COMPILED = {name: re.compile(p, re.IGNORECASE) for name, (p, _) in SIGNATURES.items()}


def forward_text(segments, dirs, placeholder: str) -> str:
    """Forward (initiator) segments as one URL-decoded text."""
    parts = [to_text(bytes(s), placeholder) for s, d in zip(segments, dirs) if d == 0 and len(s)]
    return unquote_plus(' '.join(parts)) if parts else ''


def signature_hits(flows: pd.DataFrame, placeholder: str) -> pd.DataFrame:
    """One boolean column per signature, indexed like `flows`."""
    texts = [forward_text(s, d, placeholder) for s, d in zip(flows['payload_segments'], flows['payload_dirs'])]
    return pd.DataFrame({name: [bool(rx.search(t)) for t in texts] for name, rx in _COMPILED.items()},
                        index=flows.index)


def signature_crosstab(matched: pd.DataFrame, placeholder: str) -> pd.DataFrame:
    """Per signature: flows hit, how many were matched, and the label distribution they got."""
    hits = signature_hits(matched, placeholder)
    ok = matched['match_status'].isin(MATCHED)
    rows = []
    for name, (_, expected) in SIGNATURES.items():
        h = hits[name]
        got = matched.loc[h & ok, 'raw_label'].value_counts()
        n_ok = int((h & ok).sum())
        rows.append({
            'signature': name,
            'expected_label': expected,
            'flows_hit': int(h.sum()),
            'matched': n_ok,
            'unmatched': int((h & ~ok).sum()),
            'agreement': float(got.get(expected, 0) / n_ok) if n_ok else np.nan,
            **{f'label_{k}': int(v) for k, v in got.items()},
        })
    out = pd.DataFrame(rows)
    label_cols = [c for c in out.columns if c.startswith('label_')]
    out[label_cols] = out[label_cols].fillna(0).astype(int)
    return out


def capture_spans(flows: pd.DataFrame, offset_s: float, tolerance_s: float) -> List[tuple]:
    g = flows.groupby('capture_id').agg(lo=('first_ts', 'min'), hi=('last_ts', 'max'))
    return [(lo + offset_s - tolerance_s, hi + offset_s + tolerance_s) for lo, hi in zip(g['lo'], g['hi'])]


def class_coverage(matched: pd.DataFrame, labels: pd.DataFrame, offset_s: float, tolerance_s: float) -> pd.DataFrame:
    """Per raw label, over label rows inside the capture spans: rows claimed by >= 1 matched flow."""
    ts = labels['ts'].to_numpy()
    inside = np.zeros(len(labels), dtype=bool)
    for lo, hi in capture_spans(matched, offset_s, tolerance_s):
        inside |= (ts >= lo) & (ts <= hi)
    lab = labels.loc[inside, ['row_id', 'raw_label']].copy()
    claimed = set(matched.loc[matched['match_status'].isin(MATCHED), 'label_row_id'])
    lab['claimed'] = lab['row_id'].isin(claimed)
    out = lab.groupby('raw_label')['claimed'].agg(rows_in_span='size', row_coverage='mean')
    flows = matched[matched['match_status'].isin(MATCHED)].groupby('raw_label').size().rename('flows')
    return out.join(flows, how='outer').fillna({'rows_in_span': 0, 'flows': 0}).reset_index().rename(
        columns={'index': 'raw_label'})


def _duration_bucket(d: pd.Series) -> pd.Series:
    return pd.cut(d, [-np.inf, 1, 60, 120, np.inf], labels=['<1s', '1-60s', '60-120s', '>120s']).astype(str)


def failure_breakdown(matched: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    bad = matched[~matched['match_status'].isin(MATCHED)]
    if bad.empty:
        return pd.DataFrame(columns=['match_status', 'proto', 'end_reason', 'port_class', 'duration',
                                     'key_in_labels', 'flows', 'share_of_all_flows'])
    df = pd.DataFrame({
        'match_status': bad['match_status'],
        'proto': bad['proto'].map({6: 'tcp', 17: 'udp', 1: 'icmp', 58: 'icmp6'}).fillna('other'),
        'end_reason': bad['end_reason'].fillna('none'),
        'port_class': np.where(bad['dst_port_id'].between(1, 1023), 'well_known', 'other'),
        'duration': _duration_bucket(bad['duration']),
        'key_in_labels': bad['key'].isin(set(labels['key'])),
    })
    out = df.value_counts().rename('flows').reset_index()
    out['share_of_all_flows'] = out['flows'] / len(matched)
    return out


def gate(summary: Dict, coverage: pd.DataFrame, crosstab: pd.DataFrame, thresholds) -> Dict:
    """Checks with a null threshold are skipped. Every judged value is written out."""
    t = dict(thresholds or {})
    checks = []

    def check(name, value, threshold, higher_is_better=True):
        if threshold is None or value is None or (isinstance(value, float) and np.isnan(value)):
            checks.append({'check': name, 'value': value, 'threshold': threshold, 'ok': None})
            return
        ok = value >= threshold if higher_is_better else value <= threshold
        checks.append({'check': name, 'value': float(value), 'threshold': threshold, 'ok': bool(ok)})

    check('match_rate', summary.get('match_rate'), t.get('min_match_rate'))
    check('ambiguous_rate', summary.get('ambiguous_rate'), t.get('max_ambiguous_rate'), higher_is_better=False)
    min_rows = int(t.get('min_rows_per_class') or 0)
    for r in coverage.itertuples(index=False):
        if r.rows_in_span >= max(min_rows, 1):
            check(f'row_coverage[{r.raw_label}]', r.row_coverage, t.get('min_class_row_coverage'))
    min_sig = int(t.get('min_signature_flows') or 0)
    for r in crosstab.itertuples(index=False):
        if r.matched >= max(min_sig, 1):
            check(f'signature_agreement[{r.signature}]', r.agreement, t.get('min_signature_agreement'))

    judged = [c['ok'] for c in checks if c['ok'] is not None]
    return {'pass': bool(judged) and all(judged), 'judged': len(judged), 'checks': checks}


def write_md(path: str, summary: Dict, coverage: pd.DataFrame, crosstab: pd.DataFrame,
             breakdown: pd.DataFrame, result: Dict):
    lines = ['# Matching checks (E0.3)', '',
             f"**Gate: {'PASS' if result['pass'] else 'FAIL'}** ({result['judged']} checks judged)", '',
             pd.DataFrame(result['checks']).to_markdown(index=False) if result['checks'] else '(no checks)', '',
             '## Payload signature vs matched label', '',
             'Flows whose forward payload carries an attack signature, and the label matching gave them.', '',
             crosstab.to_markdown(index=False) if len(crosstab) else '(none)', '',
             '## Label-row coverage inside the capture time spans', '',
             coverage.to_markdown(index=False) if len(coverage) else '(none)', '',
             '## Unmatched and ambiguous flows (top 20)', '',
             'key_in_labels = the 5-tuple exists in the labels at some time (time problem, not a tuple problem).', '',
             breakdown.head(20).to_markdown(index=False) if len(breakdown) else '(none)', '']
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
