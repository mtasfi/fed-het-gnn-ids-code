"""E0.1 / E0.2 reports: label mapping, has_payload decision per flow, per-class
payload availability, reason counts, and the train/test template-overlap diagnostic.

Reports contain counts and rates only; no payload text is written to them.
"""

import logging
import re
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from fhf.data.label_sources import map_labels
from fhf.data.payload_rules import FLOW_REASONS, decide

logger = logging.getLogger(__name__)

ENCRYPTED_REASONS = ('tls_record', 'high_entropy')


def apply_payload_rule(matched: pd.DataFrame, cfg) -> pd.DataFrame:
    """Matched flows -> canonical label + has_payload, reason and readable segment texts."""
    ok = matched[matched['match_status'].isin(['matched', 'matched_consistent'])].copy()
    ok['label'] = map_labels(ok['raw_label'], cfg)
    excluded = set(cfg.dataset.get('exclude_classes') or [])
    if excluded:
        ok = ok[~ok['label'].isin(excluded)]

    rule = cfg.payload
    decisions = [decide(segs, rule) for segs in ok['payload_segments']]
    ok['has_payload'] = np.array([d.has_payload for d in decisions], dtype=np.int8)
    ok['payload_reason'] = [d.reason for d in decisions]
    ok['payload_texts'] = [d.texts for d in decisions]
    ok['payload_segment_reasons'] = [d.segment_reasons for d in decisions]
    ok['payload_truncated'] = [d.truncated for d in decisions]
    ok['app_bytes'] = ok['fwd_payload_bytes'] + ok['bwd_payload_bytes']
    return ok.drop(columns=['payload_segments'])


def label_mapping_report(matched: pd.DataFrame, labels: pd.DataFrame, cfg, path: str):
    raw_counts = labels['raw_label'].value_counts()
    mapping = dict(cfg.dataset.label_map)
    matched_ok = matched[matched['match_status'].isin(['matched', 'matched_consistent'])]
    matched_rows = matched_ok.drop_duplicates('label_row_id')['raw_label'].value_counts()
    out = {
        'dataset': cfg.dataset.name,
        'excluded_classes': list(cfg.dataset.get('exclude_classes') or []),
        'merge_classes': dict(cfg.dataset.get('merge_classes') or {}),
        'raw_labels': {
            str(raw): {
                'canonical': mapping.get(raw, 'UNMAPPED'),
                'label_rows': int(n),
                'label_rows_matched_by_a_flow': int(matched_rows.get(raw, 0)),
                'row_coverage': float(matched_rows.get(raw, 0) / n) if n else 0.0,
            } for raw, n in raw_counts.items()
        },
    }
    with open(path, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
    return out


def payload_availability(labeled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls, g in labeled.groupby('label'):
        n = len(g)
        reasons = g['payload_reason'].value_counts()
        seg_reasons = Counter(r for rs in g['payload_segment_reasons'] for r in rs)
        n_seg = sum(seg_reasons.values())
        rows.append({
            'label': cls,
            'flows': n,
            'flows_with_app_bytes': int((g['app_bytes'] > 0).sum()),
            'app_bytes_median': float(g['app_bytes'].median()),
            'has_payload_rate': float(g['has_payload'].mean()),
            'readable_segment_rate': seg_reasons.get('ok', 0) / n_seg if n_seg else 0.0,
            'encrypted_unreadable_flow_rate': float(g['payload_reason'].isin(ENCRYPTED_REASONS).mean()),
            'mean_readable_segments': float(g['payload_texts'].apply(len).mean()),
            'truncated_segment_flows': int(g['payload_truncated'].sum()),
            **{f'reason_{r}': int(reasons.get(r, 0)) for r in FLOW_REASONS if r not in ('masked', 'disabled')},
        })
    return pd.DataFrame(rows).sort_values('flows', ascending=False)


def reason_counts(labeled: pd.DataFrame) -> pd.DataFrame:
    flow = labeled['payload_reason'].value_counts().rename('flows')
    seg = pd.Series(Counter(r for rs in labeled['payload_segment_reasons'] for r in rs), name='segments')
    return pd.concat([flow, seg], axis=1).fillna(0).astype(int).rename_axis('reason').reset_index()


# ---------------------------------------------------------------- template overlap
_HEX = re.compile(r'\b[0-9a-fA-F]{8,}\b')
_DIGITS = re.compile(r'\d+')
_WS = re.compile(r'\s+')

TEMPLATE_RULE = ("template = first line of the first readable payload segment; runs of >= 8 hex digits -> 'H'; "
                 "digit runs -> '0'; whitespace runs -> one space; case preserved; truncated to 256 characters")


def payload_template(texts) -> str:
    if texts is None or len(texts) == 0:
        return None
    first_line = texts[0].split('\n', 1)[0]
    t = _HEX.sub('H', first_line)
    t = _DIGITS.sub('0', t)
    return _WS.sub(' ', t).strip()[:256]


def template_overlap(labeled: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    df = split[['flow_uid', 'split']].merge(labeled[['flow_uid', 'label', 'payload_texts']], on='flow_uid')
    df['template'] = df['payload_texts'].apply(payload_template)
    df = df.dropna(subset=['template'])
    rows = []
    for cls, g in df.groupby('label'):
        train_t = set(g.loc[g.split == 'train', 'template'])
        test = g[g.split == 'test']
        test_t = set(test['template'])
        overlap = train_t & test_t
        rows.append({
            'label': cls,
            'train_payload_flows': int((g.split == 'train').sum()),
            'test_payload_flows': int(len(test)),
            'train_templates': len(train_t),
            'test_templates': len(test_t),
            'overlapping_templates': len(overlap),
            'template_overlap_rate': len(overlap) / len(test_t) if test_t else 0.0,
            'test_flows_with_seen_template_rate': float(test['template'].isin(train_t).mean()) if len(test) else 0.0,
        })
    return pd.DataFrame(rows)


def write_audit_md(path: str, cfg, extract_stats: Dict, match: Dict, avail: pd.DataFrame, decision: str):
    lines = [f"# E0 dataset audit: {cfg.dataset.name}", "",
             f"Source kind: `{cfg.dataset.source.kind}`, root: `{cfg.dataset.source.root}`", "",
             "## Extraction", ""]
    lines += [f"- {k}: {v}" for k, v in extract_stats.items()]
    lines += ["", "## Flow to label matching", ""]
    lines += [f"- {k}: {v}" for k, v in match.items()]
    lines += ["", "## Payload availability by class", "", avail.to_markdown(index=False) if len(avail) else '(none)',
              "", "## Payload-usable decision", "", decision, ""]
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
