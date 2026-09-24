"""Label adapters: read a dataset's label CSVs into one normalised frame

    row_id, label_file, ts (UTC epoch s), duration_s, proto (int), src_ip, src_port,
    dst_ip, dst_port, raw_label, key (uint64 canonical bidirectional 5-tuple hash)

Each dataset gets one small adapter; everything downstream is dataset-agnostic.
"""

import logging
import os
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROTO_NUM = {'tcp': 6, 'udp': 17, 'icmp': 1, 'ipv6-icmp': 58, 'icmp6': 58}


def canonical_key(proto, src_ip, src_port, dst_ip, dst_port) -> np.ndarray:
    """Direction-free 5-tuple -> uint64. ICMP ports are zeroed on both sides, because
    Zeek stores ICMP type/code in the port fields while the extractor uses 0."""
    proto = pd.Series(proto).astype('int64').to_numpy()
    icmp = np.isin(proto, [1, 58])
    sp = np.where(icmp, 0, pd.Series(src_port).fillna(0).astype('int64').to_numpy())
    dp = np.where(icmp, 0, pd.Series(dst_port).fillna(0).astype('int64').to_numpy())
    a = pd.Series(src_ip).astype(str).str.strip().to_numpy(dtype=object) + ':' + sp.astype(str)
    b = pd.Series(dst_ip).astype(str).str.strip().to_numpy(dtype=object) + ':' + dp.astype(str)
    lo = np.where(a <= b, a, b)
    hi = np.where(a <= b, b, a)
    joined = proto.astype(str).astype(object) + '|' + lo + '|' + hi
    return pd.util.hash_array(joined.astype(object), categorize=False).astype(np.uint64)


def _proto_to_int(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(s, errors='coerce')
    return numeric.fillna(s.map(PROTO_NUM)).fillna(-1).astype('int64')


def _port(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    # hex ports ("0x0303") appear in some Argus/UNSW rows
    hexmask = s.str.startswith('0x')
    out = pd.to_numeric(s.where(~hexmask), errors='coerce')
    if hexmask.any():
        out[hexmask] = s[hexmask].apply(lambda v: int(v, 16))
    return out.fillna(0).astype('int64')


def _finish(df: pd.DataFrame, path: str, offset: int) -> pd.DataFrame:
    df = df.dropna(subset=['ts']).copy()
    df['label_file'] = os.path.basename(path)
    df['row_id'] = [f'{os.path.basename(path)}:{i}' for i in df.index + offset]
    df['raw_label'] = df['raw_label'].fillna('').astype(str).str.strip()
    df['key'] = canonical_key(df['proto'], df['src_ip'], df['src_port'], df['dst_ip'], df['dst_port'])
    return df[['row_id', 'label_file', 'ts', 'duration_s', 'proto', 'src_ip', 'src_port', 'dst_ip', 'dst_port',
               'raw_label', 'key']]


def read_toniot_network(path: str, spec) -> pd.DataFrame:
    cols = spec['columns']
    df = pd.read_csv(path, usecols=list(cols.values()), dtype=str, low_memory=False)
    out = pd.DataFrame({
        'ts': pd.to_numeric(df[cols['ts']], errors='coerce') * (1e-3 if spec.get('ts_unit') == 'ms' else 1.0),
        'duration_s': pd.to_numeric(df[cols['duration']], errors='coerce').fillna(0.0),
        'proto': _proto_to_int(df[cols['proto']]),
        'src_ip': df[cols['src_ip']], 'src_port': _port(df[cols['src_port']]),
        'dst_ip': df[cols['dst_ip']], 'dst_port': _port(df[cols['dst_port']]),
        'raw_label': df[cols['label']],
    })
    return _finish(out, path, 0)


def read_cicids2017(path: str, spec) -> pd.DataFrame:
    cols = spec['columns']
    df = pd.read_csv(path, dtype=str, low_memory=False, encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]
    local = pd.to_datetime(df[cols['ts']].str.strip(), dayfirst=spec.get('ts_format') == 'dayfirst', errors='coerce')
    tz = spec.get('timezone')
    utc = local.dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC') if tz else local
    dur_scale = {'us': 1e-6, 'ms': 1e-3, 's': 1.0}[spec.get('duration_unit', 'us')]
    out = pd.DataFrame({
        'ts': utc.astype('int64') / 1e9,
        'duration_s': pd.to_numeric(df[cols['duration']], errors='coerce').fillna(0.0) * dur_scale,
        'proto': _proto_to_int(df[cols['proto']]),
        'src_ip': df[cols['src_ip']], 'src_port': _port(df[cols['src_port']]),
        'dst_ip': df[cols['dst_ip']], 'dst_port': _port(df[cols['dst_port']]),
        'raw_label': df[cols['label']],
    })
    out.loc[utc.isna(), 'ts'] = np.nan
    return _finish(out, path, 0)


def read_unsw_nb15(path: str, spec) -> pd.DataFrame:
    cols = spec['columns']
    df = pd.read_csv(path, header=None, dtype=str, low_memory=False, encoding='latin-1')
    label = df[cols['label']].fillna('').str.strip().replace('', 'normal')
    out = pd.DataFrame({
        'ts': pd.to_numeric(df[cols['ts']], errors='coerce'),
        'duration_s': pd.to_numeric(df[cols['duration']], errors='coerce').fillna(0.0),
        'proto': _proto_to_int(df[cols['proto']]),
        'src_ip': df[cols['src_ip']], 'src_port': _port(df[cols['src_port']]),
        'dst_ip': df[cols['dst_ip']], 'dst_port': _port(df[cols['dst_port']]),
        'raw_label': label,
    })
    return _finish(out, path, 0)


ADAPTERS: Dict[str, Callable] = {
    'toniot_network': read_toniot_network,
    'cicids2017': read_cicids2017,
    'unsw_nb15': read_unsw_nb15,
}


def load_labels(cfg, label_files: List[str]) -> pd.DataFrame:
    spec = cfg.dataset.label_source
    kind = spec['kind']
    if kind not in ADAPTERS:
        raise ValueError(f"Unknown label_source.kind '{kind}'. Options: {sorted(ADAPTERS)}")
    if not label_files:
        raise FileNotFoundError(f"No label files matched {cfg.dataset.source.label_globs}")
    frames = []
    for path in label_files:
        frame = ADAPTERS[kind](path, spec)
        logger.info(f"  {os.path.basename(path)}: {len(frame):,} label rows")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def map_labels(raw: pd.Series, cfg) -> pd.Series:
    """raw label -> canonical class; unknown raw values fail loudly (plan: every
    retained class needs an auditable mapping)."""
    mapping = dict(cfg.dataset.label_map)
    merges = dict(cfg.dataset.get('merge_classes') or {})
    unknown = sorted(set(raw.unique()) - set(mapping))
    if unknown:
        raise ValueError(
            f"Raw labels without a mapping in configs/datasets/{cfg.dataset.name}.yaml label_map: {unknown}")
    mapped = raw.map(mapping)
    return mapped.map(lambda c: merges.get(c, c))
