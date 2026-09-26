"""Flow-feature matrix used by every model (x^f_i; m_i is appended by the callers).

IP addresses and other identity columns are never features: they are rejected
here by name, so no model code path can receive them (plan pitfall 3).
"""

import re
from typing import List

import numpy as np
import pandas as pd

from fhf.data.flow_extractor import FEATURE_COLUMNS

_IP_NAME = re.compile(r'(^|_)ip(v4|v6)?($|_)', re.I)


# NetFlow-style feature set (features.set = netflow): the non-identity fields of NF-ToN-IoT
# (IN/OUT_BYTES, IN/OUT_PKTS, TCP_FLAGS, FLOW_DURATION, PROTOCOL, L4_DST_PORT), rebuilt from our flows.
# L4_SRC_PORT is left out (an identity field in this method) and L7_PROTO (nDPI) is not available.
# Used to measure how much of the flow-only strength comes from the 50 extractor features.
NETFLOW_COLUMNS = ['in_bytes', 'out_bytes', 'in_pkts', 'out_pkts', 'tcp_flags', 'duration',
                   'proto_tcp', 'proto_udp', 'proto_icmp', 'dst_port']
_FLAG_BITS = {'fin_cnt': 1, 'syn_cnt': 2, 'rst_cnt': 4, 'psh_cnt': 8, 'ack_cnt': 16, 'urg_cnt': 32}


def netflow_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['in_bytes'], out['out_bytes'] = df['fwd_bytes'], df['bwd_bytes']
    out['in_pkts'], out['out_pkts'] = df['fwd_pkts'], df['bwd_pkts']
    # NetFlow TCP_FLAGS is the OR of all flags seen in the flow, not per-flag counts
    out['tcp_flags'] = sum((df[c] > 0).astype(np.int64) * bit for c, bit in _FLAG_BITS.items()).astype(np.float64)
    for c in ('duration', 'proto_tcp', 'proto_udp', 'proto_icmp', 'dst_port'):
        out[c] = df[c]
    return out


def feature_columns(cfg) -> List[str]:
    forbidden = set(cfg.features.forbidden_columns)
    if cfg.features.get('set', 'full') == 'netflow':
        return list(NETFLOW_COLUMNS)
    cols = [c for c in FEATURE_COLUMNS if c not in set(cfg.features.get('exclude') or [])]
    bad = [c for c in cols if c in forbidden or _IP_NAME.search(c)]
    if bad:
        raise ValueError(f"Identity columns requested as model features: {bad}")
    return cols


def feature_matrix(df: pd.DataFrame, cfg) -> np.ndarray:
    if cfg.features.get('set', 'full') == 'netflow':
        return netflow_frame(df)[NETFLOW_COLUMNS].to_numpy(dtype=np.float64)
    cols = feature_columns(cfg)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Flow table lacks feature columns {missing}")
    return df[cols].to_numpy(dtype=np.float64)
