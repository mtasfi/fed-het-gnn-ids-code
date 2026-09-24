"""Per-client heterogeneous flow-payload graphs (Methodology §7).

Nodes:  flow     x = x^f_i (federated-scaled) || m_i
        payload  x = e_ij (frozen encoder embedding), only for flows with m_i = 1
Edges:  (flow, contains, payload)      flow -> each of its payload segments
        (payload, next, payload)       segment j -> j+1 of the same flow, capture order
        (flow, shares_host, flow)      same source IP, |dt| <= window, at most kappa per flow
Reverse directions are added with ToUndirected, so flow nodes receive payload
messages (rev_contains) and every relation is used in both directions.

Separate graphs are built per client and per role (train / val / test): test
flows never enter a training graph (inductive evaluation). IP addresses are
used only inside `shares_host_edges` and never become features.

Variant switches (config `graph.*`): use_payload (A1), use_next (A6),
use_shares_host (A5), homogeneous (A4); plus `payload_mask` for the E4 sweep.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T

logger = logging.getLogger(__name__)

FLOW, PAYLOAD = 'flow', 'payload'
CONTAINS = (FLOW, 'contains', PAYLOAD)
NEXT = (PAYLOAD, 'next', PAYLOAD)
SHARES_HOST = (FLOW, 'shares_host', FLOW)
REV_CONTAINS = (PAYLOAD, 'rev_contains', FLOW)


# ------------------------------------------------------------------ shares_host
def shares_host_edges(src_ip, ts, window_s: float, kappa: Optional[int], max_offsets: int = 1000) -> np.ndarray:
    """Undirected same-source edges within `window_s`, strictly capped at `kappa`
    neighbours per flow. Candidates are each flow's nearest same-source successors
    in time; they are accepted greedily in order of increasing |dt| while both
    endpoints are below the cap. Returns a [2, E] array with both directions."""
    n = len(ts)
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64)
    codes, _ = pd.factorize(pd.Series(src_ip).astype(str))
    ts = np.asarray(ts, dtype=np.float64)
    order = np.lexsort((ts, codes))
    c_sorted, t_sorted = codes[order], ts[order]

    limit = kappa if kappa is not None else max_offsets
    us, vs, dts = [], [], []
    for off in range(1, min(limit, n - 1) + 1):
        same = c_sorted[off:] == c_sorted[:-off]
        dt = t_sorted[off:] - t_sorted[:-off]
        ok = same & (dt <= window_s)
        if not ok.any():
            break
        idx = np.nonzero(ok)[0]
        us.append(order[idx])
        vs.append(order[idx + off])
        dts.append(dt[idx])
    if not us:
        return np.zeros((2, 0), dtype=np.int64)
    u, v, dt = np.concatenate(us), np.concatenate(vs), np.concatenate(dts)

    if kappa is not None:
        sel = np.argsort(dt, kind='stable')
        degree = np.zeros(n, dtype=np.int64)
        keep = np.zeros(len(sel), dtype=bool)
        for k, e in enumerate(sel):
            a, b = u[e], v[e]
            if degree[a] < kappa and degree[b] < kappa:
                degree[a] += 1
                degree[b] += 1
                keep[k] = True
        u, v = u[sel[keep]], v[sel[keep]]
    return np.stack([np.concatenate([u, v]), np.concatenate([v, u])]).astype(np.int64)


def shares_host_stats(frame: pd.DataFrame, window_s: float, kappas) -> pd.DataFrame:
    """E0 kappa selection: realised degree / density per candidate kappa (None = uncapped)."""
    rows = []
    for kappa in kappas:
        e = shares_host_edges(frame['src_ip'].to_numpy(), frame['first_ts'].to_numpy(), window_s, kappa)
        deg = np.bincount(e[0], minlength=len(frame)) if e.size else np.zeros(len(frame), dtype=int)
        rows.append({
            'kappa': 'none' if kappa is None else kappa,
            'flows': len(frame),
            'undirected_edges': int(e.shape[1] // 2),
            'mean_degree': float(deg.mean()),
            'p95_degree': float(np.percentile(deg, 95)) if len(deg) else 0.0,
            'max_degree': int(deg.max()) if len(deg) else 0,
            'isolated_rate': float((deg == 0).mean()),
            'saturated_rate': float((deg == kappa).mean()) if kappa else 0.0,
            'edges_per_flow': float(e.shape[1] / 2 / max(len(frame), 1)),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ payload masking (E4)
def mask_score(flow_uid: str, seed: int = 0) -> float:
    """Deterministic uniform [0,1) score per flow; masking at rate r masks flows
    with score < r, so the masked sets are nested (25% within 50% within 100%)."""
    h = hashlib.sha1(f'{seed}:{flow_uid}'.encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def payload_mask(flow_uids, has_payload, rate: float, seed: int = 0) -> np.ndarray:
    if rate <= 0:
        return np.zeros(len(flow_uids), dtype=bool)
    scores = np.array([mask_score(u, seed) for u in flow_uids])
    return (np.asarray(has_payload) == 1) & (scores < rate)


# ------------------------------------------------------------------ graph assembly
@dataclass
class ClientGraph:
    data: object                     # HeteroData, or Data when homogeneous
    y: torch.Tensor                  # flow labels
    flow_uid: np.ndarray
    client: int
    role: str
    stats: Dict = field(default_factory=dict)

    @property
    def num_flows(self) -> int:
        return int(self.y.shape[0])


class EmbeddingLookup:
    """Frozen payload embeddings of one encoder tag: flow_uid -> [n_segments, d]."""

    def __init__(self, emb: np.ndarray, seg_flow_uid: np.ndarray, seg_pos: np.ndarray):
        self.emb = emb
        order = np.lexsort((seg_pos, seg_flow_uid))
        self.emb, self.uid, self.pos = emb[order], seg_flow_uid[order], seg_pos[order]
        uniq, start, counts = np.unique(self.uid, return_index=True, return_counts=True)
        self.index = {u: (s, c) for u, s, c in zip(uniq, start, counts)}
        self.dim = emb.shape[1]

    def get(self, flow_uid: str) -> np.ndarray:
        s, c = self.index.get(flow_uid, (0, 0))
        return self.emb[s:s + c]


def build_graph(frame: pd.DataFrame, x_flow: np.ndarray, y: np.ndarray, cfg, emb: Optional[EmbeddingLookup],
                client: int, role: str, mask: Optional[np.ndarray] = None) -> ClientGraph:
    """frame: this client's flows for one role (needs flow_uid, src_ip, first_ts, has_payload).
    x_flow: scaled flow features (m_i is appended here after gating)."""
    g = cfg.graph
    n = len(frame)
    m = frame['has_payload'].to_numpy().astype(np.int64).copy()
    if not g.use_payload or emb is None:
        m[:] = 0                                    # A1: payload channel removed entirely
    if mask is not None:
        m[mask] = 0                                 # E4: masked flows behave exactly like m_i = 0
    x = np.concatenate([x_flow, m[:, None].astype(np.float32)], axis=1).astype(np.float32)

    # payload nodes + contains / next edges
    p_feats, contains_src, contains_dst, next_src, next_dst = [], [], [], [], []
    if g.use_payload and emb is not None:
        pid = 0
        for i, (uid, mi) in enumerate(zip(frame['flow_uid'].to_numpy(), m)):
            if not mi:
                continue
            segs = emb.get(uid)
            for j in range(len(segs)):
                p_feats.append(segs[j])
                contains_src.append(i)
                contains_dst.append(pid)
                if j > 0:
                    next_src.append(pid - 1)
                    next_dst.append(pid)
                pid += 1
    n_payload = len(p_feats)

    sh = (shares_host_edges(frame['src_ip'].to_numpy(), frame['first_ts'].to_numpy(),
                            float(g.shares_host.window_s), g.shares_host.kappa)
          if g.use_shares_host else np.zeros((2, 0), dtype=np.int64))

    stats = {'flows': n, 'payload_nodes': n_payload, 'contains_edges': len(contains_src),
             'next_edges': len(next_src) if g.use_next else 0, 'shares_host_edges_undirected': sh.shape[1] // 2,
             'has_payload_flows': int(m.sum())}

    if g.homogeneous:
        data = _homogeneous(x, p_feats, emb.dim if emb is not None else 0, contains_src, contains_dst,
                            next_src if g.use_next else [], next_dst if g.use_next else [], sh)
    else:
        data = HeteroData()
        data[FLOW].x = torch.from_numpy(x)
        data[FLOW].num_nodes = n
        if n_payload:
            data[PAYLOAD].x = torch.from_numpy(np.stack(p_feats).astype(np.float32))
            data[PAYLOAD].num_nodes = n_payload
            data[CONTAINS].edge_index = torch.tensor([contains_src, contains_dst], dtype=torch.long)
            if g.use_next and next_src:
                data[NEXT].edge_index = torch.tensor([next_src, next_dst], dtype=torch.long)
        if g.use_shares_host and sh.shape[1]:
            data[SHARES_HOST].edge_index = torch.from_numpy(sh)
        if data.edge_types:
            data = T.ToUndirected()(data)   # adds rev_contains; next/shares_host become bidirectional

    return ClientGraph(data=data, y=torch.from_numpy(np.asarray(y, dtype=np.int64)),
                       flow_uid=frame['flow_uid'].to_numpy(), client=client, role=role, stats=stats)


def _homogeneous(x_flow, p_feats, d_e, c_src, c_dst, n_src, n_dst, sh) -> Data:
    """A4: one node type, one relation. Flow and payload features share one input
    space by zero padding, [x^f || m || 0_{d_e}] and [0_{d_f+1} || e]; all edges are
    collapsed into a single undirected edge set. Flow nodes come first."""
    n, d_f1 = x_flow.shape
    n_p = len(p_feats)
    xf = np.concatenate([x_flow, np.zeros((n, d_e), np.float32)], axis=1)
    parts = [xf]
    if n_p:
        parts.append(np.concatenate([np.zeros((n_p, d_f1), np.float32), np.stack(p_feats).astype(np.float32)], axis=1))
    x = np.concatenate(parts, axis=0)
    src = list(c_src) + [n + p for p in n_src] + sh[0].tolist()
    dst = [n + p for p in c_dst] + [n + p for p in n_dst] + sh[1].tolist()
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    data = Data(x=torch.from_numpy(x), edge_index=ei)
    data = T.ToUndirected()(data)
    data.flow_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    data.flow_mask[:n] = True
    return data
