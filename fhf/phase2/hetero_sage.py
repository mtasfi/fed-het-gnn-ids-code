"""Hetero-GraphSAGE (Methodology §8.1, Eq. 6-9) and the A4 homogeneous counterpart.

    h_v^(0)   = W_tau(v) x_v                                      per-type projection
    m_{v,r}   = MEAN_{u in N_r(v)} h_u^(l)                        per relation
    h_v^(l+1) = ReLU( W_self,tau(v) h_v^(l) + sum_r W_r m_{v,r} )   HinSAGE-style self term per type
    y_hat_i   = W_c h_{v_i}^(L) + b_c                              flow nodes only

Relations are fixed at construction (all four, both directions via ToUndirected),
so every client holds the same parameter set for FedAvg even when a graph lacks
a relation (e.g. A1 has no payload nodes): absent relations are skipped in the
forward pass and their weights just stay at the broadcast values.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from fhf.phase2.build_heterograph import CONTAINS, FLOW, NEXT, PAYLOAD, REV_CONTAINS, SHARES_HOST

RELATIONS = (CONTAINS, REV_CONTAINS, NEXT, SHARES_HOST)


def _rel_key(rel: Tuple[str, str, str]) -> str:
    return '__'.join(rel)


class HeteroSAGE(nn.Module):
    def __init__(self, flow_dim: int, payload_dim: int, hidden: int, layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.proj = nn.ModuleDict({FLOW: nn.Linear(flow_dim, hidden), PAYLOAD: nn.Linear(max(payload_dim, 1), hidden)})
        self.self_lin = nn.ModuleList(
            [nn.ModuleDict({FLOW: nn.Linear(hidden, hidden), PAYLOAD: nn.Linear(hidden, hidden)}) for _ in range(layers)])
        self.rel_conv = nn.ModuleList(
            [nn.ModuleDict({_rel_key(r): SAGEConv((hidden, hidden), hidden, aggr='mean', root_weight=False, bias=False)
                            for r in RELATIONS}) for _ in range(layers)])
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> torch.Tensor:
        h = {t: self.proj[t](x) for t, x in x_dict.items() if x is not None and x.shape[0] > 0}
        for layer, (self_lin, convs) in enumerate(zip(self.self_lin, self.rel_conv)):
            out = {t: self_lin[t](v) for t, v in h.items()}
            for rel, ei in edge_index_dict.items():
                src, _, dst = rel
                key = _rel_key(rel)
                if key not in convs or src not in h or dst not in h or ei.numel() == 0:
                    continue
                out[dst] = out[dst] + convs[key]((h[src], h[dst]), ei)
            h = {t: F.dropout(F.relu(v), self.dropout, self.training) for t, v in out.items()}
        return self.head(h[FLOW])


class HomoSAGE(nn.Module):
    """A4: one node type, one relation, same depth / width / head as HeteroSAGE."""

    def __init__(self, in_dim: int, hidden: int, layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden, aggr='mean', root_weight=True) for _ in range(layers)])
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, flow_mask):
        h = self.proj(x)
        for conv in self.convs:
            h = F.dropout(F.relu(conv(h, edge_index)), self.dropout, self.training)
        return self.head(h[flow_mask])


def build_gnn(cfg, flow_dim: int, payload_dim: int, num_classes: int) -> nn.Module:
    g = cfg.gnn
    if cfg.graph.homogeneous:
        return HomoSAGE(flow_dim + payload_dim, int(g.hidden_dim), int(g.layers), num_classes, float(g.dropout))
    return HeteroSAGE(flow_dim, payload_dim, int(g.hidden_dim), int(g.layers), num_classes, float(g.dropout))


def gnn_forward(model: nn.Module, data) -> torch.Tensor:
    """Flow-node logits for either model type."""
    if isinstance(model, HomoSAGE):
        return model(data.x, data.edge_index, data.flow_mask)
    # a client graph may legitimately have no edges at all (e.g. A1 with an isolated
    # validation window); PyG's edge_index_dict raises in that case
    edges = {et: data[et].edge_index for et in data.edge_types if 'edge_index' in data[et]}
    return model(data.x_dict, edges)
