"""Phase 2 local training / inference on one client's graphs.

Each step runs the model over the whole client graph (small at sprint scale,
and exact: no neighbour sampling) and takes the loss on a mini-batch of target
flow nodes, so one local epoch = one pass over all training flows in batches
of `phase2.batch_size`. Loss: CE weighted with the shared Laplace-smoothed
inverse-frequency rule over classes present on the client (Eq. 10).
"""

import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fhf.common.utils import class_weights
from fhf.federated.client import EvalResult, FederatedClient
from fhf.phase2.build_heterograph import ClientGraph
from fhf.phase2.hetero_sage import gnn_forward


def get_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_state(model: nn.Module, state: Dict[str, torch.Tensor]):
    model.load_state_dict(state)


class GraphClient(FederatedClient):
    """A client holding one graph per role; `forward` maps (model, graph data) -> flow logits."""

    def __init__(self, cid: int, model: nn.Module, graphs: Dict[str, ClientGraph], cfg, num_classes: int,
                 device, seed: int, forward=gnn_forward, train_cfg=None):
        self.cid = cid
        self.model = model
        self.cfg = cfg
        self.tcfg = train_cfg or cfg.phase2
        self.device = device
        self.seed = seed
        self.forward = forward
        self.num_classes = num_classes
        self.graphs = {role: _to(g, device) for role, g in graphs.items()}
        self.n_train = self.graphs['train'].num_flows if 'train' in self.graphs else 0
        self.n_val = self.graphs['val'].num_flows if 'val' in self.graphs else 0
        w = class_weights(graphs['train'].y.numpy(), num_classes, cfg.training.class_weight.get('max_weight'))
        self.weights = torch.tensor(w, dtype=torch.float32, device=device)

    def fit(self, global_state, round_idx):
        m = self.model
        set_state(m, global_state)
        g = self.graphs['train']
        if g.num_flows == 0:
            return get_state(m), {'train_loss': float('nan'), 'lr': 0.0}
        opt = torch.optim.AdamW(m.parameters(), lr=float(self.tcfg.lr), weight_decay=float(self.tcfg.weight_decay))
        gen = torch.Generator().manual_seed(self.seed * 1000003 + round_idx * 101 + self.cid)
        bs = int(self.tcfg.batch_size)
        m.train()
        total, seen = 0.0, 0
        for _ in range(int(self.tcfg.local_epochs)):
            perm = torch.randperm(g.num_flows, generator=gen).to(self.device)
            for i in range(0, g.num_flows, bs):
                idx = perm[i:i + bs]
                logits = self.forward(m, g.data)
                loss = F.cross_entropy(logits[idx], g.y[idx], weight=self.weights)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), float(self.cfg.training.grad_clip))
                opt.step()
                total += loss.item() * len(idx)
                seen += len(idx)
        return get_state(m), {'train_loss': total / max(seen, 1), 'lr': float(self.tcfg.lr)}

    @torch.no_grad()
    def evaluate(self, global_state, role):
        return predict(self.model, global_state, self.graphs.get(role), self.forward)


@torch.no_grad()
def predict(model: nn.Module, state, graph: Optional[ClientGraph], forward=gnn_forward) -> EvalResult:
    if graph is None or graph.num_flows == 0:
        return EvalResult(np.array([], int), np.array([], int), float('nan'), None)
    if state is not None:
        set_state(model, state)
    model.eval()
    logits = forward(model, graph.data).float()
    loss = F.cross_entropy(logits, graph.y).item()
    probs = torch.softmax(logits, -1).cpu().numpy()
    return EvalResult(graph.y.cpu().numpy(), probs.argmax(1), loss, probs)


def _to(graph: ClientGraph, device) -> ClientGraph:
    g = copy.copy(graph)
    g.data = graph.data.to(device)
    g.y = graph.y.to(device)
    return g
