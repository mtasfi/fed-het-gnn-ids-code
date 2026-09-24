"""The single federated round loop shared by every federated model.

Round t:  broadcast global state -> each client trains locally (E local epochs)
          -> aggregate (FedAvg by default) -> every client scores the NEW global
          state on its own validation split -> selection score = sample-weighted
          mean of client validation macro-F1 -> checkpoint.
The returned model is the best-scoring round (plan: select by validation
macro-F1, test exactly once afterwards). Every round is checkpointed so a
Kaggle session limit costs at most one round (resume with --resume).
"""

import copy
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from fhf.common.metrics import classification_report, weighted_mean
from fhf.common.utils import state_bytes
from fhf.federated.client import FederatedClient

logger = logging.getLogger(__name__)


@dataclass
class FedResult:
    best_state: Dict[str, torch.Tensor]
    best_round: int
    best_score: float
    final_round: int
    final_state: Dict[str, torch.Tensor]
    history: List[Dict] = field(default_factory=list)
    bytes_up_total: int = 0
    bytes_down_total: int = 0
    bytes_up_per_client_round: float = 0.0
    bytes_down_per_client_round: float = 0.0


def run_federated(clients: List[FederatedClient], init_state: Dict[str, torch.Tensor], rounds: int,
                  aggregator: Callable, rundir, phase: str, num_classes: int, ckpt_path: Optional[str] = None,
                  early_stopping: Optional[Dict] = None, resume: bool = False,
                  log_clients: bool = True) -> FedResult:
    names = [str(i) for i in range(num_classes)]
    state = init_state
    best_state, best_round, best_score = copy.deepcopy(init_state), 0, -np.inf
    history: List[Dict] = []
    prev_scores = None
    start = 1
    up_total = down_total = 0
    patience = (early_stopping or {}).get('patience')
    min_delta = float((early_stopping or {}).get('min_delta', 0.0))
    since_best = 0

    if resume and ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state, best_state = ck['state'], ck['best_state']
        best_round, best_score, history = ck['best_round'], ck['best_score'], ck['history']
        prev_scores, start = ck.get('prev_scores'), ck['round'] + 1
        up_total, down_total, since_best = ck.get('up_total', 0), ck.get('down_total', 0), ck.get('since_best', 0)
        logger.info(f"[phase {phase}] resumed after round {ck['round']} (best round {best_round}, score {best_score:.4f})")

    last_round = start - 1
    for rnd in range(start, rounds + 1):
        t0 = time.perf_counter()
        states, counts, losses, lrs = [], [], [], []
        down = state_bytes(state)
        for client in clients:
            new_state, metrics = client.fit(state, rnd)
            states.append(new_state)
            counts.append(client.n_train)
            losses.append(metrics.get('train_loss', np.nan))
            lrs.append(metrics.get('lr', np.nan))
        bytes_down = down * len(clients)
        bytes_up = sum(state_bytes(s) for s in states)
        up_total += bytes_up
        down_total += bytes_down

        state = aggregator(states, counts, scores=prev_scores)

        f1s, n_vals, vlosses = [], [], []
        for client, tl in zip(clients, losses):
            res = client.evaluate(state, 'val')
            f1 = classification_report(res.y_true, res.y_pred, names)['summary']['macro_f1'] if len(res.y_true) else np.nan
            f1s.append(f1)
            n_vals.append(len(res.y_true))
            vlosses.append(res.loss)
            if log_clients:
                rundir.log_round({'phase': phase, 'round': rnd, 'scope': f'client{client.cid}', 'train_loss': tl,
                                  'val_loss': res.loss, 'val_macro_f1': f1, 'n_train': client.n_train,
                                  'n_val': len(res.y_true)})
        score = weighted_mean(f1s, n_vals)
        prev_scores = f1s

        improved = score > best_score + min_delta or best_round == 0
        if improved:
            best_state, best_round, best_score = copy.deepcopy(state), rnd, score
            since_best = 0
        else:
            since_best += 1

        row = {'phase': phase, 'round': rnd, 'scope': 'global',
               'train_loss': weighted_mean(losses, counts), 'val_loss': weighted_mean(vlosses, n_vals),
               'val_macro_f1': score, 'lr': float(np.nanmean(lrs)) if lrs else np.nan, 'clients': len(clients),
               'bytes_up': bytes_up, 'bytes_down': bytes_down, 'elapsed_s': time.perf_counter() - t0,
               'best_round': best_round}
        rundir.log_round(row)
        history.append(row)
        logger.info(f"[phase {phase}] round {rnd}/{rounds}: train_loss {row['train_loss']:.4f} "
                    f"val_macro_f1 {score:.4f} (best {best_score:.4f} @ {best_round}) {row['elapsed_s']:.1f}s")

        if ckpt_path:
            tmp = ckpt_path + '.tmp'
            torch.save({'round': rnd, 'state': state, 'best_state': best_state, 'best_round': best_round,
                        'best_score': best_score, 'history': history, 'prev_scores': prev_scores,
                        'up_total': up_total, 'down_total': down_total, 'since_best': since_best}, tmp)
            os.replace(tmp, ckpt_path)
        last_round = rnd
        if patience is not None and since_best >= int(patience):
            logger.info(f"[phase {phase}] early stop: {since_best} rounds without {min_delta} improvement")
            break

    n_rounds = max(last_round, 1)
    return FedResult(best_state=best_state, best_round=best_round, best_score=float(best_score),
                     final_round=last_round, final_state=state, history=history,
                     bytes_up_total=up_total, bytes_down_total=down_total,
                     bytes_up_per_client_round=up_total / (n_rounds * len(clients)),
                     bytes_down_per_client_round=down_total / (n_rounds * len(clients)))
