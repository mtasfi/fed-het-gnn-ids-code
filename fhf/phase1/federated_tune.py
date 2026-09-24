"""Phase 1: federated LoRA fine-tuning of the payload encoder (Methodology §5).

Weak supervision: every readable payload segment inherits its flow's label.
Loss: CE weighted with the shared Laplace-smoothed inverse-frequency rule over
the classes present on the client (Eq. 3). Both the LoRA adapters Delta and
the temporary head W_c are broadcast and FedAvg-aggregated every round (Eq. 4);
the head is discarded after the best round (phi*) is selected, except that A9
evaluates it once on test payloads first.
"""

import logging
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fhf.common.utils import class_weights
from fhf.federated.client import EvalResult, FederatedClient
from fhf.phase1.lora_model import PayloadEncoder

logger = logging.getLogger(__name__)


def segment_table(frame: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """One row per readable segment: flow_uid, seg_pos, text, y, client, role."""
    rows = frame.loc[frame['has_payload'] == 1, ['flow_uid', 'client', 'role', 'payload_texts']].copy()
    rows['y'] = y[frame['has_payload'].to_numpy() == 1]
    rows = rows.explode('payload_texts').dropna(subset=['payload_texts'])
    rows['seg_pos'] = rows.groupby('flow_uid').cumcount()
    return rows.rename(columns={'payload_texts': 'text'})[['flow_uid', 'seg_pos', 'text', 'y', 'client', 'role']] \
        .reset_index(drop=True)


def warmup_linear(step: int, warmup: int, total: int) -> float:
    """Linear warm-up to the base LR, then linear decay to 0 over the local steps."""
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def _batches(n: int, size: int, rng: np.random.Generator = None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


class Phase1Client(FederatedClient):
    def __init__(self, cid: int, model: PayloadEncoder, train: pd.DataFrame, val: pd.DataFrame, cfg,
                 num_classes: int, seed: int):
        self.cid = cid
        self.model = model
        self.train = train.reset_index(drop=True)
        self.val = val.reset_index(drop=True)
        self.cfg = cfg
        self.num_classes = num_classes
        self.seed = seed
        self.n_train = len(self.train)
        self.n_val = len(self.val)
        w = class_weights(self.train['y'].to_numpy(), num_classes, cfg.training.class_weight.get('max_weight'))
        self.weights = torch.tensor(w, dtype=torch.float32, device=model.device)

    def fit(self, global_state, round_idx):
        m, p1 = self.model, self.cfg.phase1
        m.load_trainable_state(global_state)
        if self.n_train == 0:
            return m.trainable_state(), {'train_loss': float('nan'), 'lr': 0.0}
        rng = np.random.default_rng(self.seed * 1000003 + round_idx * 101 + self.cid)
        data = self.train
        cap = p1.get('max_segments_per_client_round')
        if cap and len(data) > cap:
            data = data.iloc[rng.choice(len(data), int(cap), replace=False)].reset_index(drop=True)

        bs = int(self.cfg.encoder.batch_size)
        steps = int(p1.local_epochs) * math.ceil(len(data) / bs)
        opt = torch.optim.AdamW(m.trainable_parameters(), lr=float(p1.lr), weight_decay=float(p1.weight_decay))
        warm = max(1, int(float(p1.warmup_ratio) * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_linear(s, warm, steps))
        scaler = torch.amp.GradScaler('cuda', enabled=m.amp)
        m.train()
        total, n_seen, skipped = 0.0, 0, 0
        texts, ys = data['text'].tolist(), torch.tensor(data['y'].to_numpy(), device=m.device)
        for _ in range(int(p1.local_epochs)):
            for idx in _batches(len(data), bs, rng):
                _, logits = m([texts[i] for i in idx])
                loss = F.cross_entropy(logits.float(), ys[idx], weight=self.weights)
                if not torch.isfinite(loss):
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.trainable_parameters(), float(self.cfg.training.grad_clip))
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += loss.item() * len(idx)
                n_seen += len(idx)
        if skipped:
            logger.warning(f"client {self.cid}: {skipped} non-finite Phase-1 losses skipped "
                           f"(fp16 instability? set encoder.precision=fp32 and log the reason)")
        return m.trainable_state(), {'train_loss': total / max(n_seen, 1), 'lr': sched.get_last_lr()[0],
                                     'nonfinite_steps': skipped}

    @torch.no_grad()
    def evaluate(self, global_state, role):
        data = self.val if role == 'val' else getattr(self, role)
        return predict_segments(self.model, global_state, data, int(self.cfg.encoder.embed_batch_size))


@torch.no_grad()
def predict_segments(model: PayloadEncoder, state, data: pd.DataFrame, batch_size: int) -> EvalResult:
    model.load_trainable_state(state)
    model.eval()
    if len(data) == 0:
        return EvalResult(np.array([], int), np.array([], int), float('nan'), np.zeros((0, model.head.out_features)))
    texts = data['text'].tolist()
    order = np.argsort([len(t) for t in texts])      # length-sorted batches: less padding
    probs = np.zeros((len(texts), model.head.out_features), dtype=np.float32)
    loss_sum = 0.0
    ys = torch.tensor(data['y'].to_numpy(), device=model.device)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        _, logits = model([texts[j] for j in idx])
        loss_sum += F.cross_entropy(logits.float(), ys[idx], reduction='sum').item()
        probs[idx] = torch.softmax(logits.float(), -1).cpu().numpy()
    y = data['y'].to_numpy()
    return EvalResult(y, probs.argmax(1), loss_sum / len(texts), probs)
