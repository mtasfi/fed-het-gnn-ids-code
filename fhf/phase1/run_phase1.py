"""Runs Phase 1 inside an E1 / E5 run and stores phi* (best-round LoRA + head).

The checkpoint lives in the shared cache (work/<ds>/cache/phase1/<tag>/) so A9
and E6 can load it; the run directory records a pointer, the resolved LoRA
targets, the named_modules() listing and the Phase-1 rounds (phase = 1 rows).
"""

import json
import logging
import os
import time

import torch

from fhf.data.store import Store
from fhf.federated.fedavg import fedavg
from fhf.federated.server import run_federated
from fhf.phase1.embed_once import embedding_tag
from fhf.phase1.federated_tune import Phase1Client, segment_table
from fhf.phase1.lora_model import PayloadEncoder

logger = logging.getLogger(__name__)


def phase1_dir(cfg) -> str:
    return Store(cfg).phase1_dir(embedding_tag(cfg, 'lora'))


def run_phase1(cfg, prep, rundir, device) -> PayloadEncoder:
    model = PayloadEncoder(cfg, len(prep.names), device, use_lora=True)
    rundir.write_text('encoder_named_modules.txt',
                      '\n'.join(f'{n}\t{type(m).__name__}' for n, m in model.backbone.named_modules()))
    rundir.write_text('lora_targets.txt', '\n'.join(model.lora_targets))

    seg = segment_table(prep.frame, prep.y)
    clients = []
    for c in prep.clients:
        tr = seg[(seg.client == c) & (seg.role == 'train')]
        va = seg[(seg.client == c) & (seg.role == 'val')]
        client = Phase1Client(c, model, tr, va, cfg, len(prep.names), int(cfg.seed))
        client.test = seg[(seg.client == c) & (seg.role == 'test')]
        clients.append(client)
    logger.info(f"Phase 1: {len(seg):,} readable segments; per-client train segments "
                f"{[cl.n_train for cl in clients]}")

    out_dir = phase1_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    result = run_federated(clients, model.trainable_state(), int(cfg.phase1.rounds), fedavg, rundir, phase='1',
                           num_classes=len(prep.names), ckpt_path=os.path.join(out_dir, 'latest.pt'),
                           early_stopping=cfg.early_stopping, resume=True)

    torch.save(result.best_state, os.path.join(out_dir, 'best.pt'))
    meta = {
        'best_round': result.best_round,
        'best_val_macro_f1': result.best_score,
        'rounds_run': result.final_round,
        'lora_trainable_params': model.param_counts['lora_trainable'],
        'head_params': model.param_counts['head'],
        'encoder_total_params': model.param_counts['encoder_total'],
        'bytes_up_per_client_round': result.bytes_up_per_client_round,
        'bytes_down_per_client_round': result.bytes_down_per_client_round,
        'bytes_total': result.bytes_up_total + result.bytes_down_total,
        'segments_train': int(sum(cl.n_train for cl in clients)),
        'seconds_this_session': time.perf_counter() - t0,
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump({**meta, 'encoder': model.identity, 'lora_targets': model.lora_targets,
                   'lora': cfg.encoder.lora.to_dict(), 'run_id': rundir.manifest.get('run_id')}, f, indent=2)
    rundir.manifest['phase1'] = meta
    rundir.manifest['encoder_identity'] = model.identity
    rundir.write_text('phase1_checkpoint_pointer.txt', os.path.join(out_dir, 'best.pt') + '\n')

    model.load_trainable_state(result.best_state)   # phi*: the head is no longer used after this point
    return model
