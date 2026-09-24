"""E6: measured cost of the E1 model (no retraining).

Parameter inventory: |Delta| (trainable LoRA), |W_c| (Phase-1 head), |phi| (whole
encoder), |theta| (HetGNN). Communication: bytes actually serialised per client
per round, taken from the E1 run. Time: Phase 1, embedding, Phase 2 wall clock
from E1. Latency: warm per-flow p50 / p95 at a fixed batch size on this device,
split into payload encoding and graph inference. The "48x per adapted matrix
(d=768, r=8)" figure is matrix-level context only; the numbers here are
whole-model measurements.
"""

import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch

from fhf.common.config import Config
from fhf.common.rundir import hardware_identity, source_run
from fhf.common.utils import count_parameters, resolve_device, set_seed
from fhf.phase1 import embed_once
from fhf.phase1.lora_model import PayloadEncoder
from fhf.phase1.run_phase1 import phase1_dir
from fhf.phase2.build_heterograph import build_graph
from fhf.phase2.hetero_sage import build_gnn, gnn_forward
from fhf.pipeline import prepare

logger = logging.getLogger(__name__)


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def run_efficiency(cfg, rundir, resume: bool = False):
    e1_path, e1_cfg = source_run(cfg, 'E1')
    work = Config({**e1_cfg.to_dict(), 'smoke': cfg.get('smoke', False)})
    device = resolve_device(cfg.device)
    set_seed(int(cfg.seed))
    with open(os.path.join(e1_path, 'manifest.json')) as f:
        e1_manifest = json.load(f)
    e1_res = pd.read_csv(os.path.join(e1_path, 'resource_metrics.csv')).set_index('name')['value']
    rundir.manifest['source_run'] = e1_path

    prep = prepare(work)
    tag = embed_once.embedding_tag(work, 'lora')
    emb_meta = embed_once.cache_meta(work, tag)
    encoder = PayloadEncoder(work, len(prep.names), device, use_lora=True)
    encoder.load_trainable_state(torch.load(os.path.join(phase1_dir(work), 'best.pt'), map_location='cpu',
                                            weights_only=False))
    gnn = build_gnn(work, prep.x.shape[1] + 1, int(emb_meta['dim']), len(prep.names)).to(device)
    gnn.load_state_dict(torch.load(os.path.join(e1_path, 'phase2_best.pt'), map_location='cpu', weights_only=False))

    inventory = pd.DataFrame([
        {'component': 'lora_delta', 'params': encoder.param_counts['lora_trainable'], 'communicated': 'phase 1'},
        {'component': 'phase1_head_Wc', 'params': encoder.param_counts['head'], 'communicated': 'phase 1'},
        {'component': 'encoder_phi_total', 'params': encoder.param_counts['encoder_total'], 'communicated': 'never'},
        {'component': 'hetgnn_theta', 'params': count_parameters(gnn), 'communicated': 'phase 2'},
    ])
    inventory['bytes_fp32'] = inventory['params'] * 4
    inventory.to_csv(rundir.file('parameter_inventory.csv'), index=False)

    # ---- latency: one client's test flows, fixed batch, warm
    bs = int(cfg.evaluation.latency_batch_size)
    reps = int(cfg.evaluation.latency_repeats)
    test = prep.frame[(prep.frame.role == 'test')]
    client = int(test['client'].value_counts().idxmax())
    sel = ((prep.frame.role == 'test') & (prep.frame.client == client)).to_numpy()
    idx = np.nonzero(sel)[0][:bs]
    batch = prep.frame.iloc[idx]
    texts = [t for ts in batch.loc[batch.has_payload == 1, 'payload_texts'] for t in ts]
    enc_times, gnn_times = [], []
    for r in range(reps + 2):                       # first two iterations are warm-up
        _sync(device)
        t0 = time.perf_counter()
        if texts:
            vecs = embed_once.encode_texts(encoder, texts, int(work.encoder.embed_batch_size))
        _sync(device)
        t1 = time.perf_counter()
        lookup = _batch_lookup(batch, vecs if texts else np.zeros((0, int(emb_meta['dim'])), np.float16))
        g = build_graph(batch, prep.x[idx], prep.y[idx], work, lookup, client, 'test')
        with torch.no_grad():
            gnn.eval()
            gnn_forward(gnn, g.data.to(device))
        _sync(device)
        t2 = time.perf_counter()
        if r >= 2:
            enc_times.append((t1 - t0) / len(batch))
            gnn_times.append((t2 - t1) / len(batch))
    total = np.array(enc_times) + np.array(gnn_times)

    p1 = e1_manifest.get('phase1', {})
    rows = {
        'lora_delta_params': encoder.param_counts['lora_trainable'],
        'head_Wc_params': encoder.param_counts['head'],
        'encoder_phi_params': encoder.param_counts['encoder_total'],
        'hetgnn_theta_params': count_parameters(gnn),
        'phase1_bytes_up_per_client_round': p1.get('bytes_up_per_client_round'),
        'phase1_bytes_down_per_client_round': p1.get('bytes_down_per_client_round'),
        'phase1_bytes_total': p1.get('bytes_total'),
        'phase2_bytes_up_per_client_round': e1_res.get('phase2_bytes_up_per_client_round'),
        'phase2_bytes_down_per_client_round': e1_res.get('phase2_bytes_down_per_client_round'),
        'phase2_bytes_total': e1_res.get('phase2_bytes_total'),
        'scaler_stats_bytes_uploaded': e1_res.get('scaler_stats_bytes_uploaded'),
        'phase1_seconds': e1_res.get('time_phase1_s', p1.get('seconds_this_session')),
        'embedding_seconds': emb_meta.get('embed_seconds'),
        'phase2_seconds': e1_res.get('time_phase2_s'),
        'flows_encoded': emb_meta.get('flows_encoded'),
        'flows_skipped_encoder_m0': emb_meta.get('flows_skipped_m0'),
        'latency_batch_size': len(batch),
        'latency_repeats': reps,
        'latency_encode_ms_per_flow_p50': 1000 * float(np.percentile(enc_times, 50)),
        'latency_gnn_ms_per_flow_p50': 1000 * float(np.percentile(gnn_times, 50)),
        'latency_total_ms_per_flow_p50': 1000 * float(np.percentile(total, 50)),
        'latency_total_ms_per_flow_p95': 1000 * float(np.percentile(total, 95)),
        'lora_reduction_vs_full_encoder': encoder.param_counts['encoder_total'] / max(encoder.param_counts['lora_trainable'], 1),
    }
    pd.DataFrame([{'run_id': rundir.manifest['run_id'], 'device': str(device), **rows}]).to_csv(
        rundir.file('efficiency_runs.csv'), index=False)
    pd.DataFrame({'iteration': range(len(total)), 'encode_s_per_flow': enc_times, 'gnn_s_per_flow': gnn_times}) \
        .to_csv(rundir.file('latency_profile.csv'), index=False)
    rundir.write_resources([{'name': k, 'value': v, 'unit': ''} for k, v in rows.items() if v is not None])
    hw = hardware_identity()
    lines = ['# E6 efficiency report', '', f"Source run: `{e1_path}`", f"Hardware / software: `{hw}`", '',
             '## Parameters', '', inventory.to_markdown(index=False), '', '## Measurements', '']
    lines += [f"- {k}: {v}" for k, v in rows.items()]
    lines += ['', "Matrix-level context only: LoRA at d=768, r=8 trains 2dr = 12,288 instead of d^2 = 589,824 "
                  "parameters per adapted square matrix (48x). Whole-model numbers above are measured."]
    rundir.write_text('efficiency_report.md', '\n'.join(lines))
    return {'rows': rows}


def _batch_lookup(batch: pd.DataFrame, vecs: np.ndarray):
    from fhf.phase2.build_heterograph import EmbeddingLookup
    uids, pos = [], []
    for uid, texts in zip(batch.loc[batch.has_payload == 1, 'flow_uid'], batch.loc[batch.has_payload == 1, 'payload_texts']):
        uids += [uid] * len(texts)
        pos += list(range(len(texts)))
    return EmbeddingLookup(np.asarray(vecs, dtype=np.float32), np.array(uids, dtype=object), np.array(pos))
