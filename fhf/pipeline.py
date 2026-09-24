"""End-to-end runners shared by the experiment entry points.

prepare()          fixed split + stored partition -> labels, scaled flow features (federated scaler)
payload_embeddings() Phase 1 (E1 / E5 only) or cache lookup / frozen / hashed embeddings
run_fedhet()       E1 and every graph variant (A1, A2, A3, A4, A5, A6, A7, E5)
run_fed_mlp()      B1 (flow features + m_i) and A8 (+ mean payload embedding), no graph
run_central_hetgnn() B3: identical Phase-2 architecture, pooled training, no federation
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from fhf.common.config import Config
from fhf.common.metrics import classification_report, weighted_mean
from fhf.common.utils import Timer, count_parameters, peak_gpu_mb, peak_rss_mb, resolve_device, set_seed
from fhf.data.federated_scaler import FederatedScaler
from fhf.data.features import feature_matrix
from fhf.data.store import Store, class_names, encode_labels, load_run_frame
from fhf.federated.server import FedResult, run_federated
from fhf.federated.weighted_aggregate import get_aggregator
from fhf.phase1 import embed_once
from fhf.phase2.build_heterograph import ClientGraph, EmbeddingLookup, build_graph
from fhf.phase2.hetero_sage import build_gnn, gnn_forward
from fhf.phase2.train_local import GraphClient, get_state, predict

logger = logging.getLogger(__name__)

ROLES = ('train', 'val', 'test')


@dataclass
class Prepared:
    frame: pd.DataFrame
    names: List[str]
    y: np.ndarray
    x: np.ndarray               # federated-scaled flow features (m_i not included)
    scaler: FederatedScaler
    clients: List[int]


def prepare(cfg) -> Prepared:
    frame = load_run_frame(cfg)
    names = class_names(cfg)
    frame = frame[frame['label'].isin(names)].reset_index(drop=True)
    y = encode_labels(frame['label'], names)
    raw = feature_matrix(frame, cfg)
    clients = [int(c) for c in sorted(frame['client'].unique())]
    train_parts = [raw[((frame.client == c) & (frame.role == 'train')).to_numpy()] for c in clients]
    scaler = FederatedScaler(bool(cfg.features.log1p)).fit_clients(train_parts)
    x = scaler.transform(raw)
    logger.info(f"{len(frame):,} flows, {len(names)} classes {names}, {len(clients)} clients, "
                f"{x.shape[1]} flow features; scaler from {scaler.n:,} client-train rows")
    return Prepared(frame, names, y, x, scaler, clients)


# ---------------------------------------------------------------------------- payload embeddings
def payload_embeddings(cfg, prep: Prepared, rundir, timer: Timer, device) -> Optional[EmbeddingLookup]:
    source = cfg.payload.get('encoder_source', 'lora')
    if source == 'none' or not cfg.graph.get('use_payload', True):
        return None
    tag = embed_once.embedding_tag(cfg, source)
    if not embed_once.cache_exists(cfg, tag):
        if source == 'lora':
            if not cfg.phase1.get('allow_train', False):
                raise FileNotFoundError(
                    f"Embedding cache '{tag}' missing and this experiment may not run Phase 1. "
                    f"Run E1 with the same dataset / alpha / seed first.")
            from fhf.phase1.run_phase1 import run_phase1
            with timer('phase1'):
                encoder = run_phase1(cfg, prep, rundir, device)
            with timer('embedding'):
                embed_once.build_cache(cfg, prep.frame, 'lora', tag, encoder)
            del encoder
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        elif source == 'frozen':
            from fhf.phase1.lora_model import PayloadEncoder
            encoder = PayloadEncoder(cfg, len(prep.names), device, use_lora=False)
            with timer('embedding'):
                embed_once.build_cache(cfg, prep.frame, 'frozen', tag, encoder)
            del encoder
        else:
            with timer('embedding'):
                embed_once.build_cache(cfg, prep.frame, source, tag)
    elif source == 'lora' and cfg.phase1.get('allow_train', False):
        attach_phase1_record(cfg, rundir)
    meta = embed_once.cache_meta(cfg, tag)
    rundir.add_input(f'embeddings:{tag}', os.path.join(Store(cfg).embeddings_dir(tag), 'meta.json'))
    rundir.manifest['embedding_tag'] = tag
    rundir.manifest['embedding_meta'] = {k: v for k, v in meta.items() if k != 'encoder'}
    return embed_once.load_cache(cfg, tag, prep.frame['flow_uid'].to_numpy())


def attach_phase1_record(cfg, rundir):
    """E1 / E5 rerun after Phase 1 already finished (e.g. a later crash): copy the
    Phase-1 rounds and meta from the cached checkpoint so this run directory stays complete."""
    import json
    from fhf.phase1.run_phase1 import phase1_dir
    d = phase1_dir(cfg)
    latest, meta_path = os.path.join(d, 'latest.pt'), os.path.join(d, 'meta.json')
    if not (os.path.exists(latest) and os.path.exists(meta_path)):
        return
    for row in torch.load(latest, map_location='cpu', weights_only=False)['history']:
        rundir.append_csv('round_metrics.csv', {**row, 'note': 'copied from cached Phase-1 checkpoint'})
    with open(meta_path) as f:
        meta = json.load(f)
    rundir.manifest['phase1'] = {k: v for k, v in meta.items() if isinstance(v, (int, float))}
    rundir.manifest['phase1_reused_from'] = meta.get('run_id')
    rundir.write_text('phase1_checkpoint_pointer.txt', os.path.join(d, 'best.pt') + '\n')
    logger.info(f"Phase 1 already cached ({d}); its rounds were copied into this run's round_metrics.csv")


# ---------------------------------------------------------------------------- graphs
def build_client_graphs(cfg, prep: Prepared, emb: Optional[EmbeddingLookup],
                        test_mask: Optional[np.ndarray] = None) -> Dict[int, Dict[str, ClientGraph]]:
    graphs: Dict[int, Dict[str, ClientGraph]] = {}
    for c in prep.clients:
        graphs[c] = {}
        for role in ROLES:
            sel = ((prep.frame.client == c) & (prep.frame.role == role)).to_numpy()
            mask = test_mask[sel] if (test_mask is not None and role == 'test') else None
            graphs[c][role] = build_graph(prep.frame[sel], prep.x[sel], prep.y[sel], cfg, emb, c, role, mask)
    return graphs


def graph_inventory(graphs) -> pd.DataFrame:
    rows = [{'client': c, 'role': r, **g.stats} for c, roles in graphs.items() for r, g in roles.items()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- evaluation
def evaluate_test(model, state, clients_graphs: Dict[int, Dict[str, ClientGraph]], names, forward=gnn_forward,
                  device=None) -> Dict:
    ys, ps, cs, uids, probs = [], [], [], [], []
    for c, roles in clients_graphs.items():
        g = roles['test']
        if device is not None:
            g = ClientGraph(g.data.to(device), g.y.to(device), g.flow_uid, g.client, g.role, g.stats)
        res = predict(model, state, g, forward)
        if len(res.y_true) == 0:
            continue
        ys.append(res.y_true); ps.append(res.y_pred); cs.append(np.full(len(res.y_true), c))
        uids.append(g.flow_uid); probs.append(res.probs)
    y, p, cl = np.concatenate(ys), np.concatenate(ps), np.concatenate(cs)
    report = classification_report(y, p, names, client_ids=cl)
    pred = pd.DataFrame({'flow_uid': np.concatenate(uids), 'client': cl, 'y_true': y, 'y_pred': p})
    prob = np.concatenate(probs)
    for i, n in enumerate(names):
        pred[f'p_{n}'] = prob[:, i].astype(np.float32)
    return {'report': report, 'predictions': pred}


def write_outputs(rundir, names, result: Dict, extra: Dict, resources: List[Dict]):
    rundir.write_evaluation(result['report'], names, extra=extra)
    result['predictions'].to_parquet(rundir.file('predictions.parquet'), index=False)
    rundir.write_resources(resources)


def resource_rows(timer: Timer, extra: Dict) -> List[Dict]:
    rows = [{'name': f'time_{k}_s', 'value': round(v, 3), 'unit': 's'} for k, v in timer.totals.items()]
    for k, v in extra.items():
        rows.append({'name': k, 'value': v, 'unit': ''})
    rows.append({'name': 'peak_rss_mb', 'value': round(peak_rss_mb(), 1), 'unit': 'MB'})
    gpu = peak_gpu_mb()
    if gpu is not None:
        rows.append({'name': 'peak_gpu_mb', 'value': round(gpu, 1), 'unit': 'MB'})
    return rows


def _fed_extra(result: FedResult) -> Dict:
    return {'checkpoint_round': result.best_round, 'final_round': result.final_round,
            'best_val_macro_f1': result.best_score}


# ---------------------------------------------------------------------------- E1 and graph variants
def run_fedhet(cfg, rundir, resume: bool = False, test_mask: Optional[np.ndarray] = None) -> Dict:
    device = resolve_device(cfg.device)
    set_seed(int(cfg.seed))
    timer = Timer()
    prep = prepare(cfg)
    emb = payload_embeddings(cfg, prep, rundir, timer, device)

    set_seed(int(cfg.seed))
    with timer('graph_build'):
        graphs = build_client_graphs(cfg, prep, emb)
    inv = graph_inventory(graphs)
    inv.to_csv(rundir.file('graph_inventory.csv'), index=False)

    flow_dim = prep.x.shape[1] + 1
    payload_dim = emb.dim if emb is not None else int(cfg.encoder.hidden_dim)
    model = build_gnn(cfg, flow_dim, payload_dim, len(prep.names)).to(device)
    clients = [GraphClient(c, model, graphs[c], cfg, len(prep.names), device, int(cfg.seed)) for c in prep.clients]

    with timer('phase2'):
        result = run_federated(clients, get_state(model), int(cfg.phase2.rounds),
                               get_aggregator(cfg.aggregation.method), rundir, phase='2',
                               num_classes=len(prep.names), ckpt_path=rundir.file('phase2_latest.pt'),
                               early_stopping=cfg.early_stopping, resume=resume)
    torch.save(result.best_state, rundir.file('phase2_best.pt'))
    rundir.write_pointer(rundir.file('phase2_best.pt'))

    with timer('test_inference'):
        out = evaluate_test(model, result.best_state, graphs, prep.names, device=device)
    theta = count_parameters(model)
    res = resource_rows(timer, {
        'gnn_params_theta': theta,
        'phase2_bytes_up_per_client_round': result.bytes_up_per_client_round,
        'phase2_bytes_down_per_client_round': result.bytes_down_per_client_round,
        'phase2_bytes_total': result.bytes_up_total + result.bytes_down_total,
        'scaler_stats_bytes_uploaded': prep.scaler.bytes_uploaded,
        'phase2_batch_size': int(cfg.phase2.batch_size),
        'payload_nodes_total': int(inv['payload_nodes'].sum()),
        'shares_host_edges_total': int(inv['shares_host_edges_undirected'].sum()),
    })
    res += phase1_resource_rows(rundir)
    write_outputs(rundir, prep.names, out, _fed_extra(result), res)
    return {'prep': prep, 'result': result, 'model': model, 'graphs': graphs, 'emb': emb, 'out': out}


def phase1_resource_rows(rundir) -> List[Dict]:
    meta = rundir.manifest.get('phase1')
    if not meta:
        return []
    return [{'name': f'phase1_{k}', 'value': v, 'unit': ''} for k, v in meta.items() if isinstance(v, (int, float))]


# ---------------------------------------------------------------------------- B1 / A8: federated MLP
class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], num_classes: int, dropout: float):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            d = h
        layers.append(torch.nn.Linear(d, num_classes))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def mlp_forward(model, data):
    return model(data.x)


def mlp_inputs(prep: Prepared, emb: Optional[EmbeddingLookup], use_payload_mean: bool) -> np.ndarray:
    m = prep.frame['has_payload'].to_numpy().astype(np.float32)
    parts = [prep.x, m[:, None]]
    if use_payload_mean:
        mean = np.zeros((len(prep.frame), emb.dim), dtype=np.float32)
        for i, uid in enumerate(prep.frame['flow_uid'].to_numpy()):
            if m[i]:
                segs = emb.get(uid)
                if len(segs):
                    mean[i] = segs.mean(0)        # m_i = 0 -> zero payload vector, flag retained
        parts.append(mean)
    return np.concatenate(parts, axis=1).astype(np.float32)


def run_fed_mlp(cfg, rundir, resume: bool = False) -> Dict:
    device = resolve_device(cfg.device)
    set_seed(int(cfg.seed))
    timer = Timer()
    prep = prepare(cfg)
    use_emb = cfg.payload.get('encoder_source', 'none') != 'none'
    emb = payload_embeddings(cfg, prep, rundir, timer, device) if use_emb else None
    feats = mlp_inputs(prep, emb, use_emb)

    graphs = {}
    for c in prep.clients:
        graphs[c] = {}
        for role in ROLES:
            sel = ((prep.frame.client == c) & (prep.frame.role == role)).to_numpy()
            graphs[c][role] = ClientGraph(Data(x=torch.from_numpy(feats[sel])),
                                          torch.from_numpy(prep.y[sel]), prep.frame['flow_uid'].to_numpy()[sel],
                                          c, role, {'flows': int(sel.sum())})

    set_seed(int(cfg.seed))
    model = MLP(feats.shape[1], list(cfg.mlp.hidden_dims), len(prep.names), float(cfg.mlp.dropout)).to(device)
    clients = [GraphClient(c, model, graphs[c], cfg, len(prep.names), device, int(cfg.seed), forward=mlp_forward)
               for c in prep.clients]
    with timer('train'):
        result = run_federated(clients, get_state(model), int(cfg.phase2.rounds), get_aggregator(cfg.aggregation.method),
                               rundir, phase='2', num_classes=len(prep.names),
                               ckpt_path=rundir.file('mlp_latest.pt'), early_stopping=cfg.early_stopping, resume=resume)
    torch.save(result.best_state, rundir.file('mlp_best.pt'))
    rundir.write_pointer(rundir.file('mlp_best.pt'))
    with timer('test_inference'):
        out = evaluate_test(model, result.best_state, graphs, prep.names, forward=mlp_forward, device=device)
    res = resource_rows(timer, {'mlp_params': count_parameters(model), 'input_dim': feats.shape[1],
                                'bytes_up_per_client_round': result.bytes_up_per_client_round,
                                'bytes_down_per_client_round': result.bytes_down_per_client_round})
    write_outputs(rundir, prep.names, out, _fed_extra(result), res)
    return {'prep': prep, 'result': result}


# ---------------------------------------------------------------------------- B3: centralized HetGNN
def run_central_hetgnn(cfg, rundir, resume: bool = False) -> Dict:
    """Same architecture, same client graphs (so edges are identical to E1), same
    cached phi* embeddings, trained on the pooled training graphs with the same
    effective epoch budget R2 x E2; checkpoint chosen by the same sample-weighted
    mean of per-client validation macro-F1."""
    from torch_geometric.data import Batch

    device = resolve_device(cfg.device)
    set_seed(int(cfg.seed))
    timer = Timer()
    prep = prepare(cfg)
    emb = payload_embeddings(cfg, prep, rundir, timer, device)
    graphs = build_client_graphs(cfg, prep, emb)
    pooled_train = ClientGraph(Batch.from_data_list([graphs[c]['train'].data for c in prep.clients]),
                               torch.cat([graphs[c]['train'].y for c in prep.clients]),
                               np.concatenate([graphs[c]['train'].flow_uid for c in prep.clients]), -1, 'train')

    flow_dim = prep.x.shape[1] + 1
    payload_dim = emb.dim if emb is not None else int(cfg.encoder.hidden_dim)
    model = build_gnn(cfg, flow_dim, payload_dim, len(prep.names)).to(device)
    central = GraphClient(-1, model, {'train': pooled_train}, cfg, len(prep.names), device, int(cfg.seed),
                          train_cfg=Config({**cfg.phase2.to_dict(), 'local_epochs': 1}))
    val_clients = [GraphClient(c, model, graphs[c], cfg, len(prep.names), device, int(cfg.seed)) for c in prep.clients]

    epochs = int(cfg.phase2.rounds) * int(cfg.phase2.local_epochs)
    state = get_state(model)
    best = (-np.inf, state, 0)
    with timer('train'):
        for epoch in range(1, epochs + 1):
            state, m = central.fit(state, epoch)
            f1s, ns = [], []
            for vc in val_clients:
                r = vc.evaluate(state, 'val')
                f1s.append(classification_report(r.y_true, r.y_pred, prep.names)['summary']['macro_f1']
                           if len(r.y_true) else np.nan)
                ns.append(len(r.y_true))
            score = weighted_mean(f1s, ns)
            if score > best[0]:
                best = (score, {k: v.clone() for k, v in state.items()}, epoch)
            rundir.log_round({'phase': 'central', 'round': epoch, 'scope': 'global', 'train_loss': m['train_loss'],
                              'val_macro_f1': score, 'lr': m['lr'], 'clients': 0, 'bytes_up': 0, 'bytes_down': 0,
                              'best_round': best[2]})
    torch.save(best[1], rundir.file('central_best.pt'))
    rundir.write_pointer(rundir.file('central_best.pt'))
    with timer('test_inference'):
        out = evaluate_test(model, best[1], graphs, prep.names, device=device)
    res = resource_rows(timer, {'gnn_params_theta': count_parameters(model), 'epochs': epochs})
    write_outputs(rundir, prep.names, out, {'checkpoint_round': best[2], 'best_val_macro_f1': best[0]}, res)
    return {'prep': prep}
