"""B2: FedGATSage rerun on the exact E1 split, label map and seed.

The FedGATSage code is NOT copied or modified: the adapter exports our split into
the directory layout the reproduction expects, imports its modules from
`baselines.fedgatsage.repo_path`, trains it with its own loop, and scores its
ensemble on our test flows with our metric code. Every deviation from the
original script is listed in docs/b2_fedgatsage_rerun.md and in the run manifest.

Inputs come from our PCAP extractor (the same flows as E1): FedGATSage's
"payload statistics" (payload sizes per direction, etc.) are our fwd/bwd
payload byte counts. FedGATSage uses IP endpoints as graph nodes by design;
that is part of the method being compared, not of ours.
"""

import logging
import os
import subprocess
import sys
from argparse import Namespace
from contextlib import contextmanager
from typing import Dict

import numpy as np
import pandas as pd
import torch

from fhf.common.config import resolve_path
from fhf.common.metrics import classification_report
from fhf.common.utils import Timer, peak_rss_mb, resolve_device, set_seed
from fhf.pipeline import prepare

logger = logging.getLogger(__name__)

DETECTORS = ['temporal', 'content', 'behavioral']

DEVIATIONS = [
    "input flows, split, client partition and label map are E1's (PCAP-extracted flows), not NF-ToN-IoT rows",
    "flow columns mapped from our extractor (docs/b2_fedgatsage_rerun.md); payload statistics = our fwd/bwd payload bytes",
    "class_weights.pt computed from training rows only (the original preprocess used the full dataset incl. test)",
    "ensemble Random Forest fitted on pooled client VALIDATION flows and scored on the full E1 test set "
    "(the original fitted it on 50% of the test set and scored the other 50%)",
    "no checkpoint selection (none in the original): the model after the last round is evaluated",
    "per-client grouping of test predictions (worst-client macro-F1) uses the E1 client assignment",
]


def export_frame(frame: pd.DataFrame, names) -> pd.DataFrame:
    """Our flow table -> the column names FedGATSage's feature code reads (CIC + NF variants)."""
    return pd.DataFrame({
        'flow_uid': frame['flow_uid'].to_numpy(),
        'Src IP': frame['src_ip'].to_numpy(), 'Dst IP': frame['dst_ip'].to_numpy(),
        'Src Port': frame['src_port'].to_numpy(), 'Dst Port': frame['dst_port_id'].to_numpy(),
        'Protocol': frame['proto'].to_numpy(),
        'Flow Duration': frame['duration'].to_numpy() * 1e6,                 # CIC: microseconds
        'Tot Fwd Pkts': frame['fwd_pkts'].to_numpy(), 'Tot Bwd Pkts': frame['bwd_pkts'].to_numpy(),
        'TotLen Fwd Pkts': frame['fwd_payload_bytes'].to_numpy(),
        'TotLen Bwd Pkts': frame['bwd_payload_bytes'].to_numpy(),
        'Flow IAT Mean': frame['iat_mean'].to_numpy() * 1e6, 'Flow IAT Std': frame['iat_std'].to_numpy() * 1e6,
        'Flow Pkts/s': frame['pkts_per_s'].to_numpy(),
        'SYN Flag Cnt': frame['syn_cnt'].to_numpy(), 'RST Flag Cnt': frame['rst_cnt'].to_numpy(),
        'ACK Flag Cnt': frame['ack_cnt'].to_numpy(),
        'IN_BYTES': frame['fwd_bytes'].to_numpy(), 'OUT_BYTES': frame['bwd_bytes'].to_numpy(),
        'IN_PKTS': frame['fwd_pkts'].to_numpy(), 'OUT_PKTS': frame['bwd_pkts'].to_numpy(),
        'FLOW_DURATION_MILLISECONDS': frame['duration'].to_numpy() * 1e3,
        'L4_DST_PORT': frame['dst_port_id'].to_numpy(),
        'Attack': frame['label'].to_numpy(),
    })


def export_split(prep, work_dir: str) -> Dict[str, str]:
    data_dir = os.path.join(work_dir, 'data')
    f = prep.frame
    paths = {}
    for det in DETECTORS:
        d = os.path.join(data_dir, f'{det}_detector')
        os.makedirs(d, exist_ok=True)
        for k, c in enumerate(prep.clients, start=1):
            export_frame(f[(f.client == c) & (f.role == 'train')], prep.names).to_csv(
                os.path.join(d, f'client_{k}.csv'), index=False)
        export_frame(f[f.role == 'test'], prep.names).to_csv(os.path.join(d, 'test.csv'), index=False)
        export_frame(f[f.role == 'val'], prep.names).to_csv(os.path.join(d, 'val.csv'), index=False)
    import json
    with open(os.path.join(data_dir, 'label_mapper.json'), 'w') as fh:
        json.dump({n: i for i, n in enumerate(prep.names)}, fh, indent=2)
    train_y = prep.y[(f.role == 'train').to_numpy()]
    counts = np.bincount(train_y, minlength=len(prep.names)).astype(float)
    weights = np.where(counts > 0, counts.sum() / (len(prep.names) * np.maximum(counts, 1)), 0.0)
    torch.save(torch.tensor(weights, dtype=torch.float), os.path.join(data_dir, 'class_weights.pt'))
    paths['data_dir'] = data_dir
    return paths


@contextmanager
def _in_dir(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _repo_commit(repo: str) -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode().strip()
    except Exception:
        return 'unknown'


def _graph_for(loader, csv_path: str):
    df = pd.read_csv(csv_path)
    df = loader.feature_engineer.extract_features(df)
    df = loader.centrality_extractor.extract_centrality_features(df)
    df = loader.community_processor.create_community_enhanced_features(df, {})
    return loader._process_to_graph(df)


def run_fedgatsage(cfg, rundir, resume: bool = False):
    bcfg = cfg.baselines.fedgatsage
    repo = resolve_path(bcfg.repo_path)
    if not os.path.isdir(os.path.join(repo, 'src')):
        raise FileNotFoundError(f"FedGATSage code not found at {repo} (baselines.fedgatsage.repo_path). "
                                f"Clone mtasfi/Fed_GNN there, or pass --set baselines.fedgatsage.repo_path=<dir>.")
    device = resolve_device(cfg.device)
    set_seed(int(cfg.seed))
    timer = Timer()
    prep = prepare(cfg)
    work_dir = rundir.file('fedgatsage_work')
    os.makedirs(work_dir, exist_ok=True)
    export_split(prep, work_dir)
    rundir.manifest.update({'fedgatsage_repo': repo, 'fedgatsage_commit': _repo_commit(repo),
                            'deviations_from_original': DEVIATIONS})

    for sub in ('src', 'asfi-codes'):
        p = os.path.join(repo, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    from federated_learning import FedGATSageSystem           # noqa: E402  (FedGATSage code)
    from ensemble_evaluator import RandomForestEnsembleEvaluator  # noqa: E402
    from sklearn.ensemble import RandomForestClassifier

    with _in_dir(work_dir):                                  # its trainer reads data/class_weights.pt from CWD
        system = FedGATSageSystem(data_dir='data', num_clients=len(prep.clients), detector_types=DETECTORS,
                                  device=str(device), community_algorithm=bcfg.community)
        sample = system.data_loaders[DETECTORS[0]].load_client_data(1)
        input_dim = int(sample['features'].shape[1])
        system.initialize_models(input_dim=input_dim, hidden_dim=int(bcfg.hidden_dim), num_classes=len(prep.names))
        with timer('train'):
            results = system.train_federated(num_rounds=int(bcfg.num_rounds))
        for i, (loss, sec) in enumerate(zip(results['training_losses'], results['round_times']), start=1):
            rundir.log_round({'phase': 'fedgatsage', 'round': i, 'scope': 'global', 'train_loss': loss,
                              'elapsed_s': sec, 'clients': len(prep.clients)})

        evaluator = RandomForestEnsembleEvaluator(system, Namespace(demo_mode=False, output_dir=work_dir))
        loader = system.data_loaders[DETECTORS[0]]
        with timer('test_inference'):
            val_graph = _graph_for(loader, os.path.join('data', 'temporal_detector', 'val.csv'))
            test_graph = _graph_for(loader, os.path.join('data', 'temporal_detector', 'test.csv'))
            x_val, y_val = evaluator._extract_probabilities(val_graph)
            x_test, y_test = evaluator._extract_probabilities(test_graph)
            rf = RandomForestClassifier(n_estimators=int(bcfg.rf_estimators), random_state=42)
            rf.fit(x_val, y_val)
            proba_local = rf.predict_proba(x_test)

    proba = np.zeros((len(y_test), len(prep.names)), dtype=np.float32)
    proba[:, rf.classes_.astype(int)] = proba_local
    y_pred = proba.argmax(1)
    uids = test_graph['df']['flow_uid'].to_numpy()
    client = prep.frame.set_index('flow_uid').loc[uids, 'client'].to_numpy()
    report = classification_report(y_test, y_pred, prep.names, client_ids=client)

    pred = pd.DataFrame({'flow_uid': uids, 'client': client, 'y_true': y_test, 'y_pred': y_pred})
    for i, n in enumerate(prep.names):
        pred[f'p_{n}'] = proba[:, i]
    pred.to_parquet(rundir.file('predictions.parquet'), index=False)
    rundir.write_evaluation(report, prep.names, extra={'checkpoint_round': int(bcfg.num_rounds),
                                                        'test_flows_scored': int(len(y_test))})
    params = sum(p.numel() for models in system.client_models.values() for p in models[0].parameters())
    rundir.write_resources([
        {'name': 'time_train_s', 'value': round(timer.totals['train'], 2), 'unit': 's'},
        {'name': 'time_test_inference_s', 'value': round(timer.totals['test_inference'], 2), 'unit': 's'},
        {'name': 'client_gat_params_total_3_detectors', 'value': int(params), 'unit': ''},
        {'name': 'peak_rss_mb', 'value': round(peak_rss_mb(), 1), 'unit': 'MB'},
    ])
    logger.info(f"B2 FedGATSage on the E1 split: macro-F1 {report['summary']['macro_f1']:.4f}")
    return {'report': report}
