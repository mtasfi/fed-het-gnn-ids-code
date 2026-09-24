"""End-to-end smoke test on a synthetic capture and a tiny random encoder (offline).

Exercises E0 -> E1 -> every variant -> aggregation. Synthetic data and the tiny
encoder exist only to test the code paths; nothing here is a result.
B2 is skipped unless the FedGATSage repo and pyg-lib are available.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tests'))

from make_synthetic_dataset import build as build_dataset  # noqa: E402
from make_tiny_encoder import build as build_encoder  # noqa: E402


def _run(args, env_sets):
    cmd = [sys.executable] + args + [x for s in env_sets for x in ('--set', s)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


@pytest.mark.slow
def test_everything_runs(tmp_path):
    raw = build_dataset(str(tmp_path / 'raw'))
    enc = build_encoder(str(tmp_path / 'enc'))
    common = [f'dataset.source.root={raw}', f'paths.work_dir={tmp_path / "work"}',
              f'paths.runs_dir={tmp_path / "runs"}', f'paths.results_dir={tmp_path / "results"}',
              'tracking.comet=false', 'flow_extractor.workers=1',
              'split.target_flows=300', 'split.rare_class_floor=50', 'split.min_test_per_class=5']
    _run(['experiments/run_e0.py', 'all', '--dataset', 'toniot'], common)

    model = common + [f'encoder.hf_id={enc}', 'encoder.hidden_dim=32', 'graph.shares_host.kappa=5',
                      'phase1.rounds=1', 'phase2.rounds=2', 'encoder.batch_size=16', 'phase2.batch_size=64',
                      'evaluation.latency_repeats=2']
    for exp in ['E1', 'E4', 'A1', 'A5', 'B3', 'B1', 'B5', 'A2', 'A3', 'A4', 'A6', 'A7', 'A8', 'A9', 'E6']:
        _run(['experiments/run.py', '--exp', exp, '--dataset', 'toniot', '--seed', '0', '--alpha', '0.5'], model)
    _run(['experiments/run.py', '--exp', 'E5', '--dataset', 'toniot', '--seed', '0'],
         model + ['partition.alpha_sweep=[0.3, 1.0]'])
    _run(['experiments/aggregate_results.py', '--dataset', 'toniot'], common)

    res = tmp_path / 'results' / 'toniot'
    idx = pd.read_csv(res / 'runs_index.csv')
    assert (idx['status'] == 'completed').all(), idx[idx.status != 'completed'][['run_id', 'failure_reason']]
    abl = pd.read_csv(res / 'results_ablations.csv')
    assert {'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9'} <= set(abl['experiment_id'])
    assert abl['delta_macro_f1_vs_E1'].notna().all()
    for name in ('table_main.tex', 'fig_convergence.pdf', 'fig_missing_rate.pdf', 'fig_perclass.pdf'):
        assert (res / name).exists(), name
