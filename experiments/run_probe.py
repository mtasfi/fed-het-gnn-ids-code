"""Timing probe (plan §4.1): after E0, before the E1 queue.

Runs the full E1 pipeline once on ~10k real flows with ONE Phase-1 round and ONE
Phase-2 round, measures Phase 1, embedding and Phase 2 separately, and
extrapolates them to the sprint schedule. Writes probe_report.md and
probe_budget.csv next to the probe run; update the plan's budget table from them.
If the Phase-1 estimate for ModernBERT exceeds the budget, rerun with
`--encoder distilbert` and switch the queue to the fallback.

    python experiments/run_probe.py --dataset toniot [--flows 10000] [--encoder distilbert]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fhf.common.config import load_config, parse_set_args, require_locked
from fhf.common.rundir import RunDir
from fhf.common.utils import setup_logging
from fhf.data.store import load_run_frame
from fhf.pipeline import run_fedhet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='toniot')
    parser.add_argument('--flows', type=int, default=10000)
    parser.add_argument('--encoder', default=None)
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()

    cfg = load_config(args.dataset, 'E1', parse_set_args(args.set), encoder=args.encoder)
    require_locked(cfg, ['partition.alpha', 'graph.shares_host.kappa'])
    sched = {'R1': int(cfg.phase1.rounds), 'R2': int(cfg.phase2.rounds)}
    full = load_run_frame(cfg, columns=['flow_uid', 'has_payload', 'payload_texts', 'first_ts', 'label'])
    full_segments_train = int(full.loc[full.role == 'train', 'payload_texts'].apply(len).sum())
    full_segments_all = int(full['payload_texts'].apply(len).sum())
    full_flows = len(full)

    cfg.experiment = {**cfg.experiment, 'id': 'PROBE', 'variant': f"e1_{args.encoder or 'modernbert'}"}
    cfg.subset_flows = args.flows
    cfg.phase1.rounds = 1
    cfg.phase2.rounds = 1
    setup_logging('INFO')
    with RunDir(cfg, overwrite=True) as rundir:
        out = run_fedhet(cfg, rundir)
        res = pd.read_csv(rundir.file('resource_metrics.csv')).set_index('name')['value']
        frame = out['prep'].frame
        seg_train = int(frame.loc[frame.role == 'train', 'payload_texts'].apply(len).sum())
        seg_all = int(frame['payload_texts'].apply(len).sum())

        t1, te, t2 = float(res.get('time_phase1_s', 0)), float(res.get('time_embedding_s', 0)), float(res.get('time_phase2_s', 0))
        s_per_seg_round = t1 / max(seg_train, 1)
        s_per_seg_embed = te / max(seg_all, 1)
        s_per_flow_round = t2 / max(len(frame), 1)
        e1_h = (s_per_seg_round * full_segments_train * sched['R1'] + s_per_seg_embed * full_segments_all
                + s_per_flow_round * full_flows * sched['R2']) / 3600
        phase2_only_h = s_per_flow_round * full_flows * sched['R2'] / 3600
        budget = pd.DataFrame([
            {'item': 'E1 main, 2 seeds', 'runs': 2, 'est_gpu_h': 2 * e1_h},
            {'item': 'E5 alpha sweep (2 alphas, full pipeline)', 'runs': 2, 'est_gpu_h': 2 * e1_h},
            {'item': 'A1, A4, A5, A6, A7 (Phase 2 only)', 'runs': 5, 'est_gpu_h': 5 * phase2_only_h},
            {'item': 'A3 frozen encoder (embedding + Phase 2)', 'runs': 1,
             'est_gpu_h': (s_per_seg_embed * full_segments_all) / 3600 + phase2_only_h},
            {'item': 'A2 hashed n-grams (CPU transform + Phase 2)', 'runs': 1, 'est_gpu_h': phase2_only_h},
            {'item': 'B1, A8 federated MLP', 'runs': 2, 'est_gpu_h': 2 * phase2_only_h * 0.2},
            {'item': 'B3 centralized HetGNN', 'runs': 1, 'est_gpu_h': phase2_only_h},
            {'item': 'B2 FedGATSage rerun (plan estimate, not probed)', 'runs': 1, 'est_gpu_h': 2.0},
            {'item': 'E4 sweep, A9, E6 (inference only)', 'runs': 3, 'est_gpu_h': 0.3},
        ])
        budget.loc[len(budget)] = {'item': 'TOTAL', 'runs': int(budget['runs'].sum()),
                                   'est_gpu_h': float(budget['est_gpu_h'].sum())}
        budget['est_gpu_h'] = budget['est_gpu_h'].round(2)
        budget.to_csv(rundir.file('probe_budget.csv'), index=False)
        lines = ['# Timing probe', '', f"Encoder: {cfg.encoder.hf_id}; probe flows: {len(frame):,} "
                 f"(train segments {seg_train:,}); full sprint flows {full_flows:,} (train segments {full_segments_train:,})",
                 '', f"- Phase 1, one round: {t1:.1f} s  ->  {s_per_seg_round * 1e3:.2f} ms per training segment per round",
                 f"- Embedding: {te:.1f} s  ->  {s_per_seg_embed * 1e3:.2f} ms per segment",
                 f"- Phase 2, one round (E2={cfg.phase2.local_epochs}): {t2:.1f} s  ->  {s_per_flow_round * 1e3:.3f} ms per flow per round",
                 f"- Estimated single E1 run at R1={sched['R1']}, R2={sched['R2']}: {e1_h:.2f} GPU-h", '',
                 budget.to_markdown(index=False), '',
                 'Check the Phase-1 share against the 12 h session limit; if it does not fit, use --encoder distilbert.']
        rundir.write_text('probe_report.md', '\n'.join(lines))
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
