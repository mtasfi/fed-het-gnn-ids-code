"""Single entry point for every modeled experiment (E1, E4-E6, A1-A9, B1-B5).

Each (experiment, dataset, variant, alpha, seed) writes one immutable run directory
runs/{exp}/{dataset}/{variant}/alpha={a}/seed={s}/ and mirrors scalars to Comet.

Examples
    python experiments/run.py --exp E1 --dataset toniot                 # seeds from configs/experiments/E1.yaml
    python experiments/run.py --exp A1 --dataset toniot --seed 0
    python experiments/run.py --exp E5 --dataset toniot                 # loops partition.alpha_sweep
    python experiments/run.py --exp B5 --dataset toniot --variant xgboost_smote
    python experiments/run.py --exp E1 --dataset toniot --smoke         # real-data subset, 1+2 rounds
    python experiments/run.py --exp E1 --dataset toniot --encoder distilbert   # fallback encoder
    python experiments/run.py --exp E1 --dataset toniot --resume        # continue after a Kaggle session limit
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fhf.common.config import load_config, load_experiment_spec, parse_set_args, require_locked
from fhf.common.rundir import RunDir
from fhf.common.utils import setup_logging

logger = logging.getLogger('run')

GRAPH_PIPELINES = {'fedhet', 'central_hetgnn', 'missing_sweep', 'efficiency'}


def pipeline_fn(name: str):
    if name == 'fedhet':
        from fhf.pipeline import run_fedhet
        return run_fedhet
    if name == 'fed_mlp':
        from fhf.pipeline import run_fed_mlp
        return run_fed_mlp
    if name == 'central_hetgnn':
        from fhf.pipeline import run_central_hetgnn
        return run_central_hetgnn
    if name == 'classical':
        from fhf.baselines.classical_xgb_rf import run_classical
        return run_classical
    if name == 'missing_sweep':
        from fhf.missing_sweep import run_missing_sweep
        return run_missing_sweep
    if name == 'payload_head':
        from fhf.phase1.payload_head_eval import run_payload_head
        return run_payload_head
    if name == 'efficiency':
        from fhf.efficiency import run_efficiency
        return run_efficiency
    if name == 'fedgatsage':
        from fhf.baselines.fedgatsage.adapter import run_fedgatsage
        return run_fedgatsage
    raise ValueError(f"Pipeline '{name}' has no runner")


def build_cfg(args, seed, alpha, variant):
    overrides = parse_set_args(args.set)
    cfg = load_config(args.dataset, args.exp, overrides, encoder=args.encoder)
    cfg.seed = int(seed)
    if alpha is not None:
        cfg.partition.alpha = alpha
    if variant is not None:
        cfg.experiment['variant'] = variant
    if args.encoder:
        cfg.experiment['variant'] = f"{cfg.experiment['variant']}-{args.encoder}"
    if args.smoke:
        cfg.smoke = True
        cfg.phase1.rounds = min(int(cfg.phase1.rounds), 1)
        cfg.phase2.rounds = min(int(cfg.phase2.rounds), 2)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp', required=True)
    parser.add_argument('--dataset', default='toniot')
    parser.add_argument('--seed', type=int, action='append', help='repeatable; default: seeds in the experiment config')
    parser.add_argument('--alpha', action='append', help="repeatable; e.g. 0.5 or iid; default: partition.alpha")
    parser.add_argument('--variant', action='append', help='B5 only: which classical variant(s)')
    parser.add_argument('--encoder', default=None, help='encoder preset in configs/encoders/ (e.g. distilbert)')
    parser.add_argument('--set', action='append', default=[], help='config override key=value (repeatable)')
    parser.add_argument('--smoke', action='store_true', help='real-data subset, 1 Phase-1 + 2 Phase-2 rounds')
    parser.add_argument('--resume', action='store_true', help='continue from the last round checkpoint')
    parser.add_argument('--overwrite', action='store_true', help='replace a completed run directory')
    args = parser.parse_args()

    spec = load_experiment_spec(args.exp)
    if spec['pipeline'] == 'not_run':
        print(f"{args.exp} is marked not_run: {spec.get('note')}")
        return

    seeds = args.seed or spec.get('seeds') or [0]
    base = load_config(args.dataset, args.exp, parse_set_args(args.set), encoder=args.encoder)
    if args.alpha:
        alphas = [a if a == 'iid' else float(a) for a in args.alpha]
    elif spec.get('alpha_from'):
        require_locked(base, [spec['alpha_from']])
        alphas = list(base.get_path(spec['alpha_from']))
    else:
        require_locked(base, ['partition.alpha'])
        alphas = [base.partition.alpha]
    variants = args.variant or spec.get('variants') or [None]

    setup_logging('INFO')
    fn = pipeline_fn(spec['pipeline'])
    failures = 0
    for alpha in alphas:
        for seed in seeds:
            for variant in variants:
                cfg = build_cfg(args, seed, alpha, variant)
                if spec['pipeline'] in GRAPH_PIPELINES and cfg.graph.use_shares_host:
                    require_locked(cfg, ['graph.shares_host.kappa'])
                rundir = RunDir(cfg, overwrite=args.overwrite, resume=args.resume)
                try:
                    with rundir:
                        setup_logging('INFO', rundir.file('run.log'))
                        logger.info(f"=== {cfg.experiment['id']} / {cfg.experiment['variant']} alpha={alpha} "
                                    f"seed={seed} ===")
                        fn(cfg, rundir, resume=args.resume)
                except Exception:
                    failures += 1
                    logger.exception(f"Run failed and was recorded as failed in {rundir.path}")
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
