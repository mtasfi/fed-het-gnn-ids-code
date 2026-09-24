"""Run directory contract (plan §5).

runs/{experiment_id}/{dataset}/{variant}/alpha={value}/seed={seed}/
    config.yaml  manifest.json  round_metrics.csv  test_metrics.csv
    per_class_metrics.csv  confusion_matrix.csv/.png  resource_metrics.csv
    per_client_metrics.csv  predictions.parquet  checkpoint_or_pointer.txt

A completed run directory is immutable: starting the same run again refuses
unless --overwrite is given. Failed or interrupted runs are never deleted;
they are moved aside with their status and failure reason so unfavourable or
broken runs stay on record (plan: "retain crashed/NaN runs").
"""

import csv
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from fhf.common.config import REPO_ROOT, dump_yaml, resolve_path
from fhf.common.tracking import Tracker, init_tracker

logger = logging.getLogger(__name__)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def git_state() -> Dict:
    def run(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    commit = run('rev-parse', 'HEAD')
    status = run('status', '--porcelain', '--untracked-files=no')
    return {'git_commit': commit, 'git_dirty': bool(status) if status is not None else None}


def file_hash(path: str, full_limit_mb: int = 64) -> Optional[str]:
    """sha256 of the file; files above `full_limit_mb` hash size + first/last 8 MB
    (embedding caches are large, and this is a change detector, not a signature)."""
    if not os.path.exists(path):
        return None
    if os.path.isdir(path):
        h = hashlib.sha256()
        for root, _, files in sorted(os.walk(path)):
            for name in sorted(files):
                sub = file_hash(os.path.join(root, name), full_limit_mb)
                h.update(f'{os.path.relpath(os.path.join(root, name), path)}:{sub}'.encode())
        return h.hexdigest()
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        if size <= full_limit_mb * 2 ** 20:
            for block in iter(lambda: f.read(2 ** 20), b''):
                h.update(block)
        else:
            h.update(str(size).encode())
            h.update(f.read(8 * 2 ** 20))
            f.seek(-8 * 2 ** 20, os.SEEK_END)
            h.update(f.read())
    return h.hexdigest()


def hardware_identity() -> Dict:
    info = {'python': platform.python_version(), 'platform': platform.platform(), 'cpu_count': os.cpu_count()}
    try:
        import torch
        info['torch'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
            info['cuda'] = torch.version.cuda
    except ImportError:
        pass
    for pkg in ('transformers', 'peft', 'torch_geometric', 'sklearn', 'xgboost'):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            pass
    return info


def run_path(cfg) -> str:
    exp = cfg.experiment
    alpha = cfg.get_path('partition.alpha')
    root = resolve_path(cfg.paths.runs_dir)
    if cfg.get('smoke'):
        root = os.path.join(root, '_smoke')
    return os.path.join(root, str(exp['id']), str(cfg.dataset.name), str(exp.get('variant', 'main')),
                        f'alpha={alpha}', f"seed={cfg.seed}")


class RunDir:
    def __init__(self, cfg, overwrite: bool = False, resume: bool = False, tracker: Optional[Tracker] = None):
        self.cfg = cfg
        self.path = run_path(cfg)
        self.overwrite = overwrite
        self.resume = resume
        self.tracker = tracker
        self.manifest: Dict = {}
        self.inputs: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------ lifecycle
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            if self.manifest.get('status') == 'running':
                self.finish('completed')
            return False
        reason = f'{exc_type.__name__}: {exc}'
        self.write_text('failure_traceback.txt', ''.join(traceback.format_exception(exc_type, exc, tb)))
        self.finish('failed', reason)
        return False

    def start(self):
        manifest_path = os.path.join(self.path, 'manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                previous = json.load(f)
            status = previous.get('status')
            if status == 'completed' and not self.overwrite:
                raise FileExistsError(
                    f"{self.path} already holds a completed run (immutable). Use --overwrite to replace it, "
                    f"or change seed/variant.")
            if self.resume and status != 'completed':
                return self._resume(previous)
            aside = f"{self.path}.{status or 'unknown'}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            shutil.move(self.path, aside)
            logger.warning(f"Previous run ({status}) moved to {aside}")
        os.makedirs(self.path, exist_ok=True)

        dump_yaml(self.cfg.to_dict(), os.path.join(self.path, 'config.yaml'))
        exp = self.cfg.experiment
        self.manifest = {
            'run_id': f"{exp['id']}/{self.cfg.dataset.name}/{exp.get('variant', 'main')}/"
                      f"alpha={self.cfg.get_path('partition.alpha')}/seed={self.cfg.seed}",
            'experiment_id': exp['id'],
            'variant': exp.get('variant', 'main'),
            'dataset': self.cfg.dataset.name,
            'alpha': self.cfg.get_path('partition.alpha'),
            'seed': self.cfg.seed,
            'smoke': bool(self.cfg.get('smoke')),
            'start': _now(),
            'end': None,
            'status': 'running',
            'failure_reason': None,
            **git_state(),
            'hardware': hardware_identity(),
            'inputs': {},
            'outputs': {},
        }
        if self.tracker is None:
            self.tracker = init_tracker(self.cfg, self.manifest['run_id'])
        self.tracker.log_config(self.cfg.to_dict())
        self.manifest['comet_url'] = self.tracker.url
        self._write_manifest()
        logger.info(f"Run directory: {self.path}")
        return self

    def _resume(self, previous: Dict):
        """Continue an interrupted run in place: round checkpoints and round_metrics.csv are kept."""
        self.manifest = previous
        self.manifest['status'] = 'running'
        self.manifest.setdefault('resumed_at', []).append(_now())
        self.inputs = previous.get('inputs', {})
        if self.tracker is None:
            self.tracker = init_tracker(self.cfg, self.manifest['run_id'], extra_tags=['resumed'])
        self.tracker.log_config(self.cfg.to_dict())
        self._write_manifest()
        logger.info(f"Resuming run in {self.path}")
        return self

    def finish(self, status: str, reason: Optional[str] = None):
        self.manifest['status'] = status
        self.manifest['failure_reason'] = reason
        self.manifest['end'] = _now()
        self.manifest['outputs'] = {
            name: file_hash(os.path.join(self.path, name))
            for name in sorted(os.listdir(self.path))
            if name not in ('manifest.json',) and os.path.isfile(os.path.join(self.path, name))
        }
        self._write_manifest()
        if self.tracker is not None:
            self.tracker.set_status(status)
            self.tracker.end()
        logger.info(f"Run {status}: {self.path}" + (f" ({reason})" if reason else ''))

    def _write_manifest(self):
        self.manifest['inputs'] = self.inputs
        with open(os.path.join(self.path, 'manifest.json'), 'w') as f:
            json.dump(self.manifest, f, indent=2, default=str)

    # ------------------------------------------------------------ records
    def add_input(self, name: str, path: str):
        self.inputs[name] = file_hash(path)
        self._write_manifest()

    def file(self, name: str) -> str:
        return os.path.join(self.path, name)

    def write_text(self, name: str, text: str):
        with open(self.file(name), 'w') as f:
            f.write(text)

    def append_csv(self, name: str, row: Dict):
        path = self.file(name)
        exists = os.path.exists(path)
        if exists:
            with open(path) as f:
                header = next(csv.reader(f))
            missing = [k for k in row if k not in header]
            if missing:  # rewrite with the widened header rather than drop fields
                df = pd.read_csv(path)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                df.to_csv(path, index=False)
                return
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header if exists else list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in (header if exists else row.keys())})

    def log_round(self, row: Dict):
        """One row of round_metrics.csv: phase, round, scope, losses, macro-F1, lr, clients, bytes, elapsed."""
        self.append_csv('round_metrics.csv', row)
        if self.tracker is not None:
            scope = row.get('scope', 'global')
            phase = row.get('phase', '')
            step = int(row['round']) if row.get('round') is not None else None
            prefix = f"p{phase}/{scope}/" if phase != '' else f"{scope}/"
            self.tracker.log_metrics({k: v for k, v in row.items() if k not in ('phase', 'round', 'scope')},
                                     step=step, prefix=prefix)

    def write_evaluation(self, report: Dict, class_names: List[str], extra: Optional[Dict] = None,
                         prefix: str = '', status: str = 'ok'):
        """test_metrics.csv (one row), per_class_metrics.csv, confusion_matrix.csv/.png, per_client_metrics.csv."""
        tag = f'{prefix}_' if prefix else ''
        row = {'run_id': self.manifest.get('run_id'), 'status': status, **report['summary'], **(extra or {})}
        pd.DataFrame([row]).to_csv(self.file(f'{tag}test_metrics.csv'), index=False)
        report['per_class'].to_csv(self.file(f'{tag}per_class_metrics.csv'), index=False)
        cm = pd.DataFrame(report['confusion'], index=class_names, columns=class_names)
        cm.index.name = 'true\\pred'
        cm.to_csv(self.file(f'{tag}confusion_matrix.csv'))
        _plot_confusion(report['confusion'], class_names, self.file(f'{tag}confusion_matrix.png'))
        if report.get('per_client') is not None:
            report['per_client'].to_csv(self.file(f'{tag}per_client_metrics.csv'), index=False)

        if self.tracker is not None:
            mprefix = f'{prefix}/test/' if prefix else 'test/'
            self.tracker.log_metrics(report['summary'], prefix=mprefix)
            per_class = report['per_class']
            self.tracker.log_metrics({f"f1/{r.label}": r.f1 for r in per_class.itertuples()}, prefix=mprefix)
            self.tracker.log_metrics({f"support/{r.label}": r.support for r in per_class.itertuples()}, prefix=mprefix)
            self.tracker.log_confusion_matrix(report['confusion'], class_names, title=f'{tag}confusion_matrix')

    def write_resources(self, rows: List[Dict]):
        pd.DataFrame(rows).to_csv(self.file('resource_metrics.csv'), index=False)
        if self.tracker is not None:
            for r in rows:
                if 'value' in r and isinstance(r['value'], (int, float)):
                    self.tracker.log_metrics({f"resource/{r['name']}": r['value']})

    def write_pointer(self, target: str):
        self.write_text('checkpoint_or_pointer.txt', target + '\n')


def _plot_confusion(cm: np.ndarray, labels: List[str], path: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * n, 1.0 + 0.55 * n))
    ax.imshow(np.log1p(cm), cmap='Blues')
    ax.set_xticks(range(n), labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title('Raw counts (colour: log scale)', fontsize=9)
    vmax = np.log1p(cm).max() if cm.size else 1
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=6,
                    color='white' if np.log1p(cm[i, j]) > 0.6 * vmax else 'black')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def source_run(cfg, experiment_id: str, variant: str = 'main'):
    """(path, config) of the completed run another experiment builds on, e.g. E4 / A9 / E6
    reuse E1 with the same dataset, alpha and seed. Fails loudly if it is missing or incomplete."""
    from fhf.common.config import Config
    import yaml

    probe = Config(cfg.to_dict())
    probe.experiment = {**cfg.experiment, 'id': experiment_id, 'variant': variant}
    path = run_path(probe)
    manifest = os.path.join(path, 'manifest.json')
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"{experiment_id} run not found at {path}; run it first (same dataset/alpha/seed)")
    with open(manifest) as f:
        status = json.load(f).get('status')
    if status != 'completed':
        raise RuntimeError(f"{experiment_id} run at {path} has status '{status}', not completed")
    with open(os.path.join(path, 'config.yaml')) as f:
        return path, Config(yaml.safe_load(f))
