"""Comet ML mirror.

Rules (plan §5 + data-localisation claim):
  * CSVs in the run directory are the record. Comet only mirrors them, so a
    Comet outage never loses a result and never aborts a run.
  * Only three kinds of data leave the machine: numeric metrics, flattened
    config values, and tags (plus the numeric confusion matrix). No payload
    text or samples, no IP addresses, no per-flow rows, no file assets. The
    Tracker API below does not offer a way to send anything else.
"""

import logging
import math
import os
import re
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_SECRET_KEYS = re.compile(r'api_key|token|password_secret', re.I)
# A config string that looks like an IPv4/IPv6 address never goes out, whatever its key
_IP_LIKE = re.compile(r'^\s*(\d{1,3}\.){3}\d{1,3}\s*$|^\s*[0-9a-f]*:[0-9a-f:]+\s*$', re.I)


def flatten(d: Dict, prefix: str = '') -> Dict:
    out = {}
    for key, value in d.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            out.update(flatten(value, f'{name}.'))
        else:
            out[name] = value
    return out


def _safe_param(key: str, value):
    if _SECRET_KEYS.search(key):
        return None
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return None if _IP_LIKE.match(value) else value[:500]
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (int, float, str, bool)) or v is None for v in value):
            items = [v for v in value if not (isinstance(v, str) and _IP_LIKE.match(v))]
            return ','.join(str(v) for v in items)[:500]
        return None
    return None


class Tracker:
    """Wraps a Comet experiment; a no-op when Comet is disabled or unreachable."""

    def __init__(self, experiment=None):
        self.exp = experiment

    @property
    def enabled(self) -> bool:
        return self.exp is not None

    def _call(self, fn_name: str, *args, **kwargs):
        if not self.enabled:
            return
        try:
            getattr(self.exp, fn_name)(*args, **kwargs)
        except Exception as e:  # losing a mirror entry is never worth losing a run
            logger.warning(f"Comet {fn_name} failed: {e}")

    def log_config(self, cfg: Dict):
        params = {}
        for key, value in flatten(cfg).items():
            safe = _safe_param(key, value)
            if safe is not None:
                params[key] = safe
        self._call('log_parameters', params)

    def log_params(self, params: Dict):
        self._call('log_parameters', {k: v for k, v in params.items() if _safe_param(k, v) is not None})

    def log_metrics(self, metrics: Dict, step: Optional[int] = None, prefix: str = ''):
        flat = {}
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            flat[f'{prefix}{key}'] = value
        if flat:
            self._call('log_metrics', flat, step=step)

    def log_confusion_matrix(self, matrix, labels: List[str], title: str = 'confusion_matrix'):
        self._call('log_confusion_matrix', matrix=[[int(v) for v in row] for row in matrix],
                   labels=list(labels), title=title, file_name=f'{title}.json')

    def add_tags(self, tags: Iterable[str]):
        self._call('add_tags', [str(t) for t in tags])

    def set_status(self, status: str):
        self.add_tags([f'status:{status}'])

    @property
    def url(self) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            return self.exp.url
        except Exception:
            return None

    def end(self):
        self._call('end')


def standard_tags(cfg) -> List[str]:
    """Tags that make any run findable in Comet: experiment, variant, dataset, alpha, seed, tier."""
    exp = cfg.get('experiment', {})
    alpha = cfg.get_path('partition.alpha')
    tags = [
        f"exp:{exp.get('id', 'unknown')}",
        f"variant:{exp.get('variant', 'main')}",
        f"dataset:{cfg.get_path('dataset.name')}",
        f"alpha:{'iid' if alpha == 'iid' else alpha}",
        f"seed:{cfg.get('seed')}",
        f"pipeline:{exp.get('pipeline', 'unknown')}",
        'sprint' if not cfg.get('paper_final', False) else 'paper-final',
    ]
    if exp.get('tier'):
        tags.append(f"tier:{exp['tier']}")
    if cfg.get('smoke'):
        tags.append('smoke')
    for tag in exp.get('tags', []) or []:
        tags.append(str(tag))
    return tags


def _kaggle_secret(name: str):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def init_tracker(cfg, run_name: str, extra_tags: Iterable[str] = ()) -> Tracker:
    tcfg = cfg.get('tracking', {})
    if not tcfg.get('comet', False):
        logger.info("Comet mirror disabled by config")
        return Tracker(None)
    try:
        import comet_ml

        kwargs = {'project_name': tcfg.get('project', 'fedhetformer-ids')}
        if tcfg.get('workspace'):
            kwargs['workspace'] = tcfg['workspace']
        # Everything automatic is switched off: stdout capture could carry whatever
        # a library prints, code/git-patch upload is not a metric, and framework
        # auto-logging bypasses the allowlist above.
        exp_config = comet_ml.ExperimentConfig(
            name=run_name,
            auto_output_logging=None,
            parse_args=False,
            log_code=False,
            log_graph=False,
            log_git_metadata=False,
            log_git_patch=False,
            log_env_details=False,
            log_env_host=False,
            log_env_network=False,
            log_env_disk=False,
            auto_log_co2=False,
            auto_param_logging=False,
            auto_metric_logging=False,
            auto_histogram_weight_logging=False,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False,
        )
        api_key = tcfg.get('api_key') or os.environ.get('COMET_API_KEY') or _kaggle_secret('COMET_API_KEY')
        if not api_key:
            logger.warning("Comet enabled but no COMET_API_KEY (env or Kaggle Secret); CSV records only")
            return Tracker(None)
        exp = comet_ml.start(api_key=api_key, experiment_config=exp_config, **kwargs)
        tracker = Tracker(exp)
        tracker.add_tags(list(standard_tags(cfg)) + list(extra_tags))
        logger.info(f"Comet mirror: project '{kwargs['project_name']}', run '{run_name}'")
        return tracker
    except Exception as e:
        logger.warning(f"Comet unavailable ({e}); continuing with CSV records only")
        return Tracker(None)
