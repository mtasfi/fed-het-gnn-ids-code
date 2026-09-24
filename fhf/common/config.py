"""Config loading.

Merge order: configs/base.yaml -> [configs/encoders/<encoder>.yaml] -> configs/datasets/<dataset>.yaml ->
configs/experiments/<experiment>.yaml (its `overrides:` block) -> CLI `--set key=value`.
"""

import copy
import os
from typing import Any, Dict, Iterable, List, Optional

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(REPO_ROOT, 'configs')


class Config(dict):
    """Dict with attribute access. Nested dicts are wrapped once, at construction,
    so `cfg.graph.kappa = 5` writes into the real nested object."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, Config):
                self[key] = Config(value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() if isinstance(v, Config) else copy.deepcopy(v) for k, v in self.items()}

    def get_path(self, dotted: str, default=None):
        node = self
        for key in dotted.split('.'):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def deep_merge(base: Dict, override: Dict) -> Dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_nested(cfg: Dict, dotted_key: str, value: Any):
    keys = dotted_key.split('.')
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def parse_set_args(pairs: Optional[Iterable[str]]) -> Dict[str, Any]:
    """['graph.shares_host.kappa=10', 'dataset.source.kind=kaggle'] -> typed dict.
    Values are parsed as YAML, so numbers, booleans, null and lists work."""
    out = {}
    for pair in pairs or []:
        if '=' not in pair:
            raise ValueError(f"--set expects key=value, got '{pair}'")
        key, raw = pair.split('=', 1)
        out[key.strip()] = yaml.safe_load(raw)
    return out


def _read_yaml(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_experiment_spec(experiment: str) -> Dict:
    path = os.path.join(CONFIG_DIR, 'experiments', f'{experiment}.yaml')
    if not os.path.exists(path):
        available = sorted(f[:-5] for f in os.listdir(os.path.join(CONFIG_DIR, 'experiments')) if f.endswith('.yaml'))
        raise FileNotFoundError(f"No experiment config '{experiment}'. Available: {available}")
    return _read_yaml(path)


def load_config(dataset: str, experiment: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None,
                encoder: Optional[str] = None) -> Config:
    cfg = _read_yaml(os.path.join(CONFIG_DIR, 'base.yaml'))
    if encoder:
        cfg = deep_merge(cfg, _read_yaml(os.path.join(CONFIG_DIR, 'encoders', f'{encoder}.yaml')))

    ds_path = os.path.join(CONFIG_DIR, 'datasets', f'{dataset}.yaml')
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"No dataset config '{dataset}' at {ds_path}")
    cfg = deep_merge(cfg, _read_yaml(ds_path))

    if experiment:
        spec = load_experiment_spec(experiment)
        cfg = deep_merge(cfg, spec.get('overrides') or {})
        cfg['experiment'] = {k: v for k, v in spec.items() if k != 'overrides'}
        cfg['experiment']['overrides'] = spec.get('overrides') or {}
    else:
        cfg['experiment'] = {'id': 'E0', 'variant': 'data', 'pipeline': 'data'}

    for key, value in (overrides or {}).items():
        set_nested(cfg, key, value)
    cfg['cli_overrides'] = dict(overrides or {})

    return Config(cfg)


def require_locked(cfg: Config, keys: List[str]):
    """Fail loudly when a value that must be locked in E0 is still null."""
    missing = [k for k in keys if cfg.get_path(k) is None]
    if missing:
        raise RuntimeError(
            f"These settings must be locked in E0 before this run: {missing}. "
            f"Set them in configs/base.yaml (or configs/datasets/*.yaml) after reading the E0 report, "
            f"or pass --set {missing[0]}=<value> for a one-off."
        )


def resolve_path(path: str) -> str:
    """Relative paths in configs are relative to the repo root, not the CWD."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def dump_yaml(obj: Dict, path: str):
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
