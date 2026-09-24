"""Where raw files come from. Swappable per dataset config (`dataset.source.kind`)
so moving between a local disk, an attached Kaggle dataset, kagglehub or plain
HTTP downloads is a config change, not a code change.

    kind: local      root: /path/to/TON_IoT
    kind: kaggle     root: /kaggle/input/<slug>          (attached in the notebook)
    kind: kagglehub  slug: <owner>/<dataset>             (downloaded on first use)
    kind: http       urls: [..], cache_dir: data/raw/x   (downloaded on first use)

CLI example: --set dataset.source.kind=kaggle --set dataset.source.root=/kaggle/input/ton-iot-pcaps
"""

import glob
import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import List

from fhf.common.config import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class DatasetFiles:
    root: str
    pcaps: List[str]
    label_files: List[str]


def _resolve_root(source) -> str:
    kind = source.get('kind', 'local')
    if kind in ('local', 'kaggle'):
        return resolve_path(source['root'])
    if kind == 'kagglehub':
        import kagglehub  # optional dependency, only needed for this kind
        if not source.get('slug'):
            raise ValueError("dataset.source.kind=kagglehub needs dataset.source.slug")
        return kagglehub.dataset_download(source['slug'])
    if kind == 'http':
        cache = resolve_path(source['cache_dir'])
        os.makedirs(cache, exist_ok=True)
        for url in source.get('urls') or []:
            target = os.path.join(cache, os.path.basename(url.split('?')[0]))
            if not os.path.exists(target):
                logger.info(f"Downloading {url} -> {target}")
                urllib.request.urlretrieve(url, target)
        return cache
    raise ValueError(f"Unknown dataset.source.kind '{kind}' (local | kaggle | kagglehub | http)")


def _glob_all(root: str, patterns: List[str]) -> List[str]:
    found = set()
    for pattern in patterns:
        found.update(p for p in glob.glob(os.path.join(root, pattern), recursive=True) if os.path.isfile(p))
    return sorted(found)


def resolve_files(cfg) -> DatasetFiles:
    source = cfg.dataset.source
    root = _resolve_root(source)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Dataset root '{root}' does not exist. Point dataset.source.root at the dataset "
            f"(e.g. --set dataset.source.root=/kaggle/input/<slug>).")
    pcaps = [p for p in _glob_all(root, source.get('pcap_globs', [])) if not p.endswith(('.csv', '.txt', '.md'))]
    labels = _glob_all(root, source.get('label_globs', []))
    logger.info(f"{cfg.dataset.name}: root={root}, {len(pcaps)} capture files, {len(labels)} label files")
    return DatasetFiles(root=root, pcaps=pcaps, label_files=labels)
