"""Flow-feature matrix used by every model (x^f_i; m_i is appended by the callers).

IP addresses and other identity columns are never features: they are rejected
here by name, so no model code path can receive them (plan pitfall 3).
"""

import re
from typing import List

import numpy as np
import pandas as pd

from fhf.data.flow_extractor import FEATURE_COLUMNS

_IP_NAME = re.compile(r'(^|_)ip(v4|v6)?($|_)', re.I)


def feature_columns(cfg) -> List[str]:
    forbidden = set(cfg.features.forbidden_columns)
    cols = [c for c in FEATURE_COLUMNS if c not in set(cfg.features.get('exclude') or [])]
    bad = [c for c in cols if c in forbidden or _IP_NAME.search(c)]
    if bad:
        raise ValueError(f"Identity columns requested as model features: {bad}")
    return cols


def feature_matrix(df: pd.DataFrame, cfg) -> np.ndarray:
    cols = feature_columns(cfg)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Flow table lacks feature columns {missing}")
    return df[cols].to_numpy(dtype=np.float64)
