"""B5: centralized XGBoost / Random Forest, each with and without training-only SMOTE (CPU).

Inputs: the E1 flow statistics (federated global scaler applied) + has_payload.
No IP / identifier, graph edge, payload text or payload embedding.
Leakage rule: SMOTE is fitted on the training partition only, after the split;
validation and test stay naturally distributed and untouched.
Tuning: one fixed compact grid per model family (4 configurations each, shared
by the plain and SMOTE variants); selection by validation macro-F1; the
selected model is evaluated once on test.
"""

import json
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from fhf.common.metrics import classification_report
from fhf.common.utils import peak_rss_mb, set_seed
from fhf.pipeline import prepare

logger = logging.getLogger(__name__)

GRIDS = {
    'xgboost': [
        {'max_depth': 6, 'n_estimators': 300, 'learning_rate': 0.1},
        {'max_depth': 10, 'n_estimators': 300, 'learning_rate': 0.1},
        {'max_depth': 6, 'n_estimators': 600, 'learning_rate': 0.05},
        {'max_depth': 10, 'n_estimators': 600, 'learning_rate': 0.05},
    ],
    'random_forest': [
        {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1},
        {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 3},
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_leaf': 1},
        {'n_estimators': 600, 'max_depth': None, 'min_samples_leaf': 1},
    ],
}
SMOTE_K = 5


def _make_model(family: str, params: Dict, seed: int, num_classes: int):
    if family == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(**params, objective='multi:softprob', num_class=num_classes, tree_method='hist',
                             random_state=seed, n_jobs=-1, eval_metric='mlogloss')
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**params, random_state=seed, n_jobs=-1)


def apply_smote(x: np.ndarray, y: np.ndarray, seed: int, k: int = SMOTE_K) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Oversample every minority class to the majority count. A class with fewer
    than k+1 samples uses k = n_c - 1; a singleton class cannot be interpolated and
    is left as is. All of it is recorded."""
    from imblearn.over_sampling import SMOTE

    counts = pd.Series(y).value_counts()
    majority = int(counts.max())
    too_small = {int(c): int(n) for c, n in counts.items() if n < 2}
    eligible = counts[(counts >= 2) & (counts < majority)]
    audit = {'before': {int(c): int(n) for c, n in counts.items()}, 'k_requested': k, 'singleton_classes': too_small}
    if eligible.empty:
        audit.update({'k_used': None, 'after': audit['before']})
        return x, y, audit
    k_used = int(min(k, eligible.min() - 1))
    smote = SMOTE(sampling_strategy={int(c): majority for c in eligible.index}, k_neighbors=k_used, random_state=seed)
    xs, ys = smote.fit_resample(x, y)
    audit.update({'k_used': k_used, 'after': {int(c): int(n) for c, n in pd.Series(ys).value_counts().items()}})
    return xs, ys, audit


def run_classical(cfg, rundir, resume: bool = False) -> Dict:
    variant = cfg.experiment['variant']
    family = 'xgboost' if variant.startswith('xgboost') else 'random_forest'
    use_smote = variant.endswith('_smote')
    seed = int(cfg.seed)
    set_seed(seed)
    prep = prepare(cfg)
    m = prep.frame['has_payload'].to_numpy().astype(np.float32)[:, None]
    x = np.concatenate([prep.x, m], axis=1)
    role = prep.frame['role'].to_numpy()
    tr, va, te = role == 'train', role == 'val', role == 'test'
    x_tr, y_tr = x[tr], prep.y[tr]
    C = len(prep.names)

    smote_audit = {'applied': False}
    t0 = time.perf_counter()
    if use_smote:
        x_tr, y_tr, smote_audit = apply_smote(x_tr, y_tr, seed)
        smote_audit['applied'] = True
        smote_audit['sampler_seed'] = seed
    smote_seconds = time.perf_counter() - t0

    # XGBoost needs contiguous labels over the classes it sees in training
    present = np.unique(y_tr)
    to_local = {c: i for i, c in enumerate(present)}
    y_tr_local = np.array([to_local[c] for c in y_tr])

    best = None
    tuning_rows: List[Dict] = []
    for i, params in enumerate(GRIDS[family]):
        model = _make_model(family, params, seed, len(present))
        t0 = time.perf_counter()
        model.fit(x_tr, y_tr_local)
        fit_s = time.perf_counter() - t0
        pred_va = present[model.predict(x[va]).astype(int).ravel()]
        f1 = classification_report(prep.y[va], pred_va, prep.names)['summary']['macro_f1']
        tuning_rows.append({'config': i, **{k: v for k, v in params.items()}, 'val_macro_f1': f1, 'fit_seconds': fit_s})
        rundir.log_round({'phase': 'tune', 'round': i, 'scope': 'global', 'val_macro_f1': f1, 'fit_seconds': fit_s})
        if best is None or f1 > best[0]:
            best = (f1, i, model, fit_s)
    pd.DataFrame(tuning_rows).to_csv(rundir.file('tuning.csv'), index=False)

    f1, idx, model, fit_s = best
    t0 = time.perf_counter()
    proba_local = model.predict_proba(x[te])
    infer_s = time.perf_counter() - t0
    proba = np.zeros((te.sum(), C), dtype=np.float32)
    proba[:, present] = proba_local
    y_pred = proba.argmax(1)
    report = classification_report(prep.y[te], y_pred, prep.names, client_ids=prep.frame['client'].to_numpy()[te])

    pred = pd.DataFrame({'flow_uid': prep.frame['flow_uid'].to_numpy()[te], 'client': prep.frame['client'].to_numpy()[te],
                         'y_true': prep.y[te], 'y_pred': y_pred})
    for i, n in enumerate(prep.names):
        pred[f'p_{n}'] = proba[:, i]
    pred.to_parquet(rundir.file('predictions.parquet'), index=False)

    counts = pd.DataFrame({
        'label': prep.names,
        'train_before': [smote_audit.get('before', {}).get(i, int((prep.y[tr] == i).sum())) for i in range(C)],
        'train_after': [smote_audit.get('after', {}).get(i, int((y_tr == i).sum())) for i in range(C)],
        'val': [int((prep.y[va] == i).sum()) for i in range(C)],
        'test': [int((prep.y[te] == i).sum()) for i in range(C)],
    })
    counts.to_csv(rundir.file('class_count_audit.csv'), index=False)
    with open(rundir.file('smote_audit.json'), 'w') as f:
        json.dump(smote_audit, f, indent=2)
    rundir.manifest['hyperparameters'] = {'family': family, 'selected_config': idx, **GRIDS[family][idx]}

    rundir.write_evaluation(report, prep.names, extra={'checkpoint_round': idx, 'best_val_macro_f1': f1})
    rundir.write_resources([
        {'name': 'fit_seconds_selected', 'value': round(fit_s, 3), 'unit': 's'},
        {'name': 'fit_seconds_all_configs', 'value': round(sum(r['fit_seconds'] for r in tuning_rows), 3), 'unit': 's'},
        {'name': 'smote_seconds', 'value': round(smote_seconds, 3), 'unit': 's'},
        {'name': 'inference_seconds_test', 'value': round(infer_s, 4), 'unit': 's'},
        {'name': 'inference_ms_per_flow', 'value': 1000 * infer_s / max(int(te.sum()), 1), 'unit': 'ms'},
        {'name': 'peak_rss_mb', 'value': round(peak_rss_mb(), 1), 'unit': 'MB'},
        {'name': 'train_rows_after_sampling', 'value': int(len(y_tr)), 'unit': ''},
    ])
    return {'report': report}
