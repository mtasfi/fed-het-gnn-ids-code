"""The one metric implementation every run uses (plan §5).

Headline: macro-F1 and balanced accuracy over the FIXED label set of the run
(classes absent from the test split still appear in per-class output with
support 0, so they never silently vanish). Federated robustness: worst-client
macro-F1 from the same held-out predictions grouped by client.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _prf_from_confusion(cm: np.ndarray, zero_division: float):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(tp + fp > 0, tp / (tp + fp), zero_division)
        recall = np.where(tp + fn > 0, tp / (tp + fn), zero_division)
        # Same definition as sklearn: 2TP / (2TP + FP + FN), zero_division when all three are 0
        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), zero_division)
    return precision, recall, f1, tp, fp, fn, support


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: List[str],
    client_ids: Optional[Sequence[int]] = None,
    zero_division: float = 0.0,
) -> Dict:
    """Returns {'summary': {...}, 'per_class': DataFrame, 'confusion': ndarray, 'per_client': DataFrame|None}.

    Macro averages are taken over classes with test support > 0: a class that the
    split legitimately lacks cannot be scored, and averaging a constant for it in
    would move the headline by an arbitrary amount. The per-class table keeps it
    with support 0 so the gap is visible.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    precision, recall, f1, tp, fp, fn, support = _prf_from_confusion(cm, zero_division)
    present = support > 0

    summary = {
        'n_samples': int(len(y_true)),
        'accuracy': float((y_true == y_pred).mean()) if len(y_true) else float('nan'),
        'balanced_accuracy': float(recall[present].mean()) if present.any() else float('nan'),
        'macro_precision': float(precision[present].mean()) if present.any() else float('nan'),
        'macro_recall': float(recall[present].mean()) if present.any() else float('nan'),
        'macro_f1': float(f1[present].mean()) if present.any() else float('nan'),
        'weighted_f1': float(np.average(f1, weights=support)) if support.sum() else float('nan'),
        'n_classes_scored': int(present.sum()),
        'n_classes_total': int(len(class_names)),
    }

    per_class = pd.DataFrame({
        'label': class_names,
        'label_index': labels,
        'support': support.astype(int),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp.astype(int),
        'fp': fp.astype(int),
        'fn': fn.astype(int),
        'zero_division_policy': zero_division,
    })

    per_client = None
    if client_ids is not None:
        client_ids = np.asarray(client_ids)
        rows = []
        for cid in sorted(np.unique(client_ids)):
            mask = client_ids == cid
            sub = classification_report(y_true[mask], y_pred[mask], class_names, None, zero_division)['summary']
            rows.append({'client': int(cid), 'n_samples': sub['n_samples'], 'macro_f1': sub['macro_f1'],
                         'balanced_accuracy': sub['balanced_accuracy'], 'accuracy': sub['accuracy']})
        per_client = pd.DataFrame(rows)
        summary['worst_client_macro_f1'] = float(per_client['macro_f1'].min()) if len(per_client) else float('nan')
        summary['worst_client'] = int(per_client.loc[per_client['macro_f1'].idxmin(), 'client']) if len(per_client) else -1

    return {'summary': summary, 'per_class': per_class, 'confusion': cm, 'per_client': per_client}


def macro_f1(y_true, y_pred, num_classes: int) -> float:
    return classification_report(y_true, y_pred, [str(i) for i in range(num_classes)])['summary']['macro_f1']


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = ~np.isnan(values) & (weights > 0)
    if not ok.any():
        return float('nan')
    return float(np.average(values[ok], weights=weights[ok]))
