"""A9: payload-only classifier = the aggregated Phase-1 head W_c on phi*, evaluated
once on the held-out test payload segments (no training, no encoder re-tuning).

Segment -> flow rule (declared before looking at test labels): average the
softmax probabilities of a flow's readable segments, take the argmax.
Coverage: only flows with m_i = 1 can be predicted. The metric set is reported
on that clearly labelled evaluable subset, together with coverage overall and
per class, so flows with m_i = 0 never silently leave the denominator.
"""

import logging
import os

import numpy as np
import pandas as pd
import torch

from fhf.common.config import Config
from fhf.common.metrics import classification_report
from fhf.common.rundir import source_run
from fhf.common.utils import Timer, resolve_device, set_seed
from fhf.phase1.federated_tune import predict_segments, segment_table
from fhf.phase1.lora_model import PayloadEncoder
from fhf.phase1.run_phase1 import phase1_dir
from fhf.pipeline import prepare

logger = logging.getLogger(__name__)

RULE = 'mean of segment softmax probabilities, argmax'


def run_payload_head(cfg, rundir, resume: bool = False):
    e1_path, e1_cfg = source_run(cfg, 'E1')
    work = Config({**e1_cfg.to_dict(), 'smoke': cfg.get('smoke', False)})
    device = resolve_device(cfg.device)
    set_seed(int(cfg.seed))
    prep = prepare(work)
    ckpt = os.path.join(phase1_dir(work), 'best.pt')
    rundir.add_input('phase1_checkpoint', ckpt)
    rundir.manifest.update({'source_run': e1_path, 'segment_to_flow_rule': RULE})

    model = PayloadEncoder(work, len(prep.names), device, use_lora=True)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    seg = segment_table(prep.frame, prep.y)
    test_seg = seg[seg.role == 'test'].reset_index(drop=True)

    timer = Timer()
    with timer('inference'):
        res = predict_segments(model, state, test_seg, int(work.encoder.embed_batch_size))
    probs = pd.DataFrame(res.probs, columns=prep.names)
    probs['flow_uid'] = test_seg['flow_uid'].to_numpy()
    flow_probs = probs.groupby('flow_uid', sort=False)[prep.names].mean()

    test = prep.frame[prep.frame.role == 'test'].set_index('flow_uid')
    y_all = pd.Series(prep.y[(prep.frame.role == 'test').to_numpy()], index=test.index)
    evaluable = flow_probs.index
    y_true = y_all.loc[evaluable].to_numpy()
    y_pred = flow_probs.to_numpy().argmax(1)
    report = classification_report(y_true, y_pred, prep.names, client_ids=test.loc[evaluable, 'client'].to_numpy())

    coverage = len(evaluable) / max(len(test), 1)
    per_class = report['per_class']
    per_class['flows_in_test'] = [int((y_all == i).sum()) for i in range(len(prep.names))]
    per_class['coverage'] = per_class['support'] / per_class['flows_in_test'].clip(lower=1)
    report['per_class'] = per_class

    rundir.write_evaluation(report, prep.names, extra={
        'subset': 'evaluable (m_i = 1) test flows', 'coverage': coverage,
        'evaluable_flows': int(len(evaluable)), 'test_flows_total': int(len(test)), 'rule': RULE})
    pred = pd.DataFrame({'flow_uid': evaluable, 'client': test.loc[evaluable, 'client'].to_numpy(),
                         'y_true': y_true, 'y_pred': y_pred})
    for n in prep.names:
        pred[f'p_{n}'] = flow_probs[n].to_numpy().astype(np.float32)
    pred.to_parquet(rundir.file('predictions.parquet'), index=False)
    rundir.write_resources([{'name': 'time_inference_s', 'value': round(timer.totals['inference'], 3), 'unit': 's'},
                            {'name': 'segments_scored', 'value': int(len(test_seg)), 'unit': ''}])
    logger.info(f"A9 payload-only head: macro-F1 {report['summary']['macro_f1']:.4f} on {len(evaluable)} flows "
                f"(coverage {coverage:.1%})")
    return {'report': report}
