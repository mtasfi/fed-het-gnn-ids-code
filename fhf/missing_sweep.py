"""E4: missing-payload sweep on the fixed test split, using the E1 checkpoint (no retraining).

Available payloads are masked deterministically at each rate in
evaluation.mask_rates (default {0, 0.25, 0.5, 1.0}). A masked flow gets m_i = 0,
its payload nodes and incident edges are removed and the encoder is skipped for
it, i.e. it is handled exactly like a flow that never had a readable payload.
Masked sets are nested (hash-ranked per flow), so the curve is monotone in what
is removed. At 100% the detector runs on flow statistics alone.
"""

import logging

import pandas as pd
import torch

from fhf.common.config import Config
from fhf.common.rundir import source_run
from fhf.common.utils import Timer, resolve_device, set_seed
from fhf.phase1 import embed_once
from fhf.phase2.build_heterograph import payload_mask
from fhf.phase2.hetero_sage import build_gnn
from fhf.pipeline import build_client_graphs, evaluate_test, prepare

logger = logging.getLogger(__name__)


def run_missing_sweep(cfg, rundir, resume: bool = False):
    e1_path, e1_cfg = source_run(cfg, 'E1')
    rates = list(cfg.experiment.get('mask_rates', [0.0, 0.25, 0.5, 1.0]))
    # the model and graph settings must be E1's exactly; only the test inputs change
    work = Config({**e1_cfg.to_dict(), 'smoke': cfg.get('smoke', False)})
    device = resolve_device(cfg.device)
    set_seed(int(cfg.seed))
    prep = prepare(work)
    tag = embed_once.embedding_tag(work, 'lora')
    emb = embed_once.load_cache(work, tag, prep.frame['flow_uid'].to_numpy())
    state = torch.load(f'{e1_path}/phase2_best.pt', map_location='cpu', weights_only=False)
    model = build_gnn(work, prep.x.shape[1] + 1, emb.dim, len(prep.names)).to(device)
    rundir.add_input('e1_checkpoint', f'{e1_path}/phase2_best.pt')
    rundir.manifest['source_run'] = e1_path

    test = (prep.frame['role'] == 'test').to_numpy()
    rows, masks = [], []
    timer = Timer()
    for rate in rates:
        mask = payload_mask(prep.frame['flow_uid'].to_numpy(), prep.frame['has_payload'].to_numpy(), rate,
                            seed=int(cfg.seed)) & test
        with timer(f'rate_{rate}'):
            graphs = build_client_graphs(work, prep, emb, test_mask=mask)
            out = evaluate_test(model, state, graphs, prep.names, device=device)
        tag_r = f'rate{int(round(rate * 100))}'
        rundir.write_evaluation(out['report'], prep.names, prefix=tag_r, extra={'mask_rate': rate})
        out['predictions'].to_parquet(rundir.file(f'{tag_r}_predictions.parquet'), index=False)
        s = out['report']['summary']
        available = int(((prep.frame['has_payload'] == 1).to_numpy() & test).sum())
        rows.append({'mask_rate': rate, 'masked_flows': int(mask.sum()), 'payload_flows_available': available,
                     **{k: s[k] for k in ('macro_f1', 'balanced_accuracy', 'accuracy', 'worst_client_macro_f1')}})
        masks.append(pd.DataFrame({'flow_uid': prep.frame['flow_uid'].to_numpy()[mask], 'mask_rate': rate}))
        rundir.log_round({'phase': 'sweep', 'round': int(round(rate * 100)), 'scope': 'global',
                          'macro_f1': s['macro_f1'], 'balanced_accuracy': s['balanced_accuracy']})
        logger.info(f"mask {rate:.0%}: macro-F1 {s['macro_f1']:.4f} ({int(mask.sum())} flows masked)")

    sweep = pd.DataFrame(rows)
    sweep.to_csv(rundir.file('missing_sweep.csv'), index=False)
    pd.concat(masks).to_parquet(rundir.file('mask_ids.parquet'), index=False)
    base = sweep.iloc[0]
    # the run's headline row is the unmasked rate; per-rate files carry the rest
    pd.DataFrame([{'run_id': rundir.manifest['run_id'], 'status': 'ok', **base.to_dict()}]).to_csv(
        rundir.file('test_metrics.csv'), index=False)
    rundir.write_resources([{'name': f'time_{k}_s', 'value': round(v, 3), 'unit': 's'} for k, v in timer.totals.items()])
    return {'sweep': sweep}
