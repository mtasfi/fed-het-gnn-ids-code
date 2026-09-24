"""E0 data-readiness gate (CPU only).

Steps (run individually or `all`, in this order):
    extract   PCAPs -> flows + raw payload segments           (work/<ds>/flows/)
    labels    label CSVs -> normalised label rows             (work/<ds>/labels.parquet)
    offset    estimate the PCAP/label clock offset            (e0/clock_offset.json)
    match     flows <-> label rows                            (matched.parquet, match_report.md, match_failures.csv)
    payload   label map + has_payload rule + audit            (flows_labeled.parquet, dataset_audit.md, ...)
    split     blocks, train/test, Dirichlet partitions, templates  (splits/, e0/split_report.json, ...)
    kappa     shares_host density for candidate kappa         (e0/kappa_stats.csv)

Examples
    python experiments/run_e0.py all --dataset toniot
    python experiments/run_e0.py extract --dataset toniot --set dataset.source.kind=kaggle \
        --set dataset.source.root=/kaggle/input/ton-iot-pcaps
    python experiments/run_e0.py extract --dataset toniot --max-files 2 --max-packets 200000   # smoke
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fhf.common.config import dump_yaml, load_config, parse_set_args
from fhf.common.utils import setup_logging
from fhf.data import audit_payloads as audit
from fhf.data.flow_extractor import extract_all, load_flows
from fhf.data.label_sources import load_labels
from fhf.data.make_splits import make_split, save_report
from fhf.data.partition_clients import alpha_tag, partition
from fhf.data.pcap_match import estimate_clock_offset, flow_keys, match_flows, match_summary
from fhf.data.sources import resolve_files
from fhf.data.store import Store
from fhf.phase2.build_heterograph import shares_host_stats

logger = logging.getLogger('run_e0')
STEPS = ['extract', 'labels', 'offset', 'match', 'payload', 'split', 'kappa']


def step_extract(cfg, store, args):
    files = resolve_files(cfg)
    pcaps = files.pcaps
    if args.files_regex:
        import re
        pcaps = [p for p in pcaps if re.search(args.files_regex, p)]
    if args.max_files:
        pcaps = pcaps[:args.max_files]
    if not pcaps:
        raise FileNotFoundError(f"No capture files under {files.root} matching {cfg.dataset.source.pcap_globs}")
    stats = extract_all(cfg, pcaps, files.root, store.flows_dir, workers=int(cfg.flow_extractor.workers),
                        max_packets_per_unit=args.max_packets)
    stats.to_csv(store.e0('extract_stats.csv'), index=False)
    logger.info(f"Extraction done:\n{stats.to_string()}")


def step_labels(cfg, store, args):
    files = resolve_files(cfg)
    labels = load_labels(cfg, files.label_files)
    labels.to_parquet(store.labels, index=False)
    logger.info(f"{len(labels):,} label rows -> {store.labels}")


def _flows_for_matching(store):
    return load_flows(store.flows_dir, columns=['flow_uid', 'capture_id', 'first_ts', 'proto', 'src_ip', 'src_port',
                                                'dst_ip', 'dst_port_id'])


def step_offset(cfg, store, args):
    flows = _flows_for_matching(store)
    flows['key'] = flow_keys(flows)
    labels = pd.read_parquet(store.labels, columns=['ts', 'key'])
    result = estimate_clock_offset(flows, labels, float(cfg.matching.max_offset_search_s))
    with open(store.e0('clock_offset.json'), 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Clock offset estimate: {result['offset_s']} s (peak share {result['peak_share']:.2%}, "
                f"{result['pairs']} pairs). Top: {result['top'][:5]}")
    logger.info("If the peak is clear and not 0, set matching.clock_offset_s to it before `match`.")


def step_match(cfg, store, args):
    flows = load_flows(store.flows_dir)
    labels = pd.read_parquet(store.labels)
    matched, failures = match_flows(flows, labels, float(cfg.matching.time_tolerance_s),
                                    float(cfg.matching.clock_offset_s))
    matched.to_parquet(store.matched, index=False)
    failures.to_csv(store.e0('match_failures.csv'), index=False)

    summary = match_summary(matched)
    ok = matched[matched['match_status'].isin(['matched', 'matched_consistent'])]
    # per-class label-row coverage: share of each class's label rows claimed by >= 1 flow
    claimed = set(ok['label_row_id'])
    labels['claimed'] = labels['row_id'].isin(claimed)
    per_class = labels.groupby('raw_label')['claimed'].agg(['size', 'mean']).rename(
        columns={'size': 'label_rows', 'mean': 'row_match_rate'})
    per_class_flows = ok.groupby('raw_label').agg(flows=('flow_uid', 'size'),
                                                  row_shared=('row_shared', 'sum'),
                                                  consistent_multi=('match_status', lambda s: (s == 'matched_consistent').sum()))
    per_class = per_class.join(per_class_flows, how='left').fillna(0)
    per_class.to_csv(store.e0('match_by_class.csv'))

    # trace sample: random matched flows with the label row they point to (local file)
    sample = ok.sample(min(50, len(ok)), random_state=0)[
        ['flow_uid', 'capture_id', 'capture_file', 'first_ts', 'proto', 'src_ip', 'src_port', 'dst_ip',
         'dst_port_id', 'label_row_id', 'raw_label', 'match_dt_s']]
    sample.to_csv(store.e0('match_trace_sample.csv'), index=False)

    lines = ['# Flow <-> PCAP match report', '',
             f"tolerance: {cfg.matching.time_tolerance_s} s, clock offset: {cfg.matching.clock_offset_s} s", '']
    lines += [f"- {k}: {v}" for k, v in summary.items()]
    lines += ['', '## Per raw label', '', per_class.reset_index().to_markdown(index=False), '',
              'Trace sample (50 random matched flows -> label rows): match_trace_sample.csv']
    with open(store.e0('match_report.md'), 'w') as f:
        f.write('\n'.join(lines))
    with open(store.e0('match_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Match: {summary['match_rate']:.2%} matched, {summary['ambiguous_rate']:.2%} ambiguous")


def step_payload(cfg, store, args):
    matched = pd.read_parquet(store.matched)
    labels = pd.read_parquet(store.labels, columns=['row_id', 'raw_label'])
    audit.label_mapping_report(matched, labels, cfg, store.e0('label_mapping.yaml'))
    labeled = audit.apply_payload_rule(matched, cfg)
    labeled.to_parquet(store.labeled, index=False)

    avail = audit.payload_availability(labeled)
    avail.to_csv(store.e0('payload_availability_by_class.csv'), index=False)
    audit.reason_counts(labeled).to_csv(store.e0('has_payload_reason_counts.csv'), index=False)

    extract_stats = {}
    if os.path.exists(store.e0('extract_stats.csv')):
        es = pd.read_csv(store.e0('extract_stats.csv'))
        extract_stats = {c: int(es[c].sum()) for c in es.columns if c not in ('capture_id', 'skipped')
                         and pd.api.types.is_numeric_dtype(es[c])}
    match = json.load(open(store.e0('match_summary.json'))) if os.path.exists(store.e0('match_summary.json')) else {}

    dep = cfg.dataset.get('payload_dependent_classes') or []
    dep_rows = avail[avail['label'].isin(dep)]
    usable = len(dep_rows) > 0 and (dep_rows['has_payload_rate'] > 0.2).all()
    decision = (f"Payload-dependent classes {dep}: has_payload rates "
                f"{dict(zip(dep_rows['label'], dep_rows['has_payload_rate'].round(3)))}. "
                f"Automatic check (every payload-dependent class > 20% has_payload): "
                f"{'PASS' if usable else 'FAIL'}. The final payload-usable decision is the researcher's; "
                f"record it here before GPU work.")
    audit.write_audit_md(store.e0('dataset_audit.md'), cfg, extract_stats, match, avail, decision)
    logger.info(f"{len(labeled):,} labelled flows, has_payload rate {labeled['has_payload'].mean():.2%}")
    logger.info(decision)


def step_split(cfg, store, args):
    labeled = pd.read_parquet(store.labeled, columns=['flow_uid', 'capture_id', 'first_ts', 'label', 'payload_texts'])
    split, report = make_split(labeled, cfg, seed=int(cfg.split.get('seed', 0)))
    split.to_parquet(store.split, index=False)
    save_report(report, store.e0('split_report.json'))

    alphas = list(cfg.partition.alpha_candidates) + ['iid']
    for alpha in alphas:
        part, info = partition(split, int(cfg.partition.num_clients), alpha, float(cfg.split.val_fraction),
                               seed=int(cfg.partition.get('seed', 0)))
        part.to_parquet(store.partition(alpha), index=False)
        info['distribution'].to_csv(store.e0(f'client_class_distribution_alpha={alpha_tag(alpha)}.csv'), index=False)
        save_report(info['report'], store.e0(f'partition_report_alpha={alpha_tag(alpha)}.json'))

    overlap = audit.template_overlap(labeled, split)
    overlap.to_csv(store.e0('template_overlap_report.csv'), index=False)
    with open(store.e0('template_normalization_rule.txt'), 'w') as f:
        f.write(audit.TEMPLATE_RULE + '\n')

    leakage = {
        'train_test_block_overlap': report['block_overlap_train_test'],
        'flows_in_more_than_one_split': int(split['flow_uid'].duplicated().sum()),
        'split_ok': report['split_ok'],
        'problems': report['problems'],
    }
    save_report(leakage, store.e0('leakage_log.json'))
    logger.info(f"Split: {report['flows_selected']:,} flows, test fraction {report['test_fraction_realised']:.2%}, "
                f"ok={report['split_ok']}")


def step_kappa(cfg, store, args):
    df = pd.read_parquet(store.labeled, columns=['flow_uid', 'src_ip', 'first_ts'])
    alpha = cfg.partition.alpha if cfg.partition.alpha is not None else cfg.partition.alpha_candidates[0]
    part = pd.read_parquet(store.partition(alpha))
    df = part[part.role == 'train'].merge(df, on='flow_uid')
    kappas = [2, 5, 10, 20, 50, None]
    rows = []
    for client, g in df.groupby('client'):
        stats = shares_host_stats(g, float(cfg.graph.shares_host.window_s), kappas)
        stats.insert(0, 'client', client)
        rows.append(stats)
    out = pd.concat(rows)
    out.to_csv(store.e0('kappa_stats.csv'), index=False)
    logger.info(f"shares_host density per client (alpha={alpha}):\n"
                f"{out.groupby('kappa', sort=False)[['mean_degree', 'saturated_rate', 'isolated_rate']].mean()}")
    logger.info("Pick the smallest kappa before saturation climbs steeply; set graph.shares_host.kappa.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=STEPS + ['all'])
    parser.add_argument('--dataset', default='toniot')
    parser.add_argument('--set', action='append', default=[], help='config override key=value (repeatable)')
    parser.add_argument('--max-files', type=int, default=None, help='extract: only the first N capture files')
    parser.add_argument('--files-regex', default=None, help='extract: only capture files matching this regex')
    parser.add_argument('--max-packets', type=int, default=None, help='extract: stop each unit after N packets')
    args = parser.parse_args()

    cfg = load_config(args.dataset, overrides=parse_set_args(args.set))
    store = Store(cfg)
    setup_logging('INFO', store.e0('e0.log'))
    dump_yaml(cfg.to_dict(), store.e0('e0_config.yaml'))

    steps = STEPS if args.step == 'all' else [args.step]
    for step in steps:
        logger.info(f"==== E0 step: {step} ====")
        globals()[f'step_{step}'](cfg, store, args)


if __name__ == '__main__':
    main()
