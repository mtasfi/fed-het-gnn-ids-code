import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fhf.common.config import load_config
from fhf.common.metrics import classification_report
from fhf.common.utils import class_weights
from fhf.data.federated_scaler import FederatedScaler, client_stats, combine, pre_transform
from fhf.data.features import feature_columns
from fhf.data.flow_extractor import extract_all, load_flows
from fhf.data.label_sources import canonical_key
from fhf.data import match_checks
from fhf.data.make_splits import make_split
from fhf.data.partition_clients import partition
from fhf.data.payload_rules import decide, to_text
from fhf.data.pcap_match import match_flows
from fhf.phase2.build_heterograph import payload_mask, shares_host_edges
from synthetic import http_session, udp_packet, write_pcap


@pytest.fixture
def cfg():
    return load_config('toniot')


# ------------------------------------------------------------------ has_payload rule
def test_rule_is_deterministic_and_case_preserving(cfg):
    seg = [b"GET /?q=<ScRiPt>alert(1)</ScRiPt> HTTP/1.1\r\nHost: x\r\n\r\n"]
    a, b = decide(seg, cfg.payload), decide(list(seg), cfg.payload)
    assert a == b
    assert a.has_payload == 1 and a.reason == 'ok'
    assert '<ScRiPt>' in a.texts[0]


def test_rule_reason_codes(cfg):
    rule = cfg.payload
    assert decide([], rule).reason == 'no_payload'
    assert decide([b''], rule).reason == 'no_payload'
    tls = b'\x16\x03\x01\x02\x00' + bytes(range(256)) * 2
    assert decide([tls], rule).reason == 'tls_record'
    rng = np.random.default_rng(0)
    assert decide([rng.integers(0, 256, 1024, dtype=np.uint8).tobytes()], rule).reason == 'high_entropy'
    assert decide([b'\x00\x01\x02\x03abc\x00\x00\x00\x00\x00'], rule).reason == 'unreadable_binary'
    assert decide([b'ab'], rule).reason == 'too_short'


def test_rule_keeps_only_readable_segments_in_order(cfg):
    d = decide([b'\x16\x03\x01\x00\x10' + b'\x00' * 40, b"USER admin\r\n", b"PASS 123456\r\n", b"extra"], cfg.payload)
    assert d.has_payload == 1
    assert d.kept_indices == [1, 2]          # j_max = 3: the 4th segment is never considered
    assert d.texts == ['USER admin\r\n', 'PASS 123456\r\n']


def test_placeholder_replaces_runs(cfg):
    assert to_text(b'ab\x00\x01\x02cd\xffef', '<NP>') == 'ab<NP>cd<NP>ef'


# ------------------------------------------------------------------ extractor + matching
def test_extractor_and_matching(tmp_path, cfg):
    cap_dir = tmp_path / 'raw' / 'cap1'
    cap_dir.mkdir(parents=True)
    pkts = http_session('10.0.0.1', 40000, '10.0.0.9', 1000.0, b"GET /index.php?id=1' OR 1=1-- HTTP/1.1\r\n\r\n")
    pkts += http_session('10.0.0.2', 40001, '10.0.0.9', 1001.0, b"GET / HTTP/1.1\r\n\r\n")
    pkts += [(1002.0, udp_packet('10.0.0.3', 5353, '10.0.0.9', 53, b'\x12\x34\x01\x00'))]
    # split one capture across two rotated files to exercise per_directory continuity
    write_pcap(str(cap_dir / 'part1.pcap'), pkts[:5])
    write_pcap(str(cap_dir / 'part2.pcap'), pkts[5:])

    out = tmp_path / 'flows'
    extract_all(cfg, [str(cap_dir / 'part1.pcap'), str(cap_dir / 'part2.pcap')], str(tmp_path / 'raw'), str(out))
    flows = load_flows(str(out))
    assert len(flows) == 3
    sqli = flows[flows.src_port == 40000].iloc[0]
    assert sqli['src_ip'] == '10.0.0.1' and sqli['dst_port_id'] == 80   # initiator = client
    assert sqli['fwd_pkts'] == 5 and sqli['bwd_pkts'] == 3
    assert sqli['n_segments_stored'] == 2                              # retransmission dropped
    assert sqli['syn_cnt'] == 2 and sqli['fin_cnt'] == 2

    labels = pd.DataFrame({
        'row_id': ['r1', 'r2', 'r3'], 'ts': [1000.0, 1001.0, 1002.0], 'proto': [6, 6, 17],
        'src_ip': ['10.0.0.1', '10.0.0.9', '10.0.0.3'], 'src_port': [40000, 80, 5353],
        'dst_ip': ['10.0.0.9', '10.0.0.2', '10.0.0.9'], 'dst_port': [80, 40001, 53],
        'raw_label': ['injection', 'normal', 'normal'],
    })
    labels['key'] = canonical_key(labels.proto, labels.src_ip, labels.src_port, labels.dst_ip, labels.dst_port)
    matched, failures = match_flows(flows, labels, tolerance_s=2.0, offset_s=0.0)
    assert (matched.match_status == 'matched').all() and failures.empty
    assert matched.set_index('src_port').loc[40000, 'raw_label'] == 'injection'

    # a label row with a conflicting duplicate inside the tolerance is ambiguous
    dup = labels.iloc[[0]].assign(row_id='r1b', ts=1000.5, raw_label='xss')
    matched2, _ = match_flows(flows, pd.concat([labels, dup]), tolerance_s=2.0, offset_s=0.0)
    assert matched2.set_index('src_port').loc[40000, 'match_status'] == 'ambiguous_conflict'


def _matched_frame():
    """Five flows: two SQLi (one labelled injection, one labelled normal), one XSS, one
    plain GET, one unmatched UDP flow whose 5-tuple exists in the labels at another time."""
    fwd = lambda b: ([b, b'HTTP/1.1 200 OK\r\n\r\n'], [0, 1])
    segs = [fwd(b"GET /?id=1%27+UNION+SELECT+user,pass+FROM+t HTTP/1.1\r\n"),
            fwd(b"GET /?id=1' OR 1=1-- HTTP/1.1\r\n"),
            fwd(b"GET /?q=%3Cscript%3Ealert(1)%3C/script%3E HTTP/1.1\r\n"),
            fwd(b"GET /index.html HTTP/1.1\r\n"),
            ([b'\x12\x34\x01\x00'], [0])]
    return pd.DataFrame({
        'flow_uid': list('abcde'), 'capture_id': 'c', 'first_ts': [100.0, 101.0, 102.0, 103.0, 104.0],
        'last_ts': [100.5, 101.5, 102.5, 103.5, 104.0], 'proto': [6, 6, 6, 6, 17],
        'dst_port_id': [80, 80, 80, 80, 53], 'duration': [0.5, 0.5, 0.5, 0.5, 0.0],
        'end_reason': ['fin'] * 4 + ['idle'], 'key': np.array([1, 2, 3, 4, 5], dtype=np.uint64),
        'match_status': ['matched'] * 4 + ['no_label_row'],
        'label_row_id': ['r1', 'r2', 'r3', 'r4', None],
        'raw_label': ['injection', 'normal', 'xss', 'normal', None],
        'payload_segments': [s for s, _ in segs], 'payload_dirs': [d for _, d in segs],
    })


def test_match_checks_signatures_coverage_breakdown_gate():
    matched = _matched_frame()
    labels = pd.DataFrame({'row_id': ['r1', 'r2', 'r3', 'r4', 'r5', 'r_old'],
                           'ts': [100.0, 101.0, 102.0, 103.0, 103.2, 5.0],
                           'key': np.array([1, 2, 3, 4, 9, 5], dtype=np.uint64),
                           'raw_label': ['injection', 'normal', 'xss', 'normal', 'normal', 'normal']})

    ct = match_checks.signature_crosstab(matched, '<NP>').set_index('signature')
    assert ct.loc['sqli', 'flows_hit'] == 2 and ct.loc['sqli', 'matched'] == 2
    assert ct.loc['sqli', 'agreement'] == 0.5                 # one SQLi flow got `normal`
    assert ct.loc['xss', 'agreement'] == 1.0
    assert ct.loc['cmd_injection', 'flows_hit'] == 0 and np.isnan(ct.loc['cmd_injection', 'agreement'])
    # the response (bwd) is never searched
    assert not match_checks.signature_hits(matched.iloc[[3]], '<NP>').any(axis=None)

    cov = match_checks.class_coverage(matched, labels, offset_s=0.0, tolerance_s=2.0).set_index('raw_label')
    assert cov.loc['normal', 'rows_in_span'] == 3            # r_old (ts=5) is outside the capture span
    assert cov.loc['normal', 'row_coverage'] == pytest.approx(2 / 3)

    br = match_checks.failure_breakdown(matched, labels)
    assert len(br) == 1 and br.iloc[0]['proto'] == 'udp' and bool(br.iloc[0]['key_in_labels'])
    assert br.iloc[0]['share_of_all_flows'] == pytest.approx(0.2)

    summary = {'match_rate': 0.8, 'ambiguous_rate': 0.0}
    thresholds = {'min_match_rate': 0.8, 'max_ambiguous_rate': 0.05, 'min_class_row_coverage': 0.5,
                  'min_rows_per_class': 1, 'min_signature_agreement': 0.9, 'min_signature_flows': 1}
    result = match_checks.gate(summary, cov.reset_index(), ct.reset_index(), thresholds)
    failed = {c['check'] for c in result['checks'] if c['ok'] is False}
    assert not result['pass'] and failed == {'signature_agreement[sqli]'}
    # null thresholds skip checks; nothing judged is not a pass
    assert match_checks.gate(summary, cov.reset_index(), ct.reset_index(), {})['pass'] is False


# ------------------------------------------------------------------ splits / partition
def _fake_labeled(n_blocks=60, per_block=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    classes = ['normal', 'xss', 'injection', 'password', 'ddos']
    for b in range(n_blocks):
        cls = classes[b % len(classes)]
        for i in range(per_block):
            label = cls if rng.random() < 0.8 else 'normal'
            rows.append({'flow_uid': f'b{b}f{i}', 'capture_id': 'c', 'first_ts': b * 300 + i, 'label': label})
    return pd.DataFrame(rows)


def test_split_blocks_never_cross_and_partition_is_stable(cfg):
    cfg.split.target_flows = 2000
    cfg.split.rare_class_floor = 100
    cfg.split.min_test_per_class = 5
    labeled = _fake_labeled()
    split, report = make_split(labeled, cfg, seed=0)
    assert report['block_overlap_train_test'] == 0
    assert split.groupby('block_id')['split'].nunique().max() == 1
    assert report['split_ok'], report['problems']

    p1, _ = partition(split, 5, 0.5, 0.1, seed=0)
    p2, _ = partition(split, 5, 0.5, 0.1, seed=0)
    pd.testing.assert_frame_equal(p1, p2)
    assert set(p1.role) == {'train', 'val', 'test'}
    assert p1.groupby('block_id')['client'].nunique().max() == 1
    assert p1.groupby('block_id')['role'].nunique().max() == 1
    assert sorted(p1.client.unique()) == [0, 1, 2, 3, 4]


def test_split_block_count_floor_spreads_a_one_block_class(cfg):
    """A class whose flows are all in one huge block plus a few small ones must end up
    with >= min_blocks_per_class blocks, so it has both test and training flows."""
    rows = []
    for b in range(40):                                   # background: 40 normal blocks
        rows += [('normal', f'n{b}', b * 60.0 + i * 0.1) for i in range(50)]
    rows += [('scanning', 'big', 5000.0 + i * 0.001) for i in range(5000)]        # one huge block
    for b in range(6):                                    # six small scanning blocks elsewhere
        rows += [('scanning', f's{b}', 9000.0 + b * 60.0 + i * 0.1) for i in range(30)]
    df = pd.DataFrame(rows, columns=['label', 'capture_id', 'first_ts'])
    df['flow_uid'] = [f'u{i}' for i in range(len(df))]
    cfg.split.block_seconds = 60
    cfg.split.target_flows = 500
    cfg.split.rare_class_floor = 300
    cfg.split.min_test_per_class = 5

    cfg.split.min_blocks_per_class = 4
    split, new = make_split(df, cfg, seed=0)
    assert new['split_ok'], new['problems']
    assert split[split.label == 'scanning']['block_id'].nunique() >= 4
    assert split.groupby('block_id')['split'].nunique().max() == 1


# ------------------------------------------------------------------ scaler / weights / metrics
def test_federated_scaler_equals_central():
    rng = np.random.default_rng(0)
    parts = [rng.lognormal(size=(n, 4)) for n in (10, 50, 200)]
    g = combine([client_stats(pre_transform(p, True)) for p in parts])
    central = pre_transform(np.concatenate(parts), True)
    np.testing.assert_allclose(g.mean, central.mean(0), rtol=1e-10)
    np.testing.assert_allclose(g.m2 / g.n, central.var(0), rtol=1e-10)
    s = FederatedScaler(True).fit_clients(parts)
    np.testing.assert_allclose(s.transform(np.concatenate(parts)).mean(0), 0, atol=1e-5)


def test_class_weights_absent_class_is_zero_not_inf():
    w = class_weights([0, 0, 0, 1], num_classes=3)
    assert w[2] == 0 and np.isfinite(w).all() and w[1] > w[0]


def test_metrics_keep_absent_class_visible():
    rep = classification_report([0, 0, 1, 1], [0, 1, 1, 1], ['a', 'b', 'c'], client_ids=[0, 0, 1, 1])
    assert rep['summary']['n_classes_scored'] == 2
    assert rep['per_class'].set_index('label').loc['c', 'support'] == 0
    assert rep['summary']['worst_client_macro_f1'] <= rep['summary']['macro_f1'] + 1


def test_no_identity_column_is_a_feature(cfg):
    cols = feature_columns(cfg)
    assert not {'src_ip', 'dst_ip', 'first_ts', 'flow_uid', 'src_port', 'dst_port_id'} & set(cols)


# ------------------------------------------------------------------ graph pieces
def test_shares_host_cap_and_window():
    src = ['a'] * 6 + ['b'] * 3
    ts = [0, 1, 2, 3, 100, 101, 0, 1, 2]
    e = shares_host_edges(src, ts, window_s=60, kappa=2)
    deg = np.bincount(e[0], minlength=9)
    assert deg.max() <= 2
    pairs = set(map(tuple, e.T))
    assert (0, 4) not in pairs and (3, 4) not in pairs          # outside the window
    assert all(src[u] == src[v] for u, v in pairs)                # same source only


def test_mask_is_nested_and_deterministic():
    uids = [f'f{i}' for i in range(1000)]
    hp = np.ones(1000)
    m25, m50 = payload_mask(uids, hp, 0.25), payload_mask(uids, hp, 0.5)
    assert (m25 <= m50).all() and 0.2 < m25.mean() < 0.3
    assert (payload_mask(uids, hp, 0.25) == m25).all()
