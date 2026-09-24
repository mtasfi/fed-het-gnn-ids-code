"""PCAP -> bidirectional flows with statistics and the first J_max payload segments (E0).

The flow table and the payload segments are both produced here, from the raw
capture, so flow statistics and payloads come from the same packets
(NF-ToN-IoT flows cannot be used: they have no payload and no absolute start time).

Flow definition (locked in configs/base.yaml `flow_extractor`):
  * key: canonical bidirectional 5-tuple (proto, endpoint A, endpoint B), A <= B;
  * forward direction = initiator = source of the first packet; a flow first seen
    on a SYN-ACK is flipped so the client is still the initiator;
  * a flow ends on: idle_timeout, active_timeout, or (TCP) RST / FIN from both
    sides once the next SYN on the same key arrives or it idles out;
  * payload segment = application bytes of one packet, in capture order, after
    dropping TCP retransmissions (same direction + sequence number + length).
    The first J_max non-empty segments are kept (raw bytes, capped at
    max_bytes_per_segment). The has_payload decision is NOT made here: raw bytes
    are stored so the rule (payload_rules.py) can change without re-extraction.

Raw bytes stay in local Parquet under work/; nothing here is ever sent anywhere.
"""

import gzip
import hashlib
import logging
import math
import os
import re
import socket
import struct
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Dict, Iterator, List, Optional, Tuple

import dpkt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

TCP, UDP, ICMP, ICMP6 = 6, 17, 1, 58
FIN, SYN, RST, PSH, ACK, URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20
CLOSED_GRACE_S = 5.0

# Model features produced by the extractor. Identity columns (IPs, ports as
# identifiers, timestamps, capture ids) are deliberately not in this list.
FEATURE_COLUMNS = [
    'duration', 'fwd_pkts', 'bwd_pkts', 'fwd_bytes', 'bwd_bytes',
    'fwd_payload_bytes', 'bwd_payload_bytes', 'fwd_payload_pkts', 'bwd_payload_pkts',
    'fwd_len_mean', 'fwd_len_std', 'fwd_len_min', 'fwd_len_max',
    'bwd_len_mean', 'bwd_len_std', 'bwd_len_min', 'bwd_len_max',
    'iat_mean', 'iat_std', 'iat_min', 'iat_max',
    'fwd_iat_mean', 'fwd_iat_std', 'bwd_iat_mean', 'bwd_iat_std',
    'bytes_per_s', 'pkts_per_s', 'down_up_pkt_ratio', 'down_up_byte_ratio', 'avg_pkt_len',
    'syn_cnt', 'fin_cnt', 'rst_cnt', 'psh_cnt', 'ack_cnt', 'urg_cnt',
    'syn_ratio', 'fin_ratio', 'rst_ratio', 'psh_ratio', 'ack_ratio',
    'fwd_init_win', 'bwd_init_win', 'fwd_hdr_bytes', 'bwd_hdr_bytes',
    'proto_tcp', 'proto_udp', 'proto_icmp', 'dst_port', 'dst_port_wellknown',
]

IDENTITY_COLUMNS = [
    'flow_uid', 'capture_id', 'capture_file', 'first_ts', 'last_ts',
    'src_ip', 'dst_ip', 'src_port', 'dst_port_id', 'proto', 'ip_version', 'end_reason',
]


class _Dir:
    """Running statistics of one direction of a flow."""
    __slots__ = ('pkts', 'bytes', 'payload_bytes', 'payload_pkts', 'len_sum', 'len_sq', 'len_min', 'len_max',
                 'last_ts', 'iat_sum', 'iat_sq', 'iat_n', 'init_win', 'hdr_bytes', 'fin')

    def __init__(self):
        self.pkts = self.bytes = self.payload_bytes = self.payload_pkts = 0
        self.len_sum = self.len_sq = 0.0
        self.len_min = math.inf
        self.len_max = 0
        self.last_ts = None
        self.iat_sum = self.iat_sq = 0.0
        self.iat_n = 0
        self.init_win = -1
        self.hdr_bytes = 0
        self.fin = False

    def add(self, ts, ip_len, payload_len, hdr_len):
        self.pkts += 1
        self.bytes += ip_len
        self.len_sum += ip_len
        self.len_sq += ip_len * ip_len
        self.len_min = min(self.len_min, ip_len)
        self.len_max = max(self.len_max, ip_len)
        self.hdr_bytes += hdr_len
        if payload_len:
            self.payload_bytes += payload_len
            self.payload_pkts += 1
        if self.last_ts is not None:
            d = ts - self.last_ts
            self.iat_sum += d
            self.iat_sq += d * d
            self.iat_n += 1
        self.last_ts = ts


class _Flow:
    __slots__ = ('key', 'src', 'sport', 'dst', 'dport', 'proto', 'ipv', 'first_ts', 'last_ts', 'fwd', 'bwd',
                 'iat_sum', 'iat_sq', 'iat_min', 'iat_max', 'iat_n', 'flags', 'segments', 'seg_dirs',
                 'seen_seq', 'closed', 'end_reason')

    def __init__(self, key, src, sport, dst, dport, proto, ipv, ts):
        self.key = key
        self.src, self.sport, self.dst, self.dport = src, sport, dst, dport
        self.proto, self.ipv = proto, ipv
        self.first_ts = self.last_ts = ts
        self.fwd, self.bwd = _Dir(), _Dir()
        self.iat_sum = self.iat_sq = 0.0
        self.iat_min, self.iat_max, self.iat_n = math.inf, 0.0, 0
        self.flags = [0, 0, 0, 0, 0, 0]  # syn fin rst psh ack urg
        self.segments: List[bytes] = []
        self.seg_dirs: List[int] = []
        self.seen_seq = set()
        self.closed = False
        self.end_reason = None


@dataclass
class ExtractStats:
    packets: int = 0
    non_ip: int = 0
    fragments: int = 0
    parse_errors: int = 0
    other_proto: int = 0
    flows: int = 0
    retransmit_segments: int = 0
    files: List[str] = field(default_factory=list)

    def merge(self, other: 'ExtractStats'):
        for name in ('packets', 'non_ip', 'fragments', 'parse_errors', 'other_proto', 'flows', 'retransmit_segments'):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.files.extend(other.files)


# --------------------------------------------------------------------------- reading
def _open(path: str):
    with open(path, 'rb') as f:
        magic = f.read(2)
    return gzip.open(path, 'rb') if magic == b'\x1f\x8b' else open(path, 'rb')


def iter_packets(path: str) -> Iterator[Tuple[float, bytes, int]]:
    """(timestamp, raw frame, datalink type) for pcap and pcapng, gzip or not."""
    fh = _open(path)
    try:
        head = fh.read(4)
        fh.seek(0)
        if head == b'\x0a\x0d\x0d\x0a':
            reader = dpkt.pcapng.Reader(fh)
        else:
            reader = dpkt.pcap.Reader(fh)
        dlt = reader.datalink()
        for ts, buf in reader:
            yield float(ts), buf, dlt
    finally:
        fh.close()


def first_timestamp(path: str) -> float:
    try:
        for ts, _, _ in iter_packets(path):
            return ts
    except Exception:
        pass
    return math.inf


def _ip_layer(buf: bytes, dlt: int):
    if dlt == dpkt.pcap.DLT_EN10MB:
        eth = dpkt.ethernet.Ethernet(buf)
        return eth.data
    if dlt == 113:  # Linux cooked capture
        return dpkt.sll.SLL(buf).data
    if dlt == 276:  # Linux cooked v2
        return dpkt.sll2.SLL2(buf).data
    if dlt in (dpkt.pcap.DLT_NULL, dpkt.pcap.DLT_LOOP):
        return dpkt.loopback.Loopback(buf).data
    if dlt in (101, 12, 14, 228, 229):  # raw IP
        version = buf[0] >> 4 if buf else 0
        return dpkt.ip.IP(buf) if version == 4 else dpkt.ip6.IP6(buf) if version == 6 else None
    raise ValueError(f"Unsupported datalink type {dlt}")


# --------------------------------------------------------------------------- flow table
class FlowTable:
    def __init__(self, cfg, capture_id: str, sink):
        fe = cfg['flow_extractor']
        pl = cfg['payload']
        self.idle = float(fe['idle_timeout_s'])
        self.active = float(fe['active_timeout_s'])
        self.close_on_fin_rst = bool(fe['close_on_fin_rst'])
        self.include_ipv6 = bool(fe['include_ipv6'])
        self.j_max = int(pl['j_max'])
        self.fwd_only = pl['directions'] == 'forward'
        self.max_seg = int(pl['max_bytes_per_segment'])
        self.capture_id = capture_id
        self.current_file = None
        self.flows: Dict[tuple, _Flow] = {}
        self.sink = sink
        self.stats = ExtractStats()
        self._since_sweep = 0
        self._now = 0.0

    def set_file(self, path: str):
        self.current_file = path
        self.stats.files.append(path)

    def process(self, ts: float, buf: bytes, dlt: int):
        self.stats.packets += 1
        self._now = max(self._now, ts)
        try:
            ip = _ip_layer(buf, dlt)
        except Exception:
            self.stats.parse_errors += 1
            return
        if isinstance(ip, dpkt.ip.IP):
            ipv = 4
            if ip.offset:  # non-first fragment carries no L4 header
                self.stats.fragments += 1
                return
            ip_len, proto = ip.len, ip.p
            hdr = ip.hl * 4
        elif isinstance(ip, dpkt.ip6.IP6) and self.include_ipv6:
            ipv = 6
            ip_len, proto = ip.plen + 40, ip.nxt
            hdr = 40
        else:
            self.stats.non_ip += 1
            return

        l4 = ip.data
        flags, seq, win = 0, None, -1
        if proto == TCP and isinstance(l4, dpkt.tcp.TCP):
            sport, dport = l4.sport, l4.dport
            flags, seq, win = l4.flags, l4.seq, l4.win
            hdr += l4.off * 4
            payload = bytes(l4.data)
        elif proto == UDP and isinstance(l4, dpkt.udp.UDP):
            sport, dport = l4.sport, l4.dport
            hdr += 8
            payload = bytes(l4.data)
        elif proto in (ICMP, ICMP6):
            sport = dport = 0
            payload = b''
        else:
            self.stats.other_proto += 1
            return

        src, dst = bytes(ip.src), bytes(ip.dst)
        a, b = (src, sport), (dst, dport)
        key = (proto, a, b) if a <= b else (proto, b, a)

        flow = self.flows.get(key)
        if flow is not None:
            new_syn = proto == TCP and flags & SYN and not flags & ACK
            if (ts - flow.last_ts > self.idle) or (ts - flow.first_ts > self.active) or (flow.closed and new_syn):
                flow.end_reason = flow.end_reason or ('closed' if flow.closed else
                                                      'idle' if ts - flow.last_ts > self.idle else 'active')
                self._finalize(flow)
                flow = None
        if flow is None:
            if proto == TCP and flags & SYN and flags & ACK:
                # first packet is the server's SYN-ACK: the client is the destination
                flow = _Flow(key, dst, dport, src, sport, proto, ipv, ts)
            else:
                flow = _Flow(key, src, sport, dst, dport, proto, ipv, ts)
            self.flows[key] = flow

        forward = src == flow.src and sport == flow.sport
        if not forward and src == flow.dst and sport == flow.dport and flow.src == flow.dst and flow.sport == flow.dport:
            forward = True  # degenerate self-flow
        d = flow.fwd if forward else flow.bwd

        if flow.fwd.pkts + flow.bwd.pkts:
            gap = ts - flow.last_ts
            flow.iat_sum += gap
            flow.iat_sq += gap * gap
            flow.iat_min = min(flow.iat_min, gap)
            flow.iat_max = max(flow.iat_max, gap)
            flow.iat_n += 1
        flow.last_ts = max(flow.last_ts, ts)
        d.add(ts, ip_len, len(payload), hdr)

        if proto == TCP:
            f = flow.flags
            if flags & SYN: f[0] += 1
            if flags & FIN: f[1] += 1; d.fin = True
            if flags & RST: f[2] += 1
            if flags & PSH: f[3] += 1
            if flags & ACK: f[4] += 1
            if flags & URG: f[5] += 1
            if d.init_win < 0:
                d.init_win = win
            if self.close_on_fin_rst and (flags & RST or (flow.fwd.fin and flow.bwd.fin)):
                if not flow.closed:
                    flow.end_reason = 'rst' if flags & RST else 'fin'
                flow.closed = True

        if payload and len(flow.segments) < self.j_max and (forward or not self.fwd_only):
            marker = (forward, seq, len(payload)) if seq is not None else None
            if marker is not None and marker in flow.seen_seq:
                self.stats.retransmit_segments += 1
            else:
                if marker is not None:
                    flow.seen_seq.add(marker)
                flow.segments.append(payload[:self.max_seg])
                flow.seg_dirs.append(0 if forward else 1)

        self._since_sweep += 1
        if self._since_sweep >= 50000:
            self.sweep()

    def sweep(self, final: bool = False):
        self._since_sweep = 0
        now = self._now
        expired = []
        for key, flow in self.flows.items():
            idle_for = now - flow.last_ts
            if final or idle_for > self.idle or (flow.closed and idle_for > CLOSED_GRACE_S):
                flow.end_reason = flow.end_reason or ('eof' if final else 'idle')
                expired.append(flow)
        for flow in expired:
            self._finalize(flow)

    def _finalize(self, flow: _Flow):
        if self.flows.get(flow.key) is flow:
            del self.flows[flow.key]
        self.stats.flows += 1
        self.sink(flow_record(flow, self.capture_id, self.current_file))


def _mean_std(total, sq, n):
    if n <= 0:
        return 0.0, 0.0
    mean = total / n
    var = max(sq / n - mean * mean, 0.0)
    return mean, math.sqrt(var)


def _ip_str(raw: bytes) -> str:
    return socket.inet_ntop(socket.AF_INET if len(raw) == 4 else socket.AF_INET6, raw)


def flow_uid(capture_id: str, flow: _Flow) -> str:
    ident = f"{capture_id}|{flow.proto}|{flow.src.hex()}:{flow.sport}|{flow.dst.hex()}:{flow.dport}|{flow.first_ts:.6f}"
    return hashlib.sha1(ident.encode()).hexdigest()[:20]


def flow_record(flow: _Flow, capture_id: str, capture_file: Optional[str]) -> Dict:
    fwd, bwd = flow.fwd, flow.bwd
    pkts = fwd.pkts + bwd.pkts
    total_bytes = fwd.bytes + bwd.bytes
    duration = max(flow.last_ts - flow.first_ts, 0.0)
    fl_mean, fl_std = _mean_std(fwd.len_sum, fwd.len_sq, fwd.pkts)
    bl_mean, bl_std = _mean_std(bwd.len_sum, bwd.len_sq, bwd.pkts)
    iat_mean, iat_std = _mean_std(flow.iat_sum, flow.iat_sq, flow.iat_n)
    fi_mean, fi_std = _mean_std(fwd.iat_sum, fwd.iat_sq, fwd.iat_n)
    bi_mean, bi_std = _mean_std(bwd.iat_sum, bwd.iat_sq, bwd.iat_n)
    syn, fin, rst, psh, ack, urg = flow.flags
    rate_d = duration if duration > 0 else 1e-6
    return {
        'flow_uid': flow_uid(capture_id, flow),
        'capture_id': capture_id,
        'capture_file': os.path.basename(capture_file) if capture_file else None,
        'first_ts': flow.first_ts,
        'last_ts': flow.last_ts,
        'src_ip': _ip_str(flow.src),
        'dst_ip': _ip_str(flow.dst),
        'src_port': flow.sport,
        'dst_port_id': flow.dport,
        'proto': flow.proto,
        'ip_version': flow.ipv,
        'end_reason': flow.end_reason,

        'duration': duration,
        'fwd_pkts': fwd.pkts, 'bwd_pkts': bwd.pkts,
        'fwd_bytes': fwd.bytes, 'bwd_bytes': bwd.bytes,
        'fwd_payload_bytes': fwd.payload_bytes, 'bwd_payload_bytes': bwd.payload_bytes,
        'fwd_payload_pkts': fwd.payload_pkts, 'bwd_payload_pkts': bwd.payload_pkts,
        'fwd_len_mean': fl_mean, 'fwd_len_std': fl_std,
        'fwd_len_min': fwd.len_min if fwd.pkts else 0, 'fwd_len_max': fwd.len_max,
        'bwd_len_mean': bl_mean, 'bwd_len_std': bl_std,
        'bwd_len_min': bwd.len_min if bwd.pkts else 0, 'bwd_len_max': bwd.len_max,
        'iat_mean': iat_mean, 'iat_std': iat_std,
        'iat_min': flow.iat_min if flow.iat_n else 0.0, 'iat_max': flow.iat_max,
        'fwd_iat_mean': fi_mean, 'fwd_iat_std': fi_std, 'bwd_iat_mean': bi_mean, 'bwd_iat_std': bi_std,
        'bytes_per_s': total_bytes / rate_d, 'pkts_per_s': pkts / rate_d,
        'down_up_pkt_ratio': bwd.pkts / max(fwd.pkts, 1), 'down_up_byte_ratio': bwd.bytes / max(fwd.bytes, 1),
        'avg_pkt_len': total_bytes / max(pkts, 1),
        'syn_cnt': syn, 'fin_cnt': fin, 'rst_cnt': rst, 'psh_cnt': psh, 'ack_cnt': ack, 'urg_cnt': urg,
        'syn_ratio': syn / max(pkts, 1), 'fin_ratio': fin / max(pkts, 1), 'rst_ratio': rst / max(pkts, 1),
        'psh_ratio': psh / max(pkts, 1), 'ack_ratio': ack / max(pkts, 1),
        'fwd_init_win': fwd.init_win, 'bwd_init_win': bwd.init_win,
        'fwd_hdr_bytes': fwd.hdr_bytes, 'bwd_hdr_bytes': bwd.hdr_bytes,
        'proto_tcp': int(flow.proto == TCP), 'proto_udp': int(flow.proto == UDP),
        'proto_icmp': int(flow.proto in (ICMP, ICMP6)),
        'dst_port': flow.dport, 'dst_port_wellknown': int(0 < flow.dport < 1024),

        'payload_segments': list(flow.segments),
        'payload_dirs': list(flow.seg_dirs),
        'n_segments_stored': len(flow.segments),
    }


# --------------------------------------------------------------------------- units of work
def _natural_key(path: str):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', path)]


def plan_units(pcaps: List[str], continuity: str, root: str) -> List[Tuple[str, List[str]]]:
    """[(capture_id, [files in stream order])]. per_directory keeps rotated files of
    one capture in a single stream so flows crossing a file boundary stay whole."""
    if continuity == 'per_file':
        return [(os.path.relpath(p, root), [p]) for p in sorted(pcaps, key=_natural_key)]
    if continuity != 'per_directory':
        raise ValueError(f"flow_extractor.continuity must be per_file or per_directory, got {continuity}")
    groups: Dict[str, List[str]] = {}
    for p in pcaps:
        groups.setdefault(os.path.dirname(p), []).append(p)
    units = []
    for directory, files in sorted(groups.items()):
        files = sorted(files, key=lambda p: (first_timestamp(p), _natural_key(p)))
        units.append((os.path.relpath(directory, root) or '.', files))
    return units


def _unit_tag(capture_id: str) -> str:
    return hashlib.sha1(capture_id.encode()).hexdigest()[:12]


def _record_schema() -> pa.Schema:
    fields = []
    for name in IDENTITY_COLUMNS:
        if name in ('first_ts', 'last_ts'):
            fields.append(pa.field(name, pa.float64()))
        elif name in ('src_port', 'dst_port_id', 'proto', 'ip_version'):
            fields.append(pa.field(name, pa.int32()))
        else:
            fields.append(pa.field(name, pa.string()))
    for name in FEATURE_COLUMNS:
        fields.append(pa.field(name, pa.float64()))
    fields += [pa.field('payload_segments', pa.list_(pa.binary())),
               pa.field('payload_dirs', pa.list_(pa.int8())),
               pa.field('n_segments_stored', pa.int32())]
    return pa.schema(fields)


SCHEMA = _record_schema()


def extract_unit(args) -> Dict:
    cfg, capture_id, files, out_dir, max_packets = args
    tag = _unit_tag(capture_id)
    done_marker = os.path.join(out_dir, f'_done-{tag}')
    if os.path.exists(done_marker):
        return {'capture_id': capture_id, 'skipped': True}

    buffer: List[Dict] = []
    part = [0]
    chunk = int(cfg['flow_extractor']['chunk_flows'])

    def flush():
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=SCHEMA)
        pq.write_table(table, os.path.join(out_dir, f'part-{tag}-{part[0]:05d}.parquet'), compression='zstd')
        part[0] += 1
        buffer.clear()

    def sink(record):
        buffer.append(record)
        if len(buffer) >= chunk:
            flush()

    table = FlowTable(cfg, capture_id, sink)
    for path in files:
        table.set_file(path)
        try:
            for ts, buf, dlt in iter_packets(path):
                table.process(ts, buf, dlt)
                if max_packets and table.stats.packets >= max_packets:
                    break
        except (dpkt.NeedData, dpkt.UnpackError, EOFError, struct.error, OSError) as e:
            # A truncated tail is common in rotated captures; keep what was read.
            logger.warning(f"{path}: read stopped early ({type(e).__name__}: {e})")
            table.stats.parse_errors += 1
        if max_packets and table.stats.packets >= max_packets:
            break
    table.sweep(final=True)
    flush()

    stats = table.stats.__dict__.copy()
    stats['files'] = len(stats['files'])
    stats['capture_id'] = capture_id
    with open(done_marker, 'w') as f:
        f.write(repr(stats))
    return stats


def extract_all(cfg, pcaps: List[str], root: str, out_dir: str, workers: int = 1,
                max_packets_per_unit: Optional[int] = None) -> pd.DataFrame:
    """Runs the extractor over every unit (resumable: finished units are skipped)."""
    os.makedirs(out_dir, exist_ok=True)
    units = plan_units(pcaps, cfg['flow_extractor']['continuity'], root)
    logger.info(f"Extracting {len(units)} capture unit(s) from {len(pcaps)} file(s) with {workers} worker(s)")
    plain_cfg = cfg.to_dict() if hasattr(cfg, 'to_dict') else cfg
    jobs = [(plain_cfg, cid, files, out_dir, max_packets_per_unit) for cid, files in units]
    if workers > 1 and len(jobs) > 1:
        with Pool(workers) as pool:
            results = list(pool.imap_unordered(extract_unit, jobs))
    else:
        results = [extract_unit(job) for job in jobs]
    return pd.DataFrame(results)


def load_flows(flow_dir: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    parts = sorted(p for p in os.listdir(flow_dir) if p.startswith('part-') and p.endswith('.parquet'))
    if not parts:
        raise FileNotFoundError(f"No extracted flow parts in {flow_dir}; run `run_e0.py extract` first")
    return pd.concat([pq.read_table(os.path.join(flow_dir, p), columns=columns).to_pandas() for p in parts],
                     ignore_index=True)
