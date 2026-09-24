"""The has_payload rule (E0.2): one deterministic function from raw segment bytes
to (m_i, reason, segment texts). Identical bytes always give identical output.

Per segment, in this order (first matching rule wins):
    tls_record         starts with a TLS record header: type 0x14-0x17, version 0x03 0x00-0x04.
                       Content-based, never port-based.
    high_entropy       >= entropy_min_len bytes and Shannon entropy > entropy_threshold bits/byte
                       (encrypted or compressed).
    unreadable_binary  printable-byte ratio < min_printable_ratio.
    too_short          fewer than min_readable_chars printable characters.
    ok                 readable.
Per flow:
    m_i = 1 iff at least one of its (first J_max, capture-order) segments is `ok`.
    Only `ok` segments become payload nodes, in their original order.
    If m_i = 0 the reason is `no_payload` (no application bytes at all) or the
    reason of the first stored segment.
Reasons used later by the pipeline (not produced here): `masked` (E4 sweep),
`disabled` (A1: payload channel removed).

Text form: printable ASCII (0x20-0x7E, tab, LF, CR) is kept as is (case preserved,
e.g. `<ScRiPt>`); each run of other bytes becomes one `placeholder` token.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Sequence

PRINTABLE = frozenset(list(range(0x20, 0x7F)) + [0x09, 0x0A, 0x0D])
_NON_PRINTABLE_RUN = re.compile(rb'[^\x20-\x7e\t\n\r]+')

SEGMENT_REASONS = ('ok', 'tls_record', 'high_entropy', 'unreadable_binary', 'too_short')
FLOW_REASONS = ('ok', 'no_payload', 'tls_record', 'high_entropy', 'unreadable_binary', 'too_short',
                'masked', 'disabled')


@dataclass
class PayloadDecision:
    has_payload: int
    reason: str
    texts: List[str] = field(default_factory=list)       # readable segments only, capture order
    kept_indices: List[int] = field(default_factory=list)  # their positions among the stored segments
    segment_reasons: List[str] = field(default_factory=list)
    truncated: bool = False


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in Counter(data).values())


def is_tls_record(data: bytes) -> bool:
    return len(data) >= 5 and data[0] in (0x14, 0x15, 0x16, 0x17) and data[1] == 0x03 and data[2] <= 0x04


def to_text(data: bytes, placeholder: str) -> str:
    return _NON_PRINTABLE_RUN.sub(placeholder.encode('ascii'), data).decode('ascii')


def segment_reason(data: bytes, rule) -> str:
    if is_tls_record(data):
        return 'tls_record'
    if len(data) >= rule['entropy_min_len'] and shannon_entropy(data) > rule['entropy_threshold']:
        return 'high_entropy'
    printable = sum(1 for b in data if b in PRINTABLE)
    if printable / max(len(data), 1) < rule['min_printable_ratio']:
        return 'unreadable_binary'
    if printable < rule['min_readable_chars']:
        return 'too_short'
    return 'ok'


def decide(segments: Sequence[bytes], rule) -> PayloadDecision:
    """rule = cfg.payload (placeholder, thresholds, j_max, max_bytes_per_segment)."""
    segments = [] if segments is None else list(segments)   # Parquet hands back numpy arrays
    segments = [bytes(s) for s in segments if s is not None and len(s) > 0][: int(rule['j_max'])]
    if not segments:
        return PayloadDecision(0, 'no_payload')

    reasons = [segment_reason(s, rule) for s in segments]
    kept = [i for i, r in enumerate(reasons) if r == 'ok']
    truncated = any(len(s) >= int(rule['max_bytes_per_segment']) for s in segments)
    if not kept:
        return PayloadDecision(0, reasons[0], [], [], reasons, truncated)
    texts = [to_text(segments[i], rule['placeholder']) for i in kept]
    return PayloadDecision(1, 'ok', texts, kept, reasons, truncated)
