# has_payload rule (E0.2)

Implementation: `fhf/data/payload_rules.py` (`decide`). Unit tests: `tests/test_e0.py`.
Thresholds live in `configs/base.yaml` under `payload:` and are copied into every run's `config.yaml`.

## Segments

- A **segment** is the application bytes of one packet, in capture order, both directions
  (`payload.directions: both`), after TCP retransmissions are dropped (same direction,
  sequence number and length; `fhf/data/flow_extractor.py`).
- The first `j_max = 3` non-empty segments of each flow are stored raw (max 2048 bytes
  each), so this rule can change without re-extracting the PCAPs.

## Per-segment reason (first matching rule wins)

| reason | condition |
|---|---|
| `tls_record` | starts with a TLS record header: byte 0 in 0x14–0x17, byte 1 = 0x03, byte 2 ≤ 0x04. Checks the content, never the port |
| `high_entropy` | ≥ 64 bytes and Shannon entropy > 7.2 bits/byte (encrypted or compressed) |
| `unreadable_binary` | printable share (0x20–0x7E, TAB, LF, CR) < 0.75 |
| `too_short` | fewer than 4 printable characters |
| `ok` | readable |

## Per flow

- `m_i = 1` when at least one stored segment is `ok`. Only `ok` segments become payload nodes, and they keep their original order.
- `m_i = 0` otherwise. The reason is `no_payload` when there are no application bytes; otherwise it is the reason of the first stored segment.
- The pipeline adds two more reasons, which the rule itself never produces: `masked` (E4) and `disabled` (A1).

## Text form

Printable ASCII is kept exactly, including case (so `<ScRiPt>` survives). Each run of other bytes becomes one `<NP>` token. Text is truncated to 256 tokenizer tokens at encoding time.

## Notes

- An SSH banner (`SSH-2.0-…`) is readable plaintext and counts as `ok`. The encrypted packets after it fail the entropy or printable test.
- Every `m_i = 0` has a reason code. Counts per flow and per segment are in `work/<ds>/e0/has_payload_reason_counts.csv`.
