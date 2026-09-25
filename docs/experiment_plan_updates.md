# Experiment Plan v1.2 → v1.3 amendments

These amendments change the sections of the Sprint Edition PDF listed below. Everything not listed stays as in v1.2.

## B2: FedGATSage (replaces the Tier-2 B2 box and the "Direct-comparison gate for B2")

**B2 — FedGATSage rerun on the E1 split: +~2 GPU-h on both the ToN-IoT and the CIC path**

- **Question:** Does FedHetFormer-IDS outperform the closest graph-based federated prior work on the *same* split?
- **Default:** rerun. The existing reproduction was run on NF-ToN-IoT, which has no absolute flow timestamps. Its flows therefore cannot be aligned with the PCAP-derived flows that E1 uses. FedGATSage is rerun on E1's split, label map, client partition, seed and flow features. The Content GAT's payload statistics come from our PCAP extractor. The adapter and the deviations are described in `docs/b2_fedgatsage_rerun.md`.
- **Keep the old run.** The completed NF-ToN-IoT run is preserved as the *paper-faithful reproduction*. It is the motivation evidence, not the head-to-head number.
- **Exception:** the old run may be reused only if E0 shows that its flows can be matched one-to-one to ours. Record the match rate in the E0 report if this is attempted.
- **Order:** B2 moves to the start of step 4, right after Tier 1 and before A2/A3, because the main table needs it.
- **Done:** the rerun completes on the exact E1 split; every modification is documented in the run config and manifest; the old run is indexed as paper-faithful reproduction evidence.

## §4.1 Planning time budget (replacement rows)

| Work block | Planning estimate |
|---|---|
| E0 preprocessing | ~0 GPU-h (CPU only) |
| **Timing probe** (1 round, 10k flows; Phase 1 and Phase 2 timed separately) | ~0.2 GPU-h; run after E0, before the E1 queue (`experiments/run_probe.py`) |
| E1 main, 2 seeds (ModernBERT-base) | ~2.5 GPU-h |
| B1 MLP + B3 centralized | ~1 GPU-h |
| **B2 FedGATSage** | **+~2 GPU-h rerun on both paths** (the ToN-IoT run can no longer count as 0 GPU-h) |
| B5 XGBoost/RF ± SMOTE | CPU only |
| A1–A6, A8–A9 | ~3.3 GPU-h |
| E4 missing sweep | ~2 GPU-h (inference only in this implementation; expected well below) |
| E5 α sweep, 2 α values | ~3 GPU-h |
| E6/E7 | ~0.5 GPU-h |
| **ToN-primary path** | **≈14 GPU-h; ≈18 h with debugging**, plus B5 CPU time |
| CIC fallback path | ≈14 GPU-h; ≈18 h with debugging (unchanged) |

Both paths still fit a 2–3 day sprint and Kaggle's 30 h/week quota. Recalibrate this table from `probe_budget.csv` once the timing probe has run.

## §2.2 Split: 60 s blocks with a block-count floor (replaces the 300 s block rule)

**What changed:** `split.block_seconds` 300 → 60, `split.target_flows` 75,000 → 150,000, and a new
`split.min_blocks_per_class = 8`: after the flow floor, every class must be held by at least 8 selected
blocks (random unselected blocks holding the class are added). Blocks are still never cut and rows are
still never sampled.

**Why:** the first full E0 on the ToN-IoT PCAP release (18 captures, 2.68 M labelled flows) put every
attack class in one to three captures, and one busy block can hold ~100 k scanning flows. With 300 s
blocks the flow floor was met by a single block, which step 2 must keep in training, so classes ended
with no test flows and the Dirichlet partition failed at α = 0.3. We tried 16 settings (300/120/60/30 s
blocks × capture or file blocks × 75 k/150 k target): none passed. Adding the block-count floor, all 16
of 60/30 s × 4/8 blocks passed, with the partition drawable at α = 0.3, 0.5 and IID. 60 s with 8 blocks and
150 k flows was chosen because it keeps each class on at least two clients at every α.

**Known side effect:** block sizes stay uneven, so scanning dominates the test set by flow count.
Macro-F1 (the headline) is unaffected; accuracy is reported but not used as the headline.

## E0 on the PCAP release: extraction and label coverage

- **`flow_extractor.continuity = per_file`** for the Hugging Face PCAP release. Its 18 captures are
  different scenarios in one flat folder; `per_directory` (meant for rotated files of one capture) read
  them as a single stream in file-name order under one `capture_id`.
- **Ransomware is dropped.** The available `Network_dataset_*.csv` files (2–10, 12–21, 23) have no
  ransomware rows, so no flow can carry that label. `Network_dataset_1`, `11` and `22` are not in the
  release. The two ransomware captures are kept; their flows match other labels or stay unmatched.
- **Match rate is below the 0.80 gate (49 %).** Unmatched flows are mostly sub-second TCP resets and idle
  flows whose 5-tuple is absent from the label CSVs. Matched flows are reliable: SQL-injection and XSS
  signatures agree with the label for 99.7 % and 99.9 % of matched hits, and the clock offset is 0 s. The
  per-class row coverage check does not apply here, because the release holds only some of the
  captures that were recording at the same time. Only matched flows are used.
- **Payload-free classes are expected.** DoS, DDoS, scanning and backdoor carry almost no readable
  payload (0–6 %). They are detected from flow statistics and `shares_host` edges with `m_i = 0`; the
  payload-dependent classes (injection, XSS, password) keep 71–79 % `has_payload`.
