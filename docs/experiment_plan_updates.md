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

## E0 on the PCAP release (release v2): extraction, labels, decisions

Inputs: 18 captures and all 23 `Network_dataset_*.csv` files from the Hugging Face release
(`mtasfi/ton-iot-with-pcap-payload`); output: `mtasfi/toniot-fhf-processed`, tag `v2`.

- **`flow_extractor.continuity = per_file`.** The captures are different scenarios in one flat folder;
  `per_directory` (meant for rotated files of one capture) had read them as a single stream in file-name
  order under one `capture_id`.
- **Matching:** 73.1 % of extracted flows matched a label row (0 % ambiguous), clock offset 0 s. SQL-injection
  and XSS signatures agree with the label for 99.9 % of matched hits. The 0.80 match-rate gate and the
  per-class row-coverage gate fail and are accepted: the release holds only some of the captures that were
  recording at the same time, so label rows from the missing captures lower both numbers without saying
  anything about the flows we have. Only matched flows are used.
- **Classes:** all ten classes are present (3.99 M labelled flows). Ransomware (5,160) and MITM (438) are small
  and kept.
- **Payload-free classes are expected.** DoS, backdoor, ransomware, DDoS and scanning carry almost no readable
  payload (0–4 %). They are detected from flow statistics and `shares_host` edges with `m_i = 0`; the
  payload-dependent classes keep 75 % (password), 72 % (XSS) and 84 % (injection) `has_payload`.
- **Validation never empties a class from a client's training data.** Validation takes each client's latest
  blocks. The MITM captures are the last day of the recording, so every MITM training flow went to validation
  at every α and no client could learn the class. A block now stays in training if moving it would remove the
  client's last training flows of a class (`fhf/data/partition_clients.py`).
- **α = 0.5 (primary), E5 sweep [0.3, 1.0].** At 0.5 every client holds 5–7 classes and every class sits on
  2–5 clients. 0.3 and 1.0 are its neighbours and give the sweep real contrast (at 1.0 ransomware sits on one
  client).
- **κ = 10.** Uncapped, a flow has ~1,331 same-host neighbours within 60 s. Saturation stays high at every cap
  (0.91 at 2, 0.85 at 10, 0.78 at 50), so there is no knee; the cap is set by cost. κ = 10 gives ~4.8 edges
  per flow, which fits a T4 with a full-graph forward.
- **Split (60 s, ≥ 8 blocks per class, 150 k target):** 195,141 flows in 125 blocks, test fraction 26.9 %,
  every class has train and test flows.
- **B2 is not rerun** in this sprint (owner's decision); the NF-ToN-IoT reproduction stays as the reference.
