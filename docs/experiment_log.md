# Experiment log: FedHetFormer-IDS sprint (25–26 Sep 2026)

This log records, in order, what was run, why, and what it showed. It also records the decisions made along the way. Each entry gives a short result; the full outputs are in [`experiment_log_outputs/`](experiment_log_outputs/) and at the links. Times are local (UTC+6).

**Where things live**

| What | Link |
|---|---|
| Raw PCAPs + labelled CSVs (input) | https://huggingface.co/datasets/mtasfi/ton-iot-with-pcap-payload |
| E0 release (flows + payloads + splits), tags `v1`, `v2` | https://huggingface.co/datasets/mtasfi/toniot-fhf-processed (private) |
| Shared run state (runs/, E1 caches, results/) | https://huggingface.co/datasets/mtasfi/fhf-state (private) |
| Comet project | https://www.comet.com/text-films/fedhetformer-ids |
| E0 build notebook | https://www.kaggle.com/code/mtasfi/toniot-build-dataset |
| Experiment runners | https://www.kaggle.com/code/mtasfi/fhf-runner-a, https://www.kaggle.com/code/mtasfi/fhf-runner-b |
| Diagnostics notebooks | `toniot-split-explore`, `toniot-shortcut-check`, `toniot-tree-rules`, `toniot-nf-features`, `toniot-capture-heldout` (all under https://www.kaggle.com/code/mtasfi) |

Every run directory under `runs/` in `fhf-state` holds the record: `config.yaml`, `manifest.json`, `test_metrics.csv`, `per_class_metrics.csv`, `per_client_metrics.csv`, `confusion_matrix.csv`, `round_metrics.csv` and `run.log`. Comet mirrors the scalars.

---

## 1. Early attempt on a CSV without payloads (abandoned)

- **Idea:** try the no-payload pipeline on `arnobbhowmik/ton-iot-network-dataset`, the ToN-IoT `train_test_network.csv`: 211k rows, no timestamps, no payloads.
- **What happened:** we wrote a converter (pseudo-time from the row order, 19 of the 50 features). Uploading the private code bundle to Kaggle was blocked by the permission system. We dropped the idea in favour of processing the real PCAPs.
- **Takeaway:** that CSV has no timestamps, so a block split is impossible and it is useless for the thesis.

## 2. Plan: build the processed dataset once (E0 release)

- **Decision (owner):** upload the PCAPs and processed CSVs to Hugging Face, extract flows and payloads once, and publish a release that links `flows.parquet` and `payloads.parquet` through `flow_uid`. Every experiment then just imports the release.
- **Build notebook:** `toniot-build-dataset`. It runs `experiments/run_e0.py` (extract → labels → offset → match → check → payload → split → kappa), exports the release, checks that the payload texts equal Phase 1's `payload_texts`, and pushes to HF.
- **Setup problems, all solved:** Kaggle secrets are not passed to API runs until they are attached in the UI, and Internet was off. The owner made the code repo public, so no GitHub token is needed. HF upload works once `HF_TOKEN` is attached.

## 3. Smoke runs (Kaggle v5–v8): 2 PCAPs, 500k packets each

- Clock offset 0 s (99.9 % peak), match 99.9 %, 49,924 flows, 75 % with a payload, SQLi signature agreement 100 %.
- The split failed, as expected with only 7 blocks. The notebook was changed so that a split failure no longer blocks the export, and so that the HF push is tested first.
- Release export: `flows.parquet` 8.7 MB, `payloads.parquet` 9.1 MB. The payload consistency check passed.

## 4. Full E0, release v1 (Kaggle v9): 18 PCAPs, 20 of 23 label CSVs

Outputs: [`experiment_log_outputs/e0_release_v1/`](experiment_log_outputs/e0_release_v1/)

- 5.45 M flows extracted, 2.68 M labelled. **Match rate 49 %**, and no ransomware labels. `Network_dataset_1`, `11` and `22` were missing from HF.
- **The split failed at every setting.** Each attack class lives in 1–3 captures, and one 300 s block can hold about 100k flows.
- **Extractor bug found:** `continuity=per_directory` read the 18 scenario PCAPs, which sit in one flat folder, as a single stream under one `capture_id`.

## 5. Split exploration (`toniot-split-explore` v4–v5)

Scripts: [`split_explore.py`](experiment_log_outputs/split_explore.py), [`split_explore2.py`](experiment_log_outputs/split_explore2.py)

- 16 settings (300/120/60/30 s × capture/file blocks × 75k/150k target): **none passed**. The flow floor was met by a single huge block, which then had to stay in training.
- Adding a **block-count floor** (every class held by ≥ 4 or ≥ 8 blocks) made **all 16** settings pass at α = 0.3, 0.5 and IID.
- **Decision:** `block_seconds = 60`, `target_flows = 150000`, `min_blocks_per_class = 8` (commit `8a105fc`). Also `continuity = per_file` for the flat HF folder.

## 6. Full E0, release v2 (Kaggle v11): 18 PCAPs, all 23 CSVs

Outputs: [`experiment_log_outputs/e0_release_v2/`](experiment_log_outputs/e0_release_v2/)

- The owner uploaded `Network_dataset_1`, `11` and `22`.
- **3.99 M labelled flows, match 73.1 %**, 0 % ambiguous, offset 0 s. All 10 classes are present (ransomware 5,160; MITM 438).
- The matching gate reports FAIL (match rate and per-class record coverage below 0.80). **Accepted:** the release holds only some of the captures that were recording at the same time. Signature agreement is 99.9 %.
- has_payload: injection 84 %, password 75 %, XSS 72 %, normal 46 %. DoS, backdoor, ransomware, DDoS and scanning have 0–4 %.
- **Split passes:** 195,141 flows, 125 blocks, 26.9 % test.

**E0 decisions** (commit `b471595`; the reasons are next to the values in `configs/base.yaml`):
- **Validation rule:** MITM is recorded last, so every MITM training flow went to validation. A block now stays in training if moving it would remove a client's last flows of a class. The owner gave no reply within 5 minutes, so I decided, per their standing instruction. The alternative was dropping MITM.
- **α = 0.5** (sweep 0.3 and 1.0): every client holds 5–7 classes and every class sits on 2–5 clients.
- **κ = 10:** the uncapped degree is about 1,331 and saturation never shows a knee, so the cap was set by cost (about 4.8 edges per flow).
- **B2 is not rerun** (owner's decision).

## 7. Runners and first batch (Runner B, 01:41–02:25)

Outputs: [`experiment_log_outputs/runner_b_batch1/`](experiment_log_outputs/runner_b_batch1/)

`tools/kaggle_runner.py` runs the E0 tests, restores release v2, rebuilds the splits with the current code, pulls and pushes the state to `fhf-state`, and runs a queue.

| Run | Macro-F1 | Bal. acc. | Worst client | Injection | Password | XSS | Comet |
|---|---|---|---|---|---|---|---|
| B5 XGBoost | **0.865** | 0.899 | 0.655 | 0.929 | 0.928 | 0.979 | [link](https://www.comet.com/text-films/fedhetformer-ids/ca0ae5030ccd423488dbf206588fb761) |
| B5 XGBoost+SMOTE | 0.861 | 0.900 | 0.647 | 0.897 | 0.927 | 0.981 | [link](https://www.comet.com/text-films/fedhetformer-ids/5983192c57ba4bfc962cb0e7ddeda74c) |
| B5 RF+SMOTE | 0.860 | 0.894 | 0.627 | 0.909 | 0.953 | 0.961 | [link](https://www.comet.com/text-films/fedhetformer-ids/248737b35b4f4c45967cc8665786b293) |
| B5 RF | 0.854 | 0.874 | 0.592 | 0.924 | 0.953 | 0.961 | [link](https://www.comet.com/text-films/fedhetformer-ids/26f8c0411964450cb9b8d1974b7847e4) |
| A2 hashed n-grams | 0.794 | 0.830 | 0.559 | 0.738 | 0.950 | 0.977 | [link](https://www.comet.com/text-films/fedhetformer-ids/2a0a6bc8c2734b489ca0f6c384367ed9) |
| A1 no payload | 0.777 | 0.817 | 0.524 | 0.630 | 0.895 | 0.972 | [link](https://www.comet.com/text-films/fedhetformer-ids/c02580ccf78646959f8090ee5d972813) |
| B1 FedAvg MLP | 0.775 | 0.783 | 0.561 | 0.586 | 0.794 | 0.934 | [link](https://www.comet.com/text-films/fedhetformer-ids/4e8072f08d894e519decce8fff96311c) |

- B5 is trained centrally, so it is a reference, not a federated baseline. SMOTE changes it by ±0.5 points.
- **Payload helps injection (+10.8) and password (+5.5)** from A1 to A2. XSS gains nothing, since it is already 0.97 without a payload.
- B1's validation F1 was still rising at round 10, so the federated models may be under-trained with the sprint budget of R2 = 10.
- A1 ≈ B1: without payloads, `shares_host` adds little.

## 8. "Is the model cheating?" Four diagnostics (CPU)

Full write-up: [`flow_feature_diagnostics.md`](flow_feature_diagnostics.md). Script: [`shortcut_check.py`](experiment_log_outputs/shortcut_check.py).

The question came up because flow-only models scored high, while on NF-ToN-IoT (Fed_GNN) XSS, password and scanning looked inseparable.

1. **Ports, TCP window and header bytes removed:** macro-F1 0.860 → 0.864. **No leakage.** The top features are behavioural (`rst_ratio`, `syn_ratio`, `psh_cnt`, packet lengths).
2. **Decision trees:** backdoor is caught by one rule (`46 < fwd_bytes ≤ 50`, a fixed-size beacon of the tool), and DoS and scanning by 3 rules. Injection, password and XSS need depth ≥ 6 and have no single-rule shortcut.
3. **NetFlow-like 7–8 features vs. our 50:** 0.871 vs. 0.860 on the block split. The number of features is not the explanation.
4. **Capture-held-out** (train on the other captures of the class, test on the held-out one): XSS 0.98–0.99, password 0.93–1.00, DoS and DDoS ≥ 0.89. **Injection 0.81** on one capture. **Scanning** needs the rich features (NF-7: 0.02–0.16; 50 features: 0.63–0.96). Caveat: the captures of one class overlap in time (same campaign and tool).

**Conclusion:** there is no cheating. On ToN-IoT, flow statistics separate most classes, so payload claims should be scoped to **injection (and password)** in the federated setting.

## 9. E1 and E5 (Runner A from 01:41, Runner B from 02:27)

Metrics from Comet: [`experiment_log_outputs/comet/test_metrics_E1_E5_PROBE.txt`](experiment_log_outputs/comet/test_metrics_E1_E5_PROBE.txt)

| Run | Time | Macro-F1 | Worst client | Injection | Password | XSS | MITM | Ransomware | Comet |
|---|---|---|---|---|---|---|---|---|---|
| **E1 seed 0** (α=0.5) | 01:49–03:51 | **0.815** | 0.591 | 0.769 | 0.978 | 0.968 | 0.00 | 0.78 | [link](https://www.comet.com/text-films/fedhetformer-ids/2636ab6f08d447d0b4d95e2f09246044) |
| E1 seed 1 | from 03:51 | running | | | | | | | [link](https://www.comet.com/text-films/fedhetformer-ids/2c39057d09b545b298fe5455822576da) |
| E5 α=0.3 | 02:29–04:17 | 0.922 | 0.801 | 0.911 | 0.920 | 0.951 | 0.82 | 1.00 | [link](https://www.comet.com/text-films/fedhetformer-ids/60cb656925164f4fb7aba24c0a4dfd72) |
| E5 α=1.0 | from 04:17 | running | | | | | | | [link](https://www.comet.com/text-films/fedhetformer-ids/41593bcdc7b64b6886783ae83f7f0d4e) |
| PROBE (1 round, 10k flows) | 01:45–01:49 | 0.212 | – | – | – | – | – | – | [link](https://www.comet.com/text-films/fedhetformer-ids/e1adaa54d83e4f64ab20d76eace5e5a0) |

- **E1 beats every federated baseline:** +4.0 over B1, +3.8 over A1 and +2.1 over A2. The gain is on the payload-dependent classes: injection 0.63 → 0.77, password 0.90 → 0.98. E1 also has the best worst-client score.
- E1 is still 5 points below central XGBoost (0.865), and it misses MITM entirely (61 test flows).
- **Open question, partition sensitivity:** at α = 0.3, E5 scores 0.922, far above E1 at α = 0.5. Where the MITM and injection blocks land decides a lot. Seed 1 of E1 and E5 at α = 1.0 will show the variance. A second α = 0.5 partition draw may be needed.
- Each E1 seed takes about 2 h, against the planned 2.5 h for both seeds.
- PROBE is a timing run only; its score is meaningless (1 round).

## 10. E1 finished (R2 = 10) and Phase-2 rounds raised to 20

- **E1 at R2 = 10:** seed 0 scored 0.815 and seed 1 0.849, so **0.832 ± 0.023** (worst client 0.626). Averaged over both seeds, injection is 0.835 and password 0.982, above central XGBoost (0.928). MITM is 0.00 / 0.23.
- **Where the time goes** (runner log): a Phase-1 round takes about 22 min, 5 rounds about 110 min, embedding about 10 min, and a Phase-2 round about 6 s.
- For both seeds, validation macro-F1 was still rising at Phase-2 round 10, and the best round was the last one (seed 1: 0.81 → 0.92). B1 behaved the same way.
- **Decision: R2 = 20** (commit `5bf3116`) for every federated Phase-2 model. 20 is the plan's full-scale value, and it adds about 1 min per run. The Phase-1 cache tag does not include R2, so E1 reuses Phase 1 and the embeddings. The owner raised no objection within 5 minutes.
- Queued with R2 = 20: Runner A (v3) runs E1 (Phase 2 only), then E4, A5, B3, A3, E6, A4, A8, A9, A6. Runner B (v4) runs B1, A1, A2, then E5 (Phase 2 only).

## 11. E5 at R2 = 10: a non-monotonic α trend caused by lumpy blocks

| α | Macro-F1 | Worst client | Injection | Ransomware | Comet |
|---|---|---|---|---|---|
| 0.3 | 0.922 | 0.801 | 0.911 | 0.998 | [link](https://www.comet.com/text-films/fedhetformer-ids/60cb656925164f4fb7aba24c0a4dfd72) |
| 0.5 (E1 mean) | 0.832 | 0.626 | 0.835 | 0.772 | see §9 |
| 1.0 | **0.665** | 0.432 | **0.000** | **0.033** | [link](https://www.comet.com/text-films/fedhetformer-ids/41593bcdc7b64b6886783ae83f7f0d4e) |

- **Cause:** injection has 8 blocks, but one of them holds about 3.9k of injection's roughly 5.5k training flows. At α = 1.0, every injection training flow sits on client 1, while client 4 holds 611 injection *test* flows and no injection training data. After FedAvg, the global model predicts injection as DDoS (1,250 of 1,638 flows). At α = 0.3, the big block lands on the same client as most of the test flows (0.91). Ransomware behaves the same way. Full outputs: runner-b v3 output `runs/E5/.../alpha=1.0/`.
- So E5's α trend mostly reflects **which client draws the big block**, not α. Splitting blocks would break the no-cut rule.
- **Decision:** rerunning E5 with several partition draws is not affordable (Phase 1 takes about 2 h per draw). Instead, a **partition-sensitivity** analysis runs the Phase-1-free models A2 and B1 at α ∈ {0.3, 0.5, 1.0, IID} × partition seeds {0, 1, 2} (24 short runs; runner support in commit `78c3334`). E5 will be reported next to that spread.

## 12. Thesis repo sync

- PR #3 (new Dataset/Setup/Results chapters) was merged on GitHub. The E0 commit was rebased onto it (`d1e93f5`), and the E0 numbers were filled into `Tab_D_Stats` and "Outcome of the Data Audit". Still open: template overlap and probe timing.

---

## Still to run

| Runner | Queue |
|---|---|
| A (v3, running) | E1 (Phase 2, R2 = 20) → E4 → A5 → B3 → A3 → E6 → A4 → A8 → A9 → A6 |
| B (v4, running) | B1 → A1 → A2 → E5 (Phase 2, R2 = 20) |
| B (next) | partition sensitivity: A2, B1 × α {0.3, 0.5, 1.0, IID} × partition seed {0, 1, 2} |
| last | (A7) → AGG → thesis Results |
