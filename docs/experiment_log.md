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

**Why (owner):** PCAPs could not be uploaded at the time, so the owner asked to try the no-payload pipeline on a ToN-IoT CSV already on Kaggle.

- **Idea:** try the no-payload pipeline on `arnobbhowmik/ton-iot-network-dataset`, the ToN-IoT `train_test_network.csv`: 211k rows, no timestamps, no payloads.
- **What happened:** we wrote a converter (pseudo-time from the row order, 19 of the 50 features). Uploading the private code bundle to Kaggle was blocked by the permission system. We dropped the idea in favour of processing the real PCAPs.
- **Takeaway:** that CSV has no timestamps, so a block split is impossible and it is useless for the thesis.

## 2. Plan: build the processed dataset once (E0 release)

**Why (owner):** the owner wanted to upload the CSVs and PCAPs once, extract flows and payloads with a `flow_uid` linking them, and let every experiment import the result instead of re-extracting.

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

**Why:** all experiments had to finish within two days. Two Kaggle GPU sessions in parallel, with state kept on HF, lose at most one run to a session limit.

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

**Why (owner):** "Did the runner-B models cheat, e.g. through a column tied to the label such as IP or port? Fed_GNN's centralised version dropped many of those." The owner then asked ("are there too many features? NF-ToN-IoT has only 14 columns") and requested a decision tree ("can we see a direct pattern, if this then that label?"), which led to diagnostics 2 and 3.

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

**Why (owner):** "How many rounds do we run?" The round logs showed validation F1 still rising at the last round. The owner also asked whether Phase 1 would rerun (it does not; see below).

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

## 11b. R2 = 20 results (Runner B v4, Runner A v3 in progress)

| Run | R2 = 10 | **R2 = 20** |
|---|---|---|
| **E1** (seeds 0/1) | 0.815 / 0.849 → 0.832 ± 0.023 | **0.840 / 0.877 → 0.858 ± 0.027** |
| A2 hashed n-grams | 0.794 | 0.841 |
| A5 no `shares_host` | – | 0.830 |
| A1 no payload | 0.777 | 0.824 |
| B1 FedAvg MLP | 0.775 | 0.810 |
| E5 α = 0.3 | 0.922 | 0.924 |
| E5 α = 1.0 | 0.665 | **0.882** |
| B5 XGBoost (central, reference) | 0.865 | – |

- Every federated model gains 3–5 points, which confirms that R2 = 10 under-trained them.
- E1 is now level with central XGBoost (0.858 vs. 0.865); seed 1 beats it. E1 − B1 = +4.8, E1 − A1 = +3.4, E1 − A2 = +1.7. Removing `shares_host` costs about 1 point (seed 0: 0.840 vs. 0.830).
- **Correction to §11:** with R2 = 20, E5 at α = 1.0 no longer loses injection (0.665 → 0.882), so under-training was a large part of that collapse. The lumpy injection block is real, but it did not cause the collapse on its own. E5 is still non-monotonic (0.924 / 0.858 / 0.882 for α = 0.3 / 0.5 / 1.0), so the partition-sensitivity runs (Runner B v5) are still needed to separate the α effect from the partition-draw effect.
- The R2 = 10 runs are kept next to the new ones as `runs/<exp>/.../seed=0.completed-<timestamp>/`.

## 11c. Ablation batch, seed 0, R2 = 20 (Runner A v3, 06:00–06:53)

| Run | What changes vs. E1 | Macro-F1 | Worst client | Minutes |
|---|---|---|---|---|
| E1 seed 0 | – | 0.840 | 0.634 | 5.8 (Phase 2 only) |
| B3 | centralised HetGNN (no federation) | 0.867 | 0.655 | 10.3 |
| A3 | frozen encoder, no Phase-1 LoRA | **0.861** | **0.722** | 14.3 |
| A4 | homogeneous graph | 0.859 | 0.685 | 3.1 |
| A6 | no `next` edges | 0.854 | 0.673 | 2.9 |
| A8 | no graph (fused-feature MLP) | 0.841 | 0.617 | 1.8 |
| A5 | no `shares_host` | 0.830 | 0.618 | 2.0 |
| A9 | payload only (Phase-1 head) | 0.688 | 0.569 | 3.6 |
| E4 | missing-payload sweep (see run dir) | – | – | 1.2 |
| E6 | efficiency (see run dir) | – | – | 2.9 |

- On seed 0, several ablations (A3, A4, A6) score **above** E1, but E1's two seeds differ by 0.037. Single-seed differences of 1–2 points are within noise.
- **A3 > E1** on seed 0 means the federated LoRA adaptation (Phase 1) does not beat the frozen pretrained encoder here. If this holds over seeds, it is an honest negative result for Phase 1.
- A9 (payload only, 0.688) confirms that payload alone is far from enough: flow statistics carry most classes.
- B3 − E1 ≈ 2.7 points is the cost of federation on seed 0.

**Decision:** single-seed ablations cannot support these claims, and Phase-2 ablations are cheap. So **seed 1 of every ablation** (A1–A6, A8, A9, B1, B3, E4) runs, paired with E1 seed 1 (Runner A v4). Ablations will then be reported as 2-seed means next to E1's 2-seed mean.

**Bug found and fixed:** runner A pulled the shared state at 06:00 and pushed every local run dir after each experiment. This overwrote runner B's newer R2 = 20 results for A1/A2/B1/E5 in `fhf-state` with stale R2 = 10 copies. Runner B's next pushes restored them, and runners now push only the run dirs written in their own session (commit `6090aac`).

## 11d. Partition sensitivity (Runner B v5, 06:29–07:29, 24 runs, all exit 0)

**Why:** E5 gave a non-monotonic α trend (§11, §11b), so we had to separate the effect of α from the luck of one partition draw.

Macro-F1 of the Phase-1-free models over three partition draws (`partition.seed` 0/1/2, R2 = 20, training seed 0):

| Model | α | p0 | p1 | p2 | **mean** |
|---|---|---|---|---|---|
| A2 | 0.3 | 0.895 | 0.835 | 0.656 | 0.795 |
| A2 | 0.5 | 0.842 | 0.842 | 0.611 | 0.765 |
| A2 | 1.0 | 0.825 | 0.717 | 0.737 | 0.760 |
| A2 | IID | 0.804 | 0.892 | 0.929 | **0.875** |
| B1 | 0.3 | 0.824 | 0.777 | 0.625 | 0.742 |
| B1 | 0.5 | 0.810 | 0.694 | 0.621 | 0.708 |
| B1 | 1.0 | 0.656 | 0.701 | 0.626 | 0.661 |
| B1 | IID | 0.818 | 0.803 | 0.845 | **0.822** |

- At a fixed α, the partition draw moves macro-F1 by up to 0.23 (0.61–0.84), which is more than α itself does.
- Averaged over draws, the expected picture appears: IID is best, and skew hurts. A single draw per α (like E5) cannot show this.
- The main partition (α = 0.5, p0) is a **favourable draw** (A2 0.842 vs. a mean of 0.765), so single-partition numbers, including E1's 0.858, are optimistic.

**Decision:** the headline claim (E1 > federated baselines) must hold beyond one draw. **E1 is run on the α = 0.5 draws p1 and p2** (training seed 0; about 2 h each for Phase 1), paired with the A2/B1 runs above. Runner B v6 runs p1, and runner A runs p2 after its seed-1 ablations. The Phase-1 cache tag now includes the partition draw (commit `f84d254`); p0 keeps the old tag.

## 11e. Ablations over two seeds (R2 = 20; seed 1 from Runner A v4, 06:55–07:33)

**Why:** on one seed several ablations beat E1, but E1's own two seeds differ by 3.7 points, so single-seed ablation differences could not be interpreted.

Paired with E1 on the same partition (α = 0.5, p0). Values are macro-F1; Δ is the 2-seed mean minus E1's 2-seed mean.

| Run | Change vs. E1 | seed 0 | seed 1 | mean | Δ |
|---|---|---|---|---|---|
| **E1** | – | 0.840 | 0.877 | **0.858** | – |
| B3 | centralised (no federation) | 0.867 | 0.888 | 0.878 | +1.9 |
| A6 | no `next` edges | 0.854 | 0.880 | 0.867 | +0.9 |
| A4 | homogeneous graph | 0.859 | 0.868 | 0.863 | +0.5 |
| A2 | hashed n-grams instead of transformer | 0.841 | 0.863 | 0.852 | −0.6 |
| A8 | no graph (fused MLP) | 0.841 | 0.855 | 0.848 | −1.0 |
| A3 | frozen encoder (no LoRA) | 0.861 | 0.828 | 0.844 | −1.4 |
| A5 | no `shares_host` | 0.830 | 0.840 | 0.835 | **−2.3** |
| A1 | no payload | 0.824 | 0.828 | 0.826 | **−3.2** |
| B1 | FedAvg MLP, flow statistics only | 0.810 | 0.820 | 0.815 | **−4.3** |
| A9 | payload only (Phase-1 head) | 0.688 | 0.675 | 0.682 | −17.6 |

- **Consistent over both seeds:** the payload channel (A1), host context (A5), and the full method against the flow-only federated floor (B1).
- **Small or sign-unstable:** transformer vs. hashing (A2), LoRA vs. frozen (A3: +2.1 on seed 0, −4.9 on seed 1), message passing (A8).
- **No benefit:** typed nodes/relations (A4) and `next` edges (A6).
- Federation costs about 2 points (B3).
- **For the thesis:** the payload and host-context claims are supported. The claim that the *transformer + federated LoRA* beats cheaper payload encoders is not supported at this scale and must be reported as such.

## 11f. Payload-dependent classes (L7): a hashed encoder beats the transformer

**Why (owner):** "How did the payload-dependent classes do without payloads?" and "Are the results in our favour or against?"

Means over 2 seeds, R2 = 20, α = 0.5, p0. L7-F1 is the mean F1 over injection, password and XSS. Source: latest Comet run per name (`comet_latest.py`).

| Run | Injection | Password | XSS | **L7-F1** | Macro-F1 |
|---|---|---|---|---|---|
| A2 hashed n-grams | **0.909** | 0.983 | 0.966 | **0.952** | 0.852 |
| B3 centralised HetGNN | 0.936 | 0.984 | 0.936 | 0.952 | 0.878 |
| A4 homogeneous | 0.926 | 0.986 | 0.888 | 0.933 | 0.863 |
| A8 no graph | 0.872 | 0.989 | 0.938 | 0.933 | 0.848 |
| A5 no `shares_host` | 0.878 | 0.978 | 0.939 | 0.932 | 0.835 |
| **E1 transformer + LoRA** | **0.747** | 0.984 | 0.965 | **0.898** | 0.858 |
| A1 no payload | 0.755 | 0.944 | 0.971 | 0.890 | 0.826 |
| A3 frozen transformer | 0.721 | 0.937 | 0.937 | 0.865 | 0.844 |
| B1 FedAvg MLP | 0.658 | 0.878 | 0.955 | 0.830 | 0.815 |
| B5 XGBoost (central, 1 seed) | 0.929 | 0.928 | 0.979 | 0.945 | 0.865 |

- **The payload channel helps:** A1 → A2 raises L7-F1 from 0.890 to 0.952 and injection from 0.755 to 0.909.
- **The transformer + federated LoRA does not:** E1's injection F1 (0.747) is as low as no payload, and 0.16 below hashed n-grams. E1's macro-F1 edge comes from *normal* (0.83) and ransomware, not from the L7 classes.
- **Consequence for the thesis:** at this scale (195k flows, R1 = 5, weakly supervised Phase 1), the headline claim must be "payload content and host context help federated detection of application-layer attacks; a cheap hashed payload encoder is at least as good as a federated-LoRA transformer". The transformer result is a negative finding to report and discuss (likely causes: short Phase 1, flow-level weak labels on segments, only ~100k readable segments).

## 11g. Template overlap and timing probe

**Template overlap** (E0, [`template_overlap_report.csv`](experiment_log_outputs/e0_release_v2/template_overlap_report.csv)): the share of test payload templates that also occur in training.
- XSS 98 % (99 % of test flows), password 94 % (99.7 %), normal 57 % (96 %), **injection 9 % (34 %)**.
- XSS and password payloads are largely memorisable templates. Injection is the one class with mostly *novel* test payloads, so it is the real test of payload understanding. On injection, hashed n-grams (0.909) beat the transformer (0.747) by a wide margin.

**Timing probe** ([`probe_report.md`](experiment_log_outputs/probe/probe_report.md)) estimated 138 ms per training segment per Phase-1 round, which extrapolates to 14.3 GPU-h per E1 run. The measured cost was about 22 min per Phase-1 round, about 2 h per E1 run. The probe overestimated by about 7× (the per-segment cost does not scale linearly from 1.4k to 73k segments).

All latest per-run test metrics from Comet: [`comet/latest_test_metrics_all_runs.csv`](experiment_log_outputs/comet/latest_test_metrics_all_runs.csv) (script: [`comet_latest.py`](experiment_log_outputs/comet_latest.py)).

## 11h. E1 on three partition draws (α = 0.5, training seed 0)

**Why:** after §11d, the headline claim (E1 > federated baselines) had to hold beyond one lucky partition draw.

Runner B v6 ran p1 (112 min) and runner A v5 ran p2 (104 min). Phase 1 took 18–22 min per round.

| Partition | E1 | A2 hashed | B1 FedAvg MLP | E1 − A2 | E1 − B1 |
|---|---|---|---|---|---|
| p0 (main) | 0.840 | 0.842 | 0.810 | −0.2 | +3.0 |
| p1 | **0.881** | 0.842 | 0.694 | +3.9 | +18.7 |
| p2 | 0.615 | 0.611 | 0.621 | +0.4 | −0.6 |
| **mean** | **0.779** | 0.765 | 0.708 | **+1.4** | **+7.0** |

- **E1 beats the flow-only federated floor (B1) by 7.0 points on average**: clearly on p0 and p1, a tie on p2. This is the thesis's most robust headline claim.
- E1 vs. hashed payload (A2): +1.4 on average, driven by one draw (+3.9 on p1). The transformer adds a small, draw-dependent gain.
- p2 is a hard draw for every federated model (≈ 0.61–0.62). The partition draw matters as much as the model.
- Comet: E1 p1 and p2 are under `E1/toniot/psens_p1|psens_p2/...`.

## 11i. AGG and thesis results (Runner A v6, 09:30)

AGG indexed 58 runs, all completed. The outputs are in [`experiment_log_outputs/agg_results/`](experiment_log_outputs/agg_results/): main, baselines, ablations, per-class, missing sweep, non-IID, partition sensitivity, efficiency, and the LaTeX tables and PDF figures.

- **E4 (masking payloads at test time)**, E1 mean over seeds: 0 % → 0.858, 25 % → 0.849, 50 % → 0.826, **100 % → 0.507**. That is far below A1 (0.826, trained without payloads), because E1 learned to rely on payloads. The thesis states the "graceful degradation" claim only for ≤ 50 % masking.
- **E6 efficiency:** LoRA Δ has 540,672 parameters and the head W_c 7,690 (0.37 % of the 149.6 M encoder). Phase-1 upload is 2.19 MB per client and round (277× less than full fine-tuning). HetGNN θ has 303,498 parameters, uploading 1.21 MB per round. Phase 1 takes 110 min, embedding 10.2 min (43,561 flows), and Phase 2 116 s for 20 rounds. Latency is 21.7 ms per flow with the encoder and 0.06 ms for the GNN alone.
- **Why E1 is weak on injection** (confusion matrix): recall 0.91 but precision 0.64. On seed 0, 1,208 DDoS flows (HTTP floods with a readable payload) are predicted as injection. A2 labels them normal instead.
- **Convergence at R2 = 20:** the best Phase-2 round is 10 (seed 0) and 19 (seed 1); the best Phase-1 round is 4 or 5 of 5.

**Thesis updated** (fed-het-gnn-ids `7e98f56`): Results and Conclusions were rewritten on the measured numbers, all eight Chapter-7 figures were regenerated ([`thesis_figures/make_figs.py`](experiment_log_outputs/thesis_figures/make_figs.py), with inputs `thesis_numbers.json` and `fig_data.json`), the Abstract now carries the main findings, and B2/A7 are marked as not run. There are no `\PH` placeholders left in the chapters.

## 11j. A7 (Runner B v7, 2 seeds, about 3 min each)

Performance-weighted aggregation scores macro-F1 0.845 / 0.872, a mean of **0.858, identical to E1**. L7-F1 is 0.894 (−0.004) and the worst client 0.681. There is no benefit, so FedAvg is kept. Thesis updated (`A7` row, Setup note); B2 is the only planned run not done.

## 11k. NetFlow-style features vs. the 50 extractor features (Runner A v7, Runner B v8, 17:55–18:16)

**Why (owner):** "Don't take all flow features. Take only what NetFlow has, as in Fed_GNN, and see how results change with and without the extra features."

**Question (owner):** how much do the extra flow features add, when the models see only the fields NF-ToN-IoT offers (as in Fed_GNN)?
The NetFlow-8 set (`features.set=netflow`, commit `46ee603`) is in/out bytes, in/out packets, the OR of TCP flags, duration, protocol and destination port. Source port and L7_PROTO are left out. Same sprint split (the Phase-2 runs take 1–6 min each, and E1 reuses its Phase-1 cache). Two seeds each; B5 one. Full numbers: [`nf8_results.json`](experiment_log_outputs/nf8_results.json).

| Model | macro-F1 full → NF8 | Δ | L7-F1 full → NF8 | Δ |
|---|---|---|---|---|
| B5 XGBoost (central) | 0.865 → 0.871 | +0.006 | 0.945 → 0.940 | −0.005 |
| B1 FedAvg MLP | 0.815 → **0.741** | **−0.074** | 0.830 → 0.734 | −0.096 |
| A1 GNN, no payload | 0.826 → 0.830 | +0.004 | 0.890 → **0.828** | **−0.062** |
| A2 GNN + hashed payload | 0.852 → 0.871 | +0.019 | 0.952 → 0.926 | −0.026 |
| E1 GNN + transformer | 0.858 → 0.859 | +0.001 | 0.898 → 0.889 | −0.010 |

- Only the federated MLP needs the rich features overall (−7.4). The graph models keep their macro-F1, because host context (`shares_host`) makes up for the missing statistics.
- On the application-layer classes, the extra features do the payload's job. With NetFlow features, the GNN without payload loses 6.2 L7 points, but with payload nodes it loses only 1.0–2.6.
- **With NetFlow features the payload gain on L7 is much larger:** A1 → A2 goes from +6.2 to **+9.8**, and A1 → E1 from +0.8 to **+6.1**. So the thesis premise (flow-only data leave an application-layer gap that payloads fill) holds for NetFlow-level flow information. The 50 extractor features already close much of that gap.

## 11l. Why NetFlow-8 works here but not on NF-ToN-IoT: label inconsistency in NF-ToN-IoT

**Why (owner):** "When I ran XGBoost and RF on NF-ToN-IoT, accuracy, and especially the L7 classes, was not this good. How do the same 8 NetFlow features do so well here?" Follow-ups: "NF-ToN-IoT is popular and built from ToN-IoT, isn't it? Can we add the ToN-IoT payloads to the NF-ToN-IoT CSV that FedGATSage used?"

**Question (owner):** Fed_GNN's centralised XGBoost/RF on NF-ToN-IoT (Comet `fedgatsage-centralised`: macro-F1 0.50; XSS 0.30, password 0.38, scanning 0.13) failed on the L7 classes. Why do the same 8 NetFlow fields reach 0.87 on our flows?

**1. Label-collision analysis on our flows** (`toniot-collision`, [script](experiment_log_outputs/nf_label_audit/collision_analysis.py)): NetFlow-8 vector, duration in ms, all 3.99 M labelled flows. The majority-label oracle recalls XSS 99.6 %, password 99.6 %, scanning 99.9 % and DoS 99.8 %. The preliminary study found 0.0 %, 0.0 %, 31.2 % and 2.4 % on NF-ToN-IoT. Only 1.3 % of our XSS flows and 2.4 % of our injection flows are tiny (< 1 s, ≤ 3 packets).

**2. Matching NF-ToN-IoT v1 rows to our PCAP flows** (`nftoniot-payload-match`, [script](experiment_log_outputs/nf_label_audit/nftoniot_v1_match.py), [log](experiment_log_outputs/nf_label_audit/nftoniot_v1_match_log.txt)). Join on the same-direction 5-tuple, then fingerprint on exact in/out packets, bytes within 2 % and duration within 1 s.
- 78.6 % of the 1.38 M NF rows have a 5-tuple partner, and **36.0 % match uniquely**. Most matches come from `injection_normal1/4.pcap`.
- **NF label vs. ToN-IoT's own (Zeek) label on the same flow agrees only 58.5 %.** NF "xss" rows are 99.7 % Zeek-injection, NF "password" 94.9 % injection, NF "scanning" 68 % injection and 32 % DDoS, NF "ddos" 44 % injection, NF "Benign" 41 % DDoS.

**3. Payload evidence** (matched flows with a readable payload):

| NF label | SQLi signature | XSS signature | login form |
|---|---|---|---|
| injection | 52 % | 0 % | 23 % |
| xss | 48 % | **0 %** | 29 % |
| password | 48 % | 0 % | 40 % |
| scanning | 50 % | 0 % | 29 % |
| ddos | 48 % | 0 % | 49 % |

Under Zeek labels, DDoS flows carry 0 % SQLi and injection flows 52 %: the Zeek labels agree with the payloads, and the NF labels do not.

**Conclusion:** NF-ToN-IoT (v1) splits single SQL-injection sessions across the labels injection, xss, password, scanning and ddos, likely because it labels by attack-schedule time windows while several attacks ran from the same hosts at the same time. On this matched subset, the "impossible" XSS and password classes of the preliminary study come from **labels that are inconsistent with ToN-IoT's own labels and with the payloads**, not from a limit of NetFlow fields or of flow-level models. This also explains why flow-only models are strong on the original ToN-IoT labels. Attaching payloads to NF-ToN-IoT v1 rows is technically possible (36 % unique matches), but it would teach payloads to wrong labels, so it is not pursued.

## 11m. Rows where NF-ToN-IoT and ToN-IoT labels agree, and their payloads

**Why (owner):** "We cannot call a popular dataset false. How many rows match in both label sources, how many of those can get a payload, and what share for L7? Then we will know whether that data is enough to train on."

Same matching as §11l (unique fingerprint matches only).

| NF class | NF rows | matched | labels agree | agree % | agree + payload | payload % of agreed |
|---|---|---|---|---|---|---|
| injection | 468,539 | 189,360 | 185,537 | 98.0 | **157,781** | 85.0 |
| xss | 99,944 | 28,124 | **58** | 0.2 | **1** | – |
| password | 156,299 | 72,064 | **16** | 0.0 | **0** | – |
| scanning | 21,467 | 13,765 | 0 | 0.0 | 0 | – |
| dos | 17,717 | 8,377 | 0 | 0.0 | 0 | – |
| ddos | 326,345 | 110,756 | 61,553 | 55.6 | 9,407 | 15.3 |
| normal | 270,279 | 73,380 | 42,705 | 58.2 | 23,489 | 55.0 |
| backdoor | 17,247 | 244 | 244 | 100 | 0 | 0 |
| mitm | 1,295 | 482 | 404 | 83.8 | 11 | 2.7 |
| ransomware | 142 | 1 | 1 | 100 | 0 | 0 |
| **total** | 1,379,274 | 496,553 | **290,518** | 58.5 | **190,689** | 65.6 |

- **L7:** of 724,782 NF rows labelled injection, xss or password, 185,611 agree with ToN-IoT's label and 157,782 of those carry a payload. **Almost all are injection**: XSS has 58 agreeing rows (1 with payload) and password has 16 (0 with payload).
- A clean "NF-ToN-IoT ∩ ToN-IoT + payload" subset can therefore train injection, normal and DDoS, but **not XSS or password**. Their NF rows in our captures almost never carry the ToN-IoT label.
- **Caveat:** the matched rows come mostly from the two injection captures, and the XSS and password captures matched only a few hundred NF rows. NF-ToN-IoT's XSS and password rows from other captures (not in our release) may agree better. The result holds for the 18 captures we have.

## 12. Thesis repo sync

- PR #3 (new Dataset/Setup/Results chapters) was merged on GitHub. The E0 commit was rebased onto it (`d1e93f5`), and the E0 numbers were filled into `Tab_D_Stats` and "Outcome of the Data Audit". Still open: template overlap and probe timing.

---

## Still to run

| Runner | Queue |
|---|---|
| A (v4, running) | seed 1 of A1, A2, A4, A5, A6, A8, A9, B1, B3, A3, E4 |
| – | all planned runs done except B2 (owner: not needed) |

