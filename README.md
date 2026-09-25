# FedHetFormer-IDS: experiment code

This is the code for the thesis *FedHetFormer-IDS: Federated Heterogeneous Graph Learning with Transformer-Derived Payload Embeddings for Intrusion Detection*. It implements Methodology v0.9 and Experiment Plan v1.2, with the v1.3 amendments in [docs/experiment_plan_updates.md](docs/experiment_plan_updates.md).

The pipeline has three stages:
- **Phase 1**: federated LoRA fine-tuning of ModernBERT-base on payload segments. The LoRA adapters Δ and the head W_c are aggregated with FedAvg.
- **Freeze and embed once**: the encoder is frozen and every readable payload segment is embedded one time.
- **Phase 2**: federated Hetero-GraphSAGE over per-client flow–payload graphs, with `contains`, `next` and `shares_host` edges.

Raw flows and payloads never leave the machine. Comet receives only metrics, config values and tags.

## Layout

```
configs/
  base.yaml                 locked settings (E0 fills the nulls: alpha, alpha_sweep, kappa)
  datasets/                 toniot (primary), cicids2017 (fallback), unsw_nb15 (audit only)
  encoders/                 modernbert (default), distilbert (fallback)
  experiments/              one file per experiment: question, done criterion, config diff vs base
fhf/
  common/                   config merge, run-directory contract, metrics, Comet mirror, utils
  data/                     sources, flow_extractor (PCAP), label_sources, pcap_match, payload_rules,
                            audit_payloads, make_splits, partition_clients, federated_scaler, store
  phase1/                   lora_model, federated_tune, run_phase1, embed_once, payload_head_eval (A9)
  phase2/                   build_heterograph, hetero_sage, train_local
  federated/                client, server (one round loop for every model), fedavg, weighted_aggregate (A7)
  baselines/                classical_xgb_rf (B5), fedgatsage/adapter (B2)
  pipeline.py               E1 and graph variants, federated MLP (B1/A8), centralized HetGNN (B3)
  missing_sweep.py (E4)     efficiency.py (E6)
experiments/
  run_e0.py                 E0 data gate (CPU)
  run_probe.py              timing probe (after E0, before the E1 queue)
  run.py                    every modeled experiment
  aggregate_results.py      rebuilds every table and figure from the run directories
tests/                      unit tests + end-to-end smoke test (synthetic data: tests only)
docs/                       has_payload rule, B2 rerun, B4 note, plan amendments
```

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q                   # ~3 min on CPU; the smoke test uses synthetic data and a tiny random encoder
```

## Data

The primary dataset is the **original ToN-IoT network data** (UNSW share: `TON_IoT datasets`). You need two parts of it:
- the raw PCAPs (`Raw_datasets/Network_dataset_pcaps/…`), which provide the flows and payloads;
- the processed network CSVs (`Processed_datasets/Processed_Network_dataset/Network_dataset_*.csv`), which provide the labels.

Do **not** use NF-ToN-IoT: it has no payload bytes and no absolute start time.

Where the files come from is set in config, so switching sources needs no code change:

```bash
# local folder (default: data/raw/toniot)
--set dataset.source.root=/path/to/TON_IoT
# attached Kaggle dataset
--set dataset.source.kind=kaggle --set dataset.source.root=/kaggle/input/<slug>
# kagglehub download
--set dataset.source.kind=kagglehub --set dataset.source.slug=<owner>/<dataset>
# plain HTTP files
--set dataset.source.kind=http --set 'dataset.source.urls=[https://…/a.pcap, …]'
```

If a mirror uses different folder names, change `pcap_globs` and `label_globs` in `configs/datasets/toniot.yaml`.

## Workflow

### 1. E0 data gate (CPU, no GPU)

```bash
python experiments/run_e0.py extract --dataset toniot     # PCAP -> flows + first 3 payload segments (resumable)
python experiments/run_e0.py labels  --dataset toniot
python experiments/run_e0.py offset  --dataset toniot     # if the peak is not 0: --set matching.clock_offset_s=<peak>
python experiments/run_e0.py match   --dataset toniot
python experiments/run_e0.py check   --dataset toniot     # signature vs label, coverage, failures, gate
python experiments/run_e0.py payload --dataset toniot     # label map + has_payload rule + audit
python experiments/run_e0.py split   --dataset toniot     # blocks, train/test, Dirichlet partitions, templates
python experiments/run_e0.py kappa   --dataset toniot
# or all steps at once: run_e0.py all
# smoke run on part of the data: extract --max-files 2 --max-packets 500000
```

The reports go to `work/toniot/e0/`:
- `dataset_audit.md`, `payload_availability_by_class.csv`, `label_mapping.yaml`, `has_payload_reason_counts.csv`
- `match_report.md`, `match_failures.csv`, `match_trace_sample.csv`
- `match_checks.md`, `match_gate.json`, `match_signature_crosstab.csv`, `match_class_coverage.csv`, `match_failure_breakdown.csv`
- `split_report.json`, `client_class_distribution_alpha=*.csv`, `leakage_log.json`
- `template_overlap_report.csv`, `kappa_stats.csv`

**E0 stop rule:** do not start GPU work if any of these hold: matching is unreliable (`match_gate.json` fails; thresholds in `matching.gate`), a key class has lost its payloads, labels are ambiguous, or κ makes the graph dense.

Then lock these values in `configs/base.yaml`:
- `partition.alpha`: the primary α;
- `partition.alpha_sweep`: two α values for E5;
- `graph.shares_host.kappa`;
- if needed, `dataset.exclude_classes` and `matching.clock_offset_s`.

Runners refuse to start while any of these is still null.

### 2. Timing probe

```bash
python experiments/run_probe.py --dataset toniot            # 1 round, 10k flows -> probe_report.md, probe_budget.csv
python experiments/run_probe.py --dataset toniot --encoder distilbert   # only if ModernBERT's Phase 1 is over budget
```

### 3. Experiments, in plan order

```bash
python experiments/run.py --exp E1 --dataset toniot         # seeds 0 and 1; runs Phase 1 and caches phi*
python experiments/run.py --exp E4 --dataset toniot         # missing-payload sweep on the E1 checkpoint
python experiments/run.py --exp A1 --dataset toniot
python experiments/run.py --exp A5 --dataset toniot
python experiments/run.py --exp B3 --dataset toniot
python experiments/run.py --exp B1 --dataset toniot
python experiments/run.py --exp B5 --dataset toniot         # 4 CPU variants
python experiments/run.py --exp B2 --dataset toniot --set baselines.fedgatsage.repo_path=/path/to/Fed_GNN
python experiments/run.py --exp A3 --dataset toniot
python experiments/run.py --exp A2 --dataset toniot
python experiments/run.py --exp E6 --dataset toniot
python experiments/run.py --exp A4 --dataset toniot
python experiments/run.py --exp E5 --dataset toniot         # loops partition.alpha_sweep
python experiments/run.py --exp A8 --dataset toniot
python experiments/run.py --exp A9 --dataset toniot
python experiments/run.py --exp A6 --dataset toniot
python experiments/run.py --exp A7 --dataset toniot         # optional
python experiments/aggregate_results.py --dataset toniot    # rebuilds all tables and figures (E7 included)
```

Useful flags:
- `--smoke`: a real-data subset with 1 Phase-1 round and 2 Phase-2 rounds. Results go to `runs/_smoke/` and are tagged `smoke`.
- `--resume`: continue an interrupted run from its last round checkpoint (for Kaggle's session limit).
- `--overwrite`: replace a completed run. Completed runs are otherwise immutable.
- `--set key=value`: override any config value; the full resolved config is saved in the run directory.

Ablations reuse the E1 embeddings of the same dataset, α and seed, and fail loudly if E1 has not run. Phase 1 is never rerun for them.

## Outputs

Each run writes `runs/{exp}/{dataset}/{variant}/alpha={a}/seed={s}/` (plan §5):
- `config.yaml` and `manifest.json` (git hash, dirty flag, input and output hashes, status, failure reason, hardware)
- `round_metrics.csv` (phase, round, scope, losses, validation macro-F1, bytes up and down, elapsed time)
- `test_metrics.csv`, `per_class_metrics.csv`, `per_client_metrics.csv`, `confusion_matrix.csv/.png`
- `resource_metrics.csv`, `predictions.parquet`, `checkpoint_or_pointer.txt`

Failed runs are kept, moved aside with their status and traceback.

`aggregate_results.py` writes `results/<dataset>/`:
- `results_main.csv`, `results_baselines.csv`, `results_ablations.csv` (with paired deltas against E1)
- `results_missing_sweep.csv`, `results_noniid.csv`, `results_perclass.csv`, `efficiency_runs.csv`
- `table_*.tex`
- `fig_convergence.pdf`, `fig_missing_rate.pdf`, `fig_perclass.pdf`, `fig_noniid.pdf`

## Comet

Project: `fedhetformer-ids`. The API key is in `configs/base.yaml`; turn Comet off with `--set tracking.comet=false`.

Every run gets these tags: `exp:<id>`, `variant:<name>`, `dataset:<name>`, `alpha:<a>`, `seed:<s>`, `pipeline:<name>`, `tier:<t>`, `sprint`, plus `smoke`, `resumed` and `status:<completed|failed>` where they apply. Filter on them in the Comet UI or API.

Comet is a mirror only; the CSVs are the record. Only scalar metrics, config values, tags and the numeric confusion matrix are sent. Stdout, code, git patches, environment details and files are never uploaded, so payload text and IP addresses cannot reach it.

## Kaggle / Colab notebooks

`notebooks/` has one notebook per experiment (`E0_data_gate.ipynb`, `E1_main.ipynb`, …, `AGG_aggregate_results.ipynb`). Each one has a `PLATFORM = "kaggle" | "colab"` switch and clones this private repo with `GITHUB_TOKEN`. Order, state sharing between the two platforms and resuming are described in [notebooks/README.md](notebooks/README.md). To regenerate them after a setup change, run `python tools/make_notebooks.py`.

## Notes

- Synthetic PCAPs and the tiny random encoder exist only for the tests. Never report results from them.
- Sprint numbers are directional. Paper-final numbers need the five-fold time-grouped CV described in the plan.
