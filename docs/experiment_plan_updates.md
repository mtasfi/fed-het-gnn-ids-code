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
