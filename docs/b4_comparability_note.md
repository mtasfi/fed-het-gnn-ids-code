# B4: FedLLM-IDS-style comparison (status: not_run)

B4 is optional stretch work (plan v1.2). It is not implemented in this sprint, so it is recorded as `not_run` in `results_baselines.csv`.

Comparability note: FedLLM-IDS (Basheer et al., 2026) fuses LLM-derived semantic embeddings with graph embeddings in a Transformer fusion detector, and it adds differential privacy. A faithful rerun would need three things: its fusion architecture, its DP accountant settings, and its feature pipeline, all adapted to our PCAP-extracted flows. Until that exists, the thesis compares against it qualitatively (Related Work) and does not report numbers.
