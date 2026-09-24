# B2: FedGATSage rerun on the E1 split

**Default: rerun.** The earlier FedGATSage reproduction was run on NF-ToN-IoT. Those flows have no absolute timestamps, so they cannot be aligned with our PCAP-derived flows. That run stays in the thesis as the **paper-faithful reproduction** (motivation evidence: weak per-class results on password and XSS). The head-to-head number in the main table comes from this rerun. It uses the same dataset, split, client partition, label map and seed as E1.

You may reuse the old run only if E0 shows that its flows can be matched one-to-one to our flows. That is not expected.

## How it runs

`fhf/baselines/fedgatsage/adapter.py`:
1. Exports the E1 split into the layout the reproduction code expects: `data/{temporal,content,behavioral}_detector/client_k.csv` (each client's training rows), `val.csv`, `test.csv`, and `label_mapper.json` (our label order).
2. Imports the reproduction code from `baselines.fedgatsage.repo_path` (a `mtasfi/Fed_GNN` checkout). The code is not modified; its commit hash is recorded in the run manifest.
3. Trains with FedGATSage's own loop (client GATs, community abstraction, server GraphSAGE) for `num_rounds` (default 15, the paper's schedule), using `community: louvain`.
4. Fits its Random-Forest ensemble on detector probabilities plus its raw features, then scores **our full test set** with **our** metric code.

## Column mapping (our extractor → FedGATSage input)

| FedGATSage column | our flow field |
|---|---|
| Src IP / Dst IP / Src Port / Dst Port / Protocol | initiator IP / responder IP / ports / IP protocol |
| Flow Duration (µs) | `duration` × 1e6 |
| Tot Fwd Pkts / Tot Bwd Pkts | `fwd_pkts` / `bwd_pkts` |
| **TotLen Fwd Pkts / TotLen Bwd Pkts** (its payload statistics) | `fwd_payload_bytes` / `bwd_payload_bytes` |
| Flow IAT Mean / Std (µs), Flow Pkts/s | `iat_mean`, `iat_std` × 1e6, `pkts_per_s` |
| SYN / RST / ACK Flag Cnt | `syn_cnt` / `rst_cnt` / `ack_cnt` |
| IN_BYTES / OUT_BYTES / IN_PKTS / OUT_PKTS / FLOW_DURATION_MILLISECONDS / L4_DST_PORT | forward / backward bytes and packets, duration × 1e3, destination port |
| Attack | canonical E1 label |

FedGATSage uses IP endpoints as graph nodes. That is part of its design and is kept. It does not change our "no IP feature" rule, which applies to our model.

## Deviations from the original script (also in `manifest.json`)

1. Input flows, split, partition and label map are E1's (PCAP-extracted), not NF-ToN-IoT rows.
2. Columns are mapped as in the table above. Payload statistics come from our PCAP extractor.
3. `class_weights.pt` is computed from training rows only. The original preprocessing used the full dataset, including test rows.
4. The ensemble Random Forest is fitted on the pooled client **validation** flows and scored on the full test set. The original fitted it on 50% of the test set and scored the other 50%.
5. There is no checkpoint selection, because the original has none. The model after the last round is evaluated.
6. Worst-client macro-F1 groups the test predictions by the E1 client assignment.

## Requirements

The FedGATSage code needs `networkx python-louvain leidenalg igraph joblib` plus **`pyg-lib`** (for its `LinkNeighborLoader`). On Kaggle, install it with:
`pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`
(match the installed torch/CUDA versions).

Budget: about 2 GPU-h.
