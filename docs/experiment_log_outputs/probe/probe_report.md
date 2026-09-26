# Timing probe

Encoder: answerdotai/ModernBERT-base; probe flows: 9,999 (train segments 1,402); full sprint flows 195,141 (train segments 73,454)

- Phase 1, one round: 193.8 s  ->  138.21 ms per training segment per round
- Embedding: 12.5 s  ->  5.68 ms per segment
- Phase 2, one round (E2=2): 0.3 s  ->  0.030 ms per flow per round
- Estimated single E1 run at R1=5, R2=10: 14.28 GPU-h

| item                                            |   runs |   est_gpu_h |
|:------------------------------------------------|-------:|------------:|
| E1 main, 2 seeds                                |      2 |       28.55 |
| E5 alpha sweep (2 alphas, full pipeline)        |      2 |       28.55 |
| A1, A4, A5, A6, A7 (Phase 2 only)               |      5 |        0.08 |
| A3 frozen encoder (embedding + Phase 2)         |      1 |        0.18 |
| A2 hashed n-grams (CPU transform + Phase 2)     |      1 |        0.02 |
| B1, A8 federated MLP                            |      2 |        0.01 |
| B3 centralized HetGNN                           |      1 |        0.02 |
| B2 FedGATSage rerun (plan estimate, not probed) |      1 |        2    |
| E4 sweep, A9, E6 (inference only)               |      3 |        0.3  |
| TOTAL                                           |     18 |       59.71 |

Check the Phase-1 share against the 12 h session limit; if it does not fit, use --encoder distilbert.