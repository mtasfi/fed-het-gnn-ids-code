# Flow-feature diagnostics (release v2, 2026-09-26)

Four quick checks on whether the flow-only models learn attack behaviour or shortcuts.
All use centrally trained XGBoost or scikit-learn trees on the release-v2 flows, with the E0 split
(60 s blocks, ≥ 8 blocks per class, seed 0): 142,742 training and 52,399 test flows. Trees use
balanced class weights, and XGBoost uses `n_estimators=300, max_depth=8, lr=0.1`. These are
exploration runs on Kaggle, not thesis runs.

## 1. Shortcut features: ports, TCP window, header bytes

| Features removed | # features | Macro-F1 |
|---|---|---|
| none | 50 | 0.860 |
| `dst_port`, `dst_port_wellknown` | 48 | 0.855 |
| + `fwd_init_win`, `bwd_init_win` | 46 | 0.865 |
| + `fwd_hdr_bytes`, `bwd_hdr_bytes` | 44 | 0.864 |

No class loses much when these features are removed. The most important features are behavioural:
`rst_ratio`, `syn_ratio`, `psh_cnt`, packet-length max/std and `fwd_bytes`. The TCP window matters
only for *normal* (0.73 → 0.67), which points to OS or device fingerprints, not to an attack signal.

## 2. Decision trees: can a few rules separate the classes?

| Depth | Macro | Backdoor | DoS | Scanning | Ransomware | Injection | Password | XSS |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.31 | 1.00 | 0.88 | 0.00 | 0.55 | 0.00 | 0.00 | 0.00 |
| 3 | 0.49 | 1.00 | 0.97 | 0.95 | 0.89 | 0.20 | 0.00 | 0.07 |
| 4 | 0.55 | 1.00 | 0.97 | 0.97 | 0.89 | 0.29 | 0.00 | 0.07 |
| 6 | 0.80 | 1.00 | 0.97 | 0.93 | 0.89 | 0.85 | 0.86 | 0.94 |
| 10 | 0.84 | 1.00 | 1.00 | 0.99 | 0.91 | 0.88 | 0.92 | 0.96 |

- **Volumetric classes** follow simple rules. Backdoor is caught by a single rule,
  `46 < fwd_bytes ≤ 50`: its flows are one fixed-size packet, a behaviour of this one tool that will
  not carry over to other backdoors. DoS and scanning are small SYN-only or RST flows, and ransomware
  shows RST together with large forward packets.
- **Application-layer classes** (injection, password, XSS) need depth ≥ 6. No single threshold
  separates them. The best depth-2 one-vs-rest rule reaches only F1 0.40 for injection, 0.42 for
  password and 0.79 for XSS.

## 3. NetFlow-style features vs. our 50

| Features | # | Macro-F1 | Injection | Password | XSS | Scanning |
|---|---|---|---|---|---|---|
| NF-ToN-IoT-like (protocol, port, bytes, packets, OR of flags, duration) | 8 | 0.864 | 0.898 | 0.982 | 0.953 | 0.985 |
| same without port | 7 | 0.871 | 0.886 | 0.986 | 0.974 | 0.981 |
| extractor features | 50 | 0.860 | 0.910 | 0.964 | 0.979 | 0.988 |

On the random-block split, the richer features add nothing: 7–8 NetFlow-like fields already reach
0.87. Our Fed_GNN reproduction logs found XSS, password and scanning inseparable on NF-ToN-IoT. That
finding does not hold for flows rebuilt from the captures.

## 4. Capture-held-out: generalisation to an unseen capture of the same class

For each attack class with at least 2 captures, one capture is held out as the test set, and training
uses every other capture, with at most 20k flows per class. The table gives the recall on the held-out
capture.

| Class | Held-out capture | Recall, NF-7 | Recall, 50 features |
|---|---|---|---|
| XSS | XSS4 / XSS6 | 0.97 / 0.99 | 0.98 / 0.99 |
| Password | pw1 / pw5 | 0.94 / 1.00 | 0.93 / 1.00 |
| DDoS | 1 / 12 / 4 | 0.89 / 0.99 / 0.99 | 0.90 / 1.00 / 0.99 |
| DoS | 1 / 5 | 0.93 / 1.00 | 0.95 / 1.00 |
| Injection | inj1 / inj4 | 0.94 / 0.80 | 0.95 / 0.81 |
| Scanning | scan1 / scan6 | **0.16 / 0.02** | **0.96 / 0.63** |

- Flow-only models generalise across captures for most classes, so template memorisation does not
  explain the high flow-only scores.
- The richer features matter for **scanning**: NF-7 misses the unseen scanning captures almost
  completely.
- **Injection** is the weakest application-layer class on an unseen capture (0.80). It is the class
  where the payload channel should help most, which fits the A1 → A2 gain of +10.8 F1 points.
- **Caveat:** the captures of one class overlap in time. For example, injection 1 and 4, and XSS 4
  and 6, come from the same campaign and tool. The held-out capture is a different session of the
  same attack, not an unseen attacker. ToN-IoT has one tool per attack, and this limits every
  evaluation on it (threats to validity).

## What this means for the thesis

- There is no IP, port or TCP-window leakage. The feature set stays unchanged.
- On ToN-IoT, flow statistics separate most classes. The payload channel is expected to pay off on
  **injection** (and partly **password**), mainly in the federated setting, where flow-only models are
  weaker (B1/A1 injection F1 0.59–0.63 vs. central XGBoost 0.93). Claims should be scoped to these
  classes.
