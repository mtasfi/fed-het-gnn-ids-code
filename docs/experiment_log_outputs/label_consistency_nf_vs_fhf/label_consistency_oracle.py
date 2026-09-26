# Label consistency of NF-ToN-IoT v1 vs NF-ToN-IoT-FHF: predict each row's label as the majority label of all rows
# sharing the same key (in-sample upper bound; no model). Keys: 9 NetFlow fields, src>dst IP pair, both.
# Inputs: the full CSVs (Kaggle mtasfi/nftoniot, mtasfi/nftoniot-fhf) and the 1500-per-label sets of §11p.
import pandas as pd
feat = ["L4_DST_PORT", "PROTOCOL", "L7_PROTO", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS"]
def oracle(df, key):
    maj = df.groupby(key).Attack.transform(lambda s: s.value_counts().index[0])
    ok = maj == df.Attack
    return {"all": ok.mean(), **{c: ok[df.Attack == c].mean() for c in ["injection", "password", "xss", "scanning", "ddos", "dos", "Benign"] if (df.Attack == c).any()}}
srcs = {"NF full": "nforig/NF-ToN-IoT.csv", "FHF full": "nfout/nftoniot-fhf/NF-ToN-IoT.csv",
        "NF 1500": "fgsout/sets/nftoniot_bal1500.csv", "FHF 1500": "fgsout/sets/nftoniot_fhf_bal1500.csv"}
rows = []
for n, p in srcs.items():
    df = pd.read_csv(p)
    df["pair"] = df.IPV4_SRC_ADDR + ">" + df.IPV4_DST_ADDR
    rows += [{"data": n, "key": "flow vector (9 NetFlow fields)", **oracle(df, feat)},
             {"data": n, "key": "IP pair", **oracle(df, "pair")},
             {"data": n, "key": "flow vector + IP pair", **oracle(df, feat + ["pair"])}]
t = pd.DataFrame(rows).set_index(["key", "data"]).sort_index().round(3)
print(t.to_string()); t.to_csv("label_consistency_oracle.csv")
