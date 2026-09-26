FLOWS_URL = "<signed link to toniot-build-dataset release_v2/flows.parquet>"

# Label-collision analysis (as in the preliminary study on NF-ToN-IoT) on our release-v2 flows.
import urllib.request, numpy as np, pandas as pd
urllib.request.urlretrieve(FLOWS_URL, "/kaggle/working/flows.parquet")
cols = ["label","fwd_bytes","bwd_bytes","fwd_pkts","bwd_pkts","duration","proto_tcp","proto_udp","proto_icmp","dst_port",
        "fin_cnt","syn_cnt","rst_cnt","psh_cnt","ack_cnt","urg_cnt","end_reason","first_ts","capture_file"]
df = pd.read_parquet("/kaggle/working/flows.parquet", columns=[c for c in cols])
print(f"{len(df):,} labelled flows")
bits = {"fin_cnt":1,"syn_cnt":2,"rst_cnt":4,"psh_cnt":8,"ack_cnt":16,"urg_cnt":32}
nf = pd.DataFrame({
  "in_bytes": df.fwd_bytes.astype(np.int64), "out_bytes": df.bwd_bytes.astype(np.int64),
  "in_pkts": df.fwd_pkts.astype(np.int64), "out_pkts": df.bwd_pkts.astype(np.int64),
  "tcp_flags": sum((df[c] > 0).astype(np.int64) * b for c, b in bits.items()),
  "duration_ms": (df.duration * 1000).round().astype(np.int64),
  "protocol": (df.proto_tcp * 6 + df.proto_udp * 17 + df.proto_icmp * 1).astype(np.int64),
  "dst_port": df.dst_port.astype(np.int64)})
key = pd.util.hash_pandas_object(nf, index=False)
g = pd.DataFrame({"key": key.values, "label": df.label.values})
nlab = g.groupby("key")["label"].nunique()
g["collides"] = g["key"].map(nlab) > 1
maj = g.groupby(["key","label"]).size().reset_index(name="n").sort_values("n", ascending=False).drop_duplicates("key").set_index("key")["label"]
g["oracle"] = g["key"].map(maj)
res = g.groupby("label").agg(flows=("key","size"), collision_share=("collides","mean"),
                             oracle_recall=("oracle", lambda s: 0))
res["oracle_recall"] = g.assign(ok=g.oracle == g.label).groupby("label")["ok"].mean()
res["distinct_vectors"] = g.groupby("label")["key"].nunique()
print("\n===== ALL LABELLED FLOWS (NF-8, duration in ms) =====")
print((res.assign(collision_share=lambda r: (100*r.collision_share).round(1), oracle_recall=lambda r: (100*r.oracle_recall).round(1))).to_string())
# which class does each class lose to (majority of its colliding vectors)
lose = g[g.oracle != g.label].groupby(["label","oracle"]).size().rename("n").reset_index()
lose["share"] = lose.groupby("label")["n"].transform(lambda x: x / x.sum())
print("\n===== MAIN CONFUSER per class (oracle mistakes) =====")
print(lose.sort_values(["label","n"], ascending=[True,False]).groupby("label").head(2).round(3).to_string(index=False))
# short flows: share of sub-second, 1-2 packet flows per class
short = (df.duration < 1) & (df.fwd_pkts + df.bwd_pkts <= 3)
print("\n===== share of tiny flows (<1 s, <=3 packets) per class =====")
print((100*short.groupby(df.label).mean()).round(1).to_string())
