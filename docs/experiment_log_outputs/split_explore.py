# Split feasibility on the v1 release (exploration only; nothing here is locked or saved as a split).
import glob, itertools, json, os, subprocess, sys
import pandas as pd

CODE = "/kaggle/working/fed-het-gnn-ids-code"
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/mtasfi/fed-het-gnn-ids-code.git", CODE], check=True)
sys.path.insert(0, CODE)
os.chdir(CODE)
subprocess.run("pip install -q pyarrow pyyaml", shell=True)

from fhf.common.config import load_config
from fhf.data.make_splits import make_split
from fhf.data.partition_clients import partition

import urllib.request
rel = "/kaggle/working/flows.parquet"
urllib.request.urlretrieve("<signed link to toniot-build-dataset output: release_vN/flows.parquet>", rel)   # signed link to toniot-build-dataset v9 output
df = pd.read_parquet(rel, columns=["flow_uid", "capture_id", "capture_file", "first_ts", "label"])
print(f"{len(df):,} flows from {rel}")
print(df["label"].value_counts().to_string())

# time span per capture file and label mix
g = df.groupby("capture_file").agg(flows=("flow_uid", "size"), t0=("first_ts", "min"), t1=("first_ts", "max"))
g["minutes"] = ((g.t1 - g.t0) / 60).round(1)
g["top_labels"] = df.groupby("capture_file")["label"].agg(lambda s: dict(s.value_counts().head(3)))
g["t0"] = pd.to_datetime(g.t0, unit="s")
print(g.sort_values("t0").to_string())

rows = []
for blocks, key, target in itertools.product([300, 120, 60, 30], ["capture_id", "capture_file"], [75000, 150000]):
    cfg = load_config("toniot", overrides={"split.block_seconds": blocks, "split.target_flows": target})
    d = df.copy()
    d["capture_id"] = d[key].astype(str)
    split, rep = make_split(d, cfg, seed=0)
    parts = {}
    for alpha in [0.5, "iid"]:
        try:
            p, _ = partition(split, int(cfg.partition.num_clients), alpha, float(cfg.split.val_fraction), seed=0)
            per = p[p.role == "train"].groupby("label")["client"].nunique()
            parts[str(alpha)] = f"ok; clients/class min={int(per.min())}"
        except RuntimeError as e:
            parts[str(alpha)] = "FAIL"
    cc = rep["class_counts"]
    rows.append({"block_s": blocks, "block_key": key, "target": target, "flows": rep["flows_selected"],
                 "blocks": rep["blocks_selected"], "split_ok": rep["split_ok"], "n_problems": len(rep["problems"]),
                 "partition_0.5": parts["0.5"], "partition_iid": parts["iid"],
                 "min_train": min(v.get("train", 0) for v in cc.values()),
                 "min_test": min(v.get("test", 0) for v in cc.values()),
                 "problems": "; ".join(rep["problems"])[:300]})
    print(rows[-1], flush=True)

res = pd.DataFrame(rows)
res.to_csv("/kaggle/working/split_explore.csv", index=False)
print("\n===== SUMMARY =====")
print(res.drop(columns=["problems"]).to_string())
