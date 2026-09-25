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


# Variant (NOT in the code): the rare-class floor also requires >= MIN_BLOCKS blocks per class,
# adding random unselected blocks that contain the class. Everything else is fhf.data.make_splits.
import numpy as np
from fhf.data import make_splits as ms
_orig = ms.select_blocks
def select_blocks_minblocks(df, target, floor, rng, min_blocks=4):
    selected, rep = _orig(df, target, floor, rng)
    bc = ms._block_class_counts(df)
    for cls in bc.columns:
        have = [b for b in selected if bc.loc[b, cls] > 0]
        cand = [b for b in rng.permutation(bc.index[bc[cls] > 0].to_numpy()) if b not in selected]
        while len(have) < min_blocks and cand:
            b = cand.pop(); selected.add(b); have.append(b)
        rep.setdefault(cls, {})["blocks_with_class"] = len(have)
    return selected, rep

rows = []
for blocks, key, target, mb in itertools.product([60, 30], ["capture_id", "capture_file"], [75000, 150000], [4, 8]):
    ms.select_blocks = lambda df, t, f, rng, mb=mb: select_blocks_minblocks(df, t, f, rng, mb)
    cfg = load_config("toniot", overrides={"split.block_seconds": blocks, "split.target_flows": target})
    d = df.copy(); d["capture_id"] = d[key].astype(str)
    split, rep = ms.make_split(d, cfg, seed=0)
    parts = {}
    for alpha in [0.5, 0.3, "iid"]:
        try:
            pt, _ = partition(split, int(cfg.partition.num_clients), alpha, float(cfg.split.val_fraction), seed=0)
            per = pt[pt.role == "train"].groupby("label")["client"].nunique()
            parts[str(alpha)] = f"ok min_clients={int(per.min())}"
        except RuntimeError:
            parts[str(alpha)] = "FAIL"
    cc = rep["class_counts"]
    rows.append({"block_s": blocks, "key": key, "target": target, "min_blocks": mb, "flows": rep["flows_selected"],
                 "split_ok": rep["split_ok"], **{f"part_{k}": v for k, v in parts.items()},
                 "min_train": min(v.get("train", 0) for v in cc.values()),
                 "min_test": min(v.get("test", 0) for v in cc.values()),
                 "problems": "; ".join(rep["problems"])[:200]})
    print(rows[-1], flush=True)
res = pd.DataFrame(rows); res.to_csv("/kaggle/working/split_explore_minblocks.csv", index=False)
print("\n===== SUMMARY =====")
print(res.drop(columns=["problems"]).to_string())
ok = res[res.split_ok]
if len(ok):
    b = ok.iloc[0]
    ms.select_blocks = lambda df, t, f, rng, mb=int(b.min_blocks): select_blocks_minblocks(df, t, f, rng, mb)
    cfg = load_config("toniot", overrides={"split.block_seconds": int(b.block_s), "split.target_flows": int(b.target)})
    d = df.copy(); d["capture_id"] = d[b.key].astype(str)
    split, rep = ms.make_split(d, cfg, seed=0)
    print("\nclass counts for", dict(b[["block_s", "key", "target", "min_blocks"]]))
    print(pd.DataFrame(rep["class_counts"]).T.to_string())
