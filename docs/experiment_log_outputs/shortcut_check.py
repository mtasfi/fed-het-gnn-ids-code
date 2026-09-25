# Shortcut check (exploration, not a thesis run): does flow-only XGBoost lean on ports / TCP window?
import os, subprocess, sys, urllib.request, time
import numpy as np, pandas as pd
CODE = "/kaggle/working/fed-het-gnn-ids-code"
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/mtasfi/fed-het-gnn-ids-code.git", CODE], check=True)
sys.path.insert(0, CODE); os.chdir(CODE)
from fhf.common.config import load_config
from fhf.data.make_splits import make_split
from fhf.data.flow_extractor import FEATURE_COLUMNS
from sklearn.metrics import f1_score
import xgboost as xgb

urllib.request.urlretrieve(FLOWS_URL, "/kaggle/working/flows.parquet")
df = pd.read_parquet("/kaggle/working/flows.parquet", columns=["flow_uid", "capture_id", "first_ts", "label"] + FEATURE_COLUMNS)
cfg = load_config("toniot")
split, rep = make_split(df[["flow_uid", "capture_id", "first_ts", "label"]], cfg, seed=0)
d = split[["flow_uid", "split"]].merge(df, on="flow_uid")
classes = sorted(d.label.unique()); idx = {c: i for i, c in enumerate(classes)}
y = d.label.map(idx).to_numpy(); tr = (d.split == "train").to_numpy(); te = ~tr
print(f"{len(d):,} flows (train {tr.sum():,}, test {te.sum():,}), split_ok={rep['split_ok']}", flush=True)

drop = {
    "a_all": [],
    "b_no_port": ["dst_port", "dst_port_wellknown"],
    "c_no_port_win": ["dst_port", "dst_port_wellknown", "fwd_init_win", "bwd_init_win"],
    "d_no_port_win_hdr": ["dst_port", "dst_port_wellknown", "fwd_init_win", "bwd_init_win", "fwd_hdr_bytes", "bwd_hdr_bytes"],
}
rows, per = [], {}
w = pd.Series(y[tr]).map(1.0 / pd.Series(y[tr]).value_counts()).to_numpy()
for name, dr in drop.items():
    cols = [c for c in FEATURE_COLUMNS if c not in dr]
    t = time.time()
    m = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=0)
    X = lambda mask: np.log1p(np.clip(d.loc[mask, cols].to_numpy(float), 0, None))
    m.fit(X(tr), y[tr], sample_weight=w)
    f = f1_score(y[te], m.predict(X(te)), average=None, labels=range(len(classes)), zero_division=0)
    per[name] = f
    rows.append({"set": name, "n_feat": len(cols), "macro_f1": round(float(f.mean()), 4), "minutes": round((time.time() - t) / 60, 1)})
    imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False).head(10)
    print(f"\n== {name}: macro-F1 {f.mean():.4f}  top: " + ", ".join(f"{k}={v:.3f}" for k, v in imp.items()), flush=True)
print("\n===== SUMMARY =====")
print(pd.DataFrame(rows).to_string(index=False))
print("\nper-class F1")
print(pd.DataFrame(per, index=classes).round(3).to_string())
pd.DataFrame(per, index=classes).to_csv("/kaggle/working/shortcut_check.csv")
