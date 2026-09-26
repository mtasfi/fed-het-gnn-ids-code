# Release v3 = release v2 (18 captures) + release extra7 (7 XSS/password captures), no re-extraction.
# Runs in the toniot-build-dataset notebook (it has the HF_TOKEN secret). CPU only.
import json, os, shutil, subprocess, time
subprocess.run("pip install -q huggingface_hub pyarrow", shell=True)
import pandas as pd
from huggingface_hub import HfApi, snapshot_download
from kaggle_secrets import UserSecretsClient

TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
api = HfApi(token=TOKEN)
REPO, EXTRA = "mtasfi/toniot-fhf-processed", "mtasfi/toniot-fhf-processed-extra"
v2 = snapshot_download(REPO, repo_type="dataset", revision="v2", token=TOKEN)
ex = snapshot_download(EXTRA, repo_type="dataset", revision="extra7", token=TOKEN)
OUT = "/kaggle/working/release_v3"
os.makedirs(OUT, exist_ok=True)

f2, fx = pd.read_parquet(f"{v2}/flows.parquet"), pd.read_parquet(f"{ex}/flows.parquet")
print(f"v2 flows {len(f2):,} | extra7 flows {len(fx):,}")
assert set(f2.columns) == set(fx.columns), set(f2.columns) ^ set(fx.columns)
assert not (set(f2.flow_uid) & set(fx.flow_uid)), "flow_uid collision"

# the same packets captured in two files (overlapping captures) would give two identical flows
key = ["src_ip", "src_port", "dst_ip", "dst_port_id", "proto", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes"]
kf = lambda d: d[key].astype(str).agg("|".join, axis=1) + "|" + (d["first_ts"] * 1000).round().astype("int64").astype(str)
dup = kf(fx).isin(set(kf(f2)))
print(f"extra7 flows duplicating a v2 flow (same 5-tuple, counts and start ms): {int(dup.sum()):,}")
fx = fx[~dup]
flows = pd.concat([f2, fx[f2.columns]], ignore_index=True)
flows.to_parquet(f"{OUT}/flows.parquet", index=False)

p2, px = pd.read_parquet(f"{v2}/payloads.parquet"), pd.read_parquet(f"{ex}/payloads.parquet")
px = px[px.flow_uid.isin(set(fx.flow_uid))]
pay = pd.concat([p2, px[p2.columns]], ignore_index=True)
pay.to_parquet(f"{OUT}/payloads.parquet", index=False, compression="zstd")
print(f"v3: {len(flows):,} flows, {len(pay):,} payload segments")

shutil.copytree(f"{v2}/e0", f"{OUT}/e0", dirs_exist_ok=True)           # E0 decisions (alpha, kappa) come from v2
shutil.copytree(f"{v2}/e0", f"{OUT}/e0_v2", dirs_exist_ok=True)
shutil.copytree(f"{ex}/e0", f"{OUT}/e0_extra7", dirs_exist_ok=True)
pc = flows.groupby("label").agg(flows=("flow_uid", "size"), has_payload=("has_payload", "sum"))
pc["has_payload_rate"] = (pc.has_payload / pc.flows).round(4)
pc.reset_index().to_csv(f"{OUT}/payload_availability_by_class_v3.csv", index=False)
print(pc.to_string())
m2 = json.load(open(f"{v2}/manifest.json")); mx = json.load(open(f"{ex}/manifest.json"))
manifest = {"release": "v3", "built_from": {"v2": {"repo": REPO, "revision": "v2", "code_commit": m2.get("code_commit")},
                                            "extra7": {"repo": EXTRA, "revision": "extra7", "code_commit": mx.get("code_commit")}},
            "capture_files": sorted(set(m2["capture_files"]) | set(mx["capture_files"])),
            "label_files": sorted(set(m2["label_files"]) | set(mx["label_files"])),
            "flows": int(len(flows)), "payload_segments": int(len(pay)),
            "extra7_duplicates_dropped": int(dup.sum()),
            "splits": "none: runners rebuild them (experiments/run_e0.py split); thesis results use release v2",
            "per_class": pc.reset_index().to_dict(orient="records"), "built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2, default=str)
open(f"{OUT}/README.md", "w").write(f"""---
license: other
pretty_name: ToN-IoT processed flows and payloads (FedHetFormer-IDS)
---
# ToN-IoT processed flows and payloads

- tag `v2`: 18 captures, the thesis data (with splits/)
- tag `v3`: v2 + 7 XSS/password captures (`normal_XSS3/5/7/9`, `password_normal2/3/4`); {len(flows):,} flows,
  {len(pay):,} payload segments; no splits/ (rebuilt by the runners). E0 reports: `e0_v2/`, `e0_extra7/`.

`flows.parquet` (one row per labelled flow, key `flow_uid`) and `payloads.parquet` (one row per stored payload
segment, key `flow_uid`, `seg_pos`). Derived from the original ToN-IoT captures (UNSW Canberra); cite the ToN-IoT authors.
""")
# main branch = v3; v2 stays reachable by its tag. Remove v2's splits/ from main so v3 is not paired with stale splits.
files = api.list_repo_files(REPO, repo_type="dataset", revision="main")
stale = [f for f in files if f.startswith("splits/")]
api.upload_folder(folder_path=OUT, repo_id=REPO, repo_type="dataset", commit_message="release v3: v2 + 7 XSS/password captures",
                  delete_patterns=["splits/**"] if stale else None)
try:
    api.delete_tag(REPO, tag="v3", repo_type="dataset")
except Exception:
    pass
api.create_tag(REPO, tag="v3", repo_type="dataset")
print("pushed", REPO, "tag v3;", "v2 tag untouched")
