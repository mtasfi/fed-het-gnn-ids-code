# NF-ToN-IoT-FHF: NF-ToN-IoT v1 (X) relabelled with our PCAP/Zeek labels (Y = release v3) where a row matches.
# Rule: X row matches Y flows (same 5-tuple in NF orientation; packets exact, bytes within 2 %, duration within 1 s)
#   - one candidate, or several candidates that all carry the same label -> Attack := Y label, Label := 0/1 from it
#   - several candidates with different labels, or no candidate           -> row unchanged
# Output keeps X's columns, row order and values; only Attack and Label change. Runs on Kaggle CPU.
import os, subprocess, glob, shutil
subprocess.run("pip install -q huggingface_hub pyarrow", shell=True)
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
from kaggle_secrets import UserSecretsClient

TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
W = "/kaggle/working"
import urllib.request
nf_file = f"{W}/nf_src/NF-ToN-IoT.csv"; os.makedirs(os.path.dirname(nf_file), exist_ok=True)
urllib.request.urlretrieve("<signed link from Kaggle MCP download_dataset mtasfi/nftoniot>", nf_file)   # X = Kaggle dataset mtasfi/nftoniot (signed link, MCP download_dataset)
out_name = os.path.basename(nf_file)
raw = pd.read_csv(nf_file, dtype=str, keep_default_na=False)      # written back verbatim except Attack/Label
print(f"X = {out_name}: {len(raw):,} rows, columns {list(raw.columns)}")
print(raw.Attack.value_counts().to_string())

k = ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL"]
num = ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "IN_PKTS", "OUT_PKTS", "IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS"]
nf = raw[["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]].copy()
for c in num:
    nf[c] = pd.to_numeric(raw[c], errors="coerce")
nf["row"] = np.arange(len(raw))

fp = hf_hub_download("mtasfi/toniot-fhf-processed", "flows.parquet", repo_type="dataset", revision="v3", token=TOKEN)
fl = pd.read_parquet(fp, columns=["src_ip", "src_port", "dst_ip", "dst_port_id", "proto",
                                  "fwd_bytes", "bwd_bytes", "fwd_pkts", "bwd_pkts", "duration", "label"])
print(f"Y = release v3: {len(fl):,} labelled flows")
pnum = {"tcp": 6, "udp": 17, "icmp": 1}
fl["PROTOCOL"] = pd.to_numeric(fl.proto.map(lambda p: pnum.get(str(p).lower(), p)), errors="coerce")
a = fl.rename(columns={"src_ip": "IPV4_SRC_ADDR", "src_port": "L4_SRC_PORT", "dst_ip": "IPV4_DST_ADDR", "dst_port_id": "L4_DST_PORT"})
for c in ["L4_SRC_PORT", "L4_DST_PORT"]:
    a[c] = pd.to_numeric(a[c], errors="coerce")
a = a[k + ["fwd_bytes", "bwd_bytes", "fwd_pkts", "bwd_pkts", "duration", "label"]]

j = nf.merge(a, on=k, how="inner")
ok = (j.IN_PKTS == j.fwd_pkts) & (j.OUT_PKTS == j.bwd_pkts) & \
     ((j.IN_BYTES - j.fwd_bytes).abs() <= 0.02 * j.IN_BYTES.clip(lower=1) + 4) & \
     ((j.OUT_BYTES - j.bwd_bytes).abs() <= 0.02 * j.OUT_BYTES.clip(lower=1) + 4) & \
     ((j.FLOW_DURATION_MILLISECONDS / 1000 - j.duration).abs() <= 1.0)
jm = j.loc[ok, ["row", "label"]]
g = jm.groupby("row").label.agg(["nunique", "first", "size"])
use = g[g["nunique"] == 1]
print(f"\nX rows with >=1 fingerprint match: {len(g):,}; single-label (relabelled): {len(use):,} "
      f"(one candidate {int((use['size'] == 1).sum()):,}, several same-label {int((use['size'] > 1).sum()):,}); "
      f"conflicting candidates, left unchanged: {int((g['nunique'] > 1).sum()):,}")

new = use["first"].replace({"normal": "Benign"})
out = raw.copy()
before = out.Attack.copy()
out.loc[new.index, "Attack"] = new.values
out.loc[new.index, "Label"] = np.where(new.values == "Benign", "0", "1")
changed = out.Attack != before
print(f"Attack changed on {int(changed.sum()):,} rows ({100 * changed.mean():.1f}%)")
print("\n===== old Attack (rows) x new Attack (cols), changed rows =====")
print(pd.crosstab(before[changed], out.Attack[changed]).to_string())
cmp = pd.DataFrame({"before": before.value_counts(), "after": out.Attack.value_counts()}).fillna(0).astype(int)
cmp["delta"] = cmp.after - cmp.before
print("\n===== class counts =====")
print(cmp.sort_values("after", ascending=False).to_string())
print("\nbinary Label:", raw.Label.value_counts().to_dict(), "->", out.Label.value_counts().to_dict())

os.makedirs(f"{W}/nftoniot-fhf", exist_ok=True)
out.to_csv(f"{W}/nftoniot-fhf/{out_name}", index=False)
shutil.rmtree(f"{W}/nf_src")
print("wrote", f"{W}/nftoniot-fhf/{out_name}", f"{os.path.getsize(f'{W}/nftoniot-fhf/{out_name}') / 1e6:.0f} MB")
