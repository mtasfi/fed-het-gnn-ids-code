FLOWS_URL = "<signed link to toniot-build-dataset release_v2 file>"
PAYLOADS_URL = "<signed link to toniot-build-dataset release_v2 file>"

# Feasibility: can NF-ToN-IoT (v1) rows be matched to our PCAP flows (release v2) to attach payloads?
import glob, urllib.request, numpy as np, pandas as pd
print(glob.glob("/kaggle/input/**/*.csv", recursive=True)[:10])
nf_path = "/kaggle/working/NF-ToN-IoT.csv"
urllib.request.urlretrieve("<signed link from Kaggle MCP download_dataset mtasfi/nftoniot>", nf_path)   # signed link (MCP download_dataset)
nf = pd.read_csv(nf_path)
print(f"NF-ToN-IoT v1: {len(nf):,} rows"); print(nf.Attack.value_counts().to_string())
urllib.request.urlretrieve(FLOWS_URL, "/kaggle/working/flows.parquet")
urllib.request.urlretrieve(PAYLOADS_URL, "/kaggle/working/payloads.parquet")
fl = pd.read_parquet("/kaggle/working/flows.parquet", columns=["flow_uid","src_ip","src_port","dst_ip","dst_port_id","proto",
        "fwd_bytes","bwd_bytes","fwd_pkts","bwd_pkts","duration","has_payload","label","capture_file"])
print(f"our labelled flows: {len(fl):,}")
pnum = {"tcp":6,"udp":17,"icmp":1}
fl["protocol"] = fl["proto"].map(lambda p: pnum.get(str(p).lower(), p)).astype("int64", errors="ignore")
# exact 5-tuple join in the NF orientation (NF src = our initiator), then the reverse orientation
k = ["IPV4_SRC_ADDR","L4_SRC_PORT","IPV4_DST_ADDR","L4_DST_PORT","PROTOCOL"]
a = fl.rename(columns={"src_ip":"IPV4_SRC_ADDR","src_port":"L4_SRC_PORT","dst_ip":"IPV4_DST_ADDR","dst_port_id":"L4_DST_PORT","protocol":"PROTOCOL"})
for c in ["L4_SRC_PORT","L4_DST_PORT","PROTOCOL"]:
    a[c] = pd.to_numeric(a[c], errors="coerce"); nf[c] = pd.to_numeric(nf[c], errors="coerce")
nf["row"] = np.arange(len(nf))
j = nf.merge(a, on=k, how="inner")
print(f"\nNF rows with a same-direction 5-tuple partner: {j.row.nunique():,} ({100*j.row.nunique()/len(nf):.1f}%)")
# fingerprint: packets exact, bytes within 2 %, duration within 1 s
fp = (j.IN_PKTS == j.fwd_pkts) & (j.OUT_PKTS == j.bwd_pkts) & \
     ((j.IN_BYTES - j.fwd_bytes).abs() <= 0.02 * j.IN_BYTES.clip(lower=1) + 4) & \
     ((j.OUT_BYTES - j.bwd_bytes).abs() <= 0.02 * j.OUT_BYTES.clip(lower=1) + 4) & \
     ((j.FLOW_DURATION_MILLISECONDS / 1000 - j.duration).abs() <= 1.0)
jm = j[fp]
cand = jm.groupby("row").size()
uniq = cand[cand == 1].index
print(f"fingerprint-matched NF rows: {cand.size:,} ({100*cand.size/len(nf):.1f}%), unique: {uniq.size:,} ({100*uniq.size/len(nf):.1f}%)")
m = jm[jm.row.isin(uniq)]
agree = (m.Attack.str.lower().replace({"benign":"normal"}) == m.label)
print(f"label agreement (NF Attack vs our Zeek label) on unique matches: {100*agree.mean():.1f}%")
per = nf.groupby("Attack").size().rename("nf_rows").to_frame()
per["matched_unique"] = m.groupby("Attack").size()
per["with_payload"] = m[m.has_payload == 1].groupby("Attack").size()
per = per.fillna(0).astype(int)
per["match_%"] = (100*per.matched_unique/per.nf_rows).round(1)
per["payload_%_of_nf"] = (100*per.with_payload/per.nf_rows).round(1)
print("\n===== PER NF CLASS ====="); print(per.to_string())
print("\ncaptures of the matched rows:"); print(m.capture_file.value_counts().head(20).to_string())
m[["row","flow_uid","Attack","label","has_payload"]].to_parquet("/kaggle/working/nf_v1_match.parquet", index=False)

print("\n===== CROSSTAB: NF-ToN-IoT label (rows) x our Zeek label (cols), unique matches =====")
ct = pd.crosstab(m.Attack, m.label)
print(ct.to_string())
print("\n===== row-normalised % =====")
print((100*ct.div(ct.sum(axis=1), axis=0)).round(1).to_string())
print("\n===== NF label x capture (top) =====")
print(pd.crosstab(m.Attack, m.capture_file).T.sort_values("xss", ascending=False).head(8).to_string())

# Payload evidence: which attack signature do the matched flows' payloads carry, by NF label?
import re
pl = pd.read_parquet("/kaggle/working/payloads.parquet", columns=["flow_uid","text"]).dropna(subset=["text"])
pl = pl[pl.flow_uid.isin(set(m.flow_uid))]
txt = pl.groupby("flow_uid")["text"].apply(lambda s: " ".join(s))
SQLI = re.compile(r"union(\s|\+|%20)+select|select(\s|\+|%20).+from|'(\s|\+|%20)*or(\s|\+|%20)*'?\d|sleep\(|benchmark\(|information_schema|%27", re.I)
XSS = re.compile(r"<script|%3cscript|javascript:|onerror\s*=|onload\s*=|alert\(|%3csvg|<svg", re.I)
LOGIN = re.compile(r"(user(name)?|login|pass(word)?|pwd)=", re.I)
sig = pd.DataFrame({"flow_uid": txt.index,
                    "sqli": txt.str.contains(SQLI).values, "xss_sig": txt.str.contains(XSS).values,
                    "login_form": txt.str.contains(LOGIN).values})
mm = m.merge(sig, on="flow_uid", how="inner")
print("\n===== PAYLOAD SIGNATURES of matched flows with readable payload, by NF label (%) =====")
t = mm.groupby("Attack")[["sqli","xss_sig","login_form"]].mean().mul(100).round(1)
t["flows"] = mm.groupby("Attack").size()
print(t.to_string())
print("\n===== same, by our Zeek label (%) =====")
t2 = mm.groupby("label")[["sqli","xss_sig","login_form"]].mean().mul(100).round(1)
t2["flows"] = mm.groupby("label").size()
print(t2.to_string())
