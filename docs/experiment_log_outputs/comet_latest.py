import json, urllib.request, csv, sys
K = [l.split()[1] for l in open('/Users/mtasfi/research/fed-het-gnn-ids-code/configs/base.yaml') if l.strip().startswith('api_key:')][0]
get = lambda u: json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"Authorization": K})))
exps = get("https://www.comet.com/api/rest/v2/experiments?workspaceName=text-films&projectName=fedhetformer-ids")["experiments"]
latest = {}
for e in sorted(exps, key=lambda e: e.get("startTimeMillis") or 0):
    latest[e["experimentName"]] = e          # later runs (R2 = 20) replace earlier ones
rows = []
for name, e in sorted(latest.items()):
    vals = {m["name"]: m.get("valueCurrent") for m in get("https://www.comet.com/api/rest/v2/experiment/metrics/summary?experimentKey=" + e["experimentKey"])["values"]}
    row = {"name": name, "key": e["experimentKey"]}
    row.update({k[5:]: vals[k] for k in vals if k.startswith("test/")})
    rows.append(row)
cols = sorted({c for r in rows for c in r})
with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "key"] + [c for c in cols if c not in ("name", "key")]); w.writeheader(); w.writerows(rows)
print(len(rows), "experiments ->", sys.argv[1])
