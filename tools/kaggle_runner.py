"""Kaggle queue runner: restore the E0 release + shared state from Hugging Face, run a
queue of experiments, push the results back after each one.

Submitted as a Kaggle *script* kernel (GPU, Internet on). The block between the
QUEUE markers is rewritten per submission; everything else is fixed.

State lives in a private HF dataset repo (STATE_REPO), so any runner session, on
any notebook, continues from the same place:
    runs/<exp>/...                 one immutable run directory per run (the record)
    work/<dataset>/cache/...       E1 embeddings + Phase-1 checkpoints reused by ablations
    work/<dataset>/probe/...       timing-probe reports
    results/<dataset>/...          aggregate tables and figures (AGG)

Secrets (Kaggle Add-ons -> Secrets, attached to the runner notebook):
    HF_TOKEN        read the release, read/write STATE_REPO
    COMET_API_KEY   optional; Comet mirror of the scalar metrics
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

# ==== QUEUE (rewritten per submission) ====
QUEUE = ["PROBE"]                    # experiment ids in order; PROBE and AGG are special
LOCKED = {}                          # e.g. {"partition.alpha": 0.5, "graph.shares_host.kappa": 10}
EXTRA_SET = []                       # any other key=value overrides
SMOKE = False
OVERWRITE = False
BRANCH = "main"
# ==== /QUEUE ====

DATASET = "toniot"
RELEASE_REPO, RELEASE_REV = "mtasfi/toniot-fhf-processed", "v2"
STATE_REPO = "mtasfi/fhf-state"
CODE = "/kaggle/working/fed-het-gnn-ids-code"
STATE = "/kaggle/working/fhf_state"
WORK = f"{STATE}/work"


def secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def sh(cmd, check=True):
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    p = subprocess.run(cmd, shell=isinstance(cmd, str))
    if check and p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}: {cmd}")
    return p.returncode


# ---------------------------------------------------------------- setup
if not os.path.isdir(CODE):
    sh(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, "https://github.com/mtasfi/fed-het-gnn-ids-code.git", CODE])
COMMIT = subprocess.check_output(["git", "-C", CODE, "rev-parse", "HEAD"]).decode().strip()
os.chdir(CODE)
sys.path.insert(0, CODE)
print("code @", COMMIT[:10], flush=True)
sh("pip uninstall -q -y torchao", check=False)   # breaks peft's LoRA dispatch on Kaggle images
sh('pip install -q "transformers>=4.48" "peft>=0.13" torch_geometric dpkt comet_ml imbalanced-learn '
   'tabulate xgboost pyarrow huggingface_hub pytest')
sh("nvidia-smi -L", check=False)

from huggingface_hub import HfApi, snapshot_download

HF_TOKEN = secret("HF_TOKEN")
assert HF_TOKEN, "Attach the HF_TOKEN secret to this notebook"
api = HfApi(token=HF_TOKEN)
api.create_repo(STATE_REPO, repo_type="dataset", private=True, exist_ok=True)
if secret("COMET_API_KEY"):
    os.environ["COMET_API_KEY"] = secret("COMET_API_KEY")


def restore_release():
    """E0 release -> work/<ds>/flows_labeled.parquet + splits/ + e0/ (what run.py reads)."""
    import pandas as pd
    root = f"{WORK}/{DATASET}"
    if os.path.exists(f"{root}/flows_labeled.parquet"):
        return
    d = snapshot_download(RELEASE_REPO, repo_type="dataset", revision=RELEASE_REV, token=HF_TOKEN)
    os.makedirs(root, exist_ok=True)
    flows = pd.read_parquet(f"{d}/flows.parquet")
    pl = pd.read_parquet(f"{d}/payloads.parquet", columns=["flow_uid", "seg_pos", "segment_reason", "text"])
    pl = pl.sort_values(["flow_uid", "seg_pos"])
    texts = pl.dropna(subset=["text"]).groupby("flow_uid")["text"].apply(list)
    reasons = pl.groupby("flow_uid")["segment_reason"].apply(list)
    flows["payload_texts"] = flows["flow_uid"].map(texts).apply(lambda v: v if isinstance(v, list) else [])
    flows["payload_segment_reasons"] = flows["flow_uid"].map(reasons).apply(lambda v: v if isinstance(v, list) else [])
    flows.drop(columns=["n_ok_segments"], errors="ignore").to_parquet(f"{root}/flows_labeled.parquet", index=False)
    for sub in ("splits", "e0"):
        if os.path.isdir(f"{d}/{sub}"):
            shutil.copytree(f"{d}/{sub}", f"{root}/{sub}", dirs_exist_ok=True)
    assert os.path.isdir(f"{root}/splits"), "release has no splits/"
    print(f"release {RELEASE_REV}: {len(flows):,} flows restored", flush=True)


def pull_state():
    try:
        d = snapshot_download(STATE_REPO, repo_type="dataset", token=HF_TOKEN)
    except Exception as e:
        print("no state yet:", e)
        return
    for sub in ("runs", "work", "results"):
        if os.path.isdir(f"{d}/{sub}"):
            shutil.copytree(f"{d}/{sub}", f"{STATE}/{sub}", dirs_exist_ok=True)
    print("state pulled from", STATE_REPO, flush=True)


TOUCHED = set()   # experiment ids run in THIS session
SESSION_T0 = time.time()   # run dirs whose manifest is newer than this are pushed


def push_state(label):
    """Upload what later sessions need: this session's runs, caches, results. Never raw flows/payloads.
    Only run dirs of experiments executed here are pushed: two runners share STATE_REPO, and pushing
    every local run dir would overwrite the other runner's newer results with this session's stale copies."""
    fresh = []
    for m in glob.glob(f"{STATE}/runs/**/manifest.json", recursive=True):
        if os.path.getmtime(m) >= SESSION_T0:          # created or rewritten in this session
            fresh.append(os.path.relpath(os.path.dirname(m), STATE) + "/**")
    patterns = fresh + [f"work/{DATASET}/cache/**", f"work/{DATASET}/probe/**"]
    if "AGG" in TOUCHED:
        patterns.append("results/**")
    for attempt in range(3):
        try:
            api.upload_folder(folder_path=STATE, repo_id=STATE_REPO, repo_type="dataset", allow_patterns=patterns,
                              commit_message=f"{label} (code {COMMIT[:10]})")
            print("state pushed:", label, flush=True)
            return
        except Exception as e:
            print("push failed, retrying:", e, flush=True)
            time.sleep(30)


SET = [f"paths.work_dir={WORK}", f"paths.runs_dir={STATE}/runs", f"paths.results_dir={STATE}/results",
       "flow_extractor.workers=" + str(os.cpu_count() or 2)] + \
      [f"{k}={v}" for k, v in LOCKED.items() if v is not None] + list(EXTRA_SET)


def run(script, *args):
    cmd = [sys.executable, "-u", script, *map(str, args)]
    for s in SET:
        cmd += ["--set", s]
    print("$", " ".join(cmd[:8]), "...", flush=True)
    return subprocess.run(cmd).returncode


def summary():
    rows = []
    for m in sorted(glob.glob(f"{STATE}/runs/**/manifest.json", recursive=True)):
        man = json.load(open(m))
        row = {"run": os.path.relpath(os.path.dirname(m), f"{STATE}/runs"), "status": man.get("status")}
        tm = os.path.join(os.path.dirname(m), "test_metrics.csv")
        if os.path.exists(tm):
            import pandas as pd
            t = pd.read_csv(tm).iloc[0]
            row.update({k: round(float(t[k]), 4) for k in ("macro_f1", "balanced_accuracy", "accuracy",
                                                           "worst_client_macro_f1") if k in t})
        rows.append(row)
    return rows


# ---------------------------------------------------------------- queue
# E0 unit tests of this checkout (synthetic data, ~1 min): a broken split rule must not reach a run
assert subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_e0.py"]).returncode == 0, "tests failed"
restore_release()
# Rebuild splits/partitions with the checked-out code (~1 min, seeded, so train/test is
# identical to the release): the release's splits/ predate the v1.3 validation rule.
if not os.path.exists(f"{WORK}/{DATASET}/splits/.resplit_{COMMIT[:10]}"):
    assert run("experiments/run_e0.py", "split", "--dataset", DATASET) == 0, "E0 split failed"
    open(f"{WORK}/{DATASET}/splits/.resplit_{COMMIT[:10]}", "w").close()
pull_state()
time.sleep(2)
SESSION_T0 = time.time()   # after the pull: pulled run dirs carry download-time mtimes and must not count as fresh
results = {}
for exp in QUEUE:
    t0 = time.time()
    label = exp if isinstance(exp, str) else json.dumps(exp, sort_keys=True)
    if isinstance(exp, str):
        TOUCHED.add(exp)
    elif exp.get("exp"):
        TOUCHED.add(exp["exp"])
    try:
        if isinstance(exp, dict) and exp.get("kind") == "resplit":
            # partition-sensitivity analysis: redraw the client partitions with another seed
            # (train/test blocks do not depend on partition.seed, only the client assignment does)
            extra = [f"partition.seed={exp['partition_seed']}"]
            cmd = [sys.executable, "-u", "experiments/run_e0.py", "split", "--dataset", DATASET]
            for s_ in SET + extra:
                cmd += ["--set", s_]
            code = subprocess.run(cmd).returncode
        elif isinstance(exp, dict):
            # {"exp": "A2", "variant": "psens_p1", "alpha": 0.3, "seed": 1, "set": ["partition.seed=1"]}
            args = ["experiments/run.py", "--exp", exp["exp"], "--dataset", DATASET]
            if exp.get("variant"):
                args += ["--variant", exp["variant"]]
            if exp.get("seed") is not None:
                args += ["--seed", str(exp["seed"])]
            if exp.get("alpha") is not None:
                args += ["--alpha", str(exp["alpha"])]
            for s_ in exp.get("set", []):
                args += ["--set", s_]
            args += (["--overwrite"] if OVERWRITE else [])
            code = run(*args)
        elif exp == "PROBE":
            code = run("experiments/run_probe.py", "--dataset", DATASET, "--flows", 10000)
        elif exp == "AGG":
            code = run("experiments/aggregate_results.py", "--dataset", DATASET)
        else:
            flags = (["--smoke"] if SMOKE else []) + (["--overwrite"] if OVERWRITE else [])
            code = run("experiments/run.py", "--exp", exp, "--dataset", DATASET, *flags)
    except Exception:
        traceback.print_exc()
        code = -1
    results[label] = {"exit": code, "minutes": round((time.time() - t0) / 60, 1)}
    print(f"===== {label}: exit {code}, {results[label]['minutes']} min", flush=True)
    push_state(f"{label[:60]} exit {code}")

print("\n===== QUEUE RESULT =====")
print(json.dumps(results, indent=1))
print("\n===== ALL RUNS =====")
for r in summary():
    print(r)
os.makedirs("/kaggle/working", exist_ok=True)
json.dump({"queue": results, "runs": summary(), "code": COMMIT}, open("/kaggle/working/queue_result.json", "w"),
          indent=1, default=str)
