"""Generates one Kaggle/Colab notebook per experiment into notebooks/.

    python tools/make_notebooks.py

Every notebook has the same setup cells (platform switch, GitHub token, clone,
installs, persistent state) and differs only in its run cells. Edit this file
and regenerate rather than editing the notebooks by hand.
"""

import os
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, 'notebooks')
REPO = 'mtasfi/fed-het-gnn-ids-code'
DEFAULT_BRANCH = 'main'

# id, file name, title, needs GPU, prerequisite note, run cell body
NEEDS_E1 = ("Needs a **completed E1** run for the same dataset, alpha and seed in `STATE_DIR` "
            "(it reuses E1's cached Phase-1 embeddings / checkpoints; it never reruns Phase 1).")

EXPERIMENTS = [
    ('E0', 'E0_data_gate', 'E0: data-readiness gate (CPU)', False,
     'Run first. CPU only: turn the accelerator off to save GPU quota.', None),
    ('PROBE', 'PROBE_timing_probe', 'Timing probe (1 round, 10k flows)', True,
     'Run after E0 and after filling `LOCKED`. Updates the budget table before the E1 queue.', None),
    ('E1', 'E1_main', 'E1: main result (full staged pipeline)', True,
     'Needs E0 + `LOCKED`. Runs Phase 1, embeds once, runs Phase 2. Default seeds 0 and 1.', None),
    ('E4', 'E4_missing_sweep', 'E4: missing-payload sweep', True, NEEDS_E1, None),
    ('A1', 'A1_no_payload', 'A1: no payload nodes', True, 'Needs E0 + `LOCKED` (no encoder used).', None),
    ('A5', 'A5_no_shares_host', 'A5: without shares_host edges', True, NEEDS_E1, None),
    ('B3', 'B3_central_hetgnn', 'B3: centralized HetGNN', True, NEEDS_E1, None),
    ('B1', 'B1_fedavg_mlp', 'B1: FedAvg + MLP on flow statistics', True, 'Needs E0 + `LOCKED`.', None),
    ('B5', 'B5_classical_xgb_rf', 'B5: XGBoost / Random Forest, with and without SMOTE (CPU)', False,
     'Needs E0 + `LOCKED`. CPU only; runs all four variants.', None),
    ('B2', 'B2_fedgatsage', 'B2: FedGATSage rerun on the E1 split', True,
     'Needs E0 + `LOCKED`. Clones mtasfi/Fed_GNN and installs pyg-lib (see docs/b2_fedgatsage_rerun.md).', 'b2'),
    ('A3', 'A3_frozen_encoder', 'A3: frozen encoder instead of federated LoRA', True,
     'Needs E0 + `LOCKED` (embeds with the pre-trained encoder, no Phase 1).', None),
    ('A2', 'A2_hashed_ngrams', 'A2: hashed byte n-grams instead of the transformer', True,
     'Needs E0 + `LOCKED`.', None),
    ('E6', 'E6_efficiency', 'E6: efficiency and actual model size', True, NEEDS_E1, None),
    ('A4', 'A4_homogeneous', 'A4: homogeneous vs heterogeneous GNN', True, NEEDS_E1, None),
    ('E5', 'E5_noniid_sweep', 'E5: non-IID alpha sweep', True,
     'Needs E0 + `LOCKED` including `partition.alpha_sweep`. Runs the full pipeline at each alpha.', None),
    ('A8', 'A8_no_graph_mlp', 'A8: no graph, federated MLP on fused features', True, NEEDS_E1, None),
    ('A9', 'A9_payload_only_head', 'A9: payload only (Phase-1 head)', True, NEEDS_E1, None),
    ('A6', 'A6_no_next', 'A6: without next edges', True, NEEDS_E1, None),
    ('A7', 'A7_performance_weighted', 'A7: performance-weighted aggregation (optional)', True, NEEDS_E1, None),
    ('AGG', 'AGG_aggregate_results', 'Aggregate: rebuild all tables and figures (E7 included)', False,
     'Run any time; reads every run directory in `STATE_DIR`. CPU only.', None),
]


CONFIG_CELL = '''# ============================== EDIT HERE ==============================
PLATFORM = "kaggle"          # "kaggle" | "colab"
BRANCH = "{branch}"               # git branch of {repo} to run
DATASET = "toniot"

# Where the raw ToN-IoT files are (PCAPs + Processed_Network_dataset CSVs)
DATA_ROOT = {{
    "kaggle": "/kaggle/input/ton-iot",                       # attached Kaggle dataset
    "colab": "/content/drive/MyDrive/datasets/TON_IoT",      # folder on Google Drive
}}[PLATFORM]

# Persistent state = work/ (E0 outputs, embedding caches, Phase-1 checkpoints) + runs/ + results/.
# Colab: lives on Drive. Kaggle: lives in /kaggle/working (saved with the notebook version);
# to continue on Kaggle from an earlier session or from Colab, attach that state as a
# Kaggle dataset and put its path in RESTORE_FROM.
STATE_DIR = {{
    "kaggle": "/kaggle/working/fhf_state",
    "colab": "/content/drive/MyDrive/fhf_state",
}}[PLATFORM]
RESTORE_FROM = None          # e.g. "/kaggle/input/fhf-state" (Kaggle only)

# Values locked after E0 (configs/base.yaml keeps them null). Fill once, reuse everywhere.
LOCKED = {{
    "partition.alpha": None,             # e.g. 0.5
    "partition.alpha_sweep": None,       # e.g. "[0.3, 1.0]"
    "graph.shares_host.kappa": None,     # e.g. 10
    # "matching.clock_offset_s": 0,
    # "dataset.exclude_classes": "[]",
}}
EXTRA_SET = []               # any other overrides, e.g. ["phase1.max_segments_per_client_round=20000"]

SEEDS = {seeds}              # None = the seeds in configs/experiments/<id>.yaml
SMOKE = False                # True = real-data subset, 1+2 rounds, saved under runs/_smoke
RESUME = False               # True = continue an interrupted run from its last round checkpoint
OVERWRITE = False            # True = replace a completed run (runs are immutable otherwise)
# ======================================================================
'''

SETUP_CELL = '''import os, sys, glob, shutil, subprocess

IS_KAGGLE = PLATFORM == "kaggle"
assert PLATFORM in ("kaggle", "colab"), PLATFORM
WORKROOT = "/kaggle/working" if IS_KAGGLE else "/content"
CODE = f"{WORKROOT}/fed-het-gnn-ids-code"

if not IS_KAGGLE:
    from google.colab import drive
    drive.mount("/content/drive")


def get_secret(name):
    """Environment variable first, then Kaggle Secrets / Colab userdata."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        if IS_KAGGLE:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


def clone(repo, branch, dest):
    token = get_secret("GITHUB_TOKEN")
    assert token, "Set GITHUB_TOKEN (env var, Kaggle Secret or Colab secret) to clone the private repo"
    if os.path.isdir(dest):
        subprocess.run(["git", "-C", dest, "fetch", "-q", f"https://x-access-token:{token}@github.com/{repo}.git",
                        branch], check=True)
        subprocess.run(["git", "-C", dest, "checkout", "-q", "-B", branch, "FETCH_HEAD"], check=True)
    else:
        subprocess.run(["git", "clone", "-q", "-b", branch,
                        f"https://x-access-token:{token}@github.com/{repo}.git", dest], check=True)
    # never leave the token in .git/config
    subprocess.run(["git", "-C", dest, "remote", "set-url", "origin", f"https://github.com/{repo}.git"], check=True)
    rev = subprocess.check_output(["git", "-C", dest, "rev-parse", "--short", "HEAD"]).decode().strip()
    print(f"{repo}@{branch} -> {dest} ({rev})")


clone("{repo}", BRANCH, CODE)
os.chdir(CODE)
'''.replace('{repo}', REPO)

INSTALL_CELL = '''# Kaggle/Colab already ship torch; install only what is missing.
# torchao (preinstalled on Kaggle) makes peft's LoRA dispatch fail, and nothing here uses it.
!pip uninstall -q -y torchao
!pip install -q "transformers>=4.48" "peft>=0.13" torch_geometric dpkt comet_ml imbalanced-learn tabulate xgboost pyarrow
!nvidia-smi -L || echo "no GPU"
'''

STATE_CELL = '''os.makedirs(STATE_DIR, exist_ok=True)
if IS_KAGGLE and RESTORE_FROM and os.path.isdir(RESTORE_FROM):
    # copy earlier state in without overwriting anything newer that already exists here
    for src in glob.glob(os.path.join(RESTORE_FROM, "**", "*"), recursive=True):
        dst = os.path.join(STATE_DIR, os.path.relpath(src, RESTORE_FROM))
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
        elif not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    print("restored state from", RESTORE_FROM)

SET = [
    f"dataset.source.kind={'kaggle' if IS_KAGGLE else 'local'}",
    f"dataset.source.root={DATA_ROOT}",
    f"paths.work_dir={STATE_DIR}/work",
    f"paths.runs_dir={STATE_DIR}/runs",
    f"paths.results_dir={STATE_DIR}/results",
    "flow_extractor.workers=" + str(os.cpu_count() or 2),
] + [f"{k}={v}" for k, v in LOCKED.items() if v is not None] + list(EXTRA_SET)


def run(script, *args):
    """Runs a repo script with the shared overrides, streaming its log."""
    cmd = [sys.executable, "-u", script, *map(str, args)]
    for s in SET:
        cmd += ["--set", s]
    print("$", " ".join(cmd[:6]), "...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    if proc.wait() != 0:
        raise RuntimeError(f"{script} exited with {proc.returncode}")


def run_exp(exp, *extra):
    flags = []
    for s in (SEEDS or []):
        flags += ["--seed", s]
    flags += ["--smoke"] if SMOKE else []
    flags += ["--resume"] if RESUME else []
    flags += ["--overwrite"] if OVERWRITE else []
    run("experiments/run.py", "--exp", exp, "--dataset", DATASET, *flags, *extra)


def show_results(exp):
    import pandas as pd
    base = os.path.join(STATE_DIR, "runs", "_smoke" if SMOKE else "", exp, DATASET)
    rows = []
    for path in sorted(glob.glob(os.path.join(base, "**", "manifest.json"), recursive=True)):
        import json
        m = json.load(open(path))
        row = {"run": os.path.relpath(os.path.dirname(path), base), "status": m.get("status"),
               "reason": m.get("failure_reason")}
        tm = os.path.join(os.path.dirname(path), "test_metrics.csv")
        if os.path.exists(tm):
            t = pd.read_csv(tm).iloc[0]
            row.update({k: t[k] for k in ("macro_f1", "balanced_accuracy", "accuracy", "worst_client_macro_f1")
                        if k in t})
        rows.append(row)
    return pd.DataFrame(rows)
'''

PERSIST_MD = '''### Keep the state
* **Colab:** `STATE_DIR` is on Google Drive already.
* **Kaggle:** use *Save Version -> Save & Run All*; `/kaggle/working/fhf_state` becomes the notebook output.
  To continue in a later session (or on Colab), create a dataset from that output and set `RESTORE_FROM`
  (Kaggle) or copy it to `MyDrive/fhf_state` (Colab). Ablations need E1's `work/<ds>/cache` and `runs/E1`.
'''


def run_cells(exp_id: str, special):
    if exp_id == 'E0':
        return [
            new_markdown_cell('### E0 steps (each resumable; extraction skips finished capture units)'),
            new_code_cell('run("experiments/run_e0.py", "extract", "--dataset", DATASET)\n'
                          '# smoke first if you like: add "--max-files", 2, "--max-packets", 500000'),
            new_code_cell('run("experiments/run_e0.py", "labels", "--dataset", DATASET)'),
            new_code_cell('run("experiments/run_e0.py", "offset", "--dataset", DATASET)\n'
                          '# if the reported peak is clearly not 0: LOCKED["matching.clock_offset_s"] = <peak>, rerun setup'),
            new_code_cell('run("experiments/run_e0.py", "match", "--dataset", DATASET)'),
            new_code_cell('run("experiments/run_e0.py", "check", "--dataset", DATASET)\n'
                          '# payload signature vs label, per-class coverage, failure breakdown, gate (matching.gate)\n'
                          'from IPython.display import Markdown, display\n'
                          'display(Markdown(open(f"{STATE_DIR}/work/{DATASET}/e0/match_checks.md").read()))'),
            new_code_cell('run("experiments/run_e0.py", "payload", "--dataset", DATASET)'),
            new_code_cell('run("experiments/run_e0.py", "split", "--dataset", DATASET)'),
            new_code_cell('run("experiments/run_e0.py", "kappa", "--dataset", DATASET)'),
            new_markdown_cell('### Read the reports, then fill `LOCKED` (alpha, alpha_sweep, kappa) in every notebook\n'
                              'E0 stop rule: unreliable matching, lost payloads for key classes, ambiguous labels, '
                              'or a dense kappa -> stop before GPU work.'),
            new_code_cell('from IPython.display import Markdown, display\n'
                          'import pandas as pd\n'
                          'E0 = f"{STATE_DIR}/work/{DATASET}/e0"\n'
                          'display(Markdown(open(f"{E0}/dataset_audit.md").read()))\n'
                          'display(Markdown(open(f"{E0}/match_report.md").read()))\n'
                          'display(Markdown(open(f"{E0}/match_checks.md").read()))\n'
                          'print(open(f"{E0}/split_report.json").read())\n'
                          'display(pd.read_csv(f"{E0}/template_overlap_report.csv"))\n'
                          'display(pd.read_csv(f"{E0}/kappa_stats.csv"))'),
        ]
    if exp_id == 'PROBE':
        return [
            new_code_cell('run("experiments/run_probe.py", "--dataset", DATASET, "--flows", 10000)'),
            new_markdown_cell('If the Phase-1 estimate does not fit the budget, run the DistilBERT fallback probe:'),
            new_code_cell('# run("experiments/run_probe.py", "--dataset", DATASET, "--flows", 10000, "--encoder", "distilbert")'),
        ]
    if exp_id == 'AGG':
        return [
            new_code_cell('run("experiments/aggregate_results.py", "--dataset", DATASET)'),
            new_code_cell('import pandas as pd\nfrom IPython.display import display\n'
                          'R = f"{STATE_DIR}/results/{DATASET}"\n'
                          'for name in ["runs_index", "results_main", "results_baselines", "results_ablations",\n'
                          '             "results_missing_sweep", "results_noniid"]:\n'
                          '    p = f"{R}/{name}.csv"\n'
                          '    if os.path.exists(p):\n'
                          '        print(name); display(pd.read_csv(p))'),
            new_code_cell('# figures are PDFs for the thesis; preview them here\n'
                          '!pip install -q pymupdf\n'
                          'import pymupdf\nfrom IPython.display import Image, display\n'
                          'for pdf in sorted(glob.glob(f"{R}/fig_*.pdf")):\n'
                          '    png = pdf[:-4] + ".png"\n'
                          '    pymupdf.open(pdf)[0].get_pixmap(dpi=110).save(png)\n'
                          '    print(os.path.basename(pdf)); display(Image(png))'),
        ]
    cells = []
    if special == 'b2':
        cells += [
            new_markdown_cell('### FedGATSage code + its extra dependencies'),
            new_code_cell('FEDGNN_BRANCH = "main"\n'
                          'FEDGNN = f"{WORKROOT}/Fed_GNN"\n'
                          'clone("mtasfi/Fed_GNN", FEDGNN_BRANCH, FEDGNN)\n'
                          'import torch\n'
                          'tv = ".".join(torch.__version__.split("+")[0].split(".")[:2]) + ".0"\n'
                          'cu = "cu" + torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"\n'
                          '!pip install -q python-louvain leidenalg igraph networkx joblib\n'
                          '!pip install -q pyg-lib -f https://data.pyg.org/whl/torch-{tv}+{cu}.html\n'
                          'SET.append(f"baselines.fedgatsage.repo_path={FEDGNN}")'),
        ]
    cells += [
        new_code_cell(f'run_exp("{exp_id}")'),
        new_code_cell(f'show_results("{exp_id}")'),
    ]
    return cells


def build(exp_id, fname, title, gpu, prereq, special):
    spec = {}
    if exp_id not in ('E0', 'PROBE', 'AGG'):
        import yaml
        with open(os.path.join(ROOT, 'configs', 'experiments', f'{exp_id}.yaml')) as f:
            spec = yaml.safe_load(f)
    lines = [f'# {title}', '']
    if spec:
        lines += [f"**Question:** {spec.get('question', '')}", '', f"**Done when:** {spec.get('done', '')}", '']
    lines += [f"**Before running:** {prereq}", '',
              f"**Accelerator:** {'GPU (T4)' if gpu else 'none needed (CPU)'}. "
              "Internet must be on (GitHub clone, pip, Hugging Face, Comet).", '',
              'Secrets: add `GITHUB_TOKEN` (repo read access) as an environment variable, a Kaggle Secret, '
              'or a Colab secret. Results are mirrored to Comet project `fedhetformer-ids` '
              f"(tag `exp:{exp_id}`); the CSVs in `STATE_DIR/runs` are the record."]
    nb = new_notebook()
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.metadata['language_info'] = {'name': 'python'}
    nb.metadata['fedhetformer'] = {'experiment': exp_id, 'generated_by': 'tools/make_notebooks.py'}
    seeds = 'None'
    nb.cells = [
        new_markdown_cell('\n'.join(lines)),
        new_code_cell(CONFIG_CELL.format(branch=DEFAULT_BRANCH, repo=REPO, seeds=seeds)),
        new_code_cell(SETUP_CELL),
        new_code_cell(INSTALL_CELL),
        new_code_cell(STATE_CELL),
        *run_cells(exp_id, special),
        new_markdown_cell(PERSIST_MD),
    ]
    path = os.path.join(OUT, f'{fname}.ipynb')
    nbformat.write(nb, path)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    for e in EXPERIMENTS:
        print(build(*e))


if __name__ == '__main__':
    main()
