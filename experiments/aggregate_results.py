"""One command that rebuilds every sprint table and figure from the immutable run
directories (plan §5). Nothing here trains or reads Comet; CSVs are the record.

    python experiments/aggregate_results.py --dataset toniot

results/<dataset>/
    runs_index.csv              every run found, with status and failure reason (failures are kept)
    results_main.csv            E1 per seed + mean/sd                       -> table_main.tex
    results_baselines.csv       B1, B2, B3, B5 (+ B4 not_run)
    results_ablations.csv       A1-A9 with paired deltas vs E1 (same dataset, alpha, seed) -> table_ablation.tex
    results_ablation_perclass.csv
    results_missing_sweep.csv   E4                                          -> fig_missing_rate.pdf
    results_noniid.csv          E5 (+ E1 at the primary alpha)              -> fig_noniid.pdf
    results_perclass.csv        E7: E1 vs A1 vs A2 (+ B2) per class          -> table_perclass.tex, fig_perclass.pdf
    efficiency_runs.csv         E6                                          -> table_efficiency.tex
    fig_convergence.pdf         E1 Phase-1 / Phase-2 validation macro-F1 per round, best rounds marked
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml

from fhf.common.config import CONFIG_DIR, load_config, resolve_path

# Validated categorical order (dataviz reference palette, slots 1-4; light surface).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
HEADLINE = ['macro_f1', 'balanced_accuracy', 'accuracy', 'worst_client_macro_f1']
BASELINES = ['B1', 'B2', 'B3', 'B4', 'B5']
ABLATIONS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']


# ----------------------------------------------------------------------------- index
def index_runs(runs_dir: str, dataset: str) -> pd.DataFrame:
    rows = []
    for manifest in glob.glob(os.path.join(runs_dir, '*', dataset, '*', 'alpha=*', 'seed=*', 'manifest.json')):
        with open(manifest) as f:
            m = json.load(f)
        path = os.path.dirname(manifest)
        row = {k: m.get(k) for k in ('run_id', 'experiment_id', 'variant', 'dataset', 'alpha', 'seed', 'status',
                                     'failure_reason', 'start', 'end', 'git_commit', 'git_dirty', 'comet_url')}
        row['path'] = path
        tm = os.path.join(path, 'test_metrics.csv')
        if os.path.exists(tm):
            t = pd.read_csv(tm).iloc[0].to_dict()
            row.update({k: t.get(k) for k in HEADLINE + ['checkpoint_round', 'coverage'] if k in t})
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['alpha'] = df['alpha'].astype(str)
    return df.sort_values(['experiment_id', 'variant', 'alpha', 'seed']).reset_index(drop=True)


def _completed(idx, exp):
    return idx[(idx.experiment_id == exp) & (idx.status == 'completed')]


# ----------------------------------------------------------------------------- tables
def main_results(idx):
    e1 = _completed(idx, 'E1')
    rows = e1[['run_id', 'alpha', 'seed'] + [c for c in HEADLINE if c in e1]].copy()
    if len(e1):
        mean = {'run_id': 'E1 mean', 'alpha': e1['alpha'].iloc[0], 'seed': 'mean'}
        sd = {'run_id': 'E1 sd', 'alpha': e1['alpha'].iloc[0], 'seed': 'sd'}
        for c in HEADLINE:
            if c in e1:
                mean[c], sd[c] = e1[c].mean(), e1[c].std(ddof=1) if len(e1) > 1 else np.nan
        rows = pd.concat([rows, pd.DataFrame([mean, sd])], ignore_index=True)
    return rows


def paired_rows(idx, exps, ref='E1'):
    ref_runs = _completed(idx, ref).set_index(['alpha', 'seed'])
    out = []
    for exp in exps:
        for _, r in idx[idx.experiment_id == exp].iterrows():
            row = r[['experiment_id', 'variant', 'alpha', 'seed', 'status', 'failure_reason'] +
                    [c for c in HEADLINE + ['coverage'] if c in r]].to_dict()
            key = (r['alpha'], r['seed'])
            if r['status'] == 'completed' and key in ref_runs.index:
                ref_row = ref_runs.loc[key]
                for c in ('macro_f1', 'balanced_accuracy'):
                    row[f'delta_{c}_vs_{ref}'] = r.get(c, np.nan) - ref_row[c]
            out.append(row)
    return pd.DataFrame(out)


def per_class(path: str) -> pd.DataFrame:
    p = os.path.join(path, 'per_class_metrics.csv')
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def perclass_table(idx, exps=('E1', 'A1', 'A2', 'B2')):
    rows = []
    for exp in exps:
        for _, r in _completed(idx, exp).iterrows():
            pc = per_class(r['path'])
            if pc.empty:
                continue
            pc.insert(0, 'experiment_id', exp)
            pc.insert(1, 'variant', r['variant'])
            pc.insert(2, 'alpha', r['alpha'])
            pc.insert(3, 'seed', r['seed'])
            rows.append(pc)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ablation_perclass(idx):
    base = {(r['alpha'], r['seed']): per_class(r['path']).set_index('label')['f1']
            for _, r in _completed(idx, 'E1').iterrows()}
    rows = []
    for exp in ABLATIONS:
        for _, r in _completed(idx, exp).iterrows():
            ref = base.get((r['alpha'], r['seed']))
            pc = per_class(r['path'])
            if ref is None or pc.empty:
                continue
            for _, c in pc.iterrows():
                rows.append({'experiment_id': exp, 'alpha': r['alpha'], 'seed': r['seed'], 'label': c['label'],
                             'support': c['support'], 'f1': c['f1'], 'e1_f1': ref.get(c['label'], np.nan),
                             'delta_f1': c['f1'] - ref.get(c['label'], np.nan)})
    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame, cols, path: str, caption: str, label: str):
    sub = df[[c for c in cols if c in df]].copy()
    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].map(lambda v: '' if pd.isna(v) else f'{v:.3f}')
    body = sub.to_latex(index=False, escape=True)
    with open(path, 'w') as f:
        f.write(f"% generated by experiments/aggregate_results.py\n\\begin{{table}}[htbp]\n\\centering\n"
                f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\footnotesize\n{body}\\end{{table}}\n")


# ----------------------------------------------------------------------------- figures
def _axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.titlecolor': INK, 'text.color': INK})
    return plt


def fig_convergence(idx, out: str):
    e1 = _completed(idx, 'E1')
    if e1.empty:
        return
    plt = _plt()
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, phase, title in zip(axes, ('1', '2'), ('Phase 1: payload encoder', 'Phase 2: HetGNN')):
        for i, (_, r) in enumerate(e1.iterrows()):
            rm = pd.read_csv(os.path.join(r['path'], 'round_metrics.csv'))
            g = rm[(rm['phase'].astype(str) == phase) & (rm['scope'] == 'global')]
            if g.empty:
                continue
            color = SERIES[i % len(SERIES)]
            ax.plot(g['round'], g['val_macro_f1'], color=color, linewidth=2, marker='o', markersize=4,
                    label=f"seed {r['seed']}")
            best = g.loc[g['val_macro_f1'].idxmax()]
            ax.scatter([best['round']], [best['val_macro_f1']], s=70, facecolor='white', edgecolor=color,
                       linewidth=2, zorder=3)
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _axes(ax, 'round', 'validation macro-F1')
    axes[1].legend(frameon=False, fontsize=8, title='open marker = selected round', title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_line(df: pd.DataFrame, x: str, ys, labels, xlabel: str, out: str, title: str):
    if df.empty:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for i, (y, lab) in enumerate(zip(ys, labels)):
        ax.plot(df[x], df[y], color=SERIES[i], linewidth=2, marker='o', markersize=5, label=lab)
    ax.set_title(title)
    _axes(ax, xlabel, 'score')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_perclass(pc: pd.DataFrame, out: str, highlight):
    if pc.empty:
        return
    plt = _plt()
    first_seed = pc.groupby('experiment_id')['seed'].transform('min')
    pc = pc[pc['seed'] == first_seed]
    exps = [e for e in ('E1', 'A1', 'A2', 'B2') if e in set(pc['experiment_id'])]
    labels = sorted(pc['label'].unique(), key=lambda l: (l not in highlight, l))
    width = 0.8 / len(exps)
    fig, ax = plt.subplots(figsize=(max(5, 0.75 * len(labels) + 1.5), 3.4))
    x = np.arange(len(labels))
    names = {'E1': 'FedHetFormer-IDS (E1)', 'A1': 'no payload nodes (A1)', 'A2': 'hashed n-grams (A2)',
             'B2': 'FedGATSage (B2)'}
    for i, exp in enumerate(exps):
        f1 = pc[pc.experiment_id == exp].set_index('label').reindex(labels)['f1'].to_numpy()
        ax.bar(x + (i - (len(exps) - 1) / 2) * width, f1, width * 0.92, color=SERIES[i], label=names[exp])
    ax.set_xticks(x, [f'{l}*' if l in highlight else l for l in labels], rotation=30, ha='right')
    ax.set_ylim(0, 1)
    ax.set_title('Per-class F1 (* payload-dependent class)', pad=30)
    _axes(ax, '', 'F1')
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='toniot')
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    from fhf.common.config import parse_set_args
    cfg = load_config(args.dataset, overrides=parse_set_args(args.set))
    runs_dir = resolve_path(cfg.paths.runs_dir)
    out = os.path.join(resolve_path(cfg.paths.results_dir), args.dataset)
    os.makedirs(out, exist_ok=True)

    idx = index_runs(runs_dir, args.dataset)
    if idx.empty:
        print(f"No runs for dataset '{args.dataset}' under {runs_dir}")
        return
    idx.to_csv(os.path.join(out, 'runs_index.csv'), index=False)

    main_df = main_results(idx)
    main_df.to_csv(os.path.join(out, 'results_main.csv'), index=False)

    base = paired_rows(idx, BASELINES)
    with open(os.path.join(CONFIG_DIR, 'experiments', 'B4.yaml')) as f:
        b4 = yaml.safe_load(f)
    if b4.get('pipeline') == 'not_run':
        base = pd.concat([base, pd.DataFrame([{'experiment_id': 'B4', 'variant': b4['variant'],
                                               'status': 'not_run', 'failure_reason': b4.get('note')}])])
    base.to_csv(os.path.join(out, 'results_baselines.csv'), index=False)

    abl = paired_rows(idx, ABLATIONS)
    abl.to_csv(os.path.join(out, 'results_ablations.csv'), index=False)
    ablation_perclass(idx).to_csv(os.path.join(out, 'results_ablation_perclass.csv'), index=False)

    main_tab = pd.concat([main_df[main_df['seed'].astype(str) == 'mean'].assign(method='FedHetFormer-IDS (E1)'),
                          base.assign(method=base['experiment_id'] + ' ' + base['variant'].astype(str))
                          if len(base) else base])
    to_latex(main_tab, ['method', 'seed'] + HEADLINE, os.path.join(out, 'table_main.tex'),
             'Main results (sprint, directional).', 'Tab_R_Main')
    to_latex(abl, ['experiment_id', 'variant', 'seed', 'macro_f1', 'delta_macro_f1_vs_E1', 'balanced_accuracy',
                   'delta_balanced_accuracy_vs_E1', 'status'], os.path.join(out, 'table_ablation.tex'),
             'Ablations with paired deltas against E1.', 'Tab_R_Ablation')

    sweep_rows = []
    for _, r in _completed(idx, 'E4').iterrows():
        s = pd.read_csv(os.path.join(r['path'], 'missing_sweep.csv'))
        s.insert(0, 'seed', r['seed'])
        s.insert(0, 'alpha', r['alpha'])
        sweep_rows.append(s)
    sweep = pd.concat(sweep_rows) if sweep_rows else pd.DataFrame()
    sweep.to_csv(os.path.join(out, 'results_missing_sweep.csv'), index=False)
    if len(sweep):
        s0 = sweep[sweep['seed'] == sweep['seed'].min()]
        fig_line(s0, 'mask_rate', ['macro_f1', 'balanced_accuracy', 'worst_client_macro_f1'],
                 ['macro-F1', 'balanced accuracy', 'worst-client macro-F1'], 'share of readable payloads masked',
                 os.path.join(out, 'fig_missing_rate.pdf'), 'Graceful degradation (E4)')

    noniid = paired_rows(idx, ['E5', 'E1'])
    noniid.to_csv(os.path.join(out, 'results_noniid.csv'), index=False)
    ok = noniid[noniid['status'] == 'completed'].copy()
    if len(ok) and ok['alpha'].nunique() > 1:
        ok['alpha_num'] = pd.to_numeric(ok['alpha'], errors='coerce')
        agg = ok.dropna(subset=['alpha_num']).groupby('alpha_num')[['macro_f1', 'worst_client_macro_f1']].mean().reset_index()
        fig_line(agg, 'alpha_num', ['macro_f1', 'worst_client_macro_f1'], ['macro-F1', 'worst-client macro-F1'],
                 'Dirichlet alpha (lower = more skew)', os.path.join(out, 'fig_noniid.pdf'), 'Label skew (E5)')

    pc = perclass_table(idx)
    pc.to_csv(os.path.join(out, 'results_perclass.csv'), index=False)
    highlight = set(cfg.dataset.get('payload_dependent_classes') or [])
    if len(pc):
        wide = pc[pc['seed'] == pc.groupby('experiment_id')['seed'].transform('min')].pivot_table(
            index=['label'], columns='experiment_id', values='f1').reset_index()
        support = pc[pc.experiment_id == 'E1'].groupby('label')['support'].first()
        wide['test_support'] = wide['label'].map(support)
        to_latex(wide, ['label', 'test_support'] + [c for c in ('E1', 'A1', 'A2', 'B2') if c in wide],
                 os.path.join(out, 'table_perclass.tex'), 'Per-class F1 (E7).', 'Tab_R_PerClass')
        fig_perclass(pc, os.path.join(out, 'fig_perclass.pdf'), highlight)

    eff = [pd.read_csv(os.path.join(r['path'], 'efficiency_runs.csv')) for _, r in _completed(idx, 'E6').iterrows()]
    if eff:
        eff = pd.concat(eff)
        eff.to_csv(os.path.join(out, 'efficiency_runs.csv'), index=False)
        long = eff.drop(columns=['run_id']).iloc[0].rename('value').reset_index().rename(columns={'index': 'measure'})
        to_latex(long, ['measure', 'value'], os.path.join(out, 'table_efficiency.tex'),
                 'Measured efficiency of E1 (E6).', 'Tab_R_Efficiency')

    fig_convergence(idx, os.path.join(out, 'fig_convergence.pdf'))
    print(f"{len(idx)} runs indexed ({(idx.status == 'completed').sum()} completed, "
          f"{(idx.status != 'completed').sum()} other); outputs in {out}")


if __name__ == '__main__':
    main()
