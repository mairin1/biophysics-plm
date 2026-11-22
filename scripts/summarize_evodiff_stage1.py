from collections import defaultdict
from pathlib import Path
import csv
import json
import math

import numpy as np

ROOT = Path('metl-global/analysis/evodiff_stage1_20261116')
CONDITIONS = ['A', 'B', 'C', 'AB', 'BC', 'ABC']
SEEDS = [20261116, 20261117, 20261118]


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    out = ROOT / 'report'
    out.mkdir(parents=True, exist_ok=True)
    runs = [(c, s) for c in CONDITIONS for s in SEEDS] + [('BASELINE', SEEDS[0])]
    statuses = []
    observations = []
    for condition, seed in runs:
        folder = ROOT / 'runs' / f'{condition}_seed{seed}'
        complete = (folder / 'SUCCESS').exists()
        statuses.append({'condition': condition, 'seed': seed, 'complete': int(complete),
                         'folder': str(folder)})
        if not complete:
            continue
        for row in read_csv(folder / 'probe_metrics.csv'):
            row['condition'] = condition
            row['seed'] = seed
            observations.append(row)
    write_csv(out / 'run_status.csv', statuses, ['condition', 'seed', 'complete', 'folder'])
    if not observations:
        (out / 'summary.md').write_text('# Stage I first batch\n\nNo successful probe results found. Check Slurm logs.\n')
        return
    write_csv(out / 'all_probe_metrics.csv', observations,
              ['condition', 'seed', 'protein', 'split', 'target', 'n', 'r2', 'spearman', 'mae_standardized'])
    base = {}
    for r in observations:
        if r['condition'] == 'BASELINE':
            base[(r['protein'], r['split'], r['target'])] = r
    deltas = []
    for r in observations:
        k = (r['protein'], r['split'], r['target'])
        if r['condition'] == 'BASELINE' or k not in base:
            continue
        b = base[k]
        deltas.append({'condition': r['condition'], 'seed': r['seed'], 'protein': r['protein'],
                       'split': r['split'], 'target': r['target'],
                       'delta_spearman': float(r['spearman']) - float(b['spearman']),
                       'delta_r2': float(r['r2']) - float(b['r2'])})
    write_csv(out / 'paired_differences.csv', deltas,
              ['condition', 'seed', 'protein', 'split', 'target', 'delta_spearman', 'delta_r2'])
    averaged = []
    for c in CONDITIONS:
        for sp in ['val', 'test']:
            for target in ['fa_atr', 'fa_sol']:
                for seed in SEEDS:
                    group = [r for r in deltas if r['condition'] == c and r['seed'] == seed and
                             r['split'] == sp and r['target'] == target]
                    if not group:
                        continue
                    averaged.append({'condition': c, 'seed': seed, 'split': sp, 'target': target,
                                     'proteins': len(group),
                                     'mean_delta_spearman': float(np.mean([r['delta_spearman'] for r in group])),
                                     'mean_delta_r2': float(np.mean([r['delta_r2'] for r in group]))})
    write_csv(out / 'mean_paired_differences.csv', averaged,
              ['condition', 'seed', 'split', 'target', 'proteins', 'mean_delta_spearman', 'mean_delta_r2'])
    plot_errors = []
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for col, target in enumerate(['fa_atr', 'fa_sol']):
            for row, split in enumerate(['val', 'test']):
                ax = axes[row, col]
                for i, c in enumerate(CONDITIONS):
                    points = [r['mean_delta_spearman'] for r in averaged if r['condition'] == c and
                              r['split'] == split and r['target'] == target]
                    if points:
                        ax.scatter([i] * len(points), points, alpha=.55, s=35)
                        ax.plot(i, float(np.mean(points)), marker='_', markersize=17, color='black')
                ax.axhline(0, color='gray', linestyle='--', linewidth=1)
                ax.set_xticks(range(len(CONDITIONS)), CONDITIONS)
                ax.set_title(f'{target}: {split} split')
                ax.set_ylabel('Mean paired Δ Spearman vs pretrained')
                ax.grid(alpha=.2)
        fig.suptitle('Common held-out Rosetta-property probes (within-protein diagnostics only)')
        fig.tight_layout()
        fig.savefig(out / 'paired_probe_performance.png', dpi=190)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)
        for ax, c in zip(axes.flat, CONDITIONS):
            for seed in SEEDS:
                folder = ROOT / 'runs' / f'{c}_seed{seed}'
                h = read_csv(folder / 'history.csv')
                if not h:
                    continue
                epochs = [int(z['epoch']) for z in h]
                ax.plot(epochs, [float(z['train_mse']) for z in h], alpha=.3, linestyle='--')
                ax.plot(epochs, [float(z['val_mse']) for z in h], label=f'seed {seed}')
            ax.set_title(c)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Standardized MSE')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.suptitle('Learning curves within each supervised objective (do not compare y-scales across conditions)')
        fig.tight_layout()
        fig.savefig(out / 'learning_curves.png', dpi=190)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        values = []
        for c in CONDITIONS:
            epochs = []
            for s in SEEDS:
                p = ROOT / 'runs' / f'{c}_seed{s}' / 'summary.json'
                if p.exists():
                    epochs.append(json.loads(p.read_text())['best_epoch'])
            values.append(float(np.mean(epochs)) if epochs else np.nan)
        ax.bar(CONDITIONS, np.nan_to_num(values, nan=0))
        ax.set_ylabel('Mean best checkpoint epoch')
        ax.set_title('Validation-selected checkpoints')
        fig.tight_layout()
        fig.savefig(out / 'checkpoint_epochs.png', dpi=190)
        plt.close(fig)
    except Exception as e:
        plot_errors.append(repr(e))
    finished = sum(r['complete'] for r in statuses)
    lines = ['# EvoDiff Stage I: first screening batch', '',
             f'**Completed jobs:** {finished} / {len(statuses)}',
             '', 'Six known-target configurations: A, B, C, AB, BC, ABC; three seeds each, plus one pretrained baseline.',
             '', 'The primary output here is the *paired difference versus frozen pretrained EvoDiff* on common reserved Rosetta properties.',
             'Use validation probe performance for provisional decisions; inspect test results descriptively, not for choosing winners.',
             'These are within-protein tests with close-mutant overlap; they do NOT establish transfer to unseen proteins or experiments.',
             'This batch is an initial screening subset, not the complete planned ten-condition matrix.',
             '', '## Files', '',
             '- `run_status.csv`: which GPU experiments succeeded',
             '- `all_probe_metrics.csv`: every protein/seed/target/split metric',
             '- `paired_differences.csv`: matched model-minus-pretrained comparisons',
             '- `mean_paired_differences.csv`: per-seed means across proteins',
             '- `paired_probe_performance.png`: common-task comparison',
             '- `learning_curves.png`: training and validation objectives',
             '- `checkpoint_epochs.png`: training lengths',
             '', f'**Plot errors:** {plot_errors or "none"}',
             '', '## Next gates', '',
             'Audit the D family and construct a balanced eight-protein dataset before rotated protein-fold experiments.',
             'Evaluate EvoDiff original masked-sequence likelihood before declaring pretrained ability retained.',
             'Experimental ProteinGym assays remain separate and pending team agreement.', '']
    (out / 'summary.md').write_text('\n'.join(lines))
    (out / 'REPORT_READY').touch()
    print(f'Finished {finished}/{len(statuses)} jobs. Report: {out}', flush=True)


if __name__ == '__main__':
    main()
