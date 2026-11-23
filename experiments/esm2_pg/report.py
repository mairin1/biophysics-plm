"""Aggregate the per-assay ProteinGym results and compare against the baseline.

Comparisons are paired by assay: for every fine-tuned encoder, each assay's
Spearman is differenced against the frozen pretrained ESM-2 value on the same
assay, same variants, same folds. Uncertainty comes from a paired bootstrap
over assays, which is the unit that actually varies here. Three seeds on a
shared assay set describe run-to-run variation, not variation across proteins.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BASE, METHODS, SEEDS, SPLIT_SCHEMES, eval_dir, read_csv, write_csv, write_json,
)

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261122


def load_rows():
    rows = []
    missing = []
    for condition, seed in [('BASELINE', None)] + [(m, s) for m in METHODS for s in SEEDS]:
        path = eval_dir(condition, seed) / 'per_assay_metrics.csv'
        if not path.is_file():
            missing.append(str(path))
            continue
        for row in read_csv(path):
            row['spearman'] = float(row['spearman'])
            row['mean_fold_spearman'] = float(row['mean_fold_spearman'])
            rows.append(row)
    return rows, missing


def paired_bootstrap(differences, rng):
    if len(differences) < 2:
        return float('nan'), float('nan')
    draws = rng.integers(0, len(differences), size=(BOOTSTRAP_RESAMPLES, len(differences)))
    means = differences[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=BASE / 'report')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, missing = load_rows()
    if not rows:
        raise RuntimeError('No per-assay metrics found; run proteingym_eval.py first')

    baseline = {
        (r['DMS_id'], r['split_scheme']): r['spearman']
        for r in rows if r['condition'] == 'BASELINE'
    }
    if not baseline:
        raise RuntimeError('Baseline results are required for paired comparisons')

    # Per-condition aggregates.
    aggregate = defaultdict(list)
    for row in rows:
        key = (row['condition'], row['seed'], row['split_scheme'])
        if np.isfinite(row['spearman']):
            aggregate[key].append(row['spearman'])
    aggregate_rows = [
        {'condition': c, 'seed': s, 'split_scheme': k, 'n_assays': len(v),
         'mean_spearman': float(np.mean(v)), 'median_spearman': float(np.median(v))}
        for (c, s, k), v in sorted(aggregate.items())
    ]
    write_csv(args.out / 'aggregate_by_condition.csv', aggregate_rows,
              ['condition', 'seed', 'split_scheme', 'n_assays',
               'mean_spearman', 'median_spearman'])

    # Paired per-assay differences against the frozen baseline.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    paired_rows, summary_rows = [], []
    for method in METHODS:
        for seed in SEEDS:
            for scheme in SPLIT_SCHEMES:
                deltas, names = [], []
                for row in rows:
                    if (row['condition'] != method or row['seed'] != str(seed)
                            or row['split_scheme'] != scheme):
                        continue
                    base = baseline.get((row['DMS_id'], scheme))
                    if base is None or not np.isfinite(base) or not np.isfinite(row['spearman']):
                        continue
                    deltas.append(row['spearman'] - base)
                    names.append(row['DMS_id'])
                if not deltas:
                    continue
                deltas = np.asarray(deltas, dtype=np.float64)
                paired_rows.extend(
                    {'method': method, 'seed': seed, 'split_scheme': scheme,
                     'DMS_id': name, 'delta_spearman': float(d)}
                    for name, d in zip(names, deltas))
                low, high = paired_bootstrap(deltas, rng)
                summary_rows.append({
                    'method': method, 'seed': seed, 'split_scheme': scheme,
                    'n_assays': len(deltas),
                    'mean_delta': float(deltas.mean()),
                    'median_delta': float(np.median(deltas)),
                    'fraction_improved': float((deltas > 0).mean()),
                    'bootstrap_ci_low': low, 'bootstrap_ci_high': high,
                })

    write_csv(args.out / 'paired_vs_baseline.csv', paired_rows,
              ['method', 'seed', 'split_scheme', 'DMS_id', 'delta_spearman'])
    write_csv(args.out / 'paired_summary.csv', summary_rows,
              ['method', 'seed', 'split_scheme', 'n_assays', 'mean_delta',
               'median_delta', 'fraction_improved',
               'bootstrap_ci_low', 'bootstrap_ci_high'])

    lines = ['# ESM-2 biophysical supervision on ProteinGym substitutions', '']
    if missing:
        lines += ['## Incomplete', '',
                  'These result files were absent when the report ran:', '']
        lines += [f'- `{path}`' for path in missing] + ['']
    for scheme in SPLIT_SCHEMES:
        lines += [f'## Split scheme: {scheme}', '',
                  '| condition | seed | assays | mean Spearman | median Spearman |',
                  '|---|---|---:|---:|---:|']
        for row in aggregate_rows:
            if row['split_scheme'] != scheme:
                continue
            lines.append(
                f"| {row['condition']} | {row['seed'] or '-'} | {row['n_assays']} "
                f"| {row['mean_spearman']:.4f} | {row['median_spearman']:.4f} |")
        lines += ['', '### Paired change versus frozen ESM-2', '',
                  '| method | seed | assays | mean Δ | median Δ | improved | 95% CI |',
                  '|---|---|---:|---:|---:|---:|---|']
        for row in summary_rows:
            if row['split_scheme'] != scheme:
                continue
            lines.append(
                f"| {row['method']} | {row['seed']} | {row['n_assays']} "
                f"| {row['mean_delta']:+.4f} | {row['median_delta']:+.4f} "
                f"| {row['fraction_improved']:.1%} "
                f"| [{row['bootstrap_ci_low']:+.4f}, {row['bootstrap_ci_high']:+.4f}] |")
        lines.append('')
    lines += [
        '## Reading these numbers', '',
        'The `random` scheme lets near-identical variants of the same protein fall',
        'on both sides of a fold boundary, so it overstates what the representation',
        'knows. The `modulo` scheme groups folds by first mutated position; for',
        'multi-mutant assays it constrains only that first position, so some',
        'leakage remains. Confidence intervals come from a paired bootstrap over',
        'assays and do not account for seed-to-seed variation.', '',
        'Hyperparameters were inherited from the EvoDiff Check-in 2 protocol and',
        'were not re-tuned for ESM-2.', '',
    ]
    (args.out / 'summary.md').write_text('\n'.join(lines))

    write_json(args.out / 'report_summary.json', {
        'n_rows': len(rows),
        'n_baseline_assays': len(baseline) // max(1, len(SPLIT_SCHEMES)),
        'missing_result_files': missing,
        'conditions_present': sorted({r['condition'] for r in rows}),
    })
    (args.out / 'REPORT_READY').touch()
    print(f'Report written to {args.out}', flush=True)


if __name__ == '__main__':
    main()
