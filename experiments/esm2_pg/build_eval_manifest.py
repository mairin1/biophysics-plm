"""Freeze the ProteinGym evaluation design before any model is run.

Decides, once and for all models:
  * which variants of each assay are used (capped subsample),
  * the residue window for assays longer than ESM-2's 1022-token limit,
  * the cross-validation fold of every selected variant, under both the
    `random` and the position-based `modulo` split scheme.

Every model then reuses this manifest, so per-assay Spearman differences
between the baseline and a fine-tuned encoder are paired on identical variants
and identical folds.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BASE, MAX_RESIDUES, MAX_VARIANTS_PER_ASSAY, MIN_VARIANTS_PER_ASSAY, N_FOLDS,
    PG_ASSAYS, PG_REFERENCE, PG_ROOT, SUBSAMPLE_SEED, parse_mutant_positions,
    read_csv, read_json, write_json,
)


def choose_window(min_pos, max_pos, seq_len, width):
    """Residue window covering the most variants.

    Each variant is usable only if all of its mutated positions fall inside the
    window, so variant i permits window starts in
    [max_pos_i - width + 1, min_pos_i]. Accumulating those intervals in a
    difference array gives the coverage count for every start in O(n + L).
    """
    n_starts = seq_len - width + 1
    diff = np.zeros(n_starts + 1, dtype=np.int64)
    lo = np.maximum(0, max_pos - width + 1)
    hi = np.minimum(min_pos, n_starts - 1)
    valid = lo <= hi
    np.add.at(diff, lo[valid], 1)
    np.add.at(diff, hi[valid] + 1, -1)
    coverage = np.cumsum(diff[:-1])
    start = int(np.argmax(coverage))
    return start, int(coverage[start])


def process_assay(row, rng):
    """Returns (record, arrays) for one assay, or (record, None) if skipped."""
    name = row['DMS_id']
    path = PG_ASSAYS / row['DMS_filename']
    target_seq = row['target_seq']
    seq_len = len(target_seq)

    record = {
        'DMS_id': name,
        'filename': row['DMS_filename'],
        'seq_len': seq_len,
        'taxon': row['taxon'],
        'coarse_selection_type': row.get('coarse_selection_type', ''),
        'neff_l_category': row.get('MSA_Neff_L_category', ''),
        'includes_multiple_mutants': row.get('includes_multiple_mutants', ''),
    }

    data = read_csv(path)
    scores, positions, keep = [], [], []
    for i, variant in enumerate(data):
        try:
            score = float(variant['DMS_score'])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(score):
            continue
        if len(variant['mutated_sequence']) != seq_len:
            # Substitution assays preserve length; anything else is unexpected.
            record['status'] = 'skipped:length_mismatch'
            return record, None
        try:
            pos = parse_mutant_positions(variant['mutant'])
        except ValueError:
            continue
        if min(pos) < 0 or max(pos) >= seq_len:
            continue
        keep.append(i)
        scores.append(score)
        positions.append((min(pos), max(pos)))

    record['n_available'] = len(keep)
    if len(keep) < MIN_VARIANTS_PER_ASSAY:
        record['status'] = 'skipped:too_few_variants'
        return record, None

    keep = np.asarray(keep, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    min_pos = np.asarray([p[0] for p in positions], dtype=np.int64)
    max_pos = np.asarray([p[1] for p in positions], dtype=np.int64)

    if seq_len > MAX_RESIDUES:
        start, covered = choose_window(min_pos, max_pos, seq_len, MAX_RESIDUES)
        window = (start, MAX_RESIDUES)
        inside = (min_pos >= start) & (max_pos < start + MAX_RESIDUES)
        record['window_dropped'] = int((~inside).sum())
        keep, scores = keep[inside], scores[inside]
        min_pos, max_pos = min_pos[inside], max_pos[inside]
        if len(keep) < MIN_VARIANTS_PER_ASSAY:
            record['status'] = 'skipped:too_few_after_windowing'
            return record, None
    else:
        window = (0, seq_len)
        record['window_dropped'] = 0

    if len(keep) > MAX_VARIANTS_PER_ASSAY:
        pick = rng.choice(len(keep), size=MAX_VARIANTS_PER_ASSAY, replace=False)
        pick.sort()
        keep, scores = keep[pick], scores[pick]
        min_pos, max_pos = min_pos[pick], max_pos[pick]

    if np.std(scores) < 1e-12:
        record['status'] = 'skipped:constant_score'
        return record, None

    # Random folds: a plain shuffle, balanced in size.
    fold_random = np.arange(len(keep)) % N_FOLDS
    fold_random = fold_random[rng.permutation(len(keep))]

    # Modulo folds: grouped by the first mutated position, so a residue seen in
    # training never reappears in the held-out fold. For multi-mutant assays
    # this only pins the first position, so leakage through the remaining
    # positions is still possible.
    fold_modulo = (min_pos % N_FOLDS).astype(np.int64)

    record['window_start'], record['window_len'] = int(window[0]), int(window[1])
    record['n_selected'] = int(len(keep))
    record['score_std'] = float(np.std(scores))
    record['modulo_fold_sizes'] = [int((fold_modulo == f).sum()) for f in range(N_FOLDS)]
    record['status'] = 'ok'

    arrays = {
        f'{name}__rows': keep.astype(np.int64),
        f'{name}__fold_random': fold_random.astype(np.int8),
        f'{name}__fold_modulo': fold_modulo.astype(np.int8),
    }
    return record, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=BASE / 'proteingym_manifest')
    args = parser.parse_args()

    if not PG_REFERENCE.is_file():
        raise RuntimeError(
            f'Missing {PG_REFERENCE}. Run experiments/esm2_pg/fetch_proteingym.py first.'
        )
    fetch_manifest = read_json(PG_ROOT / 'fetch_manifest.json')

    reference = read_csv(PG_REFERENCE)
    rng = np.random.default_rng(SUBSAMPLE_SEED)

    records, arrays = [], {}
    for i, row in enumerate(reference, 1):
        record, assay_arrays = process_assay(row, rng)
        records.append(record)
        if assay_arrays:
            arrays.update(assay_arrays)
        if i % 25 == 0 or i == len(reference):
            print(f'{i}/{len(reference)} assays processed', flush=True)

    usable = [r for r in records if r['status'] == 'ok']
    if not usable:
        raise RuntimeError('No usable assays; refusing to write an empty manifest')

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / 'folds.npz', **arrays)
    write_json(args.out / 'assays.json', records)
    summary = {
        'proteingym_version': fetch_manifest['version'],
        'reference_sha256': fetch_manifest['reference_sha256'],
        'n_assays_in_reference': len(records),
        'n_assays_usable': len(usable),
        'status_counts': dict(Counter(r['status'] for r in records)),
        'n_variants_selected': sum(r['n_selected'] for r in usable),
        'max_variants_per_assay': MAX_VARIANTS_PER_ASSAY,
        'min_variants_per_assay': MIN_VARIANTS_PER_ASSAY,
        'subsample_seed': SUBSAMPLE_SEED,
        'n_folds': N_FOLDS,
        'max_residues': MAX_RESIDUES,
        'n_assays_windowed': sum(1 for r in usable if r['window_len'] < r['seq_len']),
        'notes': (
            'Subsample and folds are fixed here so every encoder is scored on '
            'identical variants. The modulo split groups variants by their first '
            'mutated position only.'
        ),
    }
    write_json(args.out / 'summary.json', summary)
    print(summary, flush=True)
    (args.out / 'MANIFEST_READY').touch()


if __name__ == '__main__':
    main()
