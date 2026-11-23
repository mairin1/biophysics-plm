"""Per-assay ProteinGym evaluation of one frozen ESM-2 encoder.

For each usable assay the encoder produces one mean-pooled embedding per
variant. A ridge probe is then fitted with 5-fold cross-validation and the
out-of-fold predictions are scored against the continuous DMS_score with
Spearman correlation. Nothing is pooled across assays at fit time: each assay
gets its own probe and its own correlation, which is the ProteinGym convention
and avoids letting between-protein variation stand in for within-protein
ranking skill.

Two split schemes are reported from the same embeddings:
  random  5-fold shuffle; optimistic, because near-identical variants can fall
          on both sides of a split
  modulo  folds grouped by first mutated position, so a residue seen in
          training does not reappear in the held-out fold
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BASE, CONDITIONS, ESM2_MODEL, N_FOLDS, PG_ASSAYS, RIDGE_ALPHAS, SEEDS,
    SPLIT_SCHEMES, eval_dir, read_json, spearman, write_csv, write_json,
)
from model import load_condition  # noqa: E402

TOKEN_BUDGET = 16384
INNER_VAL_FRACTION = 0.2
ALPHA_SELECTION_SEED = 991

# The GPU is required for the real run; falling back to CPU keeps the ridge and
# cross-validation code testable without one.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_selected(path, row_indices):
    """Read only the manifest-selected rows of an assay.

    The largest assay holds 537k variants, so materialising every row as a dict
    just to keep 2,000 of them wastes several hundred megabytes.
    """
    wanted = set(int(i) for i in row_indices)
    found = {}
    with open(path, newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        seq_col = header.index('mutated_sequence')
        score_col = header.index('DMS_score')
        for i, row in enumerate(reader):
            if i in wanted:
                found[i] = (row[seq_col], float(row[score_col]))
    missing = wanted - set(found)
    if missing:
        raise IndexError(f'{len(missing)} manifest rows absent from {path.name}')
    ordered = [found[int(i)] for i in row_indices]
    sequences = [s for s, _ in ordered]
    scores = np.array([v for _, v in ordered], dtype=np.float64)
    return sequences, scores


def embed_assay(model, sequences, window_start, window_len, device=DEVICE):
    """Mean-pooled embeddings for one assay's variants.

    Substitution assays preserve sequence length, so every variant shares the
    same window and no padding is needed.
    """
    token_ids = model_token_ids(model, sequences, window_start, window_len)
    batch = max(1, TOKEN_BUDGET // max(1, window_len))
    parts = []
    with torch.no_grad():
        for start in range(0, len(token_ids), batch):
            chunk = torch.from_numpy(token_ids[start:start + batch]).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=(device == 'cuda')):
                vectors = model.embed(chunk)
            parts.append(vectors.float().cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


def model_token_ids(model, sequences, window_start, window_len):
    lookup = model.residue_lookup
    stop = window_start + window_len
    out = np.empty((len(sequences), window_len), dtype=np.int64)
    for i, sequence in enumerate(sequences):
        window = sequence[window_start:stop]
        if len(window) != window_len:
            raise ValueError('Variant sequence shorter than the chosen window')
        try:
            out[i] = [lookup[aa] for aa in window]
        except KeyError as exc:
            raise ValueError(f'Unsupported residue {exc} in assay sequence') from exc
    return out


def ridge_fit_predict(train_x, train_y, test_x, alpha, device=DEVICE):
    """Ridge regression with an unpenalised intercept, via whichever of the
    primal or dual form is cheaper."""
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    z = torch.as_tensor(test_x, dtype=torch.float32, device=device)
    x_mean, y_mean = x.mean(0, keepdim=True), y.mean()
    x0, z0, y0 = x - x_mean, z - x_mean, y - y_mean
    n, d = x0.shape
    if n < d:
        gram = x0 @ x0.T
        gram.diagonal().add_(float(alpha))
        dual = torch.linalg.solve(gram, y0)
        pred = z0 @ (x0.T @ dual) + y_mean
    else:
        gram = x0.T @ x0
        gram.diagonal().add_(float(alpha))
        weights = torch.linalg.solve(gram, x0.T @ y0)
        pred = z0 @ weights + y_mean
    return pred.detach().cpu().numpy()


def standardize(train_x, *others):
    mean = train_x.mean(0, keepdims=True)
    std = np.maximum(train_x.std(0, keepdims=True), 1e-5)
    return [(a - mean) / std for a in (train_x,) + others]


def select_alpha(train_x, train_y, rng):
    """Pick the ridge penalty on an inner split of the training folds only."""
    n = len(train_y)
    n_val = max(2, int(round(INNER_VAL_FRACTION * n)))
    if n - n_val < 2:
        return RIDGE_ALPHAS[len(RIDGE_ALPHAS) // 2], float('nan')
    order = rng.permutation(n)
    val_ix, fit_ix = order[:n_val], order[n_val:]
    fit_x, val_x = standardize(train_x[fit_ix], train_x[val_ix])
    fit_y, val_y = train_y[fit_ix], train_y[val_ix]
    best_alpha, best_score = RIDGE_ALPHAS[0], -np.inf
    for alpha in RIDGE_ALPHAS:
        pred = ridge_fit_predict(fit_x, fit_y, val_x, alpha)
        score = spearman(val_y, pred)
        if np.isfinite(score) and score > best_score:
            best_alpha, best_score = alpha, score
    return best_alpha, float(best_score)


def cross_validate(features, scores, folds, rng):
    """Pooled out-of-fold Spearman, plus the mean of the per-fold values."""
    predictions = np.full(len(scores), np.nan, dtype=np.float64)
    per_fold, alphas = [], []
    for fold in range(N_FOLDS):
        test_mask = folds == fold
        train_mask = ~test_mask
        if test_mask.sum() < 3 or train_mask.sum() < 10:
            continue
        alpha, _ = select_alpha(features[train_mask], scores[train_mask], rng)
        train_x, test_x = standardize(features[train_mask], features[test_mask])
        pred = ridge_fit_predict(train_x, scores[train_mask], test_x, alpha)
        predictions[test_mask] = pred
        per_fold.append(spearman(scores[test_mask], pred))
        alphas.append(alpha)
    scored = np.isfinite(predictions)
    if scored.sum() < 10 or not per_fold:
        return None
    return {
        'spearman': spearman(scores[scored], predictions[scored]),
        'mean_fold_spearman': float(np.nanmean(per_fold)),
        'n_scored': int(scored.sum()),
        'n_folds_used': len(per_fold),
        'median_alpha': float(np.median(alphas)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True, choices=CONDITIONS)
    parser.add_argument('--seed', type=int, choices=SEEDS,
                        help='Required for every condition except BASELINE')
    parser.add_argument('--manifest', type=Path, default=BASE / 'proteingym_manifest')
    parser.add_argument('--limit', type=int, default=0,
                        help='Evaluate only the first N assays (smoke testing)')
    parser.add_argument('--max-variants', type=int, default=0,
                        help='Cap variants per assay below the manifest (smoke testing)')
    parser.add_argument('--allow-cpu', action='store_true',
                        help='Permit a CPU run; far too slow for the real sweep')
    args = parser.parse_args()
    if args.condition != 'BASELINE' and args.seed is None:
        parser.error('--seed is required unless --condition BASELINE')

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError('GPU is required (pass --allow-cpu to smoke test)')
    if not (args.manifest / 'MANIFEST_READY').is_file():
        raise RuntimeError(
            f'Build the evaluation manifest first: {args.manifest}\n'
            'Run, from the repository root:\n'
            '  python experiments/esm2_pg/fetch_proteingym.py\n'
            '  python experiments/esm2_pg/build_eval_manifest.py')

    # A truncated run is not a result. It goes to its own directory and never
    # writes SUCCESS, so it cannot be mistaken for a real evaluation and cannot
    # make the real run skip itself.
    smoke = bool(args.limit or args.max_variants)
    out = eval_dir(args.condition, args.seed)
    if smoke:
        out = out.with_name(out.name + '_smoke')
        print(f'SMOKE TEST: truncated run, writing to {out}; '
              'results are not comparable and no SUCCESS marker is written',
              flush=True)
    out.mkdir(parents=True, exist_ok=True)
    if not smoke and (out / 'SUCCESS').is_file():
        print(f'Already complete: {out}', flush=True)
        return

    assays = [a for a in read_json(args.manifest / 'assays.json') if a['status'] == 'ok']
    manifest_summary = read_json(args.manifest / 'summary.json')
    folds_npz = np.load(args.manifest / 'folds.npz')
    if args.limit:
        assays = assays[:args.limit]

    started = time.time()
    model, tokenizer = load_condition(args.condition, args.seed, device=DEVICE)
    model.residue_lookup = tokenizer.a_to_i

    rows, skipped = [], []
    for index, assay in enumerate(assays, 1):
        name = assay['DMS_id']
        selected = folds_npz[f'{name}__rows']
        keep = slice(None)
        if args.max_variants and len(selected) > args.max_variants:
            keep = slice(0, args.max_variants)
            selected = selected[keep]
        try:
            sequences, scores = load_selected(PG_ASSAYS / assay['filename'], selected)
        except (IndexError, KeyError, ValueError) as exc:
            raise RuntimeError(f'{name}: assay file no longer matches the manifest') from exc

        try:
            features = embed_assay(
                model, sequences, assay['window_start'], assay['window_len'])
        except ValueError as exc:
            skipped.append({'DMS_id': name, 'reason': str(exc)})
            continue

        for scheme in SPLIT_SCHEMES:
            folds = folds_npz[f'{name}__fold_{scheme}'].astype(np.int64)[keep]
            rng = np.random.default_rng(ALPHA_SELECTION_SEED)
            result = cross_validate(features, scores, folds, rng)
            if result is None:
                skipped.append({'DMS_id': name, 'reason': f'{scheme}:degenerate_folds'})
                continue
            rows.append({
                'condition': args.condition,
                'seed': args.seed if args.seed is not None else '',
                'DMS_id': name,
                'split_scheme': scheme,
                'n_variants': len(scores),
                'seq_len': assay['seq_len'],
                'taxon': assay['taxon'],
                'coarse_selection_type': assay['coarse_selection_type'],
                **result,
            })

        if index % 10 == 0 or index == len(assays):
            print(f'{index}/{len(assays)} assays | {time.time()-started:.0f}s', flush=True)

    if not rows:
        raise RuntimeError('No assay produced a usable correlation')

    write_csv(out / 'per_assay_metrics.csv', rows, [
        'condition', 'seed', 'DMS_id', 'split_scheme', 'n_variants', 'seq_len',
        'taxon', 'coarse_selection_type', 'spearman', 'mean_fold_spearman',
        'n_scored', 'n_folds_used', 'median_alpha'])

    summary = {
        'model': ESM2_MODEL,
        'condition': args.condition,
        'seed': args.seed,
        'proteingym_version': manifest_summary['proteingym_version'],
        'n_assays_evaluated': len({r['DMS_id'] for r in rows}),
        'n_rows': len(rows),
        'n_skipped': len(skipped),
        'skipped': skipped[:50],
        'seconds': time.time() - started,
        'metric': 'pooled out-of-fold Spearman per assay; never pooled across assays',
        'smoke_test': smoke,
        'assay_limit': args.limit or None,
        'variant_cap': args.max_variants or None,
        'status': 'SMOKE' if smoke else 'PASS',
    }
    for scheme in SPLIT_SCHEMES:
        values = [r['spearman'] for r in rows
                  if r['split_scheme'] == scheme and np.isfinite(r['spearman'])]
        if values:
            summary[f'mean_spearman_{scheme}'] = float(np.mean(values))
            summary[f'median_spearman_{scheme}'] = float(np.median(values))
    write_json(out / 'summary.json', summary)
    if smoke:
        print('SMOKE TEST COMPLETE (no SUCCESS marker written)', out, summary, flush=True)
    else:
        (out / 'SUCCESS').touch()
        print('EVALUATION SUCCESS', out, summary, flush=True)


if __name__ == '__main__':
    main()
