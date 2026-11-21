from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch import nn
from evodiff.pretrained import OA_DM_38M

FAMILIES = {
    'A': ['fa_dun', 'omega', 'p_aa_pp', 'rama_prepro', 'pro_close', 'yhh_planarity'],
    'B': ['fa_elec', 'fa_rep', 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb'],
    'C': ['fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'cbeta', 'cenpack', 'env'],
}
CONDITIONS = {'A': ['A'], 'B': ['B'], 'C': ['C'], 'AB': ['A', 'B'], 'BC': ['B', 'C'], 'ABC': ['A', 'B', 'C']}
RESERVED = ['fa_atr', 'fa_sol']
SEEDS = [20261116, 20261117, 20261118]
RUN_ROOT = Path('metl-global/analysis/evodiff_stage1_20261116')
SOURCE = Path('metl-global/exports/run_29051193')


class BiophysicsPredictor(nn.Module):
    def __init__(self, backbone, n):
        super().__init__()
        self.embedder = backbone.embedder
        self.last_norm = backbone.last_norm
        self.head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n))

    def encode(self, x):
        return self.last_norm(self.embedder(x, None)).mean(dim=1)

    def forward(self, x):
        return self.head(self.encode(x))


def load_data(tokenizer, targets, seed):
    rng = np.random.default_rng(48629)
    data = defaultdict(lambda: defaultdict(lambda: {'tokens': [], 'labels': [], 'probe': [], 'row_ids': []}))
    counts = Counter()
    with gzip.open(SOURCE / 'sequences.csv.gz', 'rt') as sf, gzip.open(SOURCE / 'rosetta_numeric.csv.gz', 'rt') as yf:
        sq, yr = csv.DictReader(sf), csv.DictReader(yf)
        assert set(targets + RESERVED).issubset(yr.fieldnames)
        for s, y in zip_longest(sq, yr):
            assert s is not None and y is not None
            assert s['row_position'] == y['row_position']
            if s['role'] != 'train' or s['split'] not in ('train', 'val', 'test'):
                continue
            p, split = s['pdb_fn'], s['split']
            counts[(p, split)] += 1
            d = data[p][split]
            d['tokens'].append([tokenizer.a_to_i[a] for a in s['sequence']])
            d['labels'].append([float(y[t]) for t in targets])
            d['probe'].append([float(y[t]) for t in RESERVED])
            d['row_ids'].append(int(s['row_position']))
    assert len(data) == 6, len(data)
    for p, subs in data.items():
        for sp, expected in [('train', 40000), ('val', 2000), ('test', 2000)]:
            assert counts[(p, sp)] == expected, (p, sp, counts[(p, sp)])
            d = subs[sp]
            d['tokens'] = np.asarray(d['tokens'], dtype=np.uint8)
            d['labels'] = np.asarray(d['labels'], dtype=np.float32)
            d['probe'] = np.asarray(d['probe'], dtype=np.float32)
            d['row_ids'] = np.asarray(d['row_ids'], dtype=np.int64)
            assert np.isfinite(d['labels']).all() and np.isfinite(d['probe']).all()
        tr = subs['train']['labels']
        mu = tr.mean(0, dtype=np.float64).astype(np.float32)
        std = tr.std(0, dtype=np.float64).astype(np.float32)
        mask = (std > 1e-4).astype(np.float32)
        if mask.sum() < 3:
            raise ValueError(f'Insufficient target variation {p}: {mask}')
        subs['stats'] = {'mean': mu.tolist(), 'std': std.tolist(), 'active': mask.astype(bool).tolist()}
        for sp in ['train', 'val', 'test']:
            d = subs[sp]
            d['labels'] = (d['labels'] - mu) / np.where(mask, std, 1.0)
            d['labels'][:, mask == 0] = 0
            d['mask'] = mask
        assert np.isfinite(subs['train']['labels']).all()
        print(f'{p}: targets active {int(mask.sum())}/{len(mask)}', flush=True)
    return data


def batches(data, split, batch_size, rng, training):
    jobs = []
    for p in sorted(data):
        n = len(data[p][split]['tokens'])
        ix = rng.permutation(n) if training else np.arange(n)
        for start in range(0, n, batch_size):
            jobs.append((p, ix[start:start + batch_size]))
    if training:
        rng.shuffle(jobs)
    return jobs


def epoch(model, data, split, batch_size, rng, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    numerator, denom = 0.0, 0
    per_sum = defaultdict(lambda: np.zeros(len(next(iter(data.values()))['stats']['active']), dtype=np.float64))
    per_rows = Counter()
    for p, ids in batches(data, split, batch_size, rng, training):
        d = data[p][split]
        tokens = torch.as_tensor(d['tokens'][ids].astype(np.int64), device='cuda')
        label = torch.as_tensor(d['labels'][ids], device='cuda')
        mask = torch.as_tensor(d['mask'], device='cuda')
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(tokens)
                sq = (out.float() - label).square()
                loss = (sq * mask).sum() / (mask.sum() * len(ids))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()
        n = len(ids)
        numerator += loss.detach().item() * n
        denom += n
        if not training:
            per_sum[p] += (sq.detach().float() * mask).sum(0).cpu().numpy()
            per_rows[p] += n
    per_target = {p: (per_sum[p] / max(1, per_rows[p])).tolist() for p in per_sum}
    return numerator / denom, per_target


def make_optimizer(model):
    return torch.optim.AdamW([
        {'params': model.embedder.parameters(), 'lr': 2e-5},
        {'params': model.last_norm.parameters(), 'lr': 2e-5},
        {'params': model.head.parameters(), 'lr': 2e-4},
    ], weight_decay=0.01)


def atomic_save(value, path):
    temp = path.with_name(path.name + '.partial')
    torch.save(value, temp)
    os.replace(temp, path)


def train(model, data, targets, condition, seed, out, resume=False):
    optimizer = make_optimizer(model)
    max_epochs, batch_size, patience, min_epochs = 40, 64, 6, 10
    iters = math.ceil(40000 / batch_size) * 6
    total_steps = iters * max_epochs
    warm_steps = max(1, round(total_steps * .05))
    def scale(step):
        if step < warm_steps:
            return max(0.01, (step + 1) / warm_steps)
        s = min(1, (step - warm_steps) / max(1, total_steps - warm_steps))
        return 0.1 + 0.9 * .5 * (1 + math.cos(math.pi * s))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    rng = np.random.default_rng(seed)
    initial = {k: v.detach().cpu().clone() for k, v in model.embedder.named_parameters()}
    stats = {p: data[p]['stats'] for p in sorted(data)}
    config = {'condition': condition, 'targets': targets, 'seed': seed, 'source': str(SOURCE),
              'epochs_max': max_epochs, 'patience': patience, 'min_epochs': min_epochs,
              'batch_size': batch_size, 'encoder_lr': 2e-5, 'head_lr': 2e-4, 'weight_decay': .01,
              'clip_grad_norm': .5, 'warmup_fraction': .05, 'minimum_lr_fraction': .1,
              'relative_min_delta': .002, 'selection': 'fine-tuning validation MSE',
              'normalization': 'train-only per protein and target',
              'note': 'This protocol measures within-protein diagnostics; not experimental transfer or independent held-out-protein generalization.'}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    start_epoch, best_loss, best_epoch = 1, math.inf, 0
    hist = []
    if resume and (out / 'last_training_state.pt').exists():
        state = torch.load(out / 'last_training_state.pt', map_location='cpu', weights_only=False)
        assert state['config'] == config
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        rng.bit_generator.state = state['numpy_rng']
        best_loss, best_epoch = state['best_loss'], state['best_epoch']
        hist = state['history']
        start_epoch = state['epoch'] + 1
    began = time.time()
    for ep in range(start_epoch, max_epochs + 1):
        tloss, _ = epoch(model, data, 'train', batch_size, rng, optimizer, scheduler)
        vloss, per_target = epoch(model, data, 'val', batch_size, rng)
        hist.append({'epoch': ep, 'train_mse': tloss, 'val_mse': vloss,
                     'encoder_lr': optimizer.param_groups[0]['lr'],
                     'head_lr': optimizer.param_groups[-1]['lr']})
        print(f'{condition} seed={seed} epoch={ep}: train={tloss:.6f}, val={vloss:.6f}, best={best_loss:.6f}', flush=True)
        improved = vloss < best_loss * (1 - .002)
        if improved:
            best_loss, best_epoch = vloss, ep
            atomic_save({'model_state_dict': model.state_dict(), 'targets': targets,
                         'condition': condition, 'seed': seed, 'normalization': stats,
                         'best_epoch': ep, 'best_val_mse': vloss}, out / 'best_model.pt')
            with open(out / 'best_val_per_target.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['protein', 'target', 'val_standardized_mse', 'active'])
                for p in sorted(per_target):
                    for name, val, active in zip(targets, per_target[p], stats[p]['active']):
                        w.writerow([p, name, val, int(active)])
        with open(out / 'history.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(hist[0]))
            w.writeheader()
            w.writerows(hist)
        stop = ep >= min_epochs and ep - best_epoch >= patience
        atomic_save({'epoch': ep, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all(), 'numpy_rng': rng.bit_generator.state,
                     'best_loss': best_loss, 'best_epoch': best_epoch, 'history': hist,
                     'config': config}, out / 'last_training_state.pt')
        if stop:
            print(f'Early stopping after {ep} epochs, best epoch {best_epoch}', flush=True)
            break
    best = torch.load(out / 'best_model.pt', weights_only=True, map_location='cpu')
    model.load_state_dict(best['model_state_dict'])
    delta_sum = 0.0
    orig_sum = 0.0
    for name, param in model.embedder.named_parameters():
        diff = param.detach().cpu().float() - initial[name].float()
        delta_sum += float(diff.square().sum())
        orig_sum += float(initial[name].float().square().sum())
    delta = (delta_sum / orig_sum) ** .5
    assert delta > 0
    return {'condition': condition, 'seed': seed, 'targets': targets, 'n_targets': len(targets),
            'best_epoch': best_epoch, 'last_epoch': hist[-1]['epoch'], 'best_val_mse': best_loss,
            'relative_encoder_delta': delta, 'seconds_in_this_invocation': time.time()-began,
            'status': 'TRAINED'}


def ranks(a):
    a = np.asarray(a)
    order = np.argsort(a, kind='mergesort')
    sr = a[order]
    result = np.zeros(len(a), dtype=float)
    cuts = np.r_[0, np.flatnonzero(sr[1:] != sr[:-1]) + 1, len(a)]
    for i, j in zip(cuts[:-1], cuts[1:]):
        result[order[i:j]] = (i + j + 1) / 2
    return result


def evaluate_probes(model, data, out):
    rng = np.random.default_rng(654321)
    selected = {'train': 2000, 'val': 500, 'test': 2000}
    records = {}
    model.eval()
    with torch.no_grad():
        for p in sorted(data):
            for split, count in selected.items():
                d = data[p][split]
                ix = np.sort(rng.choice(len(d['tokens']), size=count, replace=False)) if count < len(d['tokens']) else np.arange(len(d['tokens']))
                embs = []
                for start in range(0, len(ix), 64):
                    tokens = torch.as_tensor(d['tokens'][ix[start:start+64]].astype(np.int64), device='cuda')
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        feats = model.encode(tokens)
                    embs.append(feats.float().cpu().numpy())
                records[(p, split)] = {'x': np.concatenate(embs), 'y': d['probe'][ix],
                                       'row_ids': d['row_ids'][ix]}
    mu_y = {}
    std_y = {}
    for p in sorted(data):
        y = records[(p, 'train')]['y'].astype(np.float64)
        mu_y[p], std_y[p] = y.mean(0), y.std(0)
        if (std_y[p] < 1e-8).any():
            raise ValueError(f'Constant reserved property in probe training subset for {p}')
    allx = {}
    ally = {}
    labels = {}
    for sp in ['train', 'val', 'test']:
        allx[sp] = np.concatenate([records[(p, sp)]['x'] for p in sorted(data)], axis=0)
        ally[sp] = np.concatenate([(records[(p, sp)]['y'] - mu_y[p]) / std_y[p] for p in sorted(data)], axis=0)
        labels[sp] = [(p, records[(p, sp)]['y'], records[(p, sp)]['row_ids']) for p in sorted(data)]
    xt = torch.as_tensor(allx['train'], dtype=torch.float32, device='cuda')
    mx = xt.mean(0, keepdim=True)
    sx = xt.std(0, keepdim=True).clamp(min=1e-5)
    x_train = (xt - mx) / sx
    y_train = torch.as_tensor(ally['train'], dtype=torch.float32, device='cuda')
    one = torch.ones((len(x_train), 1), device='cuda')
    x_train = torch.cat([x_train, one], 1)
    gram = x_train.T @ x_train
    ridge_diag = torch.full((x_train.shape[1],), 100.0, device='cuda')
    ridge_diag[-1] = 0
    gram.diagonal().add_(ridge_diag)
    weights = torch.linalg.solve(gram, x_train.T @ y_train)
    output = []
    for sp in ['val', 'test']:
        x = torch.as_tensor(allx[sp], dtype=torch.float32, device='cuda')
        x = torch.cat([(x - mx) / sx, torch.ones((len(x), 1), device='cuda')], 1)
        pred = (x @ weights).cpu().numpy()
        offset = 0
        for p, observed, row_ids in labels[sp]:
            n = len(observed)
            yp = pred[offset:offset+n]
            yt = ally[sp][offset:offset+n]
            offset += n
            for k, target in enumerate(RESERVED):
                true = yt[:,k]
                estimated = yp[:,k]
                sst = float(((true-true.mean())**2).sum())
                r2 = float(1 - ((estimated-true)**2).sum() / sst) if sst > 1e-9 else float('nan')
                mae = float(np.abs(estimated-true).mean())
                correlation = float(np.corrcoef(ranks(true), ranks(estimated))[0,1])
                output.append({'protein': p, 'split': sp, 'target': target, 'n': n,
                               'r2': r2, 'spearman': correlation, 'mae_standardized': mae})
    with open(out / 'probe_metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(output[0]))
        w.writeheader()
        w.writerows(output)
    (out / 'probe_config.json').write_text(json.dumps({'properties': RESERVED, 'support_per_training_protein': 2000,
                 'validation_per_training_protein': 500, 'within_protein_test_per_protein': 2000,
                 'ridge_alpha': 100.0, 'feature_standardization': 'global train probe only',
                 'target_standardization': 'per protein, train probe only',
                 'fixed_probe_sampling_seed': 654321,
                 'warning': 'Within-protein local interpolation only; training protein variants seen during encoder fine-tuning.'}, indent=2))
    return len(output)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--condition', required=True, choices=list(CONDITIONS) + ['BASELINE'])
    pa.add_argument('--seed', required=True, type=int)
    pa.add_argument('--resume', action='store_true')
    args = pa.parse_args()
    if not (RUN_ROOT / 'AUDIT_OK').exists():
        raise RuntimeError('Audit must finish successfully before GPU experiments start')
    if args.seed not in SEEDS:
        raise ValueError('Unknown seed')
    condition = args.condition
    out = RUN_ROOT / 'runs' / f'{condition}_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'SUCCESS').exists():
        print(f'Already complete: {out}', flush=True)
        return
    targets = [x for family in CONDITIONS.get(condition, []) for x in FAMILIES[family]]
    if condition == 'BASELINE':
        targets = FAMILIES['A']
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    assert torch.cuda.is_available(), 'GPU required'
    torch.backends.cuda.matmul.allow_tf32 = True
    backbone, _, tokenizer, _ = OA_DM_38M()
    data = load_data(tokenizer, targets, args.seed)
    model = BiophysicsPredictor(backbone, len(targets)).cuda()
    print('GPU', torch.cuda.get_device_name(0), 'condition', condition, 'seed', args.seed, flush=True)
    if condition == 'BASELINE':
        summary = {'condition': 'BASELINE', 'seed': args.seed, 'n_targets': 0,
                   'status': 'PRETRAINED_FROZEN', 'train_rows': 0}
    else:
        summary = train(model, data, targets, condition, args.seed, out, args.resume)
    n = evaluate_probes(model, data, out)
    summary['probe_rows'] = n
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'SUCCESS').touch()
    print('SUCCESS', out, flush=True)


if __name__ == '__main__':
    main()
