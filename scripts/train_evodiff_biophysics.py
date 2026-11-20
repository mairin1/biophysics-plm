from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
import argparse
import csv
import gzip
import json
import random
import os

import numpy as np
import torch
from torch import nn
from evodiff.pretrained import OA_DM_38M

TARGETS = [
    'fa_dun', 'fa_elec', 'fa_intra_rep', 'fa_intra_sol_xover4',
    'fa_rep', 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc',
    'hbond_sr_bb', 'lk_ball_wtd', 'omega', 'p_aa_pp',
    'pro_close', 'rama_prepro', 'yhh_planarity', 'cbeta',
    'cenpack', 'env'
]
RESERVED = ['fa_atr', 'fa_sol', 'total_sasa', 'rg', 'pack', 'unsat_hbond']


def read_data(folder, tokenizer, max_train, max_val):
    proteins = defaultdict(lambda: defaultdict(lambda: {'tokens': [], 'labels': []}))
    counts = Counter()
    with gzip.open(folder / 'sequences.csv.gz', 'rt') as sf, gzip.open(folder / 'rosetta_numeric.csv.gz', 'rt') as lf:
        seqs, labels = csv.DictReader(sf), csv.DictReader(lf)
        assert set(TARGETS + RESERVED).issubset(labels.fieldnames)
        for s, l in zip_longest(seqs, labels):
            assert s is not None and l is not None
            assert s['row_position'] == l['row_position']
            if s['role'] != 'train' or s['split'] not in ('train', 'val'):
                continue
            protein, split = s['pdb_fn'], s['split']
            counts[(protein, split)] += 1
            limit = max_train if split == 'train' else max_val
            if limit and len(proteins[protein][split]['tokens']) >= limit:
                continue
            proteins[protein][split]['tokens'].append([tokenizer.a_to_i[a] for a in s['sequence']])
            proteins[protein][split]['labels'].append([float(l[t]) for t in TARGETS])
    assert len(proteins) == 6, len(proteins)
    assert len(counts) == 12, counts
    assert all(counts[(p, 'train')] == 40000 and counts[(p, 'val')] == 2000 for p in proteins), counts
    return proteins


def normalize(proteins):
    stats = {}
    for protein, subsets in proteins.items():
        train = np.asarray(subsets['train']['labels'], dtype=np.float32)
        mu = train.mean(0, dtype=np.float64).astype(np.float32)
        sd = train.std(0, dtype=np.float64).astype(np.float32)
        mask = (sd > 1e-4).astype(np.float32)
        assert mask.sum() >= 8, (protein, mask.sum())
        stats[protein] = {'mean': mu.tolist(), 'std': sd.tolist(), 'active': mask.astype(bool).tolist()}
        for split, subset in subsets.items():
            tokens = np.asarray(subset['tokens'], dtype=np.uint8)
            assert tokens.ndim == 2
            y = np.asarray(subset['labels'], dtype=np.float32)
            y = (y - mu) / np.where(mask > 0, sd, 1.0)
            y[:, mask == 0] = 0.0
            assert np.isfinite(y).all(), (protein, split)
            subset['tokens'] = tokens
            subset['labels'] = y
            subset['mask'] = mask
        print(protein, 'active targets:', int(mask.sum()), 'length:', len(tokens[0]), flush=True)
    return stats


class Predictor(nn.Module):
    def __init__(self, backbone, n_targets):
        super().__init__()
        self.embedder = backbone.embedder
        self.last_norm = backbone.last_norm
        self.head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_targets))

    def forward(self, x):
        embeddings = self.last_norm(self.embedder(x, None))
        return self.head(embeddings.mean(dim=1))

def load_baseline_model():
    backbone, _, tokenizer, _ = OA_DM_38M()
    model = Predictor(backbone, len(TARGETS)).cuda()
    return model, tokenizer


def run_epoch(model, datasets, optimizer, batch_size, rng, device):
    training = optimizer is not None
    model.train(training)
    jobs = []
    for protein, subset in datasets.items():
        positions = rng.permutation(len(subset['tokens'])) if training else np.arange(len(subset['tokens']))
        for start in range(0, len(positions), batch_size):
            jobs.append((protein, positions[start:start + batch_size]))
    if training:
        rng.shuffle(jobs)
    sum_loss, total_rows = 0.0, 0
    for protein, indices in jobs:
        subset = datasets[protein]
        tokens = torch.as_tensor(subset['tokens'][indices].astype(np.int64), device=device)
        targets = torch.as_tensor(subset['labels'][indices], dtype=torch.float32, device=device)
        mask = torch.as_tensor(subset['mask'], dtype=torch.float32, device=device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                predicted = model(tokens)
                err = (predicted.float() - targets).square()
                loss = (err * mask).sum() / (mask.sum() * len(indices))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        sum_loss += float(loss.detach()) * len(indices)
        total_rows += len(indices)
    return sum_loss / total_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-train-per-protein', type=int, default=0)
    parser.add_argument('--max-val-per-protein', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261116)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, tokenizer = load_baseline_model()
    datasets = read_data(args.source, tokenizer, args.max_train_per_protein, args.max_val_per_protein)
    stats = normalize(datasets)
    initial = {name: param.detach().cpu().clone() for name, param in model.embedder.named_parameters()}
    opt = torch.optim.AdamW([
        {'params': model.embedder.parameters(), 'lr': 2e-5},
        {'params': model.last_norm.parameters(), 'lr': 2e-5},
        {'params': model.head.parameters(), 'lr': 2e-4},
    ], weight_decay=0.01)
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Train:', sum(len(d['train']['tokens']) for d in datasets.values()), flush=True)
    print('Val:', sum(len(d['val']['tokens']) for d in datasets.values()), flush=True)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    trainsets = {p: d['train'] for p, d in datasets.items()}
    valsets = {p: d['val'] for p, d in datasets.items()}
    best_loss, best_epoch = float('inf'), 0
    with open(args.out / 'history.csv', 'w', newline='') as history:
        writer = csv.writer(history)
        writer.writerow(['epoch', 'train_mse', 'val_mse'])
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, trainsets, opt, args.batch_size, rng, 'cuda')
            val_loss = run_epoch(model, valsets, None, args.batch_size, rng, 'cuda')
            writer.writerow([epoch, train_loss, val_loss])
            history.flush()
            print(f'Epoch {epoch}: train={train_loss:.5f} val={val_loss:.5f}', flush=True)
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'targets': TARGETS,
                    'reserved': RESERVED,
                    'normalization': stats,
                    'best_epoch': epoch,
                    'val_mse': val_loss,
                    'seed': seed,
                }, args.out / 'best_model.pt.partial')
                os.replace(args.out / 'best_model.pt.partial', args.out / 'best_model.pt')
            if epoch - best_epoch >= 3:
                print('Early stop', flush=True)
                break
    best = torch.load(args.out / 'best_model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best['model_state_dict'])
    squared_delta, squared_initial = 0.0, 0.0
    for name, param in model.embedder.named_parameters():
        diff = param.detach().cpu().float() - initial[name].float()
        squared_delta += float(diff.square().sum())
        squared_initial += float(initial[name].float().square().sum())
    relative_delta = (squared_delta / squared_initial) ** 0.5
    assert relative_delta > 0, 'Encoder weights did not change'
    with open(args.out / 'summary.json', 'w') as f:
        json.dump({
            'model': 'EvoDiff OA_DM_38M, encoder+multitask head updated',
            'targets': TARGETS, 'reserved': RESERVED,
            'epochs': args.epochs, 'batch_size': args.batch_size, # adding args for reproducibility
            'seed': seed, 'best_epoch': best_epoch, 'best_val_mse': best_loss,
            'relative_encoder_weight_change': relative_delta,
            'train_rows': sum(len(s['tokens']) for s in trainsets.values()),
            'val_rows': sum(len(s['tokens']) for s in valsets.values()),
            'normalization': 'per protein, per target; training split only',
        }, f, indent=2)
    (args.out / 'SUCCESS').touch()
    print('POST-TRAINING COMPLETE | relative encoder delta:', relative_delta, flush=True)


if __name__ == '__main__':
    main()
