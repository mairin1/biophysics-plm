"""Fine-tune ESM-2 650M on METL-Global Rosetta targets (groups A+B+C).

Mirrors experiments/checkin2_v3/train.py. The METL data loading, per-protein
target standardisation, and batching are reused from
scripts/train_evodiff_stage1.py through a tokenizer shim, so the only things
that change relative to the EvoDiff runs are the encoder and the pooling.

Hyperparameters are inherited from the EvoDiff protocol rather than re-tuned
for ESM-2. Gradient accumulation keeps the effective batch at 64 while the
per-step batch stays small enough for a 650M-parameter encoder.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BASE, ESM2_MODEL, HEAD_LR, L2SP_COEFFICIENT, METHODS, SEEDS, SOURCE,
    TARGETS, run_dir, write_json,
)
from model import (  # noqa: E402
    Esm2Predictor, l2sp_penalty, load_encoder, optimizer_for, set_strategy,
)

sys.path.insert(0, 'scripts')
import train_evodiff_stage1 as metl  # noqa: E402

MAX_EPOCHS = 8
MIN_EPOCHS = 3
PATIENCE = 2
RELATIVE_MIN_DELTA = 0.002
MICRO_BATCH = 16
ACCUM_STEPS = 4  # effective batch 64, matching the EvoDiff runs


def atomic_torch_save(payload, path):
    tmp = path.with_suffix(path.suffix + '.partial')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def make_scheduler(optimizer, total_optimizer_steps):
    warm = max(1, round(0.05 * total_optimizer_steps))

    def scale(step):
        if step < warm:
            return max(0.01, (step + 1) / warm)
        progress = min(1.0, (step - warm) / max(1, total_optimizer_steps - warm))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def run_epoch(model, data, split, rng, optimizer=None, scheduler=None,
              anchor=None, denom=None):
    training = optimizer is not None
    model.train(training)
    jobs = metl.batches(data, split, MICRO_BATCH, rng, training)
    raw_total, reg_total, nrows = 0.0, 0.0, 0

    if training:
        optimizer.zero_grad(set_to_none=True)
    pending = 0

    for index, (protein, ids) in enumerate(jobs):
        subset = data[protein][split]
        tokens = torch.as_tensor(subset['tokens'][ids].astype(np.int64), device='cuda')
        labels = torch.as_tensor(subset['labels'][ids], dtype=torch.float32, device='cuda')
        mask = torch.as_tensor(subset['mask'], dtype=torch.float32, device='cuda')

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                predicted = model(tokens)
                squared = (predicted.float() - labels).square()
                raw_loss = (squared * mask).sum() / (mask.sum() * len(ids))
            if training and anchor is not None:
                regularization = L2SP_COEFFICIENT * l2sp_penalty(model, anchor, denom)
            else:
                regularization = raw_loss.detach() * 0.0
            if training:
                # The L2-SP term is a property of the weights, not of the batch,
                # so it is divided by the accumulation depth like the data loss.
                ((raw_loss + regularization) / ACCUM_STEPS).backward()
                pending += 1
                is_last = index == len(jobs) - 1
                if pending == ACCUM_STEPS or is_last:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 0.5)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0

        n = len(ids)
        nrows += n
        raw_total += float(raw_loss.detach()) * n
        reg_total += float(regularization.detach()) * n

    return raw_total / nrows, reg_total / nrows


def save_history(path, entries):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'epoch', 'train_mse', 'val_mse', 'mean_l2sp_penalty',
            'encoder_lr', 'head_lr'])
        writer.writeheader()
        writer.writerows(entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--seed', required=True, type=int, choices=SEEDS)
    parser.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required')
    torch.backends.cuda.matmul.allow_tf32 = True

    out = run_dir(args.method, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'TRAIN_COMPLETE').exists() and (out / 'best_model.pt').is_file():
        print(f'Training already completed: {out}', flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.time()

    encoder, tokenizer = load_encoder()
    model = Esm2Predictor(encoder, len(TARGETS), tokenizer.cls_id, tokenizer.eos_id).cuda()
    detail = set_strategy(model, args.method)

    data = metl.load_data(tokenizer, TARGETS, args.seed)
    stats = {protein: data[protein]['stats'] for protein in sorted(data)}

    optimizer, encoder_lr = optimizer_for(model, args.method)
    train_rows = sum(len(data[p]['train']['tokens']) for p in data)
    micro_per_epoch = sum(
        math.ceil(len(data[p]['train']['tokens']) / MICRO_BATCH) for p in data)
    steps_per_epoch = math.ceil(micro_per_epoch / ACCUM_STEPS)
    scheduler = make_scheduler(optimizer, steps_per_epoch * args.max_epochs)

    initial = {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters() if not name.startswith('head.')
    }
    if args.method == 'L2SP':
        anchor = {
            name: tensor.to('cuda', dtype=torch.float32)
            for name, tensor in initial.items()
        }
        denom = torch.stack([t.square().sum() for t in anchor.values()]).sum().detach()
        if float(denom) <= 0:
            raise RuntimeError('Degenerate L2-SP normaliser')
    else:
        anchor, denom = None, None

    n_pretrained = sum(p.numel() for name, p in model.named_parameters()
                       if p.requires_grad and not name.startswith('head.'))
    n_head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    if not n_pretrained or not n_head:
        raise RuntimeError('Strategy did not configure both parameter groups')

    config = {
        'model': ESM2_MODEL,
        'method': args.method,
        'seed': args.seed,
        'targets': TARGETS,
        'source': str(SOURCE),
        'architecture': 'ESM-2 650M; mean-pool final hidden states over residues',
        'strategy': detail,
        'trainable_pretrained_parameters': n_pretrained,
        'trainable_head_parameters': n_head,
        'encoder_lr': encoder_lr,
        'head_lr': HEAD_LR,
        'weight_decay': 0.01,
        'l2sp_coefficient': L2SP_COEFFICIENT if args.method == 'L2SP' else 0.0,
        'micro_batch': MICRO_BATCH,
        'accumulation_steps': ACCUM_STEPS,
        'effective_batch': MICRO_BATCH * ACCUM_STEPS,
        'max_epochs': args.max_epochs,
        'min_epochs': MIN_EPOCHS,
        'patience': PATIENCE,
        'relative_min_delta': RELATIVE_MIN_DELTA,
        'scheduler': '5% warmup + cosine to 10% of the initial rate',
        'selection': 'minimum Rosetta validation MSE on the six source proteins',
        'train_rows': train_rows,
        'notes': (
            'Hyperparameters inherited from the EvoDiff Check-in 2 protocol and '
            'not re-tuned for ESM-2; no held-out protein or ProteinGym outcome '
            'is used for checkpoint selection'
        ),
    }
    write_json(out / 'config.json', config)

    best_loss, best_epoch = math.inf, 0
    history, start_epoch = [], 1
    state_path = out / 'last_training_state.pt'
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        if state['config'] != config:
            raise RuntimeError('Resume checkpoint config differs; refusing to mix runs')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        rng = np.random.default_rng()
        rng.bit_generator.state = state['numpy_rng']
        best_loss, best_epoch = state['best_loss'], state['best_epoch']
        history, start_epoch = state['history'], state['epoch'] + 1
        print(f'Resuming at epoch {start_epoch}', flush=True)
    else:
        rng = np.random.default_rng(args.seed)

    for epoch in range(start_epoch, args.max_epochs + 1):
        train_mse, reg = run_epoch(
            model, data, 'train', rng, optimizer, scheduler, anchor, denom)
        val_mse, _ = run_epoch(model, data, 'val', rng)
        if val_mse < best_loss * (1 - RELATIVE_MIN_DELTA):
            best_loss, best_epoch = val_mse, epoch
            atomic_torch_save({
                'model_state_dict': model.state_dict(),
                'targets': TARGETS, 'condition': args.method, 'seed': args.seed,
                'normalization': stats, 'best_epoch': epoch,
                'best_val_mse': val_mse, 'config': config,
            }, out / 'best_model.pt')
        history.append({
            'epoch': epoch, 'train_mse': train_mse, 'val_mse': val_mse,
            'mean_l2sp_penalty': reg,
            'encoder_lr': optimizer.param_groups[0]['lr'],
            'head_lr': optimizer.param_groups[-1]['lr'],
        })
        save_history(out / 'history.csv', history)
        atomic_torch_save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'numpy_rng': rng.bit_generator.state, 'best_loss': best_loss,
            'best_epoch': best_epoch, 'history': history, 'config': config,
        }, state_path)
        print(f'{args.method} seed={args.seed} epoch={epoch}: train={train_mse:.5f} '
              f'val={val_mse:.5f} L2-SP={reg:.6f} best={best_loss:.5f}@{best_epoch}',
              flush=True)
        if epoch >= MIN_EPOCHS and epoch - best_epoch >= PATIENCE:
            print('Early stopping', flush=True)
            break

    if not (out / 'best_model.pt').is_file():
        raise RuntimeError('No validation-selected checkpoint saved')

    saved = torch.load(out / 'best_model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model_state_dict'])
    delta_sum = initial_sum = frozen_delta = 0.0
    for name, p in model.named_parameters():
        if name not in initial:
            continue
        diff = p.detach().cpu().float() - initial[name].float()
        delta_sum += float(diff.square().sum())
        initial_sum += float(initial[name].float().square().sum())
        if not p.requires_grad:
            frozen_delta += float(diff.abs().sum())
    if args.method == 'PARTIAL' and frozen_delta != 0.0:
        raise RuntimeError('Frozen parameters changed unexpectedly')

    summary = {
        'model': ESM2_MODEL,
        'method': args.method, 'seed': args.seed, 'best_epoch': best_epoch,
        'last_epoch': history[-1]['epoch'], 'best_val_mse': best_loss,
        'relative_pretrained_weight_change': math.sqrt(delta_sum / initial_sum),
        'frozen_weight_absolute_change': frozen_delta,
        'seconds': time.time() - started, 'status': 'TRAIN_COMPLETE',
    }
    write_json(out / 'train_summary.json', summary)
    # The resume state carries full AdamW moments for a 650M-parameter encoder,
    # roughly 8GB per run. Once the run has finished it is dead weight, and
    # keeping nine of them would cost ~70GB of pool quota.
    state_path.unlink(missing_ok=True)
    (out / 'TRAIN_COMPLETE').touch()
    print('TRAIN_COMPLETE', out, summary, flush=True)


if __name__ == '__main__':
    main()
