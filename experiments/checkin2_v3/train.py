import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from evodiff.pretrained import OA_DM_38M

from common import BASE, METHODS, SEEDS, SOURCE, TARGETS, run_dir, write_json
sys.path.insert(0, 'scripts')
import train_evodiff_stage1 as previous

MAX_EPOCHS = 40
BATCH_SIZE = 64
PATIENCE = 6
MIN_EPOCHS = 10
RELATIVE_MIN_DELTA = 0.002
L2SP_COEFFICIENT = 50.0


def atomic_torch_save(payload, path):
    tmp = path.with_suffix(path.suffix + '.partial')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def set_strategy(model, method):
    if method == 'PARTIAL':
        layers = model.embedder.layers
        if not isinstance(layers, torch.nn.ModuleList) or len(layers) < 8:
            raise RuntimeError('Cannot identify the ordered ByteNet convolutional stack')
        for parameter in model.embedder.parameters():
            parameter.requires_grad_(False)
        n = max(1, math.ceil(len(layers) / 4))
        for layer in list(layers)[-n:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        detail = {'total_bytenet_blocks':len(layers), 'trainable_last_blocks':n,
                  'frozen_first_blocks':len(layers)-n,
                  'token_embedding_frozen':True, 'last_norm_trainable':True}
    else:
        detail = {'full_bytenet_trainable':True, 'last_norm_trainable':True}
    for p in model.last_norm.parameters():
        p.requires_grad_(True)
    for p in model.head.parameters():
        p.requires_grad_(True)
    return detail


def optimizer_for(model, method):
    encoder_lr = 2e-6 if method == 'GENTLE' else 2e-5
    groups = [
        {'params':[p for p in model.embedder.parameters() if p.requires_grad], 'lr':encoder_lr},
        {'params':[p for p in model.last_norm.parameters() if p.requires_grad], 'lr':encoder_lr},
        {'params':[p for p in model.head.parameters() if p.requires_grad], 'lr':2e-4},
    ]
    if any(len(g['params']) == 0 for g in groups):
        raise RuntimeError('A trainable parameter group is unexpectedly empty')
    return torch.optim.AdamW(groups, weight_decay=0.01), encoder_lr


def penalty_for(model, anchor, denom):
    diffs = [(p.float()-anchor[name]).square().sum()
             for name,p in model.named_parameters()
             if name in anchor and p.requires_grad]
    if not diffs:
        raise RuntimeError('L2-SP has no trainable pretrained parameters')
    return torch.stack(diffs).sum() / denom


def run_epoch(model, data, split, rng, optimizer=None, scheduler=None, anchor=None, denom=None):
    training = optimizer is not None
    model.train(training)
    raw_total = 0.0
    regularization_total = 0.0
    nrows = 0
    batch_jobs = previous.batches(data, split, BATCH_SIZE, rng, training)
    for protein, ids in batch_jobs:
        d = data[protein][split]
        x = torch.as_tensor(d['tokens'][ids].astype(np.int64), device='cuda')
        y = torch.as_tensor(d['labels'][ids], dtype=torch.float32, device='cuda')
        mask = torch.as_tensor(d['mask'], dtype=torch.float32, device='cuda')
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = model(x)
                squared = (pred.float() - y).square()
                raw_loss = (squared * mask).sum() / (mask.sum() * len(ids))
            if training and anchor is not None:
                regularization = L2SP_COEFFICIENT * penalty_for(model, anchor, denom)
            else:
                regularization = raw_loss.detach() * 0.0
            loss = raw_loss + regularization
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],0.5)
                optimizer.step()
                scheduler.step()
        n = len(ids)
        nrows += n
        raw_total += float(raw_loss.detach()) * n
        regularization_total += float(regularization.detach()) * n
    return raw_total / nrows, regularization_total / nrows


def save_history(path, entries):
    with path.open('w', newline='') as f:
        fields = ['epoch', 'train_mse', 'val_mse', 'mean_l2sp_penalty', 'encoder_lr', 'head_lr']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)


def make_scheduler(optimizer):
    steps_per_epoch = math.ceil(40000 / BATCH_SIZE) * 6
    max_steps = steps_per_epoch * MAX_EPOCHS
    warm = max(1, round(0.05 * max_steps))
    def scale(step):
        if step < warm:
            return max(0.01, (step+1)/warm)
        t = min(1.0, (step-warm) / max(1, max_steps-warm))
        return 0.1 + 0.9 * 0.5 * (1+math.cos(math.pi*t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--method', required=True, choices=METHODS)
    pa.add_argument('--seed', required=True, type=int, choices=SEEDS)
    args = pa.parse_args()
    if not (BASE/'GPU_READY').is_file():
        raise RuntimeError('GPU interface and retention audit must pass before training')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required')
    torch.backends.cuda.matmul.allow_tf32 = True
    out = run_dir(args.method,args.seed)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'TRAIN_COMPLETE').exists() and (out/'best_model.pt').is_file():
        print(f'Training already completed: {out}', flush=True)
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    start=time.time()
    backbone, _, tokenizer, scheme = OA_DM_38M()
    if scheme != 'mask':
        raise RuntimeError(f'Unexpected original pretrained scheme: {scheme}')
    model = previous.BiophysicsPredictor(backbone,len(TARGETS)).cuda()
    detail = set_strategy(model,args.method)
    n_pretrained_trainable = sum(p.numel() for name,p in model.named_parameters()
                                 if p.requires_grad and not name.startswith('head.'))
    n_head_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    if not n_pretrained_trainable or not n_head_trainable:
        raise RuntimeError('Strategy did not configure pretrained and head parameters correctly')
    data = previous.load_data(tokenizer,TARGETS,args.seed)
    if sorted(data) != sorted(json.loads((BASE/'preflight.json').read_text())['source_proteins']):
        raise RuntimeError('Source-protein selection changed from the preflight')
    stats = {p:data[p]['stats'] for p in sorted(data)}
    optimizer, encoder_lr = optimizer_for(model,args.method)
    scheduler = make_scheduler(optimizer)
    initial = {name:p.detach().cpu().clone() for name,p in model.named_parameters() if not name.startswith('head.')}
    if args.method == 'L2SP':
        anchor = {name:t.to('cuda', dtype=torch.float32) for name,t in initial.items()}
        denom = torch.stack([t.square().sum() for t in anchor.values()]).sum().detach()
        assert float(denom) > 0.0
    else:
        anchor, denom = None, None
    config = {
        'method':args.method,'seed':args.seed,'targets':TARGETS,'source':str(SOURCE),
        'architecture':'EvoDiff OA_DM_38M ByteNet; mean-pool last normalized states',
        'strategy':detail,'trainable_pretrained_parameters':n_pretrained_trainable,
        'trainable_head_parameters':n_head_trainable,
        'encoder_lr':encoder_lr,'head_lr':2e-4,'weight_decay':0.01,
        'l2sp_coefficient':L2SP_COEFFICIENT if args.method=='L2SP' else 0.0,
        'l2sp_definition':'coefficient * squared_L2_distance_from_pretrained / squared_pretrained_L2_norm',
        'max_epochs':MAX_EPOCHS,'batch_size':BATCH_SIZE,'min_epochs':MIN_EPOCHS,
        'patience':PATIENCE,'relative_min_delta':RELATIVE_MIN_DELTA,
        'scheduler':'5% warmup + cosine to 10% initial rate, 40-epoch horizon',
        'selection':'minimum Rosetta validation MSE on the original six source proteins',
        'notes':'Comparison is exploratory and does not select checkpoints using held-out protein outcomes'
    }
    write_json(out/'config.json',config)
    best_loss = math.inf
    best_epoch = 0
    history=[]
    start_epoch=1
    if (out/'last_training_state.pt').exists():
        state = torch.load(out/'last_training_state.pt',map_location='cpu',weights_only=False)
        if state['config'] != config:
            raise RuntimeError('Resume checkpoint config differs; refusing to mix runs')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        rng = np.random.default_rng()
        rng.bit_generator.state = state['numpy_rng']
        best_loss,best_epoch=state['best_loss'],state['best_epoch']
        history=state['history']
        start_epoch=state['epoch']+1
        print(f'Resuming at epoch {start_epoch}',flush=True)
    else:
        rng=np.random.default_rng(args.seed)
    for epoch in range(start_epoch,MAX_EPOCHS+1):
        train_mse,reg = run_epoch(model,data,'train',rng,optimizer,scheduler,anchor,denom)
        val_mse,_ = run_epoch(model,data,'val',rng)
        improved = val_mse < best_loss*(1-RELATIVE_MIN_DELTA)
        if improved:
            best_loss,best_epoch=val_mse,epoch
            atomic_torch_save({
                'model_state_dict':model.state_dict(), 'targets':TARGETS,
                'condition':args.method,'seed':args.seed,'normalization':stats,
                'best_epoch':epoch,'best_val_mse':val_mse,'config':config
            },out/'best_model.pt')
        history.append({'epoch':epoch,'train_mse':train_mse,'val_mse':val_mse,
                        'mean_l2sp_penalty':reg,'encoder_lr':optimizer.param_groups[0]['lr'],
                        'head_lr':optimizer.param_groups[-1]['lr']})
        save_history(out/'history.csv',history)
        atomic_torch_save({
            'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':rng.bit_generator.state,
            'best_loss':best_loss,'best_epoch':best_epoch,'history':history,'config':config
        },out/'last_training_state.pt')
        print(f'{args.method} seed={args.seed} epoch={epoch}: train={train_mse:.5f} '
              f'val={val_mse:.5f} L2-SP={reg:.6f} best={best_loss:.5f}@{best_epoch}',flush=True)
        if epoch>=MIN_EPOCHS and epoch-best_epoch>=PATIENCE:
            print('Early stopping',flush=True)
            break
    if not (out/'best_model.pt').is_file():
        raise RuntimeError('No validation-selected checkpoint saved')
    saved=torch.load(out/'best_model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(saved['model_state_dict'])
    delta_sum=0.0
    initial_sum=0.0
    frozen_delta=0.0
    embed_delta_sum=0.0
    embed_initial_sum=0.0
    for name,p in model.named_parameters():
        if name not in initial:
            continue
        diff=p.detach().cpu().float()-initial[name].float()
        delta_sum+=float(diff.square().sum())
        initial_sum+=float(initial[name].float().square().sum())
        if name.startswith('embedder.'):
            embed_delta_sum+=float(diff.square().sum())
            embed_initial_sum+=float(initial[name].float().square().sum())
        if not p.requires_grad:
            frozen_delta+=float(diff.abs().sum())
    if args.method=='PARTIAL' and frozen_delta != 0.0:
        raise RuntimeError('Frozen parameters changed unexpectedly')
    result={
        'method':args.method,'seed':args.seed,'best_epoch':best_epoch,
        'last_epoch':history[-1]['epoch'],'best_val_mse':best_loss,
        'relative_pretrained_weight_change':math.sqrt(delta_sum/initial_sum),
        'relative_embedder_weight_change':math.sqrt(embed_delta_sum/embed_initial_sum),
        'frozen_weight_absolute_change':frozen_delta,'seconds':time.time()-start,
        'status':'TRAIN_COMPLETE'
    }
    write_json(out/'train_summary.json',result)
    (out/'TRAIN_COMPLETE').touch()
    print('TRAIN_COMPLETE',out, result,flush=True)

if __name__=='__main__':
    main()
