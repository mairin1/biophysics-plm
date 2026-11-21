from collections import Counter
from pathlib import Path
import json
import random
import numpy as np
import torch
from transfer_common import *


def unique_by_sequence(rows, exclude=()):
    blocked = set(exclude)
    unique = {}
    for r in sorted(rows, key=lambda x: x['row_id']):
        s = r['sequence']
        if s not in blocked and s not in unique:
            unique[s] = r
    return list(unique.values())


def choose(rows, count, seed):
    if len(rows) < count:
        raise ValueError(f'Only {len(rows)} unique candidates available, need {count}')
    rr = sorted(rows, key=lambda x:x['row_id'])
    rng = random.Random(seed)
    inds = rng.sample(range(len(rr)), count)
    return [rr[i] for i in sorted(inds)]


def permute(rows, seed):
    ordered = sorted(rows, key=lambda x: x['row_id'])
    random.Random(seed).shuffle(ordered)
    return ordered


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    oldaudit = json.loads((OLD/'audit_summary.json').read_text())
    if not (OLD/'report'/'REPORT_READY').exists() or not (OLD/'AUDIT_OK').exists():
        raise ValueError('Stage I report or audit missing')
    sha = {}
    for fname in ['sequences.csv.gz','rosetta_numeric.csv.gz','sample_manifest.csv']:
        sha[fname] = digest(SOURCE/fname)
        if sha[fname] != oldaudit['source_file_sha256'].get(fname):
            raise ValueError(f'Source has changed since Stage I: {fname}')
    expected = [(c,s) for c in CONDITIONS for s in SEEDS]
    status = []
    for c,s in expected:
        run = OLD/'runs'/f'{c}_seed{s}'
        if not (run/'SUCCESS').exists() or not (run/'best_model.pt').exists():
            raise ValueError(f'Missing Stage I run or checkpoint: {c}_{s}')
        x = torch.load(run/'best_model.pt', map_location='cpu', weights_only=True)
        targets = [x for k in c for x in FAMILIES[k]]
        if x['condition'] != c or x['seed'] != s or x['targets'] != targets:
            raise ValueError(f'Checkpoint metadata incorrect: {c}_{s}')
        keys = x['model_state_dict'].keys()
        if not any(k.startswith('embedder.') for k in keys) or not any(k.startswith('last_norm.') for k in keys):
            raise ValueError(f'Checkpoint missing actual encoder weights: {c}_{s}')
        status.append({'condition':c,'seed':s,'best_epoch':x['best_epoch'],'n_targets':len(targets),'path':str(run/'best_model.pt')})
    grouped = records()
    holdouts = sorted([p for p,x in grouped.items() if 'train' not in x])
    sources = sorted([p for p,x in grouped.items() if 'train' in x])
    if holdouts != EXPECTED_HOLDOUTS or len(sources) != 6:
        raise ValueError(f'Unexpected protein roles: {sources} and {holdouts}')
    manifest = {'metadata':{'source_sha256':sha, 'old_audit_file':str(OLD/'audit_summary.json'),
                            'targets':TARGETS, 'sources':sources,'holdouts':holdouts,
                            'support_seeds':SUPPORT_SEEDS,'support_sizes':SUPPORT_SIZES,
                            'alpha_grid':ALPHAS, 'evaluation_type':'Exploratory, two held-out METL base proteins, not ProteinGym'},
                'source':{}, 'heldout':{}, 'checkpoints':status}
    duplicate_counts = {}
    for i,p in enumerate(sources):
        tr = unique_by_sequence(grouped[p]['train'])
        val = unique_by_sequence(grouped[p]['val'], exclude={r['sequence'] for r in tr})
        tr_sel = choose(tr,2000,2000+i)
        val_sel = choose(val,500,3000+i)
        manifest['source'][p] = {'train':[r['row_id'] for r in tr_sel], 'val':[r['row_id'] for r in val_sel]}
        duplicate_counts[p] = {'train_unique':len(tr),'val_unique_no_train_overlap':len(val)}
    for i,p in enumerate(holdouts):
        h = unique_by_sequence(grouped[p]['test'])
        if len(h) < 4500:
            raise ValueError(f'Too few unique sequences in held-out protein {p}: {len(h)}')
        h = permute(h,800+i)
        support = h[:1000]
        dev = h[1000:1500]
        query = h[1500:]
        if len(query)<3000:
            raise ValueError(f'Not enough held-out query sequences for {p}')
        reps={}
        for j,seed in enumerate(SUPPORT_SEEDS):
            perm = permute(support,seed+i*1000)
            reps[str(seed)] = {str(k):[r['row_id'] for r in perm[:k]] for k in SUPPORT_SIZES}
        manifest['heldout'][p] = {'pool':[r['row_id'] for r in support], 'dev':[r['row_id'] for r in dev],
                                  'query':[r['row_id'] for r in query], 'replicates':reps}
        duplicate_counts[p]={'test_unique':len(h),'query_n':len(query)}
        query_labels = np.array([r['labels'] for r in query], dtype=float)
        for name, arr in zip(TARGETS,query_labels.T):
            if np.std(arr)<1e-8:
                print(f'WARNING: {p} query target {name} is constant',flush=True)
    as_json(manifest, ROOT/'evaluation_manifest.json')
    as_json({'source_sha256':sha,'checkpoints_validated':len(status),'proteins':duplicate_counts,
             'holdouts':holdouts,'targets':TARGETS,'status':'PASS'},ROOT/'preflight.json')
    (ROOT/'PREFLIGHT_OK').touch()
    print(f'PASS: {len(status)} checkpoints, {len(sources)} source proteins, {len(holdouts)} heldout proteins',flush=True)
    print('Manifest:', ROOT/'evaluation_manifest.json',flush=True)

if __name__=='__main__':
    main()
