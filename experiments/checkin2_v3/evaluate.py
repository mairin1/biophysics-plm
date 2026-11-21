import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import BASE, TRANSFER, METHODS, SEEDS, run_dir, write_json
sys.path.insert(0, 'scripts')
import transfer_common as tc
import evaluate_transfer as et


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', choices=SEEDS, type=int, required=True)
    args = p.parse_args()
    out = run_dir(args.method,args.seed)
    if (out/'SUCCESS').is_file():
        print('Complete already:', out, flush=True)
        return
    if not (out/'TRAIN_COMPLETE').is_file():
        raise RuntimeError('Fine-tuning did not finish: refusing transfer evaluation')
    start = time.time()
    manifest = json.loads((TRANSFER/'evaluation_manifest.json').read_text())
    if manifest['metadata']['source_sha256'] != json.loads((BASE/'preflight.json').read_text())['source_hashes']:
        raise RuntimeError('Evaluation manifest is not the data validated by the preflight')
    grouped = tc.records()
    entries = et.get_entries(grouped,manifest)
    et.OLD = BASE
    model, tokenizer, scheme = et.loading_model(args.method,args.seed)
    if scheme != 'mask':
        raise RuntimeError(f'EvoDiff masking scheme changed: {scheme}')
    features = {}
    for protein in sorted(entries):
        for split, rows in entries[protein].items():
            features[(protein,split)] = et.encode(model,rows,tokenizer)
    zero, _ = et.evaluate_zero_shot(features,entries,manifest)
    for row in zero:
        row['condition'], row['seed'] = args.method,args.seed
    tc.write_csv(out/'zero_shot_metrics.csv',zero,
                 ['condition','seed','protein','target','method','n_query','spearman','alpha','source_val_spearman'])
    few, _ = et.evaluate_few_shot(features,entries,manifest)
    for row in few:
        row['condition'], row['seed'] = args.method,args.seed
    tc.write_csv(out/'few_shot_metrics.csv',few,
                 ['condition','seed','protein','target','replicate','support_n','n_query','spearman','r2','alpha','dev_spearman','status'])
    retention = et.retention_eval(model,tokenizer,entries,manifest)
    for row in retention:
        row['condition'],row['seed']=args.method,args.seed
    tc.write_csv(out/'retention_metrics.csv',retention,
                 ['condition','seed','protein','mask_fraction','repeat','n_masked','token_nll'])
    if len(zero) != 12 or len(few) != 108 or len(retention) != 12:
        raise RuntimeError(f'Unexpected evaluation row counts: zero={len(zero)} few={len(few)} masked={len(retention)}')
    if not all(np.isfinite(float(row['token_nll'])) for row in retention):
        raise RuntimeError('Nonfinite original-objective masked NLL')
    write_json(out/'evaluation_summary.json',{
        'method':args.method,'seed':args.seed,'zero_shot_rows':len(zero),
        'few_shot_rows':len(few),'retention_rows':len(retention),
        'reference_manifest':str(TRANSFER/'evaluation_manifest.json'),
        'evaluation_seconds':time.time()-start,
        'status':'PASS',
        'interpretation':'Exploratory transfer for two proteins excluded from Rosetta encoder supervision; not independent experimental validation'
    })
    (out/'SUCCESS').touch()
    print('EVALUATION SUCCESS:',out,flush=True)

if __name__=='__main__':
    main()
