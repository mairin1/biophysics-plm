import csv
import sys
import numpy as np
import torch

from common import BASE, TRANSFER, SEEDS, write_json
sys.path.insert(0, 'scripts')
import transfer_common as tc
import evaluate_transfer as et


def load_reference(condition, seed):
    path = TRANSFER/'runs'/f'{condition}_seed{seed}'/'retention_metrics.csv'
    with path.open() as f:
        return {
            (row['protein'], row['mask_fraction'], row['repeat']):float(row['token_nll'])
            for row in csv.DictReader(f)
        }


def evaluate(condition, seed, entries, manifest):
    model, tokenizer, scheme = et.loading_model(condition, seed)
    if scheme != 'mask':
        raise RuntimeError(f'Unexpected EvoDiff corruption scheme: {scheme}')
    if not isinstance(model.embedder.layers, torch.nn.ModuleList) or len(model.embedder.layers) < 8:
        raise RuntimeError('ByteNet architecture is not the expected ordered ModuleList of convolutional blocks')
    with torch.inference_mode():
        rows = et.retention_eval(model, tokenizer, entries, manifest)
    reference = load_reference(condition, seed)
    diffs = []
    for row in rows:
        key = row['protein'], str(row['mask_fraction']), str(row['repeat'])
        if key not in reference:
            raise RuntimeError(f'Missing historical retention score for {condition}: {key}')
        diffs.append(abs(row['token_nll'] - reference[key]))
    worst = max(diffs)
    del model
    torch.cuda.empty_cache()
    if worst > 0.12:
        raise RuntimeError(f'{condition} masked-token diagnostic disagrees with prior evaluation: maximum absolute NLL deviation {worst:.4f}')
    return {'condition':condition, 'seed':seed, 'max_abs_nll_deviation':worst, 'matched_rows':len(rows)}


def main():
    if not (BASE/'PREFLIGHT_OK').exists():
        raise RuntimeError('CPU audit did not pass')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for model-interface and retention audit')
    manifest = __import__('json').loads((TRANSFER/'evaluation_manifest.json').read_text())
    grouped = tc.records()
    entries = et.get_entries(grouped, manifest)
    checked = [
        evaluate('BASELINE', SEEDS[0], entries, manifest),
        evaluate('ABC', SEEDS[0], entries, manifest)
    ]
    write_json(BASE/'gpu_sanity.json', {'status':'PASS','comparison':checked,
                                     'interpretation':'Masked-token NLL measures original diffusion-head compatibility with the updated encoder; not verified generation quality'})
    (BASE/'GPU_READY').touch()
    print('PASS: both original and Stage I ABC masked-token metrics reproduced', checked, flush=True)

if __name__ == '__main__':
    main()
