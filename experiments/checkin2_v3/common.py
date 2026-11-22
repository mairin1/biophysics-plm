from pathlib import Path
import hashlib
import json
import os

BASE = Path('metl-global/analysis/evodiff_checkin2_20261119')
STAGE1 = Path('metl-global/analysis/evodiff_stage1_20261116')
TRANSFER = Path('metl-global/analysis/evodiff_transfer_20261119')
SOURCE = Path('metl-global/exports/run_29051193')
SEEDS = [20261116, 20261117, 20261118]
METHODS = ['GENTLE', 'L2SP', 'PARTIAL']
TARGETS = [
    'fa_dun', 'omega', 'p_aa_pp', 'rama_prepro', 'pro_close', 'yhh_planarity',
    'fa_elec', 'fa_rep', 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb',
    'fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'cbeta', 'cenpack', 'env'
]

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def run_dir(method, seed):
    return BASE / 'runs' / f'{method}_seed{seed}'
