from collections import defaultdict
from pathlib import Path
import csv
import json
import math
import sys
import traceback

import numpy as np

from common import BASE, STAGE1, TRANSFER, METHODS, SEEDS, run_dir

sys.path.insert(0,'scripts')
import transfer_common as tc

ALL = ['ABC'] + METHODS


def number(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def average(xs):
    a=np.asarray([number(x) for x in xs],dtype=float)
    a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float('nan')


def read(condition,seed,filename):
    root = TRANSFER/'runs'/f'ABC_seed{seed}' if condition=='ABC' else run_dir(condition,seed)
    return tc.read_csv(root/filename)


def baseline(filename):
    return tc.read_csv(TRANSFER/'runs'/f'BASELINE_seed{SEEDS[0]}'/filename)


def paired_data(filename,metric,fields):
    original={tuple(x[k] for k in fields):x for x in baseline(filename)}
    output=[]
    for condition in ALL:
        for seed in SEEDS:
            folder = TRANSFER/'runs'/f'ABC_seed{seed}' if condition=='ABC' else run_dir(condition,seed)
            if not (folder/'SUCCESS').is_file():
                continue
            for row in read(condition,seed,filename):
                key=tuple(row[k] for k in fields)
                if key not in original:
                    raise RuntimeError(f'No matched original EvoDiff baseline: {filename}, {key}')
                base=number(original[key][metric])
                current=number(row[metric])
                diff=current-base
                out=dict(row)
                out.update({'condition':condition,'seed':seed,
                            f'baseline_{metric}':base,f'delta_{metric}':diff})
                output.append(out)
    return output


def rows_for(condition,table,target=None,support=None):
    xs=[r for r in table if r['condition']==condition]
    if target is not None: xs=[r for r in xs if r['target']==target]
    if support is not None: xs=[r for r in xs if str(r.get('support_n'))==str(support)]
    return xs


def plot_all(zero,few,retention,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=[]
    for target in ('fa_atr','fa_sol'):
        x=np.arange(len(ALL))
        means=[average([r['delta_spearman'] for r in rows_for(c,zero,target)]) for c in ALL]
        fig,ax=plt.subplots(figsize=(8,4.6))
        ax.bar(x,means)
        ax.axhline(0,color='black',lw=0.8)
        ax.set_xticks(x,ALL)
        ax.set_ylabel('Change in Spearman vs original EvoDiff')
        ax.set_title(f'Zero-shot transfer to two held-out proteins: {target}')
        fig.tight_layout()
        name=f'zero_shot_{target}.png'
        fig.savefig(folder/name,dpi=180)
        plt.close(fig)
        figures.append(name)
        fig,ax=plt.subplots(figsize=(8.5,4.7))
        for condition in ALL:
            y=[average([r['delta_spearman'] for r in rows_for(condition,few,target,b)]) for b in (32,128,512)]
            ax.plot([32,128,512],y,marker='o',label=condition)
        ax.axhline(0,color='black',lw=0.8)
        ax.set_xticks([32,128,512])
        ax.set_xlabel('Labeled support variants from held-out protein')
        ax.set_ylabel('Change in Spearman vs original EvoDiff')
        ax.set_title(f'Few-shot adaptation: {target}')
        ax.legend(ncol=5)
        fig.tight_layout()
        name=f'few_shot_{target}.png'
        fig.savefig(folder/name,dpi=180)
        plt.close(fig)
        figures.append(name)
    fig,ax=plt.subplots(figsize=(10,5))
    columns=[]
    proteins=sorted({r['protein'] for r in zero})
    for protein in proteins:
        for target in ['fa_atr','fa_sol','total_sasa','rg','pack','unsat_hbond']:
            columns.append((protein,target))
    grid=np.full((len(ALL),len(columns)),np.nan)
    for i,condition in enumerate(ALL):
        for j,(protein,target) in enumerate(columns):
            grid[i,j]=average([r['delta_spearman'] for r in zero if r['condition']==condition and r['protein']==protein and r['target']==target])
    maxabs=np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1
    maxabs=max(maxabs,0.01)
    image=ax.imshow(grid,cmap='coolwarm',vmin=-maxabs,vmax=maxabs,aspect='auto')
    ax.set_yticks(range(len(ALL)),ALL)
    ax.set_xticks(range(len(columns)),[f'{p.split("_")[0]} / {t}' for p,t in columns],rotation=65,ha='right',fontsize=8)
    for i in range(len(ALL)):
        for j in range(len(columns)):
            if np.isfinite(grid[i,j]):
                ax.text(j,i,f'{grid[i,j]:+.2f}',ha='center',va='center',fontsize=7)
    ax.set_title('Zero-shot transfer by held-out protein and property')
    fig.colorbar(image,ax=ax,label='Change in Spearman')
    fig.tight_layout()
    name='zero_shot_per_protein_heatmap.png'
    fig.savefig(folder/name,dpi=180)
    plt.close(fig)
    figures.append(name)
    fig,ax=plt.subplots(figsize=(8.5,4.7))
    for c in ALL:
        values=[average([r['delta_token_nll'] for r in retention
                         if r['condition']==c and float(r['mask_fraction'])==fraction]) for fraction in (0.15,0.5,0.8)]
        ax.plot([0.15,0.5,0.8],values,marker='o',label=c)
    ax.axhline(0,color='black',lw=0.8)
    ax.set_xticks([0.15,0.5,0.8])
    ax.set_xlabel('Masked residue fraction')
    ax.set_ylabel('Increase in masked-token NLL vs original (lower is better)')
    ax.set_title('Pretrained OADM output-head compatibility')
    ax.legend(ncol=5)
    fig.tight_layout()
    name='masked_token_retention.png'
    fig.savefig(folder/name,dpi=180)
    plt.close(fig)
    figures.append(name)
    fig,ax=plt.subplots(figsize=(8.5,5.4))
    for c in ALL:
        coords=[]
        for seed in SEEDS:
            z=average([r['delta_spearman'] for r in zero
                       if r['condition']==c and int(r['seed'])==seed and r['target'] in ('fa_atr','fa_sol')])
            n=average([r['delta_token_nll'] for r in retention
                       if r['condition']==c and int(r['seed'])==seed])
            if np.isfinite(z) and np.isfinite(n):coords.append((n,z))
        if coords:
            ax.scatter([q[0] for q in coords],[q[1] for q in coords],label=c,s=45)
            ax.scatter([average([q[0] for q in coords])],[average([q[1] for q in coords])],
                       marker='x',s=95)
    ax.scatter([0],[0],marker='*',s=140,label='Pretrained reference')
    ax.axhline(0,color='black',lw=.7)
    ax.axvline(0,color='black',lw=.7)
    ax.set_xlabel('Increase in masked-token NLL (lower is better)')
    ax.set_ylabel('Zero-shot ΔSpearman, mean of fa_atr and fa_sol (higher is better)')
    ax.set_title('Transfer-versus-retention tradeoff (exploratory, two proteins)')
    ax.legend()
    fig.tight_layout()
    name='transfer_retention_tradeoff.png'
    fig.savefig(folder/name,dpi=180)
    plt.close(fig)
    figures.append(name)
    fig,ax=plt.subplots(figsize=(9,5))
    for condition in ALL:
        curves=[]
        for seed in SEEDS:
            path=(STAGE1/'runs'/f'ABC_seed{seed}'/'history.csv') if condition=='ABC' else (run_dir(condition,seed)/'history.csv')
            if not path.is_file():continue
            rows=tc.read_csv(path)
            curves.append([(int(r['epoch']),float(r['val_mse'])) for r in rows])
        if curves:
            n=max(len(r) for r in curves)
            series=np.full((len(curves),n),np.nan)
            for i,row in enumerate(curves):
                for j,(ep,v) in enumerate(row):series[i,ep-1]=v
            xx=np.arange(1,n+1)
            means=np.nanmean(series,axis=0)
            ax.plot(xx,means,label=condition)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rosetta 18-target validation MSE')
    ax.set_title('Training trajectories (same ABC labels and validation variants)')
    ax.legend()
    fig.tight_layout()
    name='training_validation_curves.png'
    fig.savefig(folder/name,dpi=180)
    plt.close(fig)
    figures.append(name)
    return figures


def main():
    folder=BASE/'report'
    folder.mkdir(parents=True,exist_ok=True)
    status=[]
    for condition in METHODS:
        for seed in SEEDS:
            out=run_dir(condition,seed)
            okay=(out/'SUCCESS').is_file()
            status.append({'method':condition,'seed':seed,'train_complete':int((out/'TRAIN_COMPLETE').is_file()),
                           'evaluation_complete':int(okay),'path':str(out)})
    tc.write_csv(folder/'run_status.csv',status,list(status[0]))
    finished=sum(r['evaluation_complete'] for r in status)
    if finished==0:
        (folder/'summary.md').write_text('# Check-in 2 experiment\n\nNo new evaluations completed. Check Slurm logs.\n')
        (folder/'PARTIAL').touch()
        print('No new models finished',flush=True)
        return
    zero=paired_data('zero_shot_metrics.csv','spearman',['protein','target'])
    few=paired_data('few_shot_metrics.csv','spearman',['protein','target','replicate','support_n'])
    retention=paired_data('retention_metrics.csv','token_nll',['protein','mask_fraction','repeat'])
    tc.write_csv(folder/'paired_zero_shot.csv',zero,list(zero[0]))
    tc.write_csv(folder/'paired_few_shot.csv',few,list(few[0]))
    tc.write_csv(folder/'paired_retention.csv',retention,list(retention[0]))
    group=[]
    for method in ALL:
        for target in ['fa_atr','fa_sol','total_sasa','rg','pack','unsat_hbond']:
            group.append({'method':method,'target':target,
                          'mean_zero_shot_delta_spearman':average([r['delta_spearman'] for r in rows_for(method,zero,target)]),
                          'mean_few_shot_n32_delta_spearman':average([r['delta_spearman'] for r in rows_for(method,few,target,32)]),
                          'mean_few_shot_n512_delta_spearman':average([r['delta_spearman'] for r in rows_for(method,few,target,512)]),
                          'n_encoder_seeds':len({int(r['seed']) for r in rows_for(method,zero,target)})})
    tc.write_csv(folder/'aggregate_by_method.csv',group,list(group[0]))
    plots=[]
    errors=[]
    try:
        plots=plot_all(zero,few,retention,folder)
    except Exception as exc:
        errors.append(traceback.format_exc())
    lines=['# Check-in 2: preservation-aware EvoDiff fine-tuning', '',
           f'**New training and transfer evaluations complete:** {finished}/9',
           '**Comparators:** original EvoDiff and original ABC fine-tuning (3 seeds, cached Stage II evaluation)',
           '', '## Design', '',
           'Same eight reference-protein pool, original six fine-tuning proteins, ABC 18 Rosetta attributes, and exactly the Stage II held-out evaluation manifest.',
           'New methods: encoder LR 2e-6 (GENTLE), explicit L2-SP anchoring (L2SP), and early ByteNet block freezing (PARTIAL).',
           'Each uses the same Rosetta validation stopping rule, matched seeds, probes, and masking patterns.',
           '', '## Numeric summary', '',
           '| Method | Target | Zero-shot ΔSpearman | Few-shot 32 ΔSpearman | Few-shot 512 ΔSpearman |',
           '|---|---|---:|---:|---:|']
    for r in group:
        def f(x):return f'{x:+.4f}' if np.isfinite(x) else 'NA'
        if r['target'] in ('fa_atr','fa_sol'):
            lines.append(f"| {r['method']} | {r['target']} | {f(r['mean_zero_shot_delta_spearman'])} | {f(r['mean_few_shot_n32_delta_spearman'])} | {f(r['mean_few_shot_n512_delta_spearman'])} |")
    lines+=['','## Retention changes', '',
            '| Method | Mean Δ masked-token NLL (nats) |', '|---|---:|']
    for c in ALL:
        v=average([r['delta_token_nll'] for r in retention if r['condition']==c])
        lines.append(f'| {c} | {v:+.4f} |')
    lines+=['','## Figures', '']
    lines.extend(f'- `{f}`' for f in plots)
    lines+=['','## Interpretation boundaries','',
            '- Positive zero-shot delta means a source-trained linear probe ranks Rosetta labels better for the two omitted proteins. It does not establish experimental function transfer.',
            '- The original Stage II query set has already been viewed. This is exploratory testing, not a fresh held-out confirmation.',
            '- The two proteins are not independent evidence for all protein folds. Variants may be very close sequence neighbors.',
            '- All probes use Rosetta labels. Target properties are correlated, and score improvements do not imply mechanism learning.',
            '- Masked-token NLL here reuses the frozen original OADM output head with the modified ByteNet components. It is a retention-compatibility diagnostic, not measured generation quality.',
            '- Checkpoint selection uses the six training proteins only, not held-out transfer results.',
            '', f'**Plot errors:** {"none" if not errors else errors}', '']
    (folder/'summary.md').write_text('\n'.join(lines))
    if finished==9 and not errors and len(plots)>=7:
        (folder/'REPORT_READY').touch()
    else:
        (folder/'PARTIAL').touch()
    print(f'Report: new complete {finished}/9, {len(plots)} plots, {len(errors)} plot errors. {folder}',flush=True)

if __name__=='__main__':
    main()
