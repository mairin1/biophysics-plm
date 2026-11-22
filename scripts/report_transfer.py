from collections import defaultdict
from pathlib import Path
import csv
import json
import math
import numpy as np
from transfer_common import *


def fnum(x):
    try:
        y=float(x)
        return y if np.isfinite(y) else float('nan')
    except (ValueError,TypeError):
        return float('nan')


def average(values):
    a=np.array([fnum(v) for v in values],dtype=float)
    a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float('nan')


def sd(values):
    a=np.array([fnum(v) for v in values],dtype=float)
    a=a[np.isfinite(a)]
    return float(a.std(ddof=1)) if len(a)>1 else float('nan')


def paired(rows,fields,keyfields,metric,direction=1):
    base={tuple(r[k] for k in keyfields):r for r in rows if r['condition']=='BASELINE'}
    out=[]
    for r in rows:
        if r['condition']=='BASELINE':
            continue
        key=tuple(r[k] for k in keyfields)
        if key not in base:
            raise ValueError(f'Missing paired pretrained baseline for {key}')
        b=base[key]
        cur=fnum(r[metric]); orig=fnum(b[metric])
        d=cur-orig if np.isfinite(cur) and np.isfinite(orig) else float('nan')
        obj=dict(r)
        obj['baseline_'+metric]=orig
        obj['delta_'+metric]=d
        out.append(obj)
    return out


def main():
    folder=ROOT/'report'
    folder.mkdir(parents=True,exist_ok=True)
    status=[]; zero=[]; few=[]; retention=[]
    tasks=[(c,s) for c in CONDITIONS for s in SEEDS]+[('BASELINE',SEEDS[0])]
    for c,s in tasks:
        run=ROOT/'runs'/f'{c}_seed{s}'
        success=(run/'SUCCESS').exists()
        summary={}
        if (run/'summary.json').exists():
            summary=json.loads((run/'summary.json').read_text())
        retention_ok=success and summary.get('retention_status')=='COMPLETE'
        status.append({'condition':c,'seed':s,'transfer_complete':int(success),
                       'retention_complete':int(retention_ok),'retention_error':summary.get('retention_error') or ''})
        if not success:
            continue
        zero.extend(read_csv(run/'zero_shot_metrics.csv'))
        few.extend(read_csv(run/'few_shot_metrics.csv'))
        if retention_ok:
            retention.extend(read_csv(run/'retention_metrics.csv'))
    write_csv(folder/'run_status.csv',status,['condition','seed','transfer_complete','retention_complete','retention_error'])
    n_done=sum(int(r['transfer_complete']) for r in status)
    if n_done==0:
        (folder/'summary.md').write_text('# Unseen-protein transfer\n\nNo complete evaluation jobs. Inspect Slurm logs.\n')
        return
    if not any(r['condition']=='BASELINE' for r in zero):
        (folder/'summary.md').write_text('# Unseen-protein transfer\n\nPretrained baseline missing; no paired comparisons produced.\n')
        return
    z=paired(zero,None,['protein','target'],'spearman')
    fu=paired(few,None,['protein','target','replicate','support_n'],'spearman')
    re=[]
    if any(r['condition']=='BASELINE' for r in retention):
        re=paired(retention,None,['protein','mask_fraction','repeat'],'token_nll')
    write_csv(folder/'paired_zero_shot.csv',z,['condition','seed','protein','target','method','n_query','spearman','baseline_spearman','delta_spearman','alpha','source_val_spearman'])
    write_csv(folder/'paired_few_shot.csv',fu,['condition','seed','protein','target','replicate','support_n','n_query','spearman','baseline_spearman','delta_spearman','r2','alpha','dev_spearman','status'])
    write_csv(folder/'paired_retention.csv',re,['condition','seed','protein','mask_fraction','repeat','n_masked','token_nll','baseline_token_nll','delta_token_nll'])
    aggregate=[]
    for method,rr in [('zero_shot',z),('few_shot',fu)]:
        groups=defaultdict(list)
        for r in rr:
            k=(r['condition'],r['protein'],r['target'],str(r.get('support_n','0')),str(r.get('seed')))
            groups[k].append(fnum(r['delta_spearman']))
        by_seed=defaultdict(list)
        for (c,p,t,n,seed),v in groups.items():
            by_seed[(c,p,t,n)].append(average(v))
        for (c,p,t,n),perseeds in by_seed.items():
            aggregate.append({'method':method,'condition':c,'protein':p,'target':t,'support_n':n,
                              'mean_delta_spearman':average(perseeds),'sd_across_encoder_seeds':sd(perseeds),
                              'n_encoder_seeds':len(perseeds)})
    write_csv(folder/'per_protein_aggregates.csv',aggregate,
              ['method','condition','protein','target','support_n','mean_delta_spearman','sd_across_encoder_seeds','n_encoder_seeds'])
    figures=[]; plot_errors=[]
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        conds=CONDITIONS
        for target in ['fa_atr','fa_sol']:
            fig,ax=plt.subplots(figsize=(9,4.6))
            vals=[]; errs=[]
            for c in conds:
                a=[r['mean_delta_spearman'] for r in aggregate if r['method']=='zero_shot' and r['condition']==c and r['target']==target]
                vals.append(average(a)); errs.append(0 if not np.isfinite(sd(a)) else sd(a))
            ax.bar(conds,vals,yerr=errs,capsize=3)
            ax.axhline(0,lw=.8)
            ax.set_ylabel('Paired delta Spearman vs original EvoDiff')
            ax.set_title(f'Zero-shot source probe on held-out proteins: {target}')
            fig.tight_layout()
            name=f'zero_shot_{target}.png';fig.savefig(folder/name,dpi=175);plt.close(fig);figures.append(name)
            fig,ax=plt.subplots(figsize=(9,4.6))
            for c in ['B','BC','ABC']:
                v=[]
                for n in SUPPORT_SIZES:
                    a=[r['mean_delta_spearman'] for r in aggregate if r['method']=='few_shot' and r['condition']==c and r['target']==target and r['support_n']==str(n)]
                    v.append(average(a))
                ax.plot(SUPPORT_SIZES,v,marker='o',label=c)
            ax.axhline(0,lw=.8)
            ax.set_xticks(SUPPORT_SIZES)
            ax.set_xlabel('Labeled support variants per new protein')
            ax.set_ylabel('Paired delta Spearman vs original EvoDiff')
            ax.set_title(f'Few-shot adaptation on held-out proteins: {target}')
            ax.legend()
            fig.tight_layout()
            name=f'few_shot_{target}.png';fig.savefig(folder/name,dpi=175);plt.close(fig);figures.append(name)
        for method, n in [('zero_shot','0'),('few_shot','128')]:
            fig,ax=plt.subplots(figsize=(10,4.3))
            subset=[r for r in aggregate if r['method']==method and r['support_n']==n and r['target'] in ['fa_atr','fa_sol']]
            proteins=sorted({r['protein'] for r in subset})
            cols=[f'{p.split("_")[0]} / {t}' for p in proteins for t in ['fa_atr','fa_sol']]
            grid=np.full((len(CONDITIONS),len(cols)),np.nan)
            for i,c in enumerate(CONDITIONS):
                for j,(p,t) in enumerate([(p,t) for p in proteins for t in ['fa_atr','fa_sol']]):
                    v=[r['mean_delta_spearman'] for r in subset if r['condition']==c and r['protein']==p and r['target']==t]
                    if v: grid[i,j]=fnum(v[0])
            m=np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
            im=ax.imshow(grid,cmap='coolwarm',vmin=-m,vmax=m,aspect='auto')
            ax.set_xticks(np.arange(len(cols)),cols,rotation=25,ha='right')
            ax.set_yticks(np.arange(len(CONDITIONS)),CONDITIONS)
            for i in range(len(CONDITIONS)):
                for j in range(len(cols)):
                    if np.isfinite(grid[i,j]): ax.text(j,i,f'{grid[i,j]:+.3f}',ha='center',va='center',fontsize=8)
            ax.set_title(f'Per-protein transfer differences, {method}, {n} support')
            fig.colorbar(im,ax=ax,label='Delta Spearman')
            fig.tight_layout()
            name=f'heatmap_{method}.png';fig.savefig(folder/name,dpi=175);plt.close(fig);figures.append(name)
        if re:
            fig,ax=plt.subplots(figsize=(8.5,4.4))
            for c in CONDITIONS:
                xs=[];ys=[]
                for frac in [0.15,0.5,0.8]:
                    vals=[r['delta_token_nll'] for r in re if r['condition']==c and float(r['mask_fraction'])==frac]
                    xs.append(frac);ys.append(average(vals))
                ax.plot(xs,ys,marker='o',label=c)
            ax.axhline(0,lw=.8);ax.legend(ncol=3)
            ax.set_xlabel('Mask fraction')
            ax.set_ylabel('Increase in masked-token NLL (lower is better)')
            ax.set_title('Pretrained OADM head compatibility after fine-tuning')
            fig.tight_layout()
            name='retention.png';fig.savefig(folder/name,dpi=175);plt.close(fig);figures.append(name)
    except Exception as e:
        plot_errors.append(str(e))
    manifest=json.loads((ROOT/'evaluation_manifest.json').read_text())
    summary=['# EvoDiff Stage II exploratory transfer evaluation', '',
             f'**Transfer jobs complete:** {n_done}/19',
             f'**Retention evaluated:** {sum(int(x["retention_complete"]) for x in status)}/19',
             '', '## Scope', '',
             'This compares original pretrained EvoDiff with six Rosetta-supervised conditions (three seeds each) on the two proteins omitted from Stage I encoder fine-tuning.',
             'Primary zero-shot result: a ridge probe fitted to source-protein labels and evaluated without any held-out protein labels. Spearman only; raw energy offsets are not calibrated.',
             'Few-shot result: independent support, dev, and query subsets of held-out protein variants. Alpha selection uses dev only, query is untouched.',
             'The base EvoDiff checkpoint may have seen related natural sequences in its original UniRef pretraining; "unseen" refers to the Rosetta-supervised stage.',
             'This is exploratory because the user has previously viewed some held-out data, and it is not an independent experimental validation.',
             'Retention uses the original pretrained OADM output head with updated encoder components. NLL compares identical mask patterns and sequences.',
             '', '## Results tables', '',
             '- `run_status.csv`: every condition, completion and retention status',
             '- `paired_zero_shot.csv`: per-protein and per-target paired differences',
             '- `paired_few_shot.csv`: paired support-budget results',
             '- `paired_retention.csv`: pretrained OADM head compatibility',
             '- `per_protein_aggregates.csv`: averages across encoder seeds, not uncertainty across independent proteins',
             '', '## Figures', '']
    summary.extend('- `'+f+'`' for f in figures)
    summary.extend(['', '## Interpretation boundaries','',
                    'The two held-out proteins are insufficient to estimate performance across protein fold space.',
                    'Stage I variants are close mutants, and held-out support/query data may also have very close sequence neighbors; only exact duplicates were removed.',
                    'Six Rosetta diagnostic properties are correlated; do not treat their metrics as independent observations.',
                    'No ProteinGym data were used; no experimental sequence-function improvement is established.',
                    'A successful GPU job with a failed retention diagnostic counts only as transfer-complete.',
                    '',f'**Plot errors:** {plot_errors or "none"}', ''])
    (folder/'summary.md').write_text('\n'.join(summary))
    if n_done==19:
        (folder/'TRANSFER_COMPLETE').touch()
    else:
        (folder/'PARTIAL').touch()
    print(f'Transfer success {n_done}/19. Retention {sum(int(x["retention_complete"]) for x in status)}/19. Report: {folder}',flush=True)

if __name__=='__main__':
    main()
