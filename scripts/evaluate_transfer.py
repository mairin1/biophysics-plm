from pathlib import Path
import argparse
import json
import math
import os
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from transfer_common import *


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)


def standardize_features(train, *others):
    mu = train.mean(0, keepdims=True)
    sd = train.std(0, keepdims=True)
    sd = np.maximum(sd, 1e-5)
    return [(x-mu)/sd for x in (train,)+others]


def ridge_predict(trainx, trainy, otherx, alpha):
    x = to_tensor(trainx)
    y = to_tensor(trainy)
    z = to_tensor(otherx)
    x = torch.cat([x, torch.ones((len(x),1),dtype=x.dtype,device=x.device)],dim=1)
    z = torch.cat([z, torch.ones((len(z),1),dtype=z.dtype,device=z.device)],dim=1)
    if x.shape[0] < x.shape[1]:
        # Centering ensures the intercept is never regularized, even in the dual form.
        rawx = x[:,:-1]
        rawz = z[:,:-1]
        xm = rawx.mean(0,keepdim=True)
        ym = y.mean(0,keepdim=True)
        x0 = rawx-xm
        z0 = rawz-xm
        gram = x0@x0.T
        gram.diagonal().add_(float(alpha))
        dual = torch.linalg.solve(gram, y-ym)
        pred = z0@(x0.T@dual)+ym
    else:
        gram = x.T@x
        penalty = torch.full((x.shape[1],),float(alpha),device=DEVICE,dtype=x.dtype)
        penalty[-1]=0.0
        gram.diagonal().add_(penalty)
        w = torch.linalg.solve(gram, x.T@y)
        pred = z@w
    return pred.detach().cpu().numpy()


def encode(model, entries, tokenizer):
    seqs = [x['sequence'] for x in entries]
    if not seqs:
        raise ValueError('Empty feature partition')
    if len({len(s) for s in seqs}) != 1:
        raise ValueError('Mixed sequence lengths within a protein partition')
    tokens = np.asarray([[tokenizer.a_to_i[aa] for aa in s] for s in seqs], dtype=np.int64)
    parts=[]
    with torch.no_grad():
        for start in range(0,len(tokens),64):
            t = torch.from_numpy(tokens[start:start+64]).to('cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                em = model.last_norm(model.embedder(t,None)).mean(1)
            parts.append(em.float().cpu().numpy().astype(np.float32))
    return np.concatenate(parts,axis=0)


def loading_model(condition,seed):
    from evodiff.pretrained import OA_DM_38M
    backbone, collater, tokenizer, scheme = OA_DM_38M()
    if condition != 'BASELINE':
        path = OLD/'runs'/f'{condition}_seed{seed}'/'best_model.pt'
        ck = torch.load(path,map_location='cpu',weights_only=True)
        sd=ck['model_state_dict']
        embed={k[len('embedder.'):]:v for k,v in sd.items() if k.startswith('embedder.')}
        last={k[len('last_norm.'):]:v for k,v in sd.items() if k.startswith('last_norm.')}
        if not embed or not last:
            raise ValueError('Checkpoint lacks original EvoDiff encoder components')
        backbone.embedder.load_state_dict(embed,strict=True)
        backbone.last_norm.load_state_dict(last,strict=True)
    return backbone.eval().cuda(),tokenizer,scheme


def get_entries(grouped,manifest):
    by_id={r['row_id']:r for splits in grouped.values() for rows in splits.values() for r in rows}
    selected={}
    for p,sects in manifest['source'].items():
        selected[p]={k:[by_id[rid] for rid in ids] for k,ids in sects.items()}
    for p,sects in manifest['heldout'].items():
        selected[p]={k:[by_id[rid] for rid in ids] for k,ids in sects.items() if k != 'replicates'}
    return selected


def evaluate_zero_shot(features, entries, manifest):
    src=manifest['metadata']['sources']
    tr_x, va_x, tr_y, va_y=[],[],[],[]
    target_index={}
    for p in src:
        a=np.asarray([r['labels'] for r in entries[p]['train']],dtype=np.float64)
        b=np.asarray([r['labels'] for r in entries[p]['val']],dtype=np.float64)
        mu=a.mean(0); sd=a.std(0)
        active=sd>1e-8
        safe=np.where(active,sd,1.0)
        target_index[p]={'mean':mu,'std':sd,'active':active}
        tr_x.append(features[(p,'train')]); va_x.append(features[(p,'val')]); tr_y.append((a-mu)/safe); va_y.append((b-mu)/safe)
    xtr=np.concatenate(tr_x); xva=np.concatenate(va_x)
    xtr,xva=standardize_features(xtr,xva)
    ytr=np.concatenate(tr_y)
    yva=np.concatenate(va_y)
    val_per_alpha={}
    for alpha in ALPHAS:
        pred=ridge_predict(xtr,ytr,xva,alpha)
        scores=np.full((len(src),len(TARGETS)),np.nan)
        pos=0
        for j,p in enumerate(src):
            n=len(entries[p]['val'])
            for k in range(len(TARGETS)):
                scores[j,k]=spearman(yva[pos:pos+n,k],pred[pos:pos+n,k])
            pos+=n
        val_per_alpha[alpha]=np.nanmean(scores,axis=0)
    best_alpha=[max(ALPHAS,key=lambda a: (-999 if not np.isfinite(val_per_alpha[a][k]) else val_per_alpha[a][k])) for k in range(len(TARGETS))]
    pred_by_alpha={a:None for a in ALPHAS}
    output=[]
    preds={}
    for p in manifest['metadata']['holdouts']:
        q=entries[p]['query']
        xq=(features[(p,'query')]-np.concatenate(tr_x).mean(0))/np.maximum(np.concatenate(tr_x).std(0),1e-5)
        for a in ALPHAS:
            pred_by_alpha[a]=ridge_predict(xtr,ytr,xq,a)
        actual=np.asarray([r['labels'] for r in q],dtype=np.float64)
        pick=np.column_stack([pred_by_alpha[a][:,k] for k,a in enumerate(best_alpha)])
        preds[p]=pick.astype(np.float32)
        for k,t in enumerate(TARGETS):
            output.append({'protein':p,'target':t,'method':'source_trained_probe',
                           'n_query':len(q),'spearman':spearman(actual[:,k],pick[:,k]),
                           'alpha':best_alpha[k],
                           'source_val_spearman':float(val_per_alpha[best_alpha[k]][k])})
    return output,preds


def evaluate_few_shot(features,entries,manifest):
    outputs=[]
    saved={}
    for p in manifest['metadata']['holdouts']:
        idx={r['row_id']:i for i,r in enumerate(entries[p]['pool'])}
        xpool=features[(p,'pool')]
        pool=np.asarray([r['labels'] for r in entries[p]['pool']],dtype=np.float64)
        xd=features[(p,'dev')]
        yd=np.asarray([r['labels'] for r in entries[p]['dev']],dtype=np.float64)
        xq=features[(p,'query')]
        yq=np.asarray([r['labels'] for r in entries[p]['query']],dtype=np.float64)
        for repseed in SUPPORT_SEEDS:
            for n in SUPPORT_SIZES:
                ids=manifest['heldout'][p]['replicates'][str(repseed)][str(n)]
                ix=[idx[r] for r in ids]
                x=xpool[ix]
                y=pool[ix]
                mu=y.mean(0); sd=y.std(0)
                valid=sd>1e-8
                sd=np.maximum(sd,1.0e-8)
                xs,xdev,xquery=standardize_features(x,xd,xq)
                predictions_dev={}
                predictions_q={}
                for alpha in ALPHAS:
                    predictions_dev[alpha]=ridge_predict(xs,(y-mu)/sd,xdev,alpha)*sd+mu
                    predictions_q[alpha]=ridge_predict(xs,(y-mu)/sd,xquery,alpha)*sd+mu
                selected=np.full((len(yq),len(TARGETS)),np.nan,dtype=np.float32)
                for k,t in enumerate(TARGETS):
                    if not valid[k]:
                        outputs.append({'protein':p,'target':t,'replicate':repseed,'support_n':n,
                                        'n_query':len(yq),'spearman':float('nan'),'r2':float('nan'),
                                        'alpha':float('nan'),'dev_spearman':float('nan'),'status':'constant_support_target'})
                        continue
                    vals={a:spearman(yd[:,k],predictions_dev[a][:,k]) for a in ALPHAS}
                    alpha=max(ALPHAS,key=lambda a:-999 if not np.isfinite(vals[a]) else vals[a])
                    prediction=predictions_q[alpha][:,k]
                    selected[:,k]=prediction.astype(np.float32)
                    sst=np.square(yq[:,k]-yq[:,k].mean()).sum()
                    r2=float(1-(np.square(prediction-yq[:,k]).sum()/sst)) if sst>1e-10 else float('nan')
                    outputs.append({'protein':p,'target':t,'replicate':repseed,'support_n':n,
                                    'n_query':len(yq),'spearman':spearman(yq[:,k],prediction),'r2':r2,
                                    'alpha':alpha,'dev_spearman':vals[alpha],'status':'ok'})
                saved[f'{p}_seed{repseed}_n{n}']=selected
    return outputs,saved


def retention_eval(model,tokenizer,entries,manifest):
    if not hasattr(tokenizer,'mask_id'):
        raise ValueError('Installed tokenizer lacks mask_id; cannot run OADM retention')
    rows=[]
    model.eval()
    for i,p in enumerate(manifest['metadata']['holdouts']):
        e=entries[p]['query'][:64]
        orig=np.array([[tokenizer.a_to_i[a] for a in r['sequence']] for r in e],dtype=np.int64)
        rng=np.random.default_rng(2026111900+i)
        for fraction in [0.15,0.5,0.8]:
            for repeat in [0,1]:
                mask = rng.random(orig.shape)<fraction
                for j in range(len(mask)):
                    if not mask[j].any():
                        mask[j, int(rng.integers(orig.shape[1]))]=True
                masked=orig.copy()
                masked[mask]=tokenizer.mask_id
                t=torch.as_tensor(masked,dtype=torch.long,device=DEVICE)
                truth=torch.as_tensor(orig,dtype=torch.long,device=DEVICE)
                m=torch.as_tensor(mask,dtype=torch.bool,device=DEVICE)
                step=torch.zeros((len(t),),dtype=torch.long,device=DEVICE)
                with torch.no_grad():
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                        logits=model(t,step)
                    if logits.shape[:2]!=t.shape or logits.shape[-1] <= int(truth.max().item()):
                        raise ValueError(f'Unexpected OADM output shape {logits.shape} for {t.shape}')
                    ce=F.cross_entropy(logits.float()[m],truth[m],reduction='sum')
                rows.append({'protein':p,'mask_fraction':fraction,'repeat':repeat,'n_masked':int(m.sum()),
                             'token_nll':float(ce.item()/m.sum().item())})
    return rows


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--condition',required=True,choices=CONDITIONS+['BASELINE'])
    pa.add_argument('--seed',type=int,required=True)
    args=pa.parse_args()
    if not (ROOT/'PREFLIGHT_OK').exists():
        raise RuntimeError('Preflight did not pass')
    if args.condition != 'BASELINE' and args.seed not in SEEDS:
        raise ValueError('Invalid supervised run seed')
    if args.condition=='BASELINE' and args.seed!=SEEDS[0]:
        raise ValueError('Unexpected baseline seed')
    out=ROOT/'runs'/f'{args.condition}_seed{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'SUCCESS').exists():
        print('Already finished:',out,flush=True)
        return
    t0=time.time()
    torch.manual_seed(20261119)
    manifest=json.loads((ROOT/'evaluation_manifest.json').read_text())
    grouped=records()
    entries=get_entries(grouped,manifest)
    model,tokenizer,scheme=loading_model(args.condition,args.seed)
    if scheme!='mask':
        raise ValueError(f'Expected EvoDiff OA-DM mask corruption, got {scheme!r}')
    features={}
    for p in sorted(entries):
        for split,ex in entries[p].items():
            features[(p,split)]=encode(model,ex,tokenizer)
    np.savez_compressed(out/'embeddings.npz',**{f'{p}___{split}':x.astype(np.float16) for (p,split),x in features.items()})
    zero,zpred=evaluate_zero_shot(features,entries,manifest)
    for r in zero:
        r['condition'],r['seed']=args.condition,args.seed
    write_csv(out/'zero_shot_metrics.csv',zero,
              ['condition','seed','protein','target','method','n_query','spearman','alpha','source_val_spearman'])
    few,preds=evaluate_few_shot(features,entries,manifest)
    for r in few:
        r['condition'],r['seed']=args.condition,args.seed
    write_csv(out/'few_shot_metrics.csv',few,
              ['condition','seed','protein','target','replicate','support_n','n_query','spearman','r2','alpha','dev_spearman','status'])
    np.savez_compressed(out/'predictions.npz',**{**{f'zero_{p}':x for p,x in zpred.items()},**{f'few_{k}':x for k,x in preds.items()}})
    retention_error=None
    try:
        retention=retention_eval(model,tokenizer,entries,manifest)
        for r in retention:
            r['condition'],r['seed']=args.condition,args.seed
        write_csv(out/'retention_metrics.csv',retention,
                  ['condition','seed','protein','mask_fraction','repeat','n_masked','token_nll'])
    except Exception as exc:
        retention_error=f'{type(exc).__name__}: {exc}'
        (out/'retention_error.txt').write_text(traceback.format_exc())
    as_json({'condition':args.condition,'seed':args.seed,'source_sha256':manifest['metadata']['source_sha256'],
             'n_zero_rows':len(zero),'n_few_rows':len(few),
             'retention_status':'COMPLETE' if retention_error is None else 'NOT_EVALUATED',
             'retention_error':retention_error,'elapsed_seconds':time.time()-t0,
             'probe_alpha_selection':'source val for zero-shot; heldout dev for few-shot',
             'heldout_query_never_used_to_fit_models_or_hyperparameters':True},out/'summary.json')
    (out/'SUCCESS').touch()
    print('SUCCESS',out,'retention:',retention_error or 'complete',flush=True)

if __name__=='__main__':
    main()
