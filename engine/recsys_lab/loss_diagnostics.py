"""Small-vocabulary diagnostic: exact masked partition vs IID sampled partitions.

The importance estimator keeps the positive term exact and weights sampled negative
terms by1/(K*q). It is NOT TF sampled_softmax and NOT the previous unique sampler.
Its partition estimate is unbiased; log-loss/gradient at finite K are not guaranteed
unbiased. Duplicate draws remain; known positives are masked without resampling.
"""
import gc
import hashlib
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .data import ROOT,save_json
from .baselines import evaluation_users,fill,topk
from .experiments import WeightedTower,ResidualTower,load_pairs
from .metrics import evaluate
from .retrieval import exact_vector_topk
from .scoring_checks import conditional_gap


def sampled_partition_loss(scores,labels,negative_ids,probabilities,known_mask,importance):
    rows=torch.arange(len(labels),device=scores.device)
    pos=scores[rows,labels]
    neg=scores.gather(1,negative_ids)
    if importance:neg=neg-torch.log(negative_ids.shape[1]*probabilities)
    neg=neg.masked_fill(known_mask,-torch.inf)
    return (torch.logsumexp(torch.cat([pos[:,None],neg],1),1)-pos).mean()


def prepare(data,project,size=5000):
    source=load_pairs(project/'runs/history_20261103/pairs_strong.npz')
    first=min(data.cfg['train_anchors'])
    before=data.events[(data.events.day<first-1)&(data.events.action==2)&(data.events.i>=0)]
    counts=np.zeros(data.n_items,np.int64);s=before.groupby('i').size();counts[s.index]=s.to_numpy()
    vocabulary=np.lexsort((np.arange(data.n_items),-counts))[:size].astype(np.int32)
    assert (counts[vocabulary]>0).all() and (data.first_day[vocabulary]<first-1).all()
    lookup=np.full(data.n_items,-1,np.int32);lookup[vocabulary]=np.arange(size)
    keep=np.flatnonzero(lookup[source['positive']]>=0)
    pairs={key:source[key][keep] for key in ['h','actions','positive','anchors','users']}
    pairs['labels']=lookup[pairs['positive']]
    forbidden=[np.array([lookup[i] for i in source['excluded'][r] if lookup[i]>=0],np.int32) for r in keep]
    probabilities={}
    for anchor in data.cfg['train_anchors']:
        f=data.events[(data.events.day<anchor-1)&(data.events.action==2)&(data.events.i>=0)]
        cnt=np.zeros(data.n_items,np.float64);s=f.groupby('i').size();cnt[s.index]=s.to_numpy()
        w=cnt[vocabulary]**.75;w/=w.sum()
        probabilities[anchor]=.5/size+.5*w
    original_targets=data.targets
    allowed=set(map(int,vocabulary))
    data.targets=[{i if isinstance(i,int) and i in allowed else 'outside_diag:'+str(i) for i in y} for y in original_targets]
    data.h,data.actions,data.ages=data.history(data.cfg['validation_anchor'],strong_only=True)
    data.cfg['history_mode']='strong_only'
    queries=evaluation_users(data);targets=[data.targets[u] for u in queries]
    pop28=data.events[(data.events.day>=data.cfg['fit_end']-28)&(data.events.day<data.cfg['fit_end'])&(data.events.action==2)&(data.events.i>=0)].groupby('i').size()
    counts28=np.zeros(data.n_items);counts28[pop28.index]=pop28.to_numpy()
    prior=vocabulary[np.lexsort((vocabulary,-counts28[vocabulary]))]
    manifest=dict(candidate_items=size,selection_cutoff=first-1,selection='Top past click counts BEFORE first training gap',
        training_pairs=len(keep),training_users=len(np.unique(pairs['users'])),positive_items=len(np.unique(pairs['positive'])),
        vocabulary_sha256=hashlib.sha256(vocabulary.tobytes()).hexdigest(),
        comparison_scope='Diagnostic vocabulary only. Reachable Recall conditional on this vocabulary; full Recall retains all original targets.',
        vocabulary_selected_without_validation=True,history='last50 strong events in60days; empty rows retained',
        item_dropout=0,negative_draws=64,negative_sampling='IID with replacement; duplicates retained; mask known positives without resampling',
        modes=['full','mixed_uncorrected','mixed_importance'],epochs=4,validation_interval_updates=500,seed=2026)
    save_json(ROOT/'loss_diagnostic/protocol.json',manifest)
    np.save(ROOT/'loss_diagnostic/vocabulary.npy',vocabulary)
    return dict(pairs=pairs,forbidden=forbidden,q=probabilities,cdf={a:np.cumsum(q) for a,q in probabilities.items()},
                vocabulary=vocabulary,prior=prior,queries=queries,targets=targets,manifest=manifest)


@torch.inference_mode()
def predict(data,ctx,model,bias=None):
    model.eval();v=ctx['vocabulary'];ids=torch.as_tensor(v.astype(np.int64)+1,device='mps')
    items=model.item(ids)
    if bias is not None:
        items=torch.cat([items,torch.as_tensor(bias,device='mps',dtype=torch.float32)[:,None]],1)
    out=np.full((len(ctx['queries']),200),-1,np.int32)
    for lo in range(0,len(out),64):
        users=ctx['queries'][lo:lo+64]
        h=torch.as_tensor(data.h[users],device='mps',dtype=torch.long)
        a=torch.as_tensor(data.actions[users],device='mps',dtype=torch.long)
        query=model.user(h,a)
        if bias is not None:query=torch.cat([query,torch.ones((len(users),1),device='mps')],1)
        local=exact_vector_topk(query,items,np.ones(len(v),bool),200,topk)
        valid=data.h[users].any(axis=1)
        out[lo:lo+len(users)][valid]=v[local[valid]]
    return fill(out,ctx['prior'],200)


def run(data,ctx,mode,epochs=4,tag=None,residual=False):
    assert torch.backends.mps.is_available() and mode in {'full','mixed_uncorrected','mixed_importance'}
    dest=ROOT/'loss_diagnostic'/(tag or mode);dest.mkdir(exist_ok=True)
    assert not (dest/'checkpoint.pt').exists(),'Fresh tags required'
    torch.manual_seed(2026)
    model_class=ResidualTower if residual else WeightedTower
    model=model_class(data.vectors,True,'mean').to('mps')
    model.item_mlp[2].p=0.
    initial=hashlib.sha256()
    for tensor in model.state_dict().values():initial.update(tensor.detach().cpu().numpy().tobytes())
    save_json(dest/'configuration.json',dict(ctx['manifest'],mode=mode,init_sha256=initial.hexdigest(),
        user_encoder='profile_plus_mlp' if residual else 'mlp_only',residual_coefficient=1. if residual else 0.,
        learning_rate=.001,temperature=.1,gradient_clip=1.,optimizer='AdamW,weight_decay1e-5',
        importance_definition='Negative exp-score divided by K*q; positive exp-score exact',
        finite_sample_loss_unbiased=False))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    p=ctx['pairs'];v=ctx['vocabulary'];size=len(v)
    candidates=torch.as_tensor(v.astype(np.int64)+1,device='mps')
    seen=np.zeros(data.n_items+1,bool);seen[v+1]=True
    best=-1.;step=0;curves=[];fit_seconds=0.;masked=draws=0
    def validate(epoch):
        nonlocal best
        model.id_seen.copy_(torch.as_tensor(seen,device='mps'))
        pred=predict(data,ctx,model)
        m,_=evaluate(pred,ctx['targets'],data.n_items)
        row=dict(epoch=epoch,updates=step,recall_diag_at_200=m['recall_reachable@200'],
            recall_all_at_200=m['recall_all@200'],ndcg_at_200=m['ndcg@200'],fit_seconds=fit_seconds)
        curves.append(row);pd.DataFrame(curves).to_csv(dest/'learning_curve.csv',index=False)
        if row['recall_diag_at_200']>best:
            best=row['recall_diag_at_200']
            torch.save(dict(state={k:t.detach().cpu() for k,t in model.state_dict().items()},mode=mode,
                residual=residual,tag=tag or mode,
                best_updates=step,vocabulary_sha256=ctx['manifest']['vocabulary_sha256']),dest/'checkpoint.pt')
        print('LOSS_DIAGNOSTIC',mode,row,flush=True)
        model.train()
    for epoch in range(1,epochs+1):
        order=np.random.default_rng(2026+epoch).permutation(len(p['positive']))
        rng=np.random.default_rng(12026+epoch)
        loss_total=0.;examples=0
        for lo in range(0,len(order),128):
            started=time.perf_counter();model.train();rows=order[lo:lo+128];batch=len(rows)
            neg=np.empty((batch,64),np.int64);prob=np.empty((batch,64),np.float32)
            for anchor in np.unique(p['anchors'][rows]):
                which=p['anchors'][rows]==anchor
                loc=np.searchsorted(ctx['cdf'][int(anchor)],rng.random((int(which.sum()),64)))
                neg[which]=loc;prob[which]=ctx['q'][int(anchor)][loc]
            known=np.zeros((batch,size),bool)
            for j,r in enumerate(rows):known[j,ctx['forbidden'][r]]=True
            labels=p['labels'][rows]
            h=torch.as_tensor(p['h'][rows],device='mps',dtype=torch.long)
            a=torch.as_tensor(p['actions'][rows],device='mps',dtype=torch.long)
            scores=model.user(h,a)@model.item(candidates).T/.1
            label_tensor=torch.as_tensor(labels,device='mps',dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            if mode=='full':
                other=known.copy();other[np.arange(batch),labels]=False
                loss=F.cross_entropy(scores.masked_fill(torch.as_tensor(other,device='mps'),-torch.inf),label_tensor)
            else:
                collision=known[np.arange(batch)[:,None],neg]
                masked+=int(collision.sum());draws+=collision.size
                loss=sampled_partition_loss(scores,label_tensor,torch.as_tensor(neg,device='mps'),
                    torch.as_tensor(prob,device='mps'),torch.as_tensor(collision,device='mps'),mode=='mixed_importance')
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            seen[p['h'][rows].ravel()]=True;seen[0]=False
            loss_total+=loss.item()*batch;examples+=batch;step+=1
            torch.mps.synchronize();fit_seconds+=time.perf_counter()-started
            if step%500==0:validate(epoch)
        print('EPOCH_LOSS',mode,epoch,loss_total/examples,flush=True)
    if not curves or curves[-1]['updates']!=step:validate(epochs)
    checkpoint=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state']);pred=predict(data,ctx,model)
    clone=model_class(data.vectors,True,'mean',content_tensor=model.content).to('mps');clone.item_mlp[2].p=0.
    clone.load_state_dict(checkpoint['state']);np.testing.assert_array_equal(pred,predict(data,ctx,clone))
    summary,pu=evaluate(pred,ctx['targets'],data.n_items);pu.u=ctx['queries']
    diag=conditional_gap(pred,ctx['targets'],data.n_items)
    summary.update(mode=mode,tag=tag or mode,user_encoder='profile_plus_mlp' if residual else 'mlp_only',
        fit_seconds=fit_seconds,updates=step,best_updates=checkpoint['best_updates'],
        diagnostic_candidate_items=size,own_minus_shuffled=diag['gap'],shuffled_recall=diag['shuffled_mean'],
        masked_negative_fraction=masked/draws if draws else None,reload_identical=True,
        scope=f'DIAGNOSTIC{size}, not the main million-item benchmark')
    save_json(dest/'metrics.json',summary);pu.to_parquet(dest/'per_user.parquet',index=False)
    np.savez_compressed(dest/'lists.npz',query_users=ctx['queries'],recommendations=pred)
    print('DIAGNOSTIC_COMPLETE',mode,summary['recall_reachable@200'],diag,flush=True)
    del opt,model,clone,checkpoint;gc.collect();torch.mps.empty_cache()
    return summary
