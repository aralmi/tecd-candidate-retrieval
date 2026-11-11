"""Frozen-weight inference diagnostics; the prior offset is explicitly heuristic."""
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import evaluation_users,fill,popularity,topk
from .data import ROOT,save_json
from .experiments import WeightedTower
from .metrics import evaluate
from .retrieval import exact_vector_topk


def nominal_mixed_q(data):
    """Nominal proposal mixture, not exact accepted unique-negative probabilities."""
    counts=np.zeros(data.n_items,np.float64)
    frame=data.events[(data.events.day<data.cfg['fit_end'])&(data.events.action==2)&(data.events.i>=0)]
    s=frame.groupby('i').size();counts[s.index]=s.to_numpy()
    popular=counts**.75;popular/=popular.sum()
    q=.5/data.n_items+.5*popular
    assert np.isclose(q.sum(),1) and (q>0).all()
    return q


@torch.inference_mode()
def scan(data,model,items,available,bias=None):
    users=evaluation_users(data)
    raw=np.full((len(users),200),-1,np.int32)
    if bias is not None:
        # No re-normalization: [u,1] dot [v,bias] = u dot v + bias.
        items=torch.cat([items,torch.as_tensor(bias,device='mps',dtype=torch.float32)[:,None]],1)
    for lo in range(0,len(users),64):
        batch=users[lo:lo+64]
        h=torch.as_tensor(data.h[batch],device='mps',dtype=torch.long)
        a=torch.as_tensor(data.actions[batch],device='mps',dtype=torch.long)
        query=model.user(h,a)
        if bias is not None:query=torch.cat([query,torch.ones((len(batch),1),device='mps')],1)
        selected=exact_vector_topk(query,items,available,200,topk)
        valid=data.h[batch].any(axis=1);raw[lo:lo+len(batch)][valid]=selected[valid]
    return raw


def conditional_gap(pred,targets,n_items):
    reachable=[{i for i in y if isinstance(i,int)} for y in targets]
    counts=np.array([len(y) for y in reachable]);valid=counts>0
    keys=np.array([j*n_items+i for j,y in enumerate(reachable) for i in y],np.int64)
    base=np.arange(len(targets),dtype=np.int64)[:,None]*n_items
    def score(p):
        hit=np.isin(base+p,keys)&(p>=0)
        return float((hit[valid].sum(1)/counts[valid]).mean())
    shuffled=[score(pred[np.random.default_rng(2026+i).permutation(len(pred))]) for i in range(12)]
    return dict(own=score(pred),shuffled_mean=float(np.mean(shuffled)),gap=score(pred)-np.mean(shuffled))


@torch.inference_mode()
def inspect_model(data,source,tag):
    assert torch.backends.mps.is_available()
    source=Path(source);dest=ROOT/'scoring'/tag;dest.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=True)
    cfg=checkpoint['configuration'];assert checkpoint['fingerprint']==data.fingerprint
    model=WeightedTower(data.vectors,cfg['use_content'],'mean').to('mps')
    model.load_state_dict(checkpoint['state']);model.eval()
    blocks=[]
    for lo in range(0,data.n_items,4096):
        blocks.append(model.item(torch.arange(lo+1,min(lo+4096,data.n_items)+1,device='mps')))
    items=torch.cat(blocks);del blocks
    available=model.id_seen[1:].cpu().numpy()
    if cfg['use_content']:available=available|data.vector_ok
    users=evaluation_users(data);targets=[data.targets[u] for u in users]
    popular=popularity(data)
    q=nominal_mixed_q(data)
    # tau=.1, matching model.forward's dot/.1; centered logQ only removes a constant.
    logq=np.log(q);bias=(.1*(logq-logq.mean())).astype(np.float32)
    rows=[]
    for name,offset in [('raw_dot',None),('nominal_logq_offset',bias)]:
        started=time.perf_counter();raw=scan(data,model,items,available,offset)
        if offset is None:
            old=np.load(source/'raw.npy',mmap_mode='r')[users]
            np.testing.assert_array_equal(raw,old)
            save_json(dest/'checkpoint_parity.json',dict(raw_ids_identical=True,queries=len(users)))
        final=fill(raw,popular,200)
        metrics,pu=evaluate(final,targets,data.n_items)
        pu.u=users;pu.to_parquet(dest/f'{name}_per_user.parquet',index=False)
        diag=conditional_gap(final,targets,data.n_items)
        metrics.update(source_model=tag,scoring=name,personalization_gap=diag['gap'],shuffled_recall=diag['shuffled_mean'],
            seconds=time.perf_counter()-started,weights_changed=False,source_sampler=cfg['sampler'],
            caveat='Heuristic prior adjustment, NOT exact correction of rejection/dedup sampler',fingerprint=data.fingerprint)
        save_json(dest/f'{name}.json',metrics)
        np.savez_compressed(dest/f'{name}_lists.npz',query_users=users,raw=raw,recommendations=final)
        rows.append(metrics)
        print('SCORING',tag,name,metrics['recall_reachable@200'],'gap',diag['gap'],flush=True)
    # Pure prior ordering uses no user profile and is a necessary control.
    prior_ids=np.lexsort((np.arange(data.n_items),-q))
    prior_ids=prior_ids[available[prior_ids]][:200]
    prior=np.tile(prior_ids,(len(users),1)).astype(np.int32)
    prior_metrics,_=evaluate(prior,targets,data.n_items)
    save_json(dest/'prior_only.json',prior_metrics)
    save_json(dest/'protocol.json',dict(source=str(source),checkpoint_sha256=hashlib.sha256((source/'checkpoint.pt').read_bytes()).hexdigest(),
        temperature=.1,coefficient=1.,coefficient_tuned=False,proposal='0.5 uniform +0.5 past click frequency^0.75',
        exact_accepted_q_known=False,uniform_model_role='Same offset applied to uniform-trained model as a generic popularity-control',
        note='Inference-time diagnostic; not a reproduction of sampled-softmax training correction',
        reference='https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/'))
    del model,items,checkpoint;gc.collect();torch.mps.empty_cache()
    return rows
