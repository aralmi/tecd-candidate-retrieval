"""E2: same full-catalog residual model, 4 vs8 historical weeks, fixed updates."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

project=Path(__file__).resolve().parent
os.environ['RECSYS_WORKSPACE']=str(project/'runs/full_catalog_20261104')
import duckdb
import numpy as np
import pandas as pd
import torch
from recsys_lab.data import PilotData,ROOT,save_json
from recsys_lab.experiments import load_pairs,ResidualTower
from recsys_lab.baselines import fill,evaluation_users
from recsys_lab.metrics import evaluate
from recsys_lab.scoring_checks import scan,conditional_gap,nominal_mixed_q


def main(weeks=4,updates=4000):
    assert weeks in [4,8] and torch.backends.mps.is_available();torch.set_num_threads(4)
    dest=ROOT/f'weeks{weeks}';dest.mkdir(exist_ok=True)
    assert not (dest/'checkpoint.pt').exists(),'Fresh output required'
    data=PilotData();data.h,data.actions,data.ages=data.history(1293,strong_only=True)
    original_events=data.events
    if weeks==4:
        pairs=load_pairs(project/'runs/history_20261103/pairs_strong.npz')
    else:
        pairs=load_pairs(project/'runs/windows_20261103/pairs_8_windows.npz')
        # Only history reconstruction needs early events. Proposal counts and C(a)
        # remain based on the original event namespace/first_day in BOTH branches.
        c=duckdb.connect();c.execute("SET memory_limit='700MB'");c.execute('SET threads=2')
        c.read_parquet(str(project/'runs/windows_20261103/events.parquet')).create_view('early')
        f=c.execute('SELECT user_id,i,ts,day,action FROM early WHERE action>1 AND i>=0 ORDER BY user_id,ts,i,action').df();c.close()
        f['u']=pd.Index(data.users).get_indexer(f.user_id).astype(np.int32)
        assert (f.u>=0).all();data.events=f
        for a in np.unique(pairs['anchors']):
            h,actions,_=data.history(int(a),strong_only=True);ix=np.flatnonzero(pairs['anchors']==a)
            pairs['h'][ix]=h[pairs['users'][ix]];pairs['actions'][ix]=actions[pairs['users'][ix]]
        data.events=original_events;del f,h,actions
    old_n=len(pairs['positive']);keep=np.flatnonzero(pairs['h'].any(1))
    p={key:pairs[key][keep] for key in ['h','actions','positive','anchors','users']}
    forbidden=[pairs['excluded'][i] for i in keep];del pairs
    anchors=np.unique(p['anchors']);proposals={};cdfs={}
    for a in anchors:
        ids=np.flatnonzero(data.first_day<a-1)
        events=data.events[(data.events.day<a-1)&(data.events.action==2)&(data.events.i>=0)]
        cnt=np.bincount(events.i,minlength=data.n_items)[ids].astype(float)**.75
        q=.5/len(ids)+.5*cnt/cnt.sum();assert np.isclose(q.sum(),1)
        proposals[int(a)]=(ids,q);cdfs[int(a)]=np.cumsum(q)
    for a in anchors:
        assert (data.first_day[p['positive'][p['anchors']==a]]<a-1).all()
    torch.manual_seed(2026);model=ResidualTower(data.vectors,True,'mean').to('mps');model.item_mlp[2].p=0.
    digest=hashlib.sha256()
    for t in model.state_dict().values():digest.update(t.detach().cpu().numpy().tobytes())
    cfg=dict(weeks=weeks,pairs=len(keep),empty_rows_removed=old_n-len(keep),training_users=len(np.unique(p['users'])),
        positive_items=len(np.unique(p['positive'])),catalog=data.n_items,anchors=anchors.tolist(),updates=updates,
        history='strong50/past60days',negative_draws=256,proposal='IID .5uniform+.5pastclick^.75 over C(a)',
        masks='Other known positive week targets; no resampling; duplicates retained',temperature=.1,lr=.001,dropout=0,
        seed=2026,init_sha256=digest.hexdigest(),fingerprint=data.fingerprint,final_test_used=False,
        first_day_policy='Original conservative first_day held fixed for both branches',
        scope='Full historical candidate support, not a top-popularity diagnostic catalog')
    save_json(dest/'configuration.json',cfg)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    queries=evaluation_users(data);ys=[data.targets[u] for u in queries]
    ev=data.events[(data.events.day>=1264)&(data.events.day<1292)&(data.events.action==2)&(data.events.i>=0)]
    prior=np.lexsort((np.arange(data.n_items),-np.bincount(ev.i,minlength=data.n_items)))
    seen=np.zeros(data.n_items+1,bool);curves=[];best=-1.;fit=0.
    def predict(bias=None):
        model.eval()
        with torch.inference_mode():
            items=torch.cat([model.item(torch.arange(lo+1,min(lo+4096,data.n_items)+1,device='mps')) for lo in range(0,data.n_items,4096)])
            available=model.id_seen[1:].cpu().numpy()|data.vector_ok
            raw=scan(data,model,items,available,bias);pred=fill(raw,prior,200)
        del items
        return raw,pred
    def validate(step):
        nonlocal best
        model.id_seen.copy_(torch.as_tensor(seen,device='mps'))
        raw,pred=predict();m,_=evaluate(pred,ys,data.n_items)
        row=dict(updates=step,recall=m['recall_reachable@200'],full_recall=m['recall_all@200'],fit_seconds=fit,
            mps_allocated_gib=torch.mps.current_allocated_memory()/2**30,mps_driver_gib=torch.mps.driver_allocated_memory()/2**30)
        curves.append(row);pd.DataFrame(curves).to_csv(dest/'learning_curve.csv',index=False)
        if row['recall']>best:
            best=row['recall'];torch.save(dict(state={k:t.detach().cpu() for k,t in model.state_dict().items()},best_updates=step,configuration=cfg),dest/'checkpoint.pt')
            np.save(dest/'best_predictions.npy',pred)
        print('FULL_CATALOG_VALIDATION',weeks,row,flush=True)
        del raw,pred;gc.collect();torch.mps.empty_cache()
    rng=np.random.default_rng(12026);epoch=0;step=0
    while step<updates:
        epoch+=1;order=np.random.default_rng(2026+epoch).permutation(len(keep))
        for lo in range(0,len(order),128):
            if step>=updates:break
            started=time.perf_counter();model.train();rows=order[lo:lo+128];n=len(rows)
            neg=np.empty((n,256),np.int64);qdraw=np.empty((n,256),np.float32)
            for a in np.unique(p['anchors'][rows]):
                which=p['anchors'][rows]==a;ids,q=proposals[int(a)]
                loc=np.searchsorted(cdfs[int(a)],rng.random((int(which.sum()),256)))
                neg[which]=ids[loc];qdraw[which]=q[loc]
            mask=np.array([np.isin(neg[j],list(forbidden[r])) for j,r in enumerate(rows)])
            candidates=np.column_stack([p['positive'][rows],neg])+1
            h=torch.as_tensor(p['h'][rows],device='mps',dtype=torch.long);a=torch.as_tensor(p['actions'][rows],device='mps',dtype=torch.long)
            score=(model.user(h,a)[:,None,:]*model.item(torch.as_tensor(candidates,device='mps'))).sum(-1)/.1
            corrected=(score[:,1:]-torch.log(256*torch.as_tensor(qdraw,device='mps'))).masked_fill(torch.as_tensor(mask,device='mps'),-torch.inf)
            loss=(torch.logsumexp(torch.cat([score[:,:1],corrected],1),1)-score[:,0]).mean()
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            seen[candidates.ravel()]=True;seen[p['h'][rows].ravel()]=True;seen[0]=False
            torch.mps.synchronize();fit+=time.perf_counter()-started;step+=1
            if step%500==0:print('FULL_CATALOG_PROGRESS',weeks,step,'loss',float(loss.item()),flush=True)
            if step%1000==0:validate(step)
    ck=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state'])
    raw,pred=predict();np.testing.assert_array_equal(pred,np.load(dest/'best_predictions.npy'))
    q=nominal_mixed_q(data);bias=(.1*(np.log(q)-np.log(q).mean())).astype(np.float32)
    for name,(r,prediction) in [('raw',(raw,pred)),('same_fixed_prior',predict(bias))]:
        m,pu=evaluate(prediction,ys,data.n_items);pu.u=queries;gap=conditional_gap(prediction,ys,data.n_items)
        m.update(weeks=weeks,scoring=name,own_minus_shuffled=gap['gap'],fit_seconds=fit,best_updates=ck['best_updates'],
            training_pairs=len(keep),training_catalog=data.n_items,checkpoint_reload_identical=True,scope='MAIN full-catalog training and evaluation')
        save_json(dest/f'{name}_metrics.json',m);pu.to_parquet(dest/f'{name}_per_user.parquet',index=False)
        np.savez_compressed(dest/f'{name}_lists.npz',query_users=queries,raw=r,recommendations=prediction)
        print('FULL_CATALOG_RESULT',weeks,name,m['recall_reachable@200'],gap['gap'],flush=True)
    save_json(dest/'complete.json',dict(completed=True,updates=step,final_test_used=False))
    del model,opt,ck;gc.collect();torch.mps.empty_cache()


if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else 4)
