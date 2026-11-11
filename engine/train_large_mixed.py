"""Resource-led batch choice and staged large-data MPS experiments.

All models use the same item features/catalog/sampler. Only histories and training
population vary. Legacy loss and CPU id_seen bookkeeping avoid extra GPU scatters.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':os.environ['RECSYS_WORKSPACE']=str(ROOT/'runs/full_catalog_20261104')
import numpy as np
import pandas as pd
import torch
from recsys_lab.data import PilotData,save_json
from recsys_lab.experiments import ResidualTower
from recsys_lab.baselines import evaluation_users,fill
from recsys_lab.scoring_checks import scan,conditional_gap
from recsys_lab.metrics import evaluate
from recsys_lab.prefix_sequence import save_last,load_last
from recsys_lab.group_weighting import user_week_weights
from recsys_lab.contrastive import cosine_contrastive


class Inputs:
    def __init__(self,variant):
        self.root=ROOT/'runs/large_mixed_20261105';s=self.root/'data'
        assert json.loads((s/'verification.json').read_text())['passed']
        self.rows=np.load(s/f'rows_{variant}.npy')
        self.policy='strong' if variant.startswith('strong_') else 'mixed'
        self.h=np.load(s/f'{self.policy}_h.npy',mmap_mode='r')
        self.a=np.load(s/f'{self.policy}_actions.npy',mmap_mode='r')
        self.groups=np.load(s/'group_index.npy',mmap_mode='r')
        self.positive=np.load(s/'positive.npy',mmap_mode='r')
        self.anchors=np.load(s/'anchors.npy',mmap_mode='r')
        self.offsets=np.load(s/'group_offsets.npy',mmap_mode='r')
        self.proposals={}
        old=ROOT/'runs/user_week_balance_20261104/data'
        for anchor in np.unique(self.anchors):
            with np.load(old/f'proposal_{anchor}.npz') as z:
                self.proposals[int(anchor)]=tuple(z[k] for k in ['ids','q','cdf'])
        self.prior=np.load(old/'pop28.npy')
        nominal=np.load(old/'nominal_q.npy')
        self.bias=(.1*(np.log(nominal)-np.log(nominal).mean())).astype(np.float32)

    def batch(self,rows,rng):
        negative=np.empty((len(rows),256),np.int64);qdraw=np.empty(negative.shape,np.float32)
        for anchor in np.unique(self.anchors[rows]):
            which=self.anchors[rows]==anchor;ids,q,cdf=self.proposals[int(anchor)]
            loc=np.searchsorted(cdf,rng.random((int(which.sum()),256)))
            negative[which]=ids[loc];qdraw[which]=q[loc]
        group=self.groups[rows];mask=np.empty(negative.shape,bool)
        for j,g in enumerate(group):mask[j]=np.isin(negative[j],self.positive[self.offsets[g]:self.offsets[g+1]])
        return self.h[group],self.a[group],np.column_stack([self.positive[rows],negative])+1,qdraw,mask


def load_model(policy,seed=2026,use_content=True):
    data=PilotData()
    d=ROOT/'runs/large_mixed_20261105/data'
    data.h=np.load(d/f'query_{policy}_h.npy',mmap_mode='r')
    data.actions=np.load(d/f'query_{policy}_actions.npy',mmap_mode='r')
    data.events=None;data.item_map=None;data.item_ids=None;gc.collect()
    torch.manual_seed(seed);torch.mps.manual_seed(seed)
    model=ResidualTower(data.vectors,use_content,'mean').to('mps');model.item_mlp[2].p=0.
    return data,model


def step(model,optimizer,batch,weights=None,temperature=.1,monitor=None,
         loss_kind='importance',margin=.5,negative_weight=50.):
    h,a,c,qdraw,mask=batch
    ht=torch.as_tensor(h,device='mps',dtype=torch.long);at=torch.as_tensor(a,device='mps',dtype=torch.long)
    ct=torch.as_tensor(c,device='mps',dtype=torch.long)
    model.train();optimizer.zero_grad(set_to_none=True)
    cosine=(model.user(ht,at)[:,None,:]*model.item(ct)).sum(-1)
    mask_tensor=torch.as_tensor(mask,device='mps')
    if loss_kind=='importance':
        score=cosine/temperature
        corrected=(score[:,1:]-torch.log(256*torch.as_tensor(qdraw,device='mps'))).masked_fill(mask_tensor,-torch.inf)
        per_example=torch.logsumexp(torch.cat([score[:,:1],corrected],1),1)-score[:,0]
    else:
        assert loss_kind=='ccl'
        per_example,positive,negative=cosine_contrastive(cosine,mask_tensor,margin,negative_weight)
        if monitor is not None and (monitor.index+1)%100==0:
            valid=~mask_tensor;count=valid.sum().clamp_min(1)
            values=torch.stack([positive.mean(),negative.mean(),(negative*negative_weight).mean(),
                cosine[:,0].mean(),(cosine[:,1:]*valid).sum()/count,
                ((cosine[:,1:]>margin)&valid).sum()/count,valid.sum(1).float().mean()]).detach().cpu().tolist()
            monitor.loss_parts=dict(update=monitor.index+1,margin=margin,negative_weight=negative_weight,
                **dict(zip(['positive_loss','negative_hinge_mean','weighted_negative_loss','positive_cosine_mean',
                            'valid_negative_cosine_mean','active_negative_fraction','valid_negatives_per_example'],values)))
    loss=per_example.mean() if weights is None else (per_example*torch.as_tensor(weights,device='mps')).mean()
    if not torch.isfinite(loss):
        if monitor is not None:monitor.save_failure(model,optimizer,batch,'Non-finite loss before backward')
        raise FloatingPointError('Non-finite training loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if monitor is not None:monitor.before_update(model,optimizer,batch,float(loss.detach().cpu()),norm)
    optimizer.step()
    torch.mps.synchronize()
    if monitor is not None:monitor.after_update(model,optimizer,batch,float(loss.detach().cpu()))
    return float(loss.detach().cpu())


def benchmark():
    inputs=Inputs('mixed_all');data,model=load_model('mixed')
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    records=[]
    for batch_size in [128,256,512]:
        model.load_state_dict(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
        rng=np.random.default_rng(2026);timings=[];peak=0
        try:
            for t in range(20):
                rows=rng.choice(inputs.rows,batch_size,replace=False)
                start=time.perf_counter();step(model,optimizer,inputs.batch(rows,rng))
                if t>=5:timings.append(time.perf_counter()-start)
                peak=max(peak,torch.mps.driver_allocated_memory())
            records.append(dict(batch_size=batch_size,seconds_per_update=float(np.mean(timings)),
                examples_per_second=batch_size/float(np.mean(timings)),mps_driver_observed_gib=peak/2**30,
                process_highwater_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30,passed=True))
        except RuntimeError as e:
            records.append(dict(batch_size=batch_size,passed=False,error=str(e)[:300]))
        del optimizer;gc.collect();torch.mps.empty_cache()
        print('LARGE_BATCH_BENCHMARK',records[-1],flush=True)
    valid=[z for z in records if z['passed']]
    assert valid
    chosen=max(valid,key=lambda z:z['examples_per_second'])['batch_size']
    save_json(inputs.root/'batch_benchmark.json',dict(records=records,chosen_batch=chosen,
        decision='Highest measured throughput, no validation metrics used; no4GB memory target',
        observations_are_not_continuous_memory_peaks=True,final_test_used=False))


def train(variant,seed=2026,lr=.001,balanced=False,id_only=False,temperature=.1,
          maximum_epochs=3,schedule='constant',early_stop=True,run_suffix='',
          output_root=None,validation_hook=None,step_monitor=None,resume_from=None,
          loss_kind='importance',margin=.5,negative_weight=50.):
    inputs=Inputs(variant);data,model=load_model(inputs.policy,seed,not id_only)
    bench=json.loads((inputs.root/'batch_benchmark.json').read_text());batch_size=bench['chosen_batch']
    tag=f'{variant}_seed{seed}'+('' if lr==.001 else '_lr'+str(lr).replace('.','p'))+('_userweek' if balanced else '')
    if id_only:tag+='_idonly'
    if temperature!=.1:tag+='_tau'+str(temperature).replace('.','p')
    assert loss_kind in ['importance','ccl']
    if loss_kind=='ccl':tag+='_ccl_m'+str(margin).replace('.','p')+'_w'+format(negative_weight,'g')
    tag+=run_suffix
    dest=(Path(output_root) if output_root else inputs.root)/tag;dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'configuration.json').exists(),'Fresh run required'
    source=Path(__file__).read_bytes()
    (dest/'trainer_snapshot.py').write_bytes(source)
    info=json.loads((inputs.root/'data/complete.json').read_text())['subsets'].get(variant)
    if info is None:
        # Independently registered all-user strong subset shares the same caches.
        assert variant=='strong_all' and len(inputs.rows)==436991
        train_users=np.load(inputs.root/'data/users.npy',mmap_mode='r')
        info=dict(pairs=len(inputs.rows),training_users=len(np.unique(train_users[inputs.rows])),
            positive_items=len(np.unique(inputs.positive[inputs.rows])))
        assert info['training_users']==86983 and info['positive_items']==64918
    row_weights=None
    if balanced:
        train_users=np.load(inputs.root/'data/users.npy',mmap_mode='r')
        w,inverse,counts,_=user_week_weights(train_users[inputs.rows],inputs.anchors[inputs.rows])
        np.testing.assert_allclose(np.bincount(inverse,weights=w),len(w)/len(counts),rtol=1e-6)
        row_weights=np.zeros(len(inputs.positive),np.float32);row_weights[inputs.rows]=w
        np.save(dest/'selected_row_weights.npy',w)
        save_json(dest/'weight_statistics.json',dict(groups=len(counts),mean=float(w.mean()),
            minimum=float(w.min()),maximum=float(w.max()),effective_sample_size=float(w.sum()**2/(w*w).sum())))
    assert maximum_epochs>=3 and schedule in ['constant','decay_after3']
    steps_per_epoch=int(np.ceil(len(inputs.rows)/batch_size));max_steps=steps_per_epoch*maximum_epochs
    cfg=dict(variant=variant,seed=seed,**info,batch_size=batch_size,maximum_epochs=maximum_epochs,maximum_steps=max_steps,
        validation_every=500,patience_checks=2,minimum_improvement=.001,minimum_steps_before_early_stop=1000,
        history=inputs.policy+'100',catalog=data.n_items,lr=lr,temperature=temperature if loss_kind=='importance' else None,negative_draws=256,
        sampler='Same frozen as-of-anchor proposal for every variant; fixed original historical candidate support',
        init_sha256=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest(),
        trainer_sha256=hashlib.sha256(source).hexdigest(),torch_version=str(torch.__version__),
        objective='equal_user_week' if balanced else 'equal_positive_pair',
        use_content=not id_only,
        loss_kind=loss_kind,ccl_margin=margin if loss_kind=='ccl' else None,
        ccl_negative_weight=negative_weight if loss_kind=='ccl' else None,
        schedule=schedule,early_stop_enabled=early_stop,
        step_monitor=None if step_monitor is None else step_monitor.protocol,
        resume_from=None if resume_from is None else str(resume_from),
        data_context=getattr(inputs,'context',None),
        fingerprint=data.fingerprint,final_test_used=False)
    save_json(dest/'configuration.json',cfg)
    if step_monitor is not None:step_monitor.attach(dest)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    rng=np.random.default_rng(12026);seen=np.zeros(data.n_items+1,bool)
    query=evaluation_users(data);targets=[data.targets[u] for u in query]
    progress=dict(updates=0,epoch=1,cursor=0,presentations=0,fit_seconds=0.,best_recall=-1.,
        patience_best=-1.,checks_without_gain=0,curves=[],stop_reason='maximum_epochs')
    if resume_from is not None:
        import shutil
        resume_from=Path(resume_from)
        parent=resume_from.parent
        source_cfg=json.loads((parent/'configuration.json').read_text())
        for key in ['init_sha256','fingerprint','seed','pairs','batch_size','lr','temperature','maximum_steps']:
            assert cfg[key]==source_cfg[key],key
        progress=load_last(resume_from,model,optimizer,rng)
        assert progress['updates']<max_steps
        seen=model.id_seen.detach().cpu().numpy().copy()
        for name in ['checkpoint.pt','best_predictions.npy']:
            shutil.copy2(parent/name,dest/name)
        if (parent/'training_loss.csv').exists():
            previous=pd.read_csv(parent/'training_loss.csv')
            previous=previous[previous.updates<=progress['updates']]
            previous.to_csv(dest/'training_loss.csv',index=False)
        pd.DataFrame(progress['curves']).to_csv(dest/'learning_curve.csv',index=False)
        print('RESUMED_TRAINING',progress['updates'],'epoch',progress['epoch'],'cursor',progress['cursor'],flush=True)

    def predict(bias=None):
        model.eval();model.id_seen.copy_(torch.as_tensor(seen,device='mps'))
        with torch.inference_mode():
            items=torch.cat([model.item(torch.arange(lo+1,min(lo+4096,data.n_items)+1,device='mps'))
                for lo in range(0,data.n_items,4096)])
            raw=scan(data,model,items,model.id_seen[1:].cpu().numpy()|(data.vector_ok if model.use_content else False),bias)
            pred=fill(raw,inputs.prior,200)
        del items;return raw,pred

    def validate():
        raw,pred=predict();metrics,_=evaluate(pred,targets,data.n_items)
        recall=metrics['recall_reachable@200']
        row=dict(updates=progress['updates'],presentations=progress['presentations'],
            fraction_of_one_pass=progress['presentations']/len(inputs.rows),recall=recall,fit_seconds=progress['fit_seconds'])
        progress['curves'].append(row);pd.DataFrame(progress['curves']).to_csv(dest/'learning_curve.csv',index=False)
        if recall>progress['best_recall']:
            progress['best_recall']=recall
            torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                best_updates=progress['updates'],configuration=cfg),dest/'checkpoint.pt')
            np.save(dest/'best_predictions.npy',pred)
        if recall>progress['patience_best']+.001:
            progress['patience_best']=recall;progress['checks_without_gain']=0
        else:progress['checks_without_gain']+=1
        save_last(dest/'last.pt',model,optimizer,rng,progress)
        if validation_hook is not None:
            validation_hook(model,data,inputs,dest,progress['updates'],pred)
        print('LARGE_VALIDATION',variant,row,'patience',progress['checks_without_gain'],flush=True)
        del raw,pred;gc.collect();torch.mps.empty_cache()
    order=np.random.default_rng(seed+progress['epoch']).permutation(len(inputs.rows))
    losses=[];loss_rows=[]
    if resume_from is not None and (dest/'training_loss.csv').exists():
        loss_rows=pd.read_csv(dest/'training_loss.csv').to_dict('records')
    if validation_hook is not None and resume_from is None:
        validation_hook(model,data,inputs,dest,0)
    while progress['updates']<max_steps:
        if progress['cursor']>=len(inputs.rows):
            progress['epoch']+=1;progress['cursor']=0
            order=np.random.default_rng(seed+progress['epoch']).permutation(len(inputs.rows))
        rows=inputs.rows[order[progress['cursor']:progress['cursor']+batch_size]]
        if schedule=='decay_after3':
            fraction=np.clip((progress['updates']-3*steps_per_epoch)/max(1,max_steps-1-3*steps_per_epoch),0.,1.)
            current_lr=lr*(.1+.9*.5*(1+np.cos(np.pi*fraction)))
            for group in optimizer.param_groups:group['lr']=float(current_lr)
        start=time.perf_counter();batch=inputs.batch(rows,rng)
        loss=step(model,optimizer,batch,None if row_weights is None else row_weights[rows],temperature,step_monitor,
                  loss_kind,margin,negative_weight)
        seen[batch[2].ravel()]=True;seen[batch[0].ravel()]=True;seen[0]=False
        progress['fit_seconds']+=time.perf_counter()-start
        progress['updates']+=1;progress['cursor']+=len(rows);progress['presentations']+=len(rows);losses.append(loss)
        if progress['updates']%100==0:
            loss_rows.append(dict(updates=progress['updates'],presentations=progress['presentations'],mean_loss=float(np.mean(losses)),lr=optimizer.param_groups[0]['lr']))
            pd.DataFrame(loss_rows).to_csv(dest/'training_loss.csv',index=False)
            print('LARGE_PROGRESS',variant,progress['updates'],'examples',progress['presentations'],
                'loss',float(np.mean(losses)),'driverGiB',torch.mps.driver_allocated_memory()/2**30,flush=True);losses=[]
        if progress['updates']%500==0 or progress['updates']==max_steps:
            validate()
            if early_stop and progress['updates']>=1000 and progress['checks_without_gain']>=2:
                progress['stop_reason']='Two checks without0.001 improvement';break
        if step_monitor is not None and step_monitor.stop_reason:
            progress['stop_reason']=step_monitor.stop_reason
            if progress['updates']%500!=0 and progress['updates']!=max_steps:validate()
            break
    # last optimizer remains the continuation point; best weights used below.
    ck=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state'])
    seen=model.id_seen.detach().cpu().numpy().copy()
    raw,pred=predict();np.testing.assert_array_equal(pred,np.load(dest/'best_predictions.npy'))
    for score,(raw_list,recommendations) in [('raw',(raw,pred)),('same_fixed_prior',predict(inputs.bias))]:
        m,pu=evaluate(recommendations,targets,data.n_items);pu.u=query
        before,_=evaluate(raw_list,targets,data.n_items);gap=conditional_gap(recommendations,targets,data.n_items)
        m.update(variant=variant,seed=seed,lr=lr,best_updates=ck['best_updates'],completed_updates=progress['updates'],
            presentations=progress['presentations'],fraction_of_one_pass=progress['presentations']/len(inputs.rows),
            fit_seconds=progress['fit_seconds'],raw_recall=before['recall_reachable@200'],own_minus_shuffled=gap['gap'],
            unique_raw_items=len(np.unique(raw_list[raw_list>=0])),stop_reason=progress['stop_reason'],final_test_used=False)
        _,frequency=np.unique(raw_list[raw_list>=0],return_counts=True)
        share=frequency/frequency.sum() if len(frequency) else np.array([])
        m.update(temperature=cfg['temperature'],loss_kind=loss_kind,
            ccl_margin=cfg['ccl_margin'],ccl_negative_weight=cfg['ccl_negative_weight'],
            prior_coefficient=.1 if score=='same_fixed_prior' else 0.,
            top10_item_share=float(np.sort(share)[-10:].sum()),effective_item_count=float(1/(share@share)) if len(share) else 0.)
        save_json(dest/f'{score}_metrics.json',m);pu.to_parquet(dest/f'{score}_per_user.parquet',index=False)
        np.savez_compressed(dest/f'{score}_lists.npz',query_users=query,raw=raw_list,recommendations=recommendations)
        print('LARGE_RESULT',variant,score,m['recall_reachable@200'],gap['gap'],flush=True)
    save_json(dest/'complete.json',dict(completed=True,**progress,final_test_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--variant',choices=['strong_old','strong_all','mixed_old','mixed_all']);parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--balanced-user-week',action='store_true')
    parser.add_argument('--id-only',action='store_true')
    parser.add_argument('--temperature',type=float,default=.1)
    args=parser.parse_args();assert torch.backends.mps.is_available() and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    torch.set_num_threads(4)
    if args.benchmark:benchmark()
    else:assert args.variant and args.temperature>0;train(args.variant,args.seed,args.lr,args.balanced_user_week,args.id_only,args.temperature)
