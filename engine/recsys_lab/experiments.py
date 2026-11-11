"""Controlled follow-up experiments; preserves the original pilot implementation."""

import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .data import ROOT, STRENGTH, save_json
from .baselines import evaluation_users, fill, popularity, record
from .metrics import evaluate
from .neural import TwoTower, recommend


class WeightedTower(TwoTower):
    """Bind each action to its item by a fixed nonnegative action weight."""

    def user(self, history, actions):
        weights = torch.as_tensor(STRENGTH, device=history.device)[actions]
        weights = weights * (history != 0)
        tokens = self.item(history)
        value = (tokens * weights.unsqueeze(-1)).sum(1) / weights.sum(
            1, keepdim=True
        ).clamp_min(1e-12)
        return F.normalize(self.user_mlp(value), dim=-1)


class ResidualTower(WeightedTower):
    """One fixed identity skip; identical parameters/init to WeightedTower."""

    def user(self, history, actions):
        weights = torch.as_tensor(STRENGTH,device=history.device)[actions]*(history!=0)
        profile = (self.item(history)*weights.unsqueeze(-1)).sum(1)/weights.sum(1,keepdim=True).clamp_min(1e-12)
        return F.normalize(profile+self.user_mlp(profile),dim=-1)


class Sampler:
    def __init__(self, data, mode):
        assert mode in {"uniform", "mixed"}
        self.mode = mode
        self.pools, self.pop_ids, self.cdfs = {}, {}, {}
        for a in data.cfg['train_anchors']:
            self.pools[a] = np.flatnonzero(data.first_day < a - 1)
            f = data.events[(data.events.day < a-1) & (data.events.action == 2) & (data.events.i >= 0)]
            counts = f.groupby('i').size()
            ids = counts.index.to_numpy(dtype=np.int64)
            valid = data.first_day[ids] < a-1
            ids, count = ids[valid], counts.to_numpy()[valid]
            self.pop_ids[a] = ids
            self.cdfs[a] = np.cumsum(count.astype(float)**0.75)
            self.cdfs[a] /= self.cdfs[a][-1]
        self.proposals = self.rejections = self.rows = 0

    def batch(self, pairs, rows, rng):
        result = []
        for row in rows:
            anchor = int(pairs['anchors'][row])
            forbidden = pairs['excluded'][row]
            chosen = set()
            # Components have exact 32/32 counts after excluding positives/duplicates.
            for component, count in [('uniform', 64 if self.mode == 'uniform' else 32),
                                     ('popular', 0 if self.mode == 'uniform' else 32)]:
                accepted = 0
                while accepted < count:
                    if component == 'uniform':
                        proposals = rng.choice(self.pools[anchor], size=128, replace=True)
                    else:
                        ix = np.searchsorted(self.cdfs[anchor], rng.random(128))
                        proposals = self.pop_ids[anchor][ix]
                    for item in proposals:
                        item = int(item)
                        self.proposals += 1
                        if item in forbidden or item in chosen:
                            self.rejections += 1
                            continue
                        chosen.add(item)
                        accepted += 1
                        if accepted == count:
                            break
            assert len(chosen) == 64 and not (chosen & forbidden)
            result.append([int(pairs['positive'][row]), *sorted(chosen)])
            self.rows += 1
        return np.asarray(result, dtype=np.int64) + 1


def compact_metrics(data, full):
    q = evaluation_users(data)
    return evaluate(full[q], [data.targets[u] for u in q], data.n_items)[0]


def train(data, pairs, tag, weighted=False, sampler_mode='uniform', epochs=4,
          use_content=False, max_updates_per_epoch=None, learning_rate=0.001):
    assert torch.backends.mps.is_available(), 'MPS required'
    dest = ROOT/'artifacts'/tag
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest/'checkpoint.pt').exists(), 'Use a fresh experiment tag'
    torch.manual_seed(2026)
    model = (WeightedTower if weighted else TwoTower)(data.vectors, use_content, 'mean').to('mps')
    init_hash = hashlib.sha256()
    for key, value in model.state_dict().items():
        init_hash.update(value.detach().cpu().numpy().tobytes())
    cfg = dict(data.cfg, tag=tag, weighted=weighted, sampler=sampler_mode,
               epochs=epochs, use_content=use_content, learning_rate=learning_rate,
               max_updates_per_epoch=max_updates_per_epoch,
               order_rng='2026 + epoch; independent of negative sampler',
               negative_rng='12026 + epoch; independent of row ordering',
               init_sha256=init_hash.hexdigest(), evaluation_fingerprint=data.fingerprint)
    save_json(dest/'configuration.json', cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    sampler = Sampler(data, sampler_mode)
    pop, queries = popularity(data), evaluation_users(data)
    seen = np.zeros(data.n_items+1, bool)
    curves, best, best_epoch, total_fit, updates = [], -1., 0, 0., 0
    trained_users, trained_positives = set(), set()
    for epoch in range(1, epochs+1):
        start=time.perf_counter()
        model.train()
        order=np.random.default_rng(2026+epoch).permutation(len(pairs['positive']))
        negative_rng=np.random.default_rng(12026+epoch)
        order_digest=hashlib.sha256(order.tobytes()).hexdigest()
        loss_sum=n_seen=0
        for step,lo in enumerate(range(0,len(order),128)):
            if max_updates_per_epoch and step>=max_updates_per_epoch: break
            rows=order[lo:lo+128]
            candidates=sampler.batch(pairs,rows,negative_rng)
            seen[candidates.ravel()]=True
            seen[pairs['h'][rows].ravel()]=True
            seen[0]=False
            trained_users.update(map(int,pairs['users'][rows]))
            trained_positives.update(map(int,pairs['positive'][rows]))
            h=torch.as_tensor(pairs['h'][rows],device='mps',dtype=torch.long)
            a=torch.as_tensor(pairs['actions'][rows],device='mps',dtype=torch.long)
            c=torch.as_tensor(candidates,device='mps')
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(h,a,c),torch.zeros(len(rows),device='mps',dtype=torch.long))
            assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            loss_sum+=loss.item()*len(rows); n_seen+=len(rows); updates+=1
        model.id_seen.copy_(torch.as_tensor(seen,device='mps'))
        torch.mps.synchronize()
        fit=time.perf_counter()-start; total_fit+=fit
        raw,predict_seconds=recommend(data,model,'mps')
        full=fill(raw[queries],pop,data.cfg['k'])
        metrics,_=evaluate(full,[data.targets[u] for u in queries],data.n_items)
        recall=metrics['recall_reachable@200']
        row=dict(epoch=epoch,train_loss=loss_sum/n_seen,recall_at_50=metrics['recall_reachable@50'],
                 recall_at_200=recall,ndcg_at_200=metrics['ndcg@200'],fit_seconds=fit,
                 predict_seconds=predict_seconds,updates=updates,examples=n_seen,order_sha256=order_digest)
        curves.append(row); pd.DataFrame(curves).to_csv(dest/'learning_curve.csv',index=False)
        print('EPOCH',tag,json.dumps(row),flush=True)
        if recall>best:
            best,best_epoch=recall,epoch
            torch.save({'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'configuration':cfg,'best_epoch':epoch,'fingerprint':data.fingerprint},dest/'checkpoint.pt')
        if epoch-best_epoch>=3: break
    ck=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ck['state'])
    raw,predict_seconds=recommend(data,model,'mps')
    # Reload into the same class and shared fixed features, then compare exact IDs.
    clone=(WeightedTower if weighted else TwoTower)(data.vectors,use_content,'mean',content_tensor=model.content).to('mps')
    clone.load_state_dict(ck['state'])
    again,_=recommend(data,clone,'mps'); np.testing.assert_array_equal(raw,again)
    summary=record(data,tag,raw,total_fit,predict_seconds,dict(device='mps',best_epoch=best_epoch,
        epochs_completed=len(curves),training_pairs=len(pairs['positive']),optimizer_updates=updates,
        sampler=sampler_mode,weighted=weighted,reload_identical=True,learning_rate=learning_rate,
        actual_training_users=len(trained_users),actual_positive_items=len(trained_positives)))
    save_json(dest/'sampler_stats.json',dict(rows=sampler.rows,proposals=sampler.proposals,
        rejections=sampler.rejections,negative_count=64,popularity_power=0.75,
        popularity_cutoff='day < anchor-1',id_seen_includes='history, positives and sampled negatives'))
    del opt,model,clone,ck,raw,again
    gc.collect(); torch.mps.empty_cache()
    return summary


class ExcludedCSR:
    def __init__(self,flat,offsets):
        self.flat,self.offsets=flat,offsets

    def __getitem__(self,row):
        return set(map(int,self.flat[self.offsets[row]:self.offsets[row+1]]))


def load_pairs(path):
    with np.load(path,allow_pickle=False) as z:
        pairs={name:z[name] for name in ['h','actions','positive','users','anchors']}
        pairs['excluded']=ExcludedCSR(z['excluded_flat'],z['excluded_offsets'])
    assert len(pairs['h'])==len(pairs['positive'])
    assert set(np.unique(pairs['actions'])) <= {0,1,2,3,4}
    return pairs


def conditional_overfit():
    """Shared candidates and conflicting labels require user-dependent predictions."""
    assert torch.backends.mps.is_available()
    results=[]
    for weighted in [False,True]:
        torch.manual_seed(2026)
        model=(WeightedTower if weighted else TwoTower)(np.zeros((16,0),np.float32),False).to('mps')
        h=torch.zeros((16,50),device='mps',dtype=torch.long)
        h[:,0]=torch.arange(1,17,device='mps');a=torch.zeros_like(h);a[:,0]=2
        candidates=torch.arange(1,17,device='mps').repeat(16,1)
        labels=torch.arange(16,device='mps')
        opt=torch.optim.AdamW(model.parameters(),lr=.003)
        for step in range(1000):
            model.train();opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(h,a,candidates),labels)
            loss.backward();opt.step()
            if loss.item()<.03:break
        model.id_seen[1:]=True;model.eval()
        with torch.no_grad():
            scores=model(h,a,candidates)
            accuracy=float((scores.argmax(1)==labels).float().mean().item())
            final_loss=float(F.cross_entropy(scores,labels).item())
        results.append(dict(weighted=weighted,steps=step+1,accuracy=accuracy,loss=final_loss,
            shared_candidates=True,conflicting_labels=True,device='mps',
            scope='Synthetic conditional-capacity test; not recommendation quality'))
        del opt,model;torch.mps.empty_cache()
    save_json(ROOT/'artifacts/conditional_overfit.json',results)
    assert all(r['accuracy']>=.95 and r['loss']<.1 for r in results),results
    print('CONDITIONAL_OVERFIT',json.dumps(results),flush=True)
    return results


def popularity_controls(data,pairs):
    queries=evaluation_users(data); rows=[]
    for days in [7,28]:
        frame=data.events[(data.events.day>=data.cfg['fit_end']-days)&
            (data.events.day<data.cfg['fit_end'])&(data.events.action==2)&(data.events.i>=0)]
        counts=np.zeros(data.n_items,np.float64); grouped=frame.groupby('i').size()
        counts[grouped.index]=grouped.to_numpy()
        order=np.lexsort((np.arange(data.n_items),-counts))[:200]
        raw=np.full((data.n_users,200),-1,np.int32);raw[queries]=order
        rows.append(record(data,f'popularity_{days}d',raw,0.,0.,{'window_days':days}))
    counts=np.bincount(pairs['positive'],minlength=data.n_items)
    order=np.lexsort((np.arange(data.n_items),-counts))[:200]
    raw=np.full((data.n_users,200),-1,np.int32);raw[queries]=order
    rows.append(record(data,'popularity_training_pairs',raw,0.,0.,{'source':'actual neural pairs; unique user/anchor/item'}))
    return pd.DataFrame(rows)


def personalization(data,name,artifact_root=None,repeats=12):
    root=Path(artifact_root or ROOT/'artifacts')
    full=np.load(root/name/'recommendations.npy',mmap_mode='r')
    q=evaluation_users(data); own=full[q]
    ys=[data.targets[u] for u in q]
    relevant=[{int(i) for i in y if isinstance(i,int)} for y in ys]
    counts=np.array([len(y) for y in relevant]); valid=counts>0
    keys=np.array([j*data.n_items+i for j,y in enumerate(relevant) for i in y],dtype=np.int64)
    base=np.arange(len(q),dtype=np.int64)[:,None]*data.n_items
    discount=1/np.log2(np.arange(200)+2)
    ideal=np.array([discount[:min(n,200)].sum() for n in counts])
    def measure(r):
        hit=np.isin(base+r,keys)&(r>=0)
        return {'recall_at_50':float((hit[valid,:50].sum(1)/counts[valid]).mean()),
                'recall_at_200':float((hit[valid].sum(1)/counts[valid]).mean()),
                'ndcg_at_200':float(((hit[valid]*discount).sum(1)/ideal[valid]).mean())}
    actual=measure(own)
    shuffled=[measure(own[np.random.default_rng(2026+i).permutation(len(q))]) for i in range(repeats)]
    result={'method':name,'own':actual,'shuffled_mean':pd.DataFrame(shuffled).mean().to_dict(),
        'shuffled_runs':shuffled,'unique_recommended_items':int(len(np.unique(own[own>=0]))),
        'queries':len(q),'repeats':repeats,'note':'Permuted saved lists, not encoder histories; one validation window.'}
    save_json(ROOT/'artifacts/personalization'/f'{name}.json',result)
    print('PERSONALIZATION',json.dumps({k:v for k,v in result.items() if k!='shuffled_runs'}),flush=True)
    return result
