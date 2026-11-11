"""E3: paired pooling/SASRec-like gap-aware learning on the full candidate catalog.

Run --encoder pool/sequence --updates 200 for the resource pilot, then resume the
same run to6000. Validation at2000/4000/6000; the pilot does not select on Recall.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

project = Path(__file__).resolve().parent
if __name__ == '__main__':
    os.environ['RECSYS_WORKSPACE'] = str(project/'runs/full_catalog_20261104')
import numpy as np
import pandas as pd
import torch
from recsys_lab.data import PilotData, save_json
from recsys_lab.prefix_sequence import PrefixTower, save_last, load_last
from recsys_lab.baselines import evaluation_users, fill
from recsys_lab.metrics import evaluate
from recsys_lab.scoring_checks import scan, conditional_gap, nominal_mixed_q


class DayProposal:
    def __init__(self, first_day, daily_counts, days):
        self.tables = {}
        counts = np.zeros(len(first_day), np.float64)
        daily = dict(tuple(daily_counts.groupby('day', sort=True)))
        previous = 1202
        for day in sorted(days):
            for past in range(previous, int(day)):
                if past in daily:
                    f = daily[past]; counts[f.i.to_numpy()] += f.clicks.to_numpy()
            previous = int(day)
            ids = np.flatnonzero(first_day < day).astype(np.int32)
            pop = counts[ids]**.75
            q = .5/len(ids) + .5*pop/pop.sum()
            cdf = np.cumsum(q); cdf[-1] = 1.
            self.tables[int(day)] = (ids, q.astype(np.float32), cdf)

    def draw(self, days, rng, k):
        neg = np.empty((len(days), k), np.int32)
        probabilities = np.empty((len(days), k), np.float32)
        for day in np.unique(days):
            which = days == day; ids, q, cdf = self.tables[int(day)]
            loc = np.searchsorted(cdf, rng.random((int(which.sum()), k)))
            neg[which], probabilities[which] = ids[loc], q[loc]
        return neg, probabilities


def select_tail(history, actions, rows, length):
    h = np.asarray(history[rows]); a = np.asarray(actions[rows])
    counts = (h > 0).sum(1)
    offsets = np.maximum(counts-length, 0)
    indices = offsets[:, None] + np.arange(length)[None, :]
    indices = np.minimum(indices, h.shape[1]-1)
    valid = np.arange(length)[None, :] < np.minimum(counts, length)[:, None]
    return (np.take_along_axis(h, indices, 1)*valid,
            np.take_along_axis(a, indices, 1)*valid)


def tiny_check(data, manifest, keep, h, actions, encoder, length, destination):
    rng = np.random.default_rng(2026)
    rows = []
    users = set()
    for row in rng.permutation(keep):
        user = int(manifest.u.iloc[row])
        if user not in users:
            rows.append(row); users.add(user)
        if len(rows) == 16:
            break
    rows = np.asarray(rows)
    histories, acts = select_tail(h, actions, rows, length)
    positive = manifest.positive.to_numpy()[rows]
    pool = np.flatnonzero(data.first_day < int(manifest.cutoff_day.iloc[rows].min()))
    negative = rng.choice(pool, 128, replace=False)
    original = np.unique(np.r_[histories[histories > 0]-1, positive, negative])
    inverse = np.full(data.n_items, -1, np.int32); inverse[original] = np.arange(len(original))
    histories = np.where(histories > 0, inverse[np.maximum(histories-1, 0)]+1, 0)
    candidate = np.unique(np.r_[positive, negative])
    target = np.searchsorted(candidate, positive)
    candidate = inverse[candidate]+1
    torch.manual_seed(17); torch.mps.manual_seed(17)
    model = PrefixTower(data.vectors[original], encoder, length).to('mps')
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-5)
    ht = torch.as_tensor(histories, device='mps', dtype=torch.long)
    at = torch.as_tensor(acts, device='mps', dtype=torch.long)
    ct = torch.as_tensor(candidate, device='mps', dtype=torch.long)
    yt = torch.as_tensor(target, device='mps', dtype=torch.long)
    losses = []
    for _ in range(150):
        model.train(); opt.zero_grad(set_to_none=True)
        logits = model.user(ht,at) @ model.item(ct).T / .1
        loss = torch.nn.functional.cross_entropy(logits,yt)
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        losses.append(float(loss.detach().cpu()))
    model.id_seen.fill_(True); model.eval()
    with torch.inference_mode():
        correct = ((model.user(ht,at)@model.item(ct).T).argmax(1) == yt).float().mean().item()
    result = dict(encoder=encoder, examples=16, candidate_items=len(candidate), steps=150,
                  initial_loss=losses[0], final_loss=losses[-1], accuracy=correct,
                  passed=correct>=.875 and losses[-1]<.5*losses[0], device='mps')
    save_json(destination, result)
    print('PREFIX_TINY', result, flush=True)
    assert result['passed'], 'Do not train full model until tiny overfit succeeds'
    del model,opt; gc.collect(); torch.mps.empty_cache()


def main(args):
    assert torch.backends.mps.is_available(), 'MPS required; CPU fallback forbidden'
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    torch.set_num_threads(4)
    base = project/'runs/prefix_sequence_20261104'
    source = base/'data'
    manifest_info = json.loads((source/'complete.json').read_text())
    manifest = pd.read_parquet(source/'prefix_manifest.parquet')
    h = np.load(source/'history100.npy', mmap_mode='r')
    actions = np.load(source/'actions100.npy', mmap_mode='r')
    keep = np.load(source/'kept_rows.npy')
    dest = base/args.encoder; dest.mkdir(exist_ok=True)
    assert args.resume or not (dest/'configuration.json').exists(), 'Use --resume or a fresh directory'
    data = PilotData()
    data.cfg['history_length'] = args.length
    data.h,data.actions,_ = data.history(1293,strong_only=True)
    first_day = np.load(source/'first_day.npy')
    np.testing.assert_array_equal(first_day,data.first_day)
    if not args.resume:
        tiny_check(data,manifest,keep,h,actions,args.encoder,args.length,dest/'tiny_overfit.json')
    days = manifest.cutoff_day.to_numpy()
    proposal = DayProposal(first_day,pd.read_parquet(source/'proposal_daily_clicks.parquet'),np.unique(days[keep]))
    q = nominal_mixed_q(data)
    bias = (.1*(np.log(q)-np.log(q).mean())).astype(np.float32)
    ev = data.events[(data.events.day>=1264)&(data.events.day<1292)&(data.events.action==2)&(data.events.i>=0)]
    prior = np.lexsort((np.arange(data.n_items),-np.bincount(ev.i,minlength=data.n_items)))
    data.events = None; del ev; gc.collect()
    known = np.load(source/'known_day_positives.npz')
    known_keys,starts,sizes,known_items = (known[k] for k in ['keys','starts','sizes','items'])
    keys = manifest.u.to_numpy(dtype=np.int64)*2000+manifest.day.to_numpy()
    group = np.searchsorted(known_keys,keys)
    assert np.array_equal(known_keys[group],keys)
    positives = manifest.positive.to_numpy()
    torch.manual_seed(2026); torch.mps.manual_seed(2026)
    model = PrefixTower(data.vectors,args.encoder,args.length).to('mps')
    shared_hash = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(('ids.','item_mlp.','user_mlp.')):
            shared_hash.update(value.detach().cpu().numpy().tobytes())
    config = dict(encoder=args.encoder, length=args.length, layers=1 if args.encoder=='sequence' else 0,
        heads=2 if args.encoder=='sequence' else 0, dimension=64, feedforward=128,
        action_representation='Same fixed action-weighted item tokens; no new metadata',
        history_days=60,gap_hours=24,negative_draws=256,temperature=.1,lr=.001,
        batch_size=128,planned_updates=6000,validation_steps=[2000,4000,6000],seed=2026,
        item_dropout=0,attention_dropout=.1 if args.encoder=='sequence' else 0,
        train_prefixes=len(keep), training_users=int(manifest.loc[keep,'u'].nunique()),
        positive_items=int(manifest.loc[keep,'positive'].nunique()),catalog=data.n_items,
        input_sha256=manifest_info['inputs_sha256'],shared_init_sha256=shared_hash.hexdigest(),
        fingerprint=data.fingerprint,proposal='IID .5uniform+.5pastclick^.75, complete days before prefix cutoff',
        negative_mask='Other known eligible click-items of same user/target day; duplicates retained; no resampling',
        evaluation='Same weekly queries1293-1299 with gap1292 and full historical catalog',
        final_test_used=False)
    if args.encoder == 'sequence':
        config['sequence_token_scale'] = 64**.5
    if args.resume:
        assert json.loads((dest/'configuration.json').read_text()) == config, 'Resume configuration mismatch'
    else:
        save_json(dest/'configuration.json',config)
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    rng = np.random.default_rng(12026)
    progress = dict(updates=0,epoch=0,cursor=0,fit_seconds=0.,best_recall=-1.,curves=[],loss_log=[])
    if args.resume:
        progress = load_last(dest/'last.pt',model,opt,rng)
        print('PREFIX_RESUMED',args.encoder,progress['updates'],flush=True)
    users = evaluation_users(data); targets = [data.targets[u] for u in users]
    if not (dest/'training_target_popularity_metrics.json').exists():
        freq = np.bincount(positives[keep],minlength=data.n_items)
        ranks = np.lexsort((np.arange(data.n_items),-freq))[:200]
        m,pu = evaluate(np.tile(ranks,(len(users),1)),targets,data.n_items)
        pu.u = users;pu.to_parquet(dest/'training_target_popularity_per_user.parquet',index=False)
        save_json(dest/'training_target_popularity_metrics.json',m)

    def predict(offset=None):
        model.eval()
        with torch.inference_mode():
            items = torch.cat([model.item(torch.arange(lo+1,min(lo+4096,data.n_items)+1,device='mps'))
                               for lo in range(0,data.n_items,4096)])
            available = model.id_seen[1:].cpu().numpy() | data.vector_ok
            raw = scan(data,model,items,available,offset)
            prediction = fill(raw,prior,200)
        del items
        return raw,prediction

    def validate():
        raw,pred = predict(); metrics,_ = evaluate(pred,targets,data.n_items)
        row = dict(updates=progress['updates'],recall=metrics['recall_reachable@200'],
                   fit_seconds=progress['fit_seconds'])
        progress['curves'].append(row)
        pd.DataFrame(progress['curves']).to_csv(dest/'learning_curve.csv',index=False)
        if row['recall'] > progress['best_recall']:
            progress['best_recall'] = row['recall']
            torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            best_updates=progress['updates'],configuration=config),dest/'checkpoint.pt')
            np.save(dest/'best_predictions.npy',pred)
        print('PREFIX_VALIDATION',args.encoder,row,flush=True)
        del raw,pred;gc.collect();torch.mps.empty_cache()

    order = np.random.default_rng(2026+progress['epoch']).permutation(len(keep))
    window_losses = []
    while progress['updates'] < args.updates:
        if progress['cursor'] >= len(keep):
            progress['epoch'] += 1; progress['cursor'] = 0
            order = np.random.default_rng(2026+progress['epoch']).permutation(len(keep))
        cursor = progress['cursor']; rows = keep[order[cursor:cursor+128]]
        started = time.perf_counter()
        neg,probabilities = proposal.draw(days[rows],rng,256)
        masked = np.empty(neg.shape,bool)
        for j,row in enumerate(rows):
            g = group[row]; forbidden = known_items[starts[g]:starts[g]+sizes[g]]
            masked[j] = np.isin(neg[j],forbidden)
        candidates = np.column_stack([positives[rows],neg])+1
        histories,acts = select_tail(h,actions,rows,args.length)
        ht = torch.as_tensor(histories,device='mps',dtype=torch.long)
        at = torch.as_tensor(acts,device='mps',dtype=torch.long)
        ct = torch.as_tensor(candidates,device='mps',dtype=torch.long)
        model.train();opt.zero_grad(set_to_none=True)
        scores = (model.user(ht,at)[:,None,:]*model.item(ct)).sum(-1)/.1
        corrected = scores[:,1:] - torch.log(256*torch.as_tensor(probabilities,device='mps'))
        corrected = corrected.masked_fill(torch.as_tensor(masked,device='mps'),-torch.inf)
        loss = (torch.logsumexp(torch.cat([scores[:,:1],corrected],1),1)-scores[:,0]).mean()
        assert torch.isfinite(loss), 'Nonfinite loss'
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        with torch.no_grad():
            model.id_seen[ct.flatten()] = True;model.id_seen[ht.flatten()] = True;model.id_seen[0] = False
        torch.mps.synchronize()
        progress['fit_seconds'] += time.perf_counter()-started
        progress['updates'] += 1;progress['cursor'] += len(rows)
        window_losses.append(float(loss.detach().cpu()))
        if progress['updates']%200 == 0:
            row = dict(updates=progress['updates'],loss=float(np.mean(window_losses)),
                       fit_seconds=progress['fit_seconds'],
                       batch_sha256=hashlib.sha256(rows.tobytes()+candidates.tobytes()+masked.tobytes()).hexdigest(),
                       mps_allocated_gib=torch.mps.current_allocated_memory()/2**30,
                       mps_driver_gib=torch.mps.driver_allocated_memory()/2**30)
            progress['loss_log'].append(row);window_losses=[]
            pd.DataFrame(progress['loss_log']).to_csv(dest/'loss_log.csv',index=False)
            print('PREFIX_PROGRESS',args.encoder,row,flush=True)
        if progress['updates'] in config['validation_steps']:
            validate()
        if progress['updates']%1000 == 0 or progress['updates'] == args.updates:
            save_last(dest/'last.pt',model,opt,rng,progress)
    resource_name = 'resource_pilot.json' if args.updates<=200 else 'training_resources.json'
    save_json(dest/resource_name,dict(updates=progress['updates'],fit_seconds=progress['fit_seconds'],
        seconds_per_step=progress['fit_seconds']/progress['updates'],
        observed_mps_driver_gib=torch.mps.driver_allocated_memory()/2**30,
        observation_is_not_continuous_peak=True,final_test_used=False))
    if args.updates < 6000:
        print('PREFIX_PILOT_SAVED',args.encoder,progress['updates'],flush=True)
        return
    ck = torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ck['state'])
    raw,pred = predict();np.testing.assert_array_equal(pred,np.load(dest/'best_predictions.npy'))
    for name, (r, prediction) in [('raw',(raw,pred)),('same_fixed_prior',predict(bias))]:
        metrics,pu = evaluate(prediction,targets,data.n_items);pu.u=users
        diagnostic = conditional_gap(prediction,targets,data.n_items)
        metrics.update(encoder=args.encoder,best_updates=ck['best_updates'],fit_seconds=progress['fit_seconds'],
                       own_minus_shuffled=diagnostic['gap'],checkpoint_reload_identical=True)
        save_json(dest/f'{name}_metrics.json',metrics)
        pu.to_parquet(dest/f'{name}_per_user.parquet',index=False)
        np.savez_compressed(dest/f'{name}_lists.npz',query_users=users,raw=r,recommendations=prediction)
        print('PREFIX_RESULT',args.encoder,name,metrics['recall_reachable@200'],diagnostic['gap'],flush=True)
    save_json(dest/'complete.json',dict(completed=True,updates=progress['updates'],final_test_used=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder',choices=['pool','sequence'],required=True)
    parser.add_argument('--length',type=int,choices=[50,100],default=50)
    parser.add_argument('--updates',type=int,default=200)
    parser.add_argument('--resume',action='store_true')
    main(parser.parse_args())
