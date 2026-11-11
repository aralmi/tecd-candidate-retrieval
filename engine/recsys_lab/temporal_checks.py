"""Frozen simple algorithms on earlier windows; no re-tuning or final test."""
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd

from .baselines import fill
from .data import PilotData, STRENGTH, save_json
from .fusion import combine
from .metrics import evaluate


def prepare(project, destination):
    project,destination=Path(project),Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    users=np.load(project/'runs/active_history/data/processed/users.npy')
    c=duckdb.connect();c.execute("SET memory_limit='1500MB'");c.execute('SET threads=3')
    c.register('selected',pd.DataFrame({'user_id':users,'u':np.arange(len(users),dtype=np.int32)}))
    c.read_parquet(str(project/'data/raw/items.pq')).create_view('catalog')
    c.execute('CREATE TABLE items AS SELECT item_id,(row_number() OVER(ORDER BY item_id)-1)::INTEGER AS i FROM catalog')
    files=[str(project/f'data/raw/events/{d:05}.pq') for d in range(1188,1300)]
    assert all(Path(f).exists() for f in files)
    c.read_parquet(files).create_view('raw')
    path=destination/'numeric_events.parquet'
    c.execute('''COPY(SELECT DISTINCT s.u,r.timestamp AS ts,
        (r.timestamp//86400000000)::SMALLINT AS day,coalesce(i.i,-1)::INTEGER AS i,
        CASE action_type WHEN 'view' THEN 1 WHEN 'click' THEN 2 WHEN 'like' THEN 3 WHEN 'clickout' THEN 4 END::TINYINT AS action,
        CASE WHEN i.i IS NULL THEN r.item_id ELSE NULL END AS outside_id
        FROM raw r JOIN selected s USING(user_id) LEFT JOIN items i USING(item_id)
        ORDER BY s.u,ts,i,action) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)''',[str(path)])
    namespace=c.execute('SELECT count(*) FROM items').fetchone()[0]
    c.close()
    save_json(destination/'data_manifest.json',dict(event_start=1188,event_end=1299,
        users=len(users),namespace_items=namespace,candidate_rule='observed in that window-specific past90-day prefix; catalog row exists',
        population_rule='Original fixed users selected using1202-1261, earlier than every query anchor',
        final_test_used=False))
    return pd.read_parquet(path),namespace,users


def check_window(events,namespace,users,anchor,destination,project):
    start=time.perf_counter();destination=Path(destination)/f'anchor_{anchor}';destination.mkdir(exist_ok=True)
    end=anchor-1
    past=events[(events.day>=end-90)&(events.day<end)&(events.i>=0)]
    allowed=np.zeros(namespace,bool);allowed[past.i.unique()]=True
    future=events[(events.day>=anchor)&(events.day<anchor+7)&(events.action==2)]
    targets=[set() for _ in users]
    for row in future.itertuples():
        item=int(row.i)
        targets[row.u].add(item if item>=0 and allowed[item] else 'outside:'+str(item if item>=0 else row.outside_id))
    queries=np.array([u for u,y in enumerate(targets) if y],np.int32)
    y=[targets[u] for u in queries]
    d=PilotData.__new__(PilotData);d.cfg={'history_days':60,'history_length':50};d.events=events;d.n_users=len(users)
    h,_,_=d.history(anchor)
    def popular(days):
        f=past[(past.day>=end-days)&(past.action==2)]
        counts=np.zeros(namespace,np.int64);s=f.groupby('i').size();counts[s.index]=s.to_numpy()
        counts[~allowed]=-1
        order=np.lexsort((np.arange(namespace),-counts))[:200].astype(np.int32)
        assert allowed[order].all()
        return order
    p90,p28=popular(90),popular(28)
    recent=np.full((len(queries),200),-1,np.int32)
    for j,u in enumerate(queries):
        ids=list(dict.fromkeys((h[u][h[u]>0]-1)[::-1]))
        recent[j,:len(ids)]=ids
    personal=np.full_like(recent,-1)
    strong=past[(past.day>=end-60)&(past.action>1)].copy()
    strong['weight']=STRENGTH[strong.action.to_numpy()]
    daily=strong.groupby(['u','i','day'],sort=False).agg(weight=('weight','max'),last=('ts','max'))
    profile=daily.groupby(level=[0,1]).agg(weight=('weight','sum'),last=('last','max')).reset_index()
    qmap={int(u):j for j,u in enumerate(queries)}
    for u,g in profile.groupby('u',sort=False):
        if u not in qmap:continue
        order=np.lexsort((g.i.to_numpy(),-g['last'].to_numpy(),-g.weight.to_numpy()))[:200]
        items=g.i.to_numpy()[order];personal[qmap[u],:len(items)]=items
    a=fill(personal,p28,200)
    methods={'pop90':np.tile(p90,(len(queries),1)), 'pop28':np.tile(p28,(len(queries),1)),
        'recent_pop90':fill(recent,p90,200),'recent_pop28':fill(recent,p28,200),
        'personal_pop90':fill(personal,p90,200),'personal_pop28':a,
        'frozen_fusion_b25':combine(a,np.tile(p90,(len(queries),1)),175)}
    fingerprint=hashlib.sha256(users.tobytes()+np.flatnonzero(allowed).tobytes()).hexdigest()
    rows=[];per={}
    for name,pred in methods.items():
        assert allowed[pred].all()
        metrics,pu=evaluate(pred,y,namespace);pu.u=queries
        metrics.update(method=name,anchor=anchor,cohort_users=len(users),evaluated_queries=len(queries),
            candidate_items=int(allowed.sum()),namespace_items=namespace,fingerprint=fingerprint,
            catalog_coverage_in_allowed_catalog=len(np.unique(pred))/allowed.sum(),
            quota_b=25 if name=='frozen_fusion_b25' else None,
            status='Frozen algorithms; retrospective temporal robustness, not blind future confirmation')
        save_json(destination/f'{name}.json',metrics)
        pu.to_parquet(destination/f'{name}_per_user.parquet',index=False)
        np.savez_compressed(destination/f'{name}_lists.npz',query_users=queries,recommendations=pred)
        rows.append(metrics);per[name]=pu
        print('TEMPORAL',anchor,name,metrics['recall_reachable@200'],flush=True)
    comparisons=[]
    for baseline in ['personal_pop28','recent_pop90']:
        delta=(per['frozen_fusion_b25']['recall_reachable@200']-per[baseline]['recall_reachable@200']).dropna().to_numpy()
        rng=np.random.default_rng(2026);means=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)]
        comparisons.append(dict(baseline=baseline,mean_difference=float(delta.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),users=len(delta)))
    save_json(destination/'comparisons.json',comparisons)
    if anchor==1293:
        references={'pop90':'runs/active_history/artifacts/popularity','pop28':'runs/followup_20261103/artifacts/popularity_28d',
            'recent_pop90':'runs/active_history/artifacts/recent', 'recent_pop28':'runs/followup_20261103/artifacts/recent_popularity_28d',
            'personal_pop90':'runs/followup_20261103/artifacts/personal_popularity',
            'personal_pop28':'runs/followup_20261103/artifacts/personal_frequency_popularity_28d',
            'frozen_fusion_b25':'runs/followup_20261103/artifacts/followup_fusion_b25'}
        for name,rel in references.items():
            ref=json.loads((Path(project)/rel/'metrics.json').read_text())
            actual=next(m for m in rows if m['method']==name)
            for key in ['recall_reachable@50','recall_reachable@200','recall_all@200','ndcg@200']:
                assert np.isclose(actual[key],ref[key],atol=1e-12,rtol=0),(name,key,actual[key],ref[key])
        save_json(destination/'original_reproduction.json',dict(passed=True,methods=len(references),candidate_items=int(allowed.sum())))
    save_json(destination/'window_manifest.json',dict(anchor=anchor,past_start=end-90,past_end_exclusive=end,
        gap_day=end,target_end=anchor+6,candidate_items=int(allowed.sum()),query_users=len(queries),
        seconds=time.perf_counter()-start,retuned_parameters=False,future_prefix_events_used=False,
        note='Parameters chosen on1293-1299 then frozen for earlier windows. Retrospective check, not pristine holdout.'))
    return rows
