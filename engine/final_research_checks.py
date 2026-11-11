"""Registered closing checks. Final recipes are saved before reading final goals."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import duckdb
from scipy import sparse
from implicit.cpu.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits
from recsys_lab.data import save_json
from recsys_lab.baselines import fill,topk
from recsys_lab.metrics import evaluate
from recsys_lab.fusion import combine
from recsys_lab.temporal_als import training_confidence,full_query_confidence
from recsys_lab.download import download_one,REVISION
from check_repeat_cold_channels import load_window
from summarize_large_mixed import paired_ci

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/closing_checks_20261107'


def saved_rankings(query,catalog):
    sources={
        'bm25_strong300':ROOT/'runs/followup_20261103/artifacts/bm25_strong_k300/raw.npy',
        'provided_content':ROOT/'runs/active_history/artifacts/content/raw.npy',
        'als_full60':ROOT/'runs/profile_audit_20261104/query_full60_lists.npz',
        'tower8_raw':ROOT/'runs/full_catalog_20261104/weeks8/raw_lists.npz',
        'prefix_pool':ROOT/'runs/prefix_sequence_20261104/pool/raw_lists.npz',
        'sasrec_like':ROOT/'runs/prefix_sequence_20261104/sequence/raw_lists.npz',
    }
    with np.load(ROOT/'runs/investigation_20261104/temporal/anchor_1293/pop28_lists.npz') as z:
        pop=np.searchsorted(catalog,z['recommendations'][0]);np.testing.assert_array_equal(catalog[pop],z['recommendations'][0])
    rankings={}
    for name,path in sources.items():
        if path.suffix=='.npy':raw=np.load(path,mmap_mode='r')[query]
        else:
            with np.load(path) as z:
                np.testing.assert_array_equal(z['query_users'],query);raw=z['raw']
        assert raw.shape==(len(query),200) and ((raw>=-1)&(raw<len(catalog))).all()
        rankings[name]=fill(raw,pop,200)
    save_json(OUT/'development_sources.json',{k:str(v) for k,v in sources.items()})
    return rankings


def development():
    query,catalog,base,ys,*_=load_window(1293)
    rankings=saved_rankings(query,catalog);rankings['personal_pop28']=base
    rows=[]
    for name,pred in rankings.items():
        m,pu=evaluate(pred,ys,len(catalog));pu.u=query;m.update(method=name,final_test_used=False)
        rows.append(m);pu.to_parquet(OUT/f'development_{name}_per_user.parquet',index=False)
    pd.DataFrame(rows).to_csv(OUT/'development_common_pop28.csv',index=False)
    save_json(OUT/'horizon_protocol.json',dict(anchor=1293,horizons=[1,3,7],methods=['personal_pop28','prefix_pool','sasrec_like'],
        cohorts=['Users with clicks within that horizon','Common first-day clicking users across all horizons'],
        checkpoint_selection='Previously selected using weekly development Recall; exploratory shorter-horizon evaluation',
        recommendations='Same frozen rankings and common pop28 fallback',final_test_used=False))
    c=duckdb.connect();c.read_parquet(str(ROOT/'runs/investigation_20261104/temporal/numeric_events.parquet')).create_view('events')
    events=c.execute('SELECT u,i,outside_id,day FROM events WHERE day>=1293 AND day<1300 AND action=2').df();c.close()
    first=set(events.loc[events.day==1293,'u']);lookup={int(u):j for j,u in enumerate(query)}
    horizons=[]
    for days in [1,3,7]:
        targets={}
        for row in events[events.day<1293+days].itertuples():
            pos=int(np.searchsorted(catalog,row.i))
            item=pos if pos<len(catalog) and catalog[pos]==row.i else 'outside:'+str(row.i if row.i>=0 else row.outside_id)
            targets.setdefault(int(row.u),set()).add(item)
        if days==7:
            assert all(targets[int(u)]==y for u,y in zip(query,ys))
        for cohort,users in [('horizon',sorted(targets)),('first_day_matched',sorted(first))]:
            positions=[lookup[u] for u in users];goals=[targets[u] for u in users]
            for method in ['personal_pop28','prefix_pool','sasrec_like']:
                m,pu=evaluate(rankings[method][positions],goals,len(catalog));pu.u=users
                m.update(method=method,days=days,cohort=cohort,final_test_used=False);horizons.append(m)
                pu.to_parquet(OUT/f'horizon{days}_{cohort}_{method}.parquet',index=False)
    pd.DataFrame(horizons).to_csv(OUT/'horizon_comparison.csv',index=False)
    print('HORIZON_CHECK',[(r['days'],r['cohort'],r['method'],round(r['recall_reachable@200'],5)) for r in horizons],flush=True)


def final():
    frozen=dict(anchor=1301,gap_day=1300,goal_days=[1301,1307],training_days=[1210,1299],history_days=60,
        cohort='Original80018 users selected on1202..1261',catalog='Observed in original cohort past90; catalog row exists',
        baseline='Strong personal frequency dailymax60 then pop28',challenger='Same baseline plus raw ALS using existing weighted roundrobin quota10',
        als=dict(factors=32,regularization=.1,iterations=10,alpha=1.,seed=2026),
        confidence='1+20log1p(sum of daily maximum action strength)',query='Exact full60 all-action collapse consecutive repeats',
        goal='Unique weekly click items, repeats allowed',metrics=['macro reachable Recall200','macro all Recall200','NDCG200','paired user bootstrap'],
        decision='Report challenger minus baseline with95%paired user interval; retain all outcomes, no final tuning',revision=REVISION)
    freeze=OUT/'final_protocol_frozen.json'
    if freeze.exists():assert json.loads(freeze.read_text())==frozen
    else:save_json(freeze,frozen)
    c=duckdb.connect();c.read_parquet(str(ROOT/'runs/investigation_20261104/temporal/numeric_events.parquet')).create_view('events')
    past=c.execute('SELECT u,ts,day,i,action FROM events WHERE day>=1210 AND day<1300 AND i>=0 ORDER BY u,ts,i,action').df()
    catalog=np.sort(past.i.unique());past['i']=np.searchsorted(catalog,past.i.to_numpy()).astype(np.int32)
    nusers=80018;confidence=training_confidence(past,nusers,len(catalog));profile=full_query_confidence(past,1301,nusers,len(catalog))
    assert past.day.min()>=1210 and past.day.max()<1300
    model=AlternatingLeastSquares(factors=32,regularization=.1,iterations=10,alpha=1.,num_threads=4,random_state=2026)
    start=time.perf_counter()
    with threadpool_limits(1,user_api='blas'):model.fit(confidence,show_progress=False)
    fit_seconds=time.perf_counter()-start;model.save(str(OUT/'final_als_model.npz'));np.save(OUT/'final_local_to_global.npy',catalog)
    print('FINAL_FIT_COMPLETE',len(catalog),confidence.nnz,round(fit_seconds,2),flush=True)
    audit=json.loads((ROOT.parent/'Планирование/Проверка_данных_18_10_2026.json').read_text())
    entries=[e for e in audit['file_manifest'] if '/events/' in e.get('path','') and 1301<=int(Path(e['path']).stem)<=1307]
    assert len(entries)==7
    with ThreadPoolExecutor(max_workers=3) as pool:downloads=list(pool.map(download_one,entries))
    save_json(OUT/'final_downloads.json',downloads)
    userids=np.load(ROOT/'runs/active_history/data/processed/users.npy')
    c.register('users',pd.DataFrame({'user_id':userids,'u':np.arange(nusers,dtype=np.int32)}))
    c.read_parquet(str(ROOT/'data/raw/items.pq')).create_view('catalog_rows')
    c.execute('CREATE TABLE global_items AS SELECT item_id,row_number() OVER(ORDER BY item_id)-1 AS i FROM catalog_rows')
    files=[str(ROOT/f'data/raw/events/{d:05}.pq') for d in range(1301,1308)]
    c.read_parquet(files).create_view('future')
    daily=c.execute('SELECT timestamp//86400000000 AS day,count(*) AS events FROM future GROUP BY day ORDER BY day').df();assert daily.day.tolist()==list(range(1301,1308));daily.to_csv(OUT/'final_days.csv',index=False)
    target=c.execute("SELECT DISTINCT u,coalesce(i,-1) AS i,f.item_id FROM future f JOIN users USING(user_id) LEFT JOIN global_items USING(item_id) WHERE action_type='click'").df();c.close()
    q=np.sort(target.u.unique());lookup={int(u):j for j,u in enumerate(q)};ys=[set() for _ in q]
    for row in target.itertuples():
        pos=int(np.searchsorted(catalog,row.i));item=pos if pos<len(catalog) and catalog[pos]==row.i else 'outside:'+str(row.i if row.i>=0 else row.item_id)
        ys[lookup[int(row.u)]].add(item)
    p=past[(past.day>=1272)&(past.action==2)].groupby('i').size();counts=np.zeros(len(catalog),np.int64);counts[p.index]=p
    pop=np.lexsort((np.arange(len(catalog)),-counts))[:200]
    strong=past[(past.day>=1240)&(past.action>1)].copy();strong['weight']=np.array([0,.1,1,2,3],np.float32)[strong.action.to_numpy()]
    daily=strong.groupby(['u','i','day']).agg(weight=('weight','max'),last=('ts','max'));personal=daily.groupby(level=[0,1]).agg(weight=('weight','sum'),last=('last','max')).reset_index()
    raw=np.full((len(q),200),-1,np.int32)
    for u,g in personal.groupby('u',sort=False):
        if int(u) in lookup:
            order=np.lexsort((g.i.to_numpy(),-g['last'].to_numpy(),-g.weight.to_numpy()))[:200];raw[lookup[int(u)],:len(order)]=g.i.to_numpy()[order]
    base=fill(raw,pop,200);als=np.full_like(base,-1);nonempty=np.diff(profile.indptr)[q]>0
    start=time.perf_counter()
    with threadpool_limits(4,user_api='blas'):
        factors=model.recalculate_user(q,profile[q])
        for lo in range(0,len(q),64):
            scores=factors[lo:lo+64]@model.item_factors.T
            for j,row in enumerate(scores):
                if nonempty[lo+j]:als[lo+j]=topk(row,200)
    predict_seconds=time.perf_counter()-start
    rows=[];per={}
    for name,pred in [('personal_pop28',base),('personal_als_b10',combine(base,als,190))]:
        m,pu=evaluate(pred,ys,len(catalog));pu.u=q;per[name]=pu
        m.update(method=name,fit_seconds=fit_seconds if name.endswith('b10') else 0.,als_predict_seconds=predict_seconds,
            anchor=1301,final_test_used=True);rows.append(m)
        pu.to_parquet(OUT/f'final_{name}_per_user.parquet',index=False);np.savez_compressed(OUT/f'final_{name}_lists.npz',query_users=q,recommendations=pred)
    pd.DataFrame(rows).to_csv(OUT/'final_comparison.csv',index=False)
    save_json(OUT/'final_paired_difference.json',paired_ci(per['personal_als_b10'],per['personal_pop28']))
    save_json(OUT/'complete.json',dict(completed=True,final_test_used=True,final_recipes_retuned=False,source_revision=REVISION))
    print('FINAL_RESULT',[(m['method'],m['recall_reachable@200'],m['recall_all@200']) for m in rows],flush=True)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    development()
    if not (OUT/'complete.json').exists():final()
