"""Fresh five-family development comparison, using the recorded implementations."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import runpy
import shutil
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
os.environ['RECSYS_WORKSPACE']=str(ROOT/'runs/full_catalog_20261104')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import pandas as pd
import torch
from recsys_lab.data import PilotData,STRENGTH,save_json
from recsys_lab.baselines import run_baseline,evaluation_users,fill,topk
from recsys_lab.metrics import evaluate
from recsys_lab.temporal_als import full_query_confidence
from recsys_lab.tiled_linear import patch_tower_linears

BENCH=ROOT/'bench_results';BENCH.mkdir(exist_ok=True)
CONFIG=json.loads((ROOT/'benchmark_config.json').read_text())
FULL=ROOT/'runs/full_catalog_20261104';ACTIVE=ROOT/'runs/active_history'


def link_inputs(source,dest):
    source=Path(source);dest=Path(dest)
    if not source.exists():raise FileNotFoundError(source)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.symlink_to(source.resolve(),target_is_directory=source.is_dir())


def common_data():
    data=PilotData();q=evaluation_users(data)
    if data.cfg['validation_anchor']!=1293:raise ValueError('This benchmark is fixed to development1293')
    assert data.events.day.max()<=1299
    assert data.n_items==1060869 and len(q)==4335,(data.n_items,len(q))
    return data


def prepare():
    cache_value=os.environ.get('RECSYS_PREPARED_STUDY','')
    cache=Path(cache_value) if cache_value else None
    for name in ['configs','artifacts','data/processed']:(FULL/name).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ACTIVE/'configs/pilot.json',FULL/'configs/pilot.json')
    link_inputs(ROOT/'data/raw',FULL/'data/raw')
    if cache and (cache/'runs/active_history/data/processed/items.parquet').exists():
        cfg=json.loads((cache/'runs/active_history/configs/pilot.json').read_text())
        assert cfg==json.loads((FULL/'configs/pilot.json').read_text()),'Prepared cohort configuration differs'
        for name in ['events.parquet','items.parquet','users.npy']:
            link_inputs(cache/'runs/active_history/data/processed'/name,FULL/'data/processed'/name)
            link_inputs(cache/'runs/active_history/data/processed'/name,ACTIVE/'data/processed'/name)
        for rel in ['runs/large_mixed_20261105/data','runs/user_week_balance_20261104/data',
                    'runs/prefix_sequence_20261104/data','runs/windows_20261103/events.parquet']:
            if (cache/rel).exists():link_inputs(cache/rel,ROOT/rel)
        print('Reused prepared INPUTS only; no model checkpoint or recommendation is loaded.',flush=True)
    else:
        from recsys_lab.data import select_events,select_items
        # The original selector reads the shared past-only activity table.
        runpy.run_path(str(ROOT/'audit_population.py'),run_name='__main__')
        select_events();select_items()
        for name in ['events.parquet','items.parquet','users.npy']:
            link_inputs(FULL/'data/processed'/name,ACTIVE/'data/processed'/name)
    data=common_data()
    save_json(ACTIVE/'artifacts/dataset_summary.json',dict(users=data.n_users,items=data.n_items,fingerprint=data.fingerprint))
    if not (ROOT/'runs/windows_20261103/events.parquet').exists():
        import duckdb
        dest=ROOT/'runs/windows_20261103/events.parquet';dest.parent.mkdir(parents=True,exist_ok=True)
        c=duckdb.connect();c.execute("SET memory_limit='3GB'");c.execute('SET threads=4')
        c.register('selected',pd.DataFrame({'user_id':data.users}))
        c.read_parquet(str(FULL/'data/processed/items.parquet')).create_view('catalog')
        c.execute('CREATE TABLE items AS SELECT item_id,(row_number() OVER(ORDER BY item_id)-1)::INTEGER AS i FROM catalog')
        c.read_parquet([str(ROOT/f'data/raw/events/{d:05}.pq') for d in range(1174,1292)]).create_view('raw')
        c.execute('''COPY(SELECT DISTINCT e.user_id,e.timestamp AS ts,(e.timestamp//86400000000)::SMALLINT AS day,i.i,
            CASE action_type WHEN 'view' THEN 1 WHEN 'click' THEN 2 WHEN 'like' THEN 3 WHEN 'clickout' THEN 4 END::TINYINT AS action
            FROM raw e SEMI JOIN selected USING(user_id) JOIN items i USING(item_id)
            ORDER BY user_id,ts,i,action) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)''',[str(dest)])
        c.close()
    if not (ROOT/'runs/large_mixed_20261105/data/verification.json').exists():prepare_strong_pairs(data)
    if not (ROOT/'runs/prefix_sequence_20261104/data/complete.json').exists():
        from prepare_prefix_sequence import main as build_prefixes
        build_prefixes()
    query=evaluation_users(data);goals=[data.targets[u] for u in query]
    np.save(BENCH/'query_users.npy',query)
    goal_records=[dict(u=int(u),item=int(i) if isinstance(i,int) else None,outside=None if isinstance(i,int) else i)
                  for u,y in zip(query,goals) for i in y]
    pd.DataFrame(goal_records).to_parquet(BENCH/'targets.parquet',index=False)
    cfg=dict(fingerprint=data.fingerprint,anchor=1293,cutoff_exclusive=1292,target_end=1299,
        cohort_users=data.n_users,catalog=data.n_items,query_users=len(query),
        reachable_users=sum(any(isinstance(i,int) for i in y) for y in goals),final_test_used=False,
        preprocessing_cache=bool(cache),model_policy="fresh fits; prepared-input cache only")
    save_json(BENCH/'data_manifest.json',cfg);print('BENCHMARK_DATA',cfg,flush=True)


def prepare_strong_pairs(data):
    """Compact version of the existing strong_all dataset; same eligible positive pairs."""
    import duckdb
    dest=ROOT/'runs/large_mixed_20261105/data';dest.mkdir(parents=True,exist_ok=True)
    proposal_dir=ROOT/'runs/user_week_balance_20261104/data';proposal_dir.mkdir(parents=True,exist_ok=True)
    c=duckdb.connect();c.execute("SET memory_limit='4GB'");c.execute('SET threads=4')
    c.read_parquet([str(ROOT/f'data/raw/events/{d:05}.pq') for d in range(1174,1292)]).create_view('raw')
    c.read_parquet(str(FULL/'data/processed/items.parquet')).create_view('catalog')
    c.execute('CREATE TABLE items AS SELECT item_id,(row_number() OVER(ORDER BY item_id)-1)::INTEGER AS i FROM catalog')
    c.register('first',pd.DataFrame({'i':np.arange(data.n_items),'first_day':data.first_day}))
    anchors=np.array([1235,1242,1249,1256,1263,1270,1277,1284],np.int16)
    c.register('anchors',pd.DataFrame({'anchor':anchors}))
    c.execute('''CREATE TABLE strong AS SELECT DISTINCT r.user_id,r.timestamp AS ts,(r.timestamp//86400000000)::SMALLINT AS day,i.i,
        CASE action_type WHEN 'click' THEN 2 WHEN 'like' THEN 3 WHEN 'clickout' THEN 4 END::TINYINT AS action,f.first_day
        FROM raw r JOIN items i USING(item_id) JOIN first f USING(i) WHERE action_type IN ('click','like','clickout')''')
    c.execute('''CREATE TABLE targets AS SELECT DISTINCT a.anchor,e.user_id,e.i FROM strong e JOIN anchors a
        ON e.day>=a.anchor AND e.day<a.anchor+7 WHERE action=2 AND first_day<a.anchor-1''')
    c.execute('''CREATE TABLE history_users AS SELECT DISTINCT a.anchor,e.user_id FROM strong e JOIN anchors a
        ON e.day>=a.anchor-61 AND e.day<a.anchor-1 SEMI JOIN targets t ON e.user_id=t.user_id AND a.anchor=t.anchor''')
    target=c.execute('SELECT t.* FROM targets t SEMI JOIN history_users USING(anchor,user_id)').df()
    eligible=np.sort(target.user_id.unique());extra=eligible[~np.isin(eligible,data.users)]
    users=np.r_[data.users,extra];target['u']=pd.Index(users).get_indexer(target.user_id).astype(np.int32)
    target=target.sort_values(['anchor','u','i']).reset_index(drop=True)
    groups=target.groupby(['anchor','u'],sort=True).size().rename('targets').reset_index()
    keys=groups.u.to_numpy(dtype=np.int64)*2048+groups.anchor.to_numpy()
    gi=pd.Index(keys).get_indexer(target.u.to_numpy(dtype=np.int64)*2048+target.anchor.to_numpy()).astype(np.int32)
    h=np.lib.format.open_memmap(dest/'strong_h.npy',mode='w+',dtype=np.int32,shape=(len(groups),100))
    a=np.lib.format.open_memmap(dest/'strong_actions.npy',mode='w+',dtype=np.uint8,shape=h.shape)
    for anchor in anchors:
        ix=np.flatnonzero(groups.anchor.to_numpy()==anchor);us=groups.u.to_numpy()[ix]
        c.register('active_users',pd.DataFrame({'user_id':users[us],'u':us}))
        frame=c.execute('''WITH p AS(SELECT s.*,a.u FROM strong s JOIN active_users a USING(user_id) WHERE day>=? AND day<?),
            runs AS(SELECT *,lead(i) OVER w AS ni,lead(action) OVER w AS na FROM p WINDOW w AS(PARTITION BY u ORDER BY ts,i,action))
            SELECT u,i,action,ts FROM runs WHERE ni IS NULL OR ni!=i OR na!=action
            QUALIFY row_number() OVER(PARTITION BY u ORDER BY ts DESC,i DESC,action DESC)<=50 ORDER BY u,ts,i,action''',[int(anchor)-61,int(anchor)-1]).df()
        positions=frame.groupby('u',sort=False).cumcount().to_numpy();row=pd.Index(us).get_indexer(frame.u)+int(ix[0])
        h[row,positions]=frame.i.to_numpy()+1;a[row,positions]=frame.action.to_numpy()
        print('STRONG_CONTEXT',int(anchor),len(ix),len(frame),flush=True)
    h.flush();a.flush();assert np.asarray(h).any(1).all()
    arrays={'positive':target.i.to_numpy(np.int32),'anchors':target.anchor.to_numpy(np.int16),'users':target.u.to_numpy(np.int32),
            'group_index':gi,'group_offsets':np.r_[0,np.cumsum(groups.targets)].astype(np.int64),'rows_strong_all':np.arange(len(target),dtype=np.int32)}
    for name,values in arrays.items():np.save(dest/f'{name}.npy',values)
    qh,qa,_=data.history(1293,strong_only=True)
    np.save(dest/'query_strong_h.npy',np.pad(qh,((0,0),(0,50))).astype(np.int32));np.save(dest/'query_strong_actions.npy',np.pad(qa,((0,0),(0,50))).astype(np.uint8))
    for anchor in anchors:
        ids=np.flatnonzero(data.first_day<int(anchor)-1)
        past=data.events[(data.events.day<int(anchor)-1)&(data.events.action==2)&(data.events.i>=0)]
        counts=np.bincount(past.i,minlength=data.n_items)[ids].astype(float)**.75
        q=.5/len(ids)+.5*counts/counts.sum();cdf=np.cumsum(q);cdf[-1]=1.
        np.savez(proposal_dir/f'proposal_{anchor}.npz',ids=ids,q=q,cdf=cdf)
    f=data.events[(data.events.day<1292)&(data.events.action==2)&(data.events.i>=0)]
    counts=np.bincount(f.i,minlength=data.n_items).astype(float)**.75
    np.save(proposal_dir/'nominal_q.npy',.5/data.n_items+.5*counts/counts.sum())
    recent=data.events[(data.events.day>=1264)&(data.events.day<1292)&(data.events.action==2)&(data.events.i>=0)]
    np.save(proposal_dir/'pop28.npy',np.lexsort((np.arange(data.n_items),-np.bincount(recent.i,minlength=data.n_items))))
    info=dict(pairs=len(target),training_users=int(target.u.nunique()),positive_items=int(target.i.nunique()))
    save_json(dest/'complete.json',dict(subsets={'strong_all':info},compact_strong_only=True))
    save_json(dest/'verification.json',dict(passed=True,all_histories_nonempty=True,training_goal_max_day=1290))
    c.close()


def priors(data):
    result={}
    for days in [28,90]:
        f=data.events[(data.events.day>=1292-days)&(data.events.day<1292)&(data.events.action==2)&(data.events.i>=0)]
        counts=np.bincount(f.i,minlength=data.n_items)
        result[f'pop{days}']=np.lexsort((np.arange(data.n_items),-counts)).astype(np.int32)
    return result


def save_ranking(name,raw,data,prior,fit_seconds=None,predict_seconds=None):
    q=evaluation_users(data);ys=[data.targets[u] for u in q];pred=fill(raw,prior,200)
    metrics,pu=evaluate(pred,ys,data.n_items);pu.u=q
    metrics.update(method=name,fit_seconds=fit_seconds,predict_seconds=predict_seconds,anchor=1293)
    folder=BENCH/'rankings';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{name}.npz',query_users=q,recommendations=pred)
    pu.to_parquet(folder/f'{name}_per_user.parquet',index=False);save_json(folder/f'{name}.json',metrics)
    print('BENCHMARK_METRIC',name,metrics['recall_reachable@200'],flush=True)
    return metrics


def classical():
    from implicit.cpu.als import AlternatingLeastSquares
    from threadpoolctl import threadpool_limits
    data=common_data();q=evaluation_users(data);p=priors(data)
    for label in ['pop28','pop90']:
        save_ranking(label,np.tile(p[label][:200],(len(q),1)),data,p['pop28'])
    for method,label in [('personal_popularity','personal_pop28'),('content_decay','content_decay'),('als','als_tail50')]:
        metrics=run_baseline(data,method,tag=label)
        raw=np.load(FULL/'artifacts'/label/'raw.npy',mmap_mode='r')[q]
        save_ranking(label,raw,data,p['pop28'],metrics['fit_seconds'],metrics['predict_seconds'])
    # The strongest historical BM25 recipe used a strong train and strong query, K100.
    original=(data.h,data.actions,data.ages);data.h,data.actions,data.ages=data.history(1293,strong_only=True)
    data.cfg.update(fit_strong_only=True,bm25_neighbors=CONFIG['bm25']['K'])
    m=run_baseline(data,'bm25',tag='bm25')
    save_ranking('bm25',np.load(FULL/'artifacts/bm25/raw.npy',mmap_mode='r')[q],data,p['pop28'],m['fit_seconds'],m['predict_seconds'])
    data.h,data.actions,data.ages=original;data.cfg['fit_strong_only']=False
    model=AlternatingLeastSquares.load(FULL/'artifacts/als_tail50_model.npz')
    profile=full_query_confidence(data.events,1293,data.n_users,data.n_items)
    raw=np.full((len(q),200),-1,np.int32);valid=np.diff(profile.indptr)[q]>0;start=time.perf_counter()
    with threadpool_limits(4,user_api='blas'):
        factors=model.recalculate_user(q,profile[q])
        for lo in range(0,len(q),64):
            scores=factors[lo:lo+64]@model.item_factors.T
            for j,row in enumerate(scores):
                if valid[lo+j]:raw[lo+j]=topk(row,200)
    base=json.loads((FULL/'artifacts/als_tail50/metrics.json').read_text())
    save_ranking('als_full60',raw,data,p['pop28'],base['fit_seconds'],time.perf_counter()-start)
    source=ROOT/'runs/large_mixed_20261105/data';rows=np.load(source/'rows_strong_all.npy')
    positive=np.load(source/'positive.npy',mmap_mode='r');frequency=np.bincount(positive[rows],minlength=data.n_items)
    ranking=np.lexsort((np.arange(data.n_items),-frequency))
    save_ranking('training_target_pop',np.tile(ranking[:200],(len(q),1)),data,p['pop28'])


def two_tower():
    import train_large_mixed as trainer
    from recsys_lab.update_monitor import UpdateMonitor
    assert torch.backends.mps.is_available(),'MPS required'
    torch.set_num_threads(4)
    original=trainer.load_model
    def patched(policy,seed=2026,use_content=True):
        data,model=original(policy,seed,use_content);return data,patch_tower_linears(model,16384)
    trainer.load_model=patched
    trainer_root=ROOT/'runs/large_mixed_20261105'
    save_json(trainer_root/'batch_benchmark.json',dict(chosen_batch=512,source='Frozen benchmark recipe, no batch search'))
    monitor=UpdateMonitor();monitor.protocol={**monitor.protocol,'linear_backward':'all4_tiled16384'}
    tag='strong_all_seed2026_lr0p0003_benchmark8_tiledall16384';dest=BENCH/'two_tower'/tag
    resume=None;suffix='_benchmark8_tiledall16384'
    if (dest/'configuration.json').exists():
        if not (dest/'last.pt').exists():raise RuntimeError('Partial TT run has no resumable checkpoint; use a fresh RUN_NAME')
        resume=dest/'last.pt';suffix+='_resume'+str(int(time.time()))
    trainer.train('strong_all',seed=CONFIG['seed'],lr=CONFIG['two_tower']['lr'],temperature=.1,maximum_epochs=8,
                  early_stop=False,run_suffix=suffix,output_root=BENCH/'two_tower',step_monitor=monitor,resume_from=resume)
    dest=BENCH/'two_tower'/('strong_all_seed2026_lr0p0003'+suffix)
    export_neural(dest,'two_tower')


def sequence():
    import train_prefix_sequence as trainer
    assert torch.backends.mps.is_available(),'MPS required'
    torch.set_num_threads(4);original=trainer.PrefixTower
    def patched(*a,**kw):return patch_tower_linears(original(*a,**kw),16384)
    trainer.PrefixTower=patched
    dest=ROOT/'runs/prefix_sequence_20261104/sequence'
    trainer.main(SimpleNamespace(encoder='sequence',length=50,updates=6000,resume=(dest/'last.pt').exists()))
    save_json(dest/'execution_variant.json',dict(linear_backward='all4_tiled16384',common_development_anchor=1293))
    export_neural(dest,'sequence')


def export_neural(folder,family):
    data=common_data();p=priors(data);q=evaluation_users(data);table=[]
    for label in ['raw','same_fixed_prior']:
        with np.load(folder/f'{label}_lists.npz') as z:
            np.testing.assert_array_equal(z['query_users'],q)
            raw=z['raw']
        original=json.loads((folder/f'{label}_metrics.json').read_text())
        metrics=save_ranking(family+'_'+label,raw,data,p['pop28'],original.get('fit_seconds'),None)
        metrics.update(best_updates=original.get('best_updates'),scoring=label,checkpoint=str(folder.relative_to(ROOT)/'checkpoint.pt'))
        table.append(metrics)
    selected=max(table,key=lambda m:m['recall_reachable@200'])
    for suffix in ['.npz','.json','_per_user.parquet']:
        shutil.copyfile(BENCH/'rankings'/(selected['method']+suffix),BENCH/'rankings'/(family+suffix))
    save_json(BENCH/(family+'_selection.json'),dict(selected=selected,variants=table,selection_on_development=True))
    shutil.copyfile(folder/'learning_curve.csv',BENCH/(family+'_learning_curve.csv'))
    if (folder/'training_loss.csv').exists():shutil.copyfile(folder/'training_loss.csv',BENCH/(family+'_training_loss.csv'))


def quota_merge(rankings,quotas,k=200):
    """Weighted rank interleaving, preserving total length and skipping duplicates."""
    names=list(quotas);weights=np.array([quotas[n] for n in names]);assert weights.sum()==k
    credit=np.zeros(len(names),dtype=int);schedule=[]
    for _ in range(k):
        credit+=weights;chosen=int(np.argmax(credit));schedule.append(chosen);credit[chosen]-=k
    n=len(rankings[names[0]]);result=np.full((n,k),-1,np.int32)
    for row in range(n):
        pointers=np.zeros(len(names),dtype=int);seen=set();out=[]
        for preferred in schedule:
            for source in [preferred,*[j for j in range(len(names)) if j!=preferred]]:
                values=rankings[names[source]][row]
                while pointers[source]<len(values):
                    item=int(values[pointers[source]]);pointers[source]+=1
                    if item>=0 and item not in seen:
                        seen.add(item);out.append(item);break
                else:continue
                break
        result[row,:len(out)]=out
    return result


def compare():
    from summarize_large_mixed import paired_ci
    data=common_data();q=evaluation_users(data);prior=priors(data)['pop28'];rankings={};rows=[];per={}
    names=['pop28','pop90','personal_pop28','training_target_pop','bm25','content_decay','als_tail50','als_full60',
           'two_tower_raw','two_tower_same_fixed_prior','sequence_raw','sequence_same_fixed_prior','two_tower','sequence']
    for name in names:
        with np.load(BENCH/'rankings'/f'{name}.npz') as z:
            np.testing.assert_array_equal(z['query_users'],q);rankings[name]=z['recommendations']
        per[name]=pd.read_parquet(BENCH/'rankings'/f'{name}_per_user.parquet')
        metrics=json.loads((BENCH/'rankings'/f'{name}.json').read_text());metrics['method']=name;rows.append(metrics)
    pd.DataFrame(rows).to_csv(BENCH/'model_comparison.csv',index=False)
    fusions=[];targets=[data.targets[u] for u in q];base=rankings['personal_pop28']
    for name,quotas in CONFIG['fusions'].items():
        if len(quotas)==2:
            from recsys_lab.fusion import combine
            a,b=list(quotas);pred=combine(rankings[a],rankings[b],quotas[a])
        else:pred=quota_merge(rankings,quotas)
        metrics=save_ranking(name,pred,data,prior);new_per=pd.read_parquet(BENCH/'rankings'/f'{name}_per_user.parquet')
        interval=paired_ci(new_per,per['personal_pop28']);added=lost=0;am=lm=0.;valid=0
        for old,new,y in zip(base,pred,targets):
            ys={i for i in y if isinstance(i,int)}
            if not ys:continue
            valid+=1;plus=(set(new)-set(old))&ys;minus=(set(old)-set(new))&ys
            added+=len(plus);lost+=len(minus);am+=len(plus)/len(ys);lm+=len(minus)/len(ys)
        metrics.update(delta=interval['delta'],low95=interval['low95'],high95=interval['high95'],added=added,lost=lost,
                       added_macro=am/valid,lost_macro=lm/valid,quotas=json.dumps(quotas))
        assert np.isclose(metrics['delta'],metrics['added_macro']-metrics['lost_macro'])
        fusions.append(metrics)
    pd.DataFrame(fusions).sort_values('recall_reachable@200',ascending=False).to_csv(BENCH/'fusion_comparison.csv',index=False)
    selected=[n for n in ['bm25','content_decay','als_full60','two_tower','sequence']]
    overlap=pd.DataFrame(index=selected,columns=selected,dtype=float)
    for a in selected:
        for b in selected:
            overlap.loc[a,b]=np.mean([len(set(x)&set(y))/len(set(x)|set(y)) for x,y in zip(rankings[a],rankings[b])])
    overlap.to_csv(BENCH/'ranking_overlap.csv')
    save_json(BENCH/'complete.json',dict(completed=True,models=len(selected),fusions=len(fusions),anchor=1293,final_test_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','classical','two_tower','sequence','compare'])
    a=parser.parse_args();start=time.perf_counter()
    assert torch.backends.mps.is_available(),'MPS is required; neural CPU fallback is not used'
    print('PHASE_START',a.phase,'MPS',torch.backends.mps.is_available(),flush=True)
    globals()[a.phase]()
    save_json(BENCH/(a.phase+'_complete.json'),dict(completed=True,seconds=time.perf_counter()-start,phase=a.phase))
    print('PHASE_COMPLETE',a.phase,flush=True)
