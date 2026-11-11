"""Isolated execution of the existing main recipes. Never imported in saved mode."""
from pathlib import Path
import argparse
import ast
import gc
import json
import os
import runpy
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
STAGES = ['01_data_and_protocol','02_baselines_and_classical','03_two_tower',
          '04_sequence_models','05_fusion_and_robustness','06_final_results']
parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=STAGES)
args = parser.parse_args()
active = ROOT/'runs/active_history'
full = ROOT/'runs/full_catalog_20261104'
os.environ['RECSYS_WORKSPACE'] = str(full if args.stage in STAGES[2:4] else active)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','4')

import numpy as np
import pandas as pd
from recsys_lab.data import PilotData, STRENGTH, save_json
from recsys_lab.baselines import evaluation_users, fill, topk, run_baseline
from recsys_lab.metrics import evaluate
from recsys_lab.fusion import combine

OUT = ROOT/'chapter_results'/args.stage
OUT.mkdir(parents=True,exist_ok=True)


def require(*paths):
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError('Run preceding chapters first. Missing: '+', '.join(missing))


def extracted(filename, name):
    """Load a specific recorded function without executing a script's orchestration."""
    source = (ROOT/filename).read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(source,node)


def overview(rows):
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'overview.csv',index=False)
    print(table.to_string(index=False),flush=True)


def load_data():
    return PilotData()


def pop28(data):
    counts = np.zeros(data.n_items,np.int64)
    end = data.cfg['validation_anchor']-1
    s = data.events[(data.events.day>=end-28)&(data.events.day<end)&(data.events.action==2)&(data.events.i>=0)].groupby('i').size()
    counts[s.index] = s.values
    return np.lexsort((np.arange(data.n_items),-counts))


def summarize_rank(data, label, raw, prior):
    q = evaluation_users(data)
    pred = fill(raw,prior,200)
    m,pu = evaluate(pred,[data.targets[u] for u in q],data.n_items)
    pu.u = q
    m.update(method=label,source='new computation')
    pu.to_parquet(OUT/f'{label}_per_user.parquet',index=False)
    np.savez_compressed(OUT/f'{label}_lists.npz',query_users=q,raw=raw,recommendations=pred)
    return m


def stage01():
    require(ROOT/'data/raw/items.pq',*[ROOT/f'data/raw/events/{d:05}.pq' for d in range(1174,1300)])
    # Existing past-only population audit: the selection rule is unchanged.
    runpy.run_path(str(ROOT/'audit_population.py'),run_name='__main__')
    from recsys_lab.data import select_events,select_items
    select_events();select_items()
    data = load_data()
    save_json(active/'artifacts/dataset_summary.json',dict(users=data.n_users,items=data.n_items,fingerprint=data.fingerprint))
    for name in ['configs','artifacts','data']:
        dest=full/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.symlink_to((active/name).resolve(),target_is_directory=True)
    daily=data.events.groupby('day',as_index=False).agg(events=('action','size'),first_ts=('ts','min'),last_ts=('ts','max'))
    clicks=data.events[data.events.action==2].groupby('day').size()
    daily['clicks']=daily.day.map(clicks).fillna(0).astype(int)
    daily.to_csv(OUT/'daily_events.csv',index=False)
    quality=dict(rows=len(data.events),null_users=int(data.events.user_id.isna().sum()),missing_items=int(data.events.item_id.isna().sum()),null_time=int(data.events.ts.isna().sum()),invalid_actions=int((~data.events.action.isin([1,2,3,4])).sum()),day_time_mismatch=int((data.events.ts//86400000000!=data.events.day).sum()),missing_surface=int(data.events.subdomain.isna().sum()))
    pd.DataFrame([quality]).to_csv(OUT/'processed_quality.csv',index=False)
    overview([dict(method='prepared dataset',users=data.n_users,items=data.n_items,events=len(data.events),fingerprint=data.fingerprint)])


def stage02():
    require(active/'data/processed/events.parquet')
    data=load_data();q=evaluation_users(data);prior=pop28(data);rows=[]
    for method in ['popularity','recent','personal_popularity','content','als']:
        path=active/'artifacts'/method/'raw.npy'
        if not path.exists():run_baseline(data,method)
        raw=np.load(path,mmap_mode='r')[q]
        label={'popularity':'pop90','recent':'recent_pop28','personal_popularity':'personal_pop28','content':'provided_content','als':'als_tail50'}.get(method,method)
        rows.append(summarize_rank(data,label,raw,prior))
    # Same BM25 recipe as the strong300 comparison.
    previous_h,previous_a,previous_ages=data.h,data.actions,data.ages
    data.h,data.actions,data.ages=data.history(1293,strong_only=True)
    data.cfg.update(fit_strong_only=True,bm25_neighbors=300)
    if not (active/'artifacts/bm25_strong300/raw.npy').exists():run_baseline(data,'bm25','bm25_strong300')
    rows.append(summarize_rank(data,'bm25_strong300',np.load(active/'artifacts/bm25_strong300/raw.npy',mmap_mode='r')[q],prior))
    data.h,data.actions,data.ages=previous_h,previous_a,previous_ages
    data.cfg['fit_strong_only']=False
    from scipy import sparse
    from implicit.cpu.als import AlternatingLeastSquares
    model=AlternatingLeastSquares.load(active/'artifacts/als_model.npz')
    f=data.events[(data.events.day>=1232)&(data.events.day<1292)&(data.events.i>=0)]
    same=(f.u==f.u.shift(-1))&(f.i==f.i.shift(-1))&(f.action==f.action.shift(-1));f=f.loc[~same]
    matrix=sparse.csr_matrix((STRENGTH[f.action.to_numpy()],(f.u,f.i)),shape=(data.n_users,data.n_items))
    matrix.data=1+20*np.log1p(matrix.data)
    factors=model.recalculate_user(q,matrix[q]);raw=np.full((len(q),200),-1,np.int32)
    nonempty=np.diff(matrix.indptr)[q]>0
    for lo in range(0,len(q),64):
        scores=factors[lo:lo+64]@model.item_factors.T
        for j,row in enumerate(scores):
            if nonempty[lo+j]:raw[lo+j]=topk(row,200)
    rows.append(summarize_rank(data,'als_full60',raw,prior));overview(rows)
    # A global numeric event archive used by the original temporal and final recipes.
    from recsys_lab.temporal_checks import prepare
    prepare(ROOT,ROOT/'runs/investigation_20261104/temporal')


def stage03():
    require(active/'artifacts/dataset_summary.json')
    import torch
    if not torch.backends.mps.is_available():raise RuntimeError('MPS is required for the recorded NN fits. No CPU fallback.')
    from prepare_expansion import main as expand
    if not (ROOT/'runs/expansion_20261103/manifest.json').exists():expand(ROOT,200000)
    if not (ROOT/'runs/history_20261103/pairs_strong.npz').exists():runpy.run_path(str(ROOT/'prepare_strong.py'),run_name='__main__')
    if not (ROOT/'runs/windows_20261103/manifest.json').exists():runpy.run_path(str(ROOT/'prepare_windows.py'),run_name='__main__')
    import train_full_catalog
    rows=[]
    for weeks in [4,8]:
        folder=full/f'weeks{weeks}'
        if not (folder/'complete.json').exists():train_full_catalog.main(weeks=weeks,updates=4000)
        for scoring in ['raw','same_fixed_prior']:
            m=json.loads((folder/f'{scoring}_metrics.json').read_text());m['method']=f'w{weeks}_{scoring}';rows.append(m)
    overview(rows)


def stage04():
    require(ROOT/'runs/windows_20261103/events.parquet')
    import torch
    if not torch.backends.mps.is_available():raise RuntimeError('MPS is required for the recorded NN fits.')
    from prepare_prefix_sequence import main as prefixes
    if not (ROOT/'runs/prefix_sequence_20261104/data/complete.json').exists():prefixes()
    import train_prefix_sequence
    rows=[]
    for encoder in ['pool','sequence']:
        folder=ROOT/'runs/prefix_sequence_20261104'/encoder
        if not (folder/'complete.json').exists():
            train_prefix_sequence.main(SimpleNamespace(encoder=encoder,length=50,updates=6000,resume=(folder/'last.pt').exists()))
        for scoring in ['raw','same_fixed_prior']:
            m=json.loads((folder/f'{scoring}_metrics.json').read_text());m['method']=f'{encoder}_{scoring}';rows.append(m)
    overview(rows)


def simple_window(events,namespace,users,anchor):
    # Run the actual original function, omitting only its comparison with files from another machine.
    source=extracted('recsys_lab/temporal_checks.py','check_window')
    tree=ast.parse(source);function=tree.body[0]
    function.body=[node for node in function.body if not (isinstance(node,ast.If) and ast.unparse(node.test)=='anchor == 1293')]
    from recsys_lab import temporal_checks
    env=dict(vars(temporal_checks));exec(compile(ast.fix_missing_locations(tree),'<recorded-check-window>','exec'),env)
    return env['check_window'](events,namespace,users,anchor,ROOT/'runs/investigation_20261104/temporal',ROOT)


def stage05():
    require(ROOT/'runs/investigation_20261104/temporal/numeric_events.parquet')
    from implicit.cpu.als import AlternatingLeastSquares
    from threadpoolctl import threadpool_limits
    from recsys_lab.temporal_als import training_confidence,full_query_confidence,target_sets
    ref=ROOT/'runs/investigation_20261104/temporal'
    events=pd.read_parquet(ref/'numeric_events.parquet');users=np.load(active/'data/processed/users.npy')
    namespace=json.loads((ref/'data_manifest.json').read_text())['namespace_items']
    rows=[]
    for anchor in [1279,1286,1293]:
        simple_window(events,namespace,users,anchor)
        end=anchor-1;past=events[(events.day>=end-90)&(events.day<end)&(events.i>=0)].copy()
        catalog=np.sort(past.i.unique());past['i']=pd.Index(catalog).get_indexer(past.i).astype(np.int32)
        future=events[(events.day>=anchor)&(events.day<anchor+7)&(events.action==2)]
        targets=target_sets(future,catalog,len(users));q=np.array([u for u,y in enumerate(targets) if y],np.int32);ys=[targets[u] for u in q]
        def saved(name):
            with np.load(ref/f'anchor_{anchor}/{name}_lists.npz') as z:
                np.testing.assert_array_equal(z['query_users'],q)
                return pd.Index(catalog).get_indexer(z['recommendations'].ravel()).reshape(len(q),200).astype(np.int32)
        base=saved('personal_pop28');prior=saved('pop28')[0]
        m,base_per=evaluate(base,ys,len(catalog));base_per.u=q
        m.update(method='personal_pop28',anchor=anchor);rows.append(m)
        confidence=training_confidence(past,len(users),len(catalog));profile=full_query_confidence(past,anchor,len(users),len(catalog))
        for seed in [2026,2027,2028]:
            model=AlternatingLeastSquares(factors=32,regularization=.1,iterations=10,alpha=1.,num_threads=4,random_state=seed)
            start=time.perf_counter()
            with threadpool_limits(1,user_api='blas'):model.fit(confidence,show_progress=False)
            fit=time.perf_counter()-start;raw=np.full_like(base,-1);valid=np.diff(profile.indptr)[q]>0
            with threadpool_limits(4,user_api='blas'):
                factors=model.recalculate_user(q,profile[q])
                for lo in range(0,len(q),64):
                    scores=factors[lo:lo+64]@model.item_factors.T
                    for j,row in enumerate(scores):
                        if valid[lo+j]:raw[lo+j]=topk(row,200)
            for label,pred in [('als_full60',fill(raw,prior,200)),('personal_als_b10',combine(base,raw,190)),('personal_als_b25',combine(base,raw,175))]:
                m,pu=evaluate(pred,ys,len(catalog));m.update(method=label,anchor=anchor,seed=seed,fit_seconds=fit)
                added=lost=0;added_mass=lost_mass=0.
                for old,new,y in zip(base,pred,ys):
                    reachable={i for i in y if isinstance(i,int)}
                    if reachable:
                        plus=(set(new)-set(old))&reachable;minus=(set(old)-set(new))&reachable
                        added+=len(plus);lost+=len(minus);added_mass+=len(plus)/len(reachable);lost_mass+=len(minus)/len(reachable)
                n=sum(any(isinstance(i,int) for i in y) for y in ys)
                m.update(added_target_pairs=added,lost_target_pairs=lost,added_macro_mass=added_mass/n,lost_macro_mass=lost_mass/n)
                pu.u=q
                ci_env={'np':np,'pd':pd,'KEY':'recall_reachable@200'}
                exec(extracted('summarize_large_mixed.py','paired_ci'),ci_env)
                interval=ci_env['paired_ci'](pu,base_per)
                m.update(delta_vs_personal=interval['delta'],ci_low=interval['low95'],ci_high=interval['high95'])
                rows.append(m);pu.to_parquet(OUT/f'{anchor}_{seed}_{label}_per_user.parquet',index=False)
        del past,confidence,profile,model;gc.collect()
    overview(rows)


def stage06():
    require(ROOT/'runs/investigation_20261104/temporal/numeric_events.parquet',*[ROOT/f'data/raw/events/{d:05}.pq' for d in range(1301,1308)])
    from scipy import sparse
    import duckdb
    from implicit.cpu.als import AlternatingLeastSquares
    from threadpoolctl import threadpool_limits
    from recsys_lab.temporal_als import training_confidence,full_query_confidence
    ns=globals().copy();ns.update(sparse=sparse,duckdb=duckdb,AlternatingLeastSquares=AlternatingLeastSquares,
                                threadpool_limits=threadpool_limits,training_confidence=training_confidence,full_query_confidence=full_query_confidence)
    ns['OUT']=ROOT/'runs/closing_checks_20261107';ns['OUT'].mkdir(parents=True,exist_ok=True)
    ns['REVISION']='79bcf78cd4699267f561ad3fd51fe7b52af53083'
    ns['KEY']='recall_reachable@200'
    exec(extracted('summarize_large_mixed.py','paired_ci'),ns)
    source=extracted('final_research_checks.py','final')
    # Files were downloaded by the reader; there is no local audit-document dependency or network side effect.
    start=source.index("    audit=json.loads((ROOT.parent/")
    end=source.index("    userids=np.load",start)
    source=source[:start]+source[end:]
    exec(source,ns);ns['final']()
    table=pd.read_csv(ns['OUT']/'final_comparison.csv');overview(table.to_dict('records'))
    a=pd.read_parquet(ns['OUT']/'final_personal_pop28_per_user.parquet')
    b=pd.read_parquet(ns['OUT']/'final_personal_als_b10_per_user.parquet');assert a.u.equals(b.u)
    pd.DataFrame({f'{p}_{m}':df[col] for p,df in [('base',a),('als10',b)]
                  for m,col in [('recall','recall_reachable@200'),('recall_all','recall_all@200'),('ndcg','ndcg@200')]}).to_csv(OUT/'paired_metrics.csv',index=False)


if __name__=='__main__':
    started=time.perf_counter()
    globals()['stage'+str(STAGES.index(args.stage)+1).zfill(2)]()
    save_json(OUT/'execution.json',dict(stage=args.stage,completed=True,seconds=time.perf_counter()-started,
              data_source='reader-provided T-ECD raw files',result_source='fresh computation',training_used_saved_metrics=False))
