"""Explain reachable targets and test a fixed category-based candidate channel."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .baselines import fill
from .data import PilotData,STRENGTH,save_json
from .fusion import combine
from .metrics import evaluate


def contains(sorted_keys,values):
    values=np.asarray(values,dtype=np.int64)
    if not len(sorted_keys):return np.zeros(values.shape,bool)
    indices=np.searchsorted(sorted_keys,values)
    return (indices<len(sorted_keys))&(sorted_keys[np.minimum(indices,len(sorted_keys)-1)]==values)


def category_labels(seen,category,known):
    return np.select([seen,category<0,known],['repeat90','unknown_category','new_item_known_category'],default='new_item_new_category')


def decompose(targets,hits,group_column):
    """Additive macro contributions use the original full reachable denominator."""
    frame=targets[['u',group_column,'macro_weight']].copy();frame['hit']=np.asarray(hits,int)
    rows=[]
    for group,g in frame.groupby(group_column,sort=False):
        local=g.groupby('u').hit.agg(['sum','size'])
        rows.append(dict(group=group,user_item_targets=len(g),users=len(local),hits=int(g.hit.sum()),
            micro_target_fraction=len(g)/len(frame),macro_opportunity=float(g.macro_weight.sum()),
            macro_contribution=float((g.macro_weight*g.hit).sum()),
            conditional_group_macro_recall=float((local['sum']/local['size']).mean())))
    return pd.DataFrame(rows)


def prepare(project,destination):
    project,destination=Path(project),Path(destination);destination.mkdir(parents=True,exist_ok=True)
    c=duckdb.connect();c.execute("SET memory_limit='1500MB'");c.execute('SET threads=3')
    c.read_parquet(str(project/'data/raw/items.pq')).create_view('catalog')
    cats=c.execute("SELECT DISTINCT category FROM catalog WHERE category IS NOT NULL AND trim(category)<>'' AND lower(category) NOT LIKE '%uncategor%' ORDER BY category").df()
    cats['category_code']=np.arange(len(cats),dtype=np.int16);c.register('categories',cats)
    cats.to_csv(destination/'category_vocabulary.csv',index=False)
    c.execute('''CREATE TABLE items AS SELECT item_id,(row_number() OVER(ORDER BY item_id)-1)::INTEGER AS i,
        coalesce(category_code,-1)::SMALLINT AS category_code FROM catalog LEFT JOIN categories USING(category)''')
    category=c.execute('SELECT category_code FROM items ORDER BY i').fetchnumpy()['category_code']
    c.read_parquet(str(project/'runs/active_history/data/processed/items.parquet')).create_view('old_catalog')
    mapping=c.execute('''SELECT i FROM (SELECT item_id,row_number() OVER(ORDER BY item_id)-1 AS old_i FROM old_catalog)
        JOIN items USING(item_id) ORDER BY old_i''').fetchnumpy()['i']
    assert len(mapping)==1060869 and len(np.unique(mapping))==len(mapping)
    np.save(destination/'original_to_global_item.npy',mapping)
    users=np.load(project/'runs/active_history/data/processed/users.npy')
    c.register('cohort',pd.DataFrame({'user_id':users,'u':np.arange(len(users),dtype=np.int32)}))
    c.read_parquet(str(project/'runs/active_history/data/processed/events.parquet')).create_view('source')
    clicks=c.execute('''SELECT u,ts,day,i,subdomain FROM source JOIN cohort USING(user_id) JOIN items USING(item_id)
        WHERE action=2 AND day>=1279 AND day<1300 ORDER BY u,ts,i''').df();c.close()
    events=pd.read_parquet(project/'runs/investigation_20261104/temporal/numeric_events.parquet')
    save_json(destination/'data_manifest.json',dict(users=len(users),namespace_items=len(category),
        valid_categories=len(cats),uncategorized_catalog_items=int((category<0).sum()),
        category_unknown_rule='NULL,empty,or category contains uncategor; unknown is not a preference category',
        category_profile='Past60days,strong actions,daily user/item max strength; top3 categories',
        category_score='P(category|user past strong actions)*past28 click frequency(item)/past28 clicks(category)',
        novelty='Absent from that user entire available90-day prefix; no future labels used',
        fusion_quota=10,quota_tuned=False,final_test_used=False))
    return dict(project=project,destination=destination,events=events,category=category,
                category_names=cats.category.tolist(),mapping=mapping,users=users,clicks=clicks)


def context(state,anchor):
    e=state['events'];ns=len(state['category']);end=anchor-1
    past=e[(e.day>=end-90)&(e.day<end)&(e.i>=0)]
    future=e[(e.day>=anchor)&(e.day<anchor+7)&(e.action==2)].drop_duplicates(['u','i'])
    assert (future.i>=0).all(),'Unknown item namespace requires explicit outside-ID handling'
    allowed=np.zeros(ns,bool);allowed[past.i.unique()]=True
    q=np.sort(future.u.unique()).astype(np.int32);qmap={int(u):j for j,u in enumerate(q)}
    ys=[set() for _ in q]
    for row in future.itertuples():ys[qmap[row.u]].add(int(row.i) if allowed[row.i] else 'outside:'+str(row.i))
    target=future[allowed[future.i.to_numpy()]][['u','i']].copy().reset_index(drop=True)
    nr=target.groupby('u').size();n_reachable=len(nr)
    target['macro_weight']=1./target.u.map(nr)/n_reachable
    p=past[past.u.isin(q)].copy();p60=p[p.day>=end-60]
    seen90=np.unique(p.u.to_numpy(dtype=np.int64)*ns+p.i.to_numpy())
    seen60=np.unique(p60.u.to_numpy(dtype=np.int64)*ns+p60.i.to_numpy())
    d=PilotData.__new__(PilotData);d.n_users=len(q);d.cfg={'history_days':60,'history_length':50}
    d.events=p.assign(u=np.searchsorted(q,p.u.to_numpy()).astype(np.int32))
    h,_,_=d.history(anchor);hs,_,_=d.history(anchor,strong_only=True)
    def item_keys(history):
        u,pos=np.nonzero(history)
        return np.unique(q[u].astype(np.int64)*ns+history[u,pos]-1)
    nc=len(state['category_names']);cat=state['category']
    def cat_keys(frame):
        values=cat[frame.i.to_numpy()];valid=values>=0
        return np.unique(frame.u.to_numpy(dtype=np.int64)[valid]*nc+values[valid])
    def input_cat_keys(history):
        u,pos=np.nonzero(history);values=cat[history[u,pos]-1];valid=values>=0
        return np.unique(q[u[valid]].astype(np.int64)*nc+values[valid])
    target['category']=cat[target.i.to_numpy()]
    ik=target.u.to_numpy(dtype=np.int64)*ns+target.i.to_numpy()
    ck=target.u.to_numpy(dtype=np.int64)*nc+target.category.to_numpy()
    target['seen90']=contains(seen90,ik);target['seen60']=contains(seen60,ik)
    target['seen_all50']=contains(item_keys(h),ik);target['seen_strong50']=contains(item_keys(hs),ik)
    for name,keys in [('category_known90',cat_keys(p)),('category_known60',cat_keys(p60)),
                      ('category_known_strong60',cat_keys(p60[p60.action>1])),
                      ('category_in_all50',input_cat_keys(h)),('category_in_strong50',input_cat_keys(hs))]:
        target[name]=(target.category>=0)&contains(keys,ck)
    target['group']=category_labels(target.seen90.to_numpy(),target.category.to_numpy(),target.category_known90.to_numpy())
    target['retention']=np.where(target.seen90,
        np.select([target.seen_all50,target.seen60],['repeat_in_all50','repeat_in60_not_all50'],default='repeat_only90_outside60'),target.group)
    assert np.isclose(target.macro_weight.sum(),1.)
    out=state['destination']/f'anchor_{anchor}';out.mkdir(exist_ok=True)
    qpath=state['project']/f'runs/investigation_20261104/temporal/anchor_{anchor}'
    methods={}
    for name in ['pop28','personal_pop28','frozen_fusion_b25']:
        with np.load(qpath/f'{name}_lists.npz') as z:
            np.testing.assert_array_equal(z['query_users'],q);methods[name]=z['recommendations']
    return dict(anchor=anchor,end=end,past=past,profile_events=p60,q=q,ys=ys,target=target,
        allowed=allowed,seen90=seen90,methods=methods,destination=out,namespace=ns,qmap=qmap)


def category_methods(state,ctx):
    cat=state['category'];nc=len(state['category_names']);ns=ctx['namespace'];q=ctx['q']
    strong=ctx['profile_events'][ctx['profile_events'].action>1].copy()
    strong['category']=cat[strong.i.to_numpy()];strong=strong[strong.category>=0]
    strong['strength']=STRENGTH[strong.action.to_numpy()]
    daily=strong.groupby(['u','i','day'],sort=False).agg(strength=('strength','max'),category=('category','first'))
    weights=daily.reset_index().groupby(['u','category'],sort=False).strength.sum()
    profile=np.zeros((len(q),nc),np.float32)
    for (u,c),w in weights.items():profile[ctx['qmap'][u],c]=w
    f=ctx['past'][(ctx['past'].day>=ctx['end']-28)&(ctx['past'].action==2)]
    counts=np.zeros(ns,np.int32);s=f.groupby('i').size();counts[s.index]=s.to_numpy()
    valid=np.flatnonzero(ctx['allowed']&(cat>=0)&(counts>0))
    order=valid[np.lexsort((valid,-counts[valid]))]
    bycat={c:order[cat[order]==c] for c in range(nc)}
    mass=np.bincount(cat[valid],weights=counts[valid],minlength=nc)
    raw=np.full((len(q),200),-1,np.int32);novel=np.full_like(raw,-1)
    raw_scores=np.full(raw.shape,np.nan,np.float32);novel_scores=raw_scores.copy();profiles=[]
    for j,u in enumerate(q):
        fav=np.lexsort((np.arange(nc),-profile[j]))[:3];fav=fav[profile[j,fav]>0]
        confidence=profile[j]/max(float(profile[j].sum()),1e-12)
        profiles.extend(dict(u=int(u),category=int(c),weight=float(confidence[c])) for c in fav)
        for output,score_output,only_new in [(raw,raw_scores,False),(novel,novel_scores,True)]:
            ids=[];scores=[]
            for c in fav:
                cand=bycat[c]
                if only_new:cand=cand[~contains(ctx['seen90'],int(u)*ns+cand)]
                cand=cand[:200]
                ids.extend(cand);scores.extend(confidence[c]*counts[cand]/max(mass[c],1.))
            if ids:
                ids=np.asarray(ids,np.int32);scores=np.asarray(scores)
                take=np.lexsort((ids,-scores))[:200];ids=ids[take]
                output[j,:len(ids)]=ids;score_output[j,:len(ids)]=scores[take]
    assert not contains(ctx['seen90'],np.repeat(q,200).astype(np.int64)[novel.ravel()>=0]*ns+novel.ravel()[novel.ravel()>=0]).any()
    prior=ctx['methods']['pop28'][0]
    ctx['methods']['category_pop28']=fill(raw,prior,200)
    ctx['methods']['category_novel10']=combine(ctx['methods']['personal_pop28'],novel,190)
    empty=(novel<0).all(1)
    np.testing.assert_array_equal(ctx['methods']['category_novel10'][empty],ctx['methods']['personal_pop28'][empty])
    np.savez_compressed(ctx['destination']/'category_channels.npz',query_users=q,raw=raw,raw_scores=raw_scores,novel_raw=novel,novel_scores=novel_scores)
    pd.DataFrame(profiles).to_csv(ctx['destination']/'category_profiles.csv',index=False)
    save_json(ctx['destination']/'category_protocol.json',dict(top_categories=3,novel_channel_quota=10,parameters_tuned=False,
        valid_profile_queries=int((profile.sum(1)>0).sum()),queries=len(q),empty_novel_channel_queries=int(empty.sum()),
        unknown_target_category_fraction=float((ctx['target'].category<0).mean()),
        category_fallback_position_fraction=float((raw<0).mean())))


def evaluate_methods(state,ctx):
    a=ctx['anchor'];q=ctx['q'];ns=ctx['namespace'];t=ctx['target'];keys=t.u.to_numpy(dtype=np.int64)*ns+t.i.to_numpy()
    if a==1293:
        old=state['project']/'runs/active_history/artifacts'
        for name in ['als','tt_id','tt_content','attention']:
            pred=np.load(old/name/'recommendations.npy',mmap_mode='r')[q]
            assert (pred>=0).all();ctx['methods'][name]=state['mapping'][pred]
    metrics=[];groups=[];retention=[]
    for name,pred in ctx['methods'].items():
        assert ctx['allowed'][pred].all()
        m,pu=evaluate(pred,ctx['ys'],ns);pu.u=q
        if name in ['als','tt_id','tt_content','attention']:
            reference=json.loads((state['project']/'runs/active_history/artifacts'/name/'metrics.json').read_text())
            for key in ['recall_reachable@50','recall_reachable@200','recall_all@200','ndcg@200']:
                assert np.isclose(m[key],reference[key],atol=1e-12,rtol=0),(name,key)
        prediction_keys=np.sort((q.astype(np.int64)[:,None]*ns+pred).ravel())
        hit=contains(prediction_keys,keys);t['hit_'+name]=hit
        g=decompose(t,hit,'group');g['method']=name;groups.append(g)
        rg=decompose(t,hit,'retention');rg['method']=name;retention.append(rg)
        assert np.isclose(g.macro_contribution.sum(),m['recall_reachable@200'],atol=1e-12)
        m.update(method=name,anchor=a,candidate_items=int(ctx['allowed'].sum()),cohort_users=80018)
        metrics.append(m);save_json(ctx['destination']/f'{name}_metrics.json',m)
        pu.to_parquet(ctx['destination']/f'{name}_per_user.parquet',index=False)
        if name.startswith('category_'):np.savez_compressed(ctx['destination']/f'{name}_lists.npz',query_users=q,recommendations=pred)
        print('INTEREST',a,name,m['recall_reachable@200'],flush=True)
    pd.DataFrame(metrics).to_csv(ctx['destination']/'comparison.csv',index=False)
    pd.concat(groups).to_csv(ctx['destination']/'target_groups.csv',index=False)
    pd.concat(retention).to_csv(ctx['destination']/'retention_groups.csv',index=False)
    added=t.hit_category_novel10&~t.hit_personal_pop28;lost=t.hit_personal_pop28&~t.hit_category_novel10
    assert not (added&t.seen90).any()
    delta=[]
    for group,g in t.groupby('group'):
        add=added.loc[g.index];remove=lost.loc[g.index]
        delta.append(dict(group=group,added_user_item_targets=int(add.sum()),lost_user_item_targets=int(remove.sum()),
            added_macro_contribution=float(g.loc[add,'macro_weight'].sum()),lost_macro_contribution=float(g.loc[remove,'macro_weight'].sum())))
    pd.DataFrame(delta).to_csv(ctx['destination']/'added_lost_targets.csv',index=False)
    comparison=[]
    for name in ['category_pop28','category_novel10']:
        p=pd.read_parquet(ctx['destination']/f'{name}_per_user.parquet')
        b=pd.read_parquet(ctx['destination']/'personal_pop28_per_user.parquet')
        difference=(p['recall_reachable@200']-b['recall_reachable@200']).dropna().to_numpy()
        rng=np.random.default_rng(2026);samples=[rng.choice(difference,len(difference),replace=True).mean() for _ in range(1000)]
        comparison.append(dict(method=name,baseline='personal_pop28',difference=float(difference.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),users=len(difference)))
    save_json(ctx['destination']/'comparisons.json',comparison)
    t.to_parquet(ctx['destination']/'reachable_target_cases.parquet',index=False)
    t.to_csv(ctx['destination']/'reachable_target_cases.csv',index=False)
    return pd.DataFrame(metrics)


def popularity_drift(state,ctx):
    f=state['clicks'];f=f[(f.day>=ctx['anchor'])&(f.day<ctx['anchor']+7)]
    ev=state['events'];expected=int(((ev.day>=ctx['anchor'])&(ev.day<ctx['anchor']+7)&(ev.action==2)).sum())
    assert len(f)==expected
    counts=f.groupby('i').size().sort_values(ascending=False,kind='stable')
    pairs=f.drop_duplicates(['u','i']);daily=f.groupby(['day','subdomain']).size().rename('click_events').reset_index()
    daily.to_csv(ctx['destination']/'daily_surface_clicks.csv',index=False)
    surfaces=f.groupby('subdomain').agg(click_events=('i','size'),users=('u','nunique'),items=('i','nunique')).reset_index()
    surfaces['event_fraction']=surfaces.click_events/len(f)
    surfaces.to_csv(ctx['destination']/'surface_events.csv',index=False)
    first=f.sort_values(['ts','i','subdomain']).drop_duplicates(['u','i'])[['u','i','subdomain']]
    labeled=ctx['target'].merge(first,on=['u','i'],validate='one_to_one')
    assert len(labeled)==len(ctx['target'])
    for method in ctx['methods']:
        g=decompose(labeled,labeled['hit_'+method],'subdomain');g['method']=method
        path=ctx['destination']/'first_click_surface_contributions.csv'
        g.to_csv(path,index=False,mode='w' if method==next(iter(ctx['methods'])) else 'a',header=method==next(iter(ctx['methods'])))
    concentration=[]
    for k in [1,10,50,200]:
        head=set(counts.head(k).index)
        concentration.append(dict(k=k,future_click_event_fraction=float(f.i.isin(head).mean()),
            future_user_item_target_fraction=float(pairs.i.isin(head).mean()),
            reachable_macro_target_mass=float(ctx['target'].loc[ctx['target'].i.isin(head),'macro_weight'].sum())))
    pd.DataFrame(concentration).to_csv(ctx['destination']/'future_head_concentration.csv',index=False)
    heads=[]
    for item,count in counts.head(20).items():
        g=f[f.i==item];per_user=g.groupby('u').size().sort_values(ascending=False);surface=g.subdomain.value_counts()
        heads.append(dict(i=int(item),click_events=int(count),unique_users=len(per_user),max_user_fraction=float(per_user.iloc[0]/count),
            top5_users_fraction=float(per_user.head(5).sum()/count),dominant_surface=str(surface.index[0]),dominant_surface_fraction=float(surface.iloc[0]/count)))
    pd.DataFrame(heads).to_csv(ctx['destination']/'head_item_audit.csv',index=False)
    save_json(ctx['destination']/'drift_manifest.json',dict(click_events=len(f),unique_user_item_targets=len(pairs),
        users=f.u.nunique(),target_items=f.i.nunique(),interpretation='Future popularity is ex-post description, not a deployable predictor.',
        surface_note='Raw log labels; no dedicated campaign indicator. First-click attribution is descriptive, not causal.',
        events_removed_for_anomalies=0))
