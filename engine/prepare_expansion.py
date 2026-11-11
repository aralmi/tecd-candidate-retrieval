"""Broaden training users only, holding evaluation cohort, C and history fixed."""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from recsys_lab.data import stable_order


def build_pairs(events, selected, first_day, anchors, output):
    # events already deduplicated and sorted by user,timestamp,item,action.
    user_index=pd.Index(selected)
    events['u']=user_index.get_indexer(events.user_id).astype(np.int32)
    histories=[];actions=[];positives=[];users=[];moments=[];excluded=[];offsets=[0]
    details=[]
    for anchor in anchors:
        f=events[(events.day>=anchor-61)&(events.day<anchor-1)]
        same_next=(f.u==f.u.shift(-1))&(f.i==f.i.shift(-1))&(f.action==f.action.shift(-1))
        f=f.loc[~same_next].groupby('u',sort=False).tail(50)
        h=np.zeros((len(selected),50),np.int32);a=np.zeros((len(selected),50),np.uint8)
        for u,g in f.groupby('u',sort=False):
            n=len(g);h[u,:n]=g.i.to_numpy()+1;a[u,:n]=g.action.to_numpy()
        target=events[(events.day>=anchor)&(events.day<anchor+7)&(events.action==2)]
        target=target[first_day[target.i.to_numpy()]<anchor-1]
        target=target.drop_duplicates(['u','i']).sort_values(['u','i'])
        before=len(positives);train_users=0
        for u,g in target.groupby('u',sort=False):
            if not h[u].any():continue
            train_users+=1
            good=g.i.to_numpy(dtype=np.int32)
            for pos in good:
                histories.append(h[u].copy());actions.append(a[u].copy());positives.append(pos)
                users.append(u);moments.append(anchor)
                excluded.extend(good);offsets.append(len(excluded))
        details.append(dict(anchor=anchor,pairs=len(positives)-before,training_users=train_users))
        print('ANCHOR',details[-1],flush=True)
    result=dict(h=np.asarray(histories,np.int32),actions=np.asarray(actions,np.uint8),
                positive=np.asarray(positives,np.int32),users=np.asarray(users,np.int32),
                anchors=np.asarray(moments,np.int16),excluded_flat=np.asarray(excluded,np.int32),
                excluded_offsets=np.asarray(offsets,np.int64))
    np.savez_compressed(output,**result)
    return dict(pairs=len(positives),training_users=len(np.unique(users)),
        positive_items=len(np.unique(positives)),anchors=details,
        old_cohort_pairs=int((np.asarray(users)<80018).sum()),
        old_cohort_training_users=int(len(np.unique(np.asarray(users)[np.asarray(users)<80018]))))


def main(root,cap):
    root=Path(root);dest=root/'runs/expansion_20261103'
    dest.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    old=root/'runs/active_history'
    cfg=json.loads((old/'configs/pilot.json').read_text())
    old_users=np.load(old/'data/processed/users.npy')
    c=duckdb.connect();c.execute("SET memory_limit='2GB'");c.execute('SET threads=3')
    files=[str(root/f'data/raw/events/{d:05}.pq') for d in range(1202,1292)]
    c.read_parquet(files).create_view('raw_events')
    c.read_parquet(str(old/'data/processed/items.parquet')).create_view('catalog')
    c.read_parquet(str(old/'data/processed/events.parquet')).create_view('old_events')
    c.register('old_users',pd.DataFrame({'user_id':old_users}))
    c.register('anchor_source',pd.DataFrame({'anchor':cfg['train_anchors']}))
    c.execute('CREATE TABLE items AS SELECT item_id,row_number() OVER(ORDER BY item_id)-1 AS i FROM catalog')
    c.execute('CREATE TABLE first_seen AS SELECT e.item_id,min(e.day) AS first_day FROM old_events e SEMI JOIN items USING(item_id) GROUP BY item_id')
    c.execute('''CREATE VIEW events AS SELECT user_id,timestamp AS ts,
        (timestamp//86400000000)::SMALLINT AS day,i.i::INTEGER AS i,
        CASE action_type WHEN 'view' THEN 1 WHEN 'click' THEN 2 WHEN 'like' THEN 3 WHEN 'clickout' THEN 4 END::TINYINT AS action,
        f.first_day FROM raw_events e JOIN items i USING(item_id) JOIN first_seen f USING(item_id)''')
    c.execute('''CREATE TABLE targets AS SELECT DISTINCT a.anchor,e.user_id,e.i
        FROM events e JOIN anchor_source a ON e.day>=a.anchor AND e.day<a.anchor+7
        WHERE e.action=2 AND e.first_day<a.anchor-1''')
    c.execute('''CREATE TABLE histories AS SELECT DISTINCT a.anchor,e.user_id
        FROM events e JOIN anchor_source a ON e.day>=a.anchor-61 AND e.day<a.anchor-1
        SEMI JOIN targets t ON e.user_id=t.user_id AND a.anchor=t.anchor''')
    c.execute('CREATE TABLE eligible_pairs AS SELECT t.* FROM targets t SEMI JOIN histories h USING(anchor,user_id)')
    eligible=c.execute('SELECT DISTINCT user_id FROM eligible_pairs ORDER BY user_id').fetchnumpy()['user_id']
    extra=eligible[~np.isin(eligible,old_users)]
    extra=extra[stable_order(extra,2026)[:max(0,cap-len(old_users))]]
    selected=np.r_[old_users,extra]
    c.register('selected',pd.DataFrame({'user_id':selected}))
    counts=c.execute('''SELECT count(*) AS pairs,count(DISTINCT user_id) AS training_users,
        count(DISTINCT i) AS positive_items FROM eligible_pairs SEMI JOIN selected USING(user_id)''').df().iloc[0].to_dict()
    original=c.execute('SELECT count(*) FROM eligible_pairs SEMI JOIN old_users USING(user_id)').fetchone()[0]
    assert original==122105,(original,'Original pair population mismatch')
    manifest=dict(selected_cohort=len(selected),selected_extra_users=len(extra),
        eligible_training_users_full_population=len(eligible),original_cohort_pair_count=original,
        cap=cap,counts={k:int(v) for k,v in counts.items()},
        selection='Keep all old evaluation users; add deterministic sample of users with nonempty historical profile and allowed training click targets; no validation labels.',
        catalog='Fixed original 1060869 items and original historical first_seen cutoff',
        periods=cfg,final_test_used=False)
    (dest/'population_counts.json').write_text(json.dumps(manifest,indent=2))
    print('POPULATION_COUNTS',json.dumps(manifest),flush=True)
    np.save(dest/'training_users.npy',selected)
    # Column-pruned export: retain every action, remove only exact duplicates; no arbitrary event cap.
    p=dest/'events.parquet'
    c.execute('''COPY(SELECT DISTINCT user_id,ts,day,i,action FROM events
        SEMI JOIN selected USING(user_id) ORDER BY user_id,ts,i,action) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)''',[str(p)])
    first=c.execute('SELECT i.i,f.first_day FROM items i JOIN first_seen f USING(item_id) ORDER BY i.i').df()
    first_day=first.first_day.to_numpy(dtype=np.int32)
    np.save(dest/'first_day.npy',first_day)
    c.close()
    events=pd.read_parquet(p)
    manifest['events']=len(events)
    manifest['actions']={str(k):int(v) for k,v in events.action.value_counts().items()}
    pairs=build_pairs(events,selected,first_day,cfg['train_anchors'],dest/'pairs_4_windows.npz')
    assert pairs['old_cohort_pairs']==122105,pairs
    assert pairs['old_cohort_training_users']==25827,pairs
    manifest['generated']=pairs;manifest['seconds']=time.perf_counter()-start
    manifest['evaluation_users_unchanged']=True
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('EXPANSION_READY',json.dumps(manifest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=Path(__file__).resolve().parent)
    p.add_argument('--cap',type=int,default=200000);a=p.parse_args();main(a.root,a.cap)
