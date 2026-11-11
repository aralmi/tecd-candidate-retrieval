"""Replace histories only; preserve every original training positive row."""
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent;start=time.perf_counter()
dest=root/'runs/history_20261103';dest.mkdir(parents=True,exist_ok=True)
users=np.load(root/'runs/active_history/data/processed/users.npy')
c=duckdb.connect();c.execute("SET memory_limit='1GB'");c.execute('SET threads=2')
c.register('selected',pd.DataFrame({'user_id':users}))
c.read_parquet(str(root/'runs/expansion_20261103/events.parquet')).create_view('events')
events=c.execute('SELECT * FROM events SEMI JOIN selected USING(user_id) WHERE action>1 ORDER BY user_id,ts,i,action').df();c.close()
events['u']=pd.Index(users).get_indexer(events.user_id).astype(np.int32)
with np.load(root/'runs/expansion_20261103/pairs_4_windows.npz',allow_pickle=False) as z:
    chosen=np.flatnonzero(z['users']<len(users))
    pairs={k:z[k][chosen] for k in ['h','actions','positive','users','anchors']}
    flat,off=z['excluded_flat'],z['excluded_offsets']
    groups=[flat[off[r]:off[r+1]] for r in chosen]
    excluded=np.concatenate(groups);offsets=np.r_[0,np.cumsum([len(g) for g in groups])]
assert len(pairs['positive'])==122105
for anchor in [1263,1270,1277,1284]:
    frame=events[(events.day>=anchor-61)&(events.day<anchor-1)]
    same_next=(frame.u==frame.u.shift(-1))&(frame.i==frame.i.shift(-1))&(frame.action==frame.action.shift(-1))
    frame=frame.loc[~same_next].groupby('u',sort=False).tail(50)
    h=np.zeros((len(users),50),np.int32);a=np.zeros((len(users),50),np.uint8)
    for u,g in frame.groupby('u',sort=False):
        n=len(g);h[u,:n]=g.i.to_numpy()+1;a[u,:n]=g.action.to_numpy()
    rows=np.flatnonzero(pairs['anchors']==anchor)
    pairs['h'][rows]=h[pairs['users'][rows]];pairs['actions'][rows]=a[pairs['users'][rows]]
assert set(np.unique(pairs['actions']))<={0,2,3,4}
np.savez_compressed(dest/'pairs_strong.npz',**pairs,excluded_flat=excluded,excluded_offsets=offsets)
manifest={'training_pairs':len(pairs['positive']),'empty_strong_training_histories':int((pairs['h'].sum(1)==0).sum()),
    'positives_and_row_order_preserved':True,'empty_histories_retained':True,
    'policy':'Filter view before consecutive-collapse and last50, on full60-day past.',
    'changes':'Both training history and query history; same U,C,targets and uniform sampler.',
    'seconds':time.perf_counter()-start,'final_test_used':False}
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2));print('STRONG_READY',json.dumps(manifest),flush=True)
