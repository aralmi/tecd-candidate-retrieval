"""Add four earlier label windows while keeping the old population and evaluation."""
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from prepare_expansion import build_pairs


root=Path(__file__).resolve().parent
dest=root/'runs/windows_20261103';dest.mkdir(parents=True,exist_ok=True)
old=root/'runs/active_history';start=time.perf_counter()
users=np.load(old/'data/processed/users.npy')
c=duckdb.connect();c.execute("SET memory_limit='2GB'");c.execute('SET threads=3')
c.register('selected',pd.DataFrame({'user_id':users}))
c.read_parquet(str(old/'data/processed/items.parquet')).create_view('catalog')
c.execute('CREATE TABLE items AS SELECT item_id,row_number() OVER(ORDER BY item_id)-1 AS i FROM catalog')
files=[str(root/f'data/raw/events/{d:05}.pq') for d in range(1174,1292)]
assert all(Path(f).exists() for f in files)
c.read_parquet(files).create_view('raw_events')
p=dest/'events.parquet'
c.execute('''COPY(SELECT DISTINCT e.user_id,timestamp AS ts,
    (timestamp//86400000000)::SMALLINT AS day,i.i::INTEGER AS i,
    CASE action_type WHEN 'view' THEN 1 WHEN 'click' THEN 2 WHEN 'like' THEN 3 WHEN 'clickout' THEN 4 END::TINYINT AS action
    FROM raw_events e SEMI JOIN selected USING(user_id) JOIN items i USING(item_id)
    ORDER BY user_id,ts,i,action) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)''',[str(p)])
c.close();events=pd.read_parquet(p)
anchors=[1235,1242,1249,1256,1263,1270,1277,1284]
first_day=np.load(root/'runs/expansion_20261103/first_day.npy')
summary=build_pairs(events,users,first_day,anchors,dest/'pairs_8_windows.npz')
old_counts={1263:33726,1270:31444,1277:31713,1284:25222}
for row in summary['anchors']:
    if row['anchor'] in old_counts:assert row['pairs']==old_counts[row['anchor']]
summary.update(events=len(events),selected_users=len(users),seconds=time.perf_counter()-start,
    extra_history_start=1174,evaluation_history_changed=False,
    historical_catalog='Original fixed C and first_day; conservative availability remains fixed across experiments.',
    selection_note='Evaluation population fixed from past 1202-1261; earlier supervision uses these fixed users. No online-sampling claim for earlier anchors.',
    final_test_used=False)
(dest/'manifest.json').write_text(json.dumps(summary,indent=2));print('WINDOWS_READY',json.dumps(summary),flush=True)
