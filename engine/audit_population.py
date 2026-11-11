"""Аудит активности исключительно ДО первого обучающего момента."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa

root=Path(__file__).resolve().parent
frames=[];start=time.perf_counter()
for day in range(1202,1262):
    t=pq.read_table(root/f'data/raw/events/{day:05}.pq',columns=['user_id','item_id','action_type'])
    t=t.filter(pc.is_in(t['action_type'],value_set=pa.array(['click','like','clickout'],type=pa.large_string())))
    f=t.to_pandas();f['day']=day;frames.append(f)
events=pd.concat(frames,ignore_index=True)
stats=events.groupby('user_id').agg(strong_events=('item_id','size'),strong_items=('item_id','nunique'),
                                  strong_days=('day','nunique'),last_strong_day=('day','max'))
stats.to_parquet(root/'data/past_user_activity.parquet')
counts=[]
for items in [1,3,5,10,20]:
    for days in [1,2,3]:
        mask=(stats.strong_items>=items)&(stats.strong_days>=days)
        counts.append({'min_items':items,'min_days':days,'users':int(mask.sum()),
                       'past_strong_events':int(stats.loc[mask,'strong_events'].sum())})
result={'period':[1202,1261],'strong_users':len(stats),'strong_events':len(events),
        'candidate_rules':counts,'seconds':time.perf_counter()-start,'future_used':False}
(root/'artifacts/past_population_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
