"""E3 inputs: one first click per (user,item,day), 24h-gap causal prefixes."""
import hashlib
import json
import time
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from recsys_lab.data import DAY_US, save_json
from recsys_lab.prefix_sequence import prefix_arrays


def main():
    root = Path(__file__).resolve().parent
    dest = root/'runs/prefix_sequence_20261104/data'
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest/'complete.json').exists(), 'Keep completed inputs immutable'
    started = time.perf_counter()
    users = np.load(root/'runs/active_history/data/processed/users.npy')
    n_items = json.loads((root/'runs/active_history/artifacts/dataset_summary.json').read_text())['items']
    c = duckdb.connect(); c.execute("SET memory_limit='800MB'"); c.execute('SET threads=2')
    c.read_parquet(str(root/'runs/windows_20261103/events.parquet')).create_view('events')
    first = c.execute('SELECT i,min(day) AS first_day FROM events WHERE day>=1202 AND i>=0 GROUP BY i').df()
    first_day = np.full(n_items, 99999, np.int32)
    first_day[first.i] = first.first_day
    assert (first_day < 1292).all()
    np.save(dest/'first_day.npy', first_day)
    raw_count = c.execute('SELECT count(*) FROM events WHERE day>=1235 AND day<1292 AND action=2 AND i>=0').fetchone()[0]
    target = c.execute('''SELECT user_id,i AS positive,day,min(ts) AS ts,count(*) AS original_clicks
        FROM events WHERE day>=1235 AND day<1292 AND action=2 AND i>=0
        GROUP BY user_id,i,day ORDER BY user_id,ts,positive''').df()
    deduplicated = len(target)
    target['cutoff'] = target.ts-DAY_US
    target['cutoff_day'] = target.cutoff//DAY_US
    target['first_day'] = first_day[target.positive.to_numpy()]
    eligible = target.first_day < target.cutoff_day
    outside_catalog = int((~eligible).sum())
    target = target.loc[eligible].reset_index(drop=True)
    target['u'] = pd.Index(users).get_indexer(target.user_id).astype(np.int32)
    assert (target.u >= 0).all()
    f = c.execute('''SELECT user_id,i,ts,action FROM events WHERE action>1 AND i>=0
        ORDER BY user_id,ts,i,action''').df()
    counts = c.execute('''SELECT day,i,count(*) AS clicks FROM events
        WHERE day>=1202 AND day<1292 AND action=2 AND i>=0 GROUP BY day,i ORDER BY day,i''').df()
    counts.to_parquet(dest/'proposal_daily_clicks.parquet', index=False)
    c.close()
    h = open_memmap(dest/'history100.npy', mode='w+', dtype=np.int32, shape=(len(target), 100))
    a = open_memmap(dest/'actions100.npy', mode='w+', dtype=np.uint8, shape=h.shape)
    earliest = np.zeros(len(target), np.int64); latest = np.zeros_like(earliest)
    n_before = np.zeros(len(target), np.int32)
    target_groups = target.groupby('user_id', sort=False).indices
    for group_index, (uid, group) in enumerate(f.groupby('user_id', sort=False)):
        rows = target_groups.get(uid)
        if rows is None:
            continue
        values = prefix_arrays(group.ts.to_numpy(), group.i.to_numpy(), group.action.to_numpy(),
                               target.ts.to_numpy()[rows], length=100)
        h[rows], a[rows] = values[:2]
        earliest[rows], latest[rows], n_before[rows] = values[3:]
        if group_index and group_index % 20000 == 0:
            print('PREFIX_USERS', group_index, flush=True)
    h.flush(); a.flush()
    target['earliest_history_ts'] = earliest
    target['latest_history_ts'] = latest
    target['tokens_before_tail'] = n_before
    target['tokens100'] = (h > 0).sum(1).astype(np.int16)
    target['label_in50'] = False; target['label_in100'] = False
    # Last50 from the right-padded variable-length prefix, not its first50 columns.
    for lo in range(0, len(target), 8192):
        hi = min(lo+8192, len(target)); block = h[lo:hi]
        count = (block > 0).sum(1)
        own = block == (target.positive.to_numpy()[lo:hi, None]+1)
        target.loc[lo:hi-1, 'label_in100'] = own.any(1)
        target.loc[lo:hi-1, 'label_in50'] = (own & (np.arange(100)[None, :] >= count[:, None]-50)).any(1)
    nonempty = target.tokens100.gt(0)
    assert (target.loc[nonempty, 'latest_history_ts'] < target.loc[nonempty, 'cutoff']).all()
    assert (target.loc[nonempty, 'earliest_history_ts'] >= target.loc[nonempty, 'cutoff']-60*DAY_US).all()
    assert target.ts.max() < 1292*DAY_US
    assert not target.duplicated(['u','positive','day']).any()
    target.to_parquet(dest/'prefix_manifest.parquet', index=False)
    keep = np.flatnonzero(nonempty)
    np.save(dest/'kept_rows.npy', keep)
    # Other labels from the same user's target day are masked, not fed as features.
    order = np.lexsort((target.positive.to_numpy(), target.day.to_numpy(), target.u.to_numpy()))
    keys = target.u.to_numpy(dtype=np.int64)*2000+target.day.to_numpy()
    unique, starts, sizes = np.unique(keys[order], return_index=True, return_counts=True)
    np.savez_compressed(dest/'known_day_positives.npz', keys=unique, starts=starts,
                        sizes=sizes, items=target.positive.to_numpy()[order])
    digest = hashlib.sha256()
    for path in ['history100.npy','actions100.npy','prefix_manifest.parquet','first_day.npy','known_day_positives.npz']:
        with (dest/path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
    summary = dict(raw_in_catalog_clicks=raw_count, unique_user_item_days=deduplicated,
        ineligible_at_cutoff=outside_catalog, prefixes_including_empty=len(target), nonempty_prefixes=len(keep),
        empty_prefixes=int((~nonempty).sum()), training_users=int(target.loc[nonempty,'u'].nunique()),
        positive_items=int(target.loc[nonempty,'positive'].nunique()),
        cutoff_days=[int(target.cutoff_day.min()), int(target.cutoff_day.max())],
        target_days=[1235,1291], history_days=60, gap_hours=24, materialized_length=100,
        longer_than50=int((target.tokens_before_tail>50).sum()),
        label_retained50=float(target.loc[nonempty,'label_in50'].mean()),
        label_retained100=float(target.loc[nonempty,'label_in100'].mean()),
        inputs_sha256=digest.hexdigest(), seconds=time.perf_counter()-started,
        catalog_policy='Original catalog; first observed complete day strictly before exact cutoff day',
        supervision='First click per user/item/day, unit row weight; no validation labels',
        cohort_caveat='Same retrospective cohort selected1202-1261; early examples are not a prospective cohort simulation',
        final_test_used=False)
    save_json(dest/'complete.json', summary)
    print('PREFIX_INPUTS_COMPLETE', summary, flush=True)


if __name__ == '__main__':
    main()
