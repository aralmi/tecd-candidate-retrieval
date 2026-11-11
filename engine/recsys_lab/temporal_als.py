"""Frozen ALS recipes for rolling windows; all item IDs here are window-local."""
import numpy as np
import pandas as pd
from scipy import sparse
from .data import STRENGTH


def past_window(events, anchor):
    end = anchor-1
    return events[(events.day >= end-90)&(events.day < end)&(events.i >= 0)]


def training_confidence(past, n_users, n_items):
    frame = past[['u','i','day','action']].copy()
    frame['weight'] = STRENGTH[frame.action.to_numpy()]
    daily = frame.groupby(['u','i','day'],sort=False).weight.max()
    total = daily.groupby(level=[0,1]).sum()
    values = 1+20*np.log1p(total.values).astype(np.float32)
    return sparse.csr_matrix((values,(total.index.get_level_values(0),
                                     total.index.get_level_values(1))),shape=(n_users,n_items))


def full_query_confidence(past, anchor, n_users, n_items):
    # Filter BEFORE collapsing: future continuation of a run cannot remove its past.
    f = past[(past.day >= anchor-61)&(past.day < anchor-1)]
    same = (f.u==f.u.shift(-1))&(f.i==f.i.shift(-1))&(f.action==f.action.shift(-1))
    f = f.loc[~same]
    matrix = sparse.csr_matrix((STRENGTH[f.action.to_numpy()],(f.u,f.i)),shape=(n_users,n_items))
    matrix.data = 1+20*np.log1p(matrix.data)
    return matrix


def target_sets(future, local_to_global, n_users):
    local = pd.Index(local_to_global)
    positions = local.get_indexer(future.i)
    result = [set() for _ in range(n_users)]
    for row, position in zip(future.itertuples(),positions):
        if position >= 0:
            item = int(position)
        else:
            key = row.i if row.i >= 0 else row.outside_id
            item = 'outside:'+str(key)
        result[int(row.u)].add(item)
    return result
