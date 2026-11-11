"""Fixed mixed50 input policy; positive rows and evaluation population stay fixed."""
import hashlib

import numpy as np
import pandas as pd

from .data import DAY_US, ROOT, save_json


def select_mixed_frame(events, anchor, days=60, length=50):
    """Last25 strong + last25 views, transfer unused capacity, retain time order.

    The caller supplies events sorted by user/time/item/action, as in PilotData.
    Consecutive identical user/item/action events collapse to the last event BEFORE
    action strata are selected. The gap day and all future events are excluded.
    """
    assert length == 50, 'This experiment fixes 25/25; no quota search'
    frame = events.loc[
        (events.day >= anchor - days - 1)
        & (events.day < anchor - 1) & (events.i >= 0),
        ['u', 'i', 'ts', 'day', 'action'],
    ].reset_index(drop=True)
    same_next = ((frame.u == frame.u.shift(-1)) & (frame.i == frame.i.shift(-1))
                 & (frame.action == frame.action.shift(-1)))
    frame = frame.loc[~same_next].copy()
    rank = frame.groupby(['u', frame.action > 1], sort=False).cumcount(ascending=False)
    chosen = frame.loc[rank < 25]
    free = 50 - chosen.groupby('u', sort=False).size()
    rest = frame.loc[rank >= 25]
    extra_rank = rest.groupby('u', sort=False).cumcount(ascending=False)
    extra = rest.loc[extra_rank < rest.u.map(free)]
    result = pd.concat([chosen, extra]).sort_index()
    assert len(result) == 0 or result.groupby('u').size().max() <= 50
    assert (result.day < anchor - 1).all()
    return result


def mixed_history(data, anchor):
    frame = select_mixed_frame(data.events, anchor, data.cfg['history_days'])
    h = np.zeros((data.n_users, 50), np.int64)
    actions = np.zeros_like(h)
    ages = np.zeros(h.shape, np.float32)
    for u, group in frame.groupby('u', sort=False):
        n = len(group)
        h[u, :n] = group.i.to_numpy() + 1
        actions[u, :n] = group.action.to_numpy()
        ages[u, :n] = ((anchor - 1) * DAY_US - group.ts.to_numpy()) / DAY_US
    return h, actions, ages


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def replace_histories(data, ctx):
    """Change only input histories, retaining every original diagnostic pair."""
    pairs = ctx['pairs']
    invariants = {k: array_hash(pairs[k]) for k in ['positive', 'labels', 'anchors', 'users']}
    forbidden_hash = hashlib.sha256(b''.join(np.asarray(x, np.int32).tobytes()
                                          for x in ctx['forbidden'])).hexdigest()
    old_h = pairs['h'].copy()
    records = []
    for anchor in data.cfg['train_anchors']:
        rows = np.flatnonzero(pairs['anchors'] == anchor)
        h, a, _ = mixed_history(data, anchor)
        users = pairs['users'][rows]
        pairs['h'][rows] = h[users]
        pairs['actions'][rows] = a[users]
        records.append(dict(anchor=anchor, pairs=len(rows),
                            changed_fraction=float(np.any(old_h[rows] != h[users], axis=1).mean()),
                            empty_rows=int((~h[users].any(1)).sum()),
                            view_tokens=int((a[users] == 1).sum())))
    strong_query = (data.h.copy(), data.actions.copy(), data.ages.copy())
    data.h, data.actions, data.ages = mixed_history(data, data.cfg['validation_anchor'])
    for key, value in invariants.items():
        assert array_hash(pairs[key]) == value, key
    assert hashlib.sha256(b''.join(np.asarray(x, np.int32).tobytes()
                                 for x in ctx['forbidden'])).hexdigest() == forbidden_hash
    audit = dict(pair_rows=len(pairs['positive']), pair_hashes=invariants,
                 known_positive_mask_sha256=forbidden_hash, anchors=records,
                 query_users=array_hash(ctx['queries']),
                 history_policy='25 strong +25 views in past60 days, transfer spare slots, last event of consecutive duplicate run, chronological',
                 empty_pairs_retained=True, query_population_unchanged=True,
                 positives_and_masks_unchanged=True, gap_excluded=True,
                 weights_unchanged={'view': .1, 'click': 1., 'like': 2., 'clickout': 3.})
    ctx['manifest'] = dict(ctx['manifest'], history=audit['history_policy'])
    data.cfg['history_mode'] = 'mixed25_25'
    save_json(ROOT / 'mixed_history_audit.json', audit)
    np.savez_compressed(ROOT / 'mixed_training_inputs.npz', h=pairs['h'], actions=pairs['actions'],
                        **{k:pairs[k] for k in ['positive','labels','anchors','users']})
    print('MIXED_INPUTS_VERIFIED', records, flush=True)
    return strong_query
