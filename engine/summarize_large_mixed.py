"""Collect completed scale fits, paired user uncertainty and goal contributions.

Exploratory development-set comparisons; no final-test data are read.
Safe to rerun while a fit is active: only complete.json runs enter comparisons.
"""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/large_mixed_20261105'
KEY = 'recall_reachable@200'


def paired_ci(left, right, repetitions=2000):
    assert not left.u.duplicated().any() and not right.u.duplicated().any()
    joined = left[['u', KEY]].merge(right[['u', KEY]], on='u', validate='one_to_one')
    assert len(joined) == len(left) == len(right)
    delta = (joined[KEY + '_x'] - joined[KEY + '_y']).dropna().to_numpy()
    rng = np.random.default_rng(502026)
    means = np.concatenate([
        delta[rng.integers(len(delta), size=(min(100, repetitions - start), len(delta)))].mean(1)
        for start in range(0, repetitions, 100)
    ])
    lo, hi = np.quantile(means, [.025, .975])
    return dict(delta=float(delta.mean()), low95=float(lo), high95=float(hi),
                paired_users=len(delta), bootstrap_repetitions=repetitions)


def main():
    rows, curves, users, predictions = [], [], {}, {}
    losses = []
    log_names = {
        'mixed_all_seed2026': 'all', 'mixed_old_seed2026': 'oldmixed',
        'strong_old_seed2026': 'oldstrong',
        'mixed_all_seed2026_lr0p0003': 'all_lr3e4',
        'mixed_old_seed2026_lr0p0003': 'oldmixed_lr3e4',
        'strong_all_seed2026': 'allstrong',
        'strong_all_seed2026_userweek': 'allstrong_userweek',
    }
    pattern = r'LARGE_PROGRESS \w+ (\d+) examples (\d+) loss ([\d.e+-]+) driverGiB ([\d.e+-]+)'
    for name, tag in log_names.items():
        path = Path('/private/tmp') / f'recsys_train_large_{tag}_20261105.log'
        if not path.exists() or not (RUN / name).exists():
            continue
        log = path.read_text()
        (RUN / name / 'training_stdout.log').write_text(log)
        for step, examples, loss, memory in re.findall(pattern, log):
            losses.append(dict(run=name, updates=int(step), presentations=int(examples),
                mean_loss_last_100_updates=float(loss), mps_driver_observed_gib=float(memory)))
    pd.DataFrame(losses).to_csv(RUN / 'training_loss_curves.csv', index=False)
    for config in sorted(RUN.glob('*/configuration.json')):
        folder = config.parent
        cfg = json.loads(config.read_text())
        if (folder / 'learning_curve.csv').exists():
            curve = pd.read_csv(folder / 'learning_curve.csv')
            curve['run'] = folder.name
            curves.append(curve)
        if not (folder / 'complete.json').exists():
            continue
        for score in ['raw', 'same_fixed_prior']:
            metric = json.loads((folder / f'{score}_metrics.json').read_text())
            name = folder.name + '/' + score
            users[name] = pd.read_parquet(folder / f'{score}_per_user.parquet')
            with np.load(folder / f'{score}_lists.npz') as z:
                predictions[name] = (z['query_users'], z['recommendations'])
            rows.append(dict(run=folder.name, score=score, variant=cfg['variant'], lr=cfg['lr'],
                batch_size=cfg['batch_size'], train_pairs=cfg['pairs'], train_users=cfg['training_users'],
                recall=metric[KEY], recall_all=metric['recall_all@200'],
                best_updates=metric['best_updates'], completed_updates=metric['completed_updates'],
                presentations=metric['presentations'], fraction_of_one_pass=metric['fraction_of_one_pass'],
                own_minus_shuffled=metric['own_minus_shuffled'], fit_seconds=metric['fit_seconds']))
    pd.DataFrame(rows).to_csv(RUN / 'completed_comparison.csv', index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(RUN / 'all_learning_curves.csv', index=False)

    controls = {
        'recent_mixed100': RUN / 'recent_mixed100_per_user.parquet',
        'personal_pop28': ROOT / 'runs/temporal_als_20261104/anchor_1293/personal_pop28_per_user.parquet',
    }
    for name, path in controls.items():
        assert path.exists(), path
        users[name] = pd.read_parquet(path)
    for variant in ['strong_old', 'strong_all', 'mixed_old', 'mixed_all']:
        path = RUN / f'train_target_pop_{variant}_per_user.parquet'
        if path.exists():
            users['train_target_pop_' + variant] = pd.read_parquet(path)
    comparisons = []
    for name in list(predictions):
        for reference in controls:
            comparisons.append(dict(method=name, reference=reference, **paired_ci(users[name], users[reference])))
        variant = json.loads((RUN / name.split('/')[0] / 'configuration.json').read_text())['variant']
        reference = 'train_target_pop_' + variant
        if reference in users:
            comparisons.append(dict(method=name, reference=reference, **paired_ci(users[name], users[reference])))
    for suffix in ['', '_lr0p0003']:
        for score in ['raw', 'same_fixed_prior']:
            large = f'mixed_all_seed2026{suffix}/{score}'
            small = f'mixed_old_seed2026{suffix}/{score}'
            if large in users and small in users:
                comparisons.append(dict(method=large, reference=small, **paired_ci(users[large], users[small])))
    for score in ['raw', 'same_fixed_prior']:
        large = 'strong_all_seed2026/' + score
        small = 'strong_old_seed2026/' + score
        if large in users and small in users:
            comparisons.append(dict(method=large, reference=small, **paired_ci(users[large], users[small])))
        weighted = 'strong_all_seed2026_userweek/' + score
        if weighted in users and large in users:
            comparisons.append(dict(method=weighted, reference=large, **paired_ci(users[weighted], users[large])))
    pd.DataFrame(comparisons).to_csv(RUN / 'paired_development_comparisons.csv', index=False)

    cases = pd.read_parquet(ROOT / 'runs/interests_20261104/anchor_1293/reachable_target_cases.parquet')
    mapping = np.load(ROOT / 'runs/interests_20261104/original_to_global_item.npy')
    inverse = np.full(2325409, -1, np.int32)
    inverse[mapping] = np.arange(len(mapping))
    item = inverse[cases.i.to_numpy()]
    assert (item >= 0).all() and np.isclose(cases.macro_weight.sum(), 1.)
    mixed = np.load(RUN / 'data/query_mixed_h.npy', mmap_mode='r')
    strong = np.load(RUN / 'data/query_strong_h.npy', mmap_mode='r')
    in_strong = (strong[cases.u.to_numpy()] == item[:, None] + 1).any(1)
    in_mixed = (mixed[cases.u.to_numpy()] == item[:, None] + 1).any(1)
    segments = {
        'strong_history_target': in_strong,
        'view_only_history_target': in_mixed & ~in_strong,
        'outside_model_history': ~in_mixed,
    }
    breakdown = []
    for name, (query, pred) in predictions.items():
        positions = pd.Index(query).get_indexer(cases.u)
        assert (positions >= 0).all()
        hit = (pred[positions] == item[:, None]).any(1)
        assert np.isclose((cases.macro_weight.to_numpy() * hit).sum(), users[name][KEY].mean())
        for group, mask in segments.items():
            weight = cases.macro_weight.to_numpy()[mask]
            contribution = float((weight * hit[mask]).sum())
            breakdown.append(dict(method=name, segment=group, goals=int(mask.sum()),
                goal_macro_mass=float(weight.sum()), hit_macro_contribution=contribution,
                captured_fraction_of_segment_mass=contribution / weight.sum()))
    pd.DataFrame(breakdown).to_csv(RUN / 'goal_contributions.csv', index=False)
    print(pd.DataFrame(rows)[['run', 'score', 'recall', 'best_updates', 'own_minus_shuffled']].to_string(index=False))


if __name__ == '__main__':
    main()
