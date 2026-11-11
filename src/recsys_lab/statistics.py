import numpy as np

def paired_bootstrap(a, b, repetitions=2000, seed=2026):
    """Mean(b-a), paired by row; exclude rows with either metric undefined."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError('Paired arrays must have equal shapes')
    difference = (b-a)[np.isfinite(a) & np.isfinite(b)]
    if not len(difference):
        raise ValueError('No finite paired observations')
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(difference, len(difference), replace=True).mean()
                      for _ in range(repetitions)])
    lo, hi = np.quantile(means, [.025, .975])
    return dict(delta=float(difference.mean()), low95=float(lo), high95=float(hi),
                paired_users=len(difference), repetitions=repetitions, seed=seed)
