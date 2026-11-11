import json
import time
import numpy as np
import pandas as pd
from .data import ROOT, save_json
from .baselines import record, evaluation_users


def combine(a, b, quota_a):
    """Фиксированный weighted round-robin; нулевая квота отключает канал."""
    k = a.shape[1]
    if quota_a == k:
        return a.copy()
    if quota_a == 0:
        return b.copy()
    result = np.full_like(a, -1)
    schedule = [
        0 if (j * quota_a) // k != ((j - 1) * quota_a) // k else 1
        for j in range(1, k + 1)
    ]
    for u in range(len(a)):
        if not (a[u] >= 0).any() and not (b[u] >= 0).any():
            continue
        lists = [a[u], b[u]]
        pointers = [0, 0]
        seen = set()
        out = []
        for channel in schedule:
            for source in [channel, 1 - channel]:
                while pointers[source] < k:
                    item = int(lists[source][pointers[source]])
                    pointers[source] += 1
                    if item >= 0 and item not in seen:
                        seen.add(item)
                        out.append(item)
                        break
                else:
                    continue
                break
        result[u, : len(out)] = out
    return result


def run_fusion(data, names):
    summaries = {
        n: json.loads((ROOT / "artifacts" / n / "metrics.json").read_text())
        for n in names
    }
    best = max(names, key=lambda n: summaries[n]["recall_reachable@200"])
    outputs = {
        n: np.load(ROOT / "artifacts" / n / "recommendations.npy") for n in names
    }
    raw = {n: np.load(ROOT / "artifacts" / n / "raw.npy") for n in names}
    diagnostics = []
    for candidate in names:
        if candidate == best:
            continue
        additional = []
        for u in evaluation_users(data):
            y = {i for i in data.targets[u] if isinstance(i, int)}
            if y:
                additional.append(
                    len((set(outputs[candidate][u]) - set(outputs[best][u])) & y)
                    / len(y)
                )
        diagnostics.append(
            {"channel": candidate, "extra_target_fraction": float(np.mean(additional))}
        )
    other = max(diagnostics, key=lambda row: row["extra_target_fraction"])["channel"]
    save_json(
        ROOT / "artifacts/fusion_selection.json",
        {
            "first": best,
            "second": other,
            "candidates": diagnostics,
            "status": "preliminary validation selection, no final test",
        },
    )
    rows = []
    for quota in [0, 50, 100, 150, 200]:
        start = time.perf_counter()
        merged = combine(raw[best], raw[other], quota)
        merge_seconds = time.perf_counter() - start
        active = [other] if quota == 0 else [best] if quota == 200 else [best, other]
        r = record(
            data,
            f"fusion_{quota}",
            merged,
            sum(summaries[n]["fit_seconds"] for n in active),
            sum(summaries[n]["predict_seconds"] for n in active) + merge_seconds,
            {
                "channel_a": best,
                "channel_b": other,
                "quota_a": quota,
                "quota_b": 200 - quota,
                "merge_seconds": merge_seconds,
                "timing_note": "conservative sum of both channels; endpoints use only active channel",
            },
        )
        if quota in (0, 200):
            active = other if quota == 0 else best
            np.testing.assert_array_equal(
                np.load(ROOT / "artifacts" / f"fusion_{quota}" / "recommendations.npy"),
                outputs[active],
            )
        rows.append(r)
    return pd.DataFrame(rows)


def paired_bootstrap(data, name_a, name_b, n_boot=1000):
    a = pd.read_parquet(ROOT / "artifacts" / name_a / "per_user.parquet")
    b = pd.read_parquet(ROOT / "artifacts" / name_b / "per_user.parquet")
    delta = (a["recall_reachable@200"] - b["recall_reachable@200"]).dropna().to_numpy()
    rng = np.random.default_rng(data.cfg["seed"])
    means = np.array(
        [rng.choice(delta, len(delta), replace=True).mean() for _ in range(n_boot)]
    )
    result = {
        "a": name_a,
        "b": name_b,
        "mean_difference": float(delta.mean()),
        "ci95": [float(v) for v in np.quantile(means, [0.025, 0.975])],
        "users": len(delta),
        "bootstrap_samples": n_boot,
        "limitation": "one validation window; validation selection optimism; not training seed stability",
    }
    save_json(ROOT / "artifacts/bootstrap.json", result)
    return result
