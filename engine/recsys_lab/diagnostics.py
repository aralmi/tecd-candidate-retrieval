"""Разрезы из готовых выдач: без новых обучений и изменения каталога."""

import json
import numpy as np
import pandas as pd
from .data import ROOT, PilotData, save_json
from .baselines import evaluation_users


def run(data):
    anchor = data.cfg["validation_anchor"]
    past = data.events[
        (data.events.day >= anchor - data.cfg["history_days"] - 1)
        & (data.events.day < anchor - 1)
    ]
    strong = (
        past[past.action != 1]
        .groupby("u")
        .item_id.nunique()
        .reindex(range(data.n_users), fill_value=0)
        .to_numpy()
    )
    group = np.select(
        [strong == 0, strong <= 5, strong <= 20], ["0", "1–5", "6–20"], default=">20"
    )
    seen = {int(u): set(g.item_id) for u, g in past.groupby("u", sort=False)}
    known = [
        set(data.item_map.get(item, "outside:" + item) for item in seen.get(u, set()))
        for u in range(data.n_users)
    ]
    repeated = [ys & known[u] for u, ys in enumerate(data.targets)]
    novel = [ys - known[u] for u, ys in enumerate(data.targets)]
    future = data.events[
        (data.events.day >= anchor)
        & (data.events.day < anchor + 7)
        & (data.events.action == 2)
    ]
    by_surface = {}
    for surface, frame in future.groupby("subdomain"):
        targets = [set() for _ in range(data.n_users)]
        for u, g in frame.groupby("u", sort=False):
            targets[u] = {
                data.item_map.get(item, "outside:" + item)
                for item in g.item_id.unique()
            }
        by_surface[str(surface)] = targets
    rows = []
    groups = []
    metrics = {}
    paths = sorted((ROOT / "artifacts").glob("*/metrics.json"))
    for path in paths:
        name = path.parent.name
        m = json.loads(path.read_text())
        if m.get("fingerprint") != data.fingerprint:
            raise ValueError("mixed cohorts")
        predictions = np.load(path.parent / "recommendations.npy", mmap_mode="r")
        per_user = pd.read_parquet(path.parent / "per_user.parquet")
        metrics[name] = per_user["recall_reachable@200"].to_numpy()
        for label in ["0", "1–5", "6–20", ">20"]:
            mask = group == label
            subset = per_user.loc[mask]
            groups.append(
                {
                    "method": name,
                    "strong_history_items": label,
                    "cohort_users": int(mask.sum()),
                    "users_with_targets": int((subset.n_targets > 0).sum()),
                    "users_with_reachable": int((subset.n_reachable > 0).sum()),
                    "recall@200": subset["recall_reachable@200"].mean(),
                }
            )
        for label, targets in [
            ("repeated", repeated),
            ("new_to_user", novel),
            *[(f"subdomain:{s}", t) for s, t in by_surface.items()],
        ]:
            recalls = []
            n_target = 0
            reachable_targets = 0
            for u in evaluation_users(data):
                n_target += len(targets[u])
                reachable = {i for i in targets[u] if isinstance(i, int)}
                reachable_targets += len(reachable)
                if reachable:
                    recalls.append(
                        len(set(predictions[u]) & reachable) / len(reachable)
                    )
            rows.append(
                {
                    "method": name,
                    "slice": label,
                    "users_with_reachable": len(recalls),
                    "targets": n_target,
                    "reachable_targets": reachable_targets,
                    "recall@200": float(np.mean(recalls)) if recalls else None,
                }
            )
    dest = ROOT / "artifacts"
    pd.DataFrame(groups).to_csv(dest / "history_groups.csv", index=False)
    pd.DataFrame(rows).to_csv(dest / "target_slices.csv", index=False)
    comparisons = []
    for a, b in [("tt_content", "tt_id"), ("attention", "tt_content")]:
        if a not in metrics or b not in metrics:
            continue
        diff = metrics[a] - metrics[b]
        diff = diff[np.isfinite(diff)]
        rng = np.random.default_rng(data.cfg["seed"])
        means = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(1000)]
        comparisons.append(
            {
                "a": a,
                "b": b,
                "difference": float(diff.mean()),
                "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
                "users": len(diff),
            }
        )
    save_json(
        dest / "ablation_intervals.json",
        {
            "comparisons": comparisons,
            "limitation": "single seed and validation selected checkpoints; no multiplicity correction",
        },
    )
    print(
        "DIAGNOSTICS_COMPLETE",
        len(paths),
        "methods",
        len(groups),
        "group rows",
        len(rows),
        "target-slice rows",
        flush=True,
    )
    return pd.DataFrame(groups), pd.DataFrame(rows)


if __name__ == "__main__":
    run(PilotData())
