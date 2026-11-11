import numpy as np
import pandas as pd


def evaluate(recommendations, targets, n_items, ks=(50, 100, 200)):
    """Макро-метрики. Пользователи без целей исключены из Recall, но посчитаны."""
    records = []
    for u, ys in enumerate(targets):
        reachable = {i for i in ys if isinstance(i, (int, np.integer)) and 0 <= i < n_items}
        row = {
            "u": u,
            "n_targets": len(ys),
            "n_reachable": len(reachable),
            "catalog_bound": len(reachable) / len(ys) if ys else np.nan,
        }
        if not ys:
            # Recall/NDCG для пустой цели не определены. Сохраняем пользователя
            # в статистике, не перебирая заново сотни нерелевантных позиций.
            for k in ks:
                for metric in ["recall_reachable", "recall_all", "hr", "ndcg"]:
                    row[f"{metric}@{k}"] = np.nan
            records.append(row)
            continue
        for k in ks:
            ranking = [int(i) for i in recommendations[u][:k] if i >= 0]
            assert len(ranking) == len(set(ranking)), "duplicate recommendation"
            assert all(i < n_items for i in ranking)
            hits = np.array([i in reachable for i in ranking], dtype=float)
            row[f"recall_reachable@{k}"] = (
                hits.sum() / len(reachable) if reachable else np.nan
            )
            row[f"recall_all@{k}"] = hits.sum() / len(ys) if ys else np.nan
            row[f"hr@{k}"] = float(hits.any()) if reachable else np.nan
            ideal_n = min(k, len(reachable))
            ideal = np.sum(1 / np.log2(np.arange(ideal_n) + 2))
            row[f"ndcg@{k}"] = (
                float(np.sum(hits / np.log2(np.arange(len(hits)) + 2)) / ideal)
                if ideal
                else np.nan
            )
        records.append(row)
    per_user = pd.DataFrame(records)
    summary = per_user.drop(columns="u").mean().to_dict()
    summary.update(
        {
            "users": len(targets),
            "users_with_targets": int((per_user.n_targets > 0).sum()),
            "users_with_reachable": int((per_user.n_reachable > 0).sum()),
        }
    )
    used = recommendations[recommendations >= 0]
    summary["catalog_coverage@200"] = len(np.unique(used)) / n_items if n_items else 0
    return summary, per_user


def metric_self_check():
    recs = np.array([[0, 2, 1], [1, 0, 2], [0, 1, 2], [0, 1, 2]])
    targets = [{0, 1}, {2, "outside:x"}, set(), {"outside:z"}]
    s, p = evaluate(recs, targets, 3, ks=(1, 3))
    assert np.isclose(s["recall_reachable@1"], 0.25)
    assert np.isclose(s["recall_all@3"], 0.5)
    assert s["users_with_targets"] == 3 and s["users_with_reachable"] == 2
    assert np.isclose(s["catalog_bound"], 0.5)
    return {
        "passed": True,
        "checks": [
            "macro averaging",
            "unreachable targets",
            "empty targets",
            "NDCG bounds",
            "duplicates",
        ],
    }
