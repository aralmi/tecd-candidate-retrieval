import time
import numpy as np
from scipy import sparse
from implicit.cpu.als import AlternatingLeastSquares
from implicit.nearest_neighbours import BM25Recommender
from .data import ROOT, STRENGTH, save_json
from .metrics import evaluate


def topk(scores, k=200, positive_only=False):
    ids = np.flatnonzero(
        np.isfinite(scores) & ((scores > 0) if positive_only else True)
    )
    if len(ids) > k:
        cutoff = np.partition(scores[ids], -k)[-k]
        better = ids[scores[ids] > cutoff]
        ties = ids[scores[ids] == cutoff][: k - len(better)]
        ids = np.concatenate([better, ties])
    return ids[np.lexsort((ids, -scores[ids]))][:k].astype(np.int32)


def popularity(data):
    past = data.events[
        (data.events.day < data.cfg.get("fit_end", 1200)) & (data.events.i >= 0)
    ]
    click = past[past.action == 2].groupby("i").size()
    counts = np.zeros(data.n_items, dtype=np.float32)
    counts[click.index] = click.values
    # Сначала click popularity; при равных click counts — стабильный item ID.
    order = np.lexsort((np.arange(data.n_items), -counts)).astype(np.int32)
    return order


def fill(raw, popular, k=200):
    out = np.full((len(raw), k), -1, dtype=np.int32)
    for u, row in enumerate(raw):
        chosen = [int(i) for i in row if i >= 0]
        seen = set(chosen)
        for i in popular:
            if len(chosen) >= k:
                break
            if int(i) not in seen:
                chosen.append(int(i))
                seen.add(int(i))
        out[u, : len(chosen[:k])] = chosen[:k]
    return out


def evaluation_users(data):
    # Это общий набор запросов с определённым Recall, не отбор обучающей когорты.
    return np.array([u for u, y in enumerate(data.targets) if y], dtype=np.int32)


def record(data, name, raw, fit_seconds, predict_seconds, extra=None):
    pop = popularity(data)
    queries = evaluation_users(data)
    full = np.full_like(raw, -1)
    full[queries] = fill(raw[queries], pop, data.cfg["k"])
    summary, per_user = evaluate(full, data.targets, data.n_items)
    raw_summary, _ = evaluate(raw, data.targets, data.n_items)
    summary.update(
        {
            "method": name,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "evaluated_queries": len(queries),
            "raw_recall@200": raw_summary["recall_reachable@200"],
            "fallback_position_fraction": float(
                (raw[queries] < 0).sum() / raw[queries].size
            ),
            "fingerprint": data.fingerprint,
        }
    )
    summary.update(extra or {})
    dest = ROOT / "artifacts" / name
    dest.mkdir(exist_ok=True)
    np.save(dest / "raw.npy", raw)
    np.save(dest / "recommendations.npy", full)
    per_user.to_parquet(dest / "per_user.parquet", index=False)
    save_json(dest / "metrics.json", summary)
    print(
        f"RESULT {name}: Recall@200={summary['recall_reachable@200']:.5f}; "
        f"all={summary['recall_all@200']:.5f}; fit={fit_seconds:.2f}s; query={predict_seconds:.2f}s",
        flush=True,
    )
    return summary


def run_baseline(data, method, tag=None):
    label = tag or method
    queries = evaluation_users(data)
    k = data.cfg["k"]
    raw = np.full((data.n_users, k), -1, dtype=np.int32)
    start = time.perf_counter()
    extra = {
        "history_mode": data.cfg.get("history_mode", "all"),
        "fit_strong_only": data.cfg.get("fit_strong_only", False),
    }
    if method == "popularity":
        pop = popularity(data)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        raw[queries, : min(k, len(pop))] = pop[:k]
    elif method == "recent":
        fit_seconds = 0.0
        for u in queries:
            recent = list(dict.fromkeys((data.h[u][data.h[u] > 0] - 1)[::-1]))[:k]
            raw[u, : len(recent)] = recent
    elif method == "personal_popularity":
        fit_seconds = 0.0
        anchor = data.cfg["validation_anchor"]
        frame = data.events[
            (data.events.day >= anchor - data.cfg["history_days"] - 1)
            & (data.events.day < anchor - 1)
            & (data.events.action != 1)
            & (data.events.i >= 0)
        ].copy()
        frame["weight"] = STRENGTH[frame.action.to_numpy()]
        daily = frame.groupby(["u", "i", "day"], sort=False).agg(
            weight=("weight", "max"), last=("ts", "max")
        )
        profile = (
            daily.groupby(level=[0, 1])
            .agg(weight=("weight", "sum"), last=("last", "max"))
            .reset_index()
        )
        query_set = set(queries)
        for u, g in profile.groupby("u", sort=False):
            if u not in query_set:
                continue
            ordering = np.lexsort(
                (g.i.to_numpy(), -g["last"].to_numpy(), -g.weight.to_numpy())
            )[:k]
            ids = g.i.to_numpy()[ordering]
            raw[u, : len(ids)] = ids
    elif method in ("content", "content_decay"):
        fit_seconds = 0.0
        query = data.query_matrix()
        if method == "content_decay":
            us, pos = np.nonzero(data.h)
            query = sparse.csr_matrix(
                (
                    STRENGTH[data.actions[us, pos]]
                    * np.exp(-np.log(2) * data.ages[us, pos] / 7),
                    (us, data.h[us, pos] - 1),
                ),
                shape=(data.n_users, data.n_items),
            )
        profiles = np.asarray(query @ data.vectors)
        profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-12)
        start = time.perf_counter()
        if data.cfg.get("device") == "mps":
            import torch
            from .retrieval import exact_vector_topk

            assert torch.backends.mps.is_available()
            items = torch.as_tensor(data.vectors, device="mps")
            for lo in range(0, len(queries), 64):
                batch = queries[lo : lo + 64]
                q = torch.as_tensor(profiles[batch], device="mps")
                selected = exact_vector_topk(q, items, data.vector_ok, k, topk)
                valid = np.linalg.norm(profiles[batch], axis=1) > 0
                raw[batch[valid]] = selected[valid]
            torch.mps.synchronize()
            del items, q, selected
            torch.mps.empty_cache()
            extra["retrieval_device"] = "mps"
            return record(
                data, label, raw, fit_seconds, time.perf_counter() - start, extra
            )
        for lo in range(0, len(queries), 64):
            batch = queries[lo : lo + 64]
            scores = profiles[batch] @ data.vectors.T
            scores[:, ~data.vector_ok] = -np.inf
            for j, u in enumerate(batch):
                if np.linalg.norm(profiles[u]) > 0:
                    ids = topk(scores[j], k)
                    raw[u, : len(ids)] = ids
    elif method == "bm25":
        matrix = data.interaction_matrix()
        # Ограничиваем вклад сверхдлинных пользователей; общую C не урезаем.
        matrix = matrix.tolil()
        for u in range(data.n_users):
            if len(matrix.rows[u]) > 200:
                keep = np.argsort(matrix.data[u])[-200:]
                matrix.rows[u] = [matrix.rows[u][j] for j in keep]
                matrix.data[u] = [matrix.data[u][j] for j in keep]
        model = BM25Recommender(K=data.cfg.get('bm25_neighbors',100), K1=1.2, B=0.75, num_threads=4)
        model.fit(matrix.tocsr(), show_progress=False)
        model.similarity.setdiag(0)
        model.similarity.eliminate_zeros()
        model.save(str(ROOT / f"artifacts/{label}_model.npz"))
        extra["similarity_nnz"] = model.similarity.nnz
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        query = data.query_matrix()
        for lo in range(0, len(queries), 64):
            batch = queries[lo : lo + 64]
            scores = (query[batch] @ model.similarity).toarray()
            for j, u in enumerate(batch):
                ids = topk(scores[j], k, positive_only=True)
                raw[u, : len(ids)] = ids
    elif method == "als":
        matrix = data.interaction_matrix()
        model = AlternatingLeastSquares(
            factors=data.cfg.get('als_factors',32),
            regularization=data.cfg.get('als_regularization',0.1),
            alpha=1.0,
            iterations=data.cfg.get('als_iterations',10),
            num_threads=4,
            random_state=data.cfg["seed"],
        )
        # implicit ожидает C_ui в ненулевых ячейках; alpha повторно не применяется.
        confidence = matrix.copy()
        confidence.data = 1 + 20 * confidence.data
        model.fit(confidence, show_progress=False)
        model.save(str(ROOT / f"artifacts/{label}_model.npz"))
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        query = data.query_matrix()
        query.data = 1 + 20 * np.log1p(query.data)
        factors = model.recalculate_user(queries, query[queries])
        for lo in range(0, len(queries), 64):
            batch = queries[lo : lo + 64]
            scores = factors[lo : lo + 64] @ model.item_factors.T
            for j, u in enumerate(batch):
                if data.h[u].any():
                    ids = topk(scores[j], k)
                    raw[u, : len(ids)] = ids
    else:
        raise ValueError(method)
    predict_seconds = time.perf_counter() - start
    return record(data, label, raw, fit_seconds, predict_seconds, extra)
