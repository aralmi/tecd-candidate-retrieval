import numpy as np

def stable_topk(scores, k, ids=None):
    scores = np.asarray(scores)
    ids = np.arange(len(scores)) if ids is None else np.asarray(ids)
    if scores.ndim != 1 or scores.shape != ids.shape:
        raise ValueError('One-dimensional scores and IDs required')
    valid = np.isfinite(scores) & (ids >= 0)
    order = np.lexsort((ids[valid], -scores[valid]))[:k]
    return ids[valid][order]

def exact_topk(query, items, k=200, block=32768):
    """Exact blockwise dot-product search, breaking ties by local item ID."""
    query, items = np.asarray(query), np.asarray(items)
    if query.ndim != 2 or items.ndim != 2 or query.shape[1] != items.shape[1]:
        raise ValueError('Expected query[users,d] and items[catalog,d]')
    k = min(k, len(items))
    output = np.full((len(query), k), -1, dtype=np.int64)
    for row, vector in enumerate(query):
        best_ids = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float32)
        for lo in range(0, len(items), block):
            ids = np.arange(lo, min(lo+block, len(items)))
            scores = items[lo:lo+block] @ vector
            all_ids = np.r_[best_ids, ids]
            all_scores = np.r_[best_scores, scores]
            order = np.lexsort((all_ids, -all_scores))[:k]
            best_ids, best_scores = all_ids[order], all_scores[order]
        output[row] = best_ids
    return output
