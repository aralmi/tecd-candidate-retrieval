import numpy as np
import torch


@torch.inference_mode()
def exact_vector_topk(vectors, items, available, k, stable_topk):
    device = vectors.device
    n_items = len(items)
    result = np.full((len(vectors), k), -1, np.int32)
    candidate_ids = []
    candidate_scores = []
    for item_lo in range(0, n_items, 32768):
        item_hi = min(item_lo + 32768, n_items)
        if vectors.ndim == 3:
            # Exact max over interests; still one shared budget of k objects.
            block = items[item_lo:item_hi].T
            scores = vectors[:, 0].contiguous() @ block
            for interest in range(1, vectors.shape[1]):
                scores = torch.maximum(scores, vectors[:, interest].contiguous() @ block)
        else:
            scores = vectors @ items[item_lo:item_hi].T
        mask = torch.as_tensor(~available[item_lo:item_hi], device=device)
        scores[:, mask] = -torch.inf
        count = min(k, item_hi - item_lo)
        values, ids = torch.topk(scores, count, dim=1)
        # Полный порядок при граничных ties: score desc, item ID asc.
        cutoff = values[:, -1:]
        ties_all = (scores == cutoff).sum(1)
        ties_top = (values == cutoff).sum(1)
        ambiguous = (ties_all > ties_top).cpu().numpy()
        vs = values.cpu().numpy()
        ix = ids.cpu().numpy().astype(np.int32)
        for j in np.flatnonzero(ambiguous):
            row = scores[j].cpu().numpy()
            selected = stable_topk(row, count)
            ix[j, :] = -1
            vs[j, :] = -np.inf
            ix[j, : len(selected)] = selected
            vs[j, : len(selected)] = row[selected]
        ix = np.where(np.isfinite(vs), ix + item_lo, -1)
        candidate_ids.append(ix)
        candidate_scores.append(vs)
    ids = np.concatenate(candidate_ids, axis=1)
    scores = np.concatenate(candidate_scores, axis=1)
    for j in range(len(vectors)):
        valid = np.isfinite(scores[j]) & (ids[j] >= 0)
        indices = np.flatnonzero(valid)
        if len(indices) > k:
            threshold = np.partition(scores[j, indices], -k)[-k]
            better = indices[scores[j, indices] > threshold]
            tied = indices[scores[j, indices] == threshold]
            tied = tied[np.argsort(ids[j, tied])][: k - len(better)]
            indices = np.concatenate([better, tied])
        indices = indices[np.lexsort((ids[j, indices], -scores[j, indices]))]
        result[j, : len(indices)] = ids[j, indices]
    return result
