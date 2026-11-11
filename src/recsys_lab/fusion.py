import numpy as np

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
