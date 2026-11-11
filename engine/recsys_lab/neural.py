import copy
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .data import ROOT, save_json
from .baselines import evaluation_users, record, topk
from .metrics import evaluate


class TwoTower(nn.Module):
    """Общая товарная часть; pooling или causal attention для истории."""

    def __init__(self, vectors, use_content=True, encoder="mean", content_tensor=None):
        super().__init__()
        self.use_content, self.encoder_name = use_content, encoder
        if content_tensor is None:
            content_tensor = (
                torch.from_numpy(np.vstack([np.zeros((1, 300), np.float32), vectors]))
                if use_content
                else torch.zeros((1, 300))
            )
        self.register_buffer("content", content_tensor, persistent=False)
        self.register_buffer("id_seen", torch.zeros(len(vectors) + 1, dtype=torch.bool))
        self.ids = nn.Embedding(len(vectors) + 1, 32, padding_idx=0)
        self.item_mlp = nn.Sequential(
            nn.Linear(333, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64)
        )
        self.actions = nn.Embedding(5, 64, padding_idx=0)
        self.user_mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        if encoder == "attention":
            self.positions = nn.Embedding(50, 64)
            layer = nn.TransformerEncoderLayer(
                64,
                2,
                dim_feedforward=128,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            self.attention = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)

    def item(self, ids):
        content = (
            self.content[ids]
            if self.use_content
            else torch.zeros((*ids.shape, 300), device=ids.device)
        )
        present = (content.square().sum(-1, keepdim=True) > 0).float()
        known = ids if self.training else torch.where(self.id_seen[ids], ids, 0)
        value = torch.cat([self.ids(known), content, present], dim=-1)
        return F.normalize(self.item_mlp(value), dim=-1)

    def user(self, history, actions):
        valid = history != 0
        tokens = self.item(history) + self.actions(actions)
        if self.encoder_name == "mean":
            value = (tokens * valid.unsqueeze(-1)).sum(1) / valid.sum(
                1, keepdim=True
            ).clamp_min(1)
            return F.normalize(self.user_mlp(value), dim=-1)
        safe_valid = valid.clone()
        safe_valid[:, 0] = (
            True  # полностью пустая строка не создаёт all-masked attention
        )
        tokens = tokens + self.positions(
            torch.arange(history.shape[1], device=history.device)
        )
        causal = torch.triu(
            torch.ones(
                history.shape[1],
                history.shape[1],
                device=history.device,
                dtype=torch.bool,
            ),
            1,
        )
        encoded = self.attention(tokens, mask=causal, src_key_padding_mask=~safe_valid)
        last = (valid.sum(1) - 1).clamp_min(0)
        value = encoded[torch.arange(len(history), device=history.device), last]
        return F.normalize(self.user_mlp(value), dim=-1)

    def forward(self, h, actions, candidates):
        return (self.user(h, actions).unsqueeze(1) * self.item(candidates)).sum(
            -1
        ) / 0.1


def negative_batch(data, pairs, rows, rng, count=64):
    if not hasattr(data, "negative_pools"):
        data.negative_pools = {
            a: np.flatnonzero(data.first_day < a - 1) for a in data.cfg["train_anchors"]
        }
    result = []
    for row in rows:
        pool = data.negative_pools[int(pairs["anchors"][row])]
        forbidden = pairs["excluded"][row]
        n = min(count, len(pool) - len(forbidden))
        if n != count:
            raise ValueError(
                "Pilot catalog too small for fixed number of distinct negatives"
            )
        chosen = set()
        while len(chosen) < n:
            for item in rng.choice(pool, size=n * 2, replace=True):
                item = int(item)
                if item not in forbidden:
                    chosen.add(item)
                    if len(chosen) == n:
                        break
        result.append([pairs["positive"][row], *sorted(chosen)])
    return np.asarray(result, dtype=np.int64) + 1


@torch.inference_mode()
def recommend(data, model, device):
    model.eval()
    start = time.perf_counter()
    item_vectors = []
    for lo in range(0, data.n_items, 4096):
        ids = torch.arange(lo + 1, min(lo + 4096, data.n_items) + 1, device=device)
        item_vectors.append(model.item(ids))
    items = torch.cat(item_vectors)
    query_users = evaluation_users(data)
    raw = np.full((data.n_users, data.cfg["k"]), -1, np.int32)
    available = model.id_seen[1:].detach().cpu().numpy()
    if model.use_content:
        available = available | data.vector_ok
    for lo in range(0, len(query_users), 64):
        users = query_users[lo : lo + 64]
        h = torch.as_tensor(data.h[users], device=device)
        a = torch.as_tensor(data.actions[users], device=device)
        vectors = model.user(h, a)
        from .retrieval import exact_vector_topk

        selected = exact_vector_topk(vectors, items, available, data.cfg["k"], topk)
        for j, u in enumerate(users):
            if data.h[u].any():
                raw[u] = selected[j]
    if device == "mps":
        torch.mps.synchronize()
    return raw, time.perf_counter() - start


def tiny_overfit(data, pairs, steps=80):
    assert (
        torch.backends.mps.is_available()
    ), "Нейросетевые запуски разрешены только на MPS"
    torch.manual_seed(data.cfg["seed"])
    model = TwoTower(data.vectors, True, "mean").to("mps")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    rows = np.arange(min(16, len(pairs["positive"])))
    rng = np.random.default_rng(data.cfg["seed"])
    candidates = negative_batch(data, pairs, rows, rng, 16)
    h = torch.as_tensor(pairs["h"][rows], device="mps")
    a = torch.as_tensor(pairs["actions"][rows], device="mps")
    c = torch.as_tensor(candidates, device="mps")
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(h, a, c)
        loss = F.cross_entropy(
            logits, torch.zeros(len(rows), dtype=torch.long, device="mps")
        )
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    result = {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "steps": steps,
        "loss_decreased": losses[-1] < 0.5 * losses[0],
        "examples": len(rows),
        "device": "mps",
        "losses": losses,
    }
    save_json(ROOT / "artifacts/tiny_overfit.json", result)
    assert result["loss_decreased"], result
    del optimizer, model
    torch.mps.empty_cache()
    print("TINY_OVERFIT", result["initial_loss"], result["final_loss"], flush=True)
    return result


def train_model(data, pairs, name="tt_content", epochs=None, tag=None):
    assert (
        data.cfg["device"] == "mps" and torch.backends.mps.is_available()
    ), "Требуется MPS; CPU fallback отключён"
    save_json(ROOT / "artifacts/training_configuration.json", data.cfg)
    torch.manual_seed(data.cfg["seed"])
    np.random.seed(data.cfg["seed"])
    encoder = "attention" if name == "attention" else "mean"
    model = TwoTower(data.vectors, name != "tt_id", encoder).to("mps")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=data.cfg["learning_rate"], weight_decay=1e-5
    )
    rng = np.random.default_rng(data.cfg["seed"])
    n = len(pairs["positive"])
    batch_size = data.cfg["batch_size"]
    seen = np.zeros(data.n_items + 1, dtype=bool)
    curves = []
    best = -1.0
    best_epoch = 0
    training_seconds = 0.0
    artifact_name = tag or name
    dest = ROOT / "artifacts" / artifact_name
    dest.mkdir(exist_ok=True)
    for epoch in range(1, (epochs or data.cfg["epochs"]) + 1):
        model.train()
        start = time.perf_counter()
        total = 0.0
        order = rng.permutation(n)
        for lo in range(0, n, batch_size):
            rows = order[lo : lo + batch_size]
            candidates = negative_batch(
                data, pairs, rows, rng, data.cfg["negative_count"]
            )
            seen[candidates.ravel()] = True
            seen[pairs["h"][rows].ravel()] = True
            seen[0] = False
            h = torch.as_tensor(pairs["h"][rows], device="mps")
            a = torch.as_tensor(pairs["actions"][rows], device="mps")
            c = torch.as_tensor(candidates, device="mps")
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                model(h, a, c), torch.zeros(len(rows), device="mps", dtype=torch.long)
            )
            assert torch.isfinite(loss), "non-finite train loss"
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(rows)
        model.id_seen.copy_(torch.as_tensor(seen, device="mps"))
        torch.mps.synchronize()
        fit = time.perf_counter() - start
        training_seconds += fit
        raw, predict_seconds = recommend(data, model, "mps")
        from .baselines import fill, popularity

        queries = evaluation_users(data)
        full = np.full_like(raw, -1)
        full[queries] = fill(raw[queries], popularity(data), data.cfg["k"])
        metrics, _ = evaluate(full, data.targets, data.n_items)
        recall = metrics["recall_reachable@200"]
        row = {
            "epoch": epoch,
            "train_loss": total / n,
            "recall@200": recall,
            "fit_seconds": fit,
            "query_seconds": predict_seconds,
            "mps_allocated_mb": torch.mps.current_allocated_memory() / 2**20,
        }
        curves.append(row)
        pd.DataFrame(curves).to_csv(dest / "learning_curve.csv", index=False)
        print(artifact_name, row, flush=True)
        if recall > best:
            best = recall
            best_epoch = epoch
            torch.save(
                {
                    "state": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "encoder": encoder,
                    "use_content": model.use_content,
                    "cfg": data.cfg,
                    "fingerprint": data.fingerprint,
                    "best_epoch": epoch,
                },
                dest / "checkpoint.pt",
            )
        if epoch - best_epoch >= 3:
            break
    checkpoint = torch.load(
        dest / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    state = {k: v for k, v in checkpoint["state"].items() if k != "content"}
    model.load_state_dict(state)
    raw, predict_seconds = recommend(data, model, "mps")
    # Повторная загрузка в новый объект должна воспроизвести список, а не только loss.
    reloaded = TwoTower(
        data.vectors,
        checkpoint["use_content"],
        checkpoint["encoder"],
        content_tensor=model.content,
    ).to("mps")
    reloaded.load_state_dict(state)
    again, _ = recommend(data, reloaded, "mps")
    np.testing.assert_array_equal(raw, again)
    summary = record(
        data,
        artifact_name,
        raw,
        training_seconds,
        predict_seconds,
        {
            "device": "mps",
            "best_epoch": best_epoch,
            "epochs_completed": len(curves),
            "training_pairs": n,
            "reload_identical": True,
            "history_mode": data.cfg.get("history_mode", "all"),
            "batch_size": batch_size,
            "optimizer_updates": int(np.ceil(n / batch_size)) * len(curves),
            "parameters": sum(p.numel() for p in model.parameters()),
            "mps_allocated_mb": torch.mps.current_allocated_memory() / 2**20,
        },
    )
    del optimizer, model, reloaded
    torch.mps.empty_cache()
    return summary
