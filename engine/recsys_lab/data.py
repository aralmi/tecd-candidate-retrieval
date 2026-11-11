from pathlib import Path
import hashlib
import json
import time
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("RECSYS_WORKSPACE", str(PROJECT_ROOT)))
DAY_US = 86_400_000_000
ACTION = {"view": 1, "click": 2, "like": 3, "clickout": 4}
STRENGTH = np.array([0, 0.1, 1, 2, 3], dtype=np.float32)


def config():
    return json.loads((ROOT / "configs/pilot.json").read_text())


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
    )


def stable_order(users, seed):
    """SplitMix64: воспроизводимый порядок по ID, без будущих признаков."""
    x = np.asarray(users, dtype=np.uint64) ^ np.uint64(seed)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> 31)
    return np.lexsort((users, x))


def select_events(cfg=None):
    cfg = cfg or config()
    start = time.perf_counter()
    raw = ROOT / "data/raw/events"
    if cfg.get("cohort_mode") == "past_strong":
        stats = pd.read_parquet(PROJECT_ROOT / "data/past_user_activity.parquet")
        mask = (stats.strong_items >= cfg["min_strong_items"]) & (
            stats.strong_days >= cfg["min_strong_days"]
        )
        eligible = stats.index[mask].to_numpy(dtype=np.uint64)
    else:
        seen = []
        for day in range(
            cfg.get("cohort_start", 1180), cfg.get("cohort_end", 1186) + 1
        ):
            seen.append(
                pq.read_table(raw / f"{day:05}.pq", columns=["user_id"])[
                    "user_id"
                ].to_numpy()
            )
        eligible = np.unique(np.concatenate(seen))
    users = eligible[stable_order(eligible, cfg["seed"])[: cfg["n_users"]]]
    allowed = pa.array(users, type=pa.uint64())
    frames, counts = [], []
    for day in range(cfg.get("event_start", 1180), cfg.get("event_end", 1207) + 1):
        table = pq.read_table(
            raw / f"{day:05}.pq",
            columns=["timestamp", "user_id", "item_id", "action_type", "subdomain"],
        )
        table = table.filter(pc.is_in(table["user_id"], value_set=allowed))
        ts = table["timestamp"].cast(pa.int64()).to_numpy()
        assert np.all(ts // DAY_US == day), (day, "timestamp/file mismatch")
        frame = table.drop(["timestamp"]).to_pandas()
        frame["ts"] = ts
        frame["day"] = day
        frame["action"] = frame["action_type"].map(ACTION).astype("int8")
        assert not frame["action"].isna().any()
        frame = frame.drop(columns="action_type")
        frame["item_id"] = frame["item_id"].astype("category")
        frame["subdomain"] = frame["subdomain"].astype("category")
        frame["day"] = frame["day"].astype(np.int16)
        frames.append(frame)
        counts.append(
            {"day": day, "events": len(frame), "clicks": int((frame.action == 2).sum())}
        )
    item_categories = pd.api.types.union_categoricals(
        [f.item_id.array for f in frames], sort_categories=True
    )
    surfaces = pd.api.types.union_categoricals(
        [f.subdomain.array for f in frames], sort_categories=True
    )
    events = pd.concat(
        [f.drop(columns=["item_id", "subdomain"]) for f in frames], ignore_index=True
    )
    events["item_id"] = item_categories
    events["subdomain"] = surfaces
    del frames
    events = events.drop_duplicates(["user_id", "item_id", "ts", "action"])
    events = events.sort_values(
        ["user_id", "ts", "item_id", "action"], kind="stable"
    ).reset_index(drop=True)
    out = ROOT / "data/processed"
    out.mkdir(exist_ok=True)
    events.to_parquet(out / "events.parquet", index=False)
    np.save(out / "users.npy", users)
    summary = {
        "n_users": len(users),
        "eligible_past_users": len(eligible),
        "events": len(events),
        "actions": events.action.value_counts().to_dict(),
        "daily": counts,
        "seconds": time.perf_counter() - start,
        "cohort_rule": "SplitMix64(user_id XOR seed), past only",
        "cfg": cfg,
    }
    save_json(ROOT / "artifacts/cohort.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def select_items():
    import duckdb

    events = pd.read_parquet(ROOT / "data/processed/events.parquet")
    wanted = pa.array(
        sorted(
            events.loc[events.day < config().get("fit_end", 1200), "item_id"].unique()
        ),
        type=pa.large_string(),
    )
    start = time.perf_counter()
    # Исходный ARROW:schema фиксирует list[300], но nullable записи вызывают
    # ArrowInvalid при прямом чтении. Нативный Parquet reader DuckDB обходит
    # проблему; длина каждого непустого вектора проверяется ниже.
    connection = duckdb.connect()
    connection.execute("SET memory_limit='2GB'")
    connection.execute("SET threads=4")
    connection.register("wanted", pa.table({"item_id": wanted}))
    items = connection.execute(
        """SELECT i.item_id, i.brand_id, i.category,
        i.subcategory, i.price, CAST(i.embedding AS FLOAT[]) AS embedding
        FROM read_parquet(?) i SEMI JOIN wanted w USING(item_id)""",
        [str(ROOT / "data/raw/items.pq")],
    ).fetch_arrow_table()
    lengths = pc.list_value_length(items["embedding"]).drop_null().to_numpy()
    assert np.all(lengths == 300), "invalid non-null vector length"
    pq.write_table(items, ROOT / "data/processed/items.parquet")
    summary = {
        "past_item_ids": len(wanted),
        "catalog_rows": items.num_rows,
        "missing_catalog_rows": len(wanted) - items.num_rows,
        "seconds": time.perf_counter() - start,
    }
    save_json(ROOT / "artifacts/catalog.json", summary)
    print(summary, flush=True)
    return summary


class PilotData:
    def __init__(self, cfg=None):
        self.cfg = cfg or config()
        self.events = pd.read_parquet(ROOT / "data/processed/events.parquet")
        self.users = np.load(ROOT / "data/processed/users.npy")
        table = pq.read_table(ROOT / "data/processed/items.parquet").sort_by("item_id")
        self.item_ids = table["item_id"].to_pylist()
        self.item_map = {item: i for i, item in enumerate(self.item_ids)}
        self.user_map = {int(u): i for i, u in enumerate(self.users)}
        self.n_users, self.n_items = len(self.users), len(self.item_ids)
        e = table["embedding"].combine_chunks()
        present = e.is_valid().to_numpy(zero_copy_only=False)
        e = pc.fill_null(e, pa.scalar([0.0] * 300, type=e.type))
        self.vectors = (
            np.asarray(e.values.to_numpy(zero_copy_only=False))
            .reshape(-1, 300)
            .astype(np.float32, copy=True)
        )
        self.vector_ok = present.copy()
        for lo in range(0, len(self.vectors), 8192):
            block = self.vectors[lo : lo + 8192]
            norms = np.linalg.norm(block, axis=1)
            ok = (
                np.isfinite(block).all(axis=1)
                & (norms > 0)
                & self.vector_ok[lo : lo + 8192]
            )
            self.vector_ok[lo : lo + 8192] = ok
            block[~ok] = 0
            block /= np.maximum(norms[:, None], 1e-12)
            block[~ok] = 0
        self.events["u"] = self.events.user_id.map(self.user_map).astype(np.int32)
        if isinstance(self.events.item_id.dtype, pd.CategoricalDtype):
            mapping = np.array(
                [
                    self.item_map.get(item, -1)
                    for item in self.events.item_id.cat.categories
                ],
                dtype=np.int32,
            )
            codes = self.events.item_id.cat.codes.to_numpy()
            self.events["i"] = np.where(codes >= 0, mapping[np.maximum(codes, 0)], -1)
        else:
            self.events["i"] = (
                self.events.item_id.map(self.item_map).fillna(-1).astype(np.int32)
            )
        first = self.events[self.events.i >= 0].groupby("i").day.min()
        self.first_day = np.full(self.n_items, 99999, dtype=np.int32)
        self.first_day[first.index] = first.values
        self.h, self.actions, self.ages = self.history(self.cfg["validation_anchor"])
        self.targets = self.target_sets(self.cfg["validation_anchor"])
        assert self.first_day.max() < self.cfg.get("fit_end", 1200)
        self.fingerprint = hashlib.sha256(
            ("\n".join(self.item_ids)).encode() + self.users.tobytes()
        ).hexdigest()

    def history(self, anchor, strong_only=None):
        if strong_only is None:
            strong_only = self.cfg.get("history_mode", "all") == "strong_only"
        lo, hi = anchor - self.cfg["history_days"] - 1, anchor - 1
        frame = self.events[
            (self.events.day >= lo) & (self.events.day < hi) & (self.events.i >= 0)
        ]
        if strong_only:
            frame = frame[frame.action != 1]
        # Схлопываются только подряд идущие одинаковые item/action одного пользователя.
        same = (
            (frame.u == frame.u.shift())
            & (frame.i == frame.i.shift())
            & (frame.action == frame.action.shift())
        )
        frame = (
            frame.loc[~same.shift(-1, fill_value=False)]
            .groupby("u", sort=False)
            .tail(self.cfg["history_length"])
        )
        h = np.zeros((self.n_users, self.cfg["history_length"]), dtype=np.int64)
        actions = np.zeros_like(h)
        ages = np.zeros_like(h, dtype=np.float32)
        for u, group in frame.groupby("u", sort=False):
            n = len(group)
            h[u, :n] = group.i.to_numpy() + 1
            actions[u, :n] = group.action
            ages[u, :n] = (hi * DAY_US - group.ts.to_numpy()) / DAY_US
        return h, actions, ages

    def target_sets(self, anchor):
        frame = self.events[
            (self.events.day >= anchor)
            & (self.events.day < anchor + 7)
            & (self.events.action == 2)
        ]
        sets = [set() for _ in self.users]
        for u, group in frame.groupby("u", sort=False):
            # Вне каталога сохраняется строковый маркер: цель остаётся в полном знаменателе.
            sets[u] = {
                self.item_map.get(item, "outside:" + item)
                for item in group.item_id.unique()
            }
        return sets

    def interaction_matrix(self):
        frame = self.events[
            (self.events.day < self.cfg.get("fit_end", 1200)) & (self.events.i >= 0)
        ].copy()
        if self.cfg.get("fit_strong_only", False):
            frame = frame[frame.action != 1]
        frame["weight"] = STRENGTH[frame.action.to_numpy()]
        daily = frame.groupby(["u", "i", "day"], sort=False).weight.max()
        agg = daily.groupby(level=[0, 1]).sum()
        return sparse.csr_matrix(
            (
                np.log1p(agg.values).astype(np.float32),
                (agg.index.get_level_values(0), agg.index.get_level_values(1)),
            ),
            shape=(self.n_users, self.n_items),
        )

    def query_matrix(self):
        u, pos = np.nonzero(self.h)
        return sparse.csr_matrix(
            (STRENGTH[self.actions[u, pos]], (u, self.h[u, pos] - 1)),
            shape=(self.n_users, self.n_items),
        )

    def training_pairs(self):
        histories, acts, positives, pools, excluded, user_rows = [], [], [], [], [], []
        for anchor in self.cfg["train_anchors"]:
            h, a, _ = self.history(anchor)
            targets = self.target_sets(anchor)
            pool = np.flatnonzero(self.first_day < anchor - 1)
            for u, ys in enumerate(targets):
                good = sorted(
                    i
                    for i in ys
                    if isinstance(i, int) and self.first_day[i] < anchor - 1
                )
                if h[u].any() and good:
                    for positive in good:
                        histories.append(h[u])
                        acts.append(a[u])
                        positives.append(positive)
                        user_rows.append(u)
                        pools.append(anchor)
                        excluded.append(set(good))
        assert histories, "No neural training pairs"
        result = {
            "h": np.asarray(histories),
            "actions": np.asarray(acts),
            "positive": np.asarray(positives),
            "anchors": np.asarray(pools),
            "excluded": excluded,
            "users": np.asarray(user_rows, dtype=np.int32),
        }
        save_json(
            ROOT / "artifacts/training_pairs.json",
            {
                "pairs": len(positives),
                "positive_items": len(set(positives)),
                "anchors": self.cfg["train_anchors"],
                "max_history_day_rule": "day < anchor-1",
                "positive_rule": "unique click items in week intersect C(anchor)",
                "overlapping_windows": bool(np.any(np.diff(sorted(self.cfg["train_anchors"])) < 7)),
            },
        )
        return result


if __name__ == "__main__":
    import sys

    if "--items" in sys.argv:
        select_items()
    else:
        select_events()
