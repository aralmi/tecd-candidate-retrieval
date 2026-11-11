"""Скачивание только раннего пилота; финальный holdout не запрашивается."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time
import requests

from .data import ROOT, PROJECT_ROOT, config

REVISION = "79bcf78cd4699267f561ad3fd51fe7b52af53083"
BASE = f"https://huggingface.co/datasets/t-tech/T-ECD/resolve/{REVISION}/"


def download_one(entry):
    name = entry["path"]
    relative = name.removeprefix("dataset/small/marketplace/")
    dest = ROOT / "data/raw" / relative
    if dest.exists() and dest.stat().st_size == entry["size"]:
        return {"file": relative, "cached": True, "bytes": entry["size"]}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    started = time.perf_counter()
    for attempt in range(3):
        try:
            hasher = hashlib.sha256()
            with requests.get(
                BASE + name + "?download=true", stream=True, timeout=(30, 90)
            ) as response:
                response.raise_for_status()
                with tmp.open("wb") as stream:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        stream.write(chunk)
                        hasher.update(chunk)
            assert tmp.stat().st_size == entry["size"], (name, "size mismatch")
            expected = entry.get("lfs", {}).get("oid")
            if expected:
                assert hasher.hexdigest() == expected, (name, "SHA256 mismatch")
            tmp.replace(dest)
            result = {
                "file": relative,
                "bytes": dest.stat().st_size,
                "sha256": hasher.hexdigest(),
                "seconds": round(time.perf_counter() - started, 2),
            }
            print(json.dumps(result), flush=True)
            return result
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def main():
    audit = json.loads(
        (
            PROJECT_ROOT.parent / "Планирование/Проверка_данных_18_10_2026.json"
        ).read_text()
    )
    cfg = config()
    selected = [
        e
        for e in audit["file_manifest"]
        if e["type"] == "file"
        and (
            (
                "/events/" in e["path"]
                and cfg.get("event_start", 1180)
                <= int(Path(e["path"]).stem)
                <= cfg.get("event_end", 1207)
            )
            or e["path"].endswith("/items.pq")
        )
    ]
    assert (
        len(selected) == cfg.get("event_end", 1207) - cfg.get("event_start", 1180) + 2
    ), len(selected)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = [
            f.result()
            for f in as_completed([executor.submit(download_one, e) for e in selected])
        ]
    (ROOT / "artifacts/download_manifest.json").write_text(
        json.dumps(
            {"revision": REVISION, "files": results}, ensure_ascii=False, indent=2
        )
    )
    print("DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
