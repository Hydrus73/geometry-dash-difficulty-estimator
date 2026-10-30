"""Cache encoded level objects for the existing validation partition only."""
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.experiments.download_gd_level import DEFAULT_CACHE, download
from training.train import DEFAULT_CSV, read_rows


def main():
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    validation_rows = rows[boundary:2 * boundary]
    requested = [row["level_id"] for row in validation_rows]
    manifest_path = DEFAULT_CACHE / "full_validation_manifest.json"
    DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    manifest = {"split": "validation", "seed": 42, "requested_ids": requested,
                "completed": []}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("requested_ids") == requested:
            manifest = previous
    completed = set(manifest["completed"])
    for index, level_id in enumerate(requested, 1):
        if level_id in completed:
            continue
        try:
            download(level_id, DEFAULT_CACHE)
            completed.add(level_id)
            manifest["completed"] = sorted(completed)
            temp_path = manifest_path.with_suffix(".partial")
            temp_path.write_text(json.dumps(manifest), encoding="utf-8")
            temp_path.replace(manifest_path)
            if len(completed) % 10 == 0 or index == len(requested):
                print(f"cached {len(completed)}/{len(requested)} validation levels", flush=True)
        except Exception as exc:
            print(f"failed level {level_id}: {type(exc).__name__}: {exc}", flush=True)
        if index != len(requested):
            time.sleep(1.0)
    print(f"complete: {len(completed)}/{len(requested)} validation levels")


if __name__ == "__main__":
    main()
