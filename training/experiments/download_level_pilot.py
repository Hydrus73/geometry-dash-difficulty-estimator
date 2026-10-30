"""Resumably download a tier-stratified pilot of public level object strings."""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from download_gd_level import DEFAULT_CACHE, download
from training.train import DEFAULT_CSV, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-quantile", type=int, default=40)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="minimum seconds between level requests")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    selected_rows = rows[2 * boundary:] if args.split == "train" else rows[boundary:2 * boundary]
    tiers = torch.arange(1, 40, dtype=torch.float32)
    selected_rows.sort(key=lambda row: float(row["target"] @ tiers))
    randomizer = random.Random(args.seed)
    sample = []
    for quantile in range(5):
        start = quantile * len(selected_rows) // 5
        end = (quantile + 1) * len(selected_rows) // 5
        candidates = selected_rows[start:end]
        sample.extend(randomizer.sample(candidates, min(args.per_quantile, len(candidates))))
    randomizer.shuffle(sample)

    manifest_path = DEFAULT_CACHE / ("pilot_manifest.json" if args.split == "train"
                                     else "validation_pilot_manifest.json")
    DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    manifest = {"split": args.split, "seed": args.seed, "per_quantile": args.per_quantile,
                "requested_ids": [row["level_id"] for row in sample], "completed": []}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("requested_ids") == manifest["requested_ids"]:
            manifest = previous

    completed = set(manifest["completed"])
    for index, row in enumerate(sample, start=1):
        level_id = row["level_id"]
        if level_id in completed:
            continue
        try:
            download(level_id, DEFAULT_CACHE)
            completed.add(level_id)
            manifest["completed"] = sorted(completed)
            temp_path = manifest_path.with_suffix(".partial")
            temp_path.write_text(json.dumps(manifest), encoding="utf-8")
            temp_path.replace(manifest_path)
            if len(completed) % 10 == 0 or index == len(sample):
                print(f"downloaded {len(completed)}/{len(sample)} levels", flush=True)
        except Exception as exc:
            print(f"failed level {level_id}: {type(exc).__name__}: {exc}", flush=True)
        if index != len(sample):
            time.sleep(max(0.0, args.delay))

    print(f"pilot complete: {len(completed)}/{len(sample)} cached at {DEFAULT_CACHE}")


if __name__ == "__main__":
    main()
