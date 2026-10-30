"""Evaluate spatial and object-transition features from cached GD level strings."""
import json
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.experiments.download_gd_level import DEFAULT_CACHE, iter_level_objects
from training.experiments.evaluate_level_object_features import (
    prepare_ridge, predict_ridge, weighted_score,
)
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


ID_BINS, X_BINS, Y_BINS, TRANSITION_BINS = 256, 16, 8, 4096
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _features_for_level_id(level_id):
    torch.set_num_threads(1)
    record = json.loads((DEFAULT_CACHE / f"{level_id}.json").read_text(encoding="utf-8"))
    return build_features(record["header"]["4"]).numpy()


def select_sample(split, rows, boundary, per_quantile=40):
    selected = rows[2 * boundary:] if split == "train" else rows[boundary:2 * boundary]
    selected = sorted(selected, key=lambda row: float(
        row["target"] @ torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)))
    population, q_lookup = [], {}
    for q in range(5):
        group = selected[q * len(selected) // 5:(q + 1) * len(selected) // 5]
        population.append(len(group))
        for row in group:
            q_lookup[row["level_id"]] = q
    manifest_name = "pilot_manifest.json" if split == "train" else "validation_pilot_manifest.json"
    manifest = json.loads((DEFAULT_CACHE / manifest_name).read_text(encoding="utf-8"))
    sample = [next(row for row in selected if row["level_id"] == level_id)
              for level_id in manifest["requested_ids"]]
    per_quantile = manifest.get("per_quantile", per_quantile)
    weights = torch.tensor([population[q_lookup[row["level_id"]]] / per_quantile for row in sample])
    return sample, weights


def build_features(encoded):
    objects = list(iter_level_objects(encoded))
    if not objects:
        return torch.zeros(ID_BINS + X_BINS * Y_BINS * ID_BINS
                           + TRANSITION_BINS + 3 * X_BINS + 8)

    min_y = min(obj[2] for obj in objects)
    max_y = max(obj[2] for obj in objects)
    max_x = max(obj[1] for obj in objects)
    global_counts = torch.zeros(ID_BINS)
    grid_counts = torch.zeros(X_BINS * Y_BINS * ID_BINS)
    transitions = torch.zeros(TRANSITION_BINS)
    x_counts = torch.zeros(X_BINS)
    x_y_sum = torch.zeros(X_BINS)
    x_y_sq = torch.zeros(X_BINS)
    unique_ids = set()
    rotated = scaled = 0

    previous_id = None
    for object_id, x, y, has_rotation, has_scale in objects:
        hashed_id = (object_id * 2654435761) % ID_BINS
        x_bin = min(X_BINS - 1, max(0, int(x / max(max_x, 1.0) * X_BINS)))
        y_bin = min(Y_BINS - 1, max(0, int((y - min_y) / max(max_y - min_y, 1.0) * Y_BINS)))
        global_counts[hashed_id] += 1
        grid_counts[(x_bin * Y_BINS + y_bin) * ID_BINS + hashed_id] += 1
        x_counts[x_bin] += 1
        x_y_sum[x_bin] += y
        x_y_sq[x_bin] += y * y
        unique_ids.add(object_id)
        rotated += int(has_rotation)
        scaled += int(has_scale)
        if previous_id is not None:
            bigram_hash = ((previous_id * 2246822519) ^ (object_id * 3266489917)) % TRANSITION_BINS
            transitions[bigram_hash] += 1
        previous_id = object_id

    count = len(objects)
    global_counts = global_counts.sqrt() / math.sqrt(count)
    grid_counts = grid_counts.reshape(X_BINS, Y_BINS, ID_BINS)
    tile_counts = grid_counts.sum(-1, keepdim=True).clamp_min(1).sqrt()
    grid_counts = (grid_counts / tile_counts).sqrt().reshape(-1)
    transitions = transitions.sqrt() / math.sqrt(max(1, count - 1))
    for x_bin in range(X_BINS):
        if x_counts[x_bin] > 0:
            x_y_sum[x_bin] /= x_counts[x_bin]
            x_y_sq[x_bin] = (x_y_sq[x_bin] / x_counts[x_bin] - x_y_sum[x_bin] ** 2).clamp_min(0).sqrt()
    x_counts /= count
    y_mean = sum(obj[2] for obj in objects) / count
    y_std = (sum(obj[2] ** 2 for obj in objects) / count - y_mean**2) ** 0.5
    stats = torch.tensor([
        math.log1p(count), math.log1p(len(unique_ids)), math.log1p(max_x),
        y_mean, y_std, rotated / count, scaled / count,
        sum((objects[i + 1][1] - objects[i][1]) < 0 for i in range(count - 1)) / max(1, count - 1),
    ])
    return torch.cat((global_counts, grid_counts, transitions, x_counts, x_y_sum, x_y_sq, stats))


def nested_cv(features, targets, weights):
    folds = torch.arange(len(targets)) % 5
    predictions = torch.empty_like(targets)
    selected_alphas = []
    for fold in range(5):
        outer_train, outer_valid = folds != fold, folds == fold
        inner_ids = torch.arange(int(outer_train.sum())) % 4
        scores = {alpha: [] for alpha in ALPHAS}
        train_x, train_y, train_w = features[outer_train], targets[outer_train], weights[outer_train]
        for inner in range(4):
            fit, tune = inner_ids != inner, inner_ids == inner
            prepared = prepare_ridge(train_x[fit], train_y[fit], train_w[fit])
            for alpha, prediction in zip(ALPHAS, predict_ridge(prepared, train_x[tune], ALPHAS)):
                scores[alpha].append(weighted_score(prediction, train_y[tune], train_w[tune]))
        alpha = max((sum(s[1] for s in vals) / len(vals),
                     -sum(s[0] for s in vals) / len(vals), -a, a)
                    for a, vals in scores.items())[3]
        selected_alphas.append(alpha)
        prepared = prepare_ridge(features[outer_train], targets[outer_train], weights[outer_train])
        predictions[outer_valid] = predict_ridge(prepared, features[outer_valid], [alpha])[0]
    results = [weighted_score(predictions[folds == f], targets[folds == f], weights[folds == f])
               for f in range(5)]
    print(f"nested CV alphas={selected_alphas}, MAE={sum(r[0] for r in results)/5:.4f}, "
          f"within3={sum(r[1] for r in results)/5:.4f}")
    return max(set(selected_alphas), key=selected_alphas.count)


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    train_rows, train_weights = select_sample("train", rows, boundary)
    valid_rows, valid_weights = select_sample("validation", rows, boundary)
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)

    def matrix(sample, name):
        cache_path = DEFAULT_CACHE / f"{name}_sequence_features.pt"
        ids = [row["level_id"] for row in sample]
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cached.get("level_ids") == ids:
                print(f"loaded cached {name} sequence features", flush=True)
                return cached["features"]
        features = []
        with ProcessPoolExecutor(max_workers=2) as pool:
            for i, feature in enumerate(pool.map(_features_for_level_id, ids, chunksize=1), 1):
                features.append(torch.from_numpy(feature))
                if i % 25 == 0:
                    print(f"parsed {name} levels {i}/{len(sample)}", flush=True)
        features = torch.stack(features)
        torch.save({"level_ids": ids, "features": features}, cache_path)
        return features

    train_x = matrix(train_rows, "training")
    valid_x = matrix(valid_rows, "validation")
    train_y = torch.stack([row["target"] for row in train_rows]) @ tiers
    valid_y = torch.stack([row["target"] for row in valid_rows]) @ tiers
    print(f"feature matrix={tuple(train_x.shape)}, train={len(train_rows)}, validation={len(valid_rows)}")
    alpha = nested_cv(train_x, train_y, train_weights)

    prepared = prepare_ridge(train_x, train_y, train_weights)
    prediction = predict_ridge(prepared, valid_x, [alpha])[0].clamp(1, NUM_CLASSES)
    print(f"validation pilot at alpha={alpha}: {weighted_score(prediction, valid_y, valid_weights)}")
    print("Only encoded level objects and vote-distribution labels were used; server rating fields were not read.")


if __name__ == "__main__":
    main()
