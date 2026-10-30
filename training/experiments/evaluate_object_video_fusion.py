"""Test whether level-object features add signal to video predictions."""
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from download_gd_level import DEFAULT_CACHE
from evaluate_level_object_features import build_features, prepare_ridge, predict_ridge
from evaluate_validation import MODELS, predict as predict_model
from knn_dinov2 import make_representations, predict as predict_knn
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


BASE_FEATURES = ROOT / "training" / "features_dinov2_base_4fps"


def weighted_score(prediction, target, weights):
    error = (prediction - target).abs()
    weights = weights / weights.sum()
    return ((error * weights).sum().item(), ((error <= 3).float() * weights).sum().item())


def select_sample(split, rows, boundary, per_quantile=40):
    selected = rows[2 * boundary:] if split == "train" else rows[boundary:2 * boundary]
    selected = sorted(selected, key=lambda row: float(
        row["target"] @ torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)))
    population = []
    q_lookup = {}
    for q in range(5):
        start, end = q * len(selected) // 5, (q + 1) * len(selected) // 5
        group = selected[start:end]
        population.append(len(group))
        for row in group:
            q_lookup[row["level_id"]] = q
    manifest_name = "pilot_manifest.json" if split == "train" else "validation_pilot_manifest.json"
    manifest = json.loads((DEFAULT_CACHE / manifest_name).read_text(encoding="utf-8"))
    sample = [next(row for row in selected if row["level_id"] == level_id)
              for level_id in manifest["requested_ids"]]
    weights = torch.tensor([population[q_lookup[row["level_id"]]] / per_quantile for row in sample])
    return sample, weights


def object_matrix(rows, split):
    cache_path = DEFAULT_CACHE / f"{split}_structural_features.pt"
    level_ids = [row["level_id"] for row in rows]
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("level_ids") == level_ids:
            return cached["global"], cached["spatial"]
    global_features, spatial_features = [], []
    for index, row in enumerate(rows, start=1):
        record = json.loads((DEFAULT_CACHE / f"{row['level_id']}.json").read_text(encoding="utf-8"))
        global_feature, spatial_feature = build_features(record["header"]["4"])
        global_features.append(global_feature)
        spatial_features.append(spatial_feature)
        if index % 10 == 0:
            print(f"decoded object data for {index}/{len(rows)} levels", flush=True)
    result = {"level_ids": level_ids, "global": torch.stack(global_features),
              "spatial": torch.stack(spatial_features)}
    with tempfile.NamedTemporaryFile(dir=DEFAULT_CACHE, prefix=f".{split}_features.",
                                     suffix=".partial", delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        torch.save(result, temp_path)
        os.replace(temp_path, cache_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return result["global"], result["spatial"]


def cross_validated_ridge(x, y, weights, name):
    alpha_values = (0.1, 1.0, 10.0, 100.0, 1000.0)
    folds = torch.arange(len(y)) % 5
    predictions = torch.empty_like(y)
    selected = []
    for fold in range(5):
        outer_train = torch.where(folds != fold)[0]
        outer_valid = torch.where(folds == fold)[0]
        inner_folds = torch.arange(len(outer_train)) % 4
        alpha_scores = {alpha: [] for alpha in alpha_values}
        for inner in range(4):
            fit_idx = outer_train[inner_folds != inner]
            tune_idx = outer_train[inner_folds == inner]
            prepared = prepare_ridge(x[fit_idx], y[fit_idx], weights[fit_idx])
            candidates = predict_ridge(prepared, x[tune_idx], alpha_values)
            for alpha, candidate in zip(alpha_values, candidates):
                error = (candidate - y[tune_idx]).abs()
                local_weights = weights[tune_idx] / weights[tune_idx].sum()
                within = ((error <= 3).float() * local_weights).sum().item()
                mae = (error * local_weights).sum().item()
                alpha_scores[alpha].append((within, -mae))
        alpha = max((sum(v[0] for v in scores) / 4,
                     sum(v[1] for v in scores) / 4, a) for a, scores in alpha_scores.items())[2]
        selected.append(alpha)
        prepared = prepare_ridge(x[outer_train], y[outer_train], weights[outer_train])
        predictions[outer_valid] = predict_ridge(prepared, x[outer_valid], [alpha])[0]
    fold_scores = []
    for fold in range(5):
        valid = folds == fold
        fold_scores.append(weighted_score(predictions[valid], y[valid], weights[valid]))
    print(f"{name} CV alphas={selected}, MAE={sum(s[0] for s in fold_scores)/5:.4f}, "
          f"within3={sum(s[1] for s in fold_scores)/5:.4f}")
    return selected, predictions


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    train_rows, train_weights = select_sample("train", rows, boundary)
    valid_rows, valid_weights = select_sample("validation", rows, boundary)
    _, train_spatial = object_matrix(train_rows, "train")
    _, valid_spatial = object_matrix(valid_rows, "validation")

    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    train_targets = torch.stack([row["target"] for row in train_rows])
    valid_targets = torch.stack([row["target"] for row in valid_rows])
    train_actual = train_targets @ tiers
    valid_actual = valid_targets @ tiers

    alphas, _ = cross_validated_ridge(train_spatial, train_actual, train_weights, "spatial objects")
    object_alpha = max(set(alphas), key=alphas.count)
    prepared = prepare_ridge(train_spatial, train_actual, train_weights)
    object_prediction = predict_ridge(prepared, valid_spatial, [object_alpha])[0].clamp(1, NUM_CLASSES)
    print("object-only validation pilot:", weighted_score(object_prediction, valid_actual, valid_weights))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = {name: predict_model(name, valid_rows, device) for name in MODELS}
    groups = [
        torch.stack([raw[n] for n in ("videomae_pooled", "videomae_mae", "videomae_categorical")]).mean(0),
        torch.stack([raw[n] for n in ("dino_small_ordinal", "dino_small_ordinal_seed7",
                                      "dino_small_categorical")]).mean(0),
        torch.stack([raw[n] for n in ("dino_base_4s_ordinal", "dino_base_4s_categorical",
                                      "dino_base_4s_categorical_seed7",
                                      "dino_base_4s_dropout06")]).mean(0),
    ]
    video_distribution = torch.stack(groups).mean(0)
    video_median = torch.searchsorted(
        video_distribution.cumsum(-1), torch.full((len(valid_rows), 1), 0.5)
    ).squeeze(-1).float() + 1

    all_rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(all_rows)
    train_idx = torch.arange(2 * boundary, len(all_rows))
    pooled, stds = [], []
    for row in all_rows:
        cache = torch.load(BASE_FEATURES / f"{row['level_id']}.pt", map_location="cpu", weights_only=True)
        features = cache["features"].float()
        pooled.append(features.mean(0))
        stds.append(features.std(0, unbiased=False))
    visual_features = make_representations(torch.stack(pooled), torch.stack(stds))["mean"]
    position_by_id = {row["level_id"]: index for index, row in enumerate(all_rows)}
    valid_positions = torch.tensor([position_by_id[row["level_id"]] for row in valid_rows])
    knn_similarity = visual_features[valid_positions] @ visual_features[train_idx].T
    knn_distribution = predict_knn(knn_similarity,
                                   torch.stack([row["target"] for row in all_rows])[train_idx],
                                   20, 0.02)
    video_distribution = 0.9 * video_distribution + 0.1 * knn_distribution
    video_median = torch.searchsorted(
        video_distribution.cumsum(-1), torch.full((len(valid_rows), 1), 0.5)
    ).squeeze(-1).float() + 1
    print("video-only validation pilot:", weighted_score(video_median, valid_actual, valid_weights))

    fold_ids = torch.arange(len(valid_rows)) % 5
    blend_scores = []
    video_scores, object_scores = [], []
    for fold in range(5):
        train, valid = fold_ids != fold, fold_ids == fold
        video_scores.append(weighted_score(video_median[valid], valid_actual[valid], valid_weights[valid]))
        object_scores.append(weighted_score(object_prediction[valid], valid_actual[valid], valid_weights[valid]))
        best = None
        for weight in torch.linspace(0, 0.5, 11):
            prediction = (1 - weight) * video_median + weight * object_prediction
            mae, within3 = weighted_score(prediction[train], valid_actual[train], valid_weights[train])
            candidate = (within3, -mae, float(weight))
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        prediction = (1 - best[2]) * video_median[valid] + best[2] * object_prediction[valid]
        result = weighted_score(prediction, valid_actual[valid], valid_weights[valid])
        blend_scores.append(result)
        print(f"fold {fold + 1} object/video weight={best[2]:.2f}, "
              f"MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("video-only 5-fold weighted: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in video_scores) / 5, sum(x[1] for x in video_scores) / 5))
    print("object-only 5-fold weighted: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in object_scores) / 5, sum(x[1] for x in object_scores) / 5))
    print("video/object fusion 5-fold weighted: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in blend_scores) / 5, sum(x[1] for x in blend_scores) / 5))


if __name__ == "__main__":
    main()
