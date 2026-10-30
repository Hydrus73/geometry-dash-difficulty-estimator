"""Cross-validate hashed object and coarse spatial features on the pilot."""
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.experiments.download_gd_level import DEFAULT_CACHE
from training.experiments.download_gd_level import iter_level_objects
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


ID_BINS = 128
X_BINS = 8
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def build_features(encoded):
    max_x = 1.0
    count = unique_ids = 0
    y_sum = y_squared_sum = 0.0
    rotated_count = scaled_count = 0
    unique_ids = set()
    for object_id, x, y, rotated, scaled in iter_level_objects(encoded):
        max_x = max(max_x, x)
        count += 1
        unique_ids.add(object_id)
        y_sum += y
        y_squared_sum += y * y
        rotated_count += int(rotated)
        scaled_count += int(scaled)
    if count == 0:
        return torch.zeros(ID_BINS + 7), torch.zeros(ID_BINS + X_BINS * ID_BINS + 3 * X_BINS + 7)

    global_counts = torch.zeros(ID_BINS)
    spatial_counts = torch.zeros(X_BINS * ID_BINS)
    bin_count = torch.zeros(X_BINS)
    bin_y_sum = torch.zeros(X_BINS)
    bin_y_sq = torch.zeros(X_BINS)
    for object_id, x, y, rotated, scaled in iter_level_objects(encoded):
        hashed_id = (object_id * 2654435761) % ID_BINS
        x_bin = min(X_BINS - 1, max(0, int(x / max_x * X_BINS)))
        global_counts[hashed_id] += 1
        spatial_counts[x_bin * ID_BINS + hashed_id] += 1
        bin_count[x_bin] += 1
        bin_y_sum[x_bin] += y
        bin_y_sq[x_bin] += y * y

    global_counts = global_counts.sqrt() / math.sqrt(count)
    spatial_counts = spatial_counts.sqrt()
    for x_bin in range(X_BINS):
        if bin_count[x_bin] > 0:
            spatial_counts[x_bin * ID_BINS:(x_bin + 1) * ID_BINS] /= bin_count[x_bin].sqrt()
            bin_y_sum[x_bin] /= bin_count[x_bin]
            variance = bin_y_sq[x_bin] / bin_count[x_bin] - bin_y_sum[x_bin] ** 2
            bin_y_sq[x_bin] = variance.clamp_min(0).sqrt()
    bin_count /= count
    y_mean = y_sum / count
    y_std = max(0.0, y_squared_sum / count - y_mean * y_mean) ** 0.5
    stats = torch.tensor([
        math.log1p(count), math.log1p(len(unique_ids)), math.log1p(max_x),
        y_mean, y_std, rotated_count / count, scaled_count / count,
    ])
    composition = torch.cat((global_counts, stats))
    spatial = torch.cat((global_counts, spatial_counts, bin_count, bin_y_sum, bin_y_sq, stats))
    return composition, spatial


def weighted_score(prediction, target, weights):
    error = (prediction - target).abs()
    weights = weights / weights.sum()
    return ((error * weights).sum().item(), ((error <= 3).float() * weights).sum().item())


def prepare_ridge(x_train, y_train, weights):
    weights = weights / weights.mean()
    mean = (x_train * weights[:, None]).sum(0) / weights.sum()
    centered = x_train - mean
    scale = ((centered.square() * weights[:, None]).sum(0) / weights.sum()).sqrt().clamp_min(1e-5)
    x = (centered / scale)
    y_mean = (y_train * weights).sum() / weights.sum()
    y = y_train - y_mean
    root_w = weights.sqrt()
    xw = x * root_w[:, None]
    yw = y * root_w
    gram = xw @ xw.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    projected_y = eigenvectors.T @ yw
    return mean, scale, xw, y_mean, eigenvalues, eigenvectors, projected_y


def predict_ridge(prepared, x_eval, alphas):
    mean, scale, xw, y_mean, eigenvalues, eigenvectors, projected_y = prepared
    cross = ((x_eval - mean) / scale) @ xw.T
    projected_cross = cross @ eigenvectors
    return [projected_cross @ (projected_y / (eigenvalues + alpha)) + y_mean
            for alpha in alphas]


def weighted_median(values, weights):
    order = values.argsort()
    sorted_values, sorted_weights = values[order], weights[order]
    index = torch.searchsorted(sorted_weights.cumsum(0), sorted_weights.sum() / 2).item()
    return sorted_values[min(index, len(values) - 1)]


def main():
    torch.set_num_threads(4)
    all_rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(all_rows)
    boundary = math.ceil(len(all_rows) * 0.1)
    train_rows = all_rows[2 * boundary:]
    train_rows = sorted(train_rows, key=lambda row: float(
        row["target"] @ torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)))
    q_edges = [i * len(train_rows) // 5 for i in range(6)]
    q_for_id = {}
    q_population = []
    for q in range(5):
        group = train_rows[q_edges[q]:q_edges[q + 1]]
        q_population.append(len(group))
        for row in group:
            q_for_id[row["level_id"]] = q

    manifest = json.loads((DEFAULT_CACHE / "pilot_manifest.json").read_text(encoding="utf-8"))
    sampled_rows = [next(row for row in train_rows if row["level_id"] == level_id)
                    for level_id in manifest["requested_ids"]]
    compositions, spatials, targets, weights = [], [], [], []
    for index, row in enumerate(sampled_rows, start=1):
        record = json.loads((DEFAULT_CACHE / f"{row['level_id']}.json").read_text(encoding="utf-8"))
        composition, spatial = build_features(record["header"]["4"])
        compositions.append(composition)
        spatials.append(spatial)
        targets.append(row["target"] @ torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32))
        weights.append(q_population[q_for_id[row["level_id"]]] / manifest["per_quantile"])
        if index % 10 == 0:
            print(f"processed structural features for {index}/{len(sampled_rows)} levels", flush=True)
    y = torch.stack(targets)
    weights = torch.tensor(weights, dtype=torch.float32)
    features = {"global object counts": torch.stack(compositions),
                "spatial object counts": torch.stack(spatials)}
    outer_fold = torch.arange(len(y)) % 5

    for name, x in features.items():
        predictions = torch.empty_like(y)
        selected_alphas = []
        for fold in range(5):
            train = torch.where(outer_fold != fold)[0]
            valid = torch.where(outer_fold == fold)[0]
            inner_fold = torch.arange(len(train)) % 4
            alpha_scores = {alpha: [] for alpha in ALPHAS}
            for inner in range(4):
                fit_idx = train[inner_fold != inner]
                tune_idx = train[inner_fold == inner]
                prepared = prepare_ridge(x[fit_idx], y[fit_idx], weights[fit_idx])
                tune_predictions = predict_ridge(prepared, x[tune_idx], ALPHAS)
                for alpha, pred in zip(ALPHAS, tune_predictions):
                    alpha_scores[alpha].append(weighted_score(
                        pred, y[tune_idx], weights[tune_idx]))
            candidates = [(sum(s[1] for s in scores) / len(scores),
                           -sum(s[0] for s in scores) / len(scores), alpha)
                          for alpha, scores in alpha_scores.items()]
            alpha = max(candidates)[2]
            selected_alphas.append(alpha)
            prepared = prepare_ridge(x[train], y[train], weights[train])
            predictions[valid] = predict_ridge(prepared, x[valid], [alpha])[0]
        fold_scores = [weighted_score(predictions[outer_fold == fold], y[outer_fold == fold],
                                      weights[outer_fold == fold]) for fold in range(5)]
        print(f"{name}: selected_alphas={selected_alphas}, "
              f"weighted_CV_MAE={sum(s[0] for s in fold_scores) / 5:.4f}, "
              f"weighted_CV_within3={sum(s[1] for s in fold_scores) / 5:.4f}")

    baseline_predictions = torch.empty_like(y)
    for fold in range(5):
        train = outer_fold != fold
        valid = ~train
        baseline_predictions[valid] = weighted_median(y[train], weights[train])
    fold_scores = [weighted_score(baseline_predictions[outer_fold == fold], y[outer_fold == fold],
                                  weights[outer_fold == fold]) for fold in range(5)]
    print("weighted train-median baseline: "
          f"MAE={sum(s[0] for s in fold_scores) / 5:.4f}, "
          f"within3={sum(s[1] for s in fold_scores) / 5:.4f}")
    print("pilot levels:", len(y), "sample-weight range:",
          (float(weights.min()), float(weights.max())))


if __name__ == "__main__":
    main()
