"""Nested CV stacker using confidence features from independent visual models."""
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate_validation import MODELS, predict as model_predict
from knn_dinov2 import make_representations, predict as knn_predict
from training.train import DEFAULT_CSV, DEFAULT_FEATURE_DIR, NUM_CLASSES, read_rows


SMALL_DIR = ROOT / "training" / "features_dinov2_small_4fps"
BASE_DIR = ROOT / "training" / "features_dinov2_base_4fps"
BASE_1S_DIR = ROOT / "training" / "features_dinov2_base_4fps_1s"


def score(prediction, target):
    error = (prediction - target).abs()
    return error.mean().item(), (error <= 3).float().mean().item()


def fit_predict(x_train, y_train, x_eval, alpha):
    mean = x_train.mean(0)
    scale = x_train.std(0).clamp_min(1e-5)
    x = (x_train - mean) / scale
    z = (x_eval - mean) / scale
    y_mean = y_train.mean()
    y = y_train - y_mean
    beta = torch.linalg.solve(x.T @ x + alpha * torch.eye(x.shape[1]), x.T @ y)
    return z @ beta + y_mean


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    rows = rows[boundary:2 * boundary]
    actual = torch.stack([row["target"] for row in rows]) @ torch.arange(1, NUM_CLASSES + 1.)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")
    raw = {name: model_predict(name, rows, device) for name in MODELS}
    groups = {
        "VideoMAE": torch.stack([raw[n] for n in (
            "videomae_pooled", "videomae_mae", "videomae_categorical"
        )]).mean(0),
        "DINO-small": torch.stack([raw[n] for n in (
            "dino_small_ordinal", "dino_small_ordinal_seed7", "dino_small_categorical"
        )]).mean(0),
        "DINO-base": torch.stack([raw[n] for n in (
            "dino_base_4s_ordinal", "dino_base_4s_categorical",
            "dino_base_4s_categorical_seed7", "dino_base_4s_dropout06"
        )]).mean(0),
    }
    all_rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(all_rows)
    split_size = math.ceil(len(all_rows) * 0.1)
    train_idx = torch.arange(2 * split_size, len(all_rows))
    valid_idx = torch.arange(split_size, 2 * split_size)
    pooled, variability = [], []
    for row in all_rows:
        cache = torch.load(BASE_DIR / f"{row['level_id']}.pt", map_location="cpu", weights_only=True)
        features = cache["features"].float()
        pooled.append(features.mean(0))
        variability.append(features.std(0, unbiased=False))
    pooled, variability = torch.stack(pooled), torch.stack(variability)
    knn_features = make_representations(pooled, variability)["mean"]
    similarity = knn_features[valid_idx] @ knn_features[train_idx].T
    all_targets = torch.stack([row["target"] for row in all_rows])
    knn_distribution = knn_predict(similarity, all_targets[train_idx], 20, 0.02)

    distributions = list(groups.values()) + [knn_distribution]
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    feature_columns = []
    for distribution in distributions:
        mean = distribution @ tiers
        median = torch.searchsorted(
            distribution.cumsum(-1), torch.full((len(distribution), 1), 0.5)
        ).squeeze(-1).float() + 1
        variance = ((distribution * (tiers - mean[:, None]).square()).sum(-1)).sqrt()
        entropy = -(distribution.clamp_min(1e-8) * distribution.clamp_min(1e-8).log()).sum(-1)
        feature_columns.extend((mean, median, variance, entropy))
    x = torch.stack(feature_columns, dim=-1)

    outer_fold = torch.arange(len(rows)) % 5
    alpha_values = (0.01, 0.1, 1.0, 10.0, 100.0)
    outer_scores = []
    for fold in range(5):
        train = torch.where(outer_fold != fold)[0]
        valid = torch.where(outer_fold == fold)[0]
        inner_folds = torch.arange(len(train)) % 4
        candidates = []
        for alpha in alpha_values:
            scores = []
            for inner_fold in range(4):
                inner_train = train[inner_folds != inner_fold]
                inner_valid = train[inner_folds == inner_fold]
                prediction = fit_predict(x[inner_train], actual[inner_train], x[inner_valid], alpha)
                scores.append(score(prediction.clamp(1, NUM_CLASSES), actual[inner_valid]))
            candidates.append((sum(s[1] for s in scores) / 4,
                               -sum(s[0] for s in scores) / 4, alpha))
        selected_alpha = max(candidates)[2]
        prediction = fit_predict(x[train], actual[train], x[valid], selected_alpha)
        result = score(prediction.clamp(1, NUM_CLASSES), actual[valid])
        outer_scores.append(result)
        print(f"fold {fold + 1}: alpha={selected_alpha:g}, MAE={result[0]:.4f}, "
              f"within3={result[1]:.4f}")

    fixed_mean_distribution = torch.stack(list(groups.values())).mean(0)
    fixed_median = torch.searchsorted(
        fixed_mean_distribution.cumsum(-1), torch.full((len(rows), 1), 0.5)
    ).squeeze(-1).float() + 1
    knn_blend = 0.9 * fixed_mean_distribution + 0.1 * knn_distribution
    knn_median = torch.searchsorted(
        knn_blend.cumsum(-1), torch.full((len(rows), 1), 0.5)
    ).squeeze(-1).float() + 1
    baseline_scores = []
    knn_scores = []
    for fold in range(5):
        valid = outer_fold == fold
        baseline_scores.append(score(fixed_median[valid], actual[valid]))
        knn_scores.append(score(knn_median[valid], actual[valid]))
    print("nested ridge stack 5-fold: MAE={:.4f}, within3={:.4f}".format(
        sum(s[0] for s in outer_scores) / 5, sum(s[1] for s in outer_scores) / 5))
    print("fixed group-mean median 5-fold: MAE={:.4f}, within3={:.4f}".format(
        sum(s[0] for s in baseline_scores) / 5, sum(s[1] for s in baseline_scores) / 5))
    print("fixed 90/10 KNN median 5-fold: MAE={:.4f}, within3={:.4f}".format(
        sum(s[0] for s in knn_scores) / 5, sum(s[1] for s in knn_scores) / 5))


if __name__ == "__main__":
    main()
