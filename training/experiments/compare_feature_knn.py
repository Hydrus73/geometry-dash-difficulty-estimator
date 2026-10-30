"""Compare train-only-tuned KNN distributions from cached visual features."""
import math
import random
from pathlib import Path

import torch

from cross_validate_ensemble import score, window_mass_center
from evaluate_validation import MODELS, predict
from training.train import (
    DEFAULT_CSV, DEFAULT_FEATURE_DIR, NUM_CLASSES, read_rows,
)


ROOT = Path(__file__).resolve().parents[2]
FEATURES = {
    "VideoMAE": DEFAULT_FEATURE_DIR,
    "DINO-small": ROOT / "training" / "features_dinov2_small_4fps",
    "DINO-base-4s": ROOT / "training" / "features_dinov2_base_4fps",
    "DINO-base-1s": ROOT / "training" / "features_dinov2_base_4fps_1s",
    "DINO-base-8s": ROOT / "training" / "features_dinov2_base_8s",
    "CLIP-ViT-B16": ROOT / "training" / "features_clip_vitb16_4fps_1s",
}
K_VALUES = (10, 20, 40, 80)
TEMPERATURES = (0.01, 0.02, 0.04, 0.08)


def load_means(rows, feature_dir):
    means = []
    for row in rows:
        path = feature_dir / f"{row['level_id']}.pt"
        cache = torch.load(path, map_location="cpu", weights_only=True)
        means.append(cache["features"].float().mean(0))
    return torch.nn.functional.normalize(torch.stack(means), dim=-1)


def knn_distribution(similarity, labels, k, temperature):
    values, indices = similarity.topk(k, dim=-1)
    weights = torch.softmax((values - values[:, :1]) / temperature, dim=-1)
    return (labels[indices] * weights.unsqueeze(-1)).sum(1)


def select_knn_params(representations, labels, actual):
    """Tune only on a random inner holdout from the fixed training partition."""
    generator = torch.Generator().manual_seed(21)
    permutation = torch.randperm(len(labels), generator=generator)
    holdout_size = math.ceil(len(labels) * 0.1)
    inner_valid, inner_train = permutation[:holdout_size], permutation[holdout_size:]
    similarity = representations[inner_valid] @ representations[inner_train].T
    best = None
    for k in K_VALUES:
        for temperature in TEMPERATURES:
            distribution = knn_distribution(similarity, labels[inner_train], k, temperature)
            result = score(window_mass_center(distribution, 3), actual[inner_valid])
            candidate = (result[1], -result[0], -k, -temperature)
            if best is None or candidate > best[0]:
                best = (candidate, k, temperature)
    return best[1], best[2], best[0][:2]


def blend_cv(base, candidate, actual):
    fold_ids = torch.arange(len(actual)) % 5
    fold_scores, fold_weights = [], []
    for fold in range(5):
        train, valid = fold_ids != fold, fold_ids == fold
        best = None
        for weight in torch.arange(0, 0.501, 0.05):
            distribution = (1 - weight) * base[train] + weight * candidate[train]
            result = score(window_mass_center(distribution, 3), actual[train])
            choice = (result[1], -result[0], -float(weight))
            if best is None or choice > best[0]:
                best = (choice, float(weight))
        weight = best[1]
        distribution = (1 - weight) * base[valid] + weight * candidate[valid]
        result = score(window_mass_center(distribution, 3), actual[valid])
        fold_scores.append(result)
        fold_weights.append(weight)
    average = tuple(sum(result[i] for result in fold_scores) / 5 for i in range(2))
    return average, fold_weights, fold_scores


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    split_size = max(1, round(len(rows) * 0.1 + 0.499999))
    train_rows = rows[2 * split_size:]
    valid_rows = rows[split_size:2 * split_size]
    train_labels = torch.stack([row["target"] for row in train_rows])
    valid_labels = torch.stack([row["target"] for row in valid_rows])
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    train_actual, valid_actual = train_labels @ tiers, valid_labels @ tiers

    print("Building the current visual-group ensemble on CPU...", flush=True)
    grouped_names = (
        ("videomae_pooled", "videomae_mae", "videomae_categorical"),
        ("dino_small_ordinal", "dino_small_ordinal_seed7", "dino_small_categorical"),
        ("dino_base_4s_ordinal", "dino_base_4s_categorical",
         "dino_base_4s_categorical_seed7", "dino_base_4s_dropout06"),
    )
    raw_names = {name for group in grouped_names for name in group}
    raw = {name: predict(name, valid_rows, torch.device("cpu")) for name in raw_names}
    groups = torch.stack([
        torch.stack([raw[name] for name in group]).mean(0)
        for group in grouped_names
    ])
    video_groups = groups.mean(0)

    base_dir = FEATURES["DINO-base-4s"]
    train_features = load_means(train_rows, base_dir)
    valid_features = load_means(valid_rows, base_dir)
    base_knn = knn_distribution(valid_features @ train_features.T, train_labels, 20, 0.02)
    base = 0.75 * (0.9 * video_groups + 0.1 * base_knn)
    temporal = torch.load(
        Path(__file__).resolve().parent / "dinov2_1s_batched_transformer_within3_w05.pt",
        map_location="cpu", weights_only=True,
    )["validation_probabilities"].float()
    base = base + 0.25 * temporal
    baseline = score(window_mass_center(base, 3), valid_actual)
    fold_ids = torch.arange(len(valid_actual)) % 5
    baseline_folds = tuple(sum(
        score(window_mass_center(base[fold_ids == fold], 3), valid_actual[fold_ids == fold])[i]
        for fold in range(5)
    ) / 5 for i in range(2))
    print(f"fixed blend full validation={baseline}; five-fold={baseline_folds}")

    for name, feature_dir in FEATURES.items():
        print(f"Evaluating {name} KNN...", flush=True)
        train_repr = train_features if name == "DINO-base-4s" else load_means(train_rows, feature_dir)
        valid_repr = valid_features if name == "DINO-base-4s" else load_means(valid_rows, feature_dir)
        k, temperature, inner_score = select_knn_params(
            train_repr, train_labels, train_actual
        )
        distribution = knn_distribution(valid_repr @ train_repr.T, train_labels,
                                         k, temperature)
        standalone = score(window_mass_center(distribution, 3), valid_actual)
        cv_score, weights, folds = blend_cv(base, distribution, valid_actual)
        print(f"{name}: train-holdout k={k}, temp={temperature:g}, "
              f"holdout within3={inner_score[0]:.4f}; standalone={standalone}; "
              f"fold blend={cv_score}; weights={weights}")


if __name__ == "__main__":
    main()
