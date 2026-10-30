"""Test learned-metric KNN seeds and their averages in held-out validation folds."""
import itertools
import math
import random
from pathlib import Path

import torch

from cross_validate_ensemble import score, window_mass_center
from ensemble_calibration_experiments import make_components
from knn_dinov2 import predict as knn_predict
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "training" / "features_dinov2_base_4fps"
CHECKPOINTS = {
    "seed42": ROOT / "training" / "experiments" / "dinov2_metric_knn.pt",
    "seed7": ROOT / "training" / "experiments" / "dinov2_metric_knn_seed7.pt",
    "seed123": ROOT / "training" / "experiments" / "dinov2_metric_knn_seed123.pt",
}


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    train_rows, valid_rows = rows[2 * boundary:], rows[boundary:2 * boundary]
    targets = torch.stack([row["target"] for row in train_rows])
    actual = torch.stack([row["target"] for row in valid_rows]) @ torch.arange(
        1, NUM_CLASSES + 1, dtype=torch.float32)

    groups, valid_features, temporal = make_components(valid_rows)
    train_features = []
    for row in train_rows:
        item = torch.load(FEATURE_DIR / f"{row['level_id']}.pt", map_location="cpu",
                          weights_only=True)
        train_features.append(item["features"].float().mean(0))
    train_features = torch.nn.functional.normalize(torch.stack(train_features), dim=-1)
    similarities = valid_features @ train_features.T
    retrieval = knn_predict(similarities, targets, 20, 0.02)
    baseline = 0.75 * (0.9 * groups.mean(0) + 0.1 * retrieval) + 0.25 * temporal

    sources = {}
    expected_ids = [row["level_id"] for row in valid_rows]
    for name, path in CHECKPOINTS.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["validation_level_ids"] != expected_ids:
            raise ValueError(f"validation order mismatch in {path}")
        sources[name] = checkpoint["validation_probabilities"].float()
    names = list(sources)
    for count in (2, 3):
        for combo in itertools.combinations(names, count):
            sources["average_" + "_".join(combo)] = torch.stack(
                [sources[name] for name in combo]
            ).mean(0)

    folds = torch.arange(len(valid_rows)) % 5
    baseline_scores, candidate_scores, selections = [], [], []
    for fold in range(5):
        tune, held = folds != fold, folds == fold
        baseline_scores.append(score(window_mass_center(baseline[held], 3), actual[held]))
        best = None
        for name, distribution in sources.items():
            for weight in torch.arange(0, 0.301, 0.025):
                blended = (1 - weight) * baseline[tune] + weight * distribution[tune]
                result = score(window_mass_center(blended, 3), actual[tune])
                key = (result[1], -result[0], -float(weight))
                if best is None or key > best[0]:
                    best = (key, name, float(weight))
        name, weight = best[1], best[2]
        held_blend = (1 - weight) * baseline[held] + weight * sources[name][held]
        candidate_scores.append(score(window_mass_center(held_blend, 3), actual[held]))
        selections.append((name, weight))
        print(f"fold {fold + 1}: baseline={baseline_scores[-1]}, "
              f"blend={candidate_scores[-1]}, source={name}, weight={weight:.3f}")

    average = lambda items: tuple(sum(item[i] for item in items) / len(items) for i in range(2))
    print("5-fold baseline (MAE, within3):", average(baseline_scores))
    print("5-fold selected-seed blend (MAE, within3):", average(candidate_scores))
    print("selections:", selections)
    for name, distribution in sources.items():
        result = score(window_mass_center(distribution, 3), actual)
        print(f"standalone {name}: {result}")
    print("Held-out test partition was not evaluated.")


if __name__ == "__main__":
    main()
