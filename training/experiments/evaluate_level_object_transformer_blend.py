"""Check whether ordered level-object tokens complement the fixed video ensemble."""
import math
import random
from pathlib import Path

import torch

from cross_validate_ensemble import window_mass_center
from ensemble_calibration_experiments import make_components
from knn_dinov2 import predict as knn_predict
from training.experiments.download_gd_level import DEFAULT_CACHE
from training.experiments.train_level_object_transformer import select_sample
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "training" / "features_dinov2_base_4fps"
CHECKPOINT = ROOT / "training" / "experiments" / "level_object_transformer.pt"


def weighted_score(prediction, actual, weights):
    error = (prediction - actual).abs()
    weights = weights / weights.sum()
    return ((error * weights).sum().item(), ((error <= 3).float() * weights).sum().item())


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    train_rows = rows[2 * boundary:]
    full_valid_rows = rows[boundary:2 * boundary]
    pilot_rows, weights = select_sample("validation", rows, boundary)
    target_by_id = {row["level_id"]: row["target"] for row in pilot_rows}
    candidate = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    candidate_ids = candidate["validation_level_ids"]
    assert candidate_ids == [row["level_id"] for row in pilot_rows]
    candidate_distribution = candidate["validation_probabilities"].float()

    print("building fixed video/KNN/temporal ensemble on CPU", flush=True)
    groups, validation_features, temporal = make_components(full_valid_rows)
    train_features = []
    for row in train_rows:
        cache = torch.load(FEATURE_DIR / f"{row['level_id']}.pt", map_location="cpu",
                           weights_only=True)
        train_features.append(cache["features"].float().mean(0))
    train_features = torch.nn.functional.normalize(torch.stack(train_features), dim=-1)
    train_targets = torch.stack([row["target"] for row in train_rows])
    retrieval = knn_predict(validation_features @ train_features.T, train_targets, 20, 0.02)
    baseline_full = 0.75 * (0.9 * groups.mean(0) + 0.1 * retrieval) + 0.25 * temporal
    full_position = {row["level_id"]: index for index, row in enumerate(full_valid_rows)}
    pilot_position = torch.tensor([full_position[row["level_id"]] for row in pilot_rows])
    baseline = baseline_full[pilot_position]

    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    actual = torch.stack([target_by_id[row["level_id"]] for row in pilot_rows]) @ tiers
    print("candidate standalone:", weighted_score(
        window_mass_center(candidate_distribution, 3), actual, weights))

    folds = torch.arange(len(pilot_rows)) % 5
    base_scores, blend_scores, chosen_weights = [], [], []
    for fold in range(5):
        tune, held = folds != fold, folds == fold
        base_scores.append(weighted_score(
            window_mass_center(baseline[held], 3), actual[held], weights[held]))
        best = None
        for weight in torch.arange(0, 0.501, 0.025):
            mixed = (1 - weight) * baseline[tune] + weight * candidate_distribution[tune]
            result = weighted_score(window_mass_center(mixed, 3), actual[tune], weights[tune])
            choice = (result[1], -result[0], -float(weight))
            if best is None or choice > best[0]:
                best = (choice, float(weight))
        weight = best[1]
        mixed = (1 - weight) * baseline[held] + weight * candidate_distribution[held]
        result = weighted_score(window_mass_center(mixed, 3), actual[held], weights[held])
        blend_scores.append(result)
        chosen_weights.append(weight)
        print(f"fold {fold + 1}: baseline={base_scores[-1]}, blend={result}, weight={weight:.3f}")
    average = lambda items: tuple(sum(value[i] for value in items) / len(items) for i in range(2))
    print("5-fold baseline (MAE, within3):", average(base_scores))
    print("5-fold blend (MAE, within3):", average(blend_scores))
    print("selected weights:", chosen_weights)
    print("held-out test partition was not evaluated.")


if __name__ == "__main__":
    main()
