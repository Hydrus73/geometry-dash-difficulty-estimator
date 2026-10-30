"""Nested-fold KNN probe using only other validation folds as neighbors."""
import random
from pathlib import Path

import torch

from cross_validate_ensemble import score, window_mass_center
from ensemble_calibration_experiments import make_components
from knn_dinov2 import predict as knn_predict
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


ROOT = Path(__file__).resolve().parents[2]
CLIP_DIR = ROOT / "training" / "features_clip_vitb16_4fps_1s"
BASE_DIR = ROOT / "training" / "features_dinov2_base_4fps"
K_VALUES = (10, 20, 40, 80)
TEMPERATURES = (0.01, 0.02, 0.04, 0.08)


def load_means(rows, directory):
    values = []
    for row in rows:
        features = torch.load(directory / f"{row['level_id']}.pt",
                              map_location="cpu", weights_only=True)["features"].float()
        values.append(features.mean(0))
    return torch.nn.functional.normalize(torch.stack(values), dim=-1)


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = max(1, round(len(rows) * 0.1 + 0.499999))
    train_rows, valid_rows = rows[2 * boundary:], rows[boundary:2 * boundary]
    train_targets = torch.stack([row["target"] for row in train_rows])
    valid_targets = torch.stack([row["target"] for row in valid_rows])
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    actual = valid_targets @ tiers

    print("Loading validation CLIP features and current baseline...", flush=True)
    clip = load_means(valid_rows, CLIP_DIR)
    groups, base_train_features, temporal = make_components(valid_rows)
    train_dino = load_means(train_rows, BASE_DIR)
    retrieval = knn_predict(base_train_features @ train_dino.T, train_targets, 20, 0.02)
    baseline = 0.75 * (0.9 * groups.mean(0) + 0.1 * retrieval) + 0.25 * temporal

    fold_ids = torch.arange(len(valid_rows)) % 5
    baseline_scores, clip_scores, clip_blend_scores, choices = [], [], [], []
    for outer_fold in range(5):
        outer_train = fold_ids != outer_fold
        outer_valid = ~outer_train
        local = torch.where(outer_train)[0]
        inner_ids = torch.arange(len(local)) % 4
        oof_by_config = None
        for inner_fold in range(4):
            fit = inner_ids != inner_fold
            holdout = ~fit
            fit_local, holdout_local = local[fit], local[holdout]
            similarity = clip[holdout_local] @ clip[fit_local].T
            # Per-combination predictions allow the blend weight and KNN
            # hyperparameters to be selected without scoring the outer fold.
            candidates = {}
            for k in K_VALUES:
                for temperature in TEMPERATURES:
                    candidates[(k, temperature)] = knn_predict(
                        similarity, valid_targets[fit_local], k, temperature,
                    )
            # Store each inner holdout prediction in a matching tensor.
            if inner_fold == 0:
                oof_by_config = {key: torch.zeros_like(valid_targets[local]) for key in candidates}
            for key, prediction in candidates.items():
                oof_by_config[key][holdout] = prediction

        best = None
        for (k, temperature), prediction in oof_by_config.items():
            for weight in torch.arange(0, 0.501, 0.05):
                mixed = (1 - weight) * baseline[local] + weight * prediction
                result = score(window_mass_center(mixed, 3), actual[local])
                choice = (result[1], -result[0], -float(weight), -k, -temperature)
                if best is None or choice > best[0]:
                    best = (choice, k, temperature, float(weight), prediction)
        _, k, temperature, weight, oof_prediction = best

        # Recompute KNN for the outer held-out fold using only outer-train labels.
        outer_similarity = clip[outer_valid] @ clip[outer_train].T
        held_knn = knn_predict(outer_similarity, valid_targets[outer_train], k, temperature)
        mixed = (1 - weight) * baseline[outer_valid] + weight * held_knn
        score_knn = score(window_mass_center(held_knn, 3), actual[outer_valid])
        score_blend = score(window_mass_center(mixed, 3), actual[outer_valid])
        score_baseline = score(window_mass_center(baseline[outer_valid], 3), actual[outer_valid])
        clip_scores.append(score_knn)
        clip_blend_scores.append(score_blend)
        baseline_scores.append(score_baseline)
        choices.append((k, temperature, weight))
        print(f"fold {outer_fold + 1}: baseline={score_baseline}, CLIP-KNN={score_knn}, "
              f"blend={score_blend}, selected=(k={k}, temp={temperature:g}, "
              f"weight={weight:.2f})", flush=True)

    average = lambda values: tuple(sum(x[i] for x in values) / 5 for i in range(2))
    print(f"5-fold baseline (MAE, within3): {average(baseline_scores)}")
    print(f"5-fold CLIP-KNN (MAE, within3): {average(clip_scores)}")
    print(f"5-fold CLIP-KNN blend (MAE, within3): {average(clip_blend_scores)}")
    print(f"selected parameters: {choices}")


if __name__ == "__main__":
    main()
