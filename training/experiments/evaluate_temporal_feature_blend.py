"""Evaluate whether the 8-second DINO temporal model complements the baseline."""
import argparse
import random
from pathlib import Path

import torch

from cross_validate_ensemble import score, window_mass_center
from ensemble_calibration_experiments import make_components
from knn_dinov2 import predict as knn_predict
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "training" / "features_dinov2_base_4fps"
TEMPORAL_8S = Path(__file__).resolve().parent / "dinov2_8s_batched_transformer_within3_w05.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=TEMPORAL_8S)
    parser.add_argument("--folds", type=int, default=5,
                        help="number of held-out validation folds (default: 5)")
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("folds must be at least 2")
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    split_size = max(1, round(len(rows) * 0.1 + 0.499999))
    train_rows, valid_rows = rows[2 * split_size:], rows[split_size:2 * split_size]
    train_targets = torch.stack([row["target"] for row in train_rows])
    valid_targets = torch.stack([row["target"] for row in valid_rows])
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    actual = valid_targets @ tiers

    print("Building fixed baseline distribution on CPU...", flush=True)
    groups, valid_features, temporal = make_components(valid_rows)
    train_features = []
    for row in train_rows:
        features = torch.load(FEATURE_DIR / f"{row['level_id']}.pt",
                              map_location="cpu", weights_only=True)["features"].float()
        train_features.append(features.mean(0))
    train_features = torch.nn.functional.normalize(torch.stack(train_features), dim=-1)
    retrieval = knn_predict(valid_features @ train_features.T, train_targets, 20, 0.02)
    baseline = 0.75 * (0.9 * groups.mean(0) + 0.1 * retrieval) + 0.25 * temporal

    new_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    candidate = new_checkpoint["validation_probabilities"].float()
    model_name = new_checkpoint.get("model", args.checkpoint.name)
    print("candidate checkpoint:", model_name, new_checkpoint["epoch"],
          new_checkpoint["validation_scores"])
    print("standalone window3:", score(window_mass_center(candidate, 3), actual))
    fixed_candidate = 0.95 * baseline + 0.05 * candidate
    print("full validation fixed 5% blend:",
          score(window_mass_center(fixed_candidate, 3), actual))

    fold_ids = torch.arange(len(valid_rows)) % args.folds
    base_scores, blend_scores, chosen_weights = [], [], []
    for fold in range(args.folds):
        tune, held = fold_ids != fold, fold_ids == fold
        baseline_score = score(window_mass_center(baseline[held], 3), actual[held])
        best = None
        for weight in torch.arange(0, 0.501, 0.025):
            mixed = (1 - weight) * baseline[tune] + weight * candidate[tune]
            result = score(window_mass_center(mixed, 3), actual[tune])
            choice = (result[1], -result[0], -float(weight))
            if best is None or choice > best[0]:
                best = (choice, float(weight))
        weight = best[1]
        mixed = (1 - weight) * baseline[held] + weight * candidate[held]
        result = score(window_mass_center(mixed, 3), actual[held])
        base_scores.append(baseline_score)
        blend_scores.append(result)
        chosen_weights.append(weight)
        print(f"fold {fold + 1}: baseline={baseline_score}, blend={result}, "
              f"candidate weight={weight:.3f}", flush=True)

    average = lambda values: tuple(sum(x[i] for x in values) / args.folds for i in range(2))
    print(f"{args.folds}-fold baseline (MAE, within3): {average(base_scores)}")
    print(f"{args.folds}-fold candidate blend (MAE, within3): {average(blend_scores)}")
    print(f"selected weights: {chosen_weights}")


if __name__ == "__main__":
    main()
