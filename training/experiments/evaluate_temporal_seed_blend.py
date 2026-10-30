"""Fold-held-out blend of the strongest temporal Transformer seeds."""
import random
from pathlib import Path

import torch

from cross_validate_ensemble import score, window_mass_center
from ensemble_calibration_experiments import make_components
from knn_dinov2 import predict as knn_predict
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "training" / "features_dinov2_base_4fps"
SEED42_PATH = Path(__file__).resolve().parent / "dinov2_1s_batched_transformer_within3_w05.pt"
SEED7_PATH = Path(__file__).resolve().parent / "dinov2_1s_batched_transformer_seed7_within3_w05.pt"


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    split_size = max(1, round(len(rows) * 0.1 + 0.499999))
    train_rows, valid_rows = rows[2 * split_size:], rows[split_size:2 * split_size]
    train_targets = torch.stack([row["target"] for row in train_rows])
    valid_targets = torch.stack([row["target"] for row in valid_rows])
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    actual = valid_targets @ tiers

    print("Rebuilding visual baseline on CPU...", flush=True)
    groups, valid_features, _ = make_components(valid_rows)
    train_features = []
    for row in train_rows:
        values = torch.load(FEATURE_DIR / f"{row['level_id']}.pt",
                            map_location="cpu", weights_only=True)["features"].float()
        train_features.append(values.mean(0))
    train_features = torch.nn.functional.normalize(torch.stack(train_features), dim=-1)
    retrieval = knn_predict(valid_features @ train_features.T, train_targets, 20, 0.02)
    visual = 0.9 * groups.mean(0) + 0.1 * retrieval
    temporal42 = torch.load(SEED42_PATH, map_location="cpu", weights_only=True)[
        "validation_probabilities"].float()
    temporal7 = torch.load(SEED7_PATH, map_location="cpu", weights_only=True)[
        "validation_probabilities"].float()

    fold_ids = torch.arange(len(valid_rows)) % 5
    baseline_scores, ensemble_scores, selected = [], [], []
    print("seed7 standalone:", score(window_mass_center(temporal7, 3), actual))
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        baseline = 0.75 * visual + 0.25 * temporal42
        baseline_result = score(window_mass_center(baseline[held], 3), actual[held])
        best = None
        for weight42_i in range(11):
            weight42 = weight42_i * 0.05
            for weight7_i in range(11 - weight42_i):
                weight7 = weight7_i * 0.05
                visual_weight = 1 - weight42 - weight7
                mixed = visual_weight * visual[tune] + weight42 * temporal42[tune] + weight7 * temporal7[tune]
                result = score(window_mass_center(mixed, 3), actual[tune])
                choice = (result[1], -result[0], -weight7, -weight42)
                if best is None or choice > best[0]:
                    best = (choice, weight42, weight7)
        _, weight42, weight7 = best
        visual_weight = 1 - weight42 - weight7
        mixed = visual_weight * visual[held] + weight42 * temporal42[held] + weight7 * temporal7[held]
        result = score(window_mass_center(mixed, 3), actual[held])
        baseline_scores.append(baseline_result)
        ensemble_scores.append(result)
        selected.append((visual_weight, weight42, weight7))
        print(f"fold {fold + 1}: baseline={baseline_result}, three-source={result}, "
              f"weights={selected[-1]}", flush=True)

    average = lambda values: tuple(sum(x[i] for x in values) / 5 for i in range(2))
    print(f"5-fold current best (MAE, within3): {average(baseline_scores)}")
    print(f"5-fold three-source ensemble (MAE, within3): {average(ensemble_scores)}")
    print(f"weights (visual, seed42, seed7): {selected}")


if __name__ == "__main__":
    main()
