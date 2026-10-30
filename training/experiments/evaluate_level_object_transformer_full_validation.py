"""Score the pilot-trained object Transformer on the complete validation split."""
import argparse
import json
import math
import random
from pathlib import Path

import torch

from cross_validate_ensemble import score, window_mass_center
from ensemble_calibration_experiments import make_components
from knn_dinov2 import predict as knn_predict
from training.experiments.download_gd_level import DEFAULT_CACHE
from training.experiments.train_level_object_transformer import (
    ObjectTransformer, load_sequences, predict,
)
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "training" / "features_dinov2_base_4fps"
CHECKPOINT = ROOT / "training" / "experiments" / "level_object_transformer.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT,
                        help=f"model checkpoint (default: {CHECKPOINT})")
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    train_rows, valid_rows = rows[2 * boundary:], rows[boundary:2 * boundary]
    manifest = json.loads((DEFAULT_CACHE / "full_validation_manifest.json").read_text())
    if len(manifest["completed"]) != len(valid_rows):
        raise RuntimeError("Full validation level strings are not all cached yet")
    pilot_manifest = json.loads((DEFAULT_CACHE / "validation_pilot_manifest.json").read_text())
    pilot_ids = set(pilot_manifest["requested_ids"])
    pilot_mask = torch.tensor([row["level_id"] in pilot_ids for row in valid_rows])

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    sequences = load_sequences(valid_rows, "full_validation")
    model = ObjectTransformer()
    model.load_state_dict(payload["state_dict"])
    candidate = predict(model, sequences,
                        torch.stack([row["target"] for row in valid_rows]),
                        torch.ones(len(valid_rows)), torch.arange(len(valid_rows)),
                        torch.device("cpu"))

    print("building fixed video/KNN/temporal ensemble on CPU", flush=True)
    groups, validation_features, temporal = make_components(valid_rows)
    train_features = []
    for row in train_rows:
        cache = torch.load(FEATURE_DIR / f"{row['level_id']}.pt", map_location="cpu",
                           weights_only=True)
        train_features.append(cache["features"].float().mean(0))
    train_features = torch.nn.functional.normalize(torch.stack(train_features), dim=-1)
    train_targets = torch.stack([row["target"] for row in train_rows])
    retrieval = knn_predict(validation_features @ train_features.T, train_targets, 20, 0.02)
    baseline = 0.75 * (0.9 * groups.mean(0) + 0.1 * retrieval) + 0.25 * temporal
    actual = torch.stack([row["target"] for row in valid_rows]) @ torch.arange(
        1, NUM_CLASSES + 1, dtype=torch.float32)

    def report(label, mask):
        base_score = score(window_mass_center(baseline[mask], 3), actual[mask])
        model_score = score(window_mass_center(candidate[mask], 3), actual[mask])
        blended = 0.9 * baseline[mask] + 0.1 * candidate[mask]
        blend_score = score(window_mass_center(blended, 3), actual[mask])
        print(f"{label}: baseline={base_score}, object={model_score}, 10% blend={blend_score}")

    report("all validation levels", torch.ones(len(valid_rows), dtype=torch.bool))
    report("pilot levels used to select blend weight", pilot_mask)
    report("remaining validation levels", ~pilot_mask)
    fold_ids = torch.arange(len(valid_rows)) % 5
    base_scores, blend_scores, selected_weights = [], [], []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        base_scores.append(score(window_mass_center(baseline[held], 3), actual[held]))
        best = None
        for weight in torch.arange(0, 0.501, 0.025):
            mixed = (1 - weight) * baseline[tune] + weight * candidate[tune]
            candidate_score = score(window_mass_center(mixed, 3), actual[tune])
            key = (candidate_score[1], -candidate_score[0], -float(weight))
            if best is None or key > best[0]:
                best = (key, float(weight))
        weight = best[1]
        mixed = (1 - weight) * baseline[held] + weight * candidate[held]
        blend_scores.append(score(window_mass_center(mixed, 3), actual[held]))
        selected_weights.append(weight)
        print(f"fold {fold + 1}: baseline={base_scores[-1]}, held-out blend={blend_scores[-1]}, "
              f"selected weight={weight:.3f}")
    average = lambda results: tuple(sum(item[i] for item in results) / len(results)
                                    for i in range(2))
    print("5-fold baseline (MAE, within3):", average(base_scores))
    print("5-fold held-out blend (MAE, within3):", average(blend_scores))
    print("5-fold selected weights:", selected_weights)

    def readouts(distribution):
        tiers = torch.arange(1, NUM_CLASSES + 1, dtype=distribution.dtype)
        mean = distribution @ tiers
        median = torch.searchsorted(
            distribution.cumsum(-1), torch.full((len(distribution), 1), 0.5)
        ).squeeze(-1).float() + 1
        return {
            "window": window_mass_center(distribution, 3),
            "mean": mean,
            "median": median,
        }

    base_readouts, object_readouts = readouts(baseline), readouts(candidate)
    readout_scores, readout_weights = [], []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        for base_name, base_point in base_readouts.items():
            for object_name, object_point in object_readouts.items():
                for weight in torch.arange(0, 0.501, 0.025):
                    prediction = (1 - weight) * base_point[tune] + weight * object_point[tune]
                    candidate_score = score(prediction, actual[tune])
                    key = (candidate_score[1], -candidate_score[0], -float(weight))
                    if best is None or key > best[0]:
                        best = (key, base_name, object_name, float(weight))
        _, base_name, object_name, weight = best
        prediction = ((1 - weight) * base_readouts[base_name][held]
                      + weight * object_readouts[object_name][held])
        readout_scores.append(score(prediction, actual[held]))
        readout_weights.append((base_name, object_name, weight))
        print(f"readout fold {fold + 1}: base={base_name}, object={object_name}, "
              f"weight={weight:.3f}, held-out={readout_scores[-1]}")
    print("5-fold tuned point-readout blend (MAE, within3):", average(readout_scores))
    print("point-readout choices:", readout_weights)
    print(f"Candidate trained on the {payload['train_levels']}-level training pilot; "
          "no validation labels were used for training.")
    print("Held-out test partition was not evaluated.")


if __name__ == "__main__":
    main()
