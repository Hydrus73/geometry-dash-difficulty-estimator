"""Compare global KNN retrieval from one- and four-second DINO features."""
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows
from training.experiments.knn_dinov2 import predict as knn_predict
from training.experiments.evaluate_validation import MODELS, predict as predict_model
from training.experiments.cross_validate_ensemble import window_mass_center


FEATURE_DIRS = {
    "4s": ROOT / "training" / "features_dinov2_base_4fps",
    "1s": ROOT / "training" / "features_dinov2_base_4fps_1s",
}
NEIGHBORS = (5, 10, 20, 40, 80)
TEMPERATURES = (0.01, 0.02, 0.04, 0.08, 0.16)


def prediction_readouts(distribution):
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    mean = distribution @ tiers
    median = torch.searchsorted(
        distribution.cumsum(-1), torch.full((len(distribution), 1), 0.5)
    ).squeeze(-1).float() + 1
    return {"mean": mean, "median": median,
            "window+/-2": window_mass_center(distribution, 2),
            "window+/-3": window_mass_center(distribution, 3)}


def score(prediction, actual):
    error = (prediction - actual).abs()
    return error.mean().item(), (error <= 3).float().mean().item()


def video_ensemble(rows, device):
    raw = {name: predict_model(name, rows, device) for name in MODELS}
    groups = [
        torch.stack([raw[n] for n in (
            "videomae_pooled", "videomae_mae", "videomae_categorical"
        )]).mean(0),
        torch.stack([raw[n] for n in (
            "dino_small_ordinal", "dino_small_ordinal_seed7", "dino_small_categorical"
        )]).mean(0),
        torch.stack([raw[n] for n in (
            "dino_base_4s_ordinal", "dino_base_4s_categorical",
            "dino_base_4s_categorical_seed7", "dino_base_4s_dropout06"
        )]).mean(0),
    ]
    return torch.stack(groups).mean(0)


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    validation_rows = rows[boundary:2 * boundary]
    train_rows = rows[2 * boundary:]
    # Do not read feature files for the held-out test partition.
    usable_rows = rows[boundary:]
    targets = torch.stack([row["target"] for row in usable_rows])
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    actual = targets @ tiers
    validation_ids = torch.arange(boundary)
    train_ids = torch.arange(boundary, len(usable_rows))

    representations = {}
    for rate, feature_dir in FEATURE_DIRS.items():
        means, stds = [], []
        for index, row in enumerate(usable_rows, 1):
            sequence = torch.load(
                feature_dir / f"{row['level_id']}.pt", map_location="cpu", weights_only=True
            )["features"].float()
            means.append(sequence.mean(0))
            stds.append(sequence.std(0, unbiased=False))
            if rate == "1s" and index % 1500 == 0:
                print(f"pooled {rate} features for {index}/{len(usable_rows)} train+validation levels",
                      flush=True)
        mean_features = torch.stack(means)
        std_features = torch.stack(stds)
        representations[f"{rate}-mean"] = F.normalize(mean_features, dim=-1)
        representations[f"{rate}-mean+std"] = F.normalize(torch.cat((
            representations[f"{rate}-mean"], F.normalize(std_features, dim=-1)
        ), dim=-1), dim=-1)

    permutation = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(21))
    tune_count = math.ceil(len(train_ids) * 0.1)
    tune_ids, fit_ids = train_ids[permutation[:tune_count]], train_ids[permutation[tune_count:]]
    best = None
    for name, features in representations.items():
        similarities = features[tune_ids] @ features[fit_ids].T
        for k in NEIGHBORS:
            for temperature in TEMPERATURES:
                distribution = knn_predict(similarities, targets[fit_ids], k, temperature)
                for readout, point in prediction_readouts(distribution).items():
                    mae, within3 = score(point, actual[tune_ids])
                    candidate = (within3, -mae, name, k, temperature, readout)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
    _, _, name, k, temperature, readout = best
    print(f"training-only selection: representation={name}, k={k}, "
          f"temperature={temperature:g}, readout={readout}")

    similarities = representations[name][validation_ids] @ representations[name][train_ids].T
    retrieval_distribution = knn_predict(similarities, targets[train_ids], k, temperature)
    actual_valid = actual[validation_ids]
    points = prediction_readouts(retrieval_distribution)
    print("selected multirate KNN validation:", score(points[readout], actual_valid))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_distribution = video_ensemble(validation_rows, device)
    video_median = prediction_readouts(video_distribution)["median"]
    print("video ensemble median validation:", score(video_median, actual_valid))
    folds = torch.arange(len(validation_rows)) % 5
    fold_scores = []
    for fold in range(5):
        tune, held = folds != fold, folds == fold
        best_blend = None
        for retrieval_weight in torch.linspace(0, 0.5, 11):
            mixed = (1 - retrieval_weight) * video_distribution + retrieval_weight * retrieval_distribution
            for readout_name, point in prediction_readouts(mixed).items():
                mae, within3 = score(point[tune], actual_valid[tune])
                candidate = (within3, -mae, float(retrieval_weight), readout_name)
                if best_blend is None or candidate[:2] > best_blend[:2]:
                    best_blend = candidate
        weight, readout_name = best_blend[2], best_blend[3]
        mixed = (1 - weight) * video_distribution + weight * retrieval_distribution
        result = score(prediction_readouts(mixed)[readout_name][held], actual_valid[held])
        fold_scores.append(result)
        print(f"fold {fold + 1}: retrieval_weight={weight:.2f}, readout={readout_name}, "
              f"MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("five-fold multirate KNN fusion: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in fold_scores) / 5,
        sum(x[1] for x in fold_scores) / 5,
    ))


if __name__ == "__main__":
    main()
