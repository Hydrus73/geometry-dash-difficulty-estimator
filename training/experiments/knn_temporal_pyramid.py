"""Test video-only KNN with temporally ordered pools of cached DINO features."""
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


FEATURES = ROOT / "training" / "features_dinov2_base_4fps"
BIN_COUNTS = (1, 2, 4, 8)
NEIGHBORS = (10, 20, 40, 80)
TEMPERATURES = (0.02, 0.04, 0.08, 0.12)


def representation(features, bins):
    if bins == 1:
        return F.normalize(features.mean(0), dim=0)
    pooled = []
    for index in range(bins):
        start = index * len(features) // bins
        end = (index + 1) * len(features) // bins
        pooled.append(features[start:end].mean(0))
    normalized = F.normalize(torch.stack(pooled), dim=-1)
    return F.normalize(normalized.flatten(), dim=0)


def values(distribution):
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    mean = distribution @ tiers
    median = torch.searchsorted(
        distribution.cumsum(-1),
        torch.full((len(distribution), 1), 0.5),
    ).squeeze(-1).float() + 1
    return mean, median


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
    all_targets = torch.stack([row["target"] for row in rows])
    actual = all_targets @ torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)

    reps = {f"temporal-{bins}": [] for bins in BIN_COUNTS}
    global_stds = []
    for index, row in enumerate(rows, start=1):
        sequence = torch.load(
            FEATURES / f"{row['level_id']}.pt", map_location="cpu", weights_only=True
        )["features"].float()
        for bins in BIN_COUNTS:
            reps[f"temporal-{bins}"].append(representation(sequence, bins))
        global_stds.append(sequence.std(0, unbiased=False))
        if index % 1000 == 0:
            print(f"pooled video features for {index}/{len(rows)} levels", flush=True)
    for name in reps:
        reps[name] = torch.stack(reps[name])
    reps["global-mean+std"] = F.normalize(
        torch.cat((reps["temporal-1"], F.normalize(torch.stack(global_stds), dim=-1)), dim=-1),
        dim=-1,
    )

    train_ids = torch.arange(2 * boundary, len(rows))
    valid_ids = torch.arange(boundary, 2 * boundary)
    permutation = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(21))
    tune_count = math.ceil(len(train_ids) * 0.1)
    tune_ids = train_ids[permutation[:tune_count]]
    fit_ids = train_ids[permutation[tune_count:]]

    best = None
    for name, features in reps.items():
        similarities = features[tune_ids] @ features[fit_ids].T
        for k in NEIGHBORS:
            for temperature in TEMPERATURES:
                distribution = knn_predict(similarities, all_targets[fit_ids], k, temperature)
                mean, median = values(distribution)
                for readout, prediction in (("mean", mean), ("median", median)):
                    mae, within3 = score(prediction, actual[tune_ids])
                    candidate = (within3, -mae, name, k, temperature, readout)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
    _, _, name, k, temperature, readout = best
    print(f"training-only selection: representation={name}, k={k}, "
          f"temperature={temperature:.2f}, readout={readout}")

    similarities = reps[name][valid_ids] @ reps[name][train_ids].T
    retrieval_distribution = knn_predict(similarities, all_targets[train_ids], k, temperature)
    retrieval_mean, retrieval_median = values(retrieval_distribution)
    prediction = retrieval_mean if readout == "mean" else retrieval_median
    print("temporal KNN validation:", score(prediction, actual[valid_ids]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_distribution = video_ensemble(rows[boundary:2 * boundary], device)
    video_mean, video_median = values(video_distribution)
    baseline = score(video_median, actual[valid_ids])
    print("video ensemble median validation:", baseline)
    blended = 0.9 * video_distribution + 0.1 * retrieval_distribution
    _, blended_median = values(blended)
    print("fixed 90/10 video/retrieval median validation:", score(blended_median, actual[valid_ids]))

    folds = torch.arange(len(valid_ids)) % 5
    fold_scores = []
    for fold in range(5):
        tune, held = folds != fold, folds == fold
        best_blend = None
        for retrieval_weight in torch.linspace(0, 1, 21):
            mixed = (1 - retrieval_weight) * video_distribution + retrieval_weight * retrieval_distribution
            mean, median = values(mixed)
            for readout_name, point in (("mean", mean), ("median", median)):
                mae, within3 = score(point[tune], actual[valid_ids][tune])
                candidate = (within3, -mae, float(retrieval_weight), readout_name)
                if best_blend is None or candidate[:2] > best_blend[:2]:
                    best_blend = candidate
        weight, chosen_readout = best_blend[2], best_blend[3]
        mixed = (1 - weight) * video_distribution + weight * retrieval_distribution
        mean, median = values(mixed)
        point = mean if chosen_readout == "mean" else median
        result = score(point[held], actual[valid_ids][held])
        fold_scores.append(result)
        print(f"fold {fold + 1}: retrieval_weight={weight:.2f}, readout={chosen_readout}, "
              f"MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("five-fold video/retrieval fusion: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in fold_scores) / 5,
        sum(x[1] for x in fold_scores) / 5,
    ))


if __name__ == "__main__":
    main()
