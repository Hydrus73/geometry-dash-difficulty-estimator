"""Retrieval baseline using cosine neighbors in pooled DINOv2-base features."""
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


FEATURE_DIR = ROOT / "training" / "features_dinov2_base_4fps"
NEIGHBOR_COUNTS = (5, 10, 20, 40, 80, 160)
TEMPERATURES = (0.01, 0.02, 0.04, 0.08, 0.16)


def tier_readouts(distribution):
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    mean = distribution @ tiers
    median = torch.searchsorted(
        distribution.cumsum(-1), torch.full((len(distribution), 1), 0.5)
    ).squeeze(-1).float() + 1
    return mean, median


def score(prediction, actual):
    error = (prediction - actual).abs()
    return error.mean().item(), (error <= 3).float().mean().item()


def make_representations(means, stds):
    mean = torch.nn.functional.normalize(means, dim=-1)
    mean_std = torch.cat((mean, torch.nn.functional.normalize(stds, dim=-1)), dim=-1)
    return {"mean": mean, "mean+std": torch.nn.functional.normalize(mean_std, dim=-1)}


def predict(similarities, labels, k, temperature):
    sims, indices = similarities.topk(k, dim=-1)
    weights = torch.softmax((sims - sims[:, :1]) / temperature, dim=-1)
    return (labels[indices] * weights.unsqueeze(-1)).sum(1)


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    means, stds = [], []
    for row in rows:
        cache = torch.load(FEATURE_DIR / f"{row['level_id']}.pt", map_location="cpu", weights_only=True)
        features = cache["features"].float()
        means.append(features.mean(0))
        stds.append(features.std(0, unbiased=False))
    means, stds = torch.stack(means), torch.stack(stds)
    targets = torch.stack([row["target"] for row in rows])
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    actual = targets @ tiers

    train_global = torch.arange(2 * boundary, len(rows))
    valid_global = torch.arange(boundary, 2 * boundary)
    local = torch.randperm(len(train_global), generator=torch.Generator().manual_seed(21))
    inner_count = math.ceil(len(train_global) * 0.1)
    inner_valid = train_global[local[:inner_count]]
    inner_train = train_global[local[inner_count:]]

    best = None
    for representation_name, representation in make_representations(means, stds).items():
        similarity = representation[inner_valid] @ representation[inner_train].T
        for k in NEIGHBOR_COUNTS:
            for temperature in TEMPERATURES:
                distribution = predict(similarity, targets[inner_train], k, temperature)
                mean, median = tier_readouts(distribution)
                for readout, prediction in (("mean", mean), ("median", median)):
                    mae, within3 = score(prediction, actual[inner_valid])
                    candidate = (within3, -mae, representation_name, k, temperature, readout)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
    _, _, representation_name, k, temperature, readout = best
    print(f"training-holdout selection: representation={representation_name}, k={k}, "
          f"temperature={temperature:g}, readout={readout}")

    representation = make_representations(means, stds)[representation_name]
    similarities = representation[valid_global] @ representation[train_global].T
    distribution = predict(similarities, targets[train_global], k, temperature)
    mean, median = tier_readouts(distribution)
    print(f"fixed validation mean: MAE={score(mean, actual[valid_global])[0]:.4f}, "
          f"within3={score(mean, actual[valid_global])[1]:.4f}")
    print(f"fixed validation median: MAE={score(median, actual[valid_global])[0]:.4f}, "
          f"within3={score(median, actual[valid_global])[1]:.4f}")


if __name__ == "__main__":
    main()
