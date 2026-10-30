"""Check whether level ID provides a useful train-only difficulty prior."""
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


def readout(probabilities):
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    mean = probabilities @ tiers
    median = torch.searchsorted(
        probabilities.cumsum(-1), torch.full((len(probabilities), 1), 0.5)
    ).squeeze(-1).float() + 1
    return mean, median


def score(prediction, target):
    error = (prediction - target).abs()
    return error.mean().item(), (error <= 3).float().mean().item()


def predict(query_ids, train_ids, targets, bandwidth):
    distance = (query_ids[:, None] - train_ids[None, :]).abs()
    weights = torch.exp(-0.5 * (distance / bandwidth).square())
    weights /= weights.sum(-1, keepdim=True).clamp_min(1e-8)
    return weights @ targets


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    ids = torch.tensor([math.log10(float(row["level_id"])) for row in rows])
    targets = torch.stack([row["target"] for row in rows])
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    actual = targets @ tiers
    train_idx = torch.arange(2 * boundary, len(rows))
    validation_idx = torch.arange(boundary, 2 * boundary)
    random_order = torch.randperm(len(train_idx), generator=torch.Generator().manual_seed(31))
    holdout_count = math.ceil(len(train_idx) * 0.1)
    holdout_idx = train_idx[random_order[:holdout_count]]
    fitting_idx = train_idx[random_order[holdout_count:]]

    best = None
    for bandwidth in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6):
        probabilities = predict(ids[holdout_idx], ids[fitting_idx], targets[fitting_idx], bandwidth)
        mean, median = readout(probabilities)
        for label, prediction in (("mean", mean), ("median", median)):
            mae, within3 = score(prediction, actual[holdout_idx])
            candidate = (within3, -mae, bandwidth, label)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    _, _, bandwidth, label = best
    probabilities = predict(ids[validation_idx], ids[train_idx], targets[train_idx], bandwidth)
    mean, median = readout(probabilities)
    print(f"training-holdout selection: bandwidth={bandwidth:g}, readout={label}")
    print("ID prior validation mean:", score(mean, actual[validation_idx]))
    print("ID prior validation median:", score(median, actual[validation_idx]))
    print("Pearson ID/target correlation on train:",
          torch.corrcoef(torch.stack((ids[train_idx], actual[train_idx])))[0, 1].item())


if __name__ == "__main__":
    main()
