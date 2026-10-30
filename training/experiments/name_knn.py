"""Hashed character n-gram KNN baseline for level names."""
import math
import random
import re
import sys
import zlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


DIMENSIONS = 4096
NEIGHBOR_COUNTS = (5, 10, 20, 40, 80)
TEMPERATURES = (0.01, 0.03, 0.06, 0.12, 0.24)


def vectorize(names):
    matrix = torch.zeros((len(names), DIMENSIONS), dtype=torch.float32)
    for row, name in enumerate(names):
        text = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        features = [f"w:{token}" for token in text.split()]
        padded = f"^{text}$"
        for size in range(2, 6):
            features.extend(f"c:{padded[i:i + size]}" for i in range(len(padded) - size + 1))
        for feature in features:
            index = zlib.crc32(feature.encode("utf-8")) % DIMENSIONS
            matrix[row, index] += 1
    return torch.nn.functional.normalize(matrix, dim=-1)


def predict(similarity, targets, k, temperature):
    values, indices = similarity.topk(k, dim=-1)
    weights = torch.softmax((values - values[:, :1]) / temperature, dim=-1)
    return (targets[indices] * weights.unsqueeze(-1)).sum(1)


def readouts(probabilities):
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    mean = probabilities @ tiers
    median = torch.searchsorted(
        probabilities.cumsum(-1), torch.full((len(probabilities), 1), 0.5)
    ).squeeze(-1).float() + 1
    return mean, median


def score(prediction, actual):
    error = (prediction - actual).abs()
    return error.mean().item(), (error <= 3).float().mean().item()


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    names = [row.get("name", "") for row in rows]
    features = vectorize(names)
    targets = torch.stack([row["target"] for row in rows])
    actual = targets @ torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    train_idx = torch.arange(2 * boundary, len(rows))
    validation_idx = torch.arange(boundary, 2 * boundary)
    order = torch.randperm(len(train_idx), generator=torch.Generator().manual_seed(31))
    holdout_count = math.ceil(len(train_idx) * 0.1)
    holdout_idx = train_idx[order[:holdout_count]]
    fitting_idx = train_idx[order[holdout_count:]]

    similarities = features[holdout_idx] @ features[fitting_idx].T
    best = None
    for k in NEIGHBOR_COUNTS:
        for temperature in TEMPERATURES:
            probabilities = predict(similarities, targets[fitting_idx], k, temperature)
            mean, median = readouts(probabilities)
            for label, prediction in (("mean", mean), ("median", median)):
                mae, within3 = score(prediction, actual[holdout_idx])
                candidate = (within3, -mae, k, temperature, label)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    _, _, k, temperature, label = best

    similarities = features[validation_idx] @ features[train_idx].T
    probabilities = predict(similarities, targets[train_idx], k, temperature)
    mean, median = readouts(probabilities)
    print(f"training-holdout selection: k={k}, temperature={temperature:g}, readout={label}")
    print("name KNN validation mean:", score(mean, actual[validation_idx]))
    print("name KNN validation median:", score(median, actual[validation_idx]))


if __name__ == "__main__":
    main()
