"""CPU ridge ordinal baseline on cached DINOv2-base features."""
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


FEATURE_DIR = ROOT / "training" / "features_dinov2_base_4fps"
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)


def tiers_from_cdf(cdf):
    cdf = torch.cummax(cdf.clamp(0, 1), dim=-1).values
    probs = torch.cat((cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]), dim=-1)
    probs = probs.clamp_min(0)
    probs /= probs.sum(-1, keepdim=True).clamp_min(1e-8)
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    mean = probs @ tiers
    median = torch.searchsorted(probs.cumsum(-1), torch.full((len(probs), 1), 0.5)).squeeze(-1) + 1
    return probs, mean, median.float()


def metrics(pred, target):
    error = (pred - target).abs()
    return error.mean().item(), (error <= 3).float().mean().item()


def fit_ridge(train_x, train_y, eval_x, alpha):
    x_mean = train_x.mean(0)
    x_scale = train_x.std(0).clamp_min(1e-5)
    y_mean = train_y.mean(0)
    x = (train_x - x_mean) / x_scale
    y = train_y - y_mean
    gram = x.T @ x / len(x)
    rhs = x.T @ y / len(x)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    projected = eigenvectors.T @ rhs
    beta = eigenvectors @ (projected / (eigenvalues[:, None] + alpha))
    normalized_eval = (eval_x - x_mean) / x_scale
    return normalized_eval @ beta + y_mean


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    train_rows = rows[2 * boundary:]
    valid_rows = rows[boundary:2 * boundary]
    ids = [row["level_id"] for row in rows]

    pooled_mean = []
    pooled_std = []
    for level_id in ids:
        cache = torch.load(FEATURE_DIR / f"{level_id}.pt", map_location="cpu", weights_only=True)
        features = cache["features"].float()
        pooled_mean.append(features.mean(0))
        pooled_std.append(features.std(0, unbiased=False))
    pooled_mean = torch.stack(pooled_mean)
    pooled_std = torch.stack(pooled_std)
    target_dist = torch.stack([row["target"] for row in rows])
    target_cdf = target_dist.cumsum(-1)[:, :-1]
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    actual_tier = target_dist @ tiers

    # Tune pooling and ridge strength using a fixed holdout from the training split.
    shuffled = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(21))
    inner_size = max(1, math.ceil(len(train_rows) * 0.1))
    inner_valid_idx = shuffled[:inner_size]
    inner_train_idx = shuffled[inner_size:]
    train_global = torch.arange(2 * boundary, len(rows))
    inner_valid_global = train_global[inner_valid_idx]
    inner_train_global = train_global[inner_train_idx]
    valid_global = torch.arange(boundary, 2 * boundary)

    representations = {
        "mean": pooled_mean,
        "mean+std": torch.cat((pooled_mean, pooled_std), dim=-1),
    }
    best = None
    for name, features in representations.items():
        for alpha in ALPHAS:
            cdf = fit_ridge(features[inner_train_global], target_cdf[inner_train_global],
                            features[inner_valid_global], alpha)
            _, mean, median = tiers_from_cdf(cdf)
            for readout_name, prediction in (("mean", mean), ("median", median)):
                mae, within3 = metrics(prediction, actual_tier[inner_valid_global])
                print(f"training holdout {name} alpha={alpha:g} {readout_name}: "
                      f"MAE={mae:.4f} within3={within3:.4f}")
                candidate = (within3, -mae, name, alpha, readout_name)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate

    _, _, representation, alpha, readout = best
    features = representations[representation]
    cdf = fit_ridge(features[train_global], target_cdf[train_global], features[valid_global], alpha)
    probs, mean, median = tiers_from_cdf(cdf)
    prediction = mean if readout == "mean" else median
    mae, within3 = metrics(prediction, actual_tier[valid_global])
    print(f"selected on training holdout: pooling={representation}, alpha={alpha:g}, "
          f"readout={readout}")
    print(f"fixed validation: MAE={mae:.4f}, within3={within3:.4f}")
    print(f"validation mean readout: {metrics(mean, actual_tier[valid_global])}")
    print(f"validation median readout: {metrics(median, actual_tier[valid_global])}")


if __name__ == "__main__":
    main()
