"""Fold-held-out confidence gate for the fixed video/KNN + temporal blend."""
import random
from pathlib import Path

import torch

from cross_validate_ensemble import window_mass_center, score
from evaluate_validation import MODELS, predict
from knn_dinov2 import predict as knn_predict
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "training" / "features_dinov2_base_4fps"
TEMPORAL_CHECKPOINT = Path(__file__).resolve().parent / "dinov2_1s_batched_transformer_within3_w05.pt"


def make_components(rows):
    """Build the exact distributions used by the fixed best ensemble."""
    group_names = {
        "VideoMAE": ("videomae_pooled", "videomae_mae", "videomae_categorical"),
        "DINO-small": ("dino_small_ordinal", "dino_small_ordinal_seed7", "dino_small_categorical"),
        "DINO-base": ("dino_base_4s_ordinal", "dino_base_4s_categorical",
                      "dino_base_4s_categorical_seed7", "dino_base_4s_dropout06"),
    }
    raw = {name: predict(name, rows, torch.device("cpu"))
           for names in group_names.values() for name in names}
    groups = torch.stack([
        torch.stack([raw[name] for name in names]).mean(0)
        for names in group_names.values()
    ])

    all_features = []
    for row in rows:
        item = torch.load(FEATURE_DIR / f"{row['level_id']}.pt",
                          map_location="cpu", weights_only=True)
        all_features.append(item["features"].float().mean(0))
    all_features = torch.nn.functional.normalize(torch.stack(all_features), dim=-1)

    temporal = torch.load(TEMPORAL_CHECKPOINT, map_location="cpu", weights_only=True)
    temporal = temporal["validation_probabilities"].float()

    return groups, all_features, temporal


def features_for_gate(base, temporal):
    """Only prediction-derived uncertainty and disagreement features."""
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    eps = 1e-8

    def entropy(prob):
        return -(prob.clamp_min(eps) * prob.clamp_min(eps).log()).sum(-1)

    base_mean, temporal_mean = base @ tiers, temporal @ tiers
    return torch.stack((
        entropy(base), entropy(temporal),
        (temporal_mean - base_mean).abs(),
        base.max(-1).values, temporal.max(-1).values,
    ), -1)


def soft_within_three_mass(distribution, actual):
    """Smooth proxy for probability mass within three tiers of the vote mean."""
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=distribution.dtype)
    closeness = torch.sigmoid((3.25 - (tiers[None] - actual[:, None]).abs()) / 0.5)
    return (distribution * closeness).sum(-1)


def fit_gate(base, temporal, actual, features):
    """Fit a small regularized sigmoid gate; return weights in [0, .5]."""
    center = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(1e-4)
    x = (features - center) / scale
    # An all-zero parameter vector gives the global baseline temporal weight .25.
    coefficients = torch.nn.Parameter(torch.zeros(x.shape[1]))
    bias = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam((coefficients, bias), lr=0.04)
    for _ in range(350):
        optimizer.zero_grad()
        weight = 0.5 * torch.sigmoid(x @ coefficients + bias)
        distribution = base + weight[:, None] * (temporal - base)
        loss = -soft_within_three_mass(distribution, actual).mean()
        # Keep both the average blend weight and per-example deviations close
        # to the already validated 0.25 temporal weight.
        loss = loss + 0.02 * coefficients.square().sum()
        loss = loss + 0.20 * (weight - 0.25).square().mean()
        loss.backward()
        optimizer.step()
    return center, scale, coefficients.detach(), bias.detach()


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    split_size = max(1, round(len(rows) * 0.1 + 0.499999))
    train_idx = torch.arange(2 * split_size, len(rows))
    valid_rows = rows[split_size:2 * split_size]
    target = torch.stack([row["target"] for row in valid_rows])
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    actual = target @ tiers

    print("Building model and retrieval probabilities on CPU...", flush=True)
    groups, features, temporal = make_components(valid_rows)
    # Retrieval neighbors are restricted to the fixed training partition.
    train_features = []
    for row in rows[2 * split_size:]:
        item = torch.load(FEATURE_DIR / f"{row['level_id']}.pt",
                          map_location="cpu", weights_only=True)
        train_features.append(item["features"].float().mean(0))
    train_features = torch.nn.functional.normalize(torch.stack(train_features), dim=-1)
    training_targets = torch.stack([row["target"] for row in rows[2 * split_size:]])
    similarity = features @ train_features.T
    knn = knn_predict(similarity, training_targets, 20, 0.02)

    video_groups = groups.mean(0)
    fixed = 0.9 * video_groups + 0.1 * knn
    fixed_best = 0.75 * fixed + 0.25 * temporal
    gate_features = features_for_gate(fixed, temporal)
    fold_ids = torch.arange(len(valid_rows)) % 5
    fixed_scores, gate_scores, fold_weights = [], [], []

    # Compare the arithmetic pool to a weighted logarithmic opinion pool.
    # Source weights reproduce the arithmetic blend's effective weights.
    sources = torch.cat((groups, knn[None], temporal[None]), dim=0)
    source_weights = torch.tensor([0.225, 0.225, 0.225, 0.075, 0.25])
    log_pool_candidates = {}
    for temperature in (0.75, 1.0, 1.25):
        log_pool = torch.softmax(
            (source_weights[:, None, None] * sources.clamp_min(1e-7).log()).sum(0)
            / temperature,
            dim=-1,
        )
        log_pool_candidates[temperature] = log_pool
    pooling_scores = []

    for fold in range(5):
        train, valid = fold_ids != fold, fold_ids == fold
        fixed_prediction = window_mass_center(fixed_best[valid], 3)
        fixed_scores.append(score(fixed_prediction, actual[valid]))

        best_pool = None
        for temperature, log_pool in log_pool_candidates.items():
            for log_weight in torch.arange(0, 1.001, 0.1):
                mixture = (1 - log_weight) * fixed_best[train] + log_weight * log_pool[train]
                result = score(window_mass_center(mixture, 3), actual[train])
                candidate = (result[1], -result[0], -float(log_weight), -temperature)
                if best_pool is None or candidate > best_pool[0]:
                    best_pool = (candidate, float(log_weight), temperature)
        _, log_weight, temperature = best_pool
        pool_distribution = ((1 - log_weight) * fixed_best[valid]
                             + log_weight * log_pool_candidates[temperature][valid])
        pooling_scores.append(score(window_mass_center(pool_distribution, 3), actual[valid]))
        print(f"fold {fold + 1}: log-pool weight={log_weight:.1f}, temperature={temperature:.2f}, "
              f"pooled={pooling_scores[-1]}", flush=True)

        center, scale, coefficients, bias = fit_gate(
            fixed[train], temporal[train], actual[train], gate_features[train]
        )
        x = (gate_features[valid] - center) / scale
        temporal_weight = 0.5 * torch.sigmoid(x @ coefficients + bias)
        gated_distribution = fixed[valid] + temporal_weight[:, None] * (
            temporal[valid] - fixed[valid]
        )
        prediction = window_mass_center(gated_distribution, 3)
        gate_scores.append(score(prediction, actual[valid]))
        fold_weights.extend(temporal_weight.tolist())
        print(f"fold {fold + 1}: fixed={fixed_scores[-1]}, gated={gate_scores[-1]}, "
              f"mean temporal weight={temporal_weight.mean().item():.3f}", flush=True)

    fixed_avg = tuple(sum(result[i] for result in fixed_scores) / 5 for i in range(2))
    gate_avg = tuple(sum(result[i] for result in gate_scores) / 5 for i in range(2))
    print(f"5-fold fixed baseline (MAE, within3): {fixed_avg}")
    pooling_avg = tuple(sum(result[i] for result in pooling_scores) / 5 for i in range(2))
    print(f"5-fold arithmetic/logarithmic pool (MAE, within3): {pooling_avg}")
    print(f"5-fold confidence gate (MAE, within3): {gate_avg}")
    print(f"held-out temporal weight range: {min(fold_weights):.3f}..{max(fold_weights):.3f}")


if __name__ == "__main__":
    main()
