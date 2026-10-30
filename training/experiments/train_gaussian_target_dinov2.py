"""Train a pooled DINO head with Gaussian, tier-distance-aware targets.

Uses the fixed train/validation partitions only. It does not load or evaluate
the held-out test partition.
"""
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


FEATURES = ROOT / "training" / "features_dinov2_base_4fps"
OUTPUT = ROOT / "training" / "experiments" / "dinov2_gaussian_target.pt"
SEED = 42
SIGMA = 3.0
HIDDEN = 256
DROPOUT = 0.4
AUXILIARY_MEAN_LOSS = 0.2
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 8


class PooledHead(nn.Module):
    def __init__(self, input_size=4 * 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, NUM_CLASSES),
        )

    def forward(self, features):
        return self.layers(features)


def gaussian_targets(actual, tiers):
    probabilities = torch.exp(-0.5 * ((tiers[None, :] - actual[:, None]) / SIGMA).square())
    return probabilities / probabilities.sum(-1, keepdim=True)


def window_center(probabilities, radius=3):
    cumulative = torch.cat((
        torch.zeros(len(probabilities), 1), probabilities.cumsum(-1)
    ), dim=-1)
    classes = probabilities.shape[-1]
    centers = torch.arange(classes)
    lower = (centers - radius).clamp_min(0)
    upper = (centers + radius + 1).clamp_max(classes)
    return (cumulative[:, upper] - cumulative[:, lower]).argmax(-1).float() + 1


def score(prediction, actual):
    error = (prediction - actual).abs()
    return error.mean().item(), (error <= 3).float().mean().item()


def save(payload):
    with tempfile.NamedTemporaryFile(dir=OUTPUT.parent, prefix=f".{OUTPUT.name}.",
                                     suffix=".partial", delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, OUTPUT)
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    rows = read_rows(DEFAULT_CSV)
    random.Random(SEED).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    train_rows = rows[2 * boundary:]
    valid_rows = rows[boundary:2 * boundary]
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)

    def pooled_input(row):
        features = torch.load(FEATURES / f"{row['level_id']}.pt", map_location="cpu",
                              weights_only=True)["features"].float()
        bins = F.adaptive_avg_pool1d(features.T.unsqueeze(0), output_size=4)
        return bins.squeeze(0).reshape(-1)

    train_x = torch.stack([pooled_input(row) for row in train_rows])
    valid_x = torch.stack([pooled_input(row) for row in valid_rows])
    train_dist = torch.stack([row["target"] for row in train_rows])
    valid_dist = torch.stack([row["target"] for row in valid_rows])
    train_actual, valid_actual = train_dist @ tiers, valid_dist @ tiers
    train_target = gaussian_targets(train_actual, tiers)

    mean = train_x.mean(0)
    scale = train_x.std(0, unbiased=False).clamp_min(1e-5)
    train_x = (train_x - mean) / scale
    valid_x = (valid_x - mean) / scale
    model = PooledHead(train_x.shape[1])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    generator = torch.Generator().manual_seed(SEED)
    best_within, best_mae, best_epoch, stale = -1.0, float("inf"), 0, 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        for indices in permutation.split(BATCH_SIZE):
            logits = model(train_x[indices])
            probabilities = torch.softmax(logits, -1)
            pred_cdf = probabilities.cumsum(-1)[:, :-1].clamp(1e-7, 1 - 1e-7)
            target_cdf = train_target[indices].cumsum(-1)[:, :-1].clamp(0, 1)
            loss = F.binary_cross_entropy(pred_cdf, target_cdf)
            predicted_mean = probabilities @ tiers
            loss = loss + AUXILIARY_MEAN_LOSS * F.smooth_l1_loss(
                predicted_mean, train_actual[indices]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(valid_x), -1)
            predicted_mean = probabilities @ tiers
            predicted_median = torch.searchsorted(
                probabilities.cumsum(-1), torch.full((len(valid_x), 1), 0.5)
            ).squeeze(-1).float() + 1
            predicted_window = window_center(probabilities)
        val_mae, val_within = score(predicted_window, valid_actual)
        improved = val_within > best_within or (val_within == best_within and val_mae < best_mae)
        if improved:
            best_within, best_mae, best_epoch, stale = val_within, val_mae, epoch, 0
            payload = {
                "state_dict": model.state_dict(),
                "input_mean": mean,
                "input_scale": scale,
                "target_representation": f"Gaussian around vote-weighted mean, sigma={SIGMA}",
                "loss": "ordinal CDF cross-entropy + expected-tier smooth L1",
                "auxiliary_mean_loss_weight": AUXILIARY_MEAN_LOSS,
                "feature_model": "facebook/dinov2-base",
                "feature_pooling": "four adaptive temporal means",
                "parameters": parameter_count,
                "epoch": epoch,
                "validation_probabilities": probabilities,
                "validation_scores": {
                    "mean": score(predicted_mean, valid_actual),
                    "median": score(predicted_median, valid_actual),
                    "window+/-3": score(predicted_window, valid_actual),
                },
            }
            save(payload)
        else:
            stale += 1
        print(f"epoch={epoch:02d} loss={loss.detach().item():.4f} "
              f"window3_MAE={val_mae:.3f} window3_acc={val_within:.3f} "
              f"best_epoch={best_epoch}", flush=True)
        if stale >= PATIENCE:
            break

    checkpoint = torch.load(OUTPUT, map_location="cpu", weights_only=True)
    print(f"parameters={parameter_count} best_epoch={best_epoch} "
          f"validation_window3={best_within:.4f} MAE={best_mae:.4f}", flush=True)
    print("Validation readouts:", checkpoint["validation_scores"], flush=True)
    print("Held-out test partition was not loaded or evaluated.", flush=True)


if __name__ == "__main__":
    main()
