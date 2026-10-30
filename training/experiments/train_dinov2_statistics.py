"""Train a video-only statistics MLP on cached DINOv2 features.

This script uses the fixed 80/10 train/validation partitions and never reads or
scores the held-out test partition. It summarizes each video by temporal mean,
standard deviation, adjacent change, and channel-wise peak response.
"""
import math
import os
import random
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows
from training.experiments.cross_validate_ensemble import window_mass_center


FEATURES = ROOT / "training" / "features_dinov2_base_4fps"
OUTPUT = ROOT / "training" / "experiments" / "dinov2_statistics_ordinal.pt"
SEED = 42
HIDDEN = 128
DROPOUT = 0.4
BATCH_SIZE = 512
MAX_EPOCHS = 40
PATIENCE = 7
EXPECTED_LOSS_WEIGHT = 0.2


def pool_statistics(sequence):
    mean = sequence.mean(0)
    variability = sequence.std(0, unbiased=False)
    change = (
        (sequence[1:] - sequence[:-1]).abs().mean(0)
        if len(sequence) > 1 else torch.zeros_like(mean)
    )
    peak = sequence.amax(0)
    return torch.cat((mean, variability, change, peak))


class StatisticsHead(nn.Module):
    def __init__(self, input_size):
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

    def forward(self, inputs):
        return self.layers(inputs)


def ordinal_ce(logits, target):
    predicted_cdf = torch.softmax(logits, -1).cumsum(-1)[:, :-1].clamp(1e-7, 1 - 1e-7)
    target_cdf = target.cumsum(-1)[:, :-1].clamp(0, 1)
    return F.binary_cross_entropy(predicted_cdf, target_cdf)


def metrics(probabilities, actual):
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    mean = probabilities @ tiers
    median = torch.searchsorted(
        probabilities.cumsum(-1), torch.full((len(probabilities), 1), 0.5)
    ).squeeze(-1).float() + 1
    window = window_mass_center(probabilities, 3)
    return {
        "mean": mean,
        "median": median,
        "window+/-3": window,
    }, {
        name: ((prediction - actual).abs().mean().item(),
               ((prediction - actual).abs() <= 3).float().mean().item())
        for name, prediction in (("mean", mean), ("median", median), ("window+/-3", window))
    }


def save_checkpoint(payload):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=OUTPUT.parent, prefix=f".{OUTPUT.name}.",
                                     suffix=".partial", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, OUTPUT)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    rows = read_rows(DEFAULT_CSV)
    random.Random(SEED).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    train_rows = rows[2 * boundary:]
    valid_rows = rows[boundary:2 * boundary]
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)

    train_x, train_y = [], []
    for index, row in enumerate(train_rows, 1):
        sequence = torch.load(FEATURES / f"{row['level_id']}.pt", map_location="cpu",
                              weights_only=True)["features"].float()
        train_x.append(pool_statistics(sequence))
        train_y.append(row["target"])
        if index % 1000 == 0:
            print(f"pooled training features {index}/{len(train_rows)}", flush=True)
    valid_x, valid_y = [], []
    for index, row in enumerate(valid_rows, 1):
        sequence = torch.load(FEATURES / f"{row['level_id']}.pt", map_location="cpu",
                              weights_only=True)["features"].float()
        valid_x.append(pool_statistics(sequence))
        valid_y.append(row["target"])
    train_x, train_y = torch.stack(train_x), torch.stack(train_y)
    valid_x, valid_y = torch.stack(valid_x), torch.stack(valid_y)
    train_actual, valid_actual = train_y @ tiers, valid_y @ tiers

    mean = train_x.mean(0)
    scale = train_x.std(0, unbiased=False).clamp_min(1e-5)
    train_x = (train_x - mean) / scale
    valid_x = (valid_x - mean) / scale

    model = StatisticsHead(train_x.shape[1])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    generator = torch.Generator().manual_seed(SEED)
    best_within, best_mae, best_epoch, stale = -1.0, float("inf"), 0, 0
    best_payload = None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        for indices in permutation.split(BATCH_SIZE):
            logits = model(train_x[indices])
            probabilities = torch.softmax(logits, -1)
            predicted_tier = probabilities @ tiers
            loss = ordinal_ce(logits, train_y[indices])
            loss = loss + EXPECTED_LOSS_WEIGHT * F.smooth_l1_loss(
                predicted_tier, train_actual[indices]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_probabilities = torch.softmax(model(valid_x), -1)
        readouts, scores = metrics(valid_probabilities, valid_actual)
        val_mae, val_within = scores["window+/-3"]
        improved = val_within > best_within or (val_within == best_within and val_mae < best_mae)
        if improved:
            best_within, best_mae, best_epoch, stale = val_within, val_mae, epoch, 0
            best_payload = {
                "state_dict": model.state_dict(),
                "input_mean": mean,
                "input_scale": scale,
                "feature_summary": "mean,std,mean_abs_delta,max",
                "feature_model": "facebook/dinov2-base",
                "hidden_size": HIDDEN,
                "dropout": DROPOUT,
                "expected_loss_weight": EXPECTED_LOSS_WEIGHT,
                "epoch": epoch,
                "validation_scores": scores,
                "validation_probabilities": valid_probabilities,
            }
            save_checkpoint(best_payload)
        else:
            stale += 1
        print(
            f"epoch={epoch:02d} loss={loss.detach().item():.4f} "
            f"window3_MAE={scores['window+/-3'][0]:.3f} "
            f"window3_acc={scores['window+/-3'][1]:.3f} "
            f"median_MAE={scores['median'][0]:.3f} median_acc={scores['median'][1]:.3f} "
            f"best_epoch={best_epoch}", flush=True,
        )
        if stale >= PATIENCE:
            break

    print(f"parameters={parameter_count} best_epoch={best_epoch} "
          f"validation_window3={best_within:.4f} MAE={best_mae:.4f} "
          f"checkpoint={OUTPUT}", flush=True)
    print("Validation scores at best epoch:", best_payload["validation_scores"], flush=True)
    print("Held-out test partition was not loaded or evaluated.", flush=True)


if __name__ == "__main__":
    main()
