"""Train a DINO pooled regressor with a direct within-three tolerance loss.

The primary term is zero inside a +/-3-tier band and grows with the excess
error. A small smooth-L1 anchor selects a point inside that band. The fixed
held-out test partition is not evaluated.
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
OUTPUT = ROOT / "training" / "experiments" / "dinov2_tolerance_loss.pt"
SEED = 42
HIDDEN = 256
DROPOUT = 0.4
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 8
ANCHOR_WEIGHT = 0.1


class PooledRegressor(nn.Module):
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
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def score(prediction, actual):
    error = (prediction - actual).abs()
    return error.mean().item(), (error <= 3).float().mean().item()


def pool(row):
    sequence = torch.load(FEATURES / f"{row['level_id']}.pt", map_location="cpu",
                          weights_only=True)["features"].float()
    bins = F.adaptive_avg_pool1d(sequence.T.unsqueeze(0), output_size=4)
    return bins.squeeze(0).reshape(-1)


def save(payload):
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

    train_x = torch.stack([pool(row) for row in train_rows])
    valid_x = torch.stack([pool(row) for row in valid_rows])
    train_dist = torch.stack([row["target"] for row in train_rows])
    valid_dist = torch.stack([row["target"] for row in valid_rows])
    train_y, valid_y = train_dist @ tiers, valid_dist @ tiers
    mean = train_x.mean(0)
    scale = train_x.std(0, unbiased=False).clamp_min(1e-5)
    train_x, valid_x = (train_x - mean) / scale, (valid_x - mean) / scale

    model = PooledRegressor(train_x.shape[1])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    generator = torch.Generator().manual_seed(SEED)
    best_within, best_mae, best_epoch, stale = -1.0, float("inf"), 0, 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        for indices in permutation.split(BATCH_SIZE):
            prediction = model(train_x[indices])
            error = (prediction - train_y[indices]).abs()
            tolerance_penalty = F.relu(error - 3).mean()
            anchor = F.smooth_l1_loss(prediction, train_y[indices])
            loss = tolerance_penalty + ANCHOR_WEIGHT * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_prediction = model(valid_x).clamp(1, NUM_CLASSES)
        val_mae, val_within = score(valid_prediction, valid_y)
        improved = val_within > best_within or (val_within == best_within and val_mae < best_mae)
        if improved:
            best_within, best_mae, best_epoch, stale = val_within, val_mae, epoch, 0
            save({
                "state_dict": model.state_dict(),
                "input_mean": mean,
                "input_scale": scale,
                "feature_model": "facebook/dinov2-base",
                "feature_pooling": "four adaptive temporal means",
                "loss": "relu(abs(error)-3) + 0.1 * smooth_l1(error)",
                "parameters": parameter_count,
                "epoch": epoch,
                "validation_predictions": valid_prediction,
                "validation_scores": score(valid_prediction, valid_y),
            })
        else:
            stale += 1
        print(f"epoch={epoch:02d} loss={loss.detach().item():.4f} "
              f"MAE={val_mae:.3f} within3={val_within:.3f} best_epoch={best_epoch}", flush=True)
        if stale >= PATIENCE:
            break

    checkpoint = torch.load(OUTPUT, map_location="cpu", weights_only=True)
    print(f"parameters={parameter_count} best_epoch={best_epoch} "
          f"validation_MAE={best_mae:.4f} within3={best_within:.4f}", flush=True)
    print("Held-out test partition was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
