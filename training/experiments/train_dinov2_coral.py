"""Train a pooled DINO predictor with a CORAL cumulative ordinal link.

Uses four temporal feature bins and the fixed train/validation split. The held-
out test partition is not evaluated.
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
from training.experiments.cross_validate_ensemble import window_mass_center


FEATURES = ROOT / "training" / "features_dinov2_base_4fps"
OUTPUT = ROOT / "training" / "experiments" / "dinov2_coral.pt"
SEED = 42
HIDDEN = 256
DROPOUT = 0.4
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 8
AUXILIARY_MEAN_LOSS = 0.2


class PooledCoral(nn.Module):
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
            nn.Linear(HIDDEN, 1),
        )
        self.threshold_base = nn.Parameter(torch.zeros(()))
        self.threshold_steps = nn.Parameter(torch.zeros(NUM_CLASSES - 2))

    def thresholds(self):
        increments = F.softplus(self.threshold_steps)
        return torch.cat((
            self.threshold_base[None],
            self.threshold_base + increments.cumsum(0),
        ))

    def forward(self, features):
        return self.layers(features).squeeze(-1)

    def distribution(self, latent):
        survival = torch.sigmoid(latent[:, None] - self.thresholds()[None, :])
        probabilities = torch.cat((
            1 - survival[:, :1],
            survival[:, :-1] - survival[:, 1:],
            survival[:, -1:],
        ), dim=-1).clamp_min(1e-8)
        return probabilities / probabilities.sum(-1, keepdim=True)

    def initialize_thresholds(self, targets):
        survival = (1 - targets.cumsum(-1)[:, :-1].mean(0)).clamp(1e-4, 1 - 1e-4)
        thresholds = torch.logit(1 - survival)
        increments = (thresholds[1:] - thresholds[:-1]).clamp_min(1e-3)
        inverse_softplus = torch.log(torch.expm1(increments).clamp_min(1e-8))
        with torch.no_grad():
            self.threshold_base.copy_(thresholds[0])
            self.threshold_steps.copy_(inverse_softplus)


def readouts(probabilities):
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    mean = probabilities @ tiers
    median = torch.searchsorted(
        probabilities.cumsum(-1), torch.full((len(probabilities), 1), 0.5)
    ).squeeze(-1).float() + 1
    window = window_mass_center(probabilities, 3)
    return {"mean": mean, "median": median, "window+/-3": window}


def score(prediction, actual):
    error = (prediction - actual).abs()
    return error.mean().item(), (error <= 3).float().mean().item()


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

    def pool(row):
        sequence = torch.load(FEATURES / f"{row['level_id']}.pt", map_location="cpu",
                              weights_only=True)["features"].float()
        bins = F.adaptive_avg_pool1d(sequence.T.unsqueeze(0), output_size=4)
        return bins.squeeze(0).reshape(-1)

    train_x = torch.stack([pool(row) for row in train_rows])
    valid_x = torch.stack([pool(row) for row in valid_rows])
    train_target = torch.stack([row["target"] for row in train_rows])
    valid_target = torch.stack([row["target"] for row in valid_rows])
    train_actual, valid_actual = train_target @ tiers, valid_target @ tiers
    survival_target = (1 - train_target.cumsum(-1)[:, :-1]).clamp(0, 1)
    mean = train_x.mean(0)
    scale = train_x.std(0, unbiased=False).clamp_min(1e-5)
    train_x, valid_x = (train_x - mean) / scale, (valid_x - mean) / scale

    model = PooledCoral(train_x.shape[1])
    model.initialize_thresholds(train_target)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    generator = torch.Generator().manual_seed(SEED)
    best_within, best_mae, best_epoch, stale = -1.0, float("inf"), 0, 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        for indices in permutation.split(BATCH_SIZE):
            latent = model(train_x[indices])
            ordinal_logits = latent[:, None] - model.thresholds()[None, :]
            loss = F.binary_cross_entropy_with_logits(
                ordinal_logits, survival_target[indices]
            )
            probabilities = model.distribution(latent)
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
            valid_probabilities = model.distribution(model(valid_x))
        predictions = readouts(valid_probabilities)
        scores = {name: score(point, valid_actual) for name, point in predictions.items()}
        val_mae, val_within = scores["window+/-3"]
        improved = val_within > best_within or (val_within == best_within and val_mae < best_mae)
        if improved:
            best_within, best_mae, best_epoch, stale = val_within, val_mae, epoch, 0
            save({
                "state_dict": model.state_dict(),
                "input_mean": mean,
                "input_scale": scale,
                "feature_model": "facebook/dinov2-base",
                "feature_pooling": "four adaptive temporal means",
                "ordinal_link": "CORAL cumulative thresholds",
                "auxiliary_mean_loss_weight": AUXILIARY_MEAN_LOSS,
                "parameters": parameter_count,
                "epoch": epoch,
                "validation_probabilities": valid_probabilities,
                "validation_scores": scores,
            })
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
    print("Held-out test partition was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
