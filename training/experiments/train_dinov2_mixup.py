#!/usr/bin/env python3
"""Train a pooled ordinal head with feature/target MixUp on DINO features.

Only the seed-42 training and validation partitions are loaded. Validation
labels remain the original vote distributions; the fixed test partition is
never read or evaluated.
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


FEATURE_DIR = ROOT / "training" / "features_dinov2_base_4fps_1s"
OUTPUT = Path(__file__).resolve().parent / "dinov2_1s_pooled_mixup.pt"
SEED = 42
TIME_BINS = 4
BATCH_SIZE = 256
HIDDEN = 256
DROPOUT = 0.4
MIXUP_ALPHA = 0.2
MAX_EPOCHS = 100
PATIENCE = 12


class PooledHead(nn.Module):
    def __init__(self, input_size=TIME_BINS * 768):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.layers = nn.Sequential(
            nn.Linear(input_size, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, NUM_CLASSES),
        )

    def forward(self, features):
        return self.layers(self.norm(features))


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                    suffix=".partial", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def window_center(probabilities, radius=3):
    batch, classes = probabilities.shape
    best_mass = torch.full((batch,), -1.0, device=probabilities.device)
    best_tier = torch.ones(batch, device=probabilities.device)
    for start in range(classes):
        stop = min(classes, start + 2 * radius + 1)
        mass = probabilities[:, start:stop].sum(-1)
        center = torch.full_like(best_tier, (start + stop - 1) / 2 + 1)
        better = mass > best_mass
        best_mass = torch.where(better, mass, best_mass)
        best_tier = torch.where(better, center, best_tier)
    return best_tier


def metrics(probabilities, target, tiers):
    actual = target @ tiers
    mean = probabilities @ tiers
    median = torch.searchsorted(
        probabilities.cumsum(-1), torch.full((len(probabilities), 1), 0.5,
                                              device=probabilities.device)
    ).squeeze(-1).float() + 1
    window = window_center(probabilities)
    return {
        "mean": ((mean - actual).abs().mean().item(), ((mean - actual).abs() <= 3).float().mean().item()),
        "median": ((median - actual).abs().mean().item(), ((median - actual).abs() <= 3).float().mean().item()),
        "window+/-3": ((window - actual).abs().mean().item(), ((window - actual).abs() <= 3).float().mean().item()),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run outside the restricted sandbox")
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(SEED).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    valid_rows = rows[boundary:2 * boundary]
    train_rows = rows[2 * boundary:]
    print(f"Loading {len(train_rows)} train and {len(valid_rows)} validation sequences; "
          "test partition is not accessed.", flush=True)

    def load(rows):
        features, targets = [], []
        for index, row in enumerate(rows, 1):
            cache = torch.load(FEATURE_DIR / f"{row['level_id']}.pt",
                               map_location="cpu", weights_only=True)
            sequence = cache["features"].float()
            pooled = F.adaptive_avg_pool1d(sequence.T.unsqueeze(0), TIME_BINS).squeeze(0).T
            features.append(pooled.reshape(-1))
            targets.append(row["target"])
            if index % 1000 == 0:
                print(f"loaded {index}/{len(rows)}", flush=True)
        return torch.stack(features), torch.stack(targets)

    train_x, train_y = load(train_rows)
    valid_x, valid_y = load(valid_rows)
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(1e-5)
    train_x = ((train_x - mean) / std).to("cuda")
    valid_x = ((valid_x - mean) / std).to("cuda")
    train_y, valid_y = train_y.to("cuda"), valid_y.to("cuda")
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32, device="cuda")

    model = PooledHead().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    train_indices = torch.arange(len(train_x), device="cuda")
    best_acc, best_mae, best_epoch, stale = -1.0, float("inf"), 0, 0
    output = OUTPUT

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = train_indices[torch.randperm(len(train_indices), device="cuda")]
        for indices in permutation.split(BATCH_SIZE):
            x, y = train_x[indices], train_y[indices]
            pairing = torch.randperm(len(indices), device="cuda")
            lam = torch.distributions.Beta(MIXUP_ALPHA, MIXUP_ALPHA).sample().to("cuda")
            mixed_x = lam * x + (1 - lam) * x[pairing]
            mixed_y = lam * y + (1 - lam) * y[pairing]
            logits = model(mixed_x)
            probabilities = torch.softmax(logits, -1)
            predicted_cdf = probabilities.cumsum(-1)[:, :-1].clamp(1e-7, 1 - 1e-7)
            target_cdf = mixed_y.cumsum(-1)[:, :-1].clamp(0, 1)
            ordinal = F.binary_cross_entropy(predicted_cdf, target_cdf, reduction="none").mean(-1)
            predicted_tier = probabilities @ tiers
            target_tier = mixed_y @ tiers
            expected = F.smooth_l1_loss(predicted_tier, target_tier, reduction="none")
            in_window = (tiers[None, :] - target_tier[:, None]).abs() <= 3
            within = -(probabilities * in_window).sum(-1).clamp_min(1e-7).log()
            loss = (ordinal + 0.2 * expected + 0.5 * within).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            valid_probabilities = model(valid_x).softmax(-1)
        scores = metrics(valid_probabilities, valid_y, tiers)
        val_mae, val_acc = scores["window+/-3"]
        improved = val_acc > best_acc or (val_acc == best_acc and val_mae < best_mae)
        if improved:
            best_acc, best_mae, best_epoch, stale = val_acc, val_mae, epoch, 0
            save(output, {
                "validation_probabilities": valid_probabilities.detach().cpu(),
                "validation_scores": scores,
                "model": "four-bin DINO pooled MLP with MixUp",
                "feature_dir": str(FEATURE_DIR),
                "time_bins": TIME_BINS,
                "mixup_alpha": MIXUP_ALPHA,
                "parameters": sum(p.numel() for p in model.parameters()),
                "epoch": epoch,
                "feature_mean": mean,
                "feature_std": std,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            })
        else:
            stale += 1
        print(f"epoch={epoch:03d} loss={loss.item():.4f} window3_MAE={val_mae:.3f} "
              f"window3_acc={val_acc:.3f} best_epoch={best_epoch} "
              f"gpu_peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB", flush=True)
        if stale >= PATIENCE:
            break

    print(f"device={torch.cuda.get_device_name(0)} parameters={sum(p.numel() for p in model.parameters())} "
          f"best_epoch={best_epoch} validation_window3={best_acc:.4f} MAE={best_mae:.4f}")
    print("Validation readouts:", torch.load(output, map_location="cpu", weights_only=True)["validation_scores"])
    print("Held-out test partition was not evaluated.")


if __name__ == "__main__":
    main()
