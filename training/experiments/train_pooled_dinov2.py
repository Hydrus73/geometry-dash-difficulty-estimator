"""Train a compact pooled DINOv2 head with ordinal losses on cached features."""
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


FEATURE_DIR = ROOT / "training" / "features_dinov2_base_4fps"
OUTPUT_DIR = ROOT / "training" / "experiments"
SEED = 42
EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 256


class PooledHead(nn.Module):
    def __init__(self, input_size=4 * 768, hidden=256, dropout=0.4):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, x):
        return self.layers(self.norm(x))


def metrics(probabilities, targets, actual):
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    expected = probabilities @ tiers
    median = torch.searchsorted(
        probabilities.cumsum(-1), torch.full((len(probabilities), 1), 0.5)
    ).squeeze(-1).float() + 1
    exact = (probabilities.argmax(-1) == targets.argmax(-1)).float().mean().item()
    result = {}
    for name, prediction in (("mean", expected), ("median", median)):
        error = (prediction - actual).abs()
        result[name] = (error.mean().item(), (error <= 3).float().mean().item())
    return result, exact


def save_checkpoint(path, checkpoint):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".partial",
                                    delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    rows = read_rows(DEFAULT_CSV)
    random.Random(SEED).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    ids = [row["level_id"] for row in rows]
    targets = torch.stack([row["target"] for row in rows])

    # Four equal temporal bins retain coarse video order while enabling batched CPU training.
    inputs = []
    for level_id in ids:
        cache = torch.load(FEATURE_DIR / f"{level_id}.pt", map_location="cpu", weights_only=True)
        features = cache["features"].float()
        bins = F.adaptive_avg_pool1d(features.T.unsqueeze(0), output_size=4)
        inputs.append(bins.squeeze(0).reshape(-1))
    inputs = torch.stack(inputs)

    test_idx = torch.arange(boundary)
    valid_idx = torch.arange(boundary, 2 * boundary)
    train_idx = torch.arange(2 * boundary, len(rows))
    train_x = inputs[train_idx]
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(1e-5)
    inputs = (inputs - mean) / std
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    actual = targets @ tiers
    print(f"Loaded {len(rows)} features; train={len(train_idx)}, validation={len(valid_idx)}, "
          f"test={len(test_idx)} (test is not evaluated).", flush=True)

    for loss_name in ("ordinal_ce", "wasserstein"):
        torch.manual_seed(SEED)
        model = PooledHead()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
        best_within = -1.0
        best_mae = float("inf")
        best_epoch = 0
        stale = 0
        checkpoint_path = OUTPUT_DIR / f"dinov2_base_4fps_pooled_{loss_name}.pt"
        generator = torch.Generator().manual_seed(SEED)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            permutation = train_idx[torch.randperm(len(train_idx), generator=generator)]
            for batch_idx in permutation.split(BATCH_SIZE):
                logits = model(inputs[batch_idx])
                probabilities = logits.softmax(-1)
                pred_cdf = probabilities.cumsum(-1)[:, :-1]
                target_cdf = targets[batch_idx].cumsum(-1)[:, :-1].clamp(0, 1)
                if loss_name == "ordinal_ce":
                    loss = F.binary_cross_entropy(pred_cdf.clamp(1e-7, 1 - 1e-7), target_cdf)
                else:
                    loss = (pred_cdf - target_cdf).abs().sum(-1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                valid_probs = model(inputs[valid_idx]).softmax(-1)
            result, exact = metrics(valid_probs, targets[valid_idx], actual[valid_idx])
            # Median is the more robust tier readout in prior validation folds.
            val_mae, val_within = result["median"]
            improved = val_within > best_within or (val_within == best_within and val_mae < best_mae)
            if improved:
                best_within, best_mae, best_epoch, stale = val_within, val_mae, epoch, 0
                save_checkpoint(checkpoint_path, {
                    "state_dict": model.state_dict(),
                    "input_mean": mean,
                    "input_std": std,
                    "loss": loss_name,
                    "seed": SEED,
                    "best_epoch": epoch,
                    "validation_median_mae": val_mae,
                    "validation_median_within3": val_within,
                    "validation_exact": exact,
                }, checkpoint_path)
            else:
                stale += 1
            print(f"{loss_name} epoch {epoch:03d} loss={loss.detach().item():.4f} "
                  f"mean_mae={result['mean'][0]:.3f} mean_within3={result['mean'][1]:.3f} "
                  f"median_mae={val_mae:.3f} median_within3={val_within:.3f} "
                  f"exact={exact:.3f} best_epoch={best_epoch}", flush=True)
            if stale >= PATIENCE:
                break

        print(f"BEST {loss_name}: epoch={best_epoch}, median_MAE={best_mae:.4f}, "
              f"median_within3={best_within:.4f}, checkpoint={checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
