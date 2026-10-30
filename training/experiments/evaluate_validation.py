"""Evaluate saved model variants on the fixed validation split only."""
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.train import DEFAULT_CSV, DEFAULT_FEATURE_DIR, NUM_CLASSES, VideoMAERegressor, read_rows

BASE = ROOT / "training" / "experiments"
ARCHIVE = BASE / "archive" / "checkpoints"
SMALL_DIR = ROOT / "training" / "features_dinov2_small_4fps"
BASE_DIR = ROOT / "training" / "features_dinov2_base_4fps"
BASE_1S_DIR = ROOT / "training" / "features_dinov2_base_4fps_1s"
MODELS = {
    "videomae_pooled": (ARCHIVE / "ordinal_pooled_mlp.pt", DEFAULT_FEATURE_DIR, 768),
    "videomae_mae": (ARCHIVE / "ordinal_transformer_mae.pt", DEFAULT_FEATURE_DIR, 768),
    "videomae_categorical": (ARCHIVE / "ordinal_transformer_categorical.pt", DEFAULT_FEATURE_DIR, 768),
    "dino_small_ordinal": (BASE / "dinov2_small_4fps_ordinal_ce.pt", SMALL_DIR, 384),
    "dino_small_ordinal_seed7": (BASE / "dinov2_small_4fps_ordinal_ce_seed7.pt", SMALL_DIR, 384),
    "dino_small_categorical": (BASE / "dinov2_small_4fps_categorical_ce.pt", SMALL_DIR, 384),
    "dino_base_4s_ordinal": (BASE / "dinov2_base_4fps_ordinal_ce.pt", BASE_DIR, 768),
    "dino_base_4s_categorical": (BASE / "dinov2_base_4fps_categorical_ce.pt", BASE_DIR, 768),
    "dino_base_4s_categorical_seed7": (BASE / "dinov2_base_4fps_categorical_ce_seed7.pt", BASE_DIR, 768),
    "dino_base_4s_dropout06": (BASE / "dinov2_base_4fps_categorical_ce_dropout06.pt", BASE_DIR, 768),
    "dino_base_1s_ordinal": (BASE / "dinov2_base_4fps_1s_ordinal_ce.pt", BASE_1S_DIR, 768),
    "dino_base_4s_regression": (BASE / "dinov2_base_4fps_expected_tier.pt", BASE_DIR, 768),
}


def metrics(prediction: torch.Tensor, target: torch.Tensor):
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    actual_tier = (target * tiers).sum(-1)
    if prediction.ndim == 2 and prediction.shape[-1] == NUM_CLASSES:
        predicted_tier = prediction @ tiers
        exact = prediction.argmax(-1) == target.argmax(-1)
    else:
        predicted_tier = prediction.reshape(-1)
        exact = predicted_tier.clamp(1, NUM_CLASSES).round().long() - 1 == target.argmax(-1)
    error = (predicted_tier - actual_tier).abs()
    return (error.mean().item(), exact.float().mean().item(), (error <= 3).float().mean().item())


def predict(name, rows, device):
    checkpoint_path, feature_dir, input_size = MODELS[name]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = VideoMAERegressor(
        checkpoint.get("feature_size", input_size),
        checkpoint.get("temporal_hidden_size", 224),
        checkpoint.get("head_type", "temporal_transformer"),
        checkpoint.get("dropout", 0.4),
        checkpoint.get("transformer_layers", 2),
        checkpoint.get("target_mode", "distribution"),
        checkpoint.get("ordinal_link", "softmax"),
        input_size,
    )
    model.load_state_dict(checkpoint["temporal_head"], strict=False)
    model.to(device).eval()
    predictions = []
    with torch.no_grad():
        for row in rows:
            cache = torch.load(
                feature_dir / f"{row['level_id']}.pt", map_location="cpu", weights_only=True
            )
            features = cache["features"].float().to(device)
            predictions.append(model(features).cpu())
    values = torch.cat(predictions)
    return values.squeeze(-1) if checkpoint.get("target_mode") == "expected_tier" else values


def score_mixtures(groups, target, increment=0.1):
    names = list(groups)
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    best_mae = (float("inf"), None, None)
    best_within3 = (-1.0, None, None)
    denominator = round(1 / increment)
    if len(names) == 3:
        for a in range(denominator + 1):
            for b in range(denominator + 1 - a):
                weights = (a / denominator, b / denominator, 1 - (a + b) / denominator)
                p = sum(w * groups[n] for w, n in zip(weights, names))
                result = metrics(p, target)
                if result[0] < best_mae[0]:
                    best_mae = (result[0], weights, result)
                if result[2] > best_within3[0]:
                    best_within3 = (result[2], weights, result)
    elif len(names) == 4:
        for a in range(denominator + 1):
            for b in range(denominator + 1 - a):
                for c in range(denominator + 1 - a - b):
                    weights = (a / denominator, b / denominator, c / denominator,
                               1 - (a + b + c) / denominator)
                    p = sum(w * groups[n] for w, n in zip(weights, names))
                    result = metrics(p, target)
                    if result[0] < best_mae[0]:
                        best_mae = (result[0], weights, result)
                    if result[2] > best_within3[0]:
                        best_within3 = (result[2], weights, result)
    print("best mixture by MAE:", best_mae)
    print("best mixture by within3:", best_within3)


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    split_size = max(1, round(len(rows) * 0.1 + 0.499999))
    validation_rows = rows[split_size:2 * split_size]
    targets = torch.stack([row["target"] for row in validation_rows])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")
    predictions = {name: predict(name, validation_rows, device) for name in MODELS}
    for name, value in predictions.items():
        print(name, tuple(round(x, 4) for x in metrics(value, targets)))

    core = torch.stack([predictions[n] for n in (
        "videomae_pooled", "videomae_mae", "videomae_categorical"
    )]).mean(0)
    small = torch.stack([predictions[n] for n in (
        "dino_small_ordinal", "dino_small_ordinal_seed7", "dino_small_categorical"
    )]).mean(0)
    base = torch.stack([predictions[n] for n in (
        "dino_base_4s_ordinal", "dino_base_4s_categorical",
        "dino_base_4s_categorical_seed7", "dino_base_4s_dropout06"
    )]).mean(0)
    detail = predictions["dino_base_1s_ordinal"]
    print("VideoMAE group:", metrics(core, targets))
    print("DINO-small group:", metrics(small, targets))
    print("DINO-base group:", metrics(base, targets))
    print("DINO-base one-second:", metrics(detail, targets))
    score_mixtures({"VideoMAE": core, "DINO-small": small, "DINO-base": base}, targets)
    score_mixtures({"VideoMAE": core, "DINO-small": small, "DINO-base": base,
                    "one-second": detail}, targets)

    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    for label, p in (("best_mae_mix", .1 * core + .2 * small + .7 * base),
                     ("best_within3_mix", .2 * core + .1 * small + .7 * base)):
        mean = p @ tiers
        median = torch.searchsorted(p.cumsum(-1), torch.full((len(p), 1), .5)).squeeze(-1).float() + 1
        hybrid = .25 * mean + .75 * median
        print(label, "mean:", metrics(mean, targets), "median:", metrics(median, targets),
              "hybrid_25mean_75median:", metrics(hybrid, targets))


if __name__ == "__main__":
    main()
