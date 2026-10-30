"""Plot actual vs predicted tiers for one repeated split's fixed ensemble."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from training.experiments.train_dinov2_metric_knn import (
    MetricEncoder,
    actual_tiers,
    pooled_features,
)
from training.experiments.knn_dinov2 import predict as knn_predict
from training.experiments.train_dinov2_temporal_gpu import (
    TemporalOrdinalTransformer,
    read_rows,
)
from training.train import DEFAULT_CSV, NUM_CLASSES


ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).resolve().parent
DEFAULT_METRIC_CHECKPOINT = RESULTS / "all_nonval_metric_knn.pt"
DEFAULT_TEMPORAL_CHECKPOINT = RESULTS / "all_nonval_temporal.pt"


def expected_tier(probabilities):
    tiers = torch.arange(1, probabilities.shape[-1] + 1, dtype=probabilities.dtype)
    return probabilities @ tiers


def temporal_predictions(model, rows, feature_dir, device):
    result = []
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            features = torch.load(
                feature_dir / f"{row['level_id']}.pt", map_location="cpu",
                weights_only=True,
            )["features"].float()
            padding = torch.zeros((1, len(features)), dtype=torch.bool, device=device)
            logits = model(features.float().unsqueeze(0).to(device), padding)
            result.append(torch.softmax(logits.float(), -1).cpu().squeeze(0))
            if index % 1000 == 0:
                print(f"temporal predictions {index}/{len(rows)}", flush=True)
    return torch.stack(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-checkpoint", type=Path, default=DEFAULT_METRIC_CHECKPOINT)
    parser.add_argument("--temporal-checkpoint", type=Path, default=DEFAULT_TEMPORAL_CHECKPOINT)
    parser.add_argument("--title", default="All 6,062 Long levels: training and validation")
    parser.add_argument("--output", type=Path,
                        default=RESULTS / "all_nonval_train_validation_actual_vs_predicted.png")
    parser.add_argument("--predictions-output", type=Path,
                        default=RESULTS / "all_nonval_predictions.csv")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for temporal inference; run outside the restricted sandbox")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    metric = torch.load(args.metric_checkpoint,
                        map_location="cpu", weights_only=True)
    temporal = torch.load(args.temporal_checkpoint,
                          map_location="cpu", weights_only=True)
    train_ids = metric["training_level_ids"]
    validation_ids = metric["validation_level_ids"]
    if validation_ids != temporal["validation_level_ids"]:
        raise ValueError("KNN and temporal checkpoints use different validation rows")
    if train_ids != temporal["training_level_ids"]:
        raise ValueError("KNN and temporal checkpoints use different training rows")
    lookup = {row["level_id"]: row for row in read_rows(DEFAULT_CSV)}
    if set(train_ids) & set(validation_ids):
        raise ValueError("train and validation IDs overlap")
    train_rows = [lookup[level_id] for level_id in train_ids]
    validation_rows = [lookup[level_id] for level_id in validation_ids]

    metric_feature_dir = Path(metric.get(
        "feature_dir", ROOT / "training" / "features_dinov2_base_4fps"
    ))
    train_features = pooled_features(train_rows, metric["pooling"], metric_feature_dir)
    train_targets = torch.stack([row["target"] for row in train_rows])
    encoder = MetricEncoder(input_size=train_features.shape[-1]).to(device)
    encoder.load_state_dict(metric["state_dict"])
    encoder.eval()
    with torch.inference_mode():
        train_embeddings = encoder(train_features.to(device)).cpu()
    original_similarity = F.normalize(train_features, dim=-1)
    original_similarity = original_similarity @ original_similarity.T
    learned_similarity = train_embeddings @ train_embeddings.T
    train_similarity = ((1 - metric["metric_mix"]) * original_similarity
                        + metric["metric_mix"] * learned_similarity)
    train_similarity.fill_diagonal_(-1e4)
    train_knn_probs = knn_predict(
        train_similarity,
        train_targets,
        metric["neighbors"],
        metric["retrieval_temperature"],
    )
    validation_knn_probs = metric["validation_probabilities"]

    temporal_model = TemporalOrdinalTransformer(
        feature_size=temporal["feature_size"],
        dropout=temporal["dropout"],
        pooling=temporal["pooling"],
    ).to(device)
    temporal_model.load_state_dict(temporal["state_dict"])
    feature_dir = Path(temporal["feature_dir"])
    train_temporal_probs = temporal_predictions(temporal_model, train_rows, feature_dir, device)
    validation_temporal_probs = temporal["validation_probabilities"]

    train_probabilities = 0.75 * train_knn_probs + 0.25 * train_temporal_probs
    validation_probabilities = 0.75 * validation_knn_probs + 0.25 * validation_temporal_probs
    train_actual = actual_tiers(train_targets)
    validation_targets = torch.stack([row["target"] for row in validation_rows])
    validation_actual = actual_tiers(validation_targets)
    train_prediction = expected_tier(train_probabilities)
    validation_prediction = expected_tier(validation_probabilities)
    train_errors = (train_prediction - train_actual).abs()
    validation_errors = (validation_prediction - validation_actual).abs()
    train_within3 = train_errors.le(3).float().mean().item()
    validation_within3 = validation_errors.le(3).float().mean().item()

    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    axis.scatter(train_actual.numpy(), train_prediction.numpy(), s=14, alpha=0.28,
                 color="#2878b5", edgecolors="none",
                 label=f"Training (n={len(train_rows):,}, in-sample)")
    axis.scatter(validation_actual.numpy(), validation_prediction.numpy(), s=19, alpha=0.68,
                 color="#e87522", edgecolors="none",
                 label=f"Validation (n={len(validation_rows):,})")
    axis.plot([1, 39], [1, 39], color="#222222", linewidth=1.5,
              label="Perfect prediction")
    axis.plot([1, 36], [4, 39], color="#777777", linestyle="--", linewidth=1.2,
              label="±3 tiers")
    axis.plot([4, 39], [1, 36], color="#777777", linestyle="--", linewidth=1.2)
    axis.set(xlim=(1, 39), ylim=(1, 39), xlabel="Actual tier (vote-weighted mean)",
             ylabel="Predicted tier (probability-weighted mean)",
             title=(f"{args.title}: 75% learned-metric KNN + "
                    "25% temporal Transformer\nexpected-tier readout"))
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.2)
    axis.legend(loc="upper left", framealpha=0.95)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)
    with args.predictions_output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=(
            "level_id", "name", "set", "actual_tier", "predicted_tier", "absolute_error",
        ))
        writer.writeheader()
        for rows, actual, prediction, split_name in (
            (train_rows, train_actual, train_prediction, "training"),
            (validation_rows, validation_actual, validation_prediction, "validation"),
        ):
            for row, x, y in zip(rows, actual.tolist(), prediction.tolist()):
                writer.writerow({
                    "level_id": row["level_id"],
                    "name": row.get("name", ""),
                    "set": split_name,
                    "actual_tier": f"{x:.6f}",
                    "predicted_tier": f"{y:.6f}",
                    "absolute_error": f"{abs(x-y):.6f}",
                })
    print(f"saved={args.output}")
    print(f"saved={args.predictions_output}")
    print(f"levels: training={len(train_rows)} validation={len(validation_rows)}")
    print(f"training in-sample within3={train_within3:.4%} MAE={train_errors.mean():.4f}; "
          f"validation within3={validation_within3:.4%} MAE={validation_errors.mean():.4f}")
    print(f"levels total in chart={len(train_rows) + len(validation_rows)}; "
          "all non-validation levels are training data.")


if __name__ == "__main__":
    main()
