"""Learn a tier-aware DINO metric, then retrieve vote distributions by KNN."""
import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from cross_validate_ensemble import window_mass_center
from training.experiments.knn_dinov2 import predict as knn_predict
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "training" / "features_dinov2_base_4fps"
OUTPUT = ROOT / "training" / "experiments" / "dinov2_metric_knn.pt"
SEED = 42
MAX_EPOCHS = 40
PATIENCE = 8
TRAIN_TEMPERATURE = 0.12
BLEND_SIMILARITIES = (0.0, 0.5, 1.0)
NEIGHBORS = (20, 40, 80)
RETRIEVAL_TEMPERATURES = (0.03, 0.08, 0.16)


class MetricEncoder(nn.Module):
    def __init__(self, input_size=768, output_size=128, dropout=0.2):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(input_size))
        self.register_buffer("feature_std", torch.ones(input_size))
        self.projection = nn.Sequential(
            nn.Linear(input_size, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, output_size), nn.LayerNorm(output_size),
        )

    def forward(self, features):
        normalized = (features - self.feature_mean) / self.feature_std
        return F.normalize(self.projection(normalized), dim=-1)


def pooled_features(rows, pooling, feature_dir=FEATURE_DIR):
    results = []
    for row in rows:
        item = torch.load(feature_dir / f"{row['level_id']}.pt", map_location="cpu",
                          weights_only=True)
        sequence = item["features"].float()
        mean = sequence.mean(0)
        if pooling == "mean_std":
            std = sequence.std(0, unbiased=False)
            mean = torch.cat((mean, std))
        results.append(mean)
    return torch.stack(results)


def actual_tiers(targets):
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    return targets @ tiers


def loss_for_batch(embedding, targets, relation):
    tiers = torch.arange(1, NUM_CLASSES + 1, device=targets.device, dtype=targets.dtype)
    tier_distance = (tiers[:, None] - tiers[None, :]).abs()
    distance = targets @ tier_distance @ targets.T
    if relation == "within3":
        in_window = (tier_distance <= 3).to(targets.dtype)
        similarity_target = targets @ in_window @ targets.T
    else:
        similarity_target = torch.exp(-distance / 3.0)
    diagonal = torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    similarity_target.masked_fill_(diagonal, 0)
    similarity_target /= similarity_target.sum(-1, keepdim=True).clamp_min(1e-8)

    logits = embedding @ embedding.T / TRAIN_TEMPERATURE
    logits = logits.masked_fill(diagonal, -1e4)
    return -(similarity_target * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def evaluate_retrieval(similarity, labels, actual):
    max_k = max(NEIGHBORS)
    sims, indices = similarity.topk(max_k, dim=-1)
    best = None
    # `similarity` is already the selected convex mixture for one metric weight.
    for k in NEIGHBORS:
        local_sims = sims[:, :k]
        local_labels = labels[indices[:, :k]]
        for temperature in RETRIEVAL_TEMPERATURES:
            weights = torch.softmax((local_sims - local_sims[:, :1]) / temperature, dim=-1)
            distribution = (local_labels * weights.unsqueeze(-1)).sum(1)
            prediction = window_mass_center(distribution, 3)
            error = (prediction - actual).abs()
            within = (error <= 3).float().mean().item()
            mae = error.mean().item()
            candidate = (within, -mae, k, temperature, distribution)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    return best


def tune_inner(model, train_features, train_targets, fit_indices, hold_indices):
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        fit_embedding = model(train_features[fit_indices].to(device)).cpu()
        hold_embedding = model(train_features[hold_indices].to(device)).cpu()
    original = F.normalize(train_features, dim=-1)
    original_similarity = original[hold_indices] @ original[fit_indices].T
    hold_actual = actual_tiers(train_targets[hold_indices])
    raw_candidate = evaluate_retrieval(
        original_similarity, train_targets[fit_indices], hold_actual
    )
    best = None
    for metric_weight in BLEND_SIMILARITIES:
        learned_similarity = hold_embedding @ fit_embedding.T
        similarity = ((1 - metric_weight) * original_similarity
                      + metric_weight * learned_similarity)
        candidate = evaluate_retrieval(
            similarity, train_targets[fit_indices], hold_actual
        )
        choice = (candidate[0], candidate[1], -metric_weight)
        if best is None or choice > best[0]:
            best = (choice, metric_weight, candidate[2], candidate[3])
    return best[1], best[2], best[3], best[0][:2], raw_candidate[2], raw_candidate[3]


def fit(model, features, targets, indices, epochs, device, relation, verbose=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = indices[torch.randperm(len(indices), device=indices.device)]
        losses = []
        for batch_ids in permutation.split(128):
            batch_features = features[batch_ids].to(device)
            batch_targets = targets[batch_ids].to(device)
            embedding = model(batch_features)
            loss = loss_for_batch(embedding, batch_targets, relation)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())
        if verbose:
            print(f"epoch={epoch:02d} contrastive_loss={torch.stack(losses).mean().item():.4f} "
                  f"gpu_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"model and minibatch seed (default: {SEED})")
    parser.add_argument("--split-seed", type=int, default=SEED,
                        help="randomized train/validation split seed; original test IDs stay excluded")
    parser.add_argument("--validation-ids-file", type=Path,
                        help="one level ID per line; when set, all other dataset rows are training data")
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR,
                        help=f"cached DINO feature directory (default: {FEATURE_DIR})")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help=f"checkpoint destination (default: {OUTPUT})")
    parser.add_argument("--relation", choices=("exp_distance", "within3"),
                        default="exp_distance",
                        help="soft pair similarity target for contrastive learning")
    parser.add_argument("--pooling", choices=("mean", "mean_std"), default="mean",
                        help="DINO temporal pooling representation")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; run outside the restricted sandbox")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    feature_dir = args.feature_dir.resolve()
    rows = read_rows(DEFAULT_CSV)
    boundary = math.ceil(len(rows) * 0.1)
    if args.validation_ids_file:
        requested_ids = [line.strip() for line in args.validation_ids_file.read_text().splitlines()
                         if line.strip()]
        if len(requested_ids) != len(set(requested_ids)):
            parser.error("validation IDs file contains duplicate IDs")
        row_by_id = {row["level_id"]: row for row in rows}
        missing_ids = set(requested_ids) - set(row_by_id)
        if missing_ids:
            parser.error(f"validation IDs missing from dataset: {sorted(missing_ids)[:5]}")
        valid_rows = [row_by_id[level_id] for level_id in requested_ids]
        validation_id_set = set(requested_ids)
        train_rows = [row for row in rows if row["level_id"] not in validation_id_set]
    else:
        random.Random(SEED).shuffle(rows)
        development_rows = rows[boundary:]
        if args.split_seed == SEED:
            split_rows = development_rows
        else:
            split_rows = list(development_rows)
            random.Random(args.split_seed).shuffle(split_rows)
        valid_rows = split_rows[:boundary]
        train_rows = split_rows[boundary:]
    train_features = pooled_features(train_rows, args.pooling, feature_dir)
    train_targets = torch.stack([row["target"] for row in train_rows])
    valid_features = pooled_features(valid_rows, args.pooling, feature_dir)
    valid_targets = torch.stack([row["target"] for row in valid_rows])
    valid_actual = actual_tiers(valid_targets)
    print(f"train={len(train_rows)} validation={len(valid_rows)}; test partition not loaded", flush=True)

    permutation = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(21))
    hold_count = max(1, math.ceil(len(train_rows) * 0.1))
    hold_indices = permutation[:hold_count]
    fit_indices = permutation[hold_count:]

    model = MetricEncoder(input_size=train_features.shape[-1]).to(device)
    mean = train_features[fit_indices].mean(0)
    std = train_features[fit_indices].std(0).clamp_min(1e-5)
    model.feature_mean.copy_(mean.to(device))
    model.feature_std.copy_(std.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    best_selection, best_epoch, stale = None, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_indices = fit_indices[torch.randperm(len(fit_indices))]
        losses = []
        for batch_ids in epoch_indices.split(128):
            x = train_features[batch_ids].to(device)
            target = train_targets[batch_ids].to(device)
            embedding = model(x)
            loss = loss_for_batch(embedding, target, args.relation)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())

        metric_weight, k, temperature, selection, _, _ = tune_inner(
            model, train_features, train_targets, fit_indices, hold_indices
        )
        improved = best_selection is None or selection > best_selection
        if improved:
            best_selection = selection
            best_epoch = epoch
            best_params = (metric_weight, k, temperature)
            stale = 0
        else:
            stale += 1
        print(f"epoch={epoch:02d} loss={torch.stack(losses).mean().item():.4f} "
              f"inner_within3={selection[0]:.4f} inner_MAE={-selection[1]:.4f} "
              f"metric_mix={metric_weight:g} k={k} temp={temperature:g} "
              f"best_epoch={best_epoch} gpu_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB",
              flush=True)
        if stale >= PATIENCE:
            break

    metric_weight, k, temperature = best_params
    with torch.inference_mode():
        original_inner_similarity = (
            F.normalize(train_features[hold_indices], dim=-1)
            @ F.normalize(train_features[fit_indices], dim=-1).T
        )
    raw_inner = evaluate_retrieval(
        original_inner_similarity, train_targets[fit_indices],
        actual_tiers(train_targets[hold_indices])
    )
    del model, optimizer
    torch.manual_seed(args.seed)
    final_model = MetricEncoder(input_size=train_features.shape[-1]).to(device)
    final_model.feature_mean.copy_(train_features.mean(0).to(device))
    final_model.feature_std.copy_(train_features.std(0).clamp_min(1e-5).to(device))
    fit(final_model, train_features, train_targets, torch.arange(len(train_rows)),
        best_epoch, device, args.relation, verbose=True)

    final_model.eval()
    with torch.inference_mode():
        train_embedding = final_model(train_features.to(device)).cpu()
        valid_embedding = final_model(valid_features.to(device)).cpu()
    learned_similarity = valid_embedding @ train_embedding.T
    original_similarity = (F.normalize(valid_features, dim=-1)
                           @ F.normalize(train_features, dim=-1).T)
    similarities = ((1 - metric_weight) * original_similarity
                    + metric_weight * learned_similarity)
    distribution = knn_predict(similarities, train_targets, k, temperature)
    prediction = window_mass_center(distribution, 3)
    error = (prediction - valid_actual).abs()
    validation_scores = {"window_mae": error.mean().item(),
                         "window_within3": (error <= 3).float().mean().item()}
    raw_distribution = knn_predict(original_similarity, train_targets,
                                   raw_inner[2], raw_inner[3])
    raw_prediction = window_mass_center(raw_distribution, 3)
    raw_error = (raw_prediction - valid_actual).abs()
    raw_validation_scores = {"window_mae": raw_error.mean().item(),
                             "window_within3": (raw_error <= 3).float().mean().item()}
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in final_model.state_dict().items()},
        "model": "tier-aware supervised contrastive DINO KNN",
        "relation": args.relation,
        "pooling": args.pooling,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "epoch": best_epoch,
        "parameters": sum(parameter.numel() for parameter in final_model.parameters()),
        "metric_mix": metric_weight,
        "neighbors": k,
        "retrieval_temperature": temperature,
        "validation_probabilities": distribution,
        "validation_scores": validation_scores,
        "raw_knn_parameters": {"neighbors": raw_inner[2],
                                "retrieval_temperature": raw_inner[3]},
        "raw_knn_validation_probabilities": raw_distribution,
        "raw_knn_validation_scores": raw_validation_scores,
        "inner_holdout_score": best_selection,
        "validation_level_ids": [row["level_id"] for row in valid_rows],
        "training_level_ids": [row["level_id"] for row in train_rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"device={torch.cuda.get_device_name(0)} selected_epoch={best_epoch} "
          f"parameters={payload['parameters']} validation={validation_scores} "
          f"metric_mix={metric_weight:g} k={k} temp={temperature:g} "
          f"gpu_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
    print("Held-out test partition was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
