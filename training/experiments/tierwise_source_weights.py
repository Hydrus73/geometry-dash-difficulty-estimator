"""Nested-CV ensemble with smoothly tier-dependent expert weights."""
import random
from pathlib import Path

import torch

from cross_validate_ensemble import score, window_mass_center
from ensemble_calibration_experiments import make_components
from knn_dinov2 import predict as knn_predict
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "training" / "features_dinov2_base_4fps"
TEMPORAL_CHECKPOINT = Path(__file__).resolve().parent / "dinov2_1s_batched_transformer_within3_w05.pt"
BASE_WEIGHTS = torch.tensor([0.225, 0.225, 0.225, 0.075, 0.25])
PENALTIES = (0.001, 0.003, 0.01, 0.03, 0.1)


def fit_and_apply(source_probabilities, targets, penalty, apply_probabilities=None):
    """Fit class-specific source weights with ordinal smoothness and shrinkage."""
    source_count, _, class_count = source_probabilities.shape
    log_prior = BASE_WEIGHTS.log()[None, :].expand(class_count, -1)
    delta = torch.nn.Parameter(torch.zeros_like(log_prior))
    optimizer = torch.optim.LBFGS(
        [delta], lr=0.5, max_iter=80, tolerance_grad=1e-8,
        tolerance_change=1e-10, line_search_fn="strong_wolfe",
    )

    def compose(values, weights):
        # values [source, item, tier], weights [tier, source]
        result = (values.permute(1, 2, 0) * weights[None]).sum(-1)
        return result / result.sum(-1, keepdim=True).clamp_min(1e-8)

    def closure():
        optimizer.zero_grad()
        adjustment = delta - delta.mean(-1, keepdim=True)
        weights = torch.softmax(log_prior + adjustment, dim=-1)
        prediction = compose(source_probabilities, weights)
        log_prediction = prediction.clamp_min(1e-7).log()
        cross_entropy = -(targets * log_prediction).sum(-1).mean()
        curvature = adjustment[2:] - 2 * adjustment[1:-1] + adjustment[:-2]
        loss = (cross_entropy + penalty * adjustment.square().sum()
                + penalty * curvature.square().sum())
        loss.backward()
        return loss

    optimizer.step(closure)
    adjustment = delta.detach() - delta.detach().mean(-1, keepdim=True)
    weights = torch.softmax(log_prior + adjustment, dim=-1)
    if apply_probabilities is None:
        apply_probabilities = source_probabilities
    return compose(apply_probabilities, weights), weights


def evaluate(probabilities, target_tiers):
    return score(window_mass_center(probabilities, 3), target_tiers)


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    split_size = max(1, round(len(rows) * 0.1 + 0.499999))
    train_rows = rows[2 * split_size:]
    valid_rows = rows[split_size:2 * split_size]
    train_targets = torch.stack([row["target"] for row in train_rows])
    valid_targets = torch.stack([row["target"] for row in valid_rows])
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    valid_tiers = valid_targets @ tiers

    print("Building source distributions on CPU...", flush=True)
    groups, valid_features, temporal = make_components(valid_rows)
    train_features = []
    for row in train_rows:
        cache = torch.load(FEATURE_DIR / f"{row['level_id']}.pt",
                           map_location="cpu", weights_only=True)
        train_features.append(cache["features"].float().mean(0))
    train_features = torch.nn.functional.normalize(torch.stack(train_features), dim=-1)
    retrieval = knn_predict(valid_features @ train_features.T, train_targets, 20, 0.02)
    sources = torch.cat((groups, retrieval[None], temporal[None]), dim=0)
    baseline = (BASE_WEIGHTS[:, None, None] * sources).sum(0)

    fold_ids = torch.arange(len(valid_rows)) % 5
    baseline_scores, fitted_scores, chosen_penalties = [], [], []
    for outer_fold in range(5):
        outer_train, outer_valid = fold_ids != outer_fold, fold_ids == outer_fold
        local_train = torch.where(outer_train)[0]
        inner_ids = torch.arange(len(local_train)) % 4
        inner_means = {}
        for penalty in PENALTIES:
            results = []
            for inner_fold in range(4):
                inner_fit = inner_ids != inner_fold
                inner_holdout = ~inner_fit
                fit_indices, holdout_indices = local_train[inner_fit], local_train[inner_holdout]
                prediction, _ = fit_and_apply(
                    sources[:, fit_indices], valid_targets[fit_indices], penalty,
                    sources[:, holdout_indices],
                )
                results.append(evaluate(prediction, valid_tiers[holdout_indices]))
            inner_means[penalty] = tuple(sum(x[i] for x in results) / 4 for i in range(2))

        penalty = max(PENALTIES, key=lambda p: (
            inner_means[p][1], -inner_means[p][0], p,
        ))
        chosen_penalties.append(penalty)
        prediction, weights = fit_and_apply(
            sources[:, outer_train], valid_targets[outer_train], penalty,
            sources[:, outer_valid],
        )
        baseline_result = evaluate(baseline[outer_valid], valid_tiers[outer_valid])
        fitted_result = evaluate(prediction, valid_tiers[outer_valid])
        baseline_scores.append(baseline_result)
        fitted_scores.append(fitted_result)
        print(f"fold {outer_fold + 1}: penalty={penalty:g}, "
              f"baseline={baseline_result}, tierwise={fitted_result}, "
              f"mean source weights={weights.mean(0).tolist()}", flush=True)

    average = lambda values: tuple(sum(x[i] for x in values) / 5 for i in range(2))
    print(f"5-fold baseline (MAE, within3): {average(baseline_scores)}")
    print(f"5-fold tierwise source weights (MAE, within3): {average(fitted_scores)}")
    print(f"selected penalties: {chosen_penalties}")


if __name__ == "__main__":
    main()
