"""Nested-CV class-offset calibration for the current ensemble distribution."""
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
REGULARIZATIONS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)


def calibrate(probabilities, targets, regularization, application_probabilities=None):
    """Fit ordinally smoothed class logit offsets to soft vote targets."""
    classes = probabilities.shape[-1]
    log_probabilities = probabilities.clamp_min(1e-7).log()
    bias = torch.nn.Parameter(torch.zeros(classes))
    optimizer = torch.optim.LBFGS(
        [bias], lr=0.5, max_iter=80, tolerance_grad=1e-8,
        tolerance_change=1e-10, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        centered_bias = bias - bias.mean()
        log_distribution = torch.log_softmax(log_probabilities + centered_bias, dim=-1)
        cross_entropy = -(targets * log_distribution).sum(-1).mean()
        curvature = centered_bias[2:] - 2 * centered_bias[1:-1] + centered_bias[:-2]
        loss = (cross_entropy + regularization * centered_bias.square().sum()
                + 0.2 * regularization * curvature.square().sum())
        loss.backward()
        return loss

    optimizer.step(closure)
    centered_bias = bias.detach() - bias.detach().mean()
    if application_probabilities is None:
        application_probabilities = probabilities
    application_log_probabilities = application_probabilities.clamp_min(1e-7).log()
    return torch.softmax(application_log_probabilities + centered_bias, dim=-1)


def mean_score(probabilities, target_tiers):
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

    print("Building current ensemble distributions on CPU...", flush=True)
    groups, valid_features, temporal = make_components(valid_rows)
    train_features = []
    for row in train_rows:
        cache = torch.load(FEATURE_DIR / f"{row['level_id']}.pt",
                           map_location="cpu", weights_only=True)
        train_features.append(cache["features"].float().mean(0))
    train_features = torch.nn.functional.normalize(torch.stack(train_features), dim=-1)
    similarity = valid_features @ train_features.T
    retrieval = knn_predict(similarity, train_targets, 20, 0.02)
    fixed = 0.75 * (0.9 * groups.mean(0) + 0.1 * retrieval) + 0.25 * temporal

    fold_ids = torch.arange(len(valid_rows)) % 5
    baseline_scores, calibrated_scores, selected_regularizations = [], [], []
    for outer_fold in range(5):
        outer_train = fold_ids != outer_fold
        outer_valid = ~outer_train
        local_train = torch.where(outer_train)[0]
        inner_ids = torch.arange(len(local_train)) % 4

        validation_by_reg = {}
        for regularization in REGULARIZATIONS:
            inner_scores = []
            for inner_fold in range(4):
                inner_fit = inner_ids != inner_fold
                inner_holdout = ~inner_fit
                fit_indices = local_train[inner_fit]
                holdout_indices = local_train[inner_holdout]
                calibrated = calibrate(
                    fixed[fit_indices], valid_targets[fit_indices], regularization,
                    fixed[holdout_indices],
                )
                inner_scores.append(mean_score(calibrated, valid_tiers[holdout_indices]))
            average = tuple(sum(result[i] for result in inner_scores) / 4 for i in range(2))
            validation_by_reg[regularization] = average

        # Favor the stronger correction only when it wins within-three accuracy;
        # ties go to the more regularized option.
        regularization = max(
            REGULARIZATIONS,
            key=lambda value: (validation_by_reg[value][1],
                               -validation_by_reg[value][0], value),
        )
        selected_regularizations.append(regularization)
        calibrated = calibrate(
            fixed[outer_train], valid_targets[outer_train], regularization,
            fixed[outer_valid],
        )
        result = mean_score(calibrated, valid_tiers[outer_valid])
        baseline = mean_score(fixed[outer_valid], valid_tiers[outer_valid])
        baseline_scores.append(baseline)
        calibrated_scores.append(result)
        print(f"fold {outer_fold + 1}: selected_penalty={regularization:g}, "
              f"baseline={baseline}, calibrated={result}", flush=True)

    average = lambda values: tuple(sum(result[i] for result in values) / 5 for i in range(2))
    print(f"5-fold baseline (MAE, within3): {average(baseline_scores)}")
    print(f"5-fold nested distribution calibration (MAE, within3): {average(calibrated_scores)}")
    print(f"selected penalties: {selected_regularizations}")


if __name__ == "__main__":
    main()
