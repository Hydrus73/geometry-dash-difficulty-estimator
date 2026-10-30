"""Cross-validate ensemble weights and distribution readout on validation only."""
import itertools
import math
import random
from pathlib import Path

import torch

from evaluate_validation import MODELS, predict
from knn_dinov2 import make_representations, predict as knn_predict
from name_knn import vectorize as vectorize_names, predict as name_knn_predict
from level_id_prior import predict as level_id_predict
from ridge_dinov2 import FEATURE_DIR, fit_ridge, tiers_from_cdf
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


def tier_values(distributions):
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    mean = distributions @ tiers
    median = torch.searchsorted(
        distributions.cumsum(-1),
        torch.full((len(distributions), 1), 0.5, dtype=distributions.dtype),
    ).squeeze(-1).float() + 1
    return mean, median


def window_mass_center(distributions, radius):
    """Choose the tier whose +/-radius interval has greatest probability mass."""
    classes = distributions.shape[-1]
    cumulative = torch.cat((
        torch.zeros(len(distributions), 1, dtype=distributions.dtype),
        distributions.cumsum(-1),
    ), dim=-1)
    centers = torch.arange(classes)
    lower = (centers - radius).clamp_min(0)
    upper = (centers + radius + 1).clamp_max(classes)
    mass = cumulative[:, upper] - cumulative[:, lower]
    return mass.argmax(-1).float() + 1


def fit_isotonic(x, y, bins):
    """Fit a small monotone binned calibration map with pool-adjacent violators."""
    unique_x, inverse = torch.unique(x, sorted=True, return_inverse=True)
    counts = torch.zeros(len(unique_x), dtype=x.dtype)
    sums = torch.zeros(len(unique_x), dtype=y.dtype)
    counts.scatter_add_(0, inverse, torch.ones_like(x))
    sums.scatter_add_(0, inverse, y)
    unique_y = sums / counts
    blocks = []
    for indices in torch.tensor_split(torch.arange(len(unique_x)), min(bins, len(unique_x))):
        if not len(indices):
            continue
        weight = float(counts[indices].sum())
        blocks.append([
            float((unique_x[indices] * counts[indices]).sum() / weight),
            float((unique_y[indices] * counts[indices]).sum() / weight),
            weight,
        ])
        while len(blocks) > 1 and blocks[-2][1] > blocks[-1][1]:
            left, right = blocks[-2], blocks[-1]
            total = left[2] + right[2]
            merged = [
                (left[0] * left[2] + right[0] * right[2]) / total,
                (left[1] * left[2] + right[1] * right[2]) / total,
                total,
            ]
            blocks[-2:] = [merged]
    centers = torch.tensor([block[0] for block in blocks])
    values = torch.tensor([block[1] for block in blocks])
    return centers, values


def apply_isotonic(calibrator, x):
    centers, values = calibrator
    if len(centers) == 1:
        return values[0].expand_as(x)
    right = torch.searchsorted(centers, x).clamp(1, len(centers) - 1)
    left = right - 1
    fraction = ((x - centers[left]) / (centers[right] - centers[left]).clamp_min(1e-6)).clamp(0, 1)
    return values[left] + fraction * (values[right] - values[left])


def score(prediction, targets):
    errors = (prediction - targets).abs()
    return errors.mean().item(), (errors <= 3).float().mean().item()


def simplex_weights(parts=3, step=0.1):
    n = round(1 / step)
    for values in itertools.product(range(n + 1), repeat=parts - 1):
        if sum(values) <= n:
            complete = (*values, n - sum(values))
            yield tuple(v / n for v in complete)


def main():
    torch.set_num_threads(4)
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    split_size = max(1, round(len(rows) * 0.1 + 0.499999))
    rows = rows[split_size:2 * split_size]
    targets = torch.stack([row["target"] for row in rows])
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    actual = targets @ tiers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")
    raw = {name: predict(name, rows, device) for name in MODELS}
    groups = {
        "VideoMAE": torch.stack([raw[n] for n in (
            "videomae_pooled", "videomae_mae", "videomae_categorical"
        )]).mean(0),
        "DINO-small": torch.stack([raw[n] for n in (
            "dino_small_ordinal", "dino_small_ordinal_seed7", "dino_small_categorical"
        )]).mean(0),
        "DINO-base": torch.stack([raw[n] for n in (
            "dino_base_4s_ordinal", "dino_base_4s_categorical",
            "dino_base_4s_categorical_seed7", "dino_base_4s_dropout06"
        )]).mean(0),
    }
    names = tuple(groups)
    matrices = torch.stack([groups[n] for n in names])
    fold_ids = torch.arange(len(rows)) % 5
    per_fold = []

    for fold in range(5):
        train = fold_ids != fold
        valid = ~train
        best = None
        for weights in simplex_weights(len(names), 0.1):
            distribution = torch.einsum("g,gvc->vc", torch.tensor(weights), matrices[:, train])
            mean, median = tier_values(distribution)
            for alpha in torch.linspace(0, 1, 11):
                prediction = alpha * mean + (1 - alpha) * median
                mae, within3 = score(prediction, actual[train])
                candidate = (within3, -mae, weights, float(alpha))
                if best is None or candidate[:2] > best[:2]:
                    best = candidate

        weights, alpha = best[2], best[3]
        valid_distribution = torch.einsum(
            "g,gvc->vc", torch.tensor(weights), matrices[:, valid]
        )
        mean, median = tier_values(valid_distribution)
        prediction = alpha * mean + (1 - alpha) * median
        fold_mae, fold_within3 = score(prediction, actual[valid])
        per_fold.append((fold_mae, fold_within3))
        print(f"fold {fold + 1}: tuned_weights={dict(zip(names, weights))}, "
              f"mean_fraction={alpha:.1f}, MAE={fold_mae:.4f}, within3={fold_within3:.4f}")

    print("5-fold tuned readout average: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in per_fold) / len(per_fold),
        sum(x[1] for x in per_fold) / len(per_fold),
    ))

    # Compare a fixed, untuned point estimate across the same folds.
    distributions = matrices.mean(0)
    mean, median = tier_values(distributions)
    for label, prediction in (("group-mean expectation", mean), ("group-mean median", median)):
        scores = [score(prediction[fold_ids == fold], actual[fold_ids == fold]) for fold in range(5)]
        print(f"{label} 5-fold: MAE={sum(s[0] for s in scores) / 5:.4f}, "
              f"within3={sum(s[1] for s in scores) / 5:.4f}")

    # Calibrate only the point readout while keeping the probability ensemble fixed.
    calibrated_scores = []
    for fold in range(5):
        train, valid = fold_ids != fold, fold_ids == fold
        best = None
        for alpha in torch.linspace(0, 1, 11):
            base = alpha * mean + (1 - alpha) * median
            for offset in torch.arange(-1.5, 1.51, 0.25):
                mae, within3 = score(base[train] + offset, actual[train])
                candidate = (within3, -mae, float(alpha), float(offset))
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        prediction = best[2] * mean[valid] + (1 - best[2]) * median[valid] + best[3]
        result = score(prediction, actual[valid])
        calibrated_scores.append(result)
        print(f"fold {fold + 1} fixed-ensemble readout: mean_fraction={best[2]:.1f}, "
              f"offset={best[3]:+.2f}, MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("5-fold readout-calibrated average: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in calibrated_scores) / 5,
        sum(x[1] for x in calibrated_scores) / 5,
    ))

    # Test whether the weaker, independently trained linear ordinal model adds
    # complementary signal to the fixed group-mean distribution.
    all_rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(all_rows)
    split_size = (len(all_rows) + 9) // 10
    pooled, variability = [], []
    for row in all_rows:
        cache = torch.load(FEATURE_DIR / f"{row['level_id']}.pt", map_location="cpu", weights_only=True)
        features = cache["features"].float()
        pooled.append(features.mean(0))
        variability.append(features.std(0, unbiased=False))
    pooled = torch.stack(pooled)
    variability = torch.stack(variability)
    all_targets = torch.stack([row["target"] for row in all_rows])
    train_idx = torch.arange(2 * split_size, len(all_rows))
    valid_idx = torch.arange(split_size, 2 * split_size)
    ridge_cdf = fit_ridge(
        pooled[train_idx], all_targets.cumsum(-1)[train_idx, :-1], pooled[valid_idx], 0.01
    )
    ridge_probs, _, _ = tiers_from_cdf(ridge_cdf)
    blend_scores = []
    for fold in range(5):
        train, valid = fold_ids != fold, fold_ids == fold
        best = None
        for ridge_weight in torch.linspace(0, 1, 11):
            blend = (1 - ridge_weight) * distributions + ridge_weight * ridge_probs
            blend_mean, blend_median = tier_values(blend)
            for label, prediction in (("mean", blend_mean), ("median", blend_median)):
                mae, within3 = score(prediction[train], actual[train])
                candidate = (within3, -mae, float(ridge_weight), label)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        weight, label = best[2], best[3]
        blend = (1 - weight) * distributions + weight * ridge_probs
        blend_mean, blend_median = tier_values(blend)
        result = score((blend_mean if label == "mean" else blend_median)[valid], actual[valid])
        blend_scores.append(result)
        print(f"fold {fold + 1} ridge blend: ridge_weight={weight:.1f}, readout={label}, "
              f"MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("5-fold ridge blend average: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in blend_scores) / 5,
        sum(x[1] for x in blend_scores) / 5,
    ))

    # The KNN hyperparameters were selected on a separate training-only holdout.
    knn_features = make_representations(pooled, variability)["mean"]
    similarities = knn_features[valid_idx] @ knn_features[train_idx].T
    knn_distribution = knn_predict(similarities, all_targets[train_idx], 20, 0.02)
    knn_scores = []
    for fold in range(5):
        train, valid = fold_ids != fold, fold_ids == fold
        best = None
        for knn_weight in torch.linspace(0, 1, 11):
            blend = (1 - knn_weight) * distributions + knn_weight * knn_distribution
            blend_mean, blend_median = tier_values(blend)
            for label, prediction in (("mean", blend_mean), ("median", blend_median)):
                mae, within3 = score(prediction[train], actual[train])
                candidate = (within3, -mae, float(knn_weight), label)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        weight, label = best[2], best[3]
        blend = (1 - weight) * distributions + weight * knn_distribution
        blend_mean, blend_median = tier_values(blend)
        result = score((blend_mean if label == "mean" else blend_median)[valid], actual[valid])
        knn_scores.append(result)
        print(f"fold {fold + 1} KNN blend: KNN_weight={weight:.1f}, readout={label}, "
              f"MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("5-fold KNN blend average: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in knn_scores) / 5,
        sum(x[1] for x in knn_scores) / 5,
    ))
    final_blend = 0.9 * distributions + 0.1 * knn_distribution
    final_mean, final_median = tier_values(final_blend)
    print("fixed 90% group ensemble + 10% KNN validation mean:", score(final_mean, actual))
    print("fixed 90% group ensemble + 10% KNN validation median:", score(final_median, actual))

    # Compare point readouts that optimize probability mass within a tier band.
    readouts = {"mean": final_mean, "median": final_median}
    for radius in (1, 2, 3, 4, 5):
        readouts[f"window+/-{radius}"] = window_mass_center(final_blend, radius)
    for name, prediction in readouts.items():
        fold_scores = [score(prediction[fold_ids == fold], actual[fold_ids == fold])
                       for fold in range(5)]
        print(f"full validation readout {name}: {score(prediction, actual)}")
        print(f"fixed readout {name} 5-fold: "
              f"MAE={sum(x[0] for x in fold_scores) / 5:.4f}, "
              f"within3={sum(x[1] for x in fold_scores) / 5:.4f}")

    selected_scores = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        candidates = []
        for name, prediction in readouts.items():
            mae, within3 = score(prediction[tune], actual[tune])
            candidates.append((within3, -mae, name))
        selected = max(candidates)
        prediction = readouts[selected[2]]
        result = score(prediction[held], actual[held])
        selected_scores.append(result)
        print(f"fold {fold + 1} selected readout={selected[2]}: "
              f"MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("five-fold selected distribution readout: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in selected_scores) / 5,
        sum(x[1] for x in selected_scores) / 5,
    ))

    # Learn a monotone score-to-tier mapping on four validation folds, with
    # calibrator complexity selected by inner folds, then score the fifth.
    calibration_inputs = {
        "mean": final_mean,
        "median": final_median,
        "window+/-2": window_mass_center(final_blend, 2),
        "window+/-3": window_mass_center(final_blend, 3),
    }
    isotonic_scores = []
    for fold in range(5):
        tune_ids = torch.where(fold_ids != fold)[0]
        held = fold_ids == fold
        inner_fold = torch.arange(len(tune_ids)) % 4
        best = None
        for input_name, raw_prediction in calibration_inputs.items():
            for bins in (5, 10, 20, 40):
                inner_scores = []
                for inner in range(4):
                    fit_ids = tune_ids[inner_fold != inner]
                    score_ids = tune_ids[inner_fold == inner]
                    calibration = fit_isotonic(
                        raw_prediction[fit_ids], actual[fit_ids], bins
                    )
                    calibrated = apply_isotonic(calibration, raw_prediction[score_ids]).clamp(1, NUM_CLASSES)
                    inner_scores.append(score(calibrated, actual[score_ids]))
                candidate = (
                    sum(item[1] for item in inner_scores) / 4,
                    -sum(item[0] for item in inner_scores) / 4,
                    input_name,
                    bins,
                )
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        input_name, bins = best[2], best[3]
        raw_prediction = calibration_inputs[input_name]
        calibration = fit_isotonic(raw_prediction[tune_ids], actual[tune_ids], bins)
        calibrated = apply_isotonic(calibration, raw_prediction[held]).clamp(1, NUM_CLASSES)
        result = score(calibrated, actual[held])
        isotonic_scores.append(result)
        print(f"fold {fold + 1} monotone calibration: input={input_name}, bins={bins}, "
              f"MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("five-fold nested monotone calibration: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in isotonic_scores) / 5,
        sum(x[1] for x in isotonic_scores) / 5,
    ))

    # Retune group and retrieval weights for the within-three objective using
    # only four validation folds at a time.
    tuned_scores = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        for group_weights in simplex_weights(len(names), 0.2):
            group_distribution = torch.einsum(
                "g,gvc->vc", torch.tensor(group_weights), matrices
            )
            for knn_weight in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                mixed = (1 - knn_weight) * group_distribution + knn_weight * knn_distribution
                mean, median = tier_values(mixed)
                candidates = {"mean": mean, "median": median}
                for radius in (2, 3, 4):
                    candidates[f"window+/-{radius}"] = window_mass_center(mixed, radius)
                for readout_name, point in candidates.items():
                    mae, within3 = score(point[tune], actual[tune])
                    candidate = (within3, -mae, group_weights, knn_weight, readout_name)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
        group_weights, knn_weight, readout_name = best[2], best[3], best[4]
        group_distribution = torch.einsum("g,gvc->vc", torch.tensor(group_weights), matrices)
        mixed = (1 - knn_weight) * group_distribution + knn_weight * knn_distribution
        mean, median = tier_values(mixed)
        point = {"mean": mean, "median": median}
        for radius in (2, 3, 4):
            point[f"window+/-{radius}"] = window_mass_center(mixed, radius)
        result = score(point[readout_name][held], actual[held])
        tuned_scores.append(result)
        print(f"fold {fold + 1} tuned video/KNN readout: "
              f"groups={dict(zip(names, group_weights))}, KNN={knn_weight:.1f}, "
              f"readout={readout_name}, MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("five-fold tuned video/KNN readout: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in tuned_scores) / 5,
        sum(x[1] for x in tuned_scores) / 5,
    ))

    # Calibrate sharpness of the fixed ensemble distribution before making a
    # within-three decision. Select the exponent and readout on other folds.
    sharpness_scores = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        for gamma in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            calibrated = final_blend.clamp_min(1e-12).pow(gamma)
            calibrated = calibrated / calibrated.sum(-1, keepdim=True)
            mean, median = tier_values(calibrated)
            candidates = {"mean": mean, "median": median}
            for radius in (2, 3):
                candidates[f"window+/-{radius}"] = window_mass_center(calibrated, radius)
            for readout_name, point in candidates.items():
                mae, within3 = score(point[tune], actual[tune])
                candidate = (within3, -mae, gamma, readout_name)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        gamma, readout_name = best[2], best[3]
        calibrated = final_blend.clamp_min(1e-12).pow(gamma)
        calibrated = calibrated / calibrated.sum(-1, keepdim=True)
        mean, median = tier_values(calibrated)
        point = {"mean": mean, "median": median}
        for radius in (2, 3):
            point[f"window+/-{radius}"] = window_mass_center(calibrated, radius)
        result = score(point[readout_name][held], actual[held])
        sharpness_scores.append(result)
        print(f"fold {fold + 1} sharpness calibration: gamma={gamma:.2f}, "
              f"readout={readout_name}, MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("five-fold sharpness calibration: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in sharpness_scores) / 5,
        sum(x[1] for x in sharpness_scores) / 5,
    ))

    # Check whether the separately trained statistics MLP contributes
    # complementary video signal when blended with the fixed group/KNN model.
    statistics_checkpoint = torch.load(
        DEFAULT_CSV.parent.parent / "training" / "experiments" / "dinov2_statistics_ordinal.pt",
        map_location="cpu", weights_only=True,
    )
    statistics_distribution = statistics_checkpoint["validation_probabilities"].float()
    statistics_scores = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        for model_weight in torch.linspace(0, 0.5, 11):
            mixed = (1 - model_weight) * final_blend + model_weight * statistics_distribution
            mean, median = tier_values(mixed)
            candidates = {
                "mean": mean,
                "median": median,
                "window+/-2": window_mass_center(mixed, 2),
                "window+/-3": window_mass_center(mixed, 3),
            }
            for readout_name, point in candidates.items():
                mae, within3 = score(point[tune], actual[tune])
                candidate = (within3, -mae, float(model_weight), readout_name)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        weight, readout_name = best[2], best[3]
        mixed = (1 - weight) * final_blend + weight * statistics_distribution
        mean, median = tier_values(mixed)
        point = {
            "mean": mean,
            "median": median,
            "window+/-2": window_mass_center(mixed, 2),
            "window+/-3": window_mass_center(mixed, 3),
        }
        result = score(point[readout_name][held], actual[held])
        statistics_scores.append(result)
        print(f"fold {fold + 1} statistics-MLP blend: weight={weight:.2f}, "
              f"readout={readout_name}, MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("five-fold statistics-MLP blend: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in statistics_scores) / 5,
        sum(x[1] for x in statistics_scores) / 5,
    ))

    # The expected-tier DINO head uses a different loss and may contribute
    # complementary point estimates even though it is weaker alone.
    regression_prediction = raw["dino_base_4s_regression"].reshape(-1)
    base_readouts = {
        "mean": final_mean,
        "median": final_median,
        "window+/-2": window_mass_center(final_blend, 2),
        "window+/-3": window_mass_center(final_blend, 3),
    }
    regression_scores = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        for regression_weight in torch.linspace(0, 1, 21):
            for readout_name, base_prediction in base_readouts.items():
                point = (1 - regression_weight) * base_prediction + regression_weight * regression_prediction
                mae, within3 = score(point[tune], actual[tune])
                candidate = (within3, -mae, float(regression_weight), readout_name)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        weight, readout_name = best[2], best[3]
        point = (1 - weight) * base_readouts[readout_name] + weight * regression_prediction
        result = score(point[held], actual[held])
        regression_scores.append(result)
        print(f"fold {fold + 1} expected-tier head blend: weight={weight:.2f}, "
              f"base_readout={readout_name}, MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("five-fold expected-tier head blend: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in regression_scores) / 5,
        sum(x[1] for x in regression_scores) / 5,
    ))

    gaussian_checkpoint = torch.load(
        DEFAULT_CSV.parent.parent / "training" / "experiments" / "dinov2_gaussian_target.pt",
        map_location="cpu", weights_only=True,
    )
    gaussian_distribution = gaussian_checkpoint["validation_probabilities"].float()
    gaussian_scores = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        for model_weight in torch.linspace(0, 0.5, 11):
            mixed = (1 - model_weight) * final_blend + model_weight * gaussian_distribution
            mean, median = tier_values(mixed)
            candidates = {
                "mean": mean,
                "median": median,
                "window+/-2": window_mass_center(mixed, 2),
                "window+/-3": window_mass_center(mixed, 3),
            }
            for readout_name, point in candidates.items():
                mae, within3 = score(point[tune], actual[tune])
                candidate = (within3, -mae, float(model_weight), readout_name)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        weight, readout_name = best[2], best[3]
        mixed = (1 - weight) * final_blend + weight * gaussian_distribution
        mean, median = tier_values(mixed)
        point = {
            "mean": mean,
            "median": median,
            "window+/-2": window_mass_center(mixed, 2),
            "window+/-3": window_mass_center(mixed, 3),
        }
        result = score(point[readout_name][held], actual[held])
        gaussian_scores.append(result)
        print(f"fold {fold + 1} Gaussian-target model blend: weight={weight:.2f}, "
              f"readout={readout_name}, MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("five-fold Gaussian-target model blend: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in gaussian_scores) / 5,
        sum(x[1] for x in gaussian_scores) / 5,
    ))

    tolerance_checkpoint = torch.load(
        DEFAULT_CSV.parent.parent / "training" / "experiments" / "dinov2_tolerance_loss.pt",
        map_location="cpu", weights_only=True,
    )
    tolerance_prediction = tolerance_checkpoint["validation_predictions"].float()
    tolerance_scores = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        for model_weight in torch.linspace(0, 0.5, 11):
            for readout_name, base_prediction in base_readouts.items():
                point = (1 - model_weight) * base_prediction + model_weight * tolerance_prediction
                mae, within3 = score(point[tune], actual[tune])
                candidate = (within3, -mae, float(model_weight), readout_name)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        weight, readout_name = best[2], best[3]
        point = (1 - weight) * base_readouts[readout_name] + weight * tolerance_prediction
        result = score(point[held], actual[held])
        tolerance_scores.append(result)
        print(f"fold {fold + 1} tolerance-loss blend: weight={weight:.2f}, "
              f"base_readout={readout_name}, MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("five-fold tolerance-loss blend: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in tolerance_scores) / 5,
        sum(x[1] for x in tolerance_scores) / 5,
    ))

    # Test the batched one-second temporal Transformer on the same fixed
    # validation split. Its input is visual features only; target distributions
    # are read here solely for scoring and fold-local blend selection.
    fixed_video_knn = 0.9 * distributions + 0.1 * knn_distribution
    # The separately tuned one-second global-pooling retrieval may complement
    # the four-second KNN already in fixed_video_knn.
    one_second_dir = DEFAULT_CSV.parent.parent / "training" / "features_dinov2_base_4fps_1s"
    one_second_means = []
    for row in all_rows:
        features = torch.load(one_second_dir / f"{row['level_id']}.pt",
                              map_location="cpu", weights_only=True)["features"].float()
        one_second_means.append(features.mean(0))
    one_second_representations = torch.nn.functional.normalize(torch.stack(one_second_means), dim=-1)
    one_second_similarity = (
        one_second_representations[valid_idx] @ one_second_representations[train_idx].T
    )
    one_second_knn = knn_predict(one_second_similarity, all_targets[train_idx], 80, 0.01)
    level_ids = torch.tensor([math.log10(float(row["level_id"])) for row in all_rows])
    id_prior = level_id_predict(level_ids[valid_idx], level_ids[train_idx],
                                all_targets[train_idx], bandwidth=0.1)
    current_best_distribution = 0.75 * fixed_video_knn + 0.25 * torch.load(
        Path(__file__).resolve().parent / "dinov2_1s_batched_transformer_within3_w05.pt",
        map_location="cpu", weights_only=True,
    )["validation_probabilities"].float()
    one_second_scores = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        for retrieval_weight in torch.linspace(0, 0.5, 11):
            blend = ((1 - retrieval_weight) * current_best_distribution
                     + retrieval_weight * one_second_knn)
            prediction = window_mass_center(blend, 3)
            mae, within3 = score(prediction[tune], actual[tune])
            candidate = (within3, -mae, float(retrieval_weight))
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        retrieval_weight = best[2]
        blend = ((1 - retrieval_weight) * current_best_distribution
                 + retrieval_weight * one_second_knn)
        result = score(window_mass_center(blend, 3)[held], actual[held])
        one_second_scores.append(result)
        print(f"fold {fold + 1} extra one-second KNN: weight={retrieval_weight:.2f}, "
              f"MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("extra one-second KNN five-fold: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in one_second_scores) / 5,
        sum(x[1] for x in one_second_scores) / 5,
    ))
    for retrieval_weight in (0.1, 0.2, 0.3):
        blend = ((1 - retrieval_weight) * current_best_distribution
                 + retrieval_weight * one_second_knn)
        print(f"extra one-second KNN full validation at {retrieval_weight:.1f}:",
              score(window_mass_center(blend, 3), actual))
    id_scores = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        for id_weight in torch.linspace(0, 0.5, 11):
            blend = (1 - id_weight) * current_best_distribution + id_weight * id_prior
            prediction = window_mass_center(blend, 3)
            mae, within3 = score(prediction[tune], actual[tune])
            candidate = (within3, -mae, float(id_weight))
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        id_weight = best[2]
        blend = (1 - id_weight) * current_best_distribution + id_weight * id_prior
        result = score(window_mass_center(blend, 3)[held], actual[held])
        id_scores.append(result)
        print(f"fold {fold + 1} level-ID prior blend: weight={id_weight:.2f}, "
              f"MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("level-ID prior five-fold blend: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in id_scores) / 5,
        sum(x[1] for x in id_scores) / 5,
    ))
    name_vectors = vectorize_names([row.get("name", "") for row in all_rows])
    name_similarity = name_vectors[valid_idx] @ name_vectors[train_idx].T
    name_distribution = name_knn_predict(
        name_similarity, all_targets[train_idx], k=80, temperature=0.24
    )
    name_scores = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        for name_weight in torch.linspace(0, 0.5, 11):
            blend = (1 - name_weight) * fixed_video_knn + name_weight * name_distribution
            prediction = window_mass_center(blend, 3)
            mae, within3 = score(prediction[tune], actual[tune])
            candidate = (within3, -mae, float(name_weight))
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        name_weight = best[2]
        blend = (1 - name_weight) * fixed_video_knn + name_weight * name_distribution
        result = score(window_mass_center(blend, 3)[held], actual[held])
        name_scores.append(result)
        print(f"fold {fold + 1} title KNN blend: weight={name_weight:.2f}, "
              f"MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("title KNN five-fold blend: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in name_scores) / 5,
        sum(x[1] for x in name_scores) / 5,
    ))
    experiment_dir = Path(__file__).resolve().parent
    temporal_a = torch.load(experiment_dir / "dinov2_1s_batched_transformer.pt",
                            map_location="cpu", weights_only=True)["validation_probabilities"].float()
    temporal_b = torch.load(experiment_dir / "dinov2_1s_batched_transformer_seed7.pt",
                            map_location="cpu", weights_only=True)["validation_probabilities"].float()
    temporal_within3 = torch.load(
        experiment_dir / "dinov2_1s_batched_transformer_within3.pt",
        map_location="cpu", weights_only=True,
    )["validation_probabilities"].float()
    temporal_within3_w05 = torch.load(
        experiment_dir / "dinov2_1s_batched_transformer_within3_w05.pt",
        map_location="cpu", weights_only=True,
    )["validation_probabilities"].float()
    temporal_within3_w10 = torch.load(
        experiment_dir / "dinov2_1s_batched_transformer_within3_w10.pt",
        map_location="cpu", weights_only=True,
    )["validation_probabilities"].float()
    temporal_focal = torch.load(
        experiment_dir / "dinov2_1s_batched_transformer_within3_focal2.pt",
        map_location="cpu", weights_only=True,
    )["validation_probabilities"].float()
    temporal_rps = torch.load(
        experiment_dir / "dinov2_1s_batched_transformer_rps_w05.pt",
        map_location="cpu", weights_only=True,
    )["validation_probabilities"].float()
    fused_videomae_dino = torch.load(
        experiment_dir / "dino_videomae_fused_temporal_within3_w05.pt",
        map_location="cpu", weights_only=True,
    )["validation_probabilities"].float()
    delta_features = torch.load(
        experiment_dir / "dinov2_1s_temporal_deltas_within3_w05.pt",
        map_location="cpu", weights_only=True,
    )["validation_probabilities"].float()
    attention_mean_max = torch.load(
        experiment_dir / "dinov2_1s_transformer_attn_mean_max_w05.pt",
        map_location="cpu", weights_only=True,
    )["validation_probabilities"].float()
    tier_weighted = torch.load(
        experiment_dir / "dinov2_1s_transformer_tierweight025.pt",
        map_location="cpu", weights_only=True,
    )["validation_probabilities"].float()
    tta_checkpoint = torch.load(
        experiment_dir / "dinov2_1s_batched_transformer_within3_w05_tta.pt",
        map_location="cpu", weights_only=True,
    )
    swa_distribution = torch.load(
        experiment_dir / "dinov2_1s_batched_transformer_within3_w05_swa.pt",
        map_location="cpu", weights_only=True,
    )["validation_probabilities"].float()
    gru_within3 = torch.load(
        experiment_dir / "dinov2_1s_batched_gru_within3.pt",
        map_location="cpu", weights_only=True,
    )["validation_probabilities"].float()
    tcn_within3 = torch.load(
        experiment_dir / "dinov2_1s_batched_tcn_within3.pt",
        map_location="cpu", weights_only=True,
    )["validation_probabilities"].float()
    temporal_candidates = {
        "seed 42": temporal_a,
        "seed 7": temporal_b,
        "two-seed average": (temporal_a + temporal_b) / 2,
        "within-three loss": temporal_within3,
        "within-three loss weight 0.5": temporal_within3_w05,
        "within-three loss weight 1.0": temporal_within3_w10,
        "focal within-three loss": temporal_focal,
        "DINO-VideoMAE fused": fused_videomae_dino,
        "DINO temporal deltas": delta_features,
        "attention+mean+max pooling": attention_mean_max,
        "tier-density weighted loss": tier_weighted,
        "reverse test-time average": tta_checkpoint["reverse_average"].float(),
        "MC-dropout test-time average": tta_checkpoint["mc_dropout_average"].float(),
        "SWA temporal model": swa_distribution,
        "uniform 2-model Transformer+GRU": (temporal_within3_w05 + gru_within3) / 2,
        "uniform 3-architecture average": (
            temporal_within3_w05 + gru_within3 + fused_videomae_dino
        ) / 3,
        "uniform ordinal-loss variants": torch.stack((
            temporal_a, temporal_within3, temporal_within3_w05,
            temporal_within3_w10, temporal_focal, temporal_rps,
        )).mean(0),
        "bidirectional GRU": gru_within3,
        "temporal convolution": tcn_within3,
    }
    for model_name, temporal_distribution in temporal_candidates.items():
        full_blend = 0.8 * fixed_video_knn + 0.2 * temporal_distribution
        full_result = score(window_mass_center(full_blend, 3), actual)
        full_blend_25 = 0.75 * fixed_video_knn + 0.25 * temporal_distribution
        full_result_25 = score(window_mass_center(full_blend_25, 3), actual)
        full_blend_30 = 0.7 * fixed_video_knn + 0.3 * temporal_distribution
        full_result_30 = score(window_mass_center(full_blend_30, 3), actual)
        temporal_scores = []
        for fold in range(5):
            tune, held = fold_ids != fold, fold_ids == fold
            best = None
            for temporal_weight in torch.linspace(0, 0.5, 11):
                blend = ((1 - temporal_weight) * fixed_video_knn
                         + temporal_weight * temporal_distribution)
                prediction = window_mass_center(blend, 3)
                mae, within3 = score(prediction[tune], actual[tune])
                candidate = (within3, -mae, float(temporal_weight))
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            temporal_weight = best[2]
            blend = ((1 - temporal_weight) * fixed_video_knn
                     + temporal_weight * temporal_distribution)
            result = score(window_mass_center(blend, 3)[held], actual[held])
            temporal_scores.append(result)
            print(f"fold {fold + 1} {model_name} blend: weight={temporal_weight:.2f}, "
                  f"MAE={result[0]:.4f}, within3={result[1]:.4f}")
        print(f"{model_name}: full_validation_20pct_MAE={full_result[0]:.4f}, "
              f"within3={full_result[1]:.4f}; full_validation_25pct_MAE="
              f"{full_result_25[0]:.4f}, within3={full_result_25[1]:.4f}; "
              f"full_validation_30pct_MAE="
              f"{full_result_30[0]:.4f}, within3={full_result_30[1]:.4f}; five-fold MAE="
              f"{sum(x[0] for x in temporal_scores) / 5:.4f}, within3="
              f"{sum(x[1] for x in temporal_scores) / 5:.4f}")

    # Allow separate, fold-local weights for the two differently initialized
    # temporal models to test whether their errors are complementary.
    seed_blend_scores = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        for seed42_weight in torch.arange(0, 0.501, 0.05):
            for seed7_weight in torch.arange(0, 0.501 - seed42_weight + 1e-6, 0.05):
                blend = ((1 - seed42_weight - seed7_weight) * fixed_video_knn
                         + seed42_weight * temporal_a + seed7_weight * temporal_b)
                prediction = window_mass_center(blend, 3)
                mae, within3 = score(prediction[tune], actual[tune])
                candidate = (within3, -mae, float(seed42_weight), float(seed7_weight))
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        weight42, weight7 = best[2], best[3]
        blend = ((1 - weight42 - weight7) * fixed_video_knn
                 + weight42 * temporal_a + weight7 * temporal_b)
        result = score(window_mass_center(blend, 3)[held], actual[held])
        seed_blend_scores.append(result)
        print(f"fold {fold + 1} separate temporal seed weights: "
              f"seed42={weight42:.2f}, seed7={weight7:.2f}, "
              f"MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("two-seed separately weighted five-fold: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in seed_blend_scores) / 5,
        sum(x[1] for x in seed_blend_scores) / 5,
    ))
    complementary_scores = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        for seed42_weight in torch.arange(0, 0.501, 0.05):
            for within3_weight in torch.arange(0, 0.501 - seed42_weight + 1e-6, 0.05):
                blend = ((1 - seed42_weight - within3_weight) * fixed_video_knn
                         + seed42_weight * temporal_a + within3_weight * temporal_within3)
                prediction = window_mass_center(blend, 3)
                mae, within3 = score(prediction[tune], actual[tune])
                candidate = (within3, -mae, float(seed42_weight), float(within3_weight))
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        weight42, weight_w3 = best[2], best[3]
        blend = ((1 - weight42 - weight_w3) * fixed_video_knn
                 + weight42 * temporal_a + weight_w3 * temporal_within3)
        result = score(window_mass_center(blend, 3)[held], actual[held])
        complementary_scores.append(result)
        print(f"fold {fold + 1} seed42/within3 weights: seed42={weight42:.2f}, "
              f"within3={weight_w3:.2f}, MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("seed42/within-three complementary five-fold: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in complementary_scores) / 5,
        sum(x[1] for x in complementary_scores) / 5,
    ))
    fixed_complementary = (0.65 * fixed_video_knn + 0.15 * temporal_a + 0.20 * temporal_within3)
    print("fixed 65/15/20 video-KNN/seed42/within3 full validation:",
          score(window_mass_center(fixed_complementary, 3), actual))
    gru_complementary_scores = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        for gru_weight in torch.arange(0, 0.501, 0.05):
            for within3_weight in torch.arange(0, 0.501 - gru_weight + 1e-6, 0.05):
                blend = ((1 - gru_weight - within3_weight) * fixed_video_knn
                         + gru_weight * gru_within3 + within3_weight * temporal_within3)
                prediction = window_mass_center(blend, 3)
                mae, within3 = score(prediction[tune], actual[tune])
                candidate = (within3, -mae, float(gru_weight), float(within3_weight))
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        gru_weight, within3_weight = best[2], best[3]
        blend = ((1 - gru_weight - within3_weight) * fixed_video_knn
                 + gru_weight * gru_within3 + within3_weight * temporal_within3)
        result = score(window_mass_center(blend, 3)[held], actual[held])
        gru_complementary_scores.append(result)
        print(f"fold {fold + 1} GRU/Transformer weights: GRU={gru_weight:.2f}, "
              f"Transformer={within3_weight:.2f}, MAE={result[0]:.4f}, "
              f"within3={result[1]:.4f}")
    print("GRU/Transformer complementary five-fold: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in gru_complementary_scores) / 5,
        sum(x[1] for x in gru_complementary_scores) / 5,
    ))
    fixed_gru_complementary = (0.6 * fixed_video_knn + 0.2 * gru_within3
                               + 0.2 * temporal_within3)
    print("fixed 60/20/20 video-KNN/GRU/Transformer full validation:",
          score(window_mass_center(fixed_gru_complementary, 3), actual))
    fixed_seed_blend = (0.8 * fixed_video_knn + 0.15 * temporal_a + 0.05 * temporal_b)
    fixed_seed_result = score(window_mass_center(fixed_seed_blend, 3), actual)
    print("fixed 80/15/5 video-KNN/temporal-seed blend on full validation:", fixed_seed_result)

    # Calibrate the final point readout for the strongest fixed distribution,
    # fitting radius/offset only on the other four validation folds.
    best_distribution = 0.75 * fixed_video_knn + 0.25 * temporal_within3_w05
    best_mean, best_median = tier_values(best_distribution)
    readout_scores = []
    selected_readouts = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        readouts = {"mean": best_mean, "median": best_median}
        for radius in (1, 2, 3, 4):
            readouts[f"window{radius}"] = window_mass_center(best_distribution, radius)
        for label, prediction in readouts.items():
            for offset in torch.arange(-1.0, 1.001, 0.25):
                mae, within3 = score((prediction[tune] + offset).clamp(1, NUM_CLASSES), actual[tune])
                candidate = (within3, -mae, label, float(offset))
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        label, offset = best[2], best[3]
        prediction = (readouts[label] + offset).clamp(1, NUM_CLASSES)
        result = score(prediction[held], actual[held])
        readout_scores.append(result)
        selected_readouts.append((label, offset))
        print(f"fold {fold + 1} best-distribution readout: {label}, offset={offset:+.2f}, "
              f"MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("best-distribution tuned readout five-fold: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in readout_scores) / 5,
        sum(x[1] for x in readout_scores) / 5,
    ))
    readout_counts = {}
    for selected in selected_readouts:
        readout_counts[selected] = readout_counts.get(selected, 0) + 1
    common_readout = max(readout_counts, key=readout_counts.get)
    common_prediction = (readouts[common_readout[0]] + common_readout[1]).clamp(1, NUM_CLASSES)
    print(f"most common tuned readout {common_readout} full validation:",
          score(common_prediction, actual))

    # Jointly tune a simple weighted average over each pretrained-model group,
    # global retrieval, and the strongest temporal head.
    ensemble_parts = torch.cat((matrices, knn_distribution[None],
                                temporal_within3_w05[None]), dim=0)
    ensemble_names = (*names, "DINO KNN", "temporal within3")
    equal_distribution = ensemble_parts.mean(0)
    equal_full_score = score(window_mass_center(equal_distribution, 3), actual)
    equal_fold_scores = [
        score(window_mass_center(equal_distribution, 3)[fold_ids == fold],
              actual[fold_ids == fold])
        for fold in range(5)
    ]
    print("literal equal five-source average: full validation MAE={:.4f}, within3={:.4f}; "
          "five-fold MAE={:.4f}, within3={:.4f}".format(
              equal_full_score[0], equal_full_score[1],
              sum(x[0] for x in equal_fold_scores) / 5,
              sum(x[1] for x in equal_fold_scores) / 5,
          ))
    ensemble_scores = []
    for fold in range(5):
        tune, held = fold_ids != fold, fold_ids == fold
        best = None
        for weights in simplex_weights(len(ensemble_names), 0.1):
            distribution = torch.einsum("p,pvc->vc", torch.tensor(weights), ensemble_parts)
            prediction = window_mass_center(distribution, 3)
            mae, within3 = score(prediction[tune], actual[tune])
            candidate = (within3, -mae, weights)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        weights = best[2]
        distribution = torch.einsum("p,pvc->vc", torch.tensor(weights), ensemble_parts)
        result = score(window_mass_center(distribution, 3)[held], actual[held])
        ensemble_scores.append(result)
        print(f"fold {fold + 1} joint distribution average: "
              f"weights={dict(zip(ensemble_names, weights))}, "
              f"MAE={result[0]:.4f}, within3={result[1]:.4f}")
    print("five-fold joint five-source distribution average: MAE={:.4f}, within3={:.4f}".format(
        sum(x[0] for x in ensemble_scores) / 5,
        sum(x[1] for x in ensemble_scores) / 5,
    ))


if __name__ == "__main__":
    main()
