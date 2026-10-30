"""Score fixed distribution blends from the repeated-split checkpoints."""
import statistics
from pathlib import Path

import torch

from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).resolve().parent
SPLIT_SEEDS = (13, 29, 47)


def window_mass_center(probabilities, radius=3):
    cumulative = torch.cat((
        torch.zeros(len(probabilities), 1, dtype=probabilities.dtype),
        probabilities.cumsum(-1),
    ), dim=-1)
    centers = torch.arange(probabilities.shape[-1])
    lower = (centers - radius).clamp_min(0)
    upper = (centers + radius + 1).clamp_max(probabilities.shape[-1])
    return (cumulative[:, upper] - cumulative[:, lower]).argmax(-1).float() + 1


def score(probabilities, targets):
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=probabilities.dtype)
    actual = targets @ tiers
    prediction = window_mass_center(probabilities)
    errors = (prediction - actual).abs()
    return (float((errors <= 3).float().mean()), float(errors.mean()))


def random_guess_chance(targets, training_targets, observed_accuracy):
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=targets.dtype)
    actual = targets @ tiers
    lower = (actual - 3).ceil().long().clamp(1, NUM_CLASSES) - 1
    upper = (actual + 3).floor().long().clamp(1, NUM_CLASSES) - 1
    prior = training_targets.mean(0)
    cumulative_prior = torch.cat((torch.zeros(1, dtype=prior.dtype), prior.cumsum(0)))
    prior_chance = cumulative_prior[upper + 1] - cumulative_prior[lower]
    uniform_chance = (upper - lower + 1).float() / NUM_CLASSES
    result = {}
    successes = observed_accuracy * len(targets)
    for name, chances in (("uniform-tier", uniform_chance),
                          ("training-marginal", prior_chance)):
        expected = chances.sum().item()
        variance = (chances * (1 - chances)).sum().item()
        z_score = (successes - expected) / variance ** 0.5
        result[name] = (float(chances.mean()), z_score)
    return result


def main():
    targets_by_id = {row["level_id"]: row["target"] for row in read_rows(DEFAULT_CSV)}
    scores = {}
    for seed in SPLIT_SEEDS:
        metric = torch.load(RESULTS / f"metric_knn_seed{seed}.pt",
                            map_location="cpu", weights_only=True)
        temporal = torch.load(RESULTS / f"temporal_fixed10_seed{seed}.pt",
                              map_location="cpu", weights_only=True)
        assert metric["validation_level_ids"] == temporal["validation_level_ids"]
        ids = metric["validation_level_ids"]
        targets = torch.stack([targets_by_id[level_id] for level_id in ids])
        training_targets = torch.stack(
            [targets_by_id[level_id] for level_id in metric["training_level_ids"]]
        )
        metric_probs = metric["validation_probabilities"]
        raw_probs = metric["raw_knn_validation_probabilities"]
        temporal_probs = temporal["validation_probabilities"]
        methods = {
            "learned_metric_knn": metric_probs,
            "cosine_knn": raw_probs,
            "temporal_transformer": temporal_probs,
            "metric75_temporal25": 0.75 * metric_probs + 0.25 * temporal_probs,
            "metric50_temporal50": 0.50 * metric_probs + 0.50 * temporal_probs,
            "metric75_cosine25": 0.75 * metric_probs + 0.25 * raw_probs,
            "equal_three_way": (metric_probs + raw_probs + temporal_probs) / 3,
        }
        for name, probabilities in methods.items():
            scores.setdefault(name, []).append(score(probabilities, targets))
        print(f"split={seed}")
        for name, probabilities in methods.items():
            accuracy, mae = score(probabilities, targets)
            print(f"  {name}: within3={accuracy:.4%} MAE={mae:.4f}")
        accuracy, _ = score(methods["metric75_temporal25"], targets)
        chance = random_guess_chance(targets, training_targets, accuracy)
        for name, (expected, z_score) in chance.items():
            print(f"  {name} random baseline: expected_within3={expected:.4%}, "
                  f"observed_z={z_score:.2f}")

    print("mean and split standard deviation:")
    for name, values in scores.items():
        mean_accuracy = statistics.mean(value[0] for value in values)
        mean_mae = statistics.mean(value[1] for value in values)
        sd_accuracy = statistics.stdev(value[0] for value in values)
        sd_mae = statistics.stdev(value[1] for value in values)
        print(f"  {name}: within3={mean_accuracy:.4%} +/- {sd_accuracy:.4%}; "
              f"MAE={mean_mae:.4f} +/- {sd_mae:.4f}")


if __name__ == "__main__":
    main()
