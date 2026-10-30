# Repeated random-split comparison

Three methods were compared on the same three randomized train/validation
splits (seeds 13, 29, and 47). Each split used 4,848 training levels and 607
validation levels. The original seed-42 test IDs were excluded from every
split and were not evaluated. All inputs were precomputed visual DINOv2-base
features; demon-rating metadata was not used.

The temporal Transformer used the one-second feature cache, the established
within-three loss weight of 0.5, and a fixed 10-epoch training schedule. The
schedule came from the earlier pilot and was not selected using these folds'
validation scores. The two retrieval methods used the four-second cache and
selected neighbor count, temperature, and learned-metric blend using an inner
holdout drawn only from that split's training partition. The learned metric
was refit on the full training partition after selecting its epoch and
retrieval settings.

As a diagnostic, temporal models were first allowed to early-stop on each
scored validation fold. They averaged 57.33% within three (57.50%, 56.18%,
58.32%), but this uses the validation labels to choose the checkpoint and is
optimistically biased. Those checkpoints and logs are retained for comparison;
the fixed-epoch results below are the temporal scores used for ranking.

| Split seed | 1s temporal Transformer | 4s cosine KNN | 4s learned-metric KNN |
| ---: | ---: | ---: | ---: |
| 13 | 53.87% / 4.155 | 52.22% / 4.862 | 55.19% / 4.389 |
| 29 | 54.20% / 4.229 | 49.26% / 4.869 | 55.85% / 4.073 |
| 47 | 55.68% / 4.085 | 55.19% / 4.638 | 58.48% / 4.052 |
| **Mean** | **54.59% / 4.156** | **52.22% / 4.789** | **56.51% / 4.171** |

Cells show within-3-tier accuracy / MAE. Across these splits, learned-metric KNN
had the highest average within-3 accuracy, by 1.92 percentage points over the
fixed-epoch temporal Transformer. The temporal model's average MAE was lower
by 0.014 tier, effectively a tie at this sample size. Plain cosine KNN was
weaker on both measures. Learned-metric KNN selected a 100% learned-similarity
weight in two splits and 50% in the third.

## Fixed distribution blends

The learned-metric KNN and temporal probabilities were also combined with
fixed weights. The 75% KNN + 25% temporal blend beat either model alone on
both measures in all three splits:

| Split seed | 75% metric KNN + 25% temporal | 50% metric KNN + 50% temporal | Equal average of all three |
| ---: | ---: | ---: | ---: |
| 13 | 55.85% / 4.068 | 56.84% / 4.098 | 56.84% / 4.081 |
| 29 | 57.17% / 4.024 | 56.01% / 4.072 | 55.35% / 4.065 |
| 47 | 59.97% / 3.956 | 58.98% / 3.968 | 57.99% / 4.021 |
| **Mean** | **57.66% / 4.016** | **57.28% / 4.046** | **56.73% / 4.056** |

Cells show within-3-tier accuracy / MAE using the same highest-mass +/-3
readout. The 75/25 weights were fixed before scoring these splits, following
the earlier ensemble's 75/25 blend ratio; the old video-group component was
replaced with the split-specific learned-metric KNN. The blend improves mean
within-three accuracy by 1.15 percentage points and mean MAE by 0.155 tier
versus learned-metric KNN alone. It is the best of the evaluated split-safe
methods here, though its three-split mean MAE remains just above 4.

It is decisively above random guessing on this metric. Uniformly guessing one
of 39 tiers gives about 14.8% expected within-three accuracy on these
validation targets. A stronger random baseline that samples from each split's
training marginal tier distribution gives 27.3–28.6%. The ensemble scored
55.8–60.0%; its observed success count was 15.4–17.9 standard deviations
above the training-marginal random baseline in each split (and 28.6–31.4 SD
above uniform guessing). This tests the random-guess null, not significance
against other trained models.

The exact earlier ensemble that combined VideoMAE, DINO-small, DINO-base,
KNN, and temporal predictions was not re-evaluated: its saved visual heads
were fit or early-stopped on the original split. Reusing them would leak
information into these randomized validation folds. Thus the result here is
the best split-safe ensemble tested so far, not a new evaluation of every
historical ensemble recipe.

These are three repeated holdouts from the same dataset, so their results are
not independent. The temporal epoch count and loss setting were established
in an earlier seed-42 pilot; that pilot's validation set overlaps the pool
used to make these randomized splits. The fixed epoch count prevents each new
validation fold from selecting its own checkpoint, but a larger fresh dataset
or nested cross-validation would give a cleaner estimate. The strongest
previous result was a multi-model ensemble; its existing component
checkpoints were trained or early-stopped on the original fixed split, so
they were not reused here because some randomized validation rows could have
influenced those checkpoints.

Checkpoints, logs, and their validation predictions are saved in this
directory. Re-run commands from the repository root with the project venv:

```bash
training/.venv/bin/python -m training.experiments.train_dinov2_temporal_gpu \
  --split-seed 13 --seed 42 --batch-size 1 --fixed-epochs 10 \
  --within-three-loss-weight 0.5 \
  --output training/experiments/repeated_splits/temporal_fixed10_seed13.pt

PYTHONPATH=.:training/experiments training/.venv/bin/python \
  -m training.experiments.train_dinov2_metric_knn \
  --split-seed 13 --seed 42 \
  --feature-dir training/features_dinov2_base_4fps \
  --output training/experiments/repeated_splits/metric_knn_seed13.pt

training/.venv/bin/python -m training.experiments.repeated_splits.evaluate_ensembles

MPLCONFIGDIR=/tmp/gd-difficulty-mpl PYTHONPATH=.:training/experiments \
  training/.venv/bin/python -m training.experiments.repeated_splits.plot_split_predictions
```

## Expanded fit using every non-validation level

After the repeated-split comparison, split 13's 607 validation IDs were kept
fixed and both models were retrained on every other Long level. This gives
5,455 training levels plus 607 validation levels, covering all 6,062 records.
All 607 levels formerly held out as the seed-42 test set are now in training;
none overlap validation. Their IDs and names are in
`former_seed42_test_ids_now_train.csv`.

The temporal Transformer monitored validation within-three accuracy and
stopped after 27 epochs, keeping the best checkpoint from epoch 18. The
learned-metric KNN selected its epoch and retrieval settings on an inner
training-only holdout. Their fixed 75/25 blend scored 58.65% within three
(MAE 3.938) on validation. Since temporal early stopping used that validation
set, this score is optimistic. The chart's training points are in-sample;
the validation points use held-out predictions. Training KNN predictions
exclude each level itself from its neighbor set.

The complete 6,062-level chart is
`all_nonval_train_validation_actual_vs_predicted.png`. The validation IDs
used for the refit are in `split13_validation_ids.txt`. The chart now reads
the ensemble distribution as its expected tier (`sum(tier * probability)`),
so y values are floating point; this readout scores 55.52% within three and
MAE 3.657 on validation. The earlier 58.65% / 3.938 result uses the
highest-mass +/-3 window point readout.
