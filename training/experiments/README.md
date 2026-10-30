# Training experiments

Generated checkpoints and plots are kept under `archive/` instead of at the
top level of `training/`. The default run remains `training/ordinal_transformer.pt`
with its plot at `training/loss_curve_ordinal.png`.

## Quick index

| Area | Entry points |
| --- | --- |
| Best results and experiment history | This file; see [repeated split results](repeated_splits/README.md) for split-safe comparisons. |
| Core scoring and ensembles | `evaluate_validation.py`, `cross_validate_ensemble.py`, `evaluate_temporal_seed_blend.py`, `evaluate_temporal_feature_blend.py` |
| Retrieval and feature comparisons | `knn_dinov2.py`, `knn_dinov2_multirate.py`, `knn_temporal_pyramid.py`, `compare_feature_knn.py`, `ridge_dinov2.py` |
| Temporal training | `train_dinov2_temporal_gpu.py` supports cached DINO feature directories; `train_dinov2_metric_knn.py` is in `repeated_splits/` workflows. |
| Alternative losses and heads | `train_dinov2_coral.py`, `train_dinov2_mixup.py`, `train_dinov2_statistics.py`, `train_dinov2_tolerance_loss.py`, `train_gaussian_target_dinov2.py`, `train_pooled_dinov2.py` |
| Ensemble calibration | `ensemble_calibration_experiments.py`, `calibrate_ensemble_distribution.py`, `tierwise_source_weights.py` |
| Level-file experiments | `evaluate_level_object_features.py`, `evaluate_level_sequence_features.py`, `train_level_object_transformer.py`, `evaluate_object_video_fusion.py` |
| Data acquisition and feature extraction | `download_*.py`; `extract_clip_features.py` |
| Archived artifacts | `archive/` and model checkpoints are local-only and ignored by Git. |

Run Python commands from the repository root with `training/.venv/bin/python`.
Generated video, feature, model, cache, and build files are excluded by the
root `.gitignore`; source scripts, experiment notes, and compact result tables
remain versioned.

## Best results so far

All runs use the same 6,062-level dataset and deterministic 80/10/10 split
(seed 42). MAE is the error between predicted and vote-weighted mean tier.

| Experiment | Validation MAE | Validation exact | Validation within 3 | Held-out MAE | Held-out within 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three-model VideoMAE probability ensemble | 4.36 | 15.3% | 50.1% | 4.02 | 54.5% |
| Sparse DINOv2-base model (one frame/8 sec) | 4.12 | 14.3% | 53.4% | 3.55 | 57.0% |
| 70% sparse DINOv2 + 30% VideoMAE ensemble | 4.06 | 17.0% | 53.4% | 3.52 | 56.2% |
| 4 fps DINOv2-small, ordinal CE | 4.12 | 13.0% | 53.9% | 3.44 | 60.0% |
| 70% two-seed 4 fps DINOv2-small + 30% VideoMAE ensemble | 3.95 | 14.5% | 53.9% | 3.39 | 55.2% |
| 4 fps DINOv2-base, categorical CE (best seed) | 3.82 | 14.3% | 56.2% | — | — |
| 4 fps DINOv2-base, ordinal CE, one-second tokens | 4.12 | 13.5% | 51.1% | 3.56 | 56.3% |
| DINOv2-base, four-bin pooled MLP, ordinal CE (median readout) | 3.99 | 14.3% | 56.2% | — | — |
| DINOv2-base, four-bin pooled MLP, Wasserstein loss (median readout) | 4.22 | 9.9% | 56.3% | — | — |
| Validation-tuned group blend (10% VideoMAE, 20% small DINO, 70% base DINO) | 3.76 | 16.8% | 56.2% | — | — |
| Same blend, median-tier readout | 3.82 | 11.7% | 58.8% | — | — |
| Fixed 90/10 group ensemble + KNN, highest-mass +/-3 window readout | 4.07 | — | 61.4% | — | — |
| Fixed best video/KNN/temporal blend + 5% seed-42 learned-metric KNN | 3.82 | — | 64.25% | — | — |

The DINOv2-base result is a sparse-sampling experiment and is not directly
comparable to the VideoMAE cache. The DINOv2-small comparison samples the full
video at 4 fps and averages each 16-frame group into a four-second feature,
matching the VideoMAE clip interval. The cache has 6,062 feature files. Its
transformer has 910,376 trainable parameters, uses dropout 0.4, and predicts a
39-tier distribution with ordinal cross entropy. The two-seed blend weight was
selected on validation data; held-out metrics are reported for that selection.

The default trainer reads cached features and supports a configurable input
dimension. For the completed DINOv2-small extraction, train with
`--feature-model facebook/dinov2-small --input-feature-size 384` and
`--feature-dir training/features_dinov2_small_4fps`.

The later DINOv2-base runs use full-video 4 fps features. Four sampled frame
embeddings are averaged per token: once per four seconds for the base models,
and once per second for the one-second experiment. The one-second sequence did
not improve standalone validation accuracy. The validation-tuned group blend
averages the saved VideoMAE models, small-DINO models, and four-second base-DINO
models within each group. Its weights are selected on the validation split, so
the listed blend scores are validation estimates and may be optimistic. The
median-tier readout improves within-three accuracy at the cost of exact-tier
accuracy. No held-out evaluation has been run for these latest blends.

To check whether those gains survive weight selection, the group ensemble and
point readout were also evaluated with five folds within the validation split.
The equal-weight group ensemble scored 55.5% within three using its expected
tier and 58.0% using its median tier. Tuning group weights and the mean/median
blend on the other four folds scored 55.9% on the held-out fold. This did not
beat the fixed group-mean median readout, so the full-split best blend should
not be treated as a reliable improvement. The reproducible CPU-capable
evaluation is `cross_validate_ensemble.py`.

A CPU ridge ordinal model over mean-pooled DINOv2-base features was also fit on
the training split, with pooling and regularization selected on an internal
training holdout. It reached 52.4% within three on validation. Allowing its
distribution to blend with the fixed group ensemble produced 57.7% in the
five-fold validation check, below the 58.0% fixed group-mean median readout.
This linear model did not add useful complementary signal in this setup; see
`ridge_dinov2.py` and `cross_validate_ensemble.py` for the runs.

A cosine KNN model over temporally mean-pooled DINOv2-base features selected
20 neighbors and temperature 0.02 on a training-only holdout, then scored 49.1%
within three on validation using its median readout. On validation five-fold
checks, mixing 10% of its predicted distribution into the equal-weight
VideoMAE/DINO group ensemble increased median-readout within-three accuracy
from 58.0% to 58.7%; the full validation score for that fixed 90/10 blend was
58.6%. This small gain is promising but below the 58.8% full-validation result
from the median readout of the validation-tuned group blend. The retrieval
baseline is in `knn_dinov2.py`.

The within-three metric can use a different readout from the MAE-optimal
median. For the fixed 90% group ensemble + 10% global-pooling KNN distribution,
selecting the tier whose +/-3 window contains the greatest predicted
probability mass scored 61.45% within three on validation and 61.45% in the
five-fold check. The median readout scored 58.65% within three in the same
check. This is the strongest video-only within-three result so far; MAE rises
from 3.87 to 4.07, so use the median when average absolute error matters more.
The window-mass calculation and cross-validation are in
`cross_validate_ensemble.py`.

Nested fold tuning of the three group weights, KNN weight (0–50%), and readout
scored 61.62% within three, only 0.17 points above the fixed blend. Selected
weights varied somewhat across folds. A separately tuned probability-sharpness
exponent reached 60.63%, so sharpening did not help. The fixed 90/10 blend with
the metric-matched window readout remains the simpler stable choice. Its MAE is
still above 3, so the average-error target has not been met.

A KNN using the higher-rate one-second DINO cache selected global mean pooling,
80 neighbors, and temperature 0.01 using a training-only holdout. Its own
validation score was 55.7% within three using the window readout. Five-fold
video/KNN fusion reached 61.6%, with fold-selected KNN weights of 30–45%; this
is effectively tied with the coarser four-second KNN fusion. The comparison is
in `knn_dinov2_multirate.py`.

A 756k-parameter, two-layer temporal Transformer was trained on the same
one-second DINOv2-base visual extractions with ordinal cross-entropy and a
small expected-tier loss term. It uses dropout 0.3 and early stopping. Its
standalone validation score was 59.6% within three (MAE 4.12). Blending its
distribution at 15–20% into the fixed video/KNN distribution, with the same
highest-mass +/-3 readout, reached 62.0% within three and MAE 4.03 in five
folds over validation; the full validation score at a fixed 20% blend was
62.3% and MAE 4.00. A second seed alone and the two-seed average were weaker,
but separate weights of 15% for seed 42 and 5% for seed 7 produced 62.1% within
three in fold-held-out validation and 62.8% (MAE 4.02) on the full validation
split. The fold-selected weights were 15%/5% in four of five folds. This is a
small gain over seed 42 alone, with slightly higher MAE. Stronger dropout
(0.5), mild loss weighting by training vote totals, and a nested ridge stacker
did not improve it. Comparisons are in `cross_validate_ensemble.py`; the CUDA
training command is `train_dinov2_temporal_gpu.py`. Inputs are cached visual
features only; difficulty vote data supplies training targets and evaluation
labels, and demon-rating metadata is not used. The held-out test partition
remains unevaluated.

A loss-focused temporal variant adds a small negative log-probability term for
the tiers within +/-3 of the vote-weighted target mean to the ordinal CDF and
expected-tier losses. Its standalone result was 59.0% within three (MAE 4.12).
A fixed 30% blend with the video/KNN distribution scored 62.8% on full
validation (MAE 3.99); selecting the blend weight within each of five
validation folds scored 62.4% (MAE 4.02). Raising this loss term's weight to
0.5 reduced standalone within-three accuracy slightly to 58.5%, but a fixed
25% blend scored 64.1% within three on full validation (MAE 3.84). The fold
comparison selected the same 25% weight in all five folds and scored 64.1%
(MAE 3.84). This is the best result so far. The fold comparison selects blend
weights independently, but the model checkpoint itself was early-stopped on
the full validation split, so the fold score may still be optimistic. The
stronger 1.0 loss weight dropped the five-fold blend to 62.0%. The loss variant
is trained by `train_dinov2_temporal_gpu.py` with
`--within-three-loss-weight 0.5` and evaluated by `cross_validate_ensemble.py`.

Adding a separate one-second DINO KNN to the current best model selected zero
weight in four of five folds and did not improve the 64.1% fold score. A fixed
10% KNN addition scored 63.6% on full validation. Concatenating aligned DINO
and VideoMAE features into a 903k-parameter temporal Transformer peaked at
57.2% standalone and 62.8% for its fold-selected blend, below the single-DINO
model. With batch size 16 it used 5.7 GiB of GPU memory. These comparisons are
also recorded in `cross_validate_ensemble.py`.

Two other temporal encoders were compared on the same one-second features and
loss. A 489k-parameter bidirectional GRU reached 58.3% within three standalone
and 62.3% in the five-fold video/KNN blend (MAE 3.99). A 766k-parameter
dilated temporal convolution reached 57.5% standalone and 61.1% blended. The
GRU reduced MAE slightly but did not beat the within-three Transformer; mixing
the two encoders also lowered within-three accuracy. GPU peaks were 3.1 GiB
for the GRU and 6.5 GiB for the convolutional model. Both checkpoints and
their comparisons are in `training/experiments/`.

Several refinements did not improve the current best: averaging reversed
sequences or 20 MC-dropout predictions reached 62.6% and 63.1% respectively
in five-fold blending; SWA weights reached 63.8%; an extra one-second KNN
selected zero weight in four of five folds. A log-level-ID prior was selected
at zero weight in four folds as well, with a 0.114 train-only correlation
between log ID and target tier. These results and code paths are in
`cross_validate_ensemble.py`, `evaluate_temporal_tta.py`, and
`level_id_prior.py`.

Focal weighting of the within-three loss (gamma 2) reached 58.0% standalone
and 60.6% in the fold-selected blend, below the unweighted 0.5 loss term.
Nested tuning of the final point readout (mean, median, +/-1 to +/-4 window,
and offsets from -1 to +1) selected the original +/-3 window with zero offset
in all five folds.

Distribution averaging was tested directly. A literal equal-weight average of
VideoMAE, DINO-small, DINO-base, DINO KNN, and the within-three Transformer
scored 63.4% on full validation and in five folds. Equal averaging of the best
Transformer and GRU, or of several ordinal-loss variants, scored 62.0% and
62.3% in folds; a uniform three-architecture average with fold-selected blend
weight scored 63.3%. These did not beat the 64.1% result from a weighted
average: 75% video/KNN distribution plus 25% within-three Transformer. Joint
fold tuning of all five source weights was unstable and scored 61.1%.

Two additional fold-held-out ensemble refinements did not improve that result.
A regularized per-example sigmoid gate based on each distribution's entropy,
peak probability, and expected-tier disagreement scored 61.95% within three
(MAE 3.85), below the fixed blend at 64.09% (MAE 3.84). A weighted logarithmic
opinion pool was mixed with the arithmetic ensemble; its blend and temperature
were selected within each training fold, then scored 63.93% within three
(MAE 3.83) across held-out folds. It was effectively ignored in one fold and
received only 10% weight in the others. Neither change is adopted. The CPU-only
comparison is reproducible in `ensemble_calibration_experiments.py`.

Cosine KNN retrieval was also tested separately on the cached VideoMAE,
DINO-small, four-second DINO-base, and one-second DINO-base embeddings. Neighbor
count and temperature were selected on an inner holdout from the fixed training
partition. VideoMAE and DINO-small retrieval received zero weight in every
validation fold. The one-second DINO KNN received 40% weight in only one fold;
the four-second KNN already in the baseline was likewise selected only once.
These fold-selected additions scored 62.4% and 62.1% within three respectively,
below the fixed 64.1% ensemble. See `compare_feature_knn.py`.

Two nested-CV distribution refinements also failed to improve the blend. A
smoothed per-tier logit calibration, fitted to vote distributions and tuned
inside each training fold, reached 63.60% within three (MAE 3.84), versus
64.09% (MAE 3.84) for the uncalibrated distribution. Allowing the five current
sources to use different weights by tier reached 63.76% within three (MAE
3.89); selected weights stayed close to their original values in four folds
and changed substantially in one. Neither method is adopted. Reproduce these
checks with `calibrate_ensemble_distribution.py` and `tierwise_source_weights.py`.

Filtering training levels by vote count did not help. With the full 607-level
validation set unchanged, minimum vote thresholds of 25, 50, and 100 retained
3,606, 2,496, and 1,392 training examples. Standalone within-three accuracy
was 55.5%, 54.5%, and 53.9%, respectively; each model received zero weight in
all five folds. Vote count was used only to filter training rows, never as a
model input. The experiment is reproducible with `--min-votes` in
`train_dinov2_temporal_gpu.py`.

To separate sample count from vote-based selection, nested random subsets used
the same 1-second DINO Transformer setup and seed with 25%, 50%, and 75% of the
training partition (1,212, 2,424, and 3,636 levels); the full-data model used
4,848. Standalone within-three accuracy was 56.0%, 58.6%, 57.7%, and 58.5%.
The smallest subset was weaker, but results plateaued around 50% rather than
improving monotonically. The 25%, 50%, and 75% models all received zero weight
beside the existing visual/temporal ensemble in five-fold validation. All 607
validation levels were scored in each run. Use `--train-fraction` in
`train_dinov2_temporal_gpu.py` to reproduce the nested sampling.

Temporal ordinal Transformers were trained on the cached four-second and
eight-second DINO sequences with the same 0.5 within-three loss. They reached
56.0% and 54.4% standalone, respectively, and fold-held-out weight selection
chose zero weight for both in every fold. A seed-7 one-second Transformer with
the same loss reached 58.3% standalone, but also received zero weight in all
five folds beside the seed-42 model. The feature-directory option and blend
checks are in `train_dinov2_temporal_gpu.py` and `evaluate_temporal_feature_blend.py`;
the seed comparison is in `evaluate_temporal_seed_blend.py`.

An additional pretrained CLIP ViT-B/16 encoder was evaluated at 4 fps,
averaging four frame embeddings into one full-video token per second while
preserving the whole widescreen frame. The completed cache contains all 5,455
training and validation levels; extraction had zero failures, and no held-out
test features were generated. A 756k-parameter one-second temporal Transformer
with ordinal loss and the 0.5 within-three term reached 57.5% standalone
within-three accuracy (MAE 4.27). Fold-held-out blending selected zero weight
in all five folds, leaving the 64.1% baseline unchanged. A cosine CLIP KNN,
with its neighbor count and temperature tuned using only an inner holdout from
the training partition, reached 54.4% standalone and 62.6% in fold-held-out
blending; it received 20% weight in one fold and zero in the other four. A
separate nested validation-only KNN probe scored 47.1% standalone and also did
not improve the baseline. CLIP did not add useful signal in these setups. The
experiments are reproducible with `extract_clip_features.py`,
`train_dinov2_temporal_gpu.py --feature-dir training/features_clip_vitb16_4fps_1s`,
`evaluate_temporal_feature_blend.py --checkpoint training/experiments/clip_vitb16_1s_temporal_within3_w05.pt`,
`compare_feature_knn.py`, and `evaluate_clip_fold_knn.py`.

Several follow-up label and regularization experiments also failed to improve
the current blend. Power-sharpening the one-second DINO training vote
distributions by 2 and 4 reached 57.7% and 58.2% standalone within-three
accuracy; the power-2 model received zero weight in every fold, while power 4
reduced the five-fold blend to 63.4% (7.5% candidate weight in two folds).
Increasing the expected-tier Smooth L1 loss weight from 0.2 to 1.0 reached
58.8% standalone but received zero blend weight in all folds. Weighting levels
by normalized vote-distribution entropy reached 58.3% standalone and likewise
received zero blend weight throughout. The trainer exposes these as
`--target-sharpening`, `--expected-loss-weight`, and
`--target-confidence-weight`.

Feature/target MixUp (alpha 0.2) on a four-bin pooled one-second DINO summary
reached 56.5% standalone; fold-selected fusion scored 63.8%, assigning 2.5%
weight in two folds and zero in the other three. The reproducible run is
`train_dinov2_mixup.py`.

The CLIP one-second features align token-for-token with the DINO one-second
cache for all 5,455 train/validation levels. A 903k-parameter temporal
Transformer trained on their concatenated features reached 58.6% standalone,
but its five-fold blend scored 63.1%; it received 45% weight in one fold and
zero in the other four. The additional stream is supported by
`--additional-feature-dir` in `train_dinov2_temporal_gpu.py`.

Increasing inverse-frequency weighting for the ten four-tier target bins was
also tested on the one-second DINO Transformer, using exponent 0.5, the same
0.5 within-three loss term, and the full training set. It reached 57.0%
standalone within-three accuracy (MAE 4.16); the candidate received zero blend
weight in all five validation folds. The earlier exponent 0.25 run reached
56.7% standalone, received 2.5% weight in one fold and zero in the other four,
and reduced the fold score to 63.8%. Tier-density weighting is not adopted; reproduce the stronger trial with
`train_dinov2_temporal_gpu.py --tier-density-weight 0.5 --within-three-loss-weight 0.5`.

A temporal Transformer trained with ranked probability score instead of
cumulative BCE reached 58.6% within three standalone. The uniform average of
that and other loss variants reached 62.3% in the fold blend, below the best
single loss variant.

A supervised-contrastive projection over mean-pooled DINO-base features learned
similarity from the vote distributions: the pair target was
`exp(-expected tier distance / 3)`. A training-only holdout selected the
projection epoch, the mix of learned and original cosine similarities, and KNN
settings. Seed 42 selected epoch 7, learned similarity only, 40 neighbors, and
temperature 0.16. Mixing its retrieved vote distribution at 5% into the fixed
video/KNN/temporal ensemble scored 64.25% within three (MAE 3.82) on full
validation. The 5% weight was selected in every fold in both five-fold and
ten-fold checks; the corresponding held-out fold scores were 64.26% and 64.27%
within three. This is a small new best, not a result at the <=3 MAE target.

The result was seed-sensitive: seeds 7 and 2026 selected zero weight in most
folds, seed 123 was worse than baseline, and a larger search over seed averages
scored 63.60%. A within-three pairwise metric target and mean-plus-standard-
deviation input also failed to improve the blend. Use `train_dinov2_metric_knn.py`
and `evaluate_temporal_feature_blend.py --checkpoint training/experiments/dinov2_metric_knn.pt`
to reproduce seed 42; the multi-seed check is in `evaluate_metric_knn_seeds.py`.
These checks use validation for fold-held-out blend selection only. The
held-out test partition was not loaded or evaluated.

Explicit first-difference features reached 58.8% standalone and 62.3% in the
fold blend. An 866k-parameter Transformer with attention, mean, and max pooling
reached 57.8% standalone and 62.8% blended. Mild inverse-frequency tier loss
weighting, intended to counter the scarce high-tier labels, reached 56.7%
standalone and 62.0% blended. None beat the 64.1% result. The pooling model's
batch-32 run approached 15.8 GiB GPU use, so it was stopped and safely rerun at
batch 16 (4.9 GiB peak). The training options and checkpoints are in
`train_dinov2_temporal_gpu.py` and `training/experiments/`.

Hashed character n-gram retrieval over level titles selected 80 neighbors and
temperature 0.24 using a training-only holdout, but scored 33.1% within three
on validation by its median readout. Adding it to the video/KNN distribution
scored 60.8% in five-fold validation, below the visual baseline. The titles
were used as text only; no demon-rating field was used.

A 421k-parameter statistics MLP trained on temporal mean, variability, adjacent
change, and peak DINO features reached 56.0% within three with its window
readout. Blending it into the main video/retrieval model reached 61.0% in the
same five-fold check. Blending the separately trained expected-tier DINO head
reached 61.1%. Neither added useful signal over the 61.45% fixed readout.
`train_dinov2_statistics.py` records the statistics-head experiment and keeps
the held-out test partition unevaluated.

A pooled DINO MLP trained with a Gaussian target centered on each level's
vote-weighted mean (sigma 3 tiers) reached 57.8% within three using its window
readout; its fold-held-out blend reached 59.7%. This target smoothing did not
add complementary signal. See `train_gaussian_target_dinov2.py`.

A CORAL pooled head with a single latent score and ordered tier thresholds
reached 54.7% within three by its window readout (mean-tier MAE 4.00). This
stronger ordinal constraint underfit compared with the independent-logit heads.
`train_dinov2_coral.py` contains the fixed-split run; the held-out test split
was not evaluated.

Nested binned isotonic calibration from the video ensemble's mean, median, or
window readouts to expected tier performed worse, at 48.4% within three. This
nonlinear calibration overfit the small validation folds and was rejected.

A video-only temporal-pyramid KNN compared normalized feature means from 1, 2,
4, and 8 ordered time bins with global mean and mean-plus-standard-deviation
representations. The training-only holdout selected 8 bins, but this choice
generalized poorly: KNN alone reached 45.1% within three on validation. A
five-fold video/retrieval blend reached 57.5% within three, below the 58.0%
video-only fold baseline and the 58.7% fixed blend using global-pooling KNN.
Ordered-bin retrieval did not improve the current system; the experiment is
reproducible in `knn_temporal_pyramid.py`.

A four-bin pooled MLP on the same base features has about 869k parameters and
was trained in batches on CPU, using only the four-second feature cache. The
ordinal cross-entropy and Wasserstein variants both peaked near 56% within
three on the validation set, below the 58% cross-validated group-mean median
readout. Their checkpoints and the bounded training command are in
`training/experiments/` and `train_pooled_dinov2.py`.

## Level-file features

Server difficulty ratings and other vote-derived fields are label leakage and
are excluded from candidate inputs and model comparisons. An earlier metadata
experiment used that information, so its scores and category models have been
discarded. Metadata-only cache files have also been removed. The retained
full-level pilot files are used only through their encoded level strings.

We downloaded 200 train and 200 validation level strings and tested hashed
object counts and coarse spatial object histograms. The object-only model
scored about 31% within three in weighted cross-validation, far below the
video model (24.5% within three on the weighted pilot). Five-fold video/object
fusion scored 54.5% within three, below video alone at 57.0%. Simple object
histograms do not help; a stronger attempt would need to separate playable
geometry from decoration and represent object sequences or layout structure.

Randomly masking 10% of one-second DINO tokens during training (replacing their
visual features with the training-set mean while preserving timestamps) reached
57.2% within three on the fixed validation set. Fold-held-out ensemble tuning
selected zero weight in all five folds, leaving the 64.1% baseline unchanged.
This temporal token-dropout variant is available as `--token-dropout` in
`train_dinov2_temporal_gpu.py`; checkpoint:
`dinov2_1s_temporal_tokdrop01_w05.pt`. The held-out test split was not evaluated.

To test whether more level examples help the structural representation, the
training pilot was expanded from 200 to 500 levels while retaining the same
200-level validation pilot. A 37,176-feature representation combined hashed
object IDs at 16-by-8 spatial resolution with ordered object-ID transitions.
Nested training CV changed from 28.0% within three at 200 examples to 26.8% at
500; weighted validation-pilot accuracy changed from 25.5% to 26.5%. This is a
small validation movement without a CV gain for the ridge model. The data and
feature caches are under
`training/level_data/`; reproduce the comparison with
`evaluate_level_sequence_features.py`. Only encoded objects were used; server
ratings and vote-derived metadata were excluded.

A 558k-parameter two-layer Transformer over up to 128 ordered object tokens was
trained on the 500-level training pilot with an internal training-only early
stop split, then refit at the selected epoch. It reached 42.9% within three
tiers on the 200-level validation pilot by its window readout (MAE 6.14), but
its distribution added complementary signal: a 10% object-model blend was
selected in all five validation folds. On this weighted pilot the fixed
video/KNN/DINO temporal ensemble scored 61.47% within three (MAE 4.00); the
fold-held-out 10% blend scored 62.47% (MAE 3.99). This pilot improvement did
not generalize. On all 607 validation levels, the fixed ensemble scored
64.09% within three (MAE 3.84), while a fixed 10% blend scored 63.10% (MAE
3.93). Fold-held-out blend tuning chose zero weight in four of five folds and
scored 62.27% within three (MAE 3.94); tuning point readouts also chose zero
object weight in every fold.

The object Transformer was retrained on 1,000 training levels, with the same
full 607-level validation set and no validation-based early stopping. Its
standalone result was 43.16% within three (window MAE 6.07). The fixed 10%
blend reached 63.26% within three, and fold-held-out tuning chose zero weight
in all five folds, leaving the 64.09% baseline unchanged. More training levels
did not make this representation useful. Both runs used CUDA on the RTX 5080
with a 0.14 GiB peak. Reproduce them with `train_level_object_transformer.py`
and evaluate with `evaluate_level_object_transformer_full_validation.py`.
`download_full_validation_levels.py` fetches only the existing validation IDs;
`download_level_pilot.py --per-quantile 200 --split train` recreates the
1,000-level training sample. The held-out test split was not evaluated, and
level-server rating fields were not read.

Two additional label-loss variants did not beat the visual baseline. Softening
vote targets with power 0.5 and retaining the 0.5 within-three loss reached
58.2% standalone within three; the full-validation fold blend selected zero
weight in every fold. Adding soft categorical CE at weight 0.25 to ordinal CDF
BCE plus the within-three term reached 60.8% standalone, but its fold-selected
blend scored 62.77% (MAE 3.92) versus 64.09% (MAE 3.84) for the baseline. The
hybrid loss was selected at weights ranging from 0 to 0.5 across folds, so it
was unstable. Reproduce these with `--target-sharpening 0.5` and
`--categorical-loss-weight 0.25` in `train_dinov2_temporal_gpu.py`; both runs
used the RTX 5080, peaked at 9.43 GiB, and left the held-out test set untouched.

Repeated randomized train/validation splits compared the one-second temporal
Transformer, four-second cosine KNN, and four-second learned-metric KNN using
the same 4,848/607 partitions. With three splits, learned-metric KNN had the
best individual mean within-three accuracy (56.51%). A fixed 75% learned-metric
KNN + 25% temporal probability blend scored 57.66% within three (MAE 4.016),
beating either component on all three splits. The temporal score uses a fixed
10-epoch schedule to avoid selecting checkpoints on the scored validation
folds. Full results, tested blends, per-split scores, commands, and limitations are in
[`repeated_splits/README.md`](repeated_splits/README.md). The original test IDs
were excluded from every split and were not evaluated.
