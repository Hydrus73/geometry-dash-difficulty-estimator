"""Train a batched temporal Transformer on cached frame-encoder features.

Uses CUDA when available, standardizes from training clips only, and never
loads or evaluates the held-out test partition.
"""
import argparse
import copy
import math
import os
import random
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


FEATURES = ROOT / "training" / "features_dinov2_base_4fps_1s"
VIDEOMAE_FEATURES = ROOT / "training" / "features"
OUTPUT = ROOT / "training" / "experiments" / "dinov2_1s_batched_transformer.pt"
SEED = 42
HIDDEN = 192
LAYERS = 2
HEADS = 6
DROPOUT = 0.3
BATCH_SIZE = 32
MAX_EPOCHS = 60
PATIENCE = 9
EXPECTED_LOSS_WEIGHT = 0.2


class TemporalOrdinalTransformer(nn.Module):
    def __init__(self, feature_size=768, dropout=DROPOUT, pooling="attention"):
        super().__init__()
        self.pooling = pooling
        self.register_buffer("feature_mean", torch.zeros(feature_size))
        self.register_buffer("feature_std", torch.ones(feature_size))
        self.input_projection = nn.Linear(feature_size, HIDDEN)
        self.position_projection = nn.Linear(32, HIDDEN)
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN,
            nhead=HEADS,
            dim_feedforward=HIDDEN * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=LAYERS)
        self.attention = nn.Linear(HIDDEN, 1)
        if pooling == "attention_mean_max":
            self.pool_fusion = nn.Sequential(
                nn.Linear(3 * HIDDEN, HIDDEN), nn.GELU(), nn.Dropout(dropout)
            )
        else:
            self.pool_fusion = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(HIDDEN, NUM_CLASSES))

    def forward(self, features, padding_mask):
        batch_size, length, _ = features.shape
        lengths = (~padding_mask).sum(1).clamp_min(1)
        positions = torch.arange(length, device=features.device, dtype=features.dtype)[None, :]
        positions = positions / (lengths[:, None] - 1).clamp_min(1)
        frequencies = 2.0 ** torch.arange(16, device=features.device, dtype=features.dtype)
        angles = positions[:, :, None] * frequencies[None, None, :] * (2 * math.pi)
        time_code = torch.cat((angles.sin(), angles.cos()), dim=-1)
        normalized = (features - self.feature_mean) / self.feature_std
        hidden = self.input_projection(normalized) + self.position_projection(time_code)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        attention_logits = self.attention(hidden).squeeze(-1).masked_fill(padding_mask, -1e4)
        weights = torch.softmax(attention_logits, dim=1)
        pooled_attention = torch.sum(hidden * weights[:, :, None], dim=1)
        if self.pooling == "attention_mean_max":
            valid_hidden = hidden.masked_fill(padding_mask[:, :, None], 0)
            pooled_mean = valid_hidden.sum(1) / lengths[:, None]
            pooled_max = hidden.masked_fill(padding_mask[:, :, None], -1e4).amax(1)
            pooled = self.pool_fusion(torch.cat((pooled_attention, pooled_mean, pooled_max), -1))
        else:
            pooled = pooled_attention
        return self.classifier(pooled)


class TemporalOrdinalGRU(nn.Module):
    """Bidirectional recurrent pooling head for ordered one-second features."""
    def __init__(self, feature_size=768, dropout=DROPOUT, recurrent_hidden=96):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(feature_size))
        self.register_buffer("feature_std", torch.ones(feature_size))
        self.input_projection = nn.Sequential(
            nn.Linear(feature_size, HIDDEN), nn.GELU(), nn.Dropout(dropout)
        )
        self.encoder = nn.GRU(
            input_size=HIDDEN,
            hidden_size=recurrent_hidden,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        encoded_size = 2 * recurrent_hidden
        self.attention = nn.Linear(encoded_size, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoded_size, NUM_CLASSES))

    def forward(self, features, padding_mask):
        lengths = (~padding_mask).sum(1)
        normalized = (features - self.feature_mean) / self.feature_std
        hidden = self.input_projection(normalized)
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_hidden, _ = self.encoder(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_hidden, batch_first=True, total_length=features.shape[1]
        )
        attention_logits = self.attention(hidden).squeeze(-1).masked_fill(padding_mask, -1e4)
        weights = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(hidden * weights[:, :, None], dim=1)
        return self.classifier(pooled)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        padding = 2 * dilation
        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5,
                               padding=padding, dilation=dilation)
        self.norm2 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5,
                               padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, padding_mask):
        residual = hidden
        value = self.norm1(hidden).transpose(1, 2)
        value = self.dropout(F.gelu(self.conv1(value))).transpose(1, 2)
        value = value.masked_fill(padding_mask[:, :, None], 0)
        value = self.norm2(value).transpose(1, 2)
        value = self.dropout(self.conv2(value).transpose(1, 2))
        return (residual + value).masked_fill(padding_mask[:, :, None], 0)


class TemporalOrdinalTCN(nn.Module):
    """Dilated residual convolutions over normalized video time."""
    CHANNELS = 128

    def __init__(self, feature_size=768, dropout=DROPOUT):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(feature_size))
        self.register_buffer("feature_std", torch.ones(feature_size))
        self.input_projection = nn.Linear(feature_size, self.CHANNELS)
        self.position_projection = nn.Linear(32, self.CHANNELS)
        self.blocks = nn.ModuleList([
            TemporalResidualBlock(self.CHANNELS, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        ])
        self.attention = nn.Linear(self.CHANNELS, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.CHANNELS, NUM_CLASSES))

    def forward(self, features, padding_mask):
        lengths = (~padding_mask).sum(1).clamp_min(1)
        positions = torch.arange(features.shape[1], device=features.device,
                                 dtype=features.dtype)[None, :]
        positions = positions / (lengths[:, None] - 1).clamp_min(1)
        frequencies = 2.0 ** torch.arange(16, device=features.device, dtype=features.dtype)
        angles = positions[:, :, None] * frequencies[None, None, :] * (2 * math.pi)
        time_code = torch.cat((angles.sin(), angles.cos()), dim=-1)
        normalized = (features - self.feature_mean) / self.feature_std
        hidden = self.input_projection(normalized) + self.position_projection(time_code)
        hidden = hidden.masked_fill(padding_mask[:, :, None], 0)
        for block in self.blocks:
            hidden = block(hidden, padding_mask)
        attention_logits = self.attention(hidden).squeeze(-1).masked_fill(padding_mask, -1e4)
        weights = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(hidden * weights[:, :, None], dim=1)
        return self.classifier(pooled)


def collate(examples, indices, device):
    sequences = [examples[int(index)][0] for index in indices]
    targets = torch.stack([examples[int(index)][1] for index in indices]).to(device)
    sample_weights = torch.tensor(
        [examples[int(index)][2] for index in indices], dtype=torch.float32, device=device
    )
    max_length = max(len(sequence) for sequence in sequences)
    feature_size = sequences[0].shape[1]
    batch = torch.zeros(len(sequences), max_length, feature_size, dtype=torch.float32)
    padding = torch.ones(len(sequences), max_length, dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        batch[index, :length] = sequence.float()
        padding[index, :length] = False
    return batch.to(device), padding.to(device), targets, sample_weights


def window_center(probabilities, radius=3):
    cumulative = torch.cat((torch.zeros(len(probabilities), 1, device=probabilities.device),
                            probabilities.cumsum(-1)), dim=-1)
    classes = probabilities.shape[-1]
    centers = torch.arange(classes, device=probabilities.device)
    lower = (centers - radius).clamp_min(0)
    upper = (centers + radius + 1).clamp_max(classes)
    return (cumulative[:, upper] - cumulative[:, lower]).argmax(-1).float() + 1


def score(prediction, actual):
    error = (prediction - actual).abs()
    return error.mean().item(), (error <= 3).float().mean().item()


def ordinal_loss(logits, target, loss_type="cdf_bce"):
    predicted_cdf = torch.softmax(logits, -1).cumsum(-1)[:, :-1].clamp(1e-7, 1 - 1e-7)
    target_cdf = target.cumsum(-1)[:, :-1].clamp(0, 1)
    if loss_type == "rps":
        return (predicted_cdf - target_cdf).square().sum(-1)
    return F.binary_cross_entropy(predicted_cdf, target_cdf, reduction="none").mean(-1)


def evaluate(model, examples, indices, device, tiers, batch_size=BATCH_SIZE):
    model.eval()
    distributions, targets = [], []
    with torch.inference_mode():
        for batch_indices in indices.split(batch_size):
            features, padding, target, _ = collate(examples, batch_indices, device)
            distributions.append(torch.softmax(model(features, padding), -1))
            targets.append(target)
    probabilities = torch.cat(distributions)
    target = torch.cat(targets)
    actual = target @ tiers
    mean = probabilities @ tiers
    median = torch.searchsorted(
        probabilities.cumsum(-1), torch.full((len(probabilities), 1), 0.5, device=device)
    ).squeeze(-1).float() + 1
    window = window_center(probabilities, 3)
    return probabilities, {
        "mean": score(mean, actual),
        "median": score(median, actual),
        "window+/-3": score(window, actual),
    }


def save(payload, output):
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.",
                                     suffix=".partial", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42,
                        help="model initialization and minibatch seed (default: 42)")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="randomized train/validation split seed; original test IDs stay excluded")
    parser.add_argument("--validation-ids-file", type=Path,
                        help="one level ID per line; when set, all other dataset rows are training data")
    parser.add_argument("--architecture", choices=("transformer", "gru", "tcn"), default="transformer",
                        help="temporal sequence encoder (default: transformer)")
    parser.add_argument("--pooling", choices=("attention", "attention_mean_max"), default="attention",
                        help="temporal pooling (default: attention)")
    parser.add_argument("--feature-dir", type=Path, default=FEATURES,
                        help=f"cached feature directory (default: {FEATURES})")
    parser.add_argument("--additional-feature-dir", type=Path,
                        help="optional aligned second visual feature stream to concatenate")
    parser.add_argument("--include-videomae", action="store_true",
                        help="align and concatenate cached VideoMAE tokens with the selected features")
    parser.add_argument("--include-deltas", action="store_true",
                        help="concatenate the signed first difference of adjacent DINO tokens")
    parser.add_argument("--dropout", type=float, default=DROPOUT,
                        help=f"transformer/head dropout (default: {DROPOUT})")
    parser.add_argument("--token-dropout", type=float, default=0.0,
                        help="training-only probability of masking each valid temporal feature token")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"GPU minibatch size (default: {BATCH_SIZE})")
    parser.add_argument("--fixed-epochs", type=int,
                        help="train exactly this many epochs and save the final epoch; disables validation-based checkpoint selection")
    parser.add_argument("--swa", action="store_true",
                        help="evaluate and checkpoint the running average of epoch weights")
    parser.add_argument("--vote-count-weight", type=float, default=0.0,
                        help="training loss weight exponent for vote totals (default: 0 disables)")
    parser.add_argument("--target-confidence-weight", type=float, default=0.0,
                        help="exponent for downweighting high-entropy vote distributions (default: 0 disables)")
    parser.add_argument("--min-votes", type=int, default=0,
                        help="minimum vote count for training rows only (default: 0 keeps all)")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="random nested fraction of training rows to use (default: 1.0)")
    parser.add_argument("--tier-density-weight", type=float, default=0.0,
                        help="inverse-frequency loss exponent by 4-tier target bins (default: 0 disables)")
    parser.add_argument("--within-three-loss-weight", type=float, default=0.0,
                        help="weight for probability-mass loss inside +/-3 tiers (default: 0)")
    parser.add_argument("--expected-loss-weight", type=float, default=EXPECTED_LOSS_WEIGHT,
                        help=f"weight for smooth-L1 expected-tier loss (default: {EXPECTED_LOSS_WEIGHT})")
    parser.add_argument("--within-three-focal-gamma", type=float, default=0.0,
                        help="focal exponent for within-three loss (default: 0 disables focal weighting)")
    parser.add_argument("--ordinal-loss", choices=("cdf_bce", "rps"), default="cdf_bce",
                        help="ordinal distribution loss (default: cumulative BCE)")
    parser.add_argument("--categorical-loss-weight", type=float, default=0.0,
                        help="weight of an auxiliary soft categorical cross-entropy term")
    parser.add_argument("--target-sharpening", type=float, default=1.0,
                        help="power applied to training vote distributions for ordinal loss; "
                             "1 keeps the original distribution, >1 sharpens (default: 1)")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help=f"checkpoint destination (default: {OUTPUT})")
    args = parser.parse_args()
    if (args.min_votes < 0 or not 0 < args.train_fraction <= 1
            or args.target_sharpening <= 0 or args.target_confidence_weight < 0
            or not 0 <= args.token_dropout < 1 or args.categorical_loss_weight < 0
            or (args.fixed_epochs is not None and args.fixed_epochs <= 0)):
        parser.error("min-votes must be nonnegative, train-fraction in (0, 1], "
                     "target-sharpening positive, target-confidence-weight nonnegative, "
                     "token-dropout in [0, 1), categorical-loss-weight nonnegative, "
                     "and fixed-epochs positive when provided")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment; run outside the restricted sandbox")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    output = args.output.resolve()
    feature_dir = args.feature_dir.resolve()
    additional_feature_dir = (args.additional_feature_dir.resolve()
                              if args.additional_feature_dir else None)
    device = torch.device("cuda")
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
        # The seed-42 first 10% is held out from every randomized split.
        random.Random(SEED).shuffle(rows)
        development_rows = rows[boundary:]
        if args.split_seed == SEED:
            split_rows = development_rows
        else:
            split_rows = list(development_rows)
            random.Random(args.split_seed).shuffle(split_rows)
        valid_rows = split_rows[:boundary]
        train_rows = split_rows[boundary:]
    if args.min_votes:
        train_rows = [row for row in train_rows
                      if row["_target_vote_count"] >= args.min_votes]
    if args.train_fraction < 1:
        # Identical seed values produce nested subsets across requested fractions.
        random.Random(args.seed + 1000).shuffle(train_rows)
        keep = max(1, round(len(train_rows) * args.train_fraction))
        train_rows = train_rows[:keep]
    print(f"training examples={len(train_rows)} after min-votes={args.min_votes}, "
          f"train-fraction={args.train_fraction:g}; "
          f"validation examples={len(valid_rows)} (unfiltered)", flush=True)
    train_vote_counts = torch.tensor(
        [row["_target_vote_count"] for row in train_rows], dtype=torch.float64
    )
    train_vote_weights = (train_vote_counts / train_vote_counts.mean()).pow(args.vote_count_weight)
    if args.target_confidence_weight:
        training_targets = torch.stack([row["target"] for row in train_rows]).clamp_min(1e-8)
        normalized_entropy = -(training_targets * training_targets.log()).sum(-1) / math.log(NUM_CLASSES)
        confidence = (1.0 - normalized_entropy).clamp(min=0.05, max=1.0)
        train_vote_weights *= confidence.pow(args.target_confidence_weight)
    if args.tier_density_weight:
        cpu_tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
        train_target_means = torch.stack([row["target"] for row in train_rows]) @ cpu_tiers
        bin_ids = ((train_target_means - 1) / 4).floor().long().clamp(0, 9)
        bin_counts = torch.bincount(bin_ids, minlength=10).double()
        density_weights = (bin_counts.mean() / bin_counts[bin_ids].clamp_min(1)).pow(
            args.tier_density_weight
        ).clamp(max=4.0)
        train_vote_weights *= density_weights
    train_vote_weights /= train_vote_weights.mean()
    weight_by_level = {row["level_id"]: float(weight)
                       for row, weight in zip(train_rows, train_vote_weights)}
    examples = []

    def load_features(row):
        dino = torch.load(feature_dir / f"{row['level_id']}.pt", map_location="cpu",
                          weights_only=True)["features"].float()
        streams = [dino]
        if args.include_deltas:
            delta = torch.zeros_like(dino)
            delta[1:] = dino[1:] - dino[:-1]
            streams.append(delta)
        if args.include_videomae:
            videomae = torch.load(VIDEOMAE_FEATURES / f"{row['level_id']}.pt", map_location="cpu",
                                  weights_only=True)["features"].float()
            aligned_videomae = F.interpolate(
                videomae.T.unsqueeze(0), size=len(dino), mode="linear", align_corners=False
            ).squeeze(0).T
            streams.append(aligned_videomae)
        if additional_feature_dir is not None:
            additional = torch.load(additional_feature_dir / f"{row['level_id']}.pt",
                                    map_location="cpu", weights_only=True)["features"].float()
            if additional.shape != dino.shape:
                raise ValueError(f"unaligned feature shapes for level {row['level_id']}: "
                                 f"{tuple(dino.shape)} vs {tuple(additional.shape)}")
            streams.append(additional)
        return torch.cat(streams, dim=-1) if len(streams) > 1 else dino

    input_size = 768 * (1 + int(args.include_videomae) + int(args.include_deltas)
                        + int(additional_feature_dir is not None))
    feature_sum = torch.zeros(input_size, dtype=torch.float64)
    feature_square_sum = torch.zeros(input_size, dtype=torch.float64)
    feature_count = 0
    for index, row in enumerate(train_rows, 1):
        sequence = load_features(row)
        values = sequence.double()
        feature_sum += values.sum(0)
        feature_square_sum += values.square().sum(0)
        feature_count += len(sequence)
        examples.append((sequence.half(), row["target"], weight_by_level[row["level_id"]]))
        if index % 1000 == 0:
            print(f"loaded training sequences {index}/{len(train_rows)}", flush=True)
    validation_offset = len(examples)
    for index, row in enumerate(valid_rows, 1):
        sequence = load_features(row)
        examples.append((sequence.half(), row["target"], 1.0))
    train_mean = feature_sum / feature_count
    train_variance = (feature_square_sum / feature_count - train_mean.square()).clamp_min(1e-8)

    model_class = {
        "transformer": TemporalOrdinalTransformer,
        "gru": TemporalOrdinalGRU,
        "tcn": TemporalOrdinalTCN,
    }[args.architecture]
    model_kwargs = {"feature_size": input_size, "dropout": args.dropout}
    if args.architecture == "transformer":
        model_kwargs["pooling"] = args.pooling
    model = model_class(**model_kwargs).to(device)
    swa_model = copy.deepcopy(model) if args.swa else None
    swa_state = None
    swa_count = 0
    model.feature_mean.copy_(train_mean.float().to(device))
    model.feature_std.copy_(train_variance.sqrt().float().to(device))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    train_indices = torch.arange(len(train_rows))
    valid_indices = torch.arange(validation_offset, len(examples))
    tiers = torch.arange(1, NUM_CLASSES + 1, device=device, dtype=torch.float32)
    train_actual = torch.stack([row["target"] for row in train_rows]).to(device) @ tiers
    best_within, best_mae, best_epoch, stale = -1.0, float("inf"), 0, 0
    output.parent.mkdir(parents=True, exist_ok=True)

    epoch_limit = args.fixed_epochs or MAX_EPOCHS
    for epoch in range(1, epoch_limit + 1):
        model.train()
        permutation = train_indices[torch.randperm(len(train_indices))]
        for batch_indices in permutation.split(args.batch_size):
            features, padding, targets, sample_weights = collate(examples, batch_indices, device)
            if args.token_dropout:
                # Replace masked visual tokens with the training feature mean. The model then
                # sees zero-valued standardized features while retaining their time positions.
                dropped = (torch.rand(padding.shape, device=device) < args.token_dropout) & ~padding
                features = torch.where(
                    dropped[:, :, None],
                    model.feature_mean.to(features.dtype)[None, None, :],
                    features,
                )
            logits = model(features, padding)
            probabilities = torch.softmax(logits, -1)
            ordinal_targets = targets
            if args.target_sharpening != 1.0:
                ordinal_targets = targets.pow(args.target_sharpening)
                ordinal_targets = ordinal_targets / ordinal_targets.sum(-1, keepdim=True).clamp_min(1e-7)
            per_example_loss = ordinal_loss(logits, ordinal_targets, args.ordinal_loss)
            if args.categorical_loss_weight:
                categorical_loss = -(ordinal_targets * torch.log_softmax(logits, -1)).sum(-1)
                per_example_loss = per_example_loss + args.categorical_loss_weight * categorical_loss
            predicted_tier = probabilities @ tiers
            target_tier = targets @ tiers
            per_example_loss = per_example_loss + args.expected_loss_weight * F.smooth_l1_loss(
                predicted_tier, target_tier, reduction="none"
            )
            if args.within_three_loss_weight:
                in_window = (tiers[None, :] - target_tier[:, None]).abs() <= 3.0
                window_probability = (probabilities * in_window).sum(-1).clamp_min(1e-7)
                within_loss = -window_probability.log()
                if args.within_three_focal_gamma:
                    within_loss = within_loss * (1.0 - window_probability).pow(
                        args.within_three_focal_gamma
                    )
                per_example_loss = per_example_loss + args.within_three_loss_weight * within_loss
            loss = (per_example_loss * sample_weights).sum() / sample_weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        evaluation_model = model
        if args.swa:
            current_state = model.state_dict()
            if swa_state is None:
                swa_state = {key: value.detach().clone() for key, value in current_state.items()}
            else:
                next_count = swa_count + 1
                for key, value in current_state.items():
                    swa_state[key].lerp_(value.detach(), 1.0 / next_count)
            swa_count += 1
            swa_model.load_state_dict(swa_state)
            evaluation_model = swa_model
        valid_probabilities, validation_scores = evaluate(
            evaluation_model, examples, valid_indices, device, tiers, args.batch_size
        )
        val_mae, val_within = validation_scores["window+/-3"]
        improved = (epoch == args.fixed_epochs if args.fixed_epochs is not None else
                    val_within > best_within or (val_within == best_within and val_mae < best_mae))
        if improved:
            best_within, best_mae, best_epoch, stale = val_within, val_mae, epoch, 0
            save({
                "state_dict": {key: value.detach().cpu()
                               for key, value in evaluation_model.state_dict().items()},
                "feature_mean": train_mean.float(),
                "feature_std": train_variance.sqrt().float(),
                "model": f"batched temporal {args.architecture.upper()} on {feature_dir.name} features",
                "split_seed": args.split_seed,
                "training_level_ids": [row["level_id"] for row in train_rows],
                "validation_level_ids": [row["level_id"] for row in valid_rows],
                "feature_dir": str(feature_dir),
                "additional_feature_dir": str(additional_feature_dir) if additional_feature_dir else None,
                "architecture": args.architecture,
                "pooling": args.pooling if args.architecture == "transformer" else "attention",
                "swa": args.swa,
                "uses_videomae_features": args.include_videomae,
                "uses_dino_deltas": args.include_deltas,
                "feature_size": input_size,
                "hidden_size": HIDDEN,
                "layers": LAYERS,
                "heads": HEADS,
                "dropout": args.dropout,
                "token_dropout": args.token_dropout,
                "vote_count_weight": args.vote_count_weight,
                "target_confidence_weight": args.target_confidence_weight,
                "min_training_votes": args.min_votes,
                "training_fraction": args.train_fraction,
                "tier_density_weight": args.tier_density_weight,
                "within_three_loss_weight": args.within_three_loss_weight,
                "within_three_focal_gamma": args.within_three_focal_gamma,
                "expected_loss_weight": args.expected_loss_weight,
                "target_sharpening": args.target_sharpening,
                "categorical_loss_weight": args.categorical_loss_weight,
                "ordinal_loss": args.ordinal_loss,
                "parameters": parameter_count,
                "epoch": epoch,
                "fixed_epochs": args.fixed_epochs,
                "validation_probabilities": valid_probabilities.detach().cpu(),
                "validation_scores": validation_scores,
            }, output)
        else:
            stale += 1
        print(f"epoch={epoch:02d} loss={loss.detach().item():.4f} "
              f"window3_MAE={val_mae:.3f} window3_acc={val_within:.3f} "
              f"best_epoch={best_epoch} gpu_peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
              flush=True)
        if args.fixed_epochs is None and stale >= PATIENCE:
            break

    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    print(f"device={torch.cuda.get_device_name(0)} parameters={parameter_count} "
          f"best_epoch={best_epoch} validation_window3={best_within:.4f} MAE={best_mae:.4f}",
          flush=True)
    print("Validation readouts:", checkpoint["validation_scores"], flush=True)
    print("Held-out test partition was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
