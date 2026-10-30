"""Train a small ordinal Transformer over ordered objects from GD level strings."""
import argparse
import json
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.experiments.download_gd_level import DEFAULT_CACHE, iter_level_objects
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows


ID_VOCAB, MAX_TOKENS = 4096, 128
HIDDEN, LAYERS, HEADS, DROPOUT = 128, 2, 4, 0.35
OUTPUT = ROOT / "training" / "experiments" / "level_object_transformer.pt"


def select_sample(split, rows, boundary):
    selected = rows[2 * boundary:] if split == "train" else rows[boundary:2 * boundary]
    selected = sorted(selected, key=lambda row: float(
        row["target"] @ torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)))
    population, q_lookup = [], {}
    for q in range(5):
        group = selected[q * len(selected) // 5:(q + 1) * len(selected) // 5]
        population.append(len(group))
        for row in group:
            q_lookup[row["level_id"]] = q
    manifest_name = "pilot_manifest.json" if split == "train" else "validation_pilot_manifest.json"
    manifest = json.loads((DEFAULT_CACHE / manifest_name).read_text(encoding="utf-8"))
    sample = [next(row for row in selected if row["level_id"] == level_id)
              for level_id in manifest["requested_ids"]]
    per_quantile = manifest["per_quantile"]
    weights = torch.tensor([population[q_lookup[row["level_id"]]] / per_quantile for row in sample])
    return sample, weights


def sequence_for_id(level_id):
    torch.set_num_threads(1)
    record = json.loads((DEFAULT_CACHE / f"{level_id}.json").read_text(encoding="utf-8"))
    objects = list(iter_level_objects(record["header"]["4"]))
    if not objects:
        return [0], [[0.0, 0.0, 0.0, 0.0]]
    count = len(objects)
    max_x = max(1.0, max(item[1] for item in objects))
    min_y = min(item[2] for item in objects)
    max_y = max(item[2] for item in objects)
    sample_indices = np.linspace(0, count - 1, min(MAX_TOKENS, count), dtype=np.int64)
    ids, values = [], []
    for index in sample_indices:
        object_id, x, y, rotated, scaled = objects[int(index)]
        ids.append((object_id * 2654435761) % ID_VOCAB)
        y_scale = max(1.0, max_y - min_y)
        values.append((x / max_x, (y - min_y) / y_scale, float(rotated), float(scaled)))
    return ids, values


def load_sequences(rows, name):
    cache_path = DEFAULT_CACHE / f"{name}_object_tokens.pt"
    ids_by_level = [row["level_id"] for row in rows]
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache.get("level_ids") == ids_by_level:
            print(f"loaded cached {name} object tokens", flush=True)
            return cache["sequences"]
    sequence_by_id = {}
    for existing_cache in DEFAULT_CACHE.glob("*_object_tokens.pt"):
        if existing_cache == cache_path:
            continue
        cached = torch.load(existing_cache, map_location="cpu", weights_only=True)
        sequence_by_id.update(zip(cached.get("level_ids", []), cached.get("sequences", [])))
    missing = [level_id for level_id in ids_by_level if level_id not in sequence_by_id]
    if missing:
        print(f"encoding {len(missing)} uncached {name} level sequences", flush=True)
        with ProcessPoolExecutor(max_workers=2) as pool:
            new_sequences = pool.map(sequence_for_id, missing, chunksize=1)
            sequence_by_id.update({
                level_id: (torch.tensor(ids, dtype=torch.long), torch.tensor(values, dtype=torch.float32))
                for level_id, (ids, values) in zip(missing, new_sequences)
            })
    sequences = [sequence_by_id[level_id] for level_id in ids_by_level]
    torch.save({"level_ids": ids_by_level, "sequences": sequences}, cache_path)
    return sequences


def collate(sequences, targets, weights, indices, device):
    chosen = [sequences[int(i)] for i in indices]
    length = max(len(ids) for ids, _ in chosen)
    token_ids = torch.zeros(len(chosen), length, dtype=torch.long)
    numeric = torch.zeros(len(chosen), length, 4, dtype=torch.float32)
    padding = torch.ones(len(chosen), length, dtype=torch.bool)
    for i, (ids, values) in enumerate(chosen):
        n = len(ids)
        token_ids[i, :n] = torch.as_tensor(ids, dtype=torch.long)
        numeric[i, :n] = torch.as_tensor(values, dtype=torch.float32)
        padding[i, :n] = False
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    return (token_ids.to(device), numeric.to(device), padding.to(device),
            targets[index_tensor].to(device), weights[index_tensor].to(device))


class ObjectTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_embedding = nn.Embedding(ID_VOCAB, 64)
        self.value_projection = nn.Linear(4, HIDDEN)
        self.id_projection = nn.Linear(64, HIDDEN)
        self.position_embedding = nn.Embedding(MAX_TOKENS, HIDDEN)
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN, nhead=HEADS, dim_feedforward=2 * HIDDEN,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=LAYERS)
        self.attention = nn.Linear(HIDDEN, 1)
        self.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(HIDDEN, NUM_CLASSES))

    def forward(self, token_ids, numeric, padding):
        length = token_ids.shape[1]
        positions = torch.arange(length, device=token_ids.device)[None, :]
        hidden = (self.id_projection(self.id_embedding(token_ids))
                  + self.value_projection(numeric)
                  + self.position_embedding(positions))
        hidden = self.encoder(hidden, src_key_padding_mask=padding)
        weights = torch.softmax(self.attention(hidden).squeeze(-1).masked_fill(padding, -1e4), dim=1)
        return self.classifier(torch.sum(hidden * weights[:, :, None], dim=1))


def loss_for(logits, target, actual, tiers):
    cdf = torch.softmax(logits, -1).cumsum(-1)[:, :-1].clamp(1e-7, 1 - 1e-7)
    target_cdf = target.cumsum(-1)[:, :-1].clamp(0, 1)
    ordinal = F.binary_cross_entropy(cdf, target_cdf, reduction="none").mean(-1)
    probabilities = torch.softmax(logits, -1)
    expected = probabilities @ tiers
    smooth = F.smooth_l1_loss(expected, actual, reduction="none")
    in_window = (tiers[None, :] - actual[:, None]).abs() <= 3
    window_probability = (probabilities * in_window).sum(-1).clamp_min(1e-7)
    within = -window_probability.log()
    return ordinal + 0.2 * smooth + 0.5 * within


def score(probabilities, actual, weights):
    tiers = torch.arange(1, probabilities.shape[-1] + 1, dtype=probabilities.dtype)
    cumulative = torch.cat((torch.zeros(len(probabilities), 1), probabilities.cumsum(-1)), dim=-1)
    centers = torch.arange(probabilities.shape[-1])
    lower = (centers - 3).clamp_min(0)
    upper = (centers + 4).clamp_max(probabilities.shape[-1])
    window = (cumulative[:, upper] - cumulative[:, lower]).argmax(-1).float() + 1
    target_prediction = probabilities @ tiers
    error_window = (window - actual).abs()
    error_mean = (target_prediction - actual).abs()
    weights = weights / weights.sum()
    return {
        "window_mae": (error_window * weights).sum().item(),
        "window_within3": ((error_window <= 3) * weights).sum().item(),
        "mean_mae": (error_mean * weights).sum().item(),
        "mean_within3": ((error_mean <= 3) * weights).sum().item(),
    }


def predict(model, sequences, targets, weights, indices, device, batch_size=32):
    model.eval()
    all_probs = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_ids = indices[start:start + batch_size]
            token_ids, numeric, padding, _, _ = collate(
                sequences, targets, weights, batch_ids, device)
            all_probs.append(torch.softmax(model(token_ids, numeric, padding), -1).cpu())
    return torch.cat(all_probs)


def fit_epochs(model, sequences, targets, weights, fit_indices, epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32, device=device)
    for _ in range(epochs):
        model.train()
        permutation = fit_indices[torch.randperm(len(fit_indices))]
        for start in range(0, len(permutation), 32):
            batch_indices = permutation[start:start + 32]
            token_ids, numeric, padding, target, sample_weight = collate(
                sequences, targets, weights, batch_indices, device)
            actual = target @ tiers
            per_example = loss_for(model(token_ids, numeric, padding), target, actual, tiers)
            loss = (per_example * sample_weight).sum() / sample_weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help=f"checkpoint destination (default: {OUTPUT})")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; launch outside the restricted sandbox")
    torch.set_num_threads(4)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    rows = read_rows(DEFAULT_CSV)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    train_rows, train_weights = select_sample("train", rows, boundary)
    valid_rows, valid_weights = select_sample("validation", rows, boundary)
    train_sequences = load_sequences(train_rows, "training")
    valid_sequences = load_sequences(valid_rows, "validation")
    train_targets = torch.stack([row["target"] for row in train_rows])
    valid_targets = torch.stack([row["target"] for row in valid_rows])
    tiers = torch.arange(1, NUM_CLASSES + 1, dtype=torch.float32)
    train_actual = train_targets @ tiers
    valid_actual = valid_targets @ tiers

    indices = list(range(len(train_rows)))
    random.Random(43).shuffle(indices)
    split = max(1, round(len(indices) * 0.8))
    fit_indices = torch.tensor(indices[:split], dtype=torch.long)
    early_indices = torch.tensor(indices[split:], dtype=torch.long)
    model = ObjectTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    best_state, best_within, best_mae, best_epoch, stale = None, -1.0, float("inf"), 0, 0
    early_weights = train_weights[early_indices]
    for epoch in range(1, 41):
        model.train()
        permutation = fit_indices[torch.randperm(len(fit_indices))]
        for start in range(0, len(permutation), 32):
            batch_indices = permutation[start:start + 32]
            token_ids, numeric, padding, target, sample_weight = collate(
                train_sequences, train_targets, train_weights, batch_indices, device)
            actual = target @ tiers.to(device)
            per_example = loss_for(model(token_ids, numeric, padding), target, actual,
                                   tiers.to(device))
            loss = (per_example * sample_weight).sum() / sample_weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        early_probs = predict(model, train_sequences, train_targets, train_weights,
                              early_indices, device)
        early_scores = score(early_probs, train_actual[early_indices], early_weights)
        improved = (early_scores["window_within3"] > best_within
                    or (early_scores["window_within3"] == best_within
                        and early_scores["window_mae"] < best_mae))
        if improved:
            best_within = early_scores["window_within3"]
            best_mae = early_scores["window_mae"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(f"inner_epoch={epoch:02d} loss={loss.item():.4f} "
              f"inner_within3={early_scores['window_within3']:.3f} "
              f"best_epoch={best_epoch} gpu_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB",
              flush=True)
        if stale >= 8:
            break

    # Select training duration on the inner split, then refit with every sampled training level.
    del model, optimizer
    torch.manual_seed(seed)
    model = ObjectTransformer().to(device)
    fit_epochs(model, train_sequences, train_targets, train_weights,
               torch.arange(len(train_rows)), best_epoch, device)
    valid_indices = torch.arange(len(valid_rows))
    valid_probabilities = predict(model, valid_sequences, valid_targets, valid_weights,
                                  valid_indices, device)
    validation_scores = score(valid_probabilities, valid_actual, valid_weights)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    payload = {
        "model": "ordinal Transformer over ordered GD object tokens",
        "architecture": "two-layer Transformer, hashed object ID plus normalized layout values",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "validation_probabilities": valid_probabilities,
        "validation_level_ids": [row["level_id"] for row in valid_rows],
        "validation_scores": validation_scores,
        "inner_best_epoch": best_epoch,
        "inner_best_scores": {"window_within3": best_within, "window_mae": best_mae},
        "parameters": parameter_count,
        "train_levels": len(train_rows),
        "training_level_ids": [row["level_id"] for row in train_rows],
        "validation_levels": len(valid_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"device={torch.cuda.get_device_name(0)} parameters={parameter_count} "
          f"best_epoch={best_epoch} validation={validation_scores} "
          f"gpu_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
    print("Held-out test partition was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
