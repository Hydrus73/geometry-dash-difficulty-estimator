"""Validation-only temporal test-time augmentation for the best DINO model."""
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.train import DEFAULT_CSV, NUM_CLASSES, read_rows
from training.experiments.train_dinov2_temporal_gpu import (
    TemporalOrdinalTransformer,
    collate,
    score,
    window_center,
)


FEATURES = ROOT / "training" / "features_dinov2_base_4fps_1s"
CHECKPOINT = ROOT / "training" / "experiments" / "dinov2_1s_batched_transformer_within3_w05.pt"
OUTPUT = ROOT / "training" / "experiments" / "dinov2_1s_batched_transformer_within3_w05_tta.pt"
SEED = 42
BATCH_SIZE = 32
MC_PASSES = 20


def evaluate_views(model, examples, indices, tiers, device, reverse=False):
    probabilities = []
    with torch.inference_mode():
        for batch_indices in indices.split(BATCH_SIZE):
            features, padding, _, _ = collate(examples, batch_indices, device)
            if reverse:
                reversed_sequences = []
                for index in batch_indices:
                    sequence = examples[int(index)][0]
                    reversed_sequences.append((sequence.flip(0), examples[int(index)][1], 1.0))
                features, padding, _, _ = collate(
                    reversed_sequences, torch.arange(len(reversed_sequences)), device
                )
            probabilities.append(torch.softmax(model(features, padding), -1).cpu())
    return torch.cat(probabilities)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this validation experiment")
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    device = torch.device("cuda")
    rows = read_rows(DEFAULT_CSV)
    random.Random(SEED).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    valid_rows = rows[boundary:2 * boundary]
    examples = []
    for index, row in enumerate(valid_rows, 1):
        sequence = torch.load(FEATURES / f"{row['level_id']}.pt", map_location="cpu",
                              weights_only=True)["features"].float().half()
        examples.append((sequence, row["target"], 1.0))
        if index % 200 == 0:
            print(f"loaded validation features {index}/{len(valid_rows)}", flush=True)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = TemporalOrdinalTransformer(feature_size=768, dropout=checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    tiers = torch.arange(1, NUM_CLASSES + 1, device=device, dtype=torch.float32)
    indices = torch.arange(len(examples))
    targets = torch.stack([example[1] for example in examples])
    actual = targets @ tiers.cpu()

    model.eval()
    standard = evaluate_views(model, examples, indices, tiers, device)
    reversed_view = evaluate_views(model, examples, indices, tiers, device, reverse=True)
    reverse_average = (standard + reversed_view) / 2

    model.train()  # Enable only the model's dropout layers for MC inference.
    mc_sum = torch.zeros_like(standard)
    for pass_index in range(1, MC_PASSES + 1):
        current = evaluate_views(model, examples, indices, tiers, device)
        mc_sum += current
        if pass_index % 5 == 0:
            print(f"MC-dropout passes {pass_index}/{MC_PASSES}", flush=True)
    mc_average = mc_sum / MC_PASSES
    model.eval()

    def report(name, probabilities):
        predicted = window_center(probabilities, 3)
        result = score(predicted, actual)
        print(f"{name}: MAE={result[0]:.4f}, within3={result[1]:.4f}", flush=True)
        return result

    report("standard", standard)
    report("reverse average", reverse_average)
    report("MC-dropout average", mc_average)
    payload = {
        "standard": standard,
        "reverse_average": reverse_average,
        "mc_dropout_average": mc_average,
        "passes": MC_PASSES,
        "epoch": checkpoint["epoch"],
        "validation_scores": {
            "standard": report("standard (saved)", standard),
            "reverse_average": report("reverse average (saved)", reverse_average),
            "mc_dropout_average": report("MC-dropout average (saved)", mc_average),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUTPUT)
    print("Held-out test partition was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
