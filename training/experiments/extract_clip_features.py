#!/usr/bin/env python3
"""Extract full-video CLIP features at 4 fps, pooled to one vector per second.

The fixed test partition is excluded. Videos are sampled from first frame to
last at the requested rate, with frame embeddings averaged in short intervals.
"""
import argparse
import logging
import math
import os
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from training.train import DEFAULT_CSV, DEFAULT_VIDEO_DIR, read_rows, video_path


LOG = logging.getLogger("extract_clip")
MODEL_NAME = "openai/clip-vit-base-patch16"
OUTPUT_DIR = ROOT / "training" / "features_clip_vitb16_4fps_1s"


def extract_one(path, processor, model, device, sample_fps, batch_size, frames_per_feature):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f"Video has no frame-count metadata: {path}")

        frame_batch, feature_buffer, sequence = [], [], []
        next_sample_frame = 0.0
        frame_index = 0
        dtype = next(model.parameters()).dtype

        def encode_batch():
            if not frame_batch:
                return
            inputs = processor(images=frame_batch, return_tensors="pt")["pixel_values"]
            with torch.inference_mode():
                encoded = model(pixel_values=inputs.to(device=device, dtype=dtype)).pooler_output
            frame_batch.clear()
            for vector in encoded:
                feature_buffer.append(vector)
                if len(feature_buffer) == frames_per_feature:
                    sequence.append(torch.stack(feature_buffer).mean(0))
                    feature_buffer.clear()

        while frame_index < frame_count:
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index >= next_sample_frame:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                # Letterbox to the CLIP input size instead of cropping away the
                # sides of the 16:9 gameplay frame.
                resized = cv2.resize(rgb, (224, 126), interpolation=cv2.INTER_AREA)
                canvas = np.zeros((224, 224, 3), dtype=np.uint8)
                canvas[49:175] = resized
                frame_batch.append(Image.fromarray(canvas))
                next_sample_frame += fps / sample_fps
                if len(frame_batch) >= batch_size:
                    encode_batch()
            frame_index += 1

        encode_batch()
        if feature_buffer:
            sequence.append(torch.stack(feature_buffer).mean(0))
        if not sequence:
            raise RuntimeError(f"No CLIP features were produced: {path}")
        return torch.stack(sequence).float().cpu()
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--feature-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--frames-per-feature", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if (args.sample_fps <= 0 or args.frames_per_feature < 1 or args.batch_size < 1
            or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count):
        parser.error("fps, feature interval, batch size, and shard count must be positive; "
                     "shard index must be in [0, shard count)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This extraction is intended to run on the accessible CUDA GPU")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    processor = CLIPImageProcessor.from_pretrained(
        args.model_name, local_files_only=args.local_files_only,
    )
    processor.do_resize = False
    processor.do_center_crop = False
    model = CLIPVisionModel.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        torch_dtype=torch.float16,
    ).to(device).eval()

    rows = read_rows(args.csv)
    random.Random(42).shuffle(rows)
    boundary = math.ceil(len(rows) * 0.1)
    # Exclude the first 10% test partition; only training and validation videos
    # receive the new representation.
    rows = rows[boundary:]
    if args.limit is not None:
        rows = rows[:args.limit]
    rows = rows[args.shard_index::args.shard_count]
    args.feature_dir.mkdir(parents=True, exist_ok=True)

    encoded = skipped = failed = 0
    for row in tqdm(rows, desc="Extracting CLIP features", unit="video"):
        destination = args.feature_dir / f"{row['level_id']}.pt"
        if destination.is_file():
            skipped += 1
            continue
        source = video_path(row, args.video_dir)
        try:
            features = extract_one(
                source, processor, model, device, args.sample_fps,
                args.batch_size, args.frames_per_feature,
            )
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.", suffix=".partial",
                dir=args.feature_dir, delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                torch.save({
                    "features": features,
                    "model_name": args.model_name,
                    "feature_size": features.shape[-1],
                    "sample_fps": args.sample_fps,
                    "frames_per_feature": args.frames_per_feature,
                }, temporary_path)
                os.replace(temporary_path, destination)
            finally:
                temporary_path.unlink(missing_ok=True)
            encoded += 1
        except Exception as exc:
            failed += 1
            LOG.warning("Skipping %s: %s", source.name, exc)
        if (encoded + skipped + failed) % 250 == 0:
            LOG.info("progress encoded=%d reused=%d failed=%d", encoded, skipped, failed)
    LOG.info("done encoded=%d reused=%d failed=%d output=%s", encoded, skipped,
             failed, args.feature_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
