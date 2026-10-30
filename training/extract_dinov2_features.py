#!/usr/bin/env python3
"""Extract DINOv2-base frame features from the already downloaded showcases.

Example:
    HF_HOME=/tmp/hf_dinov2 python training/extract_dinov2_features.py \
        --feature-dir training/features_dinov2_4fps --sample-fps 4 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from train import DEFAULT_CSV, DEFAULT_VIDEO_DIR, read_rows, video_path


LOG = logging.getLogger("extract_dinov2")
MODEL_NAME = "facebook/dinov2-base"


def extract_one(
    path: Path,
    processor: AutoImageProcessor,
    model: AutoModel,
    device: torch.device,
    sample_fps: float,
    batch_size: int,
    frames_per_feature: int,
) -> torch.Tensor:
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
        frame_batch: list[Image.Image] = []
        frame_feature_buffer: list[torch.Tensor] = []
        clip_features: list[torch.Tensor] = []
        next_sample_frame = 0.0
        frame_index = 0
        dtype = next(model.parameters()).dtype

        def encode_batch() -> None:
            if not frame_batch:
                return
            batch = processor(images=frame_batch, return_tensors="pt")
            pixels = batch["pixel_values"].to(device=device, dtype=dtype)
            with torch.inference_mode():
                encoded = model(pixel_values=pixels).last_hidden_state[:, 0]
            frame_batch.clear()
            for vector in encoded:
                frame_feature_buffer.append(vector)
                if len(frame_feature_buffer) == frames_per_feature:
                    clip_features.append(torch.stack(frame_feature_buffer).mean(dim=0))
                    frame_feature_buffer.clear()

        while frame_index < frame_count:
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index >= next_sample_frame:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                # Preserve the full widescreen gameplay area and use a 28x16
                # patch grid instead of center-cropping the sides of the frame.
                rgb = cv2.resize(rgb, (392, 224), interpolation=cv2.INTER_AREA)
                frame_batch.append(Image.fromarray(rgb))
                next_sample_frame += fps / sample_fps
                if len(frame_batch) >= batch_size:
                    encode_batch()
            frame_index += 1
        if not frame_batch and not frame_feature_buffer and not clip_features:
            raise RuntimeError(f"No decodable frames found: {path}")
        encode_batch()
        if frame_feature_buffer:
            clip_features.append(torch.stack(frame_feature_buffer).mean(dim=0))
        if not clip_features:
            raise RuntimeError(f"No DINOv2 features were produced: {path}")
        return torch.stack(clip_features).float().cpu()
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--feature-dir", type=Path, default=Path("training/features_dinov2_4fps"))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frames-per-feature", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true", help="do not try downloading missing model files")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1, help="number of disjoint extraction workers")
    parser.add_argument("--shard-index", type=int, default=0, help="zero-based worker index")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if (
        args.sample_fps <= 0
        or args.batch_size < 1
        or args.frames_per_feature < 1
        or (args.limit is not None and args.limit < 1)
        or args.shard_count < 1
        or not 0 <= args.shard_index < args.shard_count
    ):
        parser.error(
            "sample-fps, batch-size, frames-per-feature, and limit must be positive; "
            "shard-index must be in [0, shard-count)"
        )

    device = torch.device(args.device)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    processor = AutoImageProcessor.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    processor.do_resize = False
    processor.do_center_crop = False
    model = AutoModel.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    rows = read_rows(args.csv)
    if args.limit is not None:
        rows = rows[:args.limit]
    rows = rows[args.shard_index::args.shard_count]
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    encoded_count = skipped_count = 0
    for row in tqdm(rows, desc="Extracting DINOv2 features", unit="video"):
        destination = args.feature_dir / f"{row['level_id']}.pt"
        if destination.is_file():
            skipped_count += 1
            continue
        source = video_path(row, args.video_dir)
        try:
            features = extract_one(
                source,
                processor,
                model,
                device,
                args.sample_fps,
                args.batch_size,
                args.frames_per_feature,
            )
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.", suffix=".partial", dir=args.feature_dir, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                torch.save(
                    {
                        "features": features,
                        "model_name": args.model_name,
                        "feature_size": model.config.hidden_size,
                        "sample_fps": args.sample_fps,
                        "frames_per_feature": args.frames_per_feature,
                    },
                    temporary_path,
                )
                os.replace(temporary_path, destination)
            finally:
                temporary_path.unlink(missing_ok=True)
            encoded_count += 1
        except Exception as exc:
            LOG.warning("Skipping %s: %s", source.name, exc)
    LOG.info(
        "Extracted %d videos; reused %d cached files; output %s",
        encoded_count, skipped_count, args.feature_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
