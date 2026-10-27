#!/usr/bin/env python3
"""Show the video tensors and clip features produced for one GD level.

Example:
    python training/inspect_video.py --level-id 13519

The script saves a 4x4 frame contact sheet, the preprocessed tensor for the
first VideoMAE clip, and all clip features passed to the temporal head.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch

from train import (
    DEFAULT_CSV,
    DEFAULT_VIDEO_DIR,
    MODEL_NAME,
    NUM_FRAMES,
    download_video,
    extract_video_features,
    read_rows,
)


def inspect_clip_layout(path: Path, sample_fps: float) -> tuple[list[Any], int]:
    """Return the first sampled clip and count clips without retaining all frames."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python to decode the video") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {path}")
    current: list[Any] = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        stride = max(fps / sample_fps, 1.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f"Video has no frame-count metadata: {path}")
        sampled_count = int((frame_count - 1) // stride) + 1
        clip_count = math.ceil(sampled_count / NUM_FRAMES)
        # For a preview, seek directly to the first clip's sampled frames
        # instead of decoding every frame in a potentially hour-long video.
        for sample_index in range(min(NUM_FRAMES, sampled_count)):
            frame_index = math.ceil(sample_index * stride - 1e-6)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {frame_index} from {path}")
            current.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if current and len(current) < NUM_FRAMES:
        current.extend([current[-1].copy() for _ in range(NUM_FRAMES - len(current))])
    if not current:
        raise RuntimeError(f"Video contains no decodable frames: {path}")
    return current, clip_count


def save_contact_sheet(frames: list[Any], destination: Path) -> None:
    """Save the 16 RGB frames in the first model clip as a 4x4 image."""
    import cv2
    import numpy as np

    tile_width, tile_height, label_height = 256, 144, 24
    canvas = np.zeros((4 * (tile_height + label_height), 4 * tile_width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, 4)
        x, y = column * tile_width, row * (tile_height + label_height)
        tile = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        canvas[y : y + tile_height, x : x + tile_width] = tile
        cv2.putText(canvas, f"frame {index + 1}", (x + 6, y + tile_height + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if not cv2.imwrite(str(destination), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write contact sheet: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level-id", required=True, help="GD Ladder level_id from the CSV")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, help="defaults to training/demo_<level-id>")
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--clip-batch-size", type=int, default=4)
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="show the first VideoMAE input clip without encoding all clips",
    )
    parser.add_argument("--device", help="defaults to CUDA when available, otherwise CPU")
    args = parser.parse_args()
    if args.sample_fps <= 0 or args.clip_batch_size < 1:
        parser.error("sample-fps and clip-batch-size must be positive")

    rows = read_rows(args.csv)
    row = next((item for item in rows if item.get("level_id") == args.level_id), None)
    if row is None:
        raise SystemExit(f"No usable Long level with ID {args.level_id!r} was found in {args.csv}")

    output_dir = args.output_dir or Path(__file__).resolve().parent / f"demo_{args.level_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = download_video(row, args.video_dir)
    first_clip, clip_count = inspect_clip_layout(video_path, args.sample_fps)

    from transformers import VideoMAEImageProcessor, VideoMAEModel

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)

    # This is exactly the processor output used for a single VideoMAE call.
    first_input = processor(first_clip, return_tensors="pt").pixel_values.squeeze(0)
    input_path = output_dir / "videomae_first_clip.pt"
    torch.save(first_input.cpu(), input_path)
    contact_path = output_dir / "videomae_first_clip.png"
    save_contact_sheet(first_clip, contact_path)

    rate_key = f"{args.sample_fps:g}".replace(".", "p")
    features_path = output_dir / f"temporal_head_clip_features_{rate_key}fps.pt"
    metadata_path = output_dir / f"temporal_head_clip_features_{rate_key}fps.json"
    cache_metadata = {
        "model_name": MODEL_NAME,
        "sample_fps": args.sample_fps,
        "num_frames": NUM_FRAMES,
        "video_size": video_path.stat().st_size,
        "video_mtime_ns": video_path.stat().st_mtime_ns,
    }

    clip_features: torch.Tensor | None = None
    cache_status = "not requested"
    if not args.preview_only:
        existing_metadata = None
        if metadata_path.is_file():
            try:
                existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing_metadata = None
        if features_path.is_file() and existing_metadata == cache_metadata:
            clip_features = torch.load(features_path, map_location="cpu", weights_only=True)
            cache_status = "reused"
        else:
            device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            encoder = VideoMAEModel.from_pretrained(MODEL_NAME).to(device).eval()
            # The training extractor returns one frozen CLS feature per clip.
            clip_features = extract_video_features(
                video_path, processor, encoder, device, args.sample_fps, args.clip_batch_size
            ).cpu()
            torch.save(clip_features, features_path)
            metadata_path.write_text(json.dumps(cache_metadata, indent=2), encoding="utf-8")
            cache_status = "computed and cached"

    summary = {
        "level_id": args.level_id,
        "name": row.get("name"),
        "video_url": row["video_url"],
        "sample_fps": args.sample_fps,
        "sampled_clip_count": clip_count,
        "frames_per_clip": NUM_FRAMES,
        "videomae_input_shape": list(first_input.shape),
        "temporal_head_input_shape": list(clip_features.shape) if clip_features is not None else None,
        "vote_target_shape": list(row["target"].shape),
        "feature_cache": cache_status,
        "preview_only": args.preview_only,
        "outputs": {
            "contact_sheet": str(contact_path),
            "one_videomae_input_clip": str(input_path),
            "all_temporal_clip_features": str(features_path) if clip_features is not None else None,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
