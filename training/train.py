#!/usr/bin/env python3
"""Train a 39-way GD ladder predictor with a frozen VideoMAE encoder.

Dependencies: torch, transformers, yt-dlp, and opencv-python.
Run from any directory with ``python training/train.py``. Downloaded videos
and checkpoints are stored under ``training/`` by default.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import math
import os
import re
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn
from yt_dlp import YoutubeDL


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "gdladder_demons.csv"
DEFAULT_VIDEO_DIR = ROOT / "training" / "videos"
DEFAULT_CHECKPOINT = ROOT / "training" / "videomae_gdladder.pt"
MODEL_NAME = "MCG-NJU/videomae-base"
NUM_CLASSES = 39
NUM_FRAMES = 16
LOG = logging.getLogger("train")


def read_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Read usable Long rows, taking only the first showcase link."""
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as source:
        for line_number, row in enumerate(csv.DictReader(source), start=2):
            if row.get("level_length", "").strip() != "Long":
                continue
            try:
                links = json.loads(row.get("showcase_links", ""))
                votes = ast.literal_eval(row.get("difficulty_vote_distribution", ""))
                url = links[0]
                counts = [float(votes.get(i, votes.get(str(i), 0))) for i in range(1, NUM_CLASSES + 1)]
            except (ValueError, SyntaxError, TypeError, IndexError, KeyError, AttributeError) as exc:
                LOG.warning("Skipping malformed CSV row %d: %s", line_number, exc)
                continue
            if not isinstance(url, str) or not url.startswith(("https://", "http://")):
                LOG.warning("Skipping row %d with an invalid first showcase URL", line_number)
                continue
            if not all(math.isfinite(n) and n >= 0 for n in counts) or sum(counts) == 0:
                LOG.warning("Skipping row %d with an empty or invalid vote distribution", line_number)
                continue
            row["video_url"] = url
            vote_counts = torch.tensor(counts, dtype=torch.float32)
            # Softmax(log(counts)) preserves the observed vote proportions;
            # softmax(counts) would turn large raw vote totals nearly one-hot.
            log_counts = torch.where(vote_counts > 0, vote_counts.log(), -torch.inf)
            row["target"] = torch.softmax(log_counts, dim=0)
            row["display_name"] = row.get("name") or row.get("level_id") or f"row {line_number}"
            rows.append(row)
    return rows


def video_path(row: dict[str, Any], video_dir: Path) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", row.get("level_id", ""))
    return video_dir / f"{safe_id or 'level'}.mp4"


def cached_video_is_decodable(path: Path) -> bool:
    """Reject partial or unsupported-codec cache entries before reusing them."""
    try:
        import cv2
    except ImportError:
        # The later decoder reports the missing dependency with a clearer error.
        return True
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        decoded, _ = capture.read()
        return bool(decoded)
    finally:
        capture.release()


def download_video(row: dict[str, Any], video_dir: Path) -> Path:
    """Download the best available video stream at or below 360p."""
    video_dir.mkdir(parents=True, exist_ok=True)
    destination = video_path(row, video_dir)
    if destination.is_file() and destination.stat().st_size:
        if cached_video_is_decodable(destination):
            return destination
        LOG.warning("Cached video %s cannot be decoded; downloading it again", destination)

    # Download into a per-video temporary directory to avoid exposing partial
    # files as cache hits. A merge is requested when yt-dlp selects separate
    # audio/video streams, though the format selector prefers a combined file.
    with tempfile.TemporaryDirectory(prefix="yt-", dir=video_dir) as temp_dir:
        options = {
            # Showcase audio is unused. Prefer a video-only stream because
            # YouTube often does not provide a combined A/V format at 360p.
            "format": (
                "bestvideo[height<=360][vcodec^=avc1][ext=mp4]/"
                "bestvideo[height<=360][vcodec^=avc1]/"
                "bestvideo[height<=360][ext=mp4]/bestvideo[height<=360]/"
                "best[height<=360]/bestvideo/best"
            ),
            "outtmpl": str(Path(temp_dir) / "source.%(ext)s"),
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
            "fragment_retries": 3,
        }
        with YoutubeDL(options) as downloader:
            downloader.download([row["video_url"]])
        files = [p for p in Path(temp_dir).iterdir() if p.is_file() and p.stat().st_size]
        if not files:
            raise RuntimeError("yt-dlp completed without producing a video file")
        source = max(files, key=lambda p: p.stat().st_size)
        # Atomic rename means a failed or interrupted download is never cached.
        staged = destination.with_suffix(".partial")
        source.replace(staged)
        os.replace(staged, destination)
    return destination


class VideoMAERegressor(nn.Module):
    """Frozen VideoMAE plus a temporal attention head for whole-video labels."""

    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        # Encode clip order and relative position, then model interactions
        # between clips before learning which evidence matters for the level.
        self.position_projection = nn.Linear(32, hidden_size)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=1)
        self.attention = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, NUM_CLASSES)

    def train(self, mode: bool = True) -> "VideoMAERegressor":
        super().train(mode)
        # Keep the frozen transformer in inference mode (including any dropout).
        self.encoder.eval()
        return self

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        """Predict a level distribution from ordered [clips, hidden] features."""
        clip_count = clip_features.shape[0]
        positions = torch.linspace(0, 1, clip_count, device=clip_features.device)
        frequencies = 2.0 ** torch.arange(16, device=clip_features.device, dtype=clip_features.dtype)
        angles = positions[:, None] * frequencies[None, :] * (2.0 * math.pi)
        time_encoding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        sequence = clip_features.unsqueeze(0) + self.position_projection(time_encoding).unsqueeze(0)
        sequence = self.temporal_encoder(sequence)
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
        video_feature = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(video_feature)
        return torch.softmax(logits, dim=-1)


def extract_video_features(
    path: Path,
    processor: Any,
    encoder: nn.Module,
    device: torch.device,
    sample_fps: float,
    clip_batch_size: int,
) -> torch.Tensor:
    """Encode consecutive 16-frame clips sampled across the entire video.

    VideoMAE has a fixed 16-frame input. This walks the full video at the
    requested sample rate and encodes every resulting clip. A short final
    clip is padded by repeating its last frame.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to decode videos; install opencv-python") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open downloaded video: {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        frame_stride = max(fps / sample_fps, 1.0)
        next_sample_at = 0.0
        frame_index = 0
        clip: list[Any] = []
        pending_clips: list[list[Any]] = []
        pending_pixels: list[torch.Tensor] = []
        feature_batches: list[torch.Tensor] = []

        def encode_pending() -> None:
            if not pending_clips:
                return
            for frames_in_clip in pending_clips:
                encoded_input = processor(frames_in_clip, return_tensors="pt").pixel_values
                pending_pixels.append(encoded_input.squeeze(0))
            pixel_batch = torch.stack(pending_pixels).to(device)
            with torch.no_grad():
                features = encoder(pixel_values=pixel_batch).last_hidden_state[:, 0]
            feature_batches.append(features)
            pending_clips.clear()
            pending_pixels.clear()

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index + 1e-6 >= next_sample_at:
                clip.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                next_sample_at += frame_stride
                if len(clip) == NUM_FRAMES:
                    pending_clips.append(clip)
                    clip = []
                    if len(pending_clips) >= clip_batch_size:
                        encode_pending()
            frame_index += 1

        if not clip and not pending_clips and not feature_batches:
            raise RuntimeError(f"Video has no decodable frames: {path}")
        if clip:
            clip.extend([clip[-1].copy() for _ in range(NUM_FRAMES - len(clip))])
            pending_clips.append(clip)
        encode_pending()
    finally:
        capture.release()
    return torch.cat(feature_batches, dim=0)


def prefetched_downloads(
    rows: list[dict[str, Any]], video_dir: Path, workers: int, prefetch: int
) -> Iterator[tuple[dict[str, Any], Future[Path]]]:
    """Yield futures in CSV order while bounding work to a small download queue."""
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video-download") as pool:
        pending: dict[int, Future[Path]] = {}
        next_to_submit = 0
        while next_to_submit < min(prefetch, len(rows)):
            pending[next_to_submit] = pool.submit(download_video, rows[next_to_submit], video_dir)
            next_to_submit += 1
        for index, row in enumerate(rows):
            future = pending.pop(index)
            # Submit the next download before yielding this one, so it can run
            # during frame decoding and model training for the current sample.
            if next_to_submit < len(rows):
                pending[next_to_submit] = pool.submit(download_video, rows[next_to_submit], video_dir)
                next_to_submit += 1
            yield row, future


def train(args: argparse.Namespace) -> None:
    try:
        from transformers import VideoMAEImageProcessor, VideoMAEModel
    except ImportError as exc:
        raise RuntimeError("Install transformers to load the pretrained VideoMAE model") from exc

    rows = read_rows(args.csv)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No usable Long rows found in {args.csv}")
    LOG.info("Training on %d Long levels", len(rows))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    encoder = VideoMAEModel.from_pretrained(MODEL_NAME)
    model = VideoMAERegressor(encoder, encoder.config.hidden_size).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
    LOG.info("Using %s; encoder frozen, head has %d outputs", device, NUM_CLASSES)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        trained = 0
        downloads = prefetched_downloads(rows, args.video_dir, args.download_workers, args.prefetch)
        for row, future in downloads:
            try:
                path = future.result()
                clip_features = extract_video_features(
                    path,
                    processor,
                    model.encoder,
                    device,
                    args.sample_fps,
                    args.clip_batch_size,
                )
            except Exception as exc:
                LOG.warning("Skipping %s: %s", row["display_name"], exc)
                continue

            target = row["target"].unsqueeze(0).to(device)
            optimizer.zero_grad(set_to_none=True)
            probabilities = model(clip_features)
            # One probability distribution per complete video, trained
            # against the softmaxed 39-way vote-count target.
            loss = F.kl_div(probabilities.clamp_min(1e-8).log(), target, reduction="batchmean")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            trained += 1
            LOG.info("epoch %d/%d | %d/%d | %s | KL %.5f", epoch, args.epochs, trained, len(rows), row["display_name"], float(loss.detach()))

        if not trained:
            raise RuntimeError("No videos could be downloaded and decoded")
        mean_loss = total_loss / trained
        checkpoint = {
            "model_name": MODEL_NAME,
            "num_classes": NUM_CLASSES,
            "num_frames": NUM_FRAMES,
            "epoch": epoch,
            "mean_kl_loss": mean_loss,
            "temporal_head": {
                "position_projection": model.position_projection.state_dict(),
                "temporal_encoder": model.temporal_encoder.state_dict(),
                "attention": model.attention.state_dict(),
                "classifier": model.classifier.state_dict(),
            },
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, args.checkpoint)
        LOG.info("epoch %d complete | mean KL %.5f | saved %s", epoch, mean_loss, args.checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--download-workers", type=int, default=2)
    parser.add_argument("--prefetch", type=int, default=3, help="maximum queued/in-progress downloads")
    parser.add_argument("--sample-fps", type=float, default=4.0, help="frame sampling rate across each whole video")
    parser.add_argument("--clip-batch-size", type=int, default=4, help="number of 16-frame clips encoded together")
    parser.add_argument("--limit", type=int, help="optional cap for a small run")
    parser.add_argument("--device", help="defaults to CUDA when available, otherwise CPU")
    args = parser.parse_args()
    if args.epochs < 1 or args.download_workers < 1 or args.prefetch < 1 or args.sample_fps <= 0 or args.clip_batch_size < 1:
        parser.error("epochs, download-workers, sample-fps, clip-batch-size, and prefetch must be positive")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(parse_args())
