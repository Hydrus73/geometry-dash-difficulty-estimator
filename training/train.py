#!/usr/bin/env python3
"""Train a 39-tier ordinal predictor from cached VideoMAE clip features.

This training path reads the existing feature cache and does not download videos
or run the VideoMAE encoder. Run from any directory with ``python training/train.py``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import math
import os
import random
import re
import stat
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from threading import Event, Lock
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from yt_dlp import YoutubeDL


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "gdladder_demons.csv"
DEFAULT_VIDEO_DIR = ROOT / "training" / "videos"
DEFAULT_FEATURE_DIR = ROOT / "training" / "features"
DEFAULT_CHECKPOINT = ROOT / "training" / "ordinal_transformer.pt"
DEFAULT_LOSS_PLOT = ROOT / "training" / "loss_curve_ordinal.png"
MODEL_NAME = "MCG-NJU/videomae-base"
NUM_CLASSES = 39
NUM_FRAMES = 16
LOG = logging.getLogger("train")


class YouTubeRateLimitError(RuntimeError):
    """Raised when YouTube asks this run to stop making requests."""


class UnavailableVideoError(RuntimeError):
    """Raised when yt-dlp identifies a video as private or unavailable."""


class YtDlpLogCollector:
    """Capture yt-dlp messages so failures can be classified reliably."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def debug(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)


class DownloadStartLimiter:
    """Enforce a minimum delay between uncached video download starts."""

    def __init__(self, delay: float):
        self.delay = delay
        self._lock = Lock()
        self._next_start = 0.0
        self._blocked = Event()

    def wait(self) -> None:
        with self._lock:
            if self._blocked.is_set():
                raise YouTubeRateLimitError("Further downloads stopped after a YouTube rate limit")
            start_at = max(time.monotonic(), self._next_start)
            self._next_start = start_at + self.delay
        if self._blocked.wait(max(0.0, start_at - time.monotonic())):
            raise YouTubeRateLimitError("Further downloads stopped after a YouTube rate limit")

    def block(self) -> None:
        self._blocked.set()


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
            row["_csv_original"] = dict(row)
            row["_csv_line_number"] = line_number
            row["video_url"] = url
            row["_target_vote_count"] = float(sum(counts))
            vote_counts = torch.tensor(counts, dtype=torch.float32)
            # Softmax(log(counts)) preserves the observed vote proportions;
            # softmax(counts) would turn large raw vote totals nearly one-hot.
            log_counts = torch.where(vote_counts > 0, vote_counts.log(), -torch.inf)
            row["target"] = torch.softmax(log_counts, dim=0)
            row["display_name"] = row.get("name") or row.get("level_id") or f"row {line_number}"
            rows.append(row)
    return rows


def remove_unavailable_rows_from_csv(
    csv_path: Path, unavailable_rows: list[dict[str, Any]]
) -> int:
    """Permanently remove confirmed unavailable video rows from the source CSV."""
    def row_identity(row: dict[str, str]) -> tuple[str, str]:
        # Level IDs remain stable if another process edits or reorders the CSV
        # while downloads are running. Include the showcase link to distinguish
        # duplicate IDs that point at different videos.
        level_id = row.get("level_id", "").strip()
        showcase_links = row.get("showcase_links", "").strip()
        if level_id:
            return level_id, showcase_links
        return row.get("name", "").strip(), showcase_links

    unavailable_identities = {
        row_identity(row["_csv_original"]) for row in unavailable_rows
    }
    if not unavailable_identities:
        return 0

    temporary_path: Path | None = None
    source_mode = stat.S_IMODE(csv_path.stat().st_mode)
    removed = 0
    try:
        with csv_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                return 0
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=csv_path.parent,
                prefix=f".{csv_path.name}.",
                suffix=".partial",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                writer = csv.DictWriter(temporary, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    if row_identity(row) in unavailable_identities:
                        removed += 1
                        continue
                    writer.writerow(row)
        if removed:
            os.chmod(temporary_path, source_mode)
            os.replace(temporary_path, csv_path)
            LOG.warning("Removed %d unavailable level(s) from %s", removed, csv_path)
        return removed
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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


def is_youtube_rate_limit(message: str) -> bool:
    lowered = message.lower()
    return (
        "rate limit" in lowered
        or "rate-limited" in lowered
        or "this content isn't available, try again later" in lowered
    )


def is_permanently_unavailable(message: str) -> bool:
    normalized = re.sub(r"\s+", " ", message.lower())
    patterns = (
        r"\bprivate video\b",
        r"\bvideo (?:is )?private\b",
        r"\bvideo (?:is )?unavailable\b",
        r"\bvideo (?:isn't|is not) available\b",
        r"\bvideo has been removed\b",
        r"\bvideo was removed\b",
        r"\bvideo has been deleted\b",
        r"\baccount associated with this video has been terminated\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def download_video(
    row: dict[str, Any],
    video_dir: Path,
    cookies_file: Path | None = None,
    cookies_from_browser: str | None = None,
    start_limiter: DownloadStartLimiter | None = None,
) -> Path:
    """Download the best available video stream at or below 360p."""
    video_dir.mkdir(parents=True, exist_ok=True)
    destination = video_path(row, video_dir)
    if destination.is_file() and destination.stat().st_size:
        if cached_video_is_decodable(destination):
            return destination
        LOG.warning("Cached video %s cannot be decoded; downloading it again", destination)

    if start_limiter is not None:
        start_limiter.wait()

    # Download into a per-video temporary directory to avoid exposing partial
    # files as cache hits. A merge is requested when yt-dlp selects separate
    # audio/video streams, though the format selector prefers a combined file.
    ytdlp_log = YtDlpLogCollector()
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
            "logger": ytdlp_log,
            # This script displays one tqdm bar for all worker downloads.
            # yt-dlp's own per-download bars otherwise redraw over it.
            "noprogress": True,
            "sleep_interval_requests": 1,
            "retries": 3,
            "fragment_retries": 3,
        }
        if cookies_file is not None:
            options["cookiefile"] = str(cookies_file.expanduser())
        if cookies_from_browser is not None:
            options["cookiesfrombrowser"] = (cookies_from_browser,)
        try:
            with YoutubeDL(options) as downloader:
                downloader.download([row["video_url"]])
        except Exception as exc:
            message = "\n".join((str(exc), *ytdlp_log.errors, *ytdlp_log.warnings))
            if start_limiter is not None and is_youtube_rate_limit(message):
                start_limiter.block()
                raise YouTubeRateLimitError(message) from exc
            if is_permanently_unavailable(message):
                raise UnavailableVideoError(message) from exc
            raise
        files = [p for p in Path(temp_dir).iterdir() if p.is_file() and p.stat().st_size]
        if not files:
            message = "\n".join((*ytdlp_log.errors, *ytdlp_log.warnings))
            if start_limiter is not None and is_youtube_rate_limit(message):
                start_limiter.block()
                raise YouTubeRateLimitError(message)
            if is_permanently_unavailable(message):
                raise UnavailableVideoError(message)
            raise RuntimeError(message or "yt-dlp completed without producing a video file")
        source = max(files, key=lambda p: p.stat().st_size)
        # Atomic rename means a failed or interrupted download is never cached.
        staged = destination.with_suffix(".partial")
        source.replace(staged)
        os.replace(staged, destination)
    return destination


class VideoMAERegressor(nn.Module):
    """Transformer over cached VideoMAE clip features, predicting tier probabilities."""

    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        head_type: str = "attention_pool",
        dropout: float = 0.25,
        transformer_layers: int = 2,
        target_mode: str = "distribution",
        ordinal_link: str = "softmax",
        input_feature_size: int = 768,
    ):
        super().__init__()
        self.head_type = head_type
        self.target_mode = target_mode
        self.ordinal_link = ordinal_link
        output_size = 1 if target_mode == "expected_tier" else NUM_CLASSES
        classifier_size = 1 if ordinal_link == "coral" and target_mode != "expected_tier" else output_size
        self.register_buffer("projection_mean", torch.zeros(input_feature_size))
        self.register_buffer("projection_components", torch.zeros(input_feature_size, feature_size))
        self.register_buffer("feature_mean", torch.zeros(feature_size))
        self.register_buffer("feature_std", torch.ones(feature_size))
        if ordinal_link == "coral" and target_mode != "expected_tier":
            self.ordinal_threshold_base = nn.Parameter(torch.tensor(0.0))
            self.ordinal_threshold_steps = nn.Parameter(torch.full((NUM_CLASSES - 2,), 0.5413))
        if head_type == "attention_pool":
            self.input_projection = nn.Identity()
            self.position_projection = nn.Identity()
            self.temporal_encoder = nn.Identity()
            self.attention = nn.Linear(feature_size + 1, 1)
            self.classifier = nn.Linear(feature_size + 1, classifier_size)
        elif head_type == "temporal_transformer":
            self.input_projection = nn.Linear(feature_size, hidden_size)
            self.position_projection = nn.Linear(32, hidden_size)
            temporal_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                temporal_layer, num_layers=transformer_layers
            )
            self.attention = nn.Linear(hidden_size, 1)
            self.classifier = nn.Linear(hidden_size, classifier_size)
        elif head_type == "pooled_mlp":
            # Four coarse temporal bins preserve video order without learning
            # pairwise attention between every clip.
            self.input_projection = nn.Linear(feature_size * 4, hidden_size)
            self.position_projection = nn.Identity()
            self.temporal_encoder = nn.Sequential(
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.attention = nn.Identity()
            self.classifier = nn.Linear(hidden_size, classifier_size)
        elif head_type == "statistics_mlp":
            # Mean appearance, temporal variability, mean adjacent change, and
            # the strongest channel response form a compact order-free summary.
            self.input_projection = nn.Linear(feature_size * 4, hidden_size)
            self.position_projection = nn.Identity()
            self.temporal_encoder = nn.Sequential(
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.attention = nn.Identity()
            self.classifier = nn.Linear(hidden_size, classifier_size)
        else:
            raise ValueError(f"Unknown temporal head type: {head_type}")
        self.dropout = nn.Dropout(dropout)

    def train(self, mode: bool = True) -> "VideoMAERegressor":
        super().train(mode)
        return self

    def initialize_ordinal_thresholds(self, targets: torch.Tensor) -> None:
        """Initialize CORAL cutpoints from training-label frequencies."""
        if self.ordinal_link != "coral" or self.target_mode == "expected_tier":
            return
        survival = 1 - targets.mean(dim=0).cumsum(dim=0)[:-1]
        survival = survival.clamp(1e-4, 1 - 1e-4)
        thresholds = torch.logit(1 - survival)
        increments = (thresholds[1:] - thresholds[:-1]).clamp_min(1e-3)
        inverse_softplus = torch.log(torch.expm1(increments).clamp_min(1e-8))
        with torch.no_grad():
            self.ordinal_threshold_base.copy_(thresholds[0].to(self.ordinal_threshold_base))
            self.ordinal_threshold_steps.copy_(inverse_softplus.to(self.ordinal_threshold_steps))

    def classify(self, representation: torch.Tensor) -> torch.Tensor:
        scores = self.classifier(self.dropout(representation))
        if self.ordinal_link != "coral" or self.target_mode == "expected_tier":
            return scores
        steps = F.softplus(self.ordinal_threshold_steps)
        thresholds = torch.cat((self.ordinal_threshold_base[None], self.ordinal_threshold_base + steps.cumsum(0)))
        survival = torch.sigmoid(scores - thresholds[None, :])
        probabilities = torch.cat(
            (1 - survival[:, :1], survival[:, :-1] - survival[:, 1:], survival[:, -1:]),
            dim=-1,
        ).clamp_min(1e-8)
        return probabilities.log()

    def forward_logits(self, clip_features: torch.Tensor) -> torch.Tensor:
        """Return class logits or an expected-tier estimate from ordered clips."""
        if self.projection_components.shape[1] != clip_features.shape[1]:
            clip_features = (clip_features - self.projection_mean) @ self.projection_components
        clip_features = (clip_features - self.feature_mean) / self.feature_std
        clip_count = clip_features.shape[0]
        if self.head_type == "pooled_mlp":
            bins = F.adaptive_avg_pool1d(clip_features.T.unsqueeze(0), output_size=4)
            pooled = bins.squeeze(0).reshape(1, -1)
            hidden = self.input_projection(pooled)
            hidden = self.temporal_encoder(hidden)
            return self.classify(hidden)
        if self.head_type == "statistics_mlp":
            mean = clip_features.mean(dim=0)
            variability = clip_features.std(dim=0, unbiased=False)
            change = (
                (clip_features[1:] - clip_features[:-1]).abs().mean(dim=0)
                if clip_count > 1
                else torch.zeros_like(mean)
            )
            peak = clip_features.amax(dim=0)
            pooled = torch.cat((mean, variability, change, peak), dim=0).unsqueeze(0)
            hidden = self.temporal_encoder(self.input_projection(pooled))
            return self.classify(hidden)
        if self.head_type == "attention_pool":
            positions = torch.linspace(
                0, 1, clip_count, device=clip_features.device, dtype=clip_features.dtype
            )
            scoring_features = torch.cat((clip_features, positions[:, None]), dim=-1)
            weights = torch.softmax(self.attention(scoring_features).squeeze(-1), dim=0)
            video_feature = torch.sum(clip_features * weights[:, None], dim=0)
            weighted_position = torch.sum(positions * weights).unsqueeze(0)
            pooled = torch.cat((video_feature, weighted_position), dim=0).unsqueeze(0)
            return self.classify(pooled)

        positions = torch.linspace(0, 1, clip_count, device=clip_features.device)
        frequencies = 2.0 ** torch.arange(16, device=clip_features.device, dtype=clip_features.dtype)
        angles = positions[:, None] * frequencies[None, :] * (2.0 * math.pi)
        time_encoding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        sequence = self.input_projection(clip_features).unsqueeze(0)
        sequence = sequence + self.position_projection(time_encoding).unsqueeze(0)
        sequence = self.temporal_encoder(sequence)
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
        video_feature = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.classify(video_feature)

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        """Predict a tier distribution or scalar tier from ordered clips."""
        output = self.forward_logits(clip_features)
        if self.target_mode == "expected_tier":
            return output.squeeze(-1).clamp(1, NUM_CLASSES)
        return torch.softmax(output, dim=-1)


def ordinal_distribution_loss(
    logits: torch.Tensor, target: torch.Tensor, loss_type: str = "ordinal_ce"
) -> torch.Tensor:
    """Compare cumulative predicted and observed rating distributions."""
    probabilities = torch.softmax(logits, dim=-1)
    if loss_type == "categorical_ce":
        return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")
    predicted_cdf = probabilities.cumsum(dim=-1)[..., :-1]
    target_cdf = target.cumsum(dim=-1)[..., :-1].clamp(0, 1)
    if loss_type == "rps":
        return F.mse_loss(predicted_cdf, target_cdf)
    if loss_type == "wasserstein":
        return (predicted_cdf - target_cdf).abs().sum(dim=-1).mean()
    predicted_cdf = predicted_cdf.clamp(1e-7, 1 - 1e-7)
    return F.binary_cross_entropy(predicted_cdf, target_cdf)


def subsample_clips(clip_features: torch.Tensor, keep_probability: float) -> torch.Tensor:
    """Randomly drop clips while preserving their original order."""
    if keep_probability >= 1 or clip_features.shape[0] <= 2:
        return clip_features
    keep_count = min(
        clip_features.shape[0],
        max(2, round(clip_features.shape[0] * keep_probability)),
    )
    selected = torch.randperm(clip_features.shape[0], device=clip_features.device)[:keep_count]
    return clip_features[selected.sort().values]


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


def download_videos(
    rows: list[dict[str, Any]],
    csv_path: Path,
    video_dir: Path,
    workers: int,
    download_delay: float,
    cookies_file: Path | None,
    cookies_from_browser: str | None,
) -> list[tuple[dict[str, Any], Path]]:
    """Download/cache all videos concurrently and return successful paths."""
    downloaded: list[tuple[dict[str, Any], Path]] = []
    unavailable_rows: list[dict[str, Any]] = []
    failures = 0
    rate_limiter = DownloadStartLimiter(download_delay)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video-download") as pool:
        futures = {
            pool.submit(
                download_video, row, video_dir, cookies_file, cookies_from_browser, rate_limiter
            ): row
            for row in rows
        }
        with tqdm(total=len(futures), desc="Downloading videos", unit="video") as progress:
            rate_limited = False
            for future in as_completed(futures):
                row = futures[future]
                try:
                    downloaded.append((row, future.result()))
                except YouTubeRateLimitError as exc:
                    rate_limited = True
                    rate_limiter.block()
                    LOG.error("Stopping downloads after YouTube rate limit: %s", exc)
                except UnavailableVideoError as exc:
                    failures += 1
                    unavailable_rows.append(row)
                    LOG.warning(
                        "Removing unavailable/private level %s from the CSV: %s",
                        row["display_name"], exc,
                    )
                except Exception as exc:
                    failures += 1
                    LOG.warning("Download failed for %s: %s", row["display_name"], exc)
                    if "sign in to confirm" in str(exc).lower() or "not a bot" in str(exc).lower():
                        LOG.warning(
                            "YouTube requires login cookies; rerun with --cookies PATH or "
                            "--cookies-from-browser BROWSER (for example chrome)"
                        )
                progress.update(1)
                progress.set_postfix(cached_or_downloaded=len(downloaded), failed=failures)
    remove_unavailable_rows_from_csv(csv_path, unavailable_rows)
    if rate_limited:
        raise RuntimeError(
            "YouTube rate-limited this session. Downloads were stopped; wait for the limit to "
            "expire before retrying. Successfully downloaded videos remain cached."
        )
    return downloaded


def save_loss_plot(
    train_losses: list[float], validation_losses: list[float], destination: Path
) -> None:
    """Write the training and validation ordinal loss history as a headless plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_losses, marker="o", label="Training objective")
    axis.plot(epochs, validation_losses, marker="o", label="Validation objective")
    axis.set(title="Training and validation objective", xlabel="Epoch", ylabel="Loss")
    axis.set_xticks(list(epochs))
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def feature_cache_identity(
    video: Path,
    sample_fps: float,
    hidden_size: int,
    model_revision: str | None,
) -> dict[str, Any]:
    """Describe inputs that affect the cached encoder features."""
    stat = video.stat()
    return {
        "model_name": MODEL_NAME,
        "model_revision": model_revision,
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "sample_fps": sample_fps,
        "num_frames": NUM_FRAMES,
        "hidden_size": hidden_size,
    }


def load_feature_cache(
    cache_path: Path, identity: dict[str, Any]
) -> torch.Tensor | None:
    """Load a matching CPU feature cache, ignoring corrupt or stale entries."""
    if not cache_path.is_file():
        return None
    try:
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        features = cached.get("features")
        if (
            cached.get("identity") == identity
            and isinstance(features, torch.Tensor)
            and features.ndim == 2
            and features.shape[0] > 0
            and features.shape[1] == identity["hidden_size"]
        ):
            return features
    except Exception as exc:
        LOG.warning("Could not read feature cache %s; recomputing it: %s", cache_path, exc)
    return None


def save_feature_cache(cache_path: Path, identity: dict[str, Any], features: torch.Tensor) -> None:
    """Write cached features atomically so interrupted runs leave no partial cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{cache_path.name}.", suffix=".partial", dir=cache_path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save({"identity": identity, "features": features.cpu()}, temporary_path)
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def evaluate_loss(
    model: VideoMAERegressor,
    examples: list[tuple[dict[str, Any], torch.Tensor]],
    device: torch.device,
    description: str,
    distribution_loss_type: str,
    target_mode: str,
    hard_label_mix: float,
) -> tuple[float, float, float, float, float]:
    """Measure distribution loss, tier MAE, exact accuracy, and within-tier accuracy."""
    total_loss = 0.0
    total_mae = 0.0
    correct = 0
    within_one = 0
    within_three = 0
    class_values = torch.arange(1, NUM_CLASSES + 1, device=device, dtype=torch.float32)
    with torch.no_grad(), tqdm(examples, desc=description, unit="video") as progress:
        for index, (row, clip_features) in enumerate(progress, start=1):
            logits = model.forward_logits(clip_features.to(device))
            target = row["target"].unsqueeze(0).to(device)
            target_difficulty = torch.sum(target * class_values, dim=-1)
            if target_mode == "expected_tier":
                predicted_difficulty = logits.squeeze(-1).clamp(1, NUM_CLASSES)
                loss = F.smooth_l1_loss(predicted_difficulty, target_difficulty)
                predicted_class = predicted_difficulty.round().long() - 1
            else:
                loss_target = target
                mix = 1.0 if target_mode == "hard_class" else hard_label_mix
                if mix:
                    hard_target = F.one_hot(target.argmax(dim=-1), NUM_CLASSES).float()
                    loss_target = target * (1 - mix) + hard_target * mix
                loss = ordinal_distribution_loss(logits, loss_target, distribution_loss_type)
                probabilities = torch.softmax(logits, dim=-1)
                predicted_difficulty = torch.sum(probabilities * class_values, dim=-1)
                predicted_class = probabilities.argmax(dim=-1)
            total_loss += float(loss)
            total_mae += float(torch.abs(predicted_difficulty - target_difficulty).mean())
            correct += int(predicted_class.eq(target.argmax(dim=-1)).sum())
            difference = torch.abs(predicted_difficulty - target_difficulty)
            within_one += int((difference <= 1).sum())
            within_three += int((difference <= 3).sum())
            progress.set_postfix(
                loss=f"{float(loss):.4f}",
                mean=f"{total_loss / index:.4f}",
                mae=f"{total_mae / index:.2f}",
                acc=f"{correct / index:.1%}",
            )
    return (
        total_loss / len(examples),
        total_mae / len(examples),
        correct / len(examples),
        within_one / len(examples),
        within_three / len(examples),
    )


def train(args: argparse.Namespace) -> None:
    rows = read_rows(args.csv)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No usable Long rows found in {args.csv}")
    if len(rows) < 3:
        raise RuntimeError("At least three usable levels are required for an 80/10/10 train/validation/test split")
    LOG.info("Preparing %d Long levels from cached %s features", len(rows), args.feature_model)

    initialization_seed = args.seed if args.model_seed is None else args.model_seed
    torch.manual_seed(initialization_seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    feature_dim = args.pca_components or args.input_feature_size
    model = VideoMAERegressor(
        feature_dim,
        args.temporal_hidden_size,
        args.head_type,
        args.dropout,
        args.transformer_layers,
        args.target_mode,
        args.ordinal_link,
        args.input_feature_size,
    ).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    tier_values = torch.arange(1, NUM_CLASSES + 1, device=device, dtype=torch.float32)
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    LOG.info(
        "Using %s; cached features %dD, %s head with %d trainable parameters",
        device, args.input_feature_size, args.head_type, trainable_count,
    )

    # Cached clip embeddings are the only model inputs in this experiment.
    # This avoids downloads and recomputing VideoMAE for already encoded videos.
    examples: list[tuple[dict[str, Any], torch.Tensor]] = []
    failures = 0
    cache_hits = 0
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    with tqdm(rows, desc="Loading cached features", unit="video") as progress:
        for row in progress:
            try:
                cache_path = args.feature_dir / f"{row['level_id']}.pt"
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                features = cached.get("features") if isinstance(cached, dict) else cached
                if (
                    not isinstance(features, torch.Tensor)
                    or features.ndim != 2
                    or features.shape[0] == 0
                    or features.shape[1] != args.input_feature_size
                    or not torch.isfinite(features).all()
                ):
                    raise ValueError(
                        f"expected a finite [clips, {args.input_feature_size}] feature tensor"
                    )
                features = features.float()
                cache_hits += 1
                examples.append((row, features))
            except Exception as exc:
                failures += 1
                LOG.warning("Skipping %s; cached features unavailable: %s", row["display_name"], exc)
            progress.set_postfix(loaded=len(examples), cache_hits=cache_hits, failed=failures)

    if len(examples) < 3:
        raise RuntimeError("Fewer than three cached feature files are usable; cannot split the data")
    LOG.info("Loaded %d cached feature files; skipped %d", cache_hits, failures)

    random.Random(args.seed).shuffle(examples)
    test_count = max(1, math.ceil(len(examples) * 0.1))
    validation_count = max(1, math.ceil(len(examples) * 0.1))
    test_examples = examples[:test_count]
    validation_examples = examples[test_count:test_count + validation_count]
    train_examples = examples[test_count + validation_count:]
    model.initialize_ordinal_thresholds(
        torch.stack([row["target"] for row, _ in train_examples])
    )
    LOG.info(
        "Split %d usable videos into %d training, %d validation, and %d held-out test "
        "(approximately 80/10/10)",
        len(examples), len(train_examples), len(validation_examples), len(test_examples),
    )

    if args.pca_components:
        training_features = torch.cat([features for _, features in train_examples], dim=0).to(device)
        projection_mean = training_features.mean(dim=0)
        centered = training_features - projection_mean
        covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        projection_components = eigenvectors[:, -args.pca_components:].flip(1).contiguous()
        model.projection_mean.copy_(projection_mean)
        model.projection_components.copy_(projection_components)
        projection_mean_cpu = projection_mean.cpu()
        projection_components_cpu = projection_components.cpu()
        examples = [
            (row, (features - projection_mean_cpu) @ projection_components_cpu)
            for row, features in examples
        ]
        test_examples = examples[:test_count]
        validation_examples = examples[test_count:test_count + validation_count]
        train_examples = examples[test_count + validation_count:]
        del training_features, centered, covariance, eigenvalues, eigenvectors
        LOG.info("Projected cached clips to %d PCA components", args.pca_components)

    # Standardize each cached feature channel from training clips only.
    feature_sum = torch.zeros(feature_dim, dtype=torch.float64)
    feature_square_sum = torch.zeros(feature_dim, dtype=torch.float64)
    feature_count = 0
    for _, clip_features in train_examples:
        values = clip_features.double()
        feature_sum += values.sum(dim=0)
        feature_square_sum += values.square().sum(dim=0)
        feature_count += values.shape[0]
    feature_mean = feature_sum / feature_count
    feature_variance = (feature_square_sum / feature_count - feature_mean.square()).clamp_min(1e-8)
    model.feature_mean.copy_(feature_mean.float().to(device))
    model.feature_std.copy_(feature_variance.sqrt().float().to(device))
    LOG.info("Standardized cached features using %d training clips", feature_count)
    if args.vote_count_weight:
        vote_counts = torch.tensor(
            [row["_target_vote_count"] for row, _ in train_examples], dtype=torch.float64
        )
        vote_weights = (vote_counts / vote_counts.mean()).pow(args.vote_count_weight)
        vote_weights /= vote_weights.mean()
        for (row, _), weight in zip(train_examples, vote_weights):
            row["_loss_weight"] = float(weight)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_losses: list[float] = []
    validation_losses: list[float] = []
    train_mae_history: list[float] = []
    validation_mae_history: list[float] = []
    best_validation_loss = math.inf
    best_validation_mae = math.inf
    best_epoch = 0
    early_stopping_patience = args.early_stopping_patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.Random(args.seed + epoch).shuffle(train_examples)
        with tqdm(train_examples, desc=f"Epoch {epoch}/{args.epochs} train", unit="video") as progress:
            for row, clip_features in progress:
                clip_features = clip_features.to(device)
                clip_features = subsample_clips(clip_features, args.clip_keep_probability)
                target = row["target"].unsqueeze(0).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model.forward_logits(clip_features)
                if args.target_mode == "expected_tier":
                    target_tier = torch.sum(target * tier_values, dim=-1)
                    predicted_tier = logits.squeeze(-1)
                    loss = F.smooth_l1_loss(predicted_tier, target_tier)
                else:
                    loss_target = target
                    mix = 1.0 if args.target_mode == "hard_class" else args.hard_label_mix
                    if mix:
                        hard_target = F.one_hot(target.argmax(dim=-1), NUM_CLASSES).float()
                        loss_target = target * (1 - mix) + hard_target * mix
                    loss = ordinal_distribution_loss(logits, loss_target, args.distribution_loss)
                if args.target_mode == "distribution" and args.expected_tier_loss_weight:
                    predicted_tier = torch.sum(torch.softmax(logits, dim=-1) * tier_values, dim=-1)
                    target_tier = torch.sum(target * tier_values, dim=-1)
                    tier_loss = F.smooth_l1_loss(predicted_tier, target_tier)
                    loss = loss + args.expected_tier_loss_weight * tier_loss
                loss = loss * row.get("_loss_weight", 1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                optimizer.step()
                progress.set_postfix(online_loss=f"{float(loss.detach()):.4f}")

        model.eval()
        mean_loss, mean_train_mae, mean_train_accuracy, mean_train_within_one, mean_train_within_three = evaluate_loss(
            model, train_examples, device, f"Epoch {epoch}/{args.epochs} train eval",
            args.distribution_loss, args.target_mode, args.hard_label_mix
        )
        mean_validation_loss, mean_validation_mae, mean_validation_accuracy, validation_within_one, validation_within_three = evaluate_loss(
            model, validation_examples, device, f"Epoch {epoch}/{args.epochs} validation eval",
            args.distribution_loss, args.target_mode, args.hard_label_mix
        )
        train_losses.append(mean_loss)
        validation_losses.append(mean_validation_loss)
        train_mae_history.append(mean_train_mae)
        validation_mae_history.append(mean_validation_mae)

        # The project target is tier error; retain the checkpoint with lowest
        # validation expected-tier MAE while reporting ordinal loss as a
        # distribution-quality metric.
        improved = mean_validation_mae < best_validation_mae
        if improved:
            best_validation_loss = mean_validation_loss
            best_validation_mae = mean_validation_mae
            best_epoch = epoch
            checkpoint = {
                "model_name": args.feature_model,
                "num_classes": NUM_CLASSES,
                "num_frames": NUM_FRAMES,
                "sample_fps": args.sample_fps,
                "frames_per_feature": NUM_FRAMES,
                "feature_size": feature_dim,
                "input_feature_size": args.input_feature_size,
                "pca_components": args.pca_components,
                "temporal_hidden_size": args.temporal_hidden_size,
                "head_type": args.head_type,
                "dropout": args.dropout,
                "clip_keep_probability": args.clip_keep_probability,
                "transformer_layers": args.transformer_layers,
                "weight_decay": args.weight_decay,
                "early_stopping_patience": args.early_stopping_patience,
                "expected_tier_loss_weight": args.expected_tier_loss_weight,
                "distribution_loss": args.distribution_loss,
                "target_mode": args.target_mode,
                "ordinal_link": args.ordinal_link,
                "hard_label_mix": args.hard_label_mix,
                "vote_count_weight": args.vote_count_weight,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "mean_objective_loss": mean_loss,
                "mean_validation_objective_loss": mean_validation_loss,
                "train_loss_history": train_losses,
                "validation_loss_history": validation_losses,
                "train_expected_difficulty_mae_history": train_mae_history,
                "validation_expected_difficulty_mae_history": validation_mae_history,
                "seed": args.seed,
                "model_seed": initialization_seed,
                "temporal_head": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, args.checkpoint)
        save_loss_plot(train_losses, validation_losses, args.loss_plot)
        LOG.info(
            "epoch %d/%d | train objective %.5f (MAE %.2f, exact %.1f%%, <=3 %.1f%%) | "
            "validation objective %.5f (MAE %.2f, exact %.1f%%, <=1 %.1f%%, <=3 %.1f%%) | "
            "best epoch %d | %s | plot %s",
            epoch, args.epochs, mean_loss, mean_train_mae, 100 * mean_train_accuracy, 100 * mean_train_within_three,
            mean_validation_loss, mean_validation_mae, 100 * mean_validation_accuracy,
            100 * validation_within_one, 100 * validation_within_three,
            best_epoch, "checkpoint updated" if improved else "checkpoint unchanged",
            args.loss_plot,
        )
        if epoch - best_epoch >= early_stopping_patience:
            LOG.info("Stopping after %d epochs without validation improvement", early_stopping_patience)
            break
    # Select the epoch using validation data, then evaluate the untouched test
    # split once so the reported test score is not used for model selection.
    best_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(best_checkpoint["temporal_head"])
    model.eval()
    heldout_test_loss, heldout_test_mae, heldout_test_accuracy, heldout_test_within_one, heldout_test_within_three = evaluate_loss(
        model, test_examples, device, "Held-out test eval",
        args.distribution_loss, args.target_mode, args.hard_label_mix
    )
    best_checkpoint.update({
        "heldout_test_objective_loss": heldout_test_loss,
        "heldout_test_expected_difficulty_mae": heldout_test_mae,
        "heldout_test_top_class_accuracy": heldout_test_accuracy,
        "heldout_test_within_one_tier": heldout_test_within_one,
        "heldout_test_within_three_tiers": heldout_test_within_three,
    })
    torch.save(best_checkpoint, args.checkpoint)
    LOG.info(
        "held-out test (best validation epoch %d) | objective loss %.5f | expected-difficulty "
        "MAE %.2f | exact %.1f%% | within 1 tier %.1f%% | within 3 tiers %.1f%%",
        best_epoch, heldout_test_loss, heldout_test_mae, 100 * heldout_test_accuracy,
        100 * heldout_test_within_one, 100 * heldout_test_within_three,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-model", default=MODEL_NAME, help="name of the pretrained model used for cached features")
    parser.add_argument("--input-feature-size", type=int, default=768, help="dimension of each cached feature vector")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--loss-plot", type=Path, default=DEFAULT_LOSS_PLOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--pca-components",
        type=int,
        default=0,
        help="optional PCA projection dimension for cached clip features",
    )
    parser.add_argument(
        "--temporal-hidden-size",
        type=int,
        default=224,
        help="Transformer width (must be divisible by 8; default head is about 1M parameters)",
    )
    parser.add_argument(
        "--head-type",
        choices=("attention_pool", "temporal_transformer", "pooled_mlp", "statistics_mlp"),
        default="temporal_transformer",
        help="weighted pooling head or the temporal Transformer head",
    )
    parser.add_argument("--dropout", type=float, default=0.4, help="dropout probability in the trainable head")
    parser.add_argument(
        "--clip-keep-probability",
        type=float,
        default=1.0,
        help="fraction of ordered cached clips retained per video during training (1 disables temporal clip augmentation)",
    )
    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=2,
        help="number of temporal Transformer layers",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="stop after this many epochs without lower validation tier MAE",
    )
    parser.add_argument(
        "--expected-tier-loss-weight",
        type=float,
        default=0.0,
        help="auxiliary smooth-L1 weight for the expected tier; ordinal loss remains the distribution objective",
    )
    parser.add_argument(
        "--distribution-loss",
        choices=("ordinal_ce", "rps", "wasserstein", "categorical_ce"),
        default="ordinal_ce",
        help="cumulative ordinal CE, ranked probability score, Wasserstein distance, or categorical cross entropy",
    )
    parser.add_argument(
        "--target-mode",
        choices=("distribution", "expected_tier", "hard_class"),
        default="distribution",
        help="predict the full rating distribution or regress its vote-weighted mean tier",
    )
    parser.add_argument(
        "--ordinal-link",
        choices=("softmax", "coral"),
        default="softmax",
        help="independent class logits or a cumulative ordinal link with ordered thresholds",
    )
    parser.add_argument(
        "--hard-label-mix",
        type=float,
        default=0.0,
        help="blend this fraction of the vote majority label into the soft target distribution",
    )
    parser.add_argument(
        "--vote-count-weight",
        type=float,
        default=0.0,
        help="loss-weight exponent for per-level vote counts; 0 disables confidence weighting",
    )
    parser.add_argument("--download-workers", type=int, default=1, help="number of videos to download in parallel")
    parser.add_argument(
        "--download-delay",
        type=float,
        default=7.5,
        help="minimum seconds between uncached video download starts (default: 7.5)",
    )
    cookie_options = parser.add_mutually_exclusive_group()
    cookie_options.add_argument(
        "--cookies",
        type=Path,
        help="Netscape-format cookies file exported from a YouTube-logged-in browser",
    )
    cookie_options.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        help="read YouTube login cookies from a browser, such as chrome, firefox, or edge",
    )
    parser.add_argument("--sample-fps", type=float, default=4.0, help="frame sampling rate across each whole video")
    parser.add_argument("--clip-batch-size", type=int, default=4, help="number of 16-frame clips encoded together")
    parser.add_argument("--limit", type=int, help="optional cap for a small run")
    parser.add_argument("--device", help="defaults to CUDA when available, otherwise CPU")
    parser.add_argument("--seed", type=int, default=42, help="seed for the train/test split and sample order")
    parser.add_argument("--model-seed", type=int, help="optional separate initialization seed for ensembling runs")
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.download_workers < 1
        or args.download_delay < 0
        or args.sample_fps <= 0
        or args.clip_batch_size < 1
        or args.temporal_hidden_size < 8
        or args.temporal_hidden_size % 8 != 0
        or args.transformer_layers < 1
        or args.weight_decay < 0
        or args.early_stopping_patience < 1
        or not 0 <= args.dropout < 1
        or not 0 < args.clip_keep_probability <= 1
        or args.expected_tier_loss_weight < 0
        or not 0 <= args.hard_label_mix <= 1
        or args.vote_count_weight < 0
        or args.pca_components < 0
        or args.input_feature_size < 1
        or args.pca_components >= args.input_feature_size
        or (args.ordinal_link == "coral" and args.target_mode == "expected_tier")
    ):
        parser.error(
            "epochs, download-workers, sample-fps, and clip-batch-size must be positive; "
            "download-delay must be nonnegative; temporal-hidden-size must be a positive "
            "multiple of 8; dropout must be in [0, 1); expected-tier-loss-weight must be nonnegative"
        )
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(parse_args())
