"""Video and audio decoding utilities."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    num_frames: int
    fps: float
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps


def probe_video(path: str | Path) -> VideoInfo:
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        info = VideoInfo(
            num_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()
    if info.num_frames <= 0 or not 1.0 <= info.fps <= 240.0:
        raise ValueError(f"Invalid video metadata: {video_path} ({info})")
    return info


def read_frames(
    path: str | Path,
    indices: Sequence[int],
    *,
    grayscale: bool = False,
) -> dict[int, np.ndarray]:
    wanted = sorted(set(int(index) for index in indices))
    if not wanted:
        return {}
    if wanted[0] < 0:
        raise ValueError("Frame indices must be nonnegative")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames: dict[int, np.ndarray] = {}
    wanted_set = set(wanted)
    try:
        for index in range(wanted[-1] + 1):
            if not capture.grab():
                break
            if index not in wanted_set:
                continue
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                continue
            conversion = cv2.COLOR_BGR2GRAY if grayscale else cv2.COLOR_BGR2RGB
            frames[index] = cv2.cvtColor(frame, conversion)
    finally:
        capture.release()
    return frames


def decode_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Decode mono float32 audio; a verified absent stream becomes silence."""
    video_path = Path(path)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=index", "-of", "json", str(video_path),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    if not json.loads(probe.stdout).get("streams"):
        return np.zeros(1, dtype=np.float32)
    decoded = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path), "-vn",
            "-ar", str(sample_rate), "-ac", "1", "-f", "f32le", "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=300,
    )
    waveform = np.frombuffer(decoded.stdout, dtype="<f4")
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid decoded audio: {video_path}")
    return waveform
