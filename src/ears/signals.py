"""Audio-motion evidence signals used by EARS."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .video import VideoInfo, decode_audio, read_frames


def candidate_indices(num_frames: int, count: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if count < 2:
        raise ValueError("count must be at least two")
    return np.unique(
        np.linspace(0, num_frames - 1, min(count, num_frames), dtype=np.int64)
    )


def motion_signal(
    path: str | Path,
    info: VideoInfo,
    *,
    candidates: int = 256,
    flow_scale: float = 0.25,
) -> np.ndarray:
    if not 0 < flow_scale <= 1:
        raise ValueError("flow_scale must be in (0, 1]")
    indices = candidate_indices(info.num_frames, candidates)
    frames = read_frames(path, indices, grayscale=True)
    if set(frames) != set(indices.tolist()):
        missing = sorted(set(indices.tolist()) - set(frames))
        raise RuntimeError(f"Failed to decode requested frames: {missing[:8]}")

    sampled = np.zeros(len(indices), dtype=np.float64)
    previous = None
    for position, index in enumerate(indices):
        gray = frames[int(index)]
        if flow_scale != 1:
            gray = cv2.resize(
                gray,
                None,
                fx=flow_scale,
                fy=flow_scale,
                interpolation=cv2.INTER_AREA,
            )
        if previous is not None:
            flow = cv2.calcOpticalFlowFarneback(
                previous,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            sampled[position] = float(np.mean(np.linalg.norm(flow, axis=2)))
        previous = gray
    return np.interp(np.arange(info.num_frames), indices, sampled)


def audio_signal(
    path: str | Path,
    info: VideoInfo,
    *,
    sample_rate: int = 22_050,
    segment_seconds: float = 1.0,
) -> np.ndarray:
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    waveform = decode_audio(path, sample_rate)
    samples_per_segment = max(1, round(sample_rate * segment_seconds))
    count = max(1, (len(waveform) + samples_per_segment - 1) // samples_per_segment)
    energy = np.empty(count, dtype=np.float64)
    for segment in range(count):
        start = segment * samples_per_segment
        stop = min(start + samples_per_segment, len(waveform))
        energy[segment] = float(np.mean(np.abs(waveform[start:stop])))
    centers = (np.arange(count) + 0.5) * segment_seconds
    timestamps = np.arange(info.num_frames) / info.fps
    return np.interp(timestamps, centers, energy)


def normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("Signal must be a nonempty finite vector")
    low = float(array.min())
    high = float(array.max())
    if high <= low:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def evidence_scores(
    audio: np.ndarray,
    motion: np.ndarray,
    *,
    audio_weight: float = 1.0,
    motion_weight: float = 1.0,
) -> np.ndarray:
    if audio.shape != motion.shape:
        raise ValueError("Audio and motion signals must share the frame grid")
    if min(audio_weight, motion_weight) < 0:
        raise ValueError("Evidence weights must be nonnegative")
    raw = audio_weight * normalize(audio) + motion_weight * normalize(motion)
    return 1.0 / (1.0 + np.exp(-raw))
