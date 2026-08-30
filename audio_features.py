"""
Audio-based event signal: energy E_t per time segment.
E_t = mean(|a_i|) over samples in segment t.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from video_loader import extract_audio_from_video


def compute_energy(waveform: np.ndarray) -> float:
    """Mean absolute amplitude over the whole waveform."""
    return float(np.mean(np.abs(waveform)))


def compute_energy_per_segment(
    waveform: np.ndarray,
    sr: int,
    segment_seconds: float = 1.0,
) -> np.ndarray:
    """
    Compute E_t = mean(|a_i|) for each segment of duration segment_seconds.
    Returns 1D array of length num_segments.
    """
    samples_per_seg = int(sr * segment_seconds)
    if samples_per_seg <= 0:
        return np.array([compute_energy(waveform)])
    n = len(waveform)
    num_seg = (n + samples_per_seg - 1) // samples_per_seg
    out = np.zeros(num_seg)
    for t in range(num_seg):
        start = t * samples_per_seg
        end = min(start + samples_per_seg, n)
        out[t] = np.mean(np.abs(waveform[start:end]))
    return out


def audio_energy_from_video(
    video_path: str | Path,
    sr: int = 22050,
    segment_seconds: float = 1.0,
) -> np.ndarray:
    """
    Load audio from video and return energy per segment E_t.
    """
    y = extract_audio_from_video(video_path, sr=sr)
    return compute_energy_per_segment(y, sr, segment_seconds)
