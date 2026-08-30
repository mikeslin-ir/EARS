"""
Event attention: A_t = sigmoid(alpha * E_norm + beta * M_norm).
E_t and M_t are min-max normalized to [0,1] so alpha/beta set relative influence.
Aligns audio energy (per segment) and motion magnitude (per frame) to a common frame grid.
"""
from __future__ import annotations

import numpy as np


def align_energy_to_frames(
    energy_per_segment: np.ndarray,
    num_frames: int,
) -> np.ndarray:
    """
    Upsample segment-level energy to frame-level (one value per frame).
    Linear interpolation so len(output) == num_frames.
    """
    n_seg = len(energy_per_segment)
    if n_seg <= 0 or num_frames <= 0:
        return np.zeros(max(1, num_frames))
    if n_seg == 1:
        return np.full(num_frames, energy_per_segment[0])
    x_old = np.linspace(0, 1, n_seg)
    x_new = np.linspace(0, 1, num_frames, endpoint=(num_frames == 1))
    return np.interp(x_new, x_old, energy_per_segment).astype(np.float64)


def _min_max_norm(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. If constant, return zeros."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def event_attention(
    E_t: np.ndarray,
    M_t: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """
    A_t = sigmoid(alpha * E_norm + beta * M_norm).
    E_t, M_t are min-max normalized to [0,1] so alpha/beta control relative influence.
    """
    E_norm = _min_max_norm(E_t)
    M_norm = _min_max_norm(M_t)
    raw = alpha * E_norm + beta * M_norm
    return 1.0 / (1.0 + np.exp(-raw))


def compute_event_attention_from_signals(
    energy_per_segment: np.ndarray,
    motion_per_frame: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Align energy to frame grid then compute A_t.
    Returns A_t of length len(motion_per_frame).
    """
    num_frames = len(motion_per_frame)
    E_aligned = align_energy_to_frames(energy_per_segment, num_frames)
    return event_attention(E_aligned, motion_per_frame, alpha=alpha, beta=beta)
