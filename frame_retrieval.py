"""
Retrieval agent: select top-K frames with highest event attention A_t.
Supports event-driven (by A_t) and uniform sampling (baseline).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from video_loader import load_frame_at_index


def select_top_k_indices(A_t: np.ndarray, k: int) -> np.ndarray:
    """
    Return frame indices (0-based) with highest A_t, in descending order.
    If len(A_t) <= k, returns all indices sorted by A_t descending.
    """
    n = len(A_t)
    k = min(k, n)
    if k <= 0:
        return np.array([], dtype=np.int64)
    # argsort ascending; take last k and reverse
    idx = np.argsort(A_t)[-k:][::-1]
    return idx.astype(np.int64)


def uniform_frame_indices(num_frames: int, k: int) -> np.ndarray:
    """The K segment centres: split the video into K equal parts, take the
    middle frame of each.

    This is the single definition of "uniform" in the repository; the selector
    in ``extensions`` delegates here so the baseline cannot drift between the
    two pipelines.

    It replaces an endpoint-inclusive ``linspace(0, n-1, k)``, which spent the
    budget on the first and last frames -- at K=1 it returned frame 0, the very
    first frame of the clip. On surveillance footage that is usually a fade-in
    or a title card, which understated the Uniform baseline badly: on
    XD-Violence at K=1 it scored F1 0.310 where centre sampling scores 0.641.
    """
    if num_frames <= 0 or k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, num_frames)
    centres = (np.arange(k) + 0.5) * (num_frames / k)
    return np.clip(centres.astype(np.int64), 0, num_frames - 1)


def retrieve_frames_event_driven(
    video_path: str | Path,
    A_t: np.ndarray,
    k: int,
) -> list[tuple[int, np.ndarray]]:
    """
    Load the top-k frames by A_t. Returns list of (frame_index, RGB array).
    """
    indices = select_top_k_indices(A_t, k)
    out = []
    for i in indices:
        frame = load_frame_at_index(video_path, int(i))
        if frame is not None:
            out.append((int(i), frame))
    return out


def retrieve_frames_uniform(
    video_path: str | Path,
    num_frames: int,
    k: int,
) -> list[tuple[int, np.ndarray]]:
    """Load k uniformly sampled frames. Returns list of (frame_index, RGB array)."""
    indices = uniform_frame_indices(num_frames, k)
    out = []
    for i in indices:
        frame = load_frame_at_index(video_path, int(i))
        if frame is not None:
            out.append((int(i), frame))
    return out
