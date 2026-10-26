"""Deterministic temporal peak selection."""
from __future__ import annotations

import numpy as np


def select_peaks(scores: np.ndarray, k: int, min_gap_frames: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("scores must be a nonempty finite vector")
    if k <= 0:
        raise ValueError("k must be positive")
    target = min(int(k), len(values))
    order = np.lexsort((np.arange(len(values)), -values))
    chosen: list[int] = []
    for index in order:
        candidate = int(index)
        if all(abs(candidate - existing) > min_gap_frames for existing in chosen):
            chosen.append(candidate)
            if len(chosen) == target:
                break
    if len(chosen) < target:
        chosen_set = set(chosen)
        chosen.extend(
            int(index)
            for index in order
            if int(index) not in chosen_set
        )
    return np.array(sorted(chosen[:target]), dtype=np.int64)
