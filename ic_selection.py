"""
Information Compression (IC): greedy NMS-style selection on the
evidence signal A_t with a minimum temporal gap tau (in frames).

Used by the EARS ablation runners. Falls back to plain top-K
when tau <= 0 (matches frame_retrieval.select_top_k_indices).
"""
from __future__ import annotations

import numpy as np


def tau_to_gap(tau_seconds: float, fps: float, stride: int = 1) -> int:
    """Convert an IC gap in seconds to a gap in samples of the score array.

    `stride` is the spacing, in video frames, between consecutive entries of the
    score array: 1 when scores are on the frame grid, and the motion stride when
    they are on the strided signal grid. Both pipelines must go through here.
    They previously each open-coded the arithmetic on their own grid, and one of
    them fed it `get_video_info()[2]` -- the frame width, not the frame rate --
    so a nominal 0.5 s gap became ~13 s on XD-Violence and ~5 s on UCF-Crime,
    silently giving one "global" tau two different meanings.
    """
    if tau_seconds <= 0 or fps <= 0:
        return 0
    if not 1.0 <= fps <= 240.0:
        raise ValueError(
            f"implausible fps {fps!r}; expected a frame rate. Use "
            "video_loader.probe_fps rather than unpacking get_video_info."
        )
    return max(0, int(round(tau_seconds * fps / max(int(stride), 1))))


def select_topk_ic(
    A_t: np.ndarray,
    k: int,
    tau_frames: int = 0,
) -> np.ndarray:
    """
    Greedy selection of up to k indices that maximise A_t under a minimum
    temporal gap tau_frames between chosen indices.

    Algorithm (Alg. 1 in paper):
      while |F| < k and candidates non-empty:
          t* = argmax_{t in candidates} A_t
          F  <- F U {t*}
          remove {t : |t - t*| <= tau} from candidates

    Returns indices sorted in ascending time order (ready for VLM ingestion).
    """
    n = len(A_t)
    k = min(int(k), n)
    if k <= 0:
        return np.array([], dtype=np.int64)
    if tau_frames <= 0:
        # fall back to plain top-K but return in ascending time order
        idx = np.argsort(A_t)[-k:]
        return np.sort(idx).astype(np.int64)

    scores = np.asarray(A_t, dtype=np.float64).copy()
    chosen: list[int] = []
    NEG_INF = -np.inf
    while len(chosen) < k:
        t_star = int(np.argmax(scores))
        if not np.isfinite(scores[t_star]):
            break
        chosen.append(t_star)
        lo = max(0, t_star - int(tau_frames))
        hi = min(n, t_star + int(tau_frames) + 1)
        scores[lo:hi] = NEG_INF
    return np.sort(np.array(chosen, dtype=np.int64))


def select_adaptive_ic(
    A_t: np.ndarray,
    k_max: int,
    tau_frames: int,
    threshold: float,
) -> np.ndarray:
    """
    Adaptive-K variant: keep selecting peaks until either
      (a) k_max frames have been chosen, or
      (b) the next-best candidate has A_t < threshold.

    Used to obtain the frame-budget reduction reported in Sec. 4.6.
    """
    n = len(A_t)
    if n == 0 or k_max <= 0:
        return np.array([], dtype=np.int64)
    scores = np.asarray(A_t, dtype=np.float64).copy()
    chosen: list[int] = []
    while len(chosen) < int(k_max):
        t_star = int(np.argmax(scores))
        if not np.isfinite(scores[t_star]) or scores[t_star] < float(threshold):
            break
        chosen.append(t_star)
        lo = max(0, t_star - int(tau_frames))
        hi = min(n, t_star + int(tau_frames) + 1)
        scores[lo:hi] = -np.inf
    if not chosen:
        # always return at least one frame (the global maximum) to avoid empty VLM input
        chosen = [int(np.argmax(np.asarray(A_t)))]
    return np.sort(np.array(chosen, dtype=np.int64))
