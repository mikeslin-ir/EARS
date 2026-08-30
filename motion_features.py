"""
Motion-based event signal: magnitude M_t = mean(||Flow_t(p)||) per frame.
Uses OpenCV optical flow (Farneback) between consecutive frames.
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

from video_loader import (
    get_video_info,
    iter_frames,
    load_video_capture,
    read_frames_at_indices,
)


def compute_optical_flow_magnitude(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """
    prev, curr: RGB (H, W, 3). Returns per-pixel flow magnitude (H, W).
    """
    g0 = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g0, g1, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return mag


def motion_magnitude_per_frame(path: str | Path) -> np.ndarray:
    """
    Compute M_t = mean(||Flow_t(p)||) for each frame t (between t-1 and t).
    First frame has magnitude 0 (no previous frame).
    Returns 1D array of length num_frames.
    """
    frames = list(iter_frames(path))
    n = len(frames)
    if n <= 1:
        return np.zeros(max(1, n))
    out = np.zeros(n)
    out[0] = 0.0
    for t in range(1, n):
        mag = compute_optical_flow_magnitude(frames[t - 1], frames[t])
        out[t] = float(np.mean(mag))
    return out


def motion_magnitude_downsampled(
    path: str | Path,
    stride: int = 5,
    max_frames: int | None = 1000,
    flow_scale: float = 1.0,
) -> np.ndarray:
    """
    Compute motion magnitude on every `stride`-th frame to save time.
    Returns magnitudes for indices 0, stride, 2*stride, ... (first is 0).

    `flow_scale` shrinks each frame before the Farneback solve, whose cost is
    linear in pixel count. It defaults to 1.0 so cached signals stay valid;
    see `bench_motion_proxies.py` for the cost/agreement trade-off.
    """
    cap = load_video_capture(path)
    n, _, _, _ = get_video_info(cap)
    cap.release()
    if max_frames is not None:
        n = min(n, max_frames)
    if n <= 1:
        return np.zeros(max(1, (n + stride - 1) // stride))
    indices = list(range(0, n, stride))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    frames = read_frames_at_indices(path, indices, grayscale=True)
    magnitudes = []
    prev_gray = None
    for i in indices:
        gray = frames.get(i)
        if gray is None:
            break
        if flow_scale != 1.0:
            gray = cv2.resize(gray, None, fx=flow_scale, fy=flow_scale,
                              interpolation=cv2.INTER_AREA)
        if prev_gray is None:
            magnitudes.append(0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            magnitudes.append(float(np.mean(mag)))
        prev_gray = gray
    return np.array(magnitudes)


def motion_magnitude_tvl1_downsampled(
    path: str | Path,
    stride: int = 5,
    max_frames: int | None = 2000,
    resize_short: int = 128,
) -> np.ndarray:
    """
    TV-L1 optical-flow variant of ``motion_magnitude_downsampled`` (ablation for
    the reviewer request on flow-algorithm choice). Uses scikit-image's dense
    TV-L1 solver instead of OpenCV Farneback. Frames are grayscaled and resized
    so the short side is ``resize_short`` px to keep the variational solver
    tractable; motion magnitude is the mean flow norm per sampled frame.
    Returns magnitudes for indices 0, stride, 2*stride, ... (first is 0).
    """
    from skimage.registration import optical_flow_tvl1

    cap = load_video_capture(path)
    n, _, _, _ = get_video_info(cap)
    if max_frames is not None:
        n = min(n, max_frames)
    if n <= 1:
        cap.release()
        return np.zeros(max(1, (n + stride - 1) // stride))
    indices = list(range(0, n, stride))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    def _prep(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        s = resize_short / max(1, min(h, w))
        if s < 1.0:
            gray = cv2.resize(
                gray,
                (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                interpolation=cv2.INTER_AREA,
            )
        return gray.astype(np.float32) / 255.0

    magnitudes: list[float] = []
    prev_gray = None
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        gray = _prep(frame)
        if prev_gray is None:
            magnitudes.append(0.0)
        else:
            v, u = optical_flow_tvl1(prev_gray, gray, num_warp=5, num_iter=10)
            mag = np.sqrt(u * u + v * v)
            magnitudes.append(float(np.mean(mag)))
        prev_gray = gray
    cap.release()
    return np.array(magnitudes)
