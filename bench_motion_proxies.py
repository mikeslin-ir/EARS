#!/usr/bin/env python3
"""Where does EARS's selection time actually go, and can a cheaper motion proxy
recover it?

`bench_selector_cost.py` established that EARS selection costs ~7.6 s/video
against ~6.9 s for a frozen SigLIP pass, and that 96% of EARS's time is the
motion channel. That measurement treats the motion channel as one blob. This
script splits it into seek+decode versus the Farneback solve, and times three
drop-in replacements against it on the same videos:

  full      Farneback at native resolution   (what EARS ships today)
  half/qtr  Farneback after downscaling      (flow cost is O(pixels))
  rgbdiff   mean |I_t - I_{t-1}|             (no flow at all)

Agreement with the shipped signal is reported as Spearman rho, since only the
ranking of frames feeds top-K selection -- a proxy that preserves the ordering
loses nothing downstream even if its magnitudes differ.

Usage:
  python bench_motion_proxies.py --n-per-dataset 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from audio_features import audio_energy_from_video
from run_overnight_experiments import load_xd, load_ucf
from video_loader import get_video_info, load_video_capture

MOTION_STRIDE = 5
MAX_FRAMES = 2000
SEGMENT_SECONDS = 1.0
OUT = _REPO / "results" / "motion_proxy_cost.json"

_FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15,
                  iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def _grab_grays(path: Path) -> tuple[list[np.ndarray], float]:
    """Seek every stride-th frame and return greyscale frames plus the time spent.

    This is the cost every variant pays, so isolating it bounds what any cheaper
    motion proxy can possibly save.
    """
    cap = load_video_capture(path)
    n, _, _, _ = get_video_info(cap)
    n = min(n, MAX_FRAMES)
    if n <= 1:
        cap.release()
        return [], 0.0
    indices = list(range(0, n, MOTION_STRIDE))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    t0 = time.perf_counter()
    grays: list[np.ndarray] = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    elapsed = time.perf_counter() - t0
    cap.release()
    return grays, elapsed


def _flow_signal(grays: list[np.ndarray], scale: float) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    out = [0.0]
    prev = None
    for g in grays:
        if scale != 1.0:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, **_FARNEBACK)
            out.append(float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))))
        prev = g
    return np.asarray(out[: len(grays)]), time.perf_counter() - t0


def _rgbdiff_signal(grays: list[np.ndarray]) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    out = [0.0]
    prev = None
    for g in grays:
        gf = g.astype(np.float32)
        if prev is not None:
            out.append(float(np.mean(np.abs(gf - prev))))
        prev = gf
    return np.asarray(out[: len(grays)]), time.perf_counter() - t0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.size != b.size:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def bench_video(path: Path) -> dict | None:
    grays, t_decode = _grab_grays(path)
    if len(grays) < 3:
        return None

    full, t_full = _flow_signal(grays, 1.0)
    half, t_half = _flow_signal(grays, 0.5)
    qtr, t_qtr = _flow_signal(grays, 0.25)
    rgb, t_rgb = _rgbdiff_signal(grays)

    t0 = time.perf_counter()
    _ = audio_energy_from_video(path, segment_seconds=SEGMENT_SECONDS)
    t_audio = time.perf_counter() - t0

    h, w = grays[0].shape
    return {
        "video": path.name, "T": len(grays), "height": int(h), "width": int(w),
        "t_decode_s": t_decode, "t_audio_s": t_audio,
        "t_flow_full_s": t_full, "t_flow_half_s": t_half,
        "t_flow_quarter_s": t_qtr, "t_rgbdiff_s": t_rgb,
        "rho_half": _spearman(full, half),
        "rho_quarter": _spearman(full, qtr),
        "rho_rgbdiff": _spearman(full, rgb),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-dataset", type=int, default=12)
    args = ap.parse_args()

    videos = [p for p, _ in load_xd()[: args.n_per_dataset]] + \
             [p for p, _ in load_ucf()[: args.n_per_dataset]]
    print(f"[cost] {len(videos)} videos, stride={MOTION_STRIDE}", flush=True)

    rows = []
    for i, p in enumerate(videos, 1):
        r = bench_video(Path(p))
        if r is None:
            continue
        rows.append(r)
        print(f"  [{i}/{len(videos)}] {r['video']:34s} T={r['T']:4d} "
              f"decode={r['t_decode_s']:5.2f}s flow={r['t_flow_full_s']:6.2f}s "
              f"rgbdiff={r['t_rgbdiff_s']:5.3f}s", flush=True)

    def m(k):
        return float(np.nanmean([r[k] for r in rows]))

    dec, aud = m("t_decode_s"), m("t_audio_s")
    variants = {
        "flow_full": m("t_flow_full_s"), "flow_half": m("t_flow_half_s"),
        "flow_quarter": m("t_flow_quarter_s"), "rgbdiff": m("t_rgbdiff_s"),
    }
    summary = {
        "n_videos": len(rows), "mean_T": m("T"),
        "decode_s": dec, "audio_s": aud,
        "motion_compute_s": variants,
        "total_s": {k: dec + aud + v for k, v in variants.items()},
        "spearman_vs_full_flow": {
            "flow_half": m("rho_half"), "flow_quarter": m("rho_quarter"),
            "rgbdiff": m("rho_rgbdiff"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "per_video": rows}, open(OUT, "w"), indent=2)

    print(f"\n[cost] {len(rows)} videos, mean T={summary['mean_T']:.0f} sampled frames")
    print(f"  shared cost: seek+decode {dec:.2f}s  audio {aud:.2f}s")
    print(f"\n  {'motion variant':<16}{'compute':>10}{'total':>10}{'speedup':>10}"
          f"{'rank agreement':>16}")
    base = summary["total_s"]["flow_full"]
    for k, v in variants.items():
        rho = summary["spearman_vs_full_flow"].get(k)
        rho_s = "  (reference)" if k == "flow_full" else f"{rho:>16.3f}"
        print(f"  {k:<16}{v:>10.3f}{dec + aud + v:>10.3f}"
              f"{base / (dec + aud + v):>10.1f}x{rho_s}")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
