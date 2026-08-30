#!/usr/bin/env python3
"""Does a cheaper motion channel change the frames EARS actually picks?

`bench_motion_proxies.py` showed half-resolution flow tracks the full-resolution
signal at Spearman rho=0.976 and costs 4.5x less. High rank correlation is not
the property that matters, though: only the top-K frames survive into the VLM,
and a proxy can preserve the global ordering while still moving the peak. This
script measures the thing that propagates downstream -- how often the selected
frame indices are identical -- and writes selections for each variant so the
survivors can be scored by the usual VLM runner.

    python bench_flow_scale_selection.py --dataset xd --n 120
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from audio_features import audio_energy_from_video
from event_attention import compute_event_attention_from_signals
from ic_selection import select_topk_ic, tau_to_gap
from motion_features import motion_magnitude_downsampled
from run_overnight_experiments import load_xd, load_ucf
from video_loader import get_video_info, load_video_capture

STRIDE = 5
MAX_FRAMES = 2000
SEGMENT_SECONDS = 1.0
KS = (1, 2, 4, 8)
# The configuration the held-out search froze on both datasets.
ALPHA, BETA, TAU_SEC = 1.0, 1.0, 0.5
OUT = _REPO / "results" / "flow_scale_selection.json"


def _rgbdiff(path: Path) -> np.ndarray:
    import cv2

    from video_loader import read_frames_at_indices

    cap = load_video_capture(path)
    n, _, _, _ = get_video_info(cap)
    cap.release()
    n = min(n, MAX_FRAMES)
    idx = list(range(0, n, STRIDE))
    frames = read_frames_at_indices(path, idx, grayscale=True)
    out, prev = [], None
    for i in idx:
        g = frames.get(i)
        if g is None:
            break
        gf = g.astype(np.float32)
        out.append(0.0 if prev is None else float(np.mean(np.abs(gf - prev))))
        prev = gf
    return np.asarray(out)


def _select(E_t: np.ndarray, M_t: np.ndarray, fps: float, k: int) -> tuple[int, ...]:
    """Top-K on the fused signal with the same IC gap the paper uses."""
    A_t = compute_event_attention_from_signals(E_t, M_t, alpha=ALPHA, beta=BETA)
    picked = select_topk_ic(A_t, k=k, tau_frames=tau_to_gap(TAU_SEC, fps, STRIDE))
    return tuple(sorted(int(p) * STRIDE for p in picked))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["xd", "ucf"], default="xd")
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()

    loader = load_xd if args.dataset == "xd" else load_ucf
    videos = [p for p, _ in loader()[: args.n]]
    variants = ("full", "half", "quarter", "rgbdiff")
    sels: dict[str, dict[str, dict[int, tuple[int, ...]]]] = {v: {} for v in variants}
    times: dict[str, list[float]] = {v: [] for v in variants}

    print(f"[flow-scale] {len(videos)} {args.dataset} videos, "
          f"alpha={ALPHA} beta={BETA} tau={TAU_SEC}s", flush=True)
    for n_done, p in enumerate(videos, 1):
        cap = load_video_capture(p)
        _, fps, _, _ = get_video_info(cap)
        cap.release()
        fps = float(fps) or 30.0
        E_t = audio_energy_from_video(p, segment_seconds=SEGMENT_SECONDS)

        signals = {}
        for name, fn in (
            ("full", lambda: motion_magnitude_downsampled(p, STRIDE, MAX_FRAMES, 1.0)),
            ("half", lambda: motion_magnitude_downsampled(p, STRIDE, MAX_FRAMES, 0.5)),
            ("quarter", lambda: motion_magnitude_downsampled(p, STRIDE, MAX_FRAMES, 0.25)),
            ("rgbdiff", lambda: _rgbdiff(p)),
        ):
            t0 = time.perf_counter()
            signals[name] = fn()
            times[name].append(time.perf_counter() - t0)

        if any(s.size == 0 for s in signals.values()):
            continue
        for name, M_t in signals.items():
            sels[name][p.name] = {k: _select(E_t, M_t, fps, k) for k in KS}
        if n_done % 20 == 0:
            print(f"  {n_done}/{len(videos)}", flush=True)

    ids = sorted(sels["full"])
    print(f"\nselection agreement with full-resolution flow ({len(ids)} videos)")
    print(f"{'variant':<10}{'mean cost':>11}" + "".join(f"{'K='+str(k):>9}" for k in KS)
          + f"{'any-K':>9}")
    report = {}
    for v in variants:
        row, per_k = "", {}
        for k in KS:
            same = sum(sels[v][i][k] == sels["full"][i][k] for i in ids) / len(ids)
            per_k[k] = same
            row += f"{same:>9.2f}"
        allk = sum(all(sels[v][i][k] == sels["full"][i][k] for k in KS)
                   for i in ids) / len(ids)
        report[v] = {"per_k_identical": per_k, "all_k_identical": allk,
                     "mean_cost_s": float(np.mean(times[v]))}
        print(f"{v:<10}{np.mean(times[v]):>10.3f}s{row}{allk:>9.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"dataset": args.dataset, "n_videos": len(ids),
               "alpha": ALPHA, "beta": BETA, "tau_sec": TAU_SEC,
               "variants": report,
               "selections": {v: {i: {str(k): list(s) for k, s in d.items()}
                                  for i, d in sels[v].items()} for v in variants}},
              open(OUT, "w"), indent=2)
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
