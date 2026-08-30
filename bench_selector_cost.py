#!/usr/bin/env python3
"""E5 -- selector front-end cost: EARS (flow+audio, no encoder) vs AKS (SigLIP
forward per candidate frame). Both feed the identical VLM, so the VLM cost is
common; this script isolates the *selection* overhead that differentiates them.

Per sampled video, times:
  EARS:  audio_energy_from_video + motion_magnitude_downsampled  (-> T frames)
  AKS :  decode T frames + SigLIP encode (compute_video_frame_embeddings)
Reports mean seconds/video for each and the speedup ratio.

Usage:
  python bench_selector_cost.py --n-per-dataset 12
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
from motion_features import motion_magnitude_downsampled
from semantic_features import (
    SemanticConfig, SemanticEncoder, compute_video_frame_embeddings,
)
from run_overnight_experiments import load_xd, load_ucf

MOTION_STRIDE = 5
SEGMENT_SECONDS = 1.0
OUT = _REPO / "results" / "e5_selector_cost.json"


def bench_video(path: Path, encoder: SemanticEncoder) -> dict | None:
    try:
        t0 = time.perf_counter()
        _ = audio_energy_from_video(path, segment_seconds=SEGMENT_SECONDS)
        t_audio = time.perf_counter() - t0

        t0 = time.perf_counter()
        M_t = motion_magnitude_downsampled(path, stride=MOTION_STRIDE, max_frames=2000)
        t_motion = time.perf_counter() - t0
        T = int(len(M_t))
        if T <= 0:
            return None

        # AKS front-end: decode T frames + SigLIP encode (query encode amortized).
        t0 = time.perf_counter()
        _ = compute_video_frame_embeddings(path, T, encoder, motion_stride=MOTION_STRIDE)
        t_siglip = time.perf_counter() - t0
    except Exception as e:  # skip unreadable videos
        print(f"  skip {path.name}: {e}")
        return None

    return {
        "video": path.name, "T": T,
        "ears_s": t_audio + t_motion,
        "ears_audio_s": t_audio, "ears_motion_s": t_motion,
        "aks_siglip_s": t_siglip,
    }


def summarize(rows: list[dict]) -> dict:
    def mean(key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")
    ears = mean("ears_s")
    aks = mean("aks_siglip_s")
    return {
        "n_videos": len(rows),
        "mean_T": mean("T"),
        "ears_s_per_video": ears,
        "ears_audio_s": mean("ears_audio_s"),
        "ears_motion_s": mean("ears_motion_s"),
        "aks_siglip_s_per_video": aks,
        "speedup_aks_over_ears": (aks / ears) if ears > 0 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-dataset", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    xd = load_xd()[: args.n_per_dataset]
    ucf = load_ucf()[: args.n_per_dataset]
    videos = [p for p, _ in xd] + [p for p, _ in ucf]
    print(f"[E5] benchmarking {len(videos)} videos "
          f"({len(xd)} XD + {len(ucf)} UCF)")

    encoder = SemanticEncoder(SemanticConfig(device=args.device))
    encoder.load()  # warm up (exclude model-load from timing)
    _ = encoder.encode_texts(["a violent scene"])

    rows = []
    for i, p in enumerate(videos, 1):
        r = bench_video(p, encoder)
        if r:
            rows.append(r)
            print(f"  [{i}/{len(videos)}] {r['video']:32s} T={r['T']:4d} "
                  f"EARS={r['ears_s']:.2f}s  AKS-SigLIP={r['aks_siglip_s']:.2f}s")

    summ = summarize(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summ, "per_video": rows}, open(OUT, "w"),
              indent=2, default=str)
    print("\n[E5] SUMMARY")
    print(f"  videos={summ['n_videos']}  mean T={summ['mean_T']:.0f}")
    print(f"  EARS  (flow+audio) : {summ['ears_s_per_video']:.3f} s/video")
    print(f"  AKS   (SigLIP)     : {summ['aks_siglip_s_per_video']:.3f} s/video")
    print(f"  AKS/EARS overhead  : {summ['speedup_aks_over_ears']:.1f}x")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
