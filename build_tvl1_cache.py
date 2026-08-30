"""
Build a TV-L1-motion signal cache for the XD-Violence test set (reviewer
ablation on optical-flow algorithm choice).

For every test video we reuse the *existing* audio energy E_t / total_frames /
fps from the Farneback signal cache (results/signal_cache) and only recompute
the motion signal M_t with TV-L1, writing a parallel cache under
results/tvl1_cache with the SAME per-video filename. This isolates the flow
algorithm so evaluate_xdviolence.py can be pointed at the new cache unchanged:

    python evaluate_xdviolence.py --model-id Qwen/Qwen2-VL-7B-Instruct \
        --method event_driven --frame-budget 4 \
        --signal-cache-dir results/tvl1_cache --results-dir results/tvl1

Usage:
    python build_tvl1_cache.py [--workers N] [--max-videos M] [--resize-short PX]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from config import XD_VIOLENCE_ROOT
from evaluate_xdviolence import load_dataset, MOTION_STRIDE
from motion_features import motion_magnitude_tvl1_downsampled

SRC_CACHE = Path("results/signal_cache")
DST_CACHE = Path("results/tvl1_cache")


def _hash(p: Path) -> str:
    return hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:16]


def _cf(cache_dir: Path, p: Path) -> Path:
    return cache_dir / f"{p.stem}_{_hash(p)}.npz"


def process_one(path_str: str, resize_short: int) -> tuple[str, str]:
    path = Path(path_str)
    dst = _cf(DST_CACHE, path)
    if dst.exists():
        return (path.name, "skip")
    src = _cf(SRC_CACHE, path)
    if not src.exists():
        return (path.name, "no_src_Et")
    z = np.load(src)
    E_t = z["E_t"]
    total_frames = int(z["total_frames"])
    fps = float(z["fps"]) if "fps" in z else 30.0
    M_t = motion_magnitude_tvl1_downsampled(
        path, stride=MOTION_STRIDE, max_frames=2000, resize_short=resize_short
    )
    if len(M_t) == 0:
        return (path.name, "empty_motion")
    np.savez_compressed(
        dst, E_t=E_t, M_t=M_t,
        total_frames=np.int64(total_frames), fps=np.float64(fps),
    )
    return (path.name, "ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--resize-short", type=int, default=128)
    args = ap.parse_args()

    DST_CACHE.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(root=XD_VIOLENCE_ROOT, max_videos=args.max_videos)
    paths = [str(p) for p, _ in dataset]
    print(f"Building TV-L1 cache for {len(paths)} XD-Violence videos "
          f"(workers={args.workers}, resize_short={args.resize_short})")

    counts: dict[str, int] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, p, args.resize_short): p for p in paths}
        for fut in as_completed(futs):
            name, status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if done % 25 == 0 or done == len(paths):
                print(f"  [{done}/{len(paths)}] {counts}")
    print("DONE", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
