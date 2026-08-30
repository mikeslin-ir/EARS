#!/usr/bin/env python3
"""
Standalone runtime benchmark for the EARS pipeline.

Measures per-video wall-clock for:
  (1) audio energy extraction (E_t)
  (2) Farneback optical-flow motion extraction (M_t, stride=5)
  (3) fusion + top-K selection
  (4) VLM forward pass (Qwen2-VL-7B by default) at K frames
  (5) baseline uniform sampling (frame decode only, no signals)

Runs on a small video subset of UCF-Crime and XD-Violence.

Usage:
  python bench_runtime.py --max-videos 20 --k 4 --device cuda
  python bench_runtime.py --no-vlm  # CPU-only signal timing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from audio_features import audio_energy_from_video  # noqa: E402
from motion_features import motion_magnitude_downsampled  # noqa: E402
from event_attention import compute_event_attention_from_signals  # noqa: E402
from frame_retrieval import retrieve_frames_uniform  # noqa: E402
from ic_selection import select_topk_ic, tau_to_gap  # noqa: E402
from video_loader import get_video_info, load_video_capture, load_frame_at_index, probe_fps  # noqa: E402
from config import UCF_CRIME_ROOT, XD_VIOLENCE_ROOT  # noqa: E402


MOTION_STRIDE = 5
SEGMENT_SECONDS = 1.0
ALPHA, BETA, TAU_SEC = 1.0, 1.0, 0.5


def _list_test_videos(root: Path, max_videos: int) -> list[Path]:
    """Walk test/ collecting video paths, shuffled with fixed seed for class balance."""
    test_dir = root / "test"
    if not test_dir.exists():
        return []
    out: list[Path] = []
    for sub in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        for p in sorted(sub.iterdir()):
            if p.suffix.lower() in (".mp4", ".avi", ".mkv"):
                out.append(p)
    rng = np.random.default_rng(42)
    rng.shuffle(out)
    return out[:max_videos]


def _summary(name: str, vals: list[float]) -> dict:
    if not vals:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": len(vals),
        "mean_s": float(mean(vals)),
        "std_s": float(stdev(vals)) if len(vals) > 1 else 0.0,
        "min_s": float(min(vals)),
        "max_s": float(max(vals)),
        "total_s": float(sum(vals)),
    }


def bench_one_video(path: Path, k: int, agent, prompt: str | None) -> dict[str, float]:
    times: dict[str, float] = {}

    t0 = time.perf_counter()
    E_t = audio_energy_from_video(path, segment_seconds=SEGMENT_SECONDS)
    times["audio_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    M_t = motion_magnitude_downsampled(path, stride=MOTION_STRIDE, max_frames=2000)
    times["motion_s"] = time.perf_counter() - t0

    if len(M_t) == 0:
        times["fusion_s"] = 0.0
        times["select_s"] = 0.0
        times["frame_decode_evt_s"] = 0.0
        times["ears_vlm_s"] = 0.0
        times["uniform_decode_s"] = 0.0
        times["uniform_vlm_s"] = 0.0
        times["ears_end_to_end_s"] = times["audio_s"] + times["motion_s"]
        times["uniform_end_to_end_s"] = 0.0
        return times

    t0 = time.perf_counter()
    A_t = compute_event_attention_from_signals(E_t, M_t, alpha=ALPHA, beta=BETA)
    times["fusion_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tau_frames = tau_to_gap(TAU_SEC, probe_fps(path), MOTION_STRIDE)
    top_indices = select_topk_ic(A_t, k, tau_frames=tau_frames)
    times["select_s"] = time.perf_counter() - t0

    cap = load_video_capture(path)
    try:
        total_frames, _, _, _ = get_video_info(cap)
    finally:
        cap.release()
    frame_indices = [min(int(i) * MOTION_STRIDE, total_frames - 1) for i in top_indices]
    frame_indices = sorted(set(frame_indices))[:k]

    t0 = time.perf_counter()
    evt_frames = []
    for fi in frame_indices:
        f = load_frame_at_index(path, fi)
        if f is not None:
            evt_frames.append(f)
    times["frame_decode_evt_s"] = time.perf_counter() - t0

    if agent is not None and prompt is not None and evt_frames:
        t0 = time.perf_counter()
        try:
            agent.run(evt_frames, prompt)
        except Exception as e:
            print(f"[vlm error on {path.name}] {e}", file=sys.stderr)
        times["ears_vlm_s"] = time.perf_counter() - t0
    else:
        times["ears_vlm_s"] = 0.0

    t0 = time.perf_counter()
    uni = retrieve_frames_uniform(path, total_frames, k)
    times["uniform_decode_s"] = time.perf_counter() - t0
    if agent is not None and prompt is not None and uni:
        t0 = time.perf_counter()
        try:
            agent.run([frame for _, frame in uni], prompt)
        except Exception as e:
            print(f"[uniform VLM error on {path.name}] {e}", file=sys.stderr)
        times["uniform_vlm_s"] = time.perf_counter() - t0
    else:
        times["uniform_vlm_s"] = 0.0

    times["ears_end_to_end_s"] = (
        times["audio_s"] + times["motion_s"] + times["fusion_s"]
        + times["select_s"] + times["frame_decode_evt_s"] + times["ears_vlm_s"]
    )
    times["uniform_end_to_end_s"] = times["uniform_decode_s"] + times["uniform_vlm_s"]

    return times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ucf", "xd", "both"], default="both")
    ap.add_argument("--max-videos", type=int, default=20)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-id", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--no-vlm", action="store_true", help="Skip VLM timing")
    ap.add_argument("--ucf-root", type=Path, default=UCF_CRIME_ROOT)
    ap.add_argument("--xd-root", type=Path, default=XD_VIOLENCE_ROOT)
    ap.add_argument("--output", type=Path, default=Path("results/runtime_bench.json"))
    args = ap.parse_args()

    agent = None
    prompt = None
    if not args.no_vlm:
        import torch
        if args.device == "cuda" and not torch.cuda.is_available():
            print("ERROR: --device cuda but CUDA unavailable.", file=sys.stderr)
            return 1
        from vlm_reasoning import VLMReasoningAgent
        agent = VLMReasoningAgent(model_id=args.model_id, device=args.device, max_new_tokens=8)
        print(f"[load VLM {args.model_id} on {args.device}] ...")
        agent.load()
        prompt = (
            "Given the following frames sampled from a surveillance video, "
            "determine whether the video contains anomalous or violent activity. "
            "Answer Yes or No."
        )

    out: dict = {
        "k": args.k,
        "model_id": args.model_id if not args.no_vlm else None,
        "device": args.device if not args.no_vlm else "cpu",
        "motion_stride": MOTION_STRIDE,
        "datasets": {},
    }

    datasets = []
    if args.dataset in ("ucf", "both"):
        datasets.append(("UCF-Crime", args.ucf_root))
    if args.dataset in ("xd", "both"):
        datasets.append(("XD-Violence", args.xd_root))

    for name, root in datasets:
        vids = _list_test_videos(root, args.max_videos)
        if not vids:
            print(f"[{name}] no videos found under {root/'test'}")
            continue
        print(f"\n=== {name}: {len(vids)} videos ===")
        per_video: list[dict] = []
        for i, p in enumerate(vids, 1):
            t_video = time.perf_counter()
            try:
                t = bench_one_video(p, args.k, agent, prompt)
            except Exception as e:
                print(f"[{i}/{len(vids)} {p.name}] error: {e}", file=sys.stderr)
                continue
            t["video_total_s"] = time.perf_counter() - t_video
            t["video"] = p.name
            per_video.append(t)
            ev_pipeline = t["audio_s"] + t["motion_s"] + t["fusion_s"] + t["select_s"] + t["frame_decode_evt_s"]
            print(
                f"  [{i:2d}/{len(vids)}] {p.name[:40]:40s} "
                f"audio={t['audio_s']:5.2f}s motion={t['motion_s']:6.2f}s "
                f"fuse+sel={t['fusion_s']+t['select_s']:5.3f}s "
                f"decode={t['frame_decode_evt_s']:5.2f}s "
                f"vlm(E/U)={t['ears_vlm_s']:5.2f}/{t['uniform_vlm_s']:5.2f}s  "
                f"(ears-pipeline={ev_pipeline:5.2f}s)"
            )

        keys = ["audio_s", "motion_s", "fusion_s", "select_s",
                "frame_decode_evt_s", "ears_vlm_s", "uniform_decode_s",
                "uniform_vlm_s", "ears_end_to_end_s", "uniform_end_to_end_s",
                "video_total_s"]
        summary = {k: _summary(k, [r[k] for r in per_video if k in r]) for k in keys}
        # Composite: EARS pre-processing (no VLM)
        ears_pp = [r["audio_s"] + r["motion_s"] + r["fusion_s"] + r["select_s"]
                   for r in per_video]
        summary["ears_preproc_total_s"] = _summary("ears_preproc_total_s", ears_pp)
        out["datasets"][name] = {"n_videos": len(per_video),
                                 "per_video": per_video,
                                 "summary": summary}
        # Print compact summary
        print(f"\n  --- {name} mean per video ---")
        for k in ["audio_s", "motion_s", "fusion_s", "select_s",
                  "frame_decode_evt_s", "uniform_decode_s", "ears_vlm_s",
                  "uniform_vlm_s", "ears_end_to_end_s", "uniform_end_to_end_s",
                  "video_total_s"]:
            s = summary[k]
            if s["n"]:
                print(f"    {k:24s} mean={s['mean_s']:.3f}s  std={s['std_s']:.3f}  total={s['total_s']:.1f}s")
        s = summary["ears_preproc_total_s"]
        print(f"    {'EARS preproc (a+m+f+s)':24s} mean={s['mean_s']:.3f}s  total={s['total_s']:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
