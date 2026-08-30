#!/usr/bin/env python3
"""
Ablation runner for the EARS paper.

Three ablations are supported (selected via --ablation):

  1. modality   -- (alpha, beta) sweep: audio-only, motion-only, fused.
  2. ic_gap     -- temporal NMS gap tau sweep (in seconds).
  3. adaptive_k -- adaptive frame budget by thresholding A_t after IC,
                   showing the AP / accuracy vs avg frames-per-video Pareto.

Reuses build_event_cache, VLMReasoningAgent and compute_metrics from the
existing UCF-Crime / XD-Violence evaluators (no retraining, no model change).

Usage examples:
  python run_ablations.py --dataset ucfcrime --ablation modality   --frame-budget 2
  python run_ablations.py --dataset ucfcrime --ablation ic_gap     --frame-budget 4 --gaps 0,0.5,1,2,4
  python run_ablations.py --dataset xdviolence --ablation adaptive_k --k-max 8 --thresholds 0.50,0.55,0.60,0.65,0.70

Outputs JSON to <results_dir>/ablation_<dataset>_<ablation>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
from tqdm import tqdm

from config import EventAgentConfig, VLMConfig
from audio_features import audio_energy_from_video
from motion_features import motion_magnitude_downsampled
from event_attention import compute_event_attention_from_signals
from frame_retrieval import retrieve_frames_uniform
from video_loader import get_video_info, load_video_capture, load_frame_at_index
from vlm_reasoning import VLMReasoningAgent
from ic_selection import select_topk_ic, select_adaptive_ic, tau_to_gap

from evaluate_xdviolence import (
    compute_metrics,
    MOTION_STRIDE,
    SEGMENT_SECONDS,
    load_dataset as load_xd,
    VLM_PROMPT as XD_PROMPT,
)
from evaluate_ucfcrime import (
    load_dataset as load_ucf,
    UCF_PROMPT,
)


# ---------------------------------------------------------------------------
# Raw signal cache (E_t, M_t, total_frames) -- independent of alpha/beta/tau
# ---------------------------------------------------------------------------
def build_signal_cache(
    dataset: list[tuple[Path, int]],
    cache_dir: Path | None = None,
) -> dict[Path, tuple[np.ndarray, np.ndarray, int, float]]:
    """
    Returns: path -> (E_t_segment_level, M_t_frame_level, total_frames, fps)

    If `cache_dir` is given, per-video signals are persisted to / loaded from
    .npz files keyed by absolute path hash, so subsequent runs skip extraction.
    """
    import hashlib

    def _cache_path(p: Path) -> Path | None:
        if cache_dir is None:
            return None
        h = hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:16]
        return cache_dir / f"{p.stem}_{h}.npz"

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[Path, tuple[np.ndarray, np.ndarray, int, float]] = {}
    for path, _ in tqdm(dataset, desc="Precomputing audio+motion signals", unit="video"):
        cf = _cache_path(path)
        if cf is not None and cf.exists():
            try:
                z = np.load(cf)
                cache[path] = (z["E_t"], z["M_t"], int(z["total_frames"]),
                               float(z["fps"]))
                continue
            except Exception:
                pass  # fall through to recompute
        E_t = audio_energy_from_video(path, segment_seconds=SEGMENT_SECONDS)
        M_t = motion_magnitude_downsampled(path, stride=MOTION_STRIDE, max_frames=2000)
        if len(M_t) == 0:
            continue
        cap = load_video_capture(path)
        try:
            total_frames, fps, _, _ = get_video_info(cap)
        finally:
            cap.release()
        fps_v = float(fps) if fps else 30.0
        cache[path] = (E_t, M_t, total_frames, fps_v)
        if cf is not None:
            try:
                np.savez_compressed(cf, E_t=E_t, M_t=M_t,
                                    total_frames=np.int64(total_frames),
                                    fps=np.float64(fps_v))
            except Exception:
                pass
    return cache


def signal_indices_to_frame_indices(
    signal_indices: np.ndarray,
    total_frames: int,
) -> list[int]:
    """A_t has length len(M_t); map back to original frame indices via MOTION_STRIDE."""
    out = [min(int(i) * MOTION_STRIDE, total_frames - 1) for i in signal_indices]
    return sorted(set(out))


def load_frames(video_path: Path, frame_indices: list[int]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for fi in frame_indices:
        f = load_frame_at_index(video_path, fi)
        if f is not None:
            frames.append(f)
    return frames


# ---------------------------------------------------------------------------
# One-pass VLM inference helper
# ---------------------------------------------------------------------------
def run_one_setting(
    dataset: list[tuple[Path, int]],
    cache: dict[Path, tuple[np.ndarray, np.ndarray, int, float]],
    agent: VLMReasoningAgent,
    prompt: str,
    *,
    alpha: float,
    beta: float,
    k: int,
    tau_sec: float,
    adaptive_threshold: float | None,
    desc: str,
) -> dict:
    y_true: list[int] = []
    y_pred: list[int] = []
    total_frames_sent = 0
    t0 = time.time()
    for path, gt in tqdm(dataset, desc=desc, unit="video"):
        if path not in cache:
            y_true.append(gt); y_pred.append(0)
            continue
        E_t, M_t, total_frames, fps = cache[path]
        A_t = compute_event_attention_from_signals(E_t, M_t, alpha=alpha, beta=beta)
        # Convert tau (seconds, in original frame space) to signal-space gap (motion stride).
        tau_signal = tau_to_gap(tau_sec, fps, MOTION_STRIDE)
        if adaptive_threshold is not None:
            sig_idx = select_adaptive_ic(A_t, k_max=k, tau_frames=tau_signal,
                                         threshold=adaptive_threshold)
        else:
            sig_idx = select_topk_ic(A_t, k=k, tau_frames=tau_signal)
        if sig_idx.size == 0:
            y_true.append(gt); y_pred.append(0)
            continue
        frame_indices = signal_indices_to_frame_indices(sig_idx, total_frames)[:k]
        frames = load_frames(path, frame_indices)
        if not frames:
            y_true.append(gt); y_pred.append(0)
            continue
        _, is_pos = agent.run(frames, prompt)
        y_true.append(gt)
        y_pred.append(1 if is_pos else 0)
        total_frames_sent += len(frames)
    elapsed = time.time() - t0
    metrics = compute_metrics(y_true, y_pred)
    n = len(dataset) or 1
    return {
        "metrics": metrics,
        "n_videos": len(dataset),
        "total_frames": total_frames_sent,
        "avg_frames_per_video": total_frames_sent / n,
        "elapsed_sec": elapsed,
        "alpha": alpha, "beta": beta, "k": k,
        "tau_sec": tau_sec, "adaptive_threshold": adaptive_threshold,
    }


# ---------------------------------------------------------------------------
# Ablation drivers
# ---------------------------------------------------------------------------
def run_modality_ablation(dataset, cache, agent, prompt, k: int, tau_sec: float) -> list[dict]:
    configs = [
        ("audio_only",  1.0, 0.0),
        ("motion_only", 0.0, 1.0),
        ("fused_1_1",   1.0, 1.0),
        ("audio_heavy", 2.0, 1.0),
        ("motion_heavy",1.0, 2.0),
    ]
    out = []
    for name, a, b in configs:
        r = run_one_setting(dataset, cache, agent, prompt,
                            alpha=a, beta=b, k=k, tau_sec=tau_sec,
                            adaptive_threshold=None,
                            desc=f"modality:{name} K={k}")
        r["variant"] = name
        out.append(r)
    return out


def run_ic_gap_ablation(dataset, cache, agent, prompt, k: int, alpha: float, beta: float,
                        gaps_sec: list[float]) -> list[dict]:
    out = []
    for g in gaps_sec:
        r = run_one_setting(dataset, cache, agent, prompt,
                            alpha=alpha, beta=beta, k=k, tau_sec=float(g),
                            adaptive_threshold=None,
                            desc=f"ic_gap:tau={g}s K={k}")
        out.append(r)
    return out


def run_adaptive_k_ablation(dataset, cache, agent, prompt, k_max: int, alpha: float, beta: float,
                            tau_sec: float, thresholds: list[float]) -> list[dict]:
    out = []
    for thr in thresholds:
        r = run_one_setting(dataset, cache, agent, prompt,
                            alpha=alpha, beta=beta, k=k_max, tau_sec=tau_sec,
                            adaptive_threshold=float(thr),
                            desc=f"adaptive_K_max={k_max} thr={thr}")
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", choices=["ucfcrime", "xdviolence"], required=True)
    ap.add_argument("--ablation", choices=["modality", "ic_gap", "adaptive_k"], required=True)
    ap.add_argument("--frame-budget", type=int, default=4, help="K (or K_max for adaptive)")
    ap.add_argument("--alpha", type=float, default=None, help="Override audio weight (default dataset-aware)")
    ap.add_argument("--beta",  type=float, default=None, help="Override motion weight (default 1.0)")
    ap.add_argument("--tau-sec", type=float, default=0.5, help="IC gap in seconds (modality/adaptive)")
    ap.add_argument("--gaps",       type=str, default="0,0.5,1,2,4", help="Comma-sep tau sweep (ic_gap)")
    ap.add_argument("--thresholds", type=str, default="0.50,0.55,0.60,0.65,0.70", help="Comma-sep adaptive thresholds")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "ablations")
    ap.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # dataset-specific defaults
    if args.dataset == "ucfcrime":
        loader = load_ucf
        prompt = UCF_PROMPT
        default_alpha, default_beta = 0.0, 1.0    # UCF audio is often missing
    else:
        loader = load_xd
        prompt = XD_PROMPT
        default_alpha, default_beta = 2.0, 1.0

    alpha = args.alpha if args.alpha is not None else default_alpha
    beta  = args.beta  if args.beta  is not None else default_beta

    dataset = loader(max_videos=args.max_videos)
    if not dataset:
        print("Empty dataset", file=sys.stderr); return 1
    print(f"Loaded {len(dataset)} videos")

    cache = build_signal_cache(dataset)
    print(f"Cached signals for {len(cache)} videos")

    vlm_cfg = VLMConfig(model_id=args.model_id, device=args.device)
    agent = VLMReasoningAgent(model_id=vlm_cfg.model_id, device=vlm_cfg.device,
                              max_new_tokens=vlm_cfg.max_new_tokens)
    agent.load()

    if args.ablation == "modality":
        results = run_modality_ablation(dataset, cache, agent, prompt,
                                        k=args.frame_budget, tau_sec=args.tau_sec)
    elif args.ablation == "ic_gap":
        gaps = [float(x) for x in args.gaps.split(",") if x.strip()]
        results = run_ic_gap_ablation(dataset, cache, agent, prompt,
                                      k=args.frame_budget, alpha=alpha, beta=beta,
                                      gaps_sec=gaps)
    else:  # adaptive_k
        thrs = [float(x) for x in args.thresholds.split(",") if x.strip()]
        results = run_adaptive_k_ablation(dataset, cache, agent, prompt,
                                          k_max=args.frame_budget, alpha=alpha, beta=beta,
                                          tau_sec=args.tau_sec, thresholds=thrs)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.results_dir / f"ablation_{args.dataset}_{args.ablation}.json"
    payload = {
        "dataset": args.dataset, "ablation": args.ablation,
        "frame_budget": args.frame_budget, "alpha": alpha, "beta": beta,
        "tau_sec_default": args.tau_sec,
        "n_videos": len(dataset),
        "model_id": args.model_id,
        "results": results,
    }
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
