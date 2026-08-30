#!/usr/bin/env python3
"""
Full evaluation pipeline for XD-Violence: event-driven vs uniform frame sampling
for video violence detection using a VLM. Suitable for workshop paper experiments.

Usage:
  python evaluate_xdviolence.py
  python evaluate_xdviolence.py --max-videos 20 --frame-budget 4 --method uniform
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repo root for imports
_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from config import EventAgentConfig, VLMConfig, XD_VIOLENCE_ROOT
from video_loader import get_video_info, load_video_capture, load_frame_at_index, probe_fps
from audio_features import audio_energy_from_video
from motion_features import motion_magnitude_downsampled
from event_attention import compute_event_attention_from_signals
from frame_retrieval import retrieve_frames_uniform
from ic_selection import select_topk_ic, tau_to_gap
from vlm_reasoning import VLMReasoningAgent

# Dataset and frozen paper configuration
FROZEN_CONFIG_PATH = _REPO / "configs" / "frozen_config.json"
# Prompt for violence detection (research specification)
VLM_PROMPT = (
    "Given the following frames sampled from a surveillance video, "
    "determine whether the video contains violent activity. Answer Yes or No."
)
# Frame budgets for experiments
FRAME_BUDGETS = (1, 2, 4, 6, 8, 10)
MOTION_STRIDE = 5
SEGMENT_SECONDS = 1.0


def load_dataset(
    root: Path | None = None,
    max_videos: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[tuple[Path, int]]:
    """
    Load XD-Violence test set. Returns list of (video_path, ground_truth).
    Ground truth: 0 = Normal, 1 = Violence (Fighting or Shooting).
    When max_videos is set, we shuffle the full list (fixed seed) then take the first
    max_videos, so the subset includes both Normal and Violence (avoids precision=1.0
    artifact from taking only the first class).
    """
    root = root or XD_VIOLENCE_ROOT
    test_dir = root / "test"
    if not test_dir.exists():
        return []
    out: list[tuple[Path, int]] = []
    for sub in ("Fighting", "Shooting", "Normal"):
        d = test_dir / sub
        if not d.is_dir():
            continue
        gt = 0 if sub == "Normal" else 1
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".mp4", ".avi", ".mkv"):
                out.append((p, gt))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(out)
    if max_videos is not None and max_videos > 0:
        out = out[:max_videos]
    return out


def build_event_cache(
    dataset: list[tuple[Path, int]],
    alpha: float = 2.0,
    beta: float = 1.0,
    cache_dir: Path | None = None,
) -> dict[Path, tuple[np.ndarray, int]]:
    """
    Precompute (A_t, total_frames) for every video. Reuse for all K in event_driven.
    Returns dict: path -> (A_t, total_frames).

    If `cache_dir` is given, raw (E_t, M_t, total_frames) signals are persisted
    to / read from per-video .npz files so subsequent runs (including with
    different alpha/beta or a different VLM) skip the slow audio+motion pass.
    """
    import hashlib

    def _cf(p: Path) -> Path | None:
        if cache_dir is None:
            return None
        h = hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:16]
        return cache_dir / f"{p.stem}_{h}.npz"

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[Path, tuple[np.ndarray, int]] = {}
    for path, _ in tqdm(dataset, desc="Precomputing motion+audio (event cache)", unit="video"):
        cf = _cf(path)
        if cf is not None and cf.exists():
            try:
                z = np.load(cf)
                E_t = z["E_t"]; M_t = z["M_t"]
                total_frames = int(z["total_frames"])
                A_t = compute_event_attention_from_signals(E_t, M_t, alpha=alpha, beta=beta)
                cache[path] = (A_t, total_frames)
                continue
            except Exception:
                pass
        E_t = audio_energy_from_video(path, segment_seconds=SEGMENT_SECONDS)
        M_t = motion_magnitude_downsampled(path, stride=MOTION_STRIDE, max_frames=2000)
        if len(M_t) == 0:
            continue
        A_t = compute_event_attention_from_signals(E_t, M_t, alpha=alpha, beta=beta)
        cap = load_video_capture(path)
        try:
            total_frames, fps, _, _ = get_video_info(cap)
        finally:
            cap.release()
        cache[path] = (A_t, total_frames)
        if cf is not None:
            try:
                np.savez_compressed(cf, E_t=E_t, M_t=M_t,
                                    total_frames=np.int64(total_frames),
                                    fps=np.float64(fps if fps else 30.0))
            except Exception:
                pass
    return cache


def sample_frames(
    video_path: Path,
    k: int,
    method: str,
    event_cfg: EventAgentConfig,
    event_cache: dict | None = None,
) -> list[tuple[int, np.ndarray]]:
    """
    Sample k frames from video using uniform or event-driven method.
    If event_cache is provided and path is in it, reuse cached (A_t, total_frames) for event_driven.
    Returns list of (frame_index, RGB array).
    """
    if method == "uniform":
        cap = load_video_capture(video_path)
        try:
            num_frames, _, _, _ = get_video_info(cap)
        finally:
            cap.release()
        if num_frames <= 0:
            return []
        return retrieve_frames_uniform(video_path, num_frames, k)

    # event_driven: use cache if available
    if event_cache is not None and video_path in event_cache:
        A_t, total_frames = event_cache[video_path]
    else:
        E_t = audio_energy_from_video(video_path, segment_seconds=SEGMENT_SECONDS)
        M_t = motion_magnitude_downsampled(video_path, stride=MOTION_STRIDE, max_frames=2000)
        L = len(M_t)
        if L == 0:
            return []
        A_t = compute_event_attention_from_signals(
            E_t, M_t, alpha=event_cfg.alpha, beta=event_cfg.beta
        )
        cap = load_video_capture(video_path)
        try:
            total_frames, _, _, _ = get_video_info(cap)
        finally:
            cap.release()
        if event_cache is not None:
            event_cache[video_path] = (A_t, total_frames)
    L = len(A_t)
    if L == 0:
        return []
    fps = probe_fps(video_path)
    tau_frames = tau_to_gap(event_cfg.tau_sec, fps, MOTION_STRIDE)
    top_indices = select_topk_ic(A_t, k, tau_frames=tau_frames)
    frame_indices = [min(int(i) * MOTION_STRIDE, total_frames - 1) for i in top_indices]
    frame_indices = sorted(set(frame_indices))[:k]
    frames = []
    for fi in frame_indices:
        f = load_frame_at_index(video_path, fi)
        if f is not None:
            frames.append((fi, f))
    return frames


def load_frozen_event_config(path: Path, dataset_key: str, top_k: int) -> EventAgentConfig:
    with open(path) as config_file:
        config = json.load(config_file)[dataset_key]
    return EventAgentConfig(
        alpha=float(config["alpha"]),
        beta=float(config["beta"]),
        tau_sec=float(config["tau_sec"]),
        top_k=top_k,
    )


def run_vlm(
    agent: VLMReasoningAgent,
    frames: list[tuple[int, np.ndarray]],
    prompt: str,
) -> int:
    """
    Run VLM on frames. Returns predicted class: 0 = normal, 1 = violence.
    """
    if not frames:
        return 0
    _, is_violent = agent.run([f[1] for f in frames], prompt)
    return 1 if is_violent else 0


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """
    Compute Accuracy, Precision, Recall, F1. Violence is the positive class (1).
    """
    y_t = np.array(y_true, dtype=np.int64)
    y_p = np.array(y_pred, dtype=np.int64)
    n = len(y_t)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    acc = accuracy_score(y_t, y_p)
    # When no positive in true, sklearn returns 0 by default for precision/recall
    prec = precision_score(y_t, y_p, pos_label=1, zero_division=0.0)
    rec = recall_score(y_t, y_p, pos_label=1, zero_division=0.0)
    f1 = f1_score(y_t, y_p, pos_label=1, zero_division=0.0)
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def run_evaluation(
    dataset: list[tuple[Path, int]],
    k: int,
    method: str,
    agent: VLMReasoningAgent,
    event_cfg: EventAgentConfig,
    prompt: str,
    event_cache: dict | None = None,
) -> tuple[dict[str, float], int, int]:
    """
    Run inference for all videos with given K and method.
    If event_cache is provided, event_driven reuses cached A_t/total_frames (no recompute).
    Returns (metrics_dict, total_videos_processed, total_vlm_inferences).
    """
    y_true = []
    y_pred = []
    total_frames = 0
    method_label = "Event-driven" if method == "event_driven" else "Uniform"
    desc = f"K={k} {method_label}"
    for path, gt in tqdm(dataset, desc=desc, unit="video"):
        frames = sample_frames(path, k, method, event_cfg, event_cache=event_cache)
        if not frames:
            y_pred.append(0)
            y_true.append(gt)
            continue
        pred = run_vlm(agent, frames, prompt)
        y_true.append(gt)
        y_pred.append(pred)
        total_frames += len(frames)
    metrics = compute_metrics(y_true, y_pred)
    return metrics, len(dataset), total_frames


def _results_to_dict(
    table_rows: list[tuple[str, int, float, float, float, float]],
    total_videos: int,
    per_run_log: list[tuple[str, int, int, int]],
    metadata: dict | None = None,
) -> dict:
    """Build a JSON-serializable dict of current results."""
    rows = [
        {"method": m, "k": k, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        for m, k, acc, prec, rec, f1 in table_rows
    ]
    log = [
        {"method": m, "k": k, "n_videos": n, "total_frames": tf, "avg_frames_per_video": tf / n if n else 0}
        for m, k, n, tf in per_run_log
    ]
    result = {"total_videos": total_videos, "table": rows, "per_run": log}
    if metadata is not None:
        result["metadata"] = metadata
    return result


def _print_current_table(table_rows: list[tuple[str, int, float, float, float, float]]) -> None:
    """Print just the current results table (for incremental display)."""
    if not table_rows:
        return
    print("\n  --- Current results ---")
    print("  | Method       | K | Accuracy | Precision | Recall | F1    |")
    print("  | ------------ | - | -------- | --------- | ------ | ----- |")
    for method, k, acc, prec, rec, f1 in table_rows:
        print(f"  | {method:<12} | {k} | {acc:.4f}   | {prec:.4f}     | {rec:.4f}  | {f1:.4f} |")
    print()


def print_results(
    table_rows: list[tuple[str, int, float, float, float, float]],
    total_videos: int,
    per_run_log: list[tuple[str, int, int, int]],
) -> None:
    """Print full summary table and reproducibility log. per_run_log: (method, k, n_videos, total_frames)."""
    print("\n" + "=" * 72)
    print("XD-Violence Evaluation: Event-driven vs Uniform Frame Sampling")
    print("=" * 72)
    print("\n| Method       | K | Accuracy | Precision | Recall | F1    |")
    print("| ------------ | - | -------- | --------- | ------ | ----- |")
    for method, k, acc, prec, rec, f1 in table_rows:
        print(f"| {method:<12} | {k} | {acc:.4f}   | {prec:.4f}     | {rec:.4f}  | {f1:.4f} |")
    print("\n--- Reproducibility ---")
    for method, k, n_videos, total_frames in per_run_log:
        avg_frames = total_frames / n_videos if n_videos else 0
        print(f"Videos: {n_videos}  |  Frame budget K: {k}  |  Method: {method}")
        print(f"  Total VLM inferences: {n_videos}  |  Total frames sent to VLM: {total_frames}  |  Avg frames per video: {avg_frames:.1f}")
        print()
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description="XD-Violence full evaluation pipeline")
    ap.add_argument("--root", type=Path, default=XD_VIOLENCE_ROOT, help="Dataset root")
    ap.add_argument("--max-videos", type=int, default=None, help="Cap number of test videos (default: all 504)")
    ap.add_argument("--frame-budget", type=int, default=None, choices=[1, 2, 4, 6, 8, 10, 12], help="Single K to run (default: all 2,4,6,8,10,12)")
    ap.add_argument("--method", choices=["uniform", "event_driven", "both"], default="both", help="Which sampling method(s)")
    ap.add_argument("--model-id", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results-dir", type=Path, default=None, help="Save results to this dir (e.g. results/); file: xdviolence_eval_results.json")
    ap.add_argument("--signal-cache-dir", type=Path, default=None,
                    help="If given, audio+motion signals are read/written here (skips extraction on re-run).")
    ap.add_argument("--event-config", type=Path, default=FROZEN_CONFIG_PATH,
                    help="Frozen held-out EARS configuration JSON.")
    args = ap.parse_args()

    import torch
    device = args.device
    if device == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: GPU required (--device cuda) but CUDA is not available.", file=sys.stderr)
            return 1
        try:
            torch.cuda.init()
            _ = torch.zeros(1, device="cuda")
        except RuntimeError as e:
            if "no kernel image" in str(e) or "is not compatible" in str(e):
                print("ERROR: GPU required but this PyTorch build does not support your GPU (e.g. GB10/sm_121).", file=sys.stderr)
                print("Install a compatible build, e.g. PyTorch nightly cu130: pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130", file=sys.stderr)
                return 1
            raise

    dataset = load_dataset(root=args.root, max_videos=args.max_videos)
    if not dataset:
        print("No test videos found.", file=sys.stderr)
        return 1
    n_total = len(dataset)
    n_normal = sum(1 for _, gt in dataset if gt == 0)
    n_violence = sum(1 for _, gt in dataset if gt == 1)
    print(f"Loaded {n_total} test videos from {args.root / 'test'} (Normal: {n_normal}, Violence: {n_violence})")

    budgets = [args.frame_budget] if args.frame_budget is not None else list(FRAME_BUDGETS)
    methods = ["uniform", "event_driven"] if args.method == "both" else [args.method]

    vlm_cfg = VLMConfig(model_id=args.model_id, device=device)
    agent = VLMReasoningAgent(model_id=vlm_cfg.model_id, device=vlm_cfg.device, max_new_tokens=vlm_cfg.max_new_tokens)
    agent.load()

    table_rows: list[tuple[str, int, float, float, float, float]] = []
    per_run_log: list[tuple[str, int, int, int]] = []
    results_file: Path | None = None
    if args.results_dir is not None:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        results_file = args.results_dir / "xdviolence_eval_results.json"
        print(f"Saving results to {results_file} (updated after each run)\n")

    # Precompute motion+audio once for event_driven so all K reuse the same A_t per video
    event_cache: dict | None = None
    if "event_driven" in methods:
        frozen_cfg = load_frozen_event_config(args.event_config, "xd-violence", top_k=1)
        event_cache = build_event_cache(dataset, alpha=frozen_cfg.alpha, beta=frozen_cfg.beta,
                                        cache_dir=args.signal_cache_dir)
        print(f"Event cache built for {len(event_cache)} videos; frozen config "
              f"alpha={frozen_cfg.alpha:g}, beta={frozen_cfg.beta:g}, "
              f"tau={frozen_cfg.tau_sec:g}s (reused for all K).\n")

    for k in budgets:
        event_cfg = load_frozen_event_config(args.event_config, "xd-violence", top_k=k)
        for method in methods:
            method_label = "Event-driven" if method == "event_driven" else "Uniform"
            print(f"Running K={k} {method_label} ...")
            metrics, n_proc, total_frames = run_evaluation(
                dataset, k, method, agent, event_cfg, VLM_PROMPT, event_cache=event_cache
            )
            table_rows.append((
                method_label, k,
                metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"],
            ))
            per_run_log.append((method_label, k, n_proc, total_frames))

            # Show this run and current table after each experiment
            print(f"  -> K={k} {method_label}: Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")
            _print_current_table(table_rows)

            # Incremental save to JSON
            if results_file is not None:
                data = _results_to_dict(table_rows, n_total, per_run_log, metadata={
                    "model_id": args.model_id,
                    "processor": "fast",
                    "prompt_style": "binary",
                    "event_config_path": str(args.event_config),
                    "event_config": vars(event_cfg),
                })
                with open(results_file, "w") as f:
                    json.dump(data, f, indent=2)

    print_results(table_rows, n_total, per_run_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
