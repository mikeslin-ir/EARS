#!/usr/bin/env python3
"""
Full evaluation pipeline for UCF-Crime: event-driven vs uniform frame sampling
for video anomaly detection using a VLM. Same pipeline as XD-Violence; reuses
event cache, sampling, and VLM logic.

Note: Standard UCF-Crime benchmarks report frame-level AUC (e.g. 84–90%) with
weakly-supervised models over many frames. We do video-level binary classification
with a VLM and few frames (K=1–10), so metrics are not directly comparable.
UCF-Crime test videos have no/minimal audio; event attention uses motion only (alpha=0).

Usage:
  python evaluate_ucfcrime.py
  python evaluate_ucfcrime.py --max-videos 20 --frame-budget 4 --method uniform
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
from tqdm import tqdm

from config import EventAgentConfig, VLMConfig, UCF_CRIME_ROOT
from datasets import list_ucf_crime_test
from vlm_reasoning import VLMReasoningAgent

# Reuse pipeline from XD-Violence evaluation
from evaluate_xdviolence import (
    FRAME_BUDGETS,
    build_event_cache,
    compute_metrics,
    run_evaluation,
    _results_to_dict,
    _print_current_table,
    sample_frames,
    run_vlm,
    load_frozen_event_config,
    FROZEN_CONFIG_PATH,
)

# UCF-Crime test videos typically have no/minimal audio; use motion-only event attention (alpha=0).
UCF_ALPHA = 0.0
UCF_BETA = 1.0
UCF_PROMPT = (
    "These frames are from a surveillance video. The video may show: fighting, robbery, abuse, "
    "road accidents, burglary, vandalism, explosions, or other crimes. "
    "If any anomalous or criminal event is visible in these frames, answer Yes. "
    "If the frames show only normal, safe activity, answer No. Answer with one word: Yes or No."
)

# Multi-choice prompt: forces VLM to commit to a class. Any non-Normal answer is
# treated as a positive (anomaly) prediction by VLMReasoningAgent's parser.
UCF_PROMPT_MULTICHOICE = (
    "These frames are from a surveillance video. Choose exactly one category that best "
    "describes what is happening. Reply with only the category name.\n"
    "Categories: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, "
    "RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism, Normal."
)


def load_dataset(
    root: Path | None = None,
    max_videos: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[tuple[Path, int]]:
    """
    Load UCF-Crime test set. Returns list of (video_path, ground_truth).
    Ground truth: 0 = Normal, 1 = Anomaly. Uses datasets.list_ucf_crime_test.
    """
    root = root or UCF_CRIME_ROOT
    items = list_ucf_crime_test(root=root)
    # Convert (Path, bool) -> (Path, int): False=0 (Normal), True=1 (Anomaly)
    out: list[tuple[Path, int]] = [(p, 1 if is_anom else 0) for p, is_anom in items]
    if shuffle and out:
        rng = np.random.default_rng(seed)
        rng.shuffle(out)
    if max_videos is not None and max_videos > 0:
        out = out[:max_videos]
    return out


def print_results(
    table_rows: list[tuple[str, int, float, float, float, float]],
    total_videos: int,
    per_run_log: list[tuple[str, int, int, int]],
) -> None:
    """Print full summary table and reproducibility log (UCF-Crime)."""
    print("\n" + "=" * 72)
    print("UCF-Crime Evaluation: Event-driven vs Uniform Frame Sampling")
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


def _ucf_class_from_path(p: Path) -> str:
    """Extract UCF-Crime class label from a video path.

    Anomaly videos live in `Anomaly-Videos-Part-N/<Class>/file.mp4`.
    Normal test videos live under `Testing_Normal_Videos_Anomaly/` or
    similar; we map all such files to ``Normal``.
    """
    parts = [s for s in p.parts]
    parent = p.parent.name
    if "Normal" in parent or any("Normal" in s for s in parts[-3:-1]):
        return "Normal"
    return parent


def run_evaluation_per_video(
    dataset: list[tuple[Path, int]],
    k: int,
    method: str,
    agent: VLMReasoningAgent,
    event_cfg: EventAgentConfig,
    prompt: str,
    event_cache: dict | None = None,
) -> tuple[dict[str, float], int, int, list[dict]]:
    """Like ``run_evaluation`` but also returns a per-video prediction log."""
    y_true: list[int] = []
    y_pred: list[int] = []
    per_video: list[dict] = []
    total_frames = 0
    method_label = "Event-driven" if method == "event_driven" else "Uniform"
    desc = f"K={k} {method_label}"
    for path, gt in tqdm(dataset, desc=desc, unit="video"):
        frames = sample_frames(path, k, method, event_cfg, event_cache=event_cache)
        if not frames:
            y_pred.append(0)
            y_true.append(gt)
            per_video.append({
                "video": str(path),
                "class": _ucf_class_from_path(path),
                "gt": gt,
                "pred": 0,
                "n_frames": 0,
            })
            continue
        pred = run_vlm(agent, frames, prompt)
        y_true.append(gt)
        y_pred.append(pred)
        per_video.append({
            "video": str(path),
            "class": _ucf_class_from_path(path),
            "gt": gt,
            "pred": int(pred),
            "n_frames": len(frames),
        })
        total_frames += len(frames)
    metrics = compute_metrics(y_true, y_pred)
    return metrics, len(dataset), total_frames, per_video


def main() -> int:
    ap = argparse.ArgumentParser(description="UCF-Crime full evaluation pipeline")
    ap.add_argument("--root", type=Path, default=UCF_CRIME_ROOT, help="UCF-Crime dataset root")
    ap.add_argument("--max-videos", type=int, default=None, help="Cap number of test videos (default: all)")
    ap.add_argument("--frame-budget", type=int, default=None, choices=[1, 2, 4, 6, 8, 10, 12], help="Single K to run (default: all)")
    ap.add_argument("--method", choices=["uniform", "event_driven", "both"], default="both", help="Which sampling method(s)")
    ap.add_argument("--model-id", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-style", choices=["binary", "multichoice"], default="binary",
                    help="binary: Yes/No anomaly prompt (default). multichoice: pick one of 13 UCF classes + Normal.")
    ap.add_argument("--results-name", type=str, default="ucfcrime_eval_results.json",
                    help="Filename inside --results-dir.")
    ap.add_argument("--results-dir", type=Path, default=None, help="Save results to this dir")
    ap.add_argument("--signal-cache-dir", type=Path, default=None,
                    help="If given, audio+motion signals are read/written here (skips extraction on re-run).")
    ap.add_argument("--event-config", type=Path, default=FROZEN_CONFIG_PATH,
                    help="Frozen held-out EARS configuration JSON.")
    ap.add_argument("--save-per-video", action="store_true",
                    help="Also dump per-video (path, class, gt, pred) JSON for per-class analysis.")
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
                print("ERROR: GPU required but this PyTorch build does not support your GPU.", file=sys.stderr)
                return 1
            raise

    dataset = load_dataset(root=args.root, max_videos=args.max_videos)
    if not dataset:
        print("No test videos found.", file=sys.stderr)
        return 1
    n_total = len(dataset)
    n_normal = sum(1 for _, gt in dataset if gt == 0)
    n_anomaly = sum(1 for _, gt in dataset if gt == 1)
    print(f"Loaded {n_total} test videos from {args.root} (Normal: {n_normal}, Anomaly: {n_anomaly})")

    budgets = [args.frame_budget] if args.frame_budget is not None else list(FRAME_BUDGETS)
    methods = ["uniform", "event_driven"] if args.method == "both" else [args.method]

    vlm_cfg = VLMConfig(model_id=args.model_id, device=device)
    agent = VLMReasoningAgent(model_id=vlm_cfg.model_id, device=vlm_cfg.device, max_new_tokens=vlm_cfg.max_new_tokens)
    agent.load()

    table_rows: list[tuple[str, int, float, float, float, float]] = []
    per_run_log: list[tuple[str, int, int, int]] = []
    prompt = UCF_PROMPT_MULTICHOICE if args.prompt_style == "multichoice" else UCF_PROMPT
    print(f"Prompt style: {args.prompt_style}")

    results_file: Path | None = None
    if args.results_dir is not None:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        results_file = args.results_dir / args.results_name
        print(f"Saving results to {results_file} (updated after each run)\n")

    event_cache = None
    if "event_driven" in methods:
        frozen_cfg = load_frozen_event_config(args.event_config, "ucf-crime", top_k=1)
        event_cache = build_event_cache(dataset, alpha=frozen_cfg.alpha, beta=frozen_cfg.beta,
                                        cache_dir=args.signal_cache_dir)
        print(f"Event cache built for {len(event_cache)} videos; frozen config "
              f"alpha={frozen_cfg.alpha:g}, beta={frozen_cfg.beta:g}, "
              f"tau={frozen_cfg.tau_sec:g}s (reused for all K).\n")

    for k in budgets:
        event_cfg = load_frozen_event_config(args.event_config, "ucf-crime", top_k=k)
        for method in methods:
            method_label = "Event-driven" if method == "event_driven" else "Uniform"
            print(f"Running K={k} {method_label} ...")
            if args.save_per_video:
                metrics, n_proc, total_frames, per_video = run_evaluation_per_video(
                    dataset, k, method, agent, event_cfg, prompt, event_cache=event_cache
                )
            else:
                metrics, n_proc, total_frames = run_evaluation(
                    dataset, k, method, agent, event_cfg, prompt, event_cache=event_cache
                )
                per_video = None
            table_rows.append((
                method_label, k,
                metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"],
            ))
            per_run_log.append((method_label, k, n_proc, total_frames))
            print(f"  -> K={k} {method_label}: Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")
            _print_current_table(table_rows)
            if results_file is not None:
                data = _results_to_dict(table_rows, n_total, per_run_log, metadata={
                    "model_id": args.model_id,
                    "processor": "fast",
                    "prompt_style": args.prompt_style,
                    "event_config_path": str(args.event_config),
                    "event_config": vars(event_cfg),
                })
                with open(results_file, "w") as f:
                    json.dump(data, f, indent=2)
            if per_video is not None and args.results_dir is not None:
                pv_path = args.results_dir / f"per_video_K{k}_{method}.json"
                with open(pv_path, "w") as f:
                    json.dump(per_video, f, indent=2)
                print(f"  per-video log -> {pv_path}")

    print_results(table_rows, n_total, per_run_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
