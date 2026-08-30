"""
Evaluation: run event-driven vs uniform retrieval, call VLM, compute accuracy and frame count.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable

import numpy as np

from config import EventAgentConfig, VLM_PROMPT, VLMConfig
from video_loader import get_video_info, load_video_capture
from audio_features import audio_energy_from_video
from motion_features import motion_magnitude_downsampled
from event_attention import compute_event_attention_from_signals
from frame_retrieval import (
    retrieve_frames_event_driven,
    retrieve_frames_uniform,
)
from vlm_reasoning import run_vlm_on_retrieved_frames


def run_single_video_event_driven(
    video_path: Path,
    event_cfg: EventAgentConfig,
    vlm_cfg: VLMConfig,
    segment_seconds: float = 1.0,
    motion_stride: int = 5,
) -> tuple[bool, int, str]:
    """
    Run event-driven pipeline: audio + motion -> A_t -> top-K frames -> VLM.
    Returns (predicted_violent, num_frames_sent_to_vlm, raw_answer).
    """
    E_t = audio_energy_from_video(video_path, segment_seconds=segment_seconds)
    M_t = motion_magnitude_downsampled(video_path, stride=motion_stride, max_frames=2000)
    L = len(M_t)
    if L == 0:
        return False, 0, "no_frames"
    A_t = compute_event_attention_from_signals(
        E_t, M_t, alpha=event_cfg.alpha, beta=event_cfg.beta
    )
    # Top-K indices are into downsampled grid; map to frame indices
    from frame_retrieval import select_top_k_indices
    from video_loader import get_video_info, load_video_capture
    cap = load_video_capture(video_path)
    try:
        total_frames, _, _, _ = get_video_info(cap)
    finally:
        cap.release()
    top_indices = select_top_k_indices(A_t, event_cfg.top_k)
    frame_indices = [min(int(i) * motion_stride, total_frames - 1) for i in top_indices]
    frame_indices = sorted(set(frame_indices))[: event_cfg.top_k]
    # Load frames by frame index
    from video_loader import load_frame_at_index
    frames = []
    for fi in frame_indices:
        f = load_frame_at_index(video_path, fi)
        if f is not None:
            frames.append((fi, f))
    if not frames:
        return False, 0, "no_frames"
    answer, is_violent = run_vlm_on_retrieved_frames(
        frames, VLM_PROMPT,
        model_id=vlm_cfg.model_id,
        device=vlm_cfg.device,
    )
    return is_violent, len(frames), answer


def run_single_video_uniform(
    video_path: Path,
    k: int,
    vlm_cfg: VLMConfig,
) -> tuple[bool, int, str]:
    """Uniform sampling: k frames -> VLM. Returns (predicted_violent, k, raw_answer)."""
    cap = load_video_capture(video_path)
    try:
        num_frames, _, _, _ = get_video_info(cap)
    finally:
        cap.release()
    if num_frames <= 0:
        return False, 0, "no_frames"
    frames = retrieve_frames_uniform(video_path, num_frames, k)
    if not frames:
        return False, 0, "no_frames"
    answer, is_violent = run_vlm_on_retrieved_frames(
        frames, VLM_PROMPT,
        model_id=vlm_cfg.model_id,
        device=vlm_cfg.device,
    )
    return is_violent, len(frames), answer


def evaluate(
    video_list: list[tuple[Path, bool]],
    event_cfg: EventAgentConfig,
    vlm_cfg: VLMConfig,
    use_event_driven: bool = True,
    segment_seconds: float = 1.0,
    motion_stride: int = 5,
) -> dict:
    """
    Run on video_list (path, ground_truth_violent).
    Returns dict with accuracy, total_frames_used, num_videos, etc.
    """
    correct = 0
    total_frames = 0
    results = []
    for path, gt in video_list:
        if use_event_driven:
            pred, nf, raw = run_single_video_event_driven(
                path, event_cfg, vlm_cfg,
                segment_seconds=segment_seconds,
                motion_stride=motion_stride,
            )
        else:
            pred, nf, raw = run_single_video_uniform(path, event_cfg.top_k, vlm_cfg)
        total_frames += nf
        correct += 1 if (pred == gt) else 0
        results.append({"path": str(path), "gt": gt, "pred": pred, "frames": nf, "raw": raw})
    n = len(video_list)
    return {
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "total": n,
        "total_frames_processed": total_frames,
        "avg_frames_per_video": total_frames / n if n else 0,
        "use_event_driven": use_event_driven,
        "results": results,
    }


def compare_uniform_vs_event_driven(
    video_list: list[tuple[Path, bool]],
    event_cfg: EventAgentConfig,
    vlm_cfg: VLMConfig,
    segment_seconds: float = 1.0,
    motion_stride: int = 5,
) -> dict:
    """Run both methods and return metrics for each."""
    eval_ed = evaluate(
        video_list, event_cfg, vlm_cfg,
        use_event_driven=True,
        segment_seconds=segment_seconds,
        motion_stride=motion_stride,
    )
    eval_uni = evaluate(
        video_list, event_cfg, vlm_cfg,
        use_event_driven=False,
    )
    return {
        "event_driven": eval_ed,
        "uniform": eval_uni,
    }
