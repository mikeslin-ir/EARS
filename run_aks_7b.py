#!/usr/bin/env python3
"""AKS baseline (query-aware keyframe selection) on UCF-Crime & XD-Violence.

Rebuttal experiment E2. Runs the AKS selector (aks_selection.select_aks) with
SigLIP image-text relevance to a fixed anomaly query set, on the same
Qwen2-VL-7B backbone and binary anomaly prompt as the main table, so it is a
head-to-head competitor to Uniform / EARS at each budget K.

AKS is query-aware; the anomaly task has no per-video query, so we use a small
fixed set of anomaly text prompts and take the max image-text similarity per
frame as the relevance signal (this is the standard max-over-options reduction
already used by EARS's semantic channel).

Signals reused from cache:
  - Motion/audio event cache (results/signal_cache) -> gives total_frames and T.
  - SigLIP per-frame embeddings cached in results/aks_semantic_cache.

Usage:
  python run_aks_7b.py --dataset both --k 1 2 4 6 8 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from aks_selection import select_aks


def select_topk(scores, k: int):
    """CLIP/SigLIP top-K selector: the k frames with the highest relevance.

    A query-aware, training-free baseline that shares AKS's relevance signal
    (max SigLIP image-text similarity to the anomaly queries) but replaces
    AKS's mean-std coverage recursion with a plain top-k over that relevance.
    Returns up to ``k`` sorted integer indices, matching ``select_aks``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    T = len(scores)
    if T == 0 or k <= 0:
        return np.array([], dtype=int)
    if T <= k:
        return np.arange(T, dtype=int)
    idx = np.argsort(scores)[-k:]
    return np.array(sorted(int(i) for i in idx), dtype=int)


from semantic_features import (
    SemanticConfig,
    SemanticEncoder,
    build_semantic_cache,
    compute_semantic_signal,
)
import run_overnight_experiments as roe
from run_overnight_experiments import (
    evaluate_setting,
    signal_to_frame_indices,
    load_xd,
    load_ucf,
    XD_PROMPT,
    UCF_PROMPT,
    build_event_cache,
    log,
)
from vlm_reasoning import VLMReasoningAgent

MODEL_ID_7B = "Qwen/Qwen2-VL-7B-Instruct"
SIGNAL_CACHE_DIR = _REPO / "results" / "signal_cache"
SEMANTIC_CACHE_DIR = _REPO / "results" / "aks_semantic_cache"
OUT_DIR = _REPO / "results" / "aks"

# Fixed anomaly query sets (query-aware relevance for a query-free task).
XD_QUERIES = [
    "a violent scene", "people fighting", "a shooting", "an explosion",
    "a car accident", "an assault", "abuse", "a riot",
]
UCF_QUERIES = [
    "a crime being committed", "an anomalous event", "fighting or assault",
    "robbery or burglary", "arson or a fire", "a road accident",
    "shoplifting or stealing", "vandalism",
]


def build_relevance_cache(dataset, event_cache, queries, encoder):
    """S_t (SigLIP relevance to anomaly queries) per video, index-aligned to A_t."""
    target_T = {p: int(len(event_cache[p][0])) for p, _ in dataset
                if p in event_cache}
    frame_emb = build_semantic_cache(
        dataset, encoder, SEMANTIC_CACHE_DIR, target_T,
    )
    text_emb = encoder.encode_texts(queries)
    rel: dict[Path, np.ndarray] = {}
    for p, _ in dataset:
        if p in frame_emb and frame_emb[p].shape[0] > 0:
            rel[p] = compute_semantic_signal(frame_emb[p], text_emb, mode="max")
    return rel


def run_dataset(agent, dataset, prompt, queries, name, k_list, encoder,
                select_fn=select_aks, method="aks", out_dir=OUT_DIR,
                file_prefix="aks7b"):
    tag = method.upper()
    log(f"{tag}@7B on {name}: building event cache ...")
    event_cache = build_event_cache(dataset, alpha=1.0, beta=1.0,
                                    cache_dir=SIGNAL_CACHE_DIR)
    log(f"{tag}@7B on {name}: building SigLIP relevance cache ...")
    rel = build_relevance_cache(dataset, event_cache, queries, encoder)
    log(f"{tag}@7B on {name}: {len(rel)}/{len(dataset)} videos have relevance.")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_prefix}_{name}.json"
    all_results = []
    for k in k_list:
        def sampler(path, gt, i, _k=k):
            if path not in rel or path not in event_cache:
                return []
            S_t = rel[path]
            _, total_frames = event_cache[path]
            sig_idx = select_fn(S_t, _k)
            return signal_to_frame_indices(sig_idx, total_frames, _k)

        r = evaluate_setting(dataset=dataset, agent=agent, prompt=prompt,
                             sampler=sampler, desc=f"{tag} {name} K={k}")
        r.update({"dataset": name, "k": k, "method": method,
                  "model": MODEL_ID_7B})
        all_results.append(r)
        log(f"  -> {tag} {name} K={k}: Acc={r['metrics']['accuracy']:.4f} "
            f"F1={r['metrics']['f1']:.4f}  ({r['elapsed_sec']/60:.1f} min)")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    log(f"{tag}@7B on {name}: saved -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["xd", "ucf", "both"], default="both")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10])
    ap.add_argument("--selector", choices=["aks", "topk"], default="aks",
                    help="aks = mean-std coverage recursion; "
                         "topk = CLIP/SigLIP top-K over the same relevance.")
    ap.add_argument("--model-id", default=MODEL_ID_7B)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-videos", type=int, default=None)
    args = ap.parse_args()

    roe.MODEL_ID = args.model_id
    if args.selector == "topk":
        select_fn, method = select_topk, "clip_topk"
        out_dir, file_prefix = _REPO / "results" / "topk", "topk7b"
    else:
        select_fn, method = select_aks, "aks"
        out_dir, file_prefix = OUT_DIR, "aks7b"
    log(f"{method.upper()}@7B sweep  K={args.k}  model={args.model_id}")

    encoder = SemanticEncoder(SemanticConfig(device=args.device))
    agent = VLMReasoningAgent(model_id=args.model_id, device=args.device)

    if args.dataset in ("xd", "both"):
        xd = load_xd()
        if args.max_videos:
            xd = xd[: args.max_videos]
        run_dataset(agent, xd, XD_PROMPT, XD_QUERIES, "xd-violence",
                    args.k, encoder, select_fn=select_fn, method=method,
                    out_dir=out_dir, file_prefix=file_prefix)
    if args.dataset in ("ucf", "both"):
        ucf = load_ucf()
        if args.max_videos:
            ucf = ucf[: args.max_videos]
        run_dataset(agent, ucf, UCF_PROMPT, UCF_QUERIES, "ucf-crime",
                    args.k, encoder, select_fn=select_fn, method=method,
                    out_dir=out_dir, file_prefix=file_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
