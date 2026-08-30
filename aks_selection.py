#!/usr/bin/env python3
"""Faithful reimplementation of AKS (Adaptive Keyframe Sampling, CVPR 2025).

Reference: Tang et al., "Adaptive Keyframe Sampling for Long Video
Understanding", CVPR 2025 (arXiv:2502.21271). Official code:
https://github.com/ncTimTang/AKS (frame_select.py -> ``meanstd`` recursion).

This module ports the exact selection algorithm from the official
``frame_select.py`` so it can be used as a query-aware baseline selector in
the EARS pipeline. Given a per-frame relevance score array (e.g. image-text
similarity to the prompt), ``select_aks`` returns the frame indices AKS would
choose under a budget ``k``.

The algorithm recursively bisects the timeline. A segment stops splitting
("no split") when its top-``k`` scores are substantially higher than the
segment mean (``mean_diff > t1`` and ``std > t2``) -- i.e. the relevance is
concentrated -- or when the maximum recursion depth is reached. Each resulting
leaf at depth ``d`` is allocated ``k / 2**d`` frames (its top scorers). This
jointly optimises relevance (peaked leaves keep more frames) and coverage
(the bisection spreads leaves across time).
"""
from __future__ import annotations

import heapq

import numpy as np

# AKS default hyper-parameters (frame_select.py argparse defaults).
AKS_T1 = 0.8
AKS_T2 = -100.0
AKS_ALL_DEPTH = 5


def _meanstd(dic_scores, fns, n, t1, t2, all_depth):
    """Direct port of AKS ``meanstd`` recursion (frame_select.py)."""
    split_scores, split_fn = [], []
    no_split_scores, no_split_fn = [], []
    for dic_score, fn in zip(dic_scores, fns):
        score = dic_score["score"]
        depth = dic_score["depth"]
        mean = float(np.mean(score))
        std = float(np.std(score))
        top_n = heapq.nlargest(min(n, len(score)), range(len(score)),
                               score.__getitem__)
        top_score = [score[t] for t in top_n]
        mean_diff = float(np.mean(top_score)) - mean
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth and len(score) > 1:
            half = len(score) // 2
            split_scores.append(dict(score=score[:half], depth=depth + 1))
            split_scores.append(dict(score=score[half:], depth=depth + 1))
            split_fn.append(fn[:half])
            split_fn.append(fn[half:])
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
    if split_scores:
        rec_score, rec_fn = _meanstd(split_scores, split_fn, n, t1, t2,
                                     all_depth)
    else:
        rec_score, rec_fn = [], []
    return no_split_scores + rec_score, no_split_fn + rec_fn


def select_aks(
    scores,
    k: int,
    t1: float = AKS_T1,
    t2: float = AKS_T2,
    all_depth: int = AKS_ALL_DEPTH,
) -> np.ndarray:
    """Select ``k`` frame indices from a relevance-score array using AKS.

    Args
    ----
    scores: (T,) per-frame relevance scores (higher = more relevant).
    k:      frame budget.
    t1,t2,all_depth: AKS hyper-parameters (paper defaults).

    Returns
    -------
    np.ndarray of up to ``k`` sorted integer indices into ``scores``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    T = len(scores)
    if T == 0 or k <= 0:
        return np.array([], dtype=int)
    if T <= k:
        return np.arange(T, dtype=int)

    smin, smax = float(np.min(scores)), float(np.max(scores))
    if smax - smin < 1e-8:
        # Flat relevance -> AKS degenerates to uniform coverage.
        return np.unique(np.linspace(0, T - 1, k).round().astype(int))

    norm = (scores - smin) / (smax - smin)  # AKS normalises to [0, 1]
    leaves_score, leaves_fn = _meanstd(
        [dict(score=norm, depth=0)], [list(range(T))], k, t1, t2, all_depth
    )

    # AKS depth-weighted allocation: leaf at depth d contributes int(k / 2**d)
    # of its top-scoring frames. This is designed for large budgets (k=64);
    # at the tight budgets EARS targets (k<=10) the deep leaves get 0 frames.
    out: list[int] = []
    for s, f in zip(leaves_score, leaves_fn):
        f_num = int(k / (2 ** s["depth"]))
        if f_num < 1:
            continue
        sc = s["score"]
        topk = heapq.nlargest(min(f_num, len(sc)), range(len(sc)),
                              sc.__getitem__)
        out.extend(int(f[t]) for t in topk)

    out = sorted(set(out))
    if len(out) > k:
        return np.array(
            sorted(sorted(out, key=lambda idx: scores[idx], reverse=True)[:k]),
            dtype=int,
        )
    if len(out) == k:
        return np.array(out, dtype=int)

    # Small-k reconciliation: instead of uniform padding, add the argmax frame
    # of each remaining leaf, ordered by leaf peak relevance. This keeps AKS
    # relevance-driven (one high-evidence frame per adaptive segment) rather
    # than collapsing to uniform coverage when the depth allocation under-fills.
    chosen = set(out)
    leaf_peaks = []
    for s, f in zip(leaves_score, leaves_fn):
        sc = s["score"]
        j = int(max(range(len(sc)), key=sc.__getitem__))
        fi = int(f[j])
        if fi not in chosen:
            leaf_peaks.append((float(sc[j]), fi))
    leaf_peaks.sort(reverse=True)
    for _, fi in leaf_peaks:
        if len(out) >= k:
            break
        if fi not in chosen:
            out.append(fi)
            chosen.add(fi)
    if len(out) < k:  # last resort: top remaining frames by global relevance
        for fi in sorted(range(T), key=lambda i: scores[i], reverse=True):
            if len(out) >= k:
                break
            if fi not in chosen:
                out.append(fi)
                chosen.add(fi)
    return np.array(sorted(out), dtype=int)


if __name__ == "__main__":
    # Tiny self-test: a peaked signal should concentrate; flat -> uniform.
    rng = np.random.default_rng(0)
    peaked = np.zeros(100)
    peaked[40:45] = 1.0
    peaked += 0.01 * rng.standard_normal(100)
    print("peaked K=4 ->", select_aks(peaked, 4))
    print("flat   K=4 ->", select_aks(np.ones(100), 4))
