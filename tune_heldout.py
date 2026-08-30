#!/usr/bin/env python3
"""Held-out hyper-parameter tuning for EARS.

This script tunes (alpha, beta, tau)
on a held-out split *disjoint from the test set* (XD-Violence train/, UCF-Crime
training videos), freezes the winner, and writes it to
results/heldout/heldout_config.json for the main-table evaluation.

Uses the same Qwen2-VL-7B backbone and binary anomaly prompt as Table 1, so the
selected config transfers directly to the reported protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config import XD_VIOLENCE_ROOT, UCF_CRIME_ROOT
from event_attention import compute_event_attention_from_signals
from ic_selection import select_topk_ic, tau_to_gap
from audio_features import audio_energy_from_video
from motion_features import motion_magnitude_downsampled
from video_loader import load_video_capture, get_video_info, probe_fps
from evaluate_xdviolence import (
    MOTION_STRIDE, SEGMENT_SECONDS, VLM_PROMPT as XD_PROMPT,
)
from evaluate_ucfcrime import UCF_PROMPT
from run_overnight_experiments import evaluate_setting, signal_to_frame_indices
from vlm_reasoning import VLMReasoningAgent

MODEL_ID_7B = "Qwen/Qwen2-VL-7B-Instruct"
SIGNAL_CACHE_DIR = _REPO / "results" / "signal_cache"
OUT_DIR = _REPO / "results" / "heldout"

# Grids (small-K regime where selection matters). beta fixed to 1 (reference).
# Both datasets sweep the same alpha values. UCF-Crime is mostly silent (223 of
# its 290 test videos carry no usable audio track), so alpha is expected to be
# inconsequential there -- but that has to come out of the search rather than be
# assumed by it, or the "no test-set tuning" claim only holds for XD.
_ALPHAS = (0.0, 1.0, 2.0)
_TAUS = (0.0, 0.5, 1.0)
XD_GRID = [(a, 1.0, t) for a in _ALPHAS for t in _TAUS]
UCF_GRID = [(a, 1.0, t) for a in _ALPHAS for t in _TAUS]
TUNE_K = [1, 2]


# --------------------------------------------------------------------------- #
# Held-out (train-split) loaders, disjoint from the evaluation test sets.
# --------------------------------------------------------------------------- #
def load_xd_train(n_per_class: int, seed: int = 7) -> list[tuple[Path, int]]:
    root = XD_VIOLENCE_ROOT / "train"
    rng = np.random.default_rng(seed)
    out: list[tuple[Path, int]] = []
    for sub, gt in (("Fighting", 1), ("Shooting", 1), ("Normal", 0)):
        d = root / sub
        if not d.is_dir():
            continue
        vids = [p for p in sorted(d.iterdir())
                if p.suffix.lower() in (".mp4", ".avi", ".mkv")]
        rng.shuffle(vids)
        take = n_per_class // 2 if gt == 1 else n_per_class
        out.extend((p, gt) for p in vids[:take])
    rng.shuffle(out)
    return out


def load_ucf_train(n_per_class: int, seed: int = 7) -> list[tuple[Path, int]]:
    root = UCF_CRIME_ROOT
    rng = np.random.default_rng(seed)
    # Anomaly training videos listed in Anomaly_Train.txt (Class/file.mp4).
    train_list = root / "Anomaly_Train.txt"
    anom_rel = []
    if train_list.exists():
        anom_rel = [ln.strip().replace("\\", "/")
                    for ln in train_list.read_text().splitlines() if ln.strip()]
    anom_set = set(anom_rel)
    anomalies: list[Path] = []
    for part in sorted(root.glob("Anomaly-Videos-Part-*")):
        for cls in sorted(part.iterdir()):
            if not cls.is_dir():
                continue
            for p in sorted(cls.iterdir()):
                rel = f"{cls.name}/{p.name}"
                if rel in anom_set and p.suffix.lower() == ".mp4":
                    anomalies.append(p)
    normals: list[Path] = []
    for part in ("Training-Normal-Videos-Part-1", "Training-Normal-Videos-Part-2"):
        d = root / part
        if d.is_dir():
            normals.extend(p for p in sorted(d.iterdir())
                           if p.suffix.lower() == ".mp4")
    rng.shuffle(anomalies)
    rng.shuffle(normals)
    out = [(p, 1) for p in anomalies[:n_per_class]] + \
          [(p, 0) for p in normals[:n_per_class]]
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------- #
# Raw signal loading (E_t, M_t, total_frames, fps), cached per video.
# --------------------------------------------------------------------------- #
def _cf(p: Path) -> Path:
    h = hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:16]
    return SIGNAL_CACHE_DIR / f"{p.stem}_{h}.npz"


def load_raw_signals(dataset) -> dict:
    from tqdm import tqdm
    SIGNAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sig: dict[Path, tuple] = {}
    for path, _ in tqdm(dataset, desc="raw signals", unit="vid"):
        cf = _cf(path)
        if cf.exists():
            try:
                z = np.load(cf)
                # The cached `fps` field predates the get_video_info unpacking
                # fix and holds the frame width, so re-probe instead of trusting
                # it. Probing reads a container property and decodes nothing.
                sig[path] = (z["E_t"], z["M_t"], int(z["total_frames"]),
                             probe_fps(path))
                continue
            except Exception:
                pass
        E_t = audio_energy_from_video(path, segment_seconds=SEGMENT_SECONDS)
        M_t = motion_magnitude_downsampled(path, stride=MOTION_STRIDE, max_frames=2000)
        if len(M_t) == 0:
            continue
        cap = load_video_capture(path)
        try:
            total_frames, fps, _, _ = get_video_info(cap)
        finally:
            cap.release()
        fps = float(fps) if fps else 30.0
        sig[path] = (E_t, M_t, total_frames, fps)
        try:
            np.savez_compressed(cf, E_t=E_t, M_t=M_t,
                                total_frames=np.int64(total_frames),
                                fps=np.float64(fps))
        except Exception:
            pass
    return sig


def make_sampler(sig, alpha, beta, tau_sec, k):
    def sampler(path, gt, i):
        if path not in sig:
            return []
        E_t, M_t, total_frames, fps = sig[path]
        A_t = compute_event_attention_from_signals(E_t, M_t, alpha=alpha, beta=beta)
        sig_idx = select_topk_ic(
            A_t, k=k, tau_frames=tau_to_gap(tau_sec, fps, MOTION_STRIDE)
        )
        return signal_to_frame_indices(sig_idx, total_frames, k)
    return sampler


def tune(agent, dataset, prompt, grid, name, ks=None):
    """Grid-search (alpha, beta, tau) on a split disjoint from the test set.

    `ks` restricts which budgets are evaluated. At K=1 the IC gap cannot bind --
    a single frame has no neighbour to suppress -- so tau-only re-runs may pass
    `ks=[2]` and reuse the K=1 column from an earlier run.
    """
    ks = list(ks or TUNE_K)
    sig = load_raw_signals(dataset)
    print(f"[{name}] held-out videos with signals: {len(sig)}/{len(dataset)}")
    results = []
    for (alpha, beta, tau) in grid:
        f1s = []
        for k in ks:
            r = evaluate_setting(dataset=dataset, agent=agent, prompt=prompt,
                                 sampler=make_sampler(sig, alpha, beta, tau, k),
                                 desc=f"{name} a={alpha} t={tau} K={k}")
            f1s.append(r["metrics"]["f1"])
        mean_f1 = float(np.mean(f1s))
        results.append({"alpha": alpha, "beta": beta, "tau_sec": tau,
                        "k_f1": dict(zip(ks, f1s)), "mean_f1": mean_f1})
        print(f"[{name}] a={alpha} b={beta} tau={tau}: "
              f"K-F1={[round(x,3) for x in f1s]} mean={mean_f1:.3f}")
    best = max(results, key=lambda r: r["mean_f1"])
    print(f"[{name}] BEST: alpha={best['alpha']} beta={best['beta']} "
          f"tau={best['tau_sec']} mean_f1={best['mean_f1']:.3f}")
    return {"grid_results": results, "best": best,
            "n_videos": len(dataset), "tune_k": ks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["xd", "ucf", "both"], default="both")
    ap.add_argument("--n-per-class", type=int, default=60)
    ap.add_argument("--model-id", default=MODEL_ID_7B)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ks", nargs="+", type=int, default=None,
                    help="budgets to evaluate (default 1 2); pass 2 to re-run "
                         "only the tau-sensitive column")
    ap.add_argument("--out-name", default="heldout_tuning.json",
                    help="filename under results/heldout/")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agent = VLMReasoningAgent(model_id=args.model_id, device=args.device)
    out = {}
    if args.dataset in ("xd", "both"):
        xd = load_xd_train(args.n_per_class)
        out["xd-violence"] = tune(agent, xd, XD_PROMPT, XD_GRID, "xd-violence", args.ks)
    if args.dataset in ("ucf", "both"):
        ucf = load_ucf_train(args.n_per_class)
        out["ucf-crime"] = tune(agent, ucf, UCF_PROMPT, UCF_GRID, "ucf-crime", args.ks)

    with open(OUT_DIR / args.out_name, "w") as f:
        json.dump(out, f, indent=2, default=str)
    if args.ks and list(args.ks) != list(TUNE_K):
        # A partial sweep cannot freeze the config on its own: the selection
        # criterion is the mean over TUNE_K, and this run only measured some of
        # it. Merge with the reusable columns first.
        print(f"Partial sweep (K={args.ks}); not overwriting heldout_config.json.")
        return 0
    cfg = {ds: {"alpha": v["best"]["alpha"], "beta": v["best"]["beta"],
                "tau_sec": v["best"]["tau_sec"]} for ds, v in out.items()}
    with open(OUT_DIR / "heldout_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("Frozen held-out config:", json.dumps(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
