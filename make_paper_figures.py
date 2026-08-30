#!/usr/bin/env python3
"""
Generate paper figures from existing JSON results.

Produces figures under assets/figures/.

  fig_f1_vs_k.pdf          -- F1 vs K, UCF-Crime + XD-Violence, EARS vs Uniform
  fig_recall_vs_k.pdf      -- Same axes, recall (cleaner story for K=1,2)
  fig_per_class_ucf.pdf    -- Per-class F1 bar chart on UCF-Crime (needs raw per-video preds)
  fig_adaptive_pareto.pdf  -- F1 vs avg frames/video (filled once ablation runs)
  fig_framelevel_curves.pdf-- PR + ROC on XD frame-level (re-styled, with our 3 variants)

Each plot has paper-friendly sizing for a single-column width (~3.3in).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO   = Path(__file__).resolve().parent
RESULTS = REPO / "results"
FIGDIR  = REPO / "assets" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

UNIFORM = dict(color="#888888", marker="o", linestyle="--")
EARS    = dict(color="#c0392b", marker="s", linestyle="-")


# ---------------------------------------------------------------------------
# 1. F1 / Recall vs K (UCF-Crime + XD-Violence)
# ---------------------------------------------------------------------------
def _split_by_method(rows: list[dict]):
    out: dict[str, dict[int, dict]] = {}
    for r in rows:
        out.setdefault(r["method"], {})[int(r["k"])] = r
    return out


def plot_metric_vs_k(metric: str, fname: str, ylabel: str) -> Path:
    headline_dir = RESULTS / "headline"
    ucf = json.loads((headline_dir / "ucfcrime_ears_qwen2vl7b.json").read_text())["table"]
    ucf += json.loads((headline_dir / "ucfcrime_uniform_qwen2vl7b.json").read_text())["table"]
    xd = json.loads((headline_dir / "xdviolence_ears_qwen2vl7b.json").read_text())["table"]
    xd_uniform = json.loads(
        (headline_dir / "xdviolence_uniform_qwen2vl7b.json").read_text()
    )["rows"]
    xd += [
        {"method": "Uniform", "k": row["k"], **row["metrics"]}
        for row in xd_uniform
    ]
    ucf_s, xd_s = _split_by_method(ucf), _split_by_method(xd)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2), sharey=True)

    for ax, store, title in [(axes[0], ucf_s, "UCF-Crime"),
                             (axes[1], xd_s,  "XD-Violence")]:
        for m, style, label in [("Uniform", UNIFORM, "Uniform"),
                                ("Event-driven", EARS, "EARS (Ours)")]:
            ks = sorted(store.get(m, {}).keys())
            ys = [store[m][k][metric] for k in ks]
            ax.plot(ks, ys, label=label, **style)
        ax.set_title(title)
        ax.set_xlabel("Frame budget $K$")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_xticks(sorted(set(list(store.get("Uniform", {}).keys()) +
                                 list(store.get("Event-driven", {}).keys()))))
    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc="lower right", frameon=True)

    out = FIGDIR / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"[wrote] {out}")
    return out


# ---------------------------------------------------------------------------
# 2. Frame-level PR + ROC curves (composite figure)
# ---------------------------------------------------------------------------
def plot_framelevel_composite() -> Path | None:
    pr_src  = RESULTS / "framelevel_ap" / "framelevel_pr_curves.png"
    roc_src = RESULTS / "framelevel_ap" / "framelevel_roc_curves.png"
    if not pr_src.exists() or not roc_src.exists():
        print("[skip] framelevel PR/ROC images missing")
        return None
    from matplotlib.image import imread
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, img_path, title in [(axes[0], pr_src, "Precision-Recall"),
                                (axes[1], roc_src, "ROC")]:
        ax.imshow(imread(img_path))
        ax.set_title(title)
        ax.axis("off")
    out = FIGDIR / "fig_framelevel_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[wrote] {out}")
    return out


# ---------------------------------------------------------------------------
# 3. Adaptive-K Pareto (placeholder data until ablation runs)
# ---------------------------------------------------------------------------
def plot_adaptive_pareto() -> Path:
    # Larger canvas + per-panel x-limits so the EARS cluster is readable
    # and the dense $\theta$ annotations don't overlap each other.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for ax, dset, title in [
        (axes[0], "ucfcrime",   "UCF-Crime"),
        (axes[1], "xdviolence", "XD-Violence"),
    ]:
        src = RESULTS / "ablations" / f"ablation_{dset}_adaptive_k.json"
        ears_xmax = None
        if src.exists():
            data = json.loads(src.read_text())
            rs = data["results"]
            avg_frames = np.array([r["avg_frames_per_video"] for r in rs])
            f1s        = np.array([r["metrics"]["f1"]         for r in rs])
            thr        = np.array([r["adaptive_threshold"]    for r in rs])
            order = np.argsort(avg_frames)
            avg_frames, f1s, thr = avg_frames[order], f1s[order], thr[order]
            ax.plot(avg_frames, f1s, **EARS, label="EARS adaptive-$K$")
            ears_xmax = float(avg_frames.max())

            # Stagger label offsets so consecutive $\theta$ tags don't collide
            # and use a small white bbox for legibility on top of grid/markers.
            offsets = [(8, 8), (8, -12), (-30, 10), (-32, -14),
                       (10, 18), (-36, 20)]
            bbox = dict(boxstyle="round,pad=0.15", fc="white",
                        ec="none", alpha=0.85)
            for i, (x, y, t) in enumerate(zip(avg_frames, f1s, thr)):
                dx, dy = offsets[i % len(offsets)]
                ax.annotate(
                    f"$\\theta$={t:.2f}", (x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, ha="left", va="center", bbox=bbox,
                    arrowprops=dict(arrowstyle="-", color="0.4",
                                    lw=0.5, shrinkA=0, shrinkB=2),
                )

        # Reference: uniform baselines from the headline K-sweep
        hd = RESULTS / f"{dset}_eval_results.json"
        uniform_xs: list[float] = []
        if hd.exists():
            tab = json.loads(hd.read_text())["table"]
            uni = [r for r in tab if r["method"] == "Uniform"]
            if uni:
                xs = [r["k"] for r in uni]
                ys = [r["f1"] for r in uni]
                ax.scatter(xs, ys, color=UNIFORM["color"], marker="x", s=45,
                           label="Uniform", zorder=3)
                uniform_xs = [float(x) for x in xs]

        # On XD-Violence the Uniform K=6 point sits far to the right and
        # crushes the EARS cluster near x=2.2-2.5. Clip x so the cluster
        # is readable, and keep a couple of nearby Uniform points for context.
        if dset == "xdviolence" and ears_xmax is not None:
            nearby = [x for x in uniform_xs if x <= ears_xmax + 1.2]
            xhi = max([ears_xmax + 0.6] + nearby) if nearby else ears_xmax + 0.6
            ax.set_xlim(min(uniform_xs + [ears_xmax]) - 0.2, xhi)

        ax.set_xlabel("Avg frames per video")
        ax.set_ylabel("F1")
        ax.set_title(f"Adaptive-$K$: {title}")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()
    out = FIGDIR / "fig_adaptive_pareto.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[wrote] {out}")
    return out


# ---------------------------------------------------------------------------
# 4. Per-class breakdown on UCF-Crime (placeholder until per-video preds saved)
# ---------------------------------------------------------------------------
def plot_per_class_ucf() -> Path:
    # Optional file written by future runs that include per-video predictions
    src = RESULTS / "ucfcrime_per_class.json"
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    if src.exists():
        data = json.loads(src.read_text())
        classes = list(data.keys())
        uni = [data[c]["uniform_f1"] for c in classes]
        ev  = [data[c]["event_f1"]   for c in classes]
        x = np.arange(len(classes))
        w = 0.4
        ax.bar(x - w/2, uni, width=w, color=UNIFORM["color"], label="Uniform")
        ax.bar(x + w/2, ev,  width=w, color=EARS["color"],    label="EARS")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_ylabel("F1")
        ax.legend()
    else:
        ax.text(0.5, 0.5,
                "Per-class breakdown: re-run evaluate_ucfcrime.py with per-video "
                "predictions saved to results/ucfcrime_per_class.json",
                ha="center", va="center", transform=ax.transAxes,
                style="italic", color="#888", wrap=True, fontsize=8)
        ax.axis("off")
    ax.set_title("Per-class F1, UCF-Crime ($K\\!=\\!2$)")
    out = FIGDIR / "fig_per_class_ucf.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[wrote] {out}")
    return out


def main() -> int:
    plot_metric_vs_k("f1",     "fig_f1_vs_k.pdf",     "F1")
    plot_metric_vs_k("recall", "fig_recall_vs_k.pdf", "Recall")
    plot_framelevel_composite()
    if any((RESULTS / "ablations").glob("ablation_*_adaptive_k.json")):
        plot_adaptive_pareto()
    else:
        print("[skip] adaptive-K artifacts missing; keeping the published figure")
    if (RESULTS / "ucfcrime_per_class.json").exists():
        plot_per_class_ucf()
    else:
        print("[skip] per-class predictions missing; keeping the published figure")
    print("\nAll figures written to:", FIGDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
