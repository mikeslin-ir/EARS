"""Generate Figure 1 (teaser/motivation) for the EARS paper.

Produces:
    - assets/figures/fig_teaser.pdf  (vector for LaTeX)
    - assets/figures/fig_teaser.png  (preview)
    - assets/figures/fig_teaser.pptx (fully editable PowerPoint)

The figure contrasts (a) Uniform sampling (baseline), (b) Query-aware /
learned selection (prior work), and (c) our Evidence-Aware (motion+audio)
selection on a sparse-event video.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# ----- Paths -----
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "assets", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ----- Style -----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colors
C_BG = "#FFFFFF"
C_TIMELINE = "#E5E7EB"
C_EVENT = "#EF4444"        # critical event region (red)
C_UNIFORM = "#6B7280"      # gray
C_QUERY = "#0EA5E9"        # blue
C_OURS = "#10B981"         # green
C_AUDIO = "#8B5CF6"        # purple
C_MOTION = "#F59E0B"       # amber
C_VLM = "#1F2937"

# ----- Synthetic evidence signals (shared across panels) -----
np.random.seed(0)
T = 200  # timeline length
t = np.arange(T)

def gauss(c, s, a=1.0):
    return a * np.exp(-0.5 * ((t - c) / s) ** 2)

# A sparse-event video: one short violent event near the middle
motion = (
    0.18 * np.abs(np.sin(t * 0.05)) +
    0.05 * np.random.rand(T) +
    gauss(118, 7, 0.95) +
    gauss(126, 4, 0.55)
)
audio = (
    0.12 + 0.04 * np.random.rand(T) +
    gauss(120, 5, 0.9) +
    gauss(60, 14, 0.18)  # background chatter
)
evidence = 0.55 * motion + 0.55 * audio
evidence = evidence / evidence.max()
motion = motion / motion.max()
audio = audio / audio.max()

# Event ground-truth region
EVT_START, EVT_END = 108, 134

# Frame sampling positions (K=6 budget)
K = 6
uniform_idx = np.linspace(10, T - 11, K).astype(int)

# Query-aware: assume it nominally biases toward middle but is imprecise / costly
query_idx = np.array([25, 70, 100, 130, 155, 180])

# Ours: peaks of evidence (top-K with simple NMS)
def topk_nms(sig, k, win=8):
    s = sig.copy()
    picks = []
    for _ in range(k):
        i = int(np.argmax(s))
        picks.append(i)
        lo, hi = max(0, i - win), min(len(s), i + win + 1)
        s[lo:hi] = -1
    return np.array(sorted(picks))

ours_idx = topk_nms(evidence, K, win=10)

# =====================================================================
# Matplotlib figure
# =====================================================================
fig = plt.figure(figsize=(13.2, 5.6))
gs = fig.add_gridspec(
    3, 3,
    height_ratios=[1.0, 1.0, 1.0],
    width_ratios=[1, 1, 1],
    hspace=0.55, wspace=0.18,
    left=0.045, right=0.985, top=0.90, bottom=0.07,
)

panel_titles = [
    ("(a) Uniform sampling", "Assumes evidence is evenly distributed\n$\\Rightarrow$ misses brief critical events",  C_UNIFORM),
    ("(b) Query-aware / learned selection", "Needs extra VLM passes or training\n$\\Rightarrow$ costly, query-specific",           C_QUERY),
    ("(c) EARS (Ours): Evidence-Aware",     "Motion $+$ audio score $\\Rightarrow$ training-free,\nfocuses budget on real evidence", C_OURS),
]

picks_list = [uniform_idx, query_idx, ours_idx]

for col, ((title, subtitle, color), picks) in enumerate(zip(panel_titles, picks_list)):
    # ---- Row 0: filmstrip of frames ----
    ax_strip = fig.add_subplot(gs[0, col])
    ax_strip.set_xlim(0, T)
    ax_strip.set_ylim(0, 1)
    ax_strip.axis("off")
    # ground truth event band
    ax_strip.add_patch(Rectangle((EVT_START, 0.05), EVT_END - EVT_START, 0.9,
                                 facecolor=C_EVENT, alpha=0.18, edgecolor="none"))
    # filmstrip background
    ax_strip.add_patch(Rectangle((0, 0.30), T, 0.40, facecolor="#F3F4F6",
                                 edgecolor="#9CA3AF", linewidth=0.6))
    # frame ticks (uniform mini frames for visual feel)
    for x in np.linspace(0, T, 41):
        ax_strip.plot([x, x], [0.30, 0.70], color="#D1D5DB", lw=0.4)
    # picked frames
    for x in picks:
        in_evt = EVT_START <= x <= EVT_END
        face = color
        edge = "black"
        ax_strip.add_patch(Rectangle((x - 2.5, 0.28), 5, 0.44,
                                     facecolor=face, edgecolor=edge, linewidth=0.9, zorder=3))
        # marker above
        ax_strip.plot(x, 0.85, marker="v", color=face, markeredgecolor="black",
                      markersize=7, zorder=4)
    # event label
    ax_strip.text((EVT_START + EVT_END) / 2, 0.06, "critical event",
                  ha="center", va="bottom", fontsize=8, color=C_EVENT, weight="bold")
    # frames hit / missed
    hits = int(np.sum((picks >= EVT_START) & (picks <= EVT_END)))
    ax_strip.text(T - 2, 0.95,
                  f"evidence frames: {hits}/{K}",
                  ha="right", va="top", fontsize=8.5,
                  color="#111827",
                  bbox=dict(boxstyle="round,pad=0.25",
                            facecolor="white", edgecolor=color, linewidth=0.8))
    ax_strip.set_title(title, fontsize=12, weight="bold", color=color, pad=6)

    # ---- Row 1: signal panel ----
    ax_sig = fig.add_subplot(gs[1, col])
    ax_sig.set_xlim(0, T)
    ax_sig.set_ylim(-0.05, 1.15)
    ax_sig.set_xticks([])
    ax_sig.set_yticks([])
    for s in ("top", "right"):
        ax_sig.spines[s].set_visible(False)
    ax_sig.spines["left"].set_color("#9CA3AF")
    ax_sig.spines["bottom"].set_color("#9CA3AF")
    ax_sig.axvspan(EVT_START, EVT_END, color=C_EVENT, alpha=0.10)

    if col == 0:
        # uniform: flat "assumed" importance
        ax_sig.plot(t, np.full_like(t, 0.5, dtype=float), color=C_UNIFORM, lw=2)
        ax_sig.fill_between(t, 0, 0.5, color=C_UNIFORM, alpha=0.12)
        ax_sig.text(T / 2, 0.62, "assumed uniform importance",
                    ha="center", fontsize=8.5, color=C_UNIFORM, style="italic")
    elif col == 1:
        # query-aware: noisy attention from VLM around query
        qsig = 0.25 + 0.4 * gauss(95, 25) + 0.3 * gauss(160, 18) + 0.05 * np.random.rand(T)
        qsig = qsig / qsig.max() * 0.95
        ax_sig.plot(t, qsig, color=C_QUERY, lw=2)
        ax_sig.fill_between(t, 0, qsig, color=C_QUERY, alpha=0.15)
        ax_sig.text(T / 2, 1.05, "query-conditioned attention  (requires VLM passes)",
                    ha="center", fontsize=8.5, color=C_QUERY, style="italic")
    else:
        # ours: motion + audio -> evidence
        ax_sig.plot(t, motion, color=C_MOTION, lw=1.3, label="motion", alpha=0.9)
        ax_sig.plot(t, audio,  color=C_AUDIO,  lw=1.3, label="audio",  alpha=0.9)
        ax_sig.plot(t, evidence, color=C_OURS, lw=2.2, label="evidence")
        ax_sig.fill_between(t, 0, evidence, color=C_OURS, alpha=0.12)
        ax_sig.legend(loc="upper left", fontsize=7.5, frameon=False,
                      ncol=3, handlelength=1.2, columnspacing=0.8)

    # mark picks on signal
    for x in picks:
        ax_sig.axvline(x, ymin=0.0, ymax=0.92, color=color, lw=0.8, alpha=0.55, linestyle="--")

    ax_sig.set_xlabel("time", fontsize=9, color="#374151")

    # ---- Row 2: outcome / caption ----
    ax_cap = fig.add_subplot(gs[2, col])
    ax_cap.axis("off")
    ax_cap.set_xlim(0, 1); ax_cap.set_ylim(0, 1)

    # outcome box
    if col == 0:
        outcome = "VLM answer:  \"normal activity\"  ✗"
        outcome_color = "#B91C1C"
    elif col == 1:
        outcome = "VLM answer: partially correct,  high cost"
        outcome_color = "#B45309"
    else:
        outcome = "VLM answer:  \"fighting\"  ✓   ($\\downarrow$ frames, $\\uparrow$ recall)"
        outcome_color = "#047857"

    ax_cap.add_patch(FancyBboxPatch(
        (0.02, 0.55), 0.96, 0.40,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="#F9FAFB", edgecolor=color, linewidth=1.2,
    ))
    ax_cap.text(0.5, 0.83, subtitle, ha="center", va="center",
                fontsize=9.2, color="#1F2937")
    ax_cap.text(0.5, 0.62, outcome, ha="center", va="center",
                fontsize=9.5, color=outcome_color, weight="bold")

    # arrow from signal to VLM box
    ax_cap.annotate("", xy=(0.5, 0.20), xytext=(0.5, 0.50),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#6B7280"))
    ax_cap.add_patch(FancyBboxPatch(
        (0.32, 0.02), 0.36, 0.18,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=C_VLM, edgecolor="black", linewidth=0.6,
    ))
    ax_cap.text(0.5, 0.11, "VLM reasoning", ha="center", va="center",
                fontsize=9, color="white", weight="bold")

# ----- Title / banner -----
fig.suptitle(
    "Evidence-Aware Frame Selection for Sparse-Event Video Reasoning",
    fontsize=13.5, weight="bold", y=0.985,
)

out_pdf = os.path.join(FIG_DIR, "fig_teaser.pdf")
out_png = os.path.join(FIG_DIR, "fig_teaser.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"[ok] wrote {out_pdf}")
print(f"[ok] wrote {out_png}")

# =====================================================================
# PowerPoint (fully editable: shapes + text, no rasterized image)
# =====================================================================
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

def hex_to_rgb(h):
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

SW = prs.slide_width
SH = prs.slide_height

# Title
tb = slide.shapes.add_textbox(Inches(0.3), Inches(0.15),
                              SW - Inches(0.6), Inches(0.55))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = "Evidence-Aware Frame Selection for Sparse-Event Video Reasoning"
r.font.size = Pt(24); r.font.bold = True
r.font.color.rgb = hex_to_rgb("#111827")

# Column layout
COL_W = Inches(4.30)
COL_GAP = Inches(0.12)
COL_LEFT0 = Inches(0.20)
TOP_PANEL = Inches(0.85)
PANEL_W = COL_W

def col_left(i):
    return Emu(int(COL_LEFT0) + i * (int(COL_W) + int(COL_GAP)))

def add_text(slide, left, top, width, height, text, *,
             size=12, bold=False, color="#111827", align=PP_ALIGN.LEFT,
             italic=False):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05); tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.02); tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
    r.font.color.rgb = hex_to_rgb(color)
    return box

def add_rect(slide, left, top, width, height, fill, line=None, line_w=0.75):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shp.fill.solid(); shp.fill.fore_color.rgb = hex_to_rgb(fill)
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = hex_to_rgb(line)
        shp.line.width = Pt(line_w)
    shp.shadow.inherit = False
    return shp

def add_round(slide, left, top, width, height, fill, line, line_w=1.0):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shp.fill.solid(); shp.fill.fore_color.rgb = hex_to_rgb(fill)
    shp.line.color.rgb = hex_to_rgb(line); shp.line.width = Pt(line_w)
    shp.adjustments[0] = 0.15
    return shp

def add_triangle_marker(slide, cx, top, color, size=Inches(0.16)):
    shp = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Emu(int(cx) - int(size)//2),
                                 top, size, size)
    shp.fill.solid(); shp.fill.fore_color.rgb = hex_to_rgb(color)
    shp.line.color.rgb = hex_to_rgb("#111827"); shp.line.width = Pt(0.5)
    return shp

def add_arrow(slide, x1, y1, x2, y2, color="#6B7280", w=1.5):
    conn = slide.shapes.add_connector(2, x1, y1, x2, y2)  # straight
    conn.line.color.rgb = hex_to_rgb(color); conn.line.width = Pt(w)
    # arrowhead
    from pptx.oxml.ns import qn
    ln = conn.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"),
                          {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return conn

# Build each column
strip_top   = TOP_PANEL + Inches(0.55)
strip_h     = Inches(0.55)
sig_top     = strip_top + strip_h + Inches(0.20)
sig_h       = Inches(1.35)
caption_top = sig_top + sig_h + Inches(0.15)
caption_h   = Inches(1.10)
vlm_top     = caption_top + caption_h + Inches(0.30)
vlm_h       = Inches(0.55)

T_PX = 1.0  # normalized timeline width fraction

panels = [
    {
        "title": "(a) Uniform sampling",
        "color": C_UNIFORM,
        "picks": uniform_idx.tolist(),
        "subtitle": "Assumes evidence is evenly distributed → misses brief critical events",
        "outcome": "VLM answer: \"normal activity\"  ✗",
        "outcome_color": "#B91C1C",
        "sig_label": "assumed uniform importance",
        "sig_kind": "flat",
    },
    {
        "title": "(b) Query-aware / learned selection",
        "color": C_QUERY,
        "picks": query_idx.tolist(),
        "subtitle": "Needs extra VLM passes or training → costly, query-specific",
        "outcome": "VLM answer: partially correct, high cost",
        "outcome_color": "#B45309",
        "sig_label": "query-conditioned attention (extra VLM passes)",
        "sig_kind": "bumps",
    },
    {
        "title": "(c) EARS (Ours): Evidence-Aware",
        "color": C_OURS,
        "picks": ours_idx.tolist(),
        "subtitle": "Motion + audio score → training-free, focuses budget on real evidence",
        "outcome": "VLM answer: \"fighting\"  ✓   (↓ frames, ↑ recall)",
        "outcome_color": "#047857",
        "sig_label": "motion + audio = evidence",
        "sig_kind": "evidence",
    },
]

for i, P in enumerate(panels):
    left = col_left(i)
    # Title
    add_text(slide, left, TOP_PANEL, COL_W, Inches(0.45),
             P["title"], size=16, bold=True, color=P["color"], align=PP_ALIGN.CENTER)

    # --- Filmstrip ---
    # event band (behind)
    band_x = Emu(int(left) + int(COL_W * EVT_START / T))
    band_w = Emu(int(COL_W * (EVT_END - EVT_START) / T))
    add_rect(slide, band_x, strip_top - Inches(0.08),
             band_w, strip_h + Inches(0.16),
             fill="#FECACA")  # light red event background
    # filmstrip rectangle
    add_rect(slide, left, strip_top, COL_W, strip_h,
             fill="#F3F4F6", line="#9CA3AF", line_w=0.6)
    # frame tick marks
    n_ticks = 25
    for k in range(1, n_ticks):
        xk = Emu(int(left) + int(COL_W * k / n_ticks))
        add_rect(slide, xk, strip_top + Inches(0.04),
                 Emu(int(Inches(0.01))), strip_h - Inches(0.08),
                 fill="#D1D5DB")
    # picked frames
    for x in P["picks"]:
        cx = Emu(int(left) + int(COL_W * x / T))
        fw = Inches(0.18)
        fx = Emu(int(cx) - int(fw)//2)
        add_rect(slide, fx, strip_top - Inches(0.02),
                 fw, strip_h + Inches(0.04),
                 fill=P["color"], line="#111827", line_w=0.6)
        # marker
        add_triangle_marker(slide, cx, strip_top - Inches(0.30),
                            P["color"], size=Inches(0.18))
    # event label
    add_text(slide, band_x, strip_top + strip_h + Inches(0.02),
             band_w, Inches(0.22),
             "critical event", size=9, bold=True,
             color="#B91C1C", align=PP_ALIGN.CENTER)
    # hits badge
    hits = int(np.sum((np.array(P["picks"]) >= EVT_START) &
                       (np.array(P["picks"]) <= EVT_END)))
    badge_w = Inches(1.55); badge_h = Inches(0.28)
    badge_left = Emu(int(left) + int(COL_W) - int(badge_w) - int(Inches(0.05)))
    add_round(slide, badge_left, strip_top - Inches(0.42),
              badge_w, badge_h, fill="#FFFFFF", line=P["color"], line_w=1.0)
    add_text(slide, badge_left, strip_top - Inches(0.42),
             badge_w, badge_h,
             f"evidence frames: {hits}/{K}",
             size=9, bold=True, color="#111827", align=PP_ALIGN.CENTER)

    # --- Signal panel ---
    add_rect(slide, left, sig_top, COL_W, sig_h,
             fill="#FFFFFF", line="#9CA3AF", line_w=0.6)
    # event shading
    add_rect(slide, band_x, sig_top, band_w, sig_h, fill="#FEE2E2")
    # signal poly-line via freeform
    from pptx.util import Emu as _Emu
    def add_polyline(points_xy_norm, color, width_pt=1.6):
        # points_xy_norm: list of (xf, yf) in [0,1] within panel
        if len(points_xy_norm) < 2: return
        x0 = int(left) + int(COL_W * points_xy_norm[0][0])
        y0 = int(sig_top) + int(sig_h * (1 - points_xy_norm[0][1]))
        ff = slide.shapes.build_freeform(_Emu(x0), _Emu(y0), scale=1.0)
        pts = []
        for (xf, yf) in points_xy_norm[1:]:
            xx = int(left) + int(COL_W * xf)
            yy = int(sig_top) + int(sig_h * (1 - yf))
            pts.append((xx, yy))
        ff.add_line_segments(pts, close=False)
        shp = ff.convert_to_shape()
        shp.fill.background()
        shp.line.color.rgb = hex_to_rgb(color)
        shp.line.width = Pt(width_pt)
        return shp

    n_pts = 80
    xs_norm = np.linspace(0.02, 0.98, n_pts)
    sample_idx = (xs_norm * (T - 1)).astype(int)

    if P["sig_kind"] == "flat":
        ys = np.full(n_pts, 0.55)
        add_polyline(list(zip(xs_norm, ys)), C_UNIFORM, 2.0)
    elif P["sig_kind"] == "bumps":
        qsig = 0.25 + 0.4 * gauss(95, 25) + 0.3 * gauss(160, 18)
        qsig = qsig / qsig.max() * 0.85
        ys = qsig[sample_idx] * 0.85 + 0.05
        add_polyline(list(zip(xs_norm, ys)), C_QUERY, 2.0)
    else:
        ys_m = motion[sample_idx] * 0.75 + 0.05
        ys_a = audio[sample_idx]  * 0.75 + 0.05
        ys_e = evidence[sample_idx] * 0.85 + 0.05
        add_polyline(list(zip(xs_norm, ys_m)), C_MOTION, 1.2)
        add_polyline(list(zip(xs_norm, ys_a)), C_AUDIO, 1.2)
        add_polyline(list(zip(xs_norm, ys_e)), C_OURS, 2.2)

    # signal label
    add_text(slide, left, sig_top + Inches(0.02),
             COL_W, Inches(0.22),
             P["sig_label"], size=9, italic=True,
             color=P["color"], align=PP_ALIGN.CENTER)

    # legend for ours
    if P["sig_kind"] == "evidence":
        leg_top = sig_top + sig_h - Inches(0.28)
        lx = int(left) + int(Inches(0.12))
        for j, (lab, col) in enumerate([("motion", C_MOTION),
                                        ("audio",  C_AUDIO),
                                        ("evidence", C_OURS)]):
            sw_left = Emu(lx + j * int(Inches(1.3)))
            add_rect(slide, sw_left, leg_top + Inches(0.06),
                     Inches(0.18), Inches(0.05), fill=col)
            add_text(slide, Emu(int(sw_left) + int(Inches(0.22))),
                     leg_top, Inches(1.0), Inches(0.22),
                     lab, size=8, color="#374151")

    # vertical dashed lines at picks (approximated as thin rects)
    for x in P["picks"]:
        cx = Emu(int(left) + int(COL_W * x / T))
        add_rect(slide, cx, sig_top, Emu(int(Inches(0.012))),
                 sig_h, fill=P["color"])

    # --- Caption box ---
    add_round(slide, left, caption_top, COL_W, caption_h,
              fill="#F9FAFB", line=P["color"], line_w=1.2)
    add_text(slide, Emu(int(left) + int(Inches(0.10))),
             caption_top + Inches(0.10),
             Emu(int(COL_W) - 2 * int(Inches(0.10))),
             Inches(0.50),
             P["subtitle"], size=11, color="#1F2937", align=PP_ALIGN.CENTER)
    add_text(slide, Emu(int(left) + int(Inches(0.10))),
             caption_top + Inches(0.58),
             Emu(int(COL_W) - 2 * int(Inches(0.10))),
             Inches(0.40),
             P["outcome"], size=11.5, bold=True,
             color=P["outcome_color"], align=PP_ALIGN.CENTER)

    # arrow to VLM
    cx_col = int(left) + int(COL_W) // 2
    add_arrow(slide, Emu(cx_col), caption_top + caption_h,
              Emu(cx_col), vlm_top, color="#6B7280", w=1.5)

    # VLM box
    vlm_w = Inches(2.0)
    vlm_left = Emu(cx_col - int(vlm_w) // 2)
    add_round(slide, vlm_left, vlm_top, vlm_w, vlm_h,
              fill=C_VLM, line="#000000", line_w=0.6)
    add_text(slide, vlm_left, vlm_top, vlm_w, vlm_h,
             "VLM reasoning", size=12, bold=True,
             color="#FFFFFF", align=PP_ALIGN.CENTER)

# Footer note
add_text(slide, Inches(0.3), Inches(7.05),
         SW - Inches(0.6), Inches(0.35),
         "Frame budget K = 6.  EARS concentrates the budget on motion+audio "
         "evidence peaks, recovering the brief critical event "
         "without extra VLM forward passes or training.",
         size=10, italic=True, color="#4B5563", align=PP_ALIGN.CENTER)

out_pptx = os.path.join(FIG_DIR, "fig_teaser.pptx")
prs.save(out_pptx)
print(f"[ok] wrote {out_pptx}")
