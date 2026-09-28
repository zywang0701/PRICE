"""Shared matplotlib style for the AdaptiveTTT (PRICE) paper figures.

2026-09-07 restyle to the house figure style (~/.claude/skills/figure-style):
serif type matching the Times/STIX body, BOXED axes (all four spines), dotted
light grid on both axes, ticks out, framed legends, panel captions written as
findings ("(a) ...") placed under the x label via `panel_caption`, and the
number the prose quotes annotated on the panel (`header`, dimension lines).

Method colors are FIXED by Figure 1 (images/fig_illustration.pdf, hand-drawn,
not restyled) and never re-mapped:
    PRICE (ours)      purple  #5B3FA8   solid, heavy
    SC   (tau = 0)    teal    #0F766E
    fixed / weighted  orange  #B45309
    BoN  (tau = inf)  blue    #1D4ED8
Every other role comes from the house palette (`PALETTE`): gray #4D4D4D for
reference lines and medians, accent red #C0392B for one-off emphasis (crossing
markers, dimension lines), steel blue / teal / rose / olive for non-method
series, and the light/dark pairs for the allocation boxes.
ESC is dark gold #B8860B (2026-09-24; it was steel blue, which Yifan found too close
to SC teal in the deployed frontier).

Every figure script does `import paperstyle as ps; ps.setup()` and writes both
pdf and png into AdaptiveTTT/images via `ps.save`.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import os
ROOT = pathlib.Path(__file__).resolve().parents[1]          # repo root
PAPER = pathlib.Path(os.environ.get("PRICE_PAPER_DIR", ROOT / "build"))   # where tables/ and images/ are written
IMAGES = PAPER / "images"

# ---- house palette (roles fixed once per paper) -----------------------------
PALETTE = {
    "gray": "#4D4D4D", "orange": "#D9740F", "teal": "#2A9D8F", "blue": "#3D6FA3",
    "rose": "#C9707A", "accent": "#C0392B", "purple": "#7B3FA0", "olive": "#6E8B3D",
    "shade": "#EEF0F3", "callout_fill": "#FDF3EC",
    "gold": "#B8860B",   # dark gold, added 2026-09-24 for ESC: warm against SC teal / PRICE purple, not the fixed-tau orange
    # light/dark pairs (light = reference / low budget, dark = ours / high budget)
    "blue_light": "#C5D3E3", "blue_dark": "#4A78A8",
    "orange_light": "#E8C9B3", "orange_dark": "#D97B3A",
}

# ---- entity -> color, set by Figure 1; never re-mapped ----------------------
METHOD = {
    "SC":       "#0F766E",   # teal   (Figure 1: SC, tau = 0)
    "BoN":      "#1D4ED8",   # blue   (Figure 1: BoN, tau -> inf)
    "fixed":    "#B45309",   # orange (Figure 1: score-weighted, pinned rung)
    "adaptive": "#5B3FA8",   # purple (Figure 1: PRICE, ours)
    "ac":       PALETTE["rose"],     # Adaptive-Consistency
    "deepconf": PALETTE["olive"],    # DeepConf online
    "esc":      PALETTE["gold"],     # ESC (sec. 5.2 deployed frontier, dash-dot); steel blue until 2026-09-24, too close to SC teal (Yifan)
}
# marker + line style per method, so color is never the only encoding
METHOD_MARKER = {"SC": "o", "BoN": "s", "fixed": "D", "adaptive": None,
                 "ac": "^", "deepconf": "v", "esc": "x"}
METHOD_LS = {"SC": "--", "BoN": ":", "fixed": "-.", "adaptive": "-",
             "ac": "--", "deepconf": ":", "esc": "-."}
METHOD_LABEL = {"SC": "SC", "BoN": "BoN", "fixed": "fixed $\\tau$",
                "adaptive": "PRICE (ours)", "ac": "Adaptive-Consistency",
                "deepconf": "DeepConf", "esc": "ESC"}
BASE_ALPHA = 1.0             # baselines at full opacity; weight/style carry the emphasis

# score palette: five distinguishable hues from the house palette + markers
SCORE = {
    "phi_prm_last": PALETTE["blue"],
    "phi_conf":     PALETTE["teal"],
    "phi_deepconf": PALETTE["orange"],
    "phi_lik":      PALETTE["purple"],
    "phi_selfcert": PALETTE["rose"],
}
SCORE_MARKER = {"phi_prm_last": "o", "phi_conf": "s", "phi_deepconf": "D",
                "phi_lik": "^", "phi_selfcert": "v"}
SCORE_LABEL = {
    "phi_prm_last": "Skywork PRM", "phi_conf": "Self-evaluation",
    "phi_lik": "Likelihood", "phi_selfcert": "Self-certainty",
    "phi_deepconf": "DeepConf", "phi_prm_prod": "PRM (product)",
    "phi_prm_min": "PRM (min)",
}
MODEL_LABEL = {
    "llama31-8b_math500": "Llama-3.1-8B",
    "llama32-3b_math500": "Llama-3.2-3B",
    "qwen25-1.5b_math500": "Qwen2.5-1.5B",
}
BUDGET2 = [PALETTE["blue_light"], PALETTE["blue_dark"]]     # low / high budget
GAMMA_RAMP = ["#C5D3E3", "#9DB6D2", "#7597BE", "#4A78A8", "#345A82", "#22405E"]

LW_BASE = 1.3
LW_OURS = 2.0
CROSS = PALETTE["accent"]    # crossing marker + its vertical dashed guide (accent red)
BAND_ALPHA = 0.15
TEXTWIDTH_IN = 5.5           # iclr2025 \textwidth (the arXiv main uses the same)

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    "axes.grid": True, "axes.grid.axis": "both",
    "grid.linestyle": ":", "grid.linewidth": 0.6, "grid.color": "#BBBBBB",
    "axes.axisbelow": True,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.5, "lines.markersize": 4.5,
    "legend.frameon": True, "legend.framealpha": 1.0,
    "legend.edgecolor": "#BBBBBB", "legend.fancybox": False,
    "legend.handlelength": 1.6, "legend.borderpad": 0.4,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
}


def setup() -> None:
    plt.rcParams.update(RC)


def panel_caption(ax, text, dy=-0.30):
    """'(a) Finding' below the x-axis label (a claim, not a variable name)."""
    return ax.text(0.5, dy, text, transform=ax.transAxes, ha="center", va="top",
                   fontsize=9.5)


def box_axes(ax):
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.8); s.set_color("#333333")


def header(ax, text, color, x=0.03, ha="left"):
    """Bold colored phrase at the top of the axes carrying the take-away number."""
    return ax.text(x, 0.96, text, transform=ax.transAxes, ha=ha, va="top",
                   fontsize=8.5, fontweight="bold", color=color)


def save(fig, name: str, extra=None, pad: float = 0.02) -> None:
    """Write vector PDF (the manuscript asset) + a PNG proof for quick review.
    Pass outside-axes artists (shared legends, panel captions) as `extra`."""
    IMAGES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(IMAGES / f"{name}.{ext}", bbox_inches="tight", pad_inches=pad,
                    bbox_extra_artists=extra)
    print(f"  wrote images/{name}.pdf (+.png)")
