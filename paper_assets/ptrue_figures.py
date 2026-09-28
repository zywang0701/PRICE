#!/usr/bin/env python3
"""ptrue_figures -- house-style figures for the P(True) experiments (sec. 5 and appendix).

Reads only the saved lab_review artifacts (E21, E22, E23, E25, E26); no fitting, no
replay. Every figure follows experiments/delivery/scripts/paperstyle.py (2026-09-07
house style: serif STIX, boxed axes, dotted grid, fixed method colors, panel captions
written as findings) and is written to AdaptiveTTT/images via ps.save (pdf + png).

    fig_ptrue_rate_validation_{qwen,llama}   large-budget law diagnostic (E26)
    fig_ptrue_oracle_gain_main               PRICE-oracle minus best fixed tau, with CIs (E21)
    fig_ptrue_oracle_frontier_main           appendix: split-pool frontiers (E21)
    fig_ptrue_deployed_frontier              PRICE-deployed vs SC / BoN on realized risk (E22, E23, E25)
    fig_ptrue_alloc_levels_{qwen,llama}      tokens and counts by MATH level (E26)
    fig_ptrue_temperature_usage              terminal temperatures of PRICE-deployed (E25)

Run: python paper_assets/ptrue_figures.py [--experiment-root DIR]
(ptrue_paper_assets.py and ptrue_rate_allocation.py call into this module.)
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
PAPER = pathlib.Path(os.environ.get("PRICE_PAPER_DIR", REPO / "build"))
sys.path.insert(0, str(REPO / "paper_assets"))
import paperstyle as ps  # noqa: E402

ps.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

DEFAULT_ROOT = REPO / "experiments"
GAMMA = 0.0029262
LABELS = {"qwen": "Qwen2.5-1.5B", "llama": "Llama-3.2-3B"}
GRAY, RED, PURPLE = ps.PALETTE["gray"], ps.PALETTE["accent"], ps.METHOD["adaptive"]


def hull(cost, reward):
    """Indices of the increasing upper concave hull in MGF coordinates (feasible mixtures)."""
    order = sorted(range(len(cost)), key=lambda i: (cost[i], -reward[i]))
    keep = []
    for i in order:
        if keep and (cost[i] == cost[keep[-1]] or reward[i] <= reward[keep[-1]]):
            continue
        while len(keep) > 1:
            a, b = keep[-2:]
            if (reward[b] - reward[a]) / (cost[b] - cost[a]) > (reward[i] - reward[b]) / (cost[i] - cost[b]):
                break
            keep.pop()
        keep.append(i)
    return np.array(keep)


def frontier_on(cgrid, cost, reward):
    """Feasible-mixture frontier sampled on an MGF-cost grid (flat once saturated)."""
    h = hull(cost, reward)
    return np.interp(cgrid, cost[h], reward[h])


def cost_at(acc, cost, reward):
    """MGF cost at which the feasible frontier first reaches accuracy `acc`."""
    h = hull(cost, reward)
    return float(np.interp(acc, reward[h], cost[h]))


def read(path):
    return json.loads(pathlib.Path(path).read_text())


# ---------------------------------------------------------------------------
# 1. the large-budget law against the data (E26)
# ---------------------------------------------------------------------------
def rate_validation(tag, root=DEFAULT_ROOT):
    d = pathlib.Path(root) / "E26_ptrue_rate_allocation" / tag
    z = np.load(d / "rate_arrays.npz")
    r = read(d / "rate_results.json")["main"]
    keep = z["gap"] >= 5e-4
    b, g = z["b"][keep], z["gap"][keep]
    o = np.argsort(b); b, g = b[o], g[o]
    rates = np.sort(z["exchange_rates"][np.isfinite(z["exchange_rates"])])
    slope, icpt = r["terminal_slope"], r["intercept"]
    med = float(np.median(rates)); mult = med / slope
    pct = 100.0 * np.searchsorted(rates, slope) / len(rates)

    fig, (ax, axh) = plt.subplots(1, 2, figsize=(ps.TEXTWIDTH_IN, 2.3))
    ax.plot(b / 1e3, g, "-", color=PURPLE, lw=ps.LW_OURS, zorder=4,
            label="gap $\\widehat R^{\\infty}-\\widehat R(b)$")
    xs = np.linspace(b.min() + 0.40 * (b.max() - b.min()), b.max(), 50)
    ys = np.exp(icpt - slope * xs)
    ax.plot(xs / 1e3, ys, ls="--", lw=1.2, color=RED, zorder=3, label="terminal-slope fit")
    ax.annotate(f"terminal slope\n${slope*1e4:.2f}\\times10^{{-4}}$/token",
                (xs[3] / 1e3, ys[3]), textcoords="offset points", xytext=(3, 5),
                fontsize=8.5, color=RED, ha="left", va="bottom", zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(5e-4 * 0.6, 0.5)
    ax.set_xlim(b.min() / 1e3, b.max() / 1e3)
    ax.set_xlabel("risk budget $b$ (k tokens)")
    ax.set_ylabel("gap to ceiling (log)")
    ax.legend(loc="lower left", handlelength=1.7)

    bins = np.geomspace(rates.min() * 0.85, rates.max() * 1.15, 26)
    axh.hist(rates * 1e4, bins=bins * 1e4, color=ps.PALETTE["blue"], alpha=0.5,
             edgecolor="#2F5680", lw=0.4, zorder=2)
    axh.set_xscale("log")
    ytop = axh.get_ylim()[1] * 1.15
    axh.set_ylim(0, ytop)
    axh.axvline(slope * 1e4, color=RED, lw=1.4, ls="--", zorder=3)
    axh.annotate("terminal slope", (slope * 1e4, ytop * 0.97), textcoords="offset points",
                 xytext=(4, 0), fontsize=8.5, color=RED, ha="left", va="top")
    axh.axvline(med * 1e4, color=GRAY, lw=1.4, ls="--", zorder=3)
    axh.annotate("median", (med * 1e4, ytop * 0.97), textcoords="offset points",
                 xytext=(4, 0), fontsize=8.5, color=GRAY, ha="left", va="top")
    y_arrow = ytop * 0.74
    axh.annotate("", (med * 1e4, y_arrow), (slope * 1e4, y_arrow),
                 arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.9), zorder=3)
    axh.annotate(f"${mult:.1f}\\times$", (np.sqrt(slope * med) * 1e4, y_arrow),
                 textcoords="offset points", xytext=(0, 3), fontsize=8.5, color=GRAY,
                 ha="center", va="bottom", zorder=3)
    axh.set_xlabel("per-query exchange rate ($\\times10^{-4}$/token)")
    axh.set_ylabel("number of queries")
    caps = [ps.panel_caption(ax, "(a) Gap to the ceiling decays exponentially in $b$", dy=-0.37),
            ps.panel_caption(axh, "(b) Terminal slope sits at the slow edge,\n"
                             f"${mult:.1f}\\times$ below the median", dy=-0.37)]
    fig.subplots_adjust(left=0.10, right=0.99, top=0.97, bottom=0.39, wspace=0.32)
    ps.save(fig, f"fig_ptrue_rate_validation_{tag}", extra=caps)
    plt.close(fig)
    stats = dict(terminal_slope_1e4=round(slope * 1e4, 2), median_rate_1e4=round(med * 1e4, 2),
                 median_over_slope=round(mult, 1), slope_percentile=round(pct, 2),
                 fitted_queries=int(len(rates)), ceiling_pct=round(r["ceiling_pct"], 2))
    print(f"  rate_validation[{tag}]: {stats}")
    return stats


# ---------------------------------------------------------------------------
# 2. PRICE-oracle minus the hindsight-best fixed temperature, with intervals (E21)
# ---------------------------------------------------------------------------
def oracle_gain(root=DEFAULT_ROOT):
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXTWIDTH_IN, 2.2), sharey=True)
    caps, stats = [], {}
    for i, (ax, tag) in enumerate(zip(axes, LABELS)):
        e = read(pathlib.Path(root) / "E21_ptrue_adaptive_count" / tag / "common_tie/results.json")
        c = e["contrasts"]["best_fixed"]
        m, lo, hi = (np.array(c[k]) for k in ("mean", "lo", "hi"))
        x = np.arange(len(m))
        ax.axhline(0, color=GRAY, lw=0.8, zorder=1)
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o", color=PURPLE, ms=4.5, mec="white",
                    mew=0.5, capsize=2, lw=1.2, zorder=4)
        for xi, mi, hi_ in zip(x, m, hi):
            ax.annotate(f"+{mi:.1f}", (xi, hi_), textcoords="offset points", xytext=(0, 2.5),
                        ha="center", va="bottom", fontsize=7.5, color=PURPLE)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(bb / 1000)}k" for bb in e["budgets"]])
        ax.set_xlim(-0.5, len(m) - 0.5)
        ax.set_ylim(-0.3, 4.9)
        ax.set_xlabel("realized budget $b$ on the held-out half")
        caps.append(ps.panel_caption(
            ax, f"({'ab'[i]}) {LABELS[tag]}: adaptive voting adds\n"
                f"${m.min():.1f}$ to ${m.max():.1f}$ pp over the best fixed $\\tau$", dy=-0.30))
        stats[tag] = dict(gain_min=round(float(m.min()), 2), gain_max=round(float(m.max()), 2),
                          all_ci_positive=bool((lo > 0).all()))
    axes[0].set_ylabel("gain over best\nfixed $\\tau$ (pp)")
    fig.subplots_adjust(left=0.10, right=0.99, top=0.95, bottom=0.36, wspace=0.10)
    ps.save(fig, "fig_ptrue_oracle_gain_main", extra=caps)
    plt.close(fig)
    print(f"  oracle_gain: {stats}")
    return stats


# ---------------------------------------------------------------------------
# 3. appendix: split-pool frontiers, joint vs best fixed vs SC (E21)
# ---------------------------------------------------------------------------
def oracle_frontier(root=DEFAULT_ROOT):
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXTWIDTH_IN, 2.35))
    caps = []
    ticks = [2000, 3000, 5000, 8000, 12000, 20000, 30000]
    for ax, letter, tag in zip(axes, "ab", LABELS):
        curves = read(pathlib.Path(root) / "E21_ptrue_adaptive_count" / tag / "common_tie/curves.json")
        grid = np.geomspace(np.exp(GAMMA * 1800), np.exp(GAMMA * 32000), 800)

        def on_grid(name):
            d = curves[name]
            return frontier_on(grid, np.exp(GAMMA * np.array(d["B_risk"])), np.array(d["B_accuracy"]) / 100)

        fixed = np.max([on_grid(f"fixed_{k}") for k in range(13)], axis=0)
        x = np.log(grid) / GAMMA
        ax.plot(x, on_grid("fixed_0"), ls=ps.METHOD_LS["SC"], color=ps.METHOD["SC"], lw=ps.LW_BASE, zorder=2)
        ax.plot(x, fixed, ls=ps.METHOD_LS["fixed"], color=ps.METHOD["fixed"], lw=ps.LW_BASE, zorder=3)
        ax.plot(x, on_grid("joint"), ls="-", color=PURPLE, lw=ps.LW_OURS, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks(ticks); ax.set_xticklabels([f"{t // 1000}k" for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(1800, 32000)
        ax.set_xlabel("realized budget $b$ on B (log scale)")
        caps.append(ps.panel_caption(ax, f"({letter}) {LABELS[tag]}: PRICE-oracle stays above\n"
                                         "every fixed rule with per-query counts", dy=-0.37))
    axes[0].set_ylabel("accuracy")
    handles = [Line2D([], [], color=ps.METHOD["SC"], ls=ps.METHOD_LS["SC"], lw=ps.LW_BASE),
               Line2D([], [], color=ps.METHOD["fixed"], ls=ps.METHOD_LS["fixed"], lw=ps.LW_BASE),
               Line2D([], [], color=PURPLE, ls="-", lw=ps.LW_OURS)]
    leg = fig.legend(handles, ["SC, per-query count", "best fixed $\\tau$, per-query count",
                               "PRICE-oracle (count and vote)"],
                     loc="lower center", ncol=3, handlelength=2.2, columnspacing=1.4,
                     bbox_to_anchor=(0.5, -0.08))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.97, bottom=0.42, wspace=0.28)
    ps.save(fig, "fig_ptrue_oracle_frontier_main", extra=caps + [leg])
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. PRICE-deployed against SC and ESC on the realized entropic-risk budget (E22/E23/E25)
#    Curves only. Between swept operating points every curve mixes the two adjacent policies
#    on the MGF scale C = e^{gamma b}, exactly as the tables read a sweep at a budget (2026-09-18,
#    review fix: the earlier version interpolated in log budget, which understated the savings).
#    SC uses all its swept counts, ESC the upper frontier of its window/cap settings, and
#    PRICE-deployed its raw price sweep. The dimension line marks the largest saving vs SC.
# ---------------------------------------------------------------------------
def mgf_curve(b, r, xgrid):
    """Accuracy of the MGF-scale mixture frontier through the points (b, r) at budgets xgrid."""
    o = np.argsort(b); c, rr = np.exp(GAMMA * b[o]), r[o]
    return np.interp(np.exp(GAMMA * xgrid), c, rr)


def mgf_budget_at(acc, b, r):
    """Budget at which the MGF-scale mixture frontier through (b, r) first reaches accuracy `acc`."""
    o = np.argsort(b); c, rr = np.exp(GAMMA * b[o]), r[o]
    i = int(np.argmax(rr >= acc))
    assert i > 0 and rr[i] >= acc
    t = (acc - rr[i - 1]) / (rr[i] - rr[i - 1])
    return float(np.log((1 - t) * c[i - 1] + t * c[i]) / GAMMA)


def deployed_frontier(root=DEFAULT_ROOT):
    root = pathlib.Path(root)
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXTWIDTH_IN, 2.55))
    caps, stats = [], {}
    for ax, letter, tag in zip(axes, "ab", LABELS):
        d = read(root / "E25_required_adaptive" / tag / "results.json")     # Table 3 budgets
        zs = np.load(root / "E23_ptrue_table3" / tag / "baselines" / "sc.npz")
        ze = np.load(root / "E23_ptrue_table3" / tag / "baselines" / "esc.npz")
        z = np.load(root / "E22_ptrue_deployed" / tag / "test" / "regression_w4.npz")
        bsc, rsc = np.log(zs["C"].mean(1)) / GAMMA, zs["R"].mean(1)
        ce, re_ = ze["C"].mean(1), ze["R"].mean(1)
        h = hull(ce, re_); besc, resc = np.log(ce[h]) / GAMMA, re_[h]
        bp, rp = np.log(z["C"].mean(1)) / GAMMA, z["R"].mean(1)
        xmin, xmax = bsc.min() * 0.92, sorted(d["budgets"])[-2] * 1.1     # up to the fifth budget of Table 3
        xs = np.geomspace(bsc.min(), min(bsc.max(), xmax), 3000)
        ysc = mgf_curve(bsc, rsc, xs)
        ax.plot(xs, ysc, ls=ps.METHOD_LS["SC"], lw=ps.LW_BASE, color=ps.METHOD["SC"], zorder=2)
        xe = np.geomspace(besc.min(), min(besc.max(), xmax), 3000)
        yesc = mgf_curve(besc, resc, xe)
        ax.plot(xe, yesc, ls=ps.METHOD_LS["esc"], lw=ps.LW_BASE, color=ps.METHOD["esc"], zorder=3)
        xp = np.geomspace(bp.min(), min(bp.max(), xmax), 3000)
        yp = mgf_curve(bp, rp, xp)
        ax.plot(xp, yp, ls="-", color=PURPLE, lw=ps.LW_OURS, zorder=5)

        # largest budget saving against SC at matched accuracy, both read on the MGF scale
        accs = np.linspace(max(rsc.min(), rp.min()) + 1e-4, min(rsc.max(), rp.max()) - 1e-4, 1500)
        b_sc = np.array([mgf_budget_at(a, bsc, rsc) for a in accs])
        b_pr = np.array([mgf_budget_at(a, bp, rp) for a in accs])
        inside = b_sc <= xmax
        accs, b_sc, b_pr = accs[inside], b_sc[inside], b_pr[inside]
        ratio = b_sc / b_pr; k = int(np.argmax(ratio))
        acc0, lo, hi, best = accs[k], b_pr[k], b_sc[k], ratio[k]
        ok = (accs >= resc.min()) & (accs <= resc.max())
        b_esc = np.array([mgf_budget_at(a, besc, resc) for a in accs[ok]])
        r_esc = b_esc / b_pr[ok]; ke = int(np.argmax(r_esc))

        ys = np.concatenate([ysc, yp, yesc]); top, bot = ys.max(), ys.min(); span = top - bot
        y_dim = top + 0.10 * span
        for xend in (lo, hi):
            ax.plot([xend, xend], [acc0, y_dim], ls=(0, (2, 2)), lw=0.7, color=RED, alpha=0.85, zorder=1.5)
            ax.plot([xend], [y_dim], marker="|", ms=4.5, mew=1.0, color=RED, zorder=1.5)
        ax.plot([lo, hi], [y_dim, y_dim], ls=(0, (4, 2.5)), lw=0.9, color=RED, zorder=1.5)
        ax.annotate(f"up to ${best:.1f}\\times$ smaller budget", (np.sqrt(lo * hi), y_dim),
                    textcoords="offset points", xytext=(0, 1.8), ha="center", va="bottom", fontsize=8, color=RED)
        ax.set_ylim(bot - 0.04 * span, y_dim + 0.20 * span)
        ax.set_xscale("log"); ax.set_xlim(xmin, xmax)
        ticks = [t for t in (1000, 2000, 5000, 10000, 20000, 40000) if xmin <= t <= xmax]
        ax.set_xticks(ticks); ax.set_xticklabels([f"{t // 1000}k" for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("entropic-risk budget $b$ (log scale)")
        caps.append(ps.panel_caption(ax, f"({letter}) {LABELS[tag]}: up to ${best:.1f}\\times$ smaller\n"
                                         "risk budget than SC at matched accuracy", dy=-0.36))
        stats[tag] = dict(sc_peak=dict(accuracy_pct=round(100 * acc0, 2), sc_budget=round(hi), price_budget=round(lo), ratio=round(best, 2)),
                          esc_peak=dict(accuracy_pct=round(100 * accs[ok][ke], 2), esc_budget=round(b_esc[ke]),
                                        price_budget=round(b_pr[ok][ke]), ratio=round(r_esc[ke], 2)),
                          at_table_anchor=dict(accuracy_pct=round(d["baseline_accuracy_pct"]["sc"][3], 2),
                                               ratio=round(d["budgets"][3] / mgf_budget_at(d["baseline_accuracy_pct"]["sc"][3] / 100, bp, rp), 2)))
    axes[0].set_ylabel("accuracy")
    handles = [Line2D([], [], color=ps.METHOD["SC"], ls=ps.METHOD_LS["SC"], lw=ps.LW_BASE),
               Line2D([], [], color=ps.METHOD["esc"], ls=ps.METHOD_LS["esc"], lw=ps.LW_BASE),
               Line2D([], [], color=PURPLE, ls="-", lw=ps.LW_OURS)]
    leg = fig.legend(handles, ["SC", "ESC", "PRICE-deployed"], loc="lower center", ncol=3,
                     handlelength=2.2, columnspacing=1.6, bbox_to_anchor=(0.5, -0.07))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.97, bottom=0.42, wspace=0.28)
    ps.save(fig, "fig_ptrue_deployed_frontier", extra=caps + [leg])
    plt.close(fig)
    print(f"  deployed_frontier: {stats}")
    return stats


# ---------------------------------------------------------------------------
# 5. where the budget goes: tokens and counts by MATH level (E26)
# ---------------------------------------------------------------------------
LEVELS = [1, 2, 3, 4, 5]
BLUES3 = [(ps.PALETTE["blue_light"], ps.PALETTE["blue_dark"]), ("#7597BE", "#345A82"),
          (ps.PALETTE["blue_dark"], "#22405E")]
ORANGES3 = [(ps.PALETTE["orange_light"], ps.PALETTE["orange_dark"]), ("#E0A070", "#B85F22"),
            (ps.PALETTE["orange_dark"], "#8F4515")]
ALLOC_BUDGETS = [2000, 8000, 32000]


def boxes(ax, vals, L, pair):
    """fig_mechanism conventions: IQR box, p10-p90 whiskers, dark-gray median, mean dot-line."""
    col, dark = pair
    half = 0.27; means = []
    for lv in LEVELS:
        v = vals[L == lv]
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        p10, p90 = np.percentile(v, [10, 90])
        ax.add_patch(Rectangle((lv - half, q1), 2 * half, max(q3 - q1, 1e-9), facecolor=col,
                               alpha=0.5, edgecolor=col, lw=0.9, zorder=3))
        ax.plot([lv, lv], [p10, q1], color=col, lw=0.9, zorder=3)
        ax.plot([lv, lv], [q3, p90], color=col, lw=0.9, zorder=3)
        for y in (p10, p90):
            ax.plot([lv - 0.11, lv + 0.11], [y, y], color=col, lw=0.9, zorder=3)
        ax.plot([lv - half, lv + half], [med, med], color=GRAY, lw=1.5, zorder=4)
        means.append((lv, v.mean()))
    mx, my = zip(*means)
    ax.plot(mx, my, "-", color=dark, lw=1.2, zorder=5)
    ax.plot(mx, my, "o", color=dark, ms=3.4, mec="white", mew=0.5, zorder=6)
    ax.set_xticks(LEVELS); ax.set_xticklabels([f"L{v}" for v in LEVELS])
    ax.set_xlim(0.45, 5.55)


def allocation(tag, root=DEFAULT_ROOT):
    z = np.load(pathlib.Path(root) / "E26_ptrue_rate_allocation" / tag / "allocation_query_means.npz")
    L = z["levels"]
    fig, axes = plt.subplots(2, 3, figsize=(ps.TEXTWIDTH_IN, 3.4))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.92, bottom=0.30, wspace=0.42, hspace=0.30)
    caps, stats = [], {}
    for ci, bt in enumerate(ALLOC_BUDGETS):
        tok, cnt = z[f"b{bt}__M"] / 1000.0, z[f"b{bt}__N"]
        m1, m5 = float(np.median(cnt[L == 1])), float(np.median(cnt[L == 5]))
        t1, t5 = float(tok[L == 1].mean()), float(tok[L == 5].mean())
        stats[bt] = dict(median_count_L1=round(m1, 2), median_count_L5=round(m5, 2),
                         mean_tokens_L1=round(1000 * t1), mean_tokens_L5=round(1000 * t5))
        direction = ["fewer rollouts\non harder queries", "harder queries now\nget more rollouts",
                     "far more rollouts\non harder queries"][ci]
        assert (m5 < m1) == (ci == 0), (tag, bt, m1, m5)   # captions state the L1-vs-L5 medians
        for ri, (vals, ylab, pairs) in enumerate(((tok, "generated\ntokens (k)", BLUES3),
                                                  (cnt, "rollout count", ORANGES3))):
            ax = axes[ri, ci]
            boxes(ax, vals, L, pairs[ci])
            if ri == 0:
                ax.set_title(f"$b={bt // 1000}$k tokens", pad=3, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(ylab)
            if ri == 1:
                ax.set_xlabel("MATH level")
                caps.append(ps.panel_caption(ax, f"({'abc'[ci]}) $b={bt // 1000}$k: {direction}", dy=-0.46))
    ps.save(fig, f"fig_ptrue_alloc_levels_{tag}", extra=caps)
    plt.close(fig)
    print(f"  allocation[{tag}]: {stats}")
    return stats


# ---------------------------------------------------------------------------
# 6. terminal temperatures of PRICE-deployed at the table budgets (E25)
# ---------------------------------------------------------------------------
def temperature_usage(root=DEFAULT_ROOT):
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXTWIDTH_IN, 2.45))
    cmap = LinearSegmentedColormap.from_list("price", ["#FFFFFF", PURPLE])
    caps, stats = [], {}
    for ax, letter, tag in zip(axes, "ab", LABELS):
        d = read(pathlib.Path(root) / "E25_required_adaptive" / tag / "results.json")
        shares = 100 * np.array([u["temperature_share"] for u in d["policies"]["sequential_adaptive"]["usage"]])
        im = ax.imshow(shares, aspect="auto", cmap=cmap, vmin=0, vmax=45, zorder=2)
        ax.grid(False)
        ax.set_xticks(range(shares.shape[1]))
        ax.set_xticklabels([r"$\infty$" if t == "infinity" else f"{t:.2g}" for t in d["temperature_grid"]],
                           rotation=60, ha="right", rotation_mode="anchor", fontsize=7)
        ax.set_yticks(range(shares.shape[0]))
        ax.set_yticklabels(d["headers"], fontsize=7.5)
        for i in range(shares.shape[0]):
            for j in range(shares.shape[1]):
                v = shares[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.3,
                        color="white" if v > 22 else "#333333", zorder=3)
        ax.set_xlabel("voting temperature $\\tau$")
        if letter == "a":
            ax.set_ylabel("risk budget $b$")
        sc = shares[:, 0]
        nonmodal = [u["nonmodal_query_mode_pct"] for u in d["policies"]["sequential_adaptive"]["usage"]]
        caps.append(ps.panel_caption(ax, f"({letter}) {LABELS[tag]}: SC takes only ${sc.min():.0f}$ to ${sc.max():.0f}\\%$\n"
                                         "of the paths, the rest spread over the grid", dy=-0.52))
        stats[tag] = dict(sc_share_min=round(float(sc.min()), 1), sc_share_max=round(float(sc.max()), 1),
                          nonmodal_query_pct=[round(x, 1) for x in nonmodal])
    fig.subplots_adjust(left=0.09, right=0.90, top=0.97, bottom=0.44, wspace=0.18)
    cax = fig.add_axes([0.915, 0.44, 0.014, 0.53])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("share of terminal paths (%)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_linewidth(0.8)
    ps.save(fig, "fig_ptrue_temperature_usage", extra=caps)
    plt.close(fig)
    print(f"  temperature_usage: {stats}")
    return stats


def draw_all(root=DEFAULT_ROOT):
    out = {}
    for tag in LABELS:
        out[f"rate_{tag}"] = rate_validation(tag, root)
        out[f"alloc_{tag}"] = allocation(tag, root)
    out["oracle_gain"] = oracle_gain(root)
    oracle_frontier(root)
    out["deployed"] = deployed_frontier(root)
    out["temperature"] = temperature_usage(root)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=pathlib.Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(draw_all(args.experiment_root), indent=1, default=str))
