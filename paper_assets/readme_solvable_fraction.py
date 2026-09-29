"""README exhibit for Theory takeaway 1: the fraction of MATH-500 queries solvable by each voting rule.

A query counts as solvable by temperature tau when the Boltzmann vote at tau over the query's full pool of
128 logged rollouts (the finite stand-in for n -> infinity) returns a correct answer. Adaptive voting solves
a query when some temperature on the 13-point grid does. P(True) scores go through the deployed rank
transform (the CDF fit on the calibration queries, experiments/E22_ptrue_deployed/<cell>/cdf.npy when the
deployed branch has been run, otherwise the CDF of the MATH-train pool). In-sample: a ceiling exhibit,
not a budgeted comparison.

Run after the oracle branch: python paper_assets/readme_solvable_fraction.py  -> assets/fig_solvable.png
"""
import pathlib, sys
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper_assets")); sys.path.insert(0, str(REPO / "experiments" / "E16_answer_sets"))
import paperstyle as ps
import data as D
import matplotlib.pyplot as plt

CELLS = {"qwen": ("Qwen2.5-1.5B", "qwen25-1.5b_mathtrain"), "llama": ("Llama-3.2-3B", "llama32-3b_mathtrain")}


def transform_cdf(tag, train_cell):
    f = REPO / "experiments" / "E22_ptrue_deployed" / tag / "cdf.npy"
    if f.exists():
        return np.load(f), "calibration-fit CDF (E22)"
    pool = pd.read_parquet(REPO / "outputs" / "cells" / train_cell / "pools" / "pool.parquet", columns=["phi_conf"])
    return np.sort(np.nan_to_num(pool.phi_conf.to_numpy(float))), "MATH-train pool CDF"


def solvable_fractions(tag):
    raw = D.load(tag, "test"); Q, R = raw["clu"].shape; etas = raw["etas"]
    z = np.load(REPO / "experiments" / "E20_nonprm_scores" / tag / "raw_scores.npz")
    assert np.array_equal(z["qid"], raw["qid"])
    ptrue = z["values"][list(z["names"]).index("ptrue")]
    cdf, source = transform_cdf(tag, CELLS[tag][1])
    phi = np.searchsorted(cdf, ptrue, side="right") / (len(cdf) + 1.0)
    solv = np.zeros((Q, len(etas)), bool)
    for i in range(Q):
        winners = D.votes(phi[i], raw["clu"][i], int(raw["none"][i]), etas, np.arange(R)[None, :])[0, -1, :]
        solv[i] = D.score_winners(winners, raw["clu"][i], raw["good"][i])
    per_tau = solv.mean(0)
    return dict(SC=per_tau[0], BoN=per_tau[-1], fixed=per_tau.max(), fixed_tau=float(etas[per_tau.argmax()]),
                adaptive=solv.any(1).mean(), Q=Q, source=source)


def main():
    ps.setup()
    res = {tag: solvable_fractions(tag) for tag in CELLS}
    for tag, r in res.items():
        print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXTWIDTH_IN, 2.3), sharey=True)
    keys = ["SC", "BoN", "fixed", "adaptive"]
    for ax, (tag, r) in zip(axes, res.items()):
        labels = ["SC\n($\\tau=0$)", "BoN\n($\\tau=\\infty$)", f"best fixed $\\tau$\n($\\tau={r['fixed_tau']:.2g}$)", "adaptive\nvoting"]
        vals = [100 * r[k] for k in keys]
        bars = ax.bar(range(4), vals, color=[ps.METHOD[k] for k in keys], width=0.62, edgecolor="#333333", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
        gain = 100 * (r["adaptive"] - r["fixed"])
        ps.header(ax, f"+{gain:.1f} pp over the best fixed rule", ps.METHOD["adaptive"], x=0.03)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 92); ax.set_title(CELLS[tag][0]); ax.grid(axis="x", visible=False); ps.box_axes(ax)
    axes[0].set_ylabel("solvable queries (% of MATH-500)")
    fig.tight_layout(w_pad=1.5)
    out = REPO / "assets" / "fig_solvable.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.03); print("wrote", out)


if __name__ == "__main__":
    main()
