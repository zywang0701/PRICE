"""experiments/common.py -- shared helpers for the experiment chain (2026-09-16).

Resolves the repository root (override with PRICE_ROOT), the Table 3 column budgets (config/tab_data_<cell>.json),
and provides the hull / paired-bootstrap machinery of E003's judge with Table 3's columns.
"""
import os, sys, json, pickle
import numpy as np
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))           # <repo>/experiments
EXP = os.environ.get("PRICE_ROOT", os.path.dirname(HERE))   # repo root: outputs/cells/<cell>/... live here
CONFIG = os.path.join(EXP, "config")                         # tab_data_<cell>.json (column budgets)
FINAL = os.path.join(EXP, "legacy_artifacts")                # optional: pre-P(True) deployed artifacts (not released)
if EXP not in sys.path:
    sys.path.insert(0, EXP)

CELLS = {"qwen": dict(lab="lab_ov", cell="qwen25-1.5b_math500", train="qwen25-1.5b_mathtrain", label="Qwen2.5-1.5B"),
         "llama": dict(lab="lab_ov_l3", cell="llama32-3b_math500", train="llama32-3b_mathtrain", label="Llama-3.2-3B")}
T, K = 64, 13
LOGLAMS = np.linspace(-450, 6, 305)
LAMS = np.logspace(-6.5, -1.5, 120)


def table3_columns(tag):
    d = json.load(open(os.path.join(CONFIG, f"tab_data_{tag}.json")))
    return {"mgf": [float(c) for c in d["b"]["cols"]], "lin": [float(c) for c in d["a"]["cols"]],
            "mgf_hdr": d["b"]["colhdr"], "lin_hdr": d["a"]["colhdr"], "base_b": d["b"]["base"], "base_a": d["a"]["base"]}


def load_final(tag):
    """Final deployed artifacts: replay tensors, states, records, baselines. Returns dict."""
    d = os.path.join(FINAL, tag)
    z = np.load(os.path.join(d, "replay.npz"))
    st = np.load(os.path.join(d, "e003_states.npz"))
    rec = pickle.load(open(os.path.join(d, "e003_records.pkl"), "rb"))
    base = pickle.load(open(os.path.join(d, "baselines_replay.pkl"), "rb"))
    return dict(corr=z["corr"], tok=z["tok"].astype(np.float32), ell_draw=z["ell_draw"], qid=z["qid"], etas=z["etas"],
                gain=st["gain"], kmat=st["kmat"], Lte=st["Lte"], logM_rep=st["logM_rep"], lgain=st["lgain"], logcost=st["logcost"],
                recs=rec["recs"], gamma=float(rec["gamma"]), Q=int(rec["Q"]), Sn=int(rec["Sn"]), base=base)


def stops_at(lgain, logcost, gain, Lte, x, cost_kind):
    """First stopping count per (q,s) at price x (log-lambda on the mgf axis, lambda on the lin axis)."""
    Tn = lgain.shape[2]
    stop = (lgain <= x + logcost) if cost_kind == "mgf" else (gain <= x * Lte)
    stop = stop.copy(); stop[:, :, Tn - 1] = True
    any_ = stop.any(2); m = stop.argmax(2) + 1
    return np.where(any_, m, Tn)


def sweep_rows(F, kmat=None, cost_kind="mgf", grid=None):
    """e003-format rows for a (possibly modified) vote matrix kmat on the final states."""
    corr, tok, gamma = F["corr"], F["tok"], F["gamma"]
    Q, Sn = F["Q"], F["Sn"]; qi = np.arange(Q)[:, None]; si = np.arange(Sn)[None, :]
    kmat = F["kmat"] if kmat is None else kmat
    grid = (LOGLAMS if cost_kind == "mgf" else LAMS) if grid is None else grid
    rows = []
    for x in grid:
        ms = stops_at(F["lgain"], F["logcost"], F["gain"], F["Lte"], x, cost_kind)
        kk = kmat[qi, si, ms - 1]
        tk = tok[qi, si, ms - 1]; cc = corr[qi, si, ms - 1, kk]
        rows.append((logsumexp(gamma * tk, axis=1), cc.mean(1), tk.mean(1), float(ms.mean()), float((ms % 2 == 1).mean())))
    return rows


def hull_at(pts, xs, xf, idx):
    """Monotone upper hull of (x, mean reward over idx) evaluated at xs (nan left of the first point)."""
    x = np.array([xf(p, idx) for p in pts]); y = np.array([p[1][idx].mean() for p in pts])
    o = np.argsort(x); x, y = x[o], np.maximum.accumulate(y[o])
    return np.array([np.interp(c, x, y, left=np.nan) for c in xs])


def xf_mgf(gamma, Sn):
    return lambda p, idx: (logsumexp(p[0][idx]) - np.log(len(idx) * Sn)) / gamma


def xf_lin():
    return lambda p, idx: p[2][idx].mean()


def paired_hull_bootstrap(recs_new, recs_ref, gamma, Sn, cols, cost_kind, nboot=800, seed=0):
    """Hull(new) - hull(ref) at cols, with a paired query bootstrap. Returns dict."""
    Q = len(recs_new[0][0])
    xf = xf_mgf(gamma, Sn) if cost_kind == "mgf" else xf_lin()
    A = [(r[0], r[1], r[2]) for r in recs_new]; B = [(r[0], r[1], r[2]) for r in recs_ref]
    idx_all = np.arange(Q)
    hA, hB = hull_at(A, cols, xf, idx_all), hull_at(B, cols, xf, idx_all)
    rng = np.random.default_rng(seed)
    D = np.zeros((nboot, len(cols)))
    for t in range(nboot):
        idx = rng.integers(0, Q, Q)
        D[t] = hull_at(A, cols, xf, idx) - hull_at(B, cols, xf, idx)
    d = hA - hB
    return {"cols": list(map(float, cols)), "new": hA.tolist(), "ref": hB.tolist(), "d_pp": (100 * d).tolist(),
            "ci_lo_pp": (100 * np.nanquantile(D, .025, axis=0)).tolist(), "ci_hi_pp": (100 * np.nanquantile(D, .975, axis=0)).tolist(),
            "P_new_gt_ref": np.nanmean(D > 0, axis=0).tolist(), "nboot": nboot}


def fmt_judge(name, J):
    s = [f"== {name} ==", "  cols : " + "".join(f"{c:9.0f}" for c in J["cols"]),
         "  new  : " + "".join(f"{100*v:9.2f}" for v in J["new"]), "  ref  : " + "".join(f"{100*v:9.2f}" for v in J["ref"]),
         "  d pp : " + "".join(f"{v:+9.2f}" for v in J["d_pp"]),
         "  95%CI: " + "".join(f"[{lo:+.2f},{hi:+.2f}]".rjust(15) for lo, hi in zip(J["ci_lo_pp"], J["ci_hi_pp"])),
         "  P(>) : " + "".join(f"{v:9.2f}" for v in J["P_new_gt_ref"])]
    return "\n".join(s)
