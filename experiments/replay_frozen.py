"""experiments/replay_frozen.py <qwen|llama> -- E02: rebuild the evaluation replay tensors with the
score transform FROZEN on the calibration corpus.

Identical to lab_ov*/anytime/replay.py (same seed, permutations, features, tie rules) except that
the DeepConf score of every MATH-500 rollout is mapped through the empirical CDF of the MATH-train
pool (475,200 rollouts) instead of the MATH-500 pool's own rank transform. The transform is monotone,
so SC / BoN / abstention handling are unchanged; only the finite-temperature weights and the
score-valued features move. Writes experiments/E02_frozen/<cell>/replay.npz (+ transform stats).
Run: OMP_NUM_THREADS=1 python3 replay_frozen.py qwen
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); P4 = os.path.dirname(HERE)
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, HERE); import common as C
TAG = sys.argv[1]; W = C.CELLS[TAG]
sys.path.insert(0, os.path.join(HERE, "legacy"))
Hn = __import__("harness_" + TAG)   # legacy/harness_<cell>.py: loads the lp/lp3 prep artifacts of this cell
from awv.engine import normalize_score
S, T, K = 64, 64, 13
OUT = os.path.join(HERE, "E02_frozen", TAG); os.makedirs(OUT, exist_ok=True)
ROOT = C.EXP


def frozen_transform(train_raw):
    srt = np.sort(np.nan_to_num(np.asarray(train_raw, float), nan=0.0)); N = len(srt)
    return lambda x: np.searchsorted(srt, np.nan_to_num(np.asarray(x, float), nan=0.0), side="right") / (N + 1.0)


def build(seed=0):
    t0 = time.time(); tr, te = Hn.load_all(); etas = te["etas"]
    cell = f"{ROOT}/outputs/cells/{W['cell']}"
    p = pd.read_parquet(cell + "/pools/pool.parquet", columns=["query_id", "rollout_id", "phi_deepconf2", "correct", "ell_tokens"])
    c = pd.read_parquet(cell + "/lp/clusters.parquet")
    train_raw = pd.read_parquet(f"{ROOT}/outputs/cells/{W['train']}/pools/pool.parquet", columns=["phi_deepconf2"])["phi_deepconf2"].values
    F = frozen_transform(train_raw)
    phi_trans = normalize_score(p["phi_deepconf2"].values)       # the transductive transform (reference only)
    p["phi"] = F(p["phi_deepconf2"].values)                        # FROZEN transform
    stats = {"n_train": int(len(train_raw)), "n_eval": int(len(p)),
             "eval_frozen_quantiles": np.quantile(p["phi"].values, [0.05, 0.25, 0.5, 0.75, 0.95]).round(4).tolist(),
             "eval_transductive_quantiles": np.quantile(phi_trans, [0.05, 0.25, 0.5, 0.75, 0.95]).round(4).tolist(),
             "mean_abs_shift": float(np.abs(p["phi"].values - phi_trans).mean()),
             "eval_raw_quantiles": np.quantile(np.nan_to_num(p["phi_deepconf2"].values), [0.05, 0.5, 0.95]).round(4).tolist(),
             "train_raw_quantiles": np.quantile(np.nan_to_num(train_raw), [0.05, 0.5, 0.95]).round(4).tolist()}
    print("transform stats:", json.dumps(stats), flush=True)
    json.dump(stats, open(os.path.join(OUT, "transform_stats.json"), "w"), indent=1)
    df = p.merge(c, on=["query_id", "rollout_id"]).sort_values(["query_id", "rollout_id"]).reset_index(drop=True)
    assert len(df) == 64000
    Q = len(te["qid"]); rng = np.random.default_rng(seed)
    corr = np.zeros((Q, S, T, K), bool); tok = np.zeros((Q, S, T), np.int32); share = np.zeros((Q, S, T, K), np.float32)
    cnt1 = np.zeros((Q, S, T), np.int16); cnt2 = np.zeros((Q, S, T), np.int16); ndist = np.zeros((Q, S, T), np.int16); nvalid = np.zeros((Q, S, T), np.int16)
    phimax = np.zeros((Q, S, T), np.float32); phimean = np.zeros((Q, S, T), np.float32); Lbar = np.zeros(Q); perms = np.zeros((Q, S, T), np.int16)
    single = np.zeros(Q); ell_draw = np.zeros((Q, S, T), np.int32); feats = np.zeros((Q, S, T, 15), np.float32)
    ar = np.arange(1, T + 1)[None, :]
    g = dict(tuple(df.groupby("query_id")))
    for i, qid in enumerate(te["qid"]):
        d = g[int(qid)]; R = len(d); assert R == 128
        cl = d["cluster"].values; tgt = int(d["target_cluster"].iloc[0]); none = int(d["none_cluster"].iloc[0])
        phi = d["phi"].values; ell = d["ell_tokens"].values; Lbar[i] = ell.mean(); single[i] = d["correct"].mean()
        assert tgt != none
        ids = np.unique(cl); Cn = len(ids); cid = np.searchsorted(ids, cl)
        # searchsorted gives an insertion position, not proof that gold was sampled.
        tgt_c = int(np.searchsorted(ids, tgt)) if tgt in ids else -1
        valid = cl != none
        P = np.stack([rng.permutation(R)[:T] for _ in range(S)])
        perms[i] = P
        pc = cid[P]; pv = valid[P]; pphi = phi[P]; pell = ell[P]
        ell_draw[i] = pell; tok[i] = np.cumsum(pell, 1)
        oh = np.zeros((S, T, Cn)); oh[np.arange(S)[:, None], np.arange(T)[None, :], pc] = 1.0; oh *= pv[:, :, None]
        cum_cnt = np.cumsum(oh, 1)
        srt = -np.sort(-cum_cnt, 2); cnt1[i] = srt[:, :, 0]; cnt2[i] = srt[:, :, 1] if Cn > 1 else 0
        ndist[i] = (cum_cnt > 0).sum(2); nvalid[i] = np.cumsum(pv, 1)
        mx = np.full((S, T, Cn), -1.0)
        for s in range(S):
            cur = np.full(Cn, -1.0)
            for m in range(T):
                if pv[s, m] and pphi[s, m] > cur[pc[s, m]]: cur[pc[s, m]] = pphi[s, m]
                mx[s, m] = cur
        phimax[i] = mx.max(2); phimean[i] = np.cumsum(pphi, 1) / np.arange(1, T + 1)
        cum_phi = np.cumsum(np.where(pv, pphi, 0.0)[:, :, None] * oh, 1)
        plur = cum_cnt.argmax(2)
        c_pl = np.take_along_axis(cum_cnt, plur[:, :, None], 2)[:, :, 0]; s_pl = np.take_along_axis(cum_phi, plur[:, :, None], 2)[:, :, 0]
        nv = nvalid[i].astype(float); anyv = nv > 0
        ph_pl = np.where(anyv, s_pl / np.maximum(c_pl, 1), phimean[i])
        rest_n = nv - c_pl; rest_s = cum_phi.sum(2) - s_pl
        ph_ot = np.where(anyv & (rest_n > 0), rest_s / np.maximum(rest_n, 1), ph_pl)
        top_is_plur = np.where(anyv, (mx.argmax(2) == plur).astype(float), 0.0)
        top = np.where(anyv, c_pl, 0.0); nd = ndist[i].astype(float)
        cmean = phimean[i]; cmax = np.maximum.accumulate(pphi, 1); cmin = np.minimum.accumulate(pphi, 1)
        csq = np.cumsum(pphi ** 2, 1) / ar; cstd = np.sqrt(np.maximum(csq - cmean ** 2, 0.0))
        lmean = np.cumsum(pell, 1) / ar; lmax = np.maximum.accumulate(pell, 1)
        feats[i] = np.stack([top / ar, nd / ar, nv / ar, cmean, cmax, cmin, cstd, np.log(lmean + 1), np.log(lmax + 1),
                             (top == 1).astype(float), (top == ar).astype(float), ph_pl, ph_ot, ph_pl - ph_ot, top_is_plur], -1)
        for k in range(K):
            if np.isinf(etas[k]):
                win = mx.argmax(2); ok = mx.max(2) > -0.5
                corr[i, :, :, k] = ok & (win == tgt_c); share[i, :, :, k] = np.where(ok, 1.0, 0.0)
            else:
                w = np.exp(etas[k] * pphi)[:, :, None] * oh
                cw = np.cumsum(w, 1); tot = cw.sum(2)
                score = cw + 1e-9 * (mx + 1.0)
                win = score.argmax(2); ok = tot > 0
                corr[i, :, :, k] = ok & (win == tgt_c); share[i, :, :, k] = np.where(ok, cw.max(2) / np.maximum(tot, 1e-300), 0.0)
        if i % 50 == 0: print(f"  {i}/{Q}  {time.time()-t0:.0f}s", flush=True)
    out = os.path.join(OUT, "replay.npz")
    np.savez_compressed(out, corr=corr, tok=tok, share=share, cnt1=cnt1, cnt2=cnt2, ndist=ndist, nvalid=nvalid, phimax=phimax, phimean=phimean,
                        Lbar=Lbar, single=single, perms=perms, ell_draw=ell_draw, feats=feats, qid=te["qid"], etas=etas)
    print(f"saved {out} in {time.time()-t0:.0f}s", flush=True)
    # cross-check against the pre-freeze transductive replay (not released): SC and BoN must be identical (monotone transform)
    legacy = os.path.join(C.FINAL, TAG, "replay.npz")
    if not os.path.exists(legacy):
        print("legacy transductive replay not found; skipping the cross-check", flush=True); return
    z0 = np.load(legacy)
    same_perm = np.array_equal(z0["perms"], perms)
    d_sc = float(np.abs(z0["corr"][:, :, :, 0].astype(float) - corr[:, :, :, 0]).mean())
    d_bon = float(np.abs(z0["corr"][:, :, :, K - 1].astype(float) - corr[:, :, :, K - 1]).mean())
    d_mid = float(np.abs(z0["corr"][:, :, :, 4].astype(float) - corr[:, :, :, 4]).mean())
    print(f"check vs transductive replay: same permutations={same_perm}; mean|d corr| SC={d_sc:.5f} BoN={d_bon:.5f} k=4={d_mid:.5f}", flush=True)
    json.dump({"same_permutations": bool(same_perm), "mean_abs_dcorr_sc": d_sc, "mean_abs_dcorr_bon": d_bon, "mean_abs_dcorr_k4": d_mid},
              open(os.path.join(OUT, "replay_check.json"), "w"), indent=1)


if __name__ == "__main__":
    build()
