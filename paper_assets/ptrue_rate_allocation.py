"""Recompute P(True) finite-range rate diagnostics and deployed level allocation.

Uses frozen E22/E25 scores and controllers; writes isolated E26 artifacts.
Run: python paper_assets/ptrue_rate_allocation.py
Use --reuse-tables to reuse verified E26 reward tables when regenerating plots.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

import os
REPO = Path(__file__).resolve().parents[1]
PAPER = Path(os.environ.get("PRICE_PAPER_DIR", REPO / "build"))
ROOT = REPO / "experiments"
EXP = REPO
OUT = ROOT / "E26_ptrue_rate_allocation"
sys.path.insert(0, str(ROOT / "E20_nonprm_scores"))
from kernel import reward_tables

GAMMA, SEED, S, N = .0029262, 20260927, 512, 64
LABELS = {"qwen": "Qwen2.5-1.5B", "llama": "Llama-3.2-3B"}
CELLS = {"qwen": "qwen25-1.5b_math500", "llama": "llama32-3b_math500"}
BLUE, PURPLE = "#0052af", "#5b3fa8"
START = time.time()


def log(*args):
    print(f"[{time.time()-START:.1f}s]", *args, flush=True)


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(tag):
    rawpath = ROOT / "E16_answer_sets" / tag / "test_data.npz"
    raw = dict(np.load(rawpath))
    pool = EXP / "outputs/cells" / CELLS[tag] / "pools/pool.parquet"
    override = ROOT / "E22_ptrue_deployed" / tag / "evaluation_ptrue.parquet"
    scorepath = override if override.exists() else pool
    scores = pd.read_parquet(scorepath, columns=["query_id", "rollout_id", "phi_conf"])
    assert not scores.duplicated(["query_id", "rollout_id"]).any()
    keys = pd.MultiIndex.from_arrays([np.repeat(raw["qid"], 128), raw["rid"].ravel()])
    score = scores.set_index(["query_id", "rollout_id"]).loc[keys, "phi_conf"].to_numpy().reshape(-1, 128)
    assert np.isfinite(score).all() and ((0 <= score) & (score <= 1)).all()
    lengths = pd.read_parquet(pool, columns=["query_id", "rollout_id", "ell_tokens"])
    np.testing.assert_array_equal(lengths.set_index(["query_id", "rollout_id"]).loc[keys, "ell_tokens"].to_numpy().reshape(-1, 128), raw["ell"])
    cdfpath = ROOT / "E22_ptrue_deployed" / tag / "cdf.npy"
    cdf = np.load(cdfpath)
    raw["ptrue"] = score
    raw["phi"] = np.searchsorted(cdf, score, side="right") / (len(cdf) + 1.)
    raw["logM"] = logsumexp(GAMMA * raw["ell"], axis=1) - np.log(128)
    metapath = EXP / "outputs/cells" / CELLS[tag] / "pools/queries.parquet"
    meta = pd.read_parquet(metapath, columns=["query_id", "level"]).set_index("query_id")
    raw["levels"] = meta.loc[raw["qid"], "level"].to_numpy(int)
    assert set(raw["levels"]) == set(range(1, 6))
    provenance = {str(p): sha(p) for p in [rawpath, scorepath, cdfpath, metapath, ROOT/"E20_nonprm_scores/kernel.py"]}
    return raw, provenance


def direct_rewards(phi, clu, none, good, etas, perms):
    """Independent batch vote calculation at all counts (no incremental winner)."""
    result = np.zeros((13, 64))
    correct = set(clu[good].tolist())
    classes = np.unique(clu[clu != none])
    for perm in perms:
        for n in range(1, 65):
            idx = perm[:n]; valid = clu[idx] != none; idx = idx[valid]
            if not len(idx):
                continue
            mx = np.array([phi[idx[clu[idx] == a]].max() if np.any(clu[idx] == a) else -1 for a in classes])
            for k, tau in enumerate(etas):
                if np.isinf(tau):
                    totals = mx
                else:
                    totals = np.array([np.exp(tau*phi[idx[clu[idx] == a]]).sum() for a in classes]) + 1e-9*(mx+1)
                result[k, n-1] += classes[totals.argmax()] in correct
    return result / len(perms)


def prepare(tag, reuse):
    raw, provenance = load_inputs(tag)
    dest = OUT/tag; dest.mkdir(exist_ok=True)
    if reuse and (dest/"reward_tables.npz").exists():
        assert json.loads((dest/"provenance.json").read_text())["source_sha256"] == provenance
        return raw, dict(np.load(dest/"reward_tables.npz"))
    V = np.zeros((len(raw["qid"]), 13, N))
    validations = []
    for qi, qid in enumerate(raw["qid"]):
        classes = np.unique(raw["clu"][qi]); clu = np.searchsorted(classes, raw["clu"][qi]).astype(np.int64)
        none = int(np.searchsorted(classes, raw["none"][qi])) if raw["none"][qi] in classes else -1
        rng = np.random.default_rng([SEED, int(qid)])
        perms = np.stack([rng.permutation(128)[:N] for _ in range(S)])
        v, _ = reward_tables(raw["phi"][qi], clu, none, raw["good"][qi], raw["strict_good"][qi], perms, raw["etas"])
        V[qi] = v[1]
        if qi in [0, 73, 181, 307, len(V)-1]:
            independent = direct_rewards(raw["phi"][qi], clu, none, raw["good"][qi], raw["etas"], perms[:8])
            small, _ = reward_tables(raw["phi"][qi], clu, none, raw["good"][qi], raw["strict_good"][qi], perms[:8], raw["etas"])
            np.testing.assert_allclose(independent, small[1], atol=1e-12)
            validations.append(int(qid))
        if qi % 100 == 0:
            log(tag, "reward tables", qi, "/", len(V))
    assert ((0 <= V) & (V <= 1)).all()
    np.testing.assert_allclose(V*S, np.rint(V*S), atol=1e-12)
    np.testing.assert_allclose(V[:, :, 0], np.broadcast_to(V[:, :1, 0], V[:, :, 0].shape), atol=1e-12)
    data = dict(V=V, logM=raw["logM"], qid=raw["qid"], levels=raw["levels"], etas=raw["etas"])
    np.savez_compressed(dest/"reward_tables.npz", **data)
    dump(dest/"provenance.json", dict(source_sha256=provenance, score="E22/E25 P(True), fit-only CDF", gamma=GAMMA,
         seed=SEED, paths=S, horizon=N, pool_size=128, verified_direct_vote_query_ids=validations,
         direct_vote_checks=len(validations)*8*N*13, status="passed"))
    return raw, data


def oracle_curve(V, logM, points=400):
    q, k, nmax = V.shape
    logprices = np.linspace(0., -300., points)
    env = V.max(1); tau = V.argmax(1)
    logcost = logM[:, None] * np.arange(1, nmax+1)
    ceiling = V.max((1, 2))
    ns, ks, rewards, budgets, gaps = [], [], [], [], []
    for logprice in logprices:
        # Cost is independent of temperature: exact joint argmax factorizes.
        objective = env - np.exp(np.clip(logprice+logcost, -745, 700))
        count = objective.argmax(1)
        kval = tau[np.arange(q), count]
        rq = V[np.arange(q), kval, count]
        budgets.append((logsumexp(logcost[np.arange(q), count])-np.log(q))/GAMMA)
        gaps.append(float((ceiling-rq).mean()))
        rewards.append(float(rq.mean()));ns.append(count+1);ks.append(kval)
    ns, ks = np.array(ns), np.array(ks)
    for i in np.linspace(0,points-1,13).astype(int):
        cost = np.exp(np.clip(logprices[i]+logcost,-745,700))[:,None,:]
        best = (V-cost).max((1,2))
        chosen = V[np.arange(q),ks[i],ns[i]-1] - cost[np.arange(q),0,ns[i]-1]
        np.testing.assert_allclose(chosen,best,rtol=0,atol=1e-12)
    return dict(b=np.array(budgets),gap=np.array(gaps),reward=np.array(rewards),n=ns,k=ks,
                logprices=logprices,ceiling=float(ceiling.mean()))


def regression(x,y):
    slope, intercept = np.polyfit(x,y,1)
    resid = y-(intercept+slope*x)
    r2 = 1-float(resid@resid)/float(((y-y.mean())**2).sum()) if np.ptp(y)>0 else 0.
    return float(slope),float(intercept),r2


def rates(V, logM, qids, threshold):
    records=[]; q,k,nmax=V.shape;counts=np.arange(1,nmax+1); values=np.full(q,np.nan)
    for i in range(q):
        eligible=np.flatnonzero(V[i,:,-1]>=threshold)
        candidates=[]
        for ki in eligible:
            err=1-V[i,ki];fit=err>0
            if fit.sum()<3:continue
            slope,intercept,r2=regression(counts[fit],np.log(err[fit]))
            candidates.append(dict(k=int(ki),decay=-slope,intercept=intercept,r2=r2,counts=counts[fit].tolist()))
        positive=[c for c in candidates if c["decay"]>0]
        if positive:
            best=max(positive,key=lambda c:c["decay"]); values[i]=best["decay"]*GAMMA/logM[i]
            rec=dict(status="fitted",exchange_rate=float(values[i]),**best)
        else:
            rec=dict(status="not_proxy" if not len(eligible) else "insufficient_points" if not candidates else "nonpositive_slope")
        records.append(dict(query_id=int(qids[i]),eligible_temperatures=eligible.tolist(),**rec))
    return values,records


def rate_summary(data,nmax,threshold,points=400):
    V=data["V"][:,:,:nmax];curve=oracle_curve(V,data["logM"],points)
    keep=curve["gap"]>=5e-4;b,g=curve["b"][keep],curve["gap"][keep]
    tail=b>=b.min()+.75*(b.max()-b.min())
    assert tail.sum()>=3
    slope,intercept,r2=regression(b[tail],np.log(g[tail])); terminal=-slope
    values,records=rates(V,data["logM"],data["qid"],threshold);finite=np.sort(values[np.isfinite(values)])
    assert terminal>0 and len(finite)>0
    rec=dict(N=nmax,threshold=threshold,price_points=points,queries=len(V),
        proxy_queries=sum(x["status"]!="not_proxy" for x in records),fitted_queries=len(finite),
        insufficient_points=sum(x["status"]=="insufficient_points" for x in records),
        nonpositive_slope=sum(x["status"]=="nonpositive_slope" for x in records),
        terminal_slope=terminal,intercept=intercept,terminal_r_squared=r2,
        minimum_rate=float(finite.min()),median_rate=float(np.median(finite)),
        median_over_terminal=float(np.median(finite)/terminal),
        terminal_percentile=float(100*np.searchsorted(finite,terminal)/len(finite)),
        kept_budget_range=[float(b.min()),float(b.max())],fit_budget_range=[float(b[tail].min()),float(b[tail].max())],
        fit_points=int(tail.sum()),fit_grid_indices=np.flatnonzero(keep)[tail].tolist(),
        gap_range=[float(g.min()),float(g.max())],ceiling_pct=100*curve["ceiling"])
    return rec,curve,values,records


def run_rates(tag,data):
    sensitivity=[];dest=OUT/tag
    for nmax in [32,48,64]:
        for threshold in [.99,.999]:
            rec,curve,values,records=rate_summary(data,nmax,threshold)
            sensitivity.append(rec)
            if nmax==64 and threshold==.999:
                main=rec
                np.savez_compressed(dest/"rate_arrays.npz",**curve,exchange_rates=values,qid=data["qid"])
                dump(dest/"query_rate_fits.json",records)
            log(tag,"rate",nmax,threshold,"slope",round(rec["terminal_slope"]*1e4,3),"median/terminal",round(rec["median_over_terminal"],2),"percentile",round(rec["terminal_percentile"],2))
    fine,*_=rate_summary(data,64,.999,799)
    result=dict(main=main,sensitivity=sensitivity,nested_grid=fine,
                nested_grid_slope_relative_change=fine["terminal_slope"]/main["terminal_slope"]-1)
    dump(dest/"rate_results.json",result)
    pd.DataFrame(sensitivity).to_csv(dest/"rate_sensitivity.csv",index=False)
    return result


def hull(cost,reward):
    h=[]
    for i in sorted(range(len(cost)),key=lambda j:(cost[j],-reward[j])):
        if h and (cost[i]==cost[h[-1]] or reward[i]<=reward[h[-1]]):continue
        while len(h)>1:
            a,b=h[-2:]
            if (reward[b]-reward[a])/(cost[b]-cost[a])>(reward[i]-reward[b])/(cost[i]-cost[b]):break
            h.pop()
        h.append(i)
    return np.array(h)


def budget_point(d,b):
    c,r=d["C"].mean(1),d["R"].mean(1);h=hull(c,r);target=np.exp(GAMMA*b)
    assert c[h[0]]<=target<=c[h[-1]]
    j=np.searchsorted(c[h],target);lo,hi=int(h[max(0,j-1)]),int(h[j])
    theta=float((target-c[lo])/(c[hi]-c[lo])) if hi!=lo else 0.
    return dict(lo=lo,hi=hi,theta=theta)


def run_allocation(tag,raw):
    source=ROOT/"E22_ptrue_deployed"/tag
    states=np.load(source/"test_states.npz");replay=np.load(source/"test.npz")
    d=dict(np.load(source/"test/regression_w4.npz"))
    for z in [states,replay,d]:np.testing.assert_array_equal(z["qid"],raw["qid"])
    np.testing.assert_allclose(replay["logM"],raw["logM"],atol=1e-12)
    gain=states["envg"][1];lc=states["logcost"];kmat=states["regression_k"]
    tok=replay["tok"];corr=replay["corr"];Q,SN,T=tok.shape
    qi=np.arange(Q)[:,None];si=np.arange(SN)[None,:]
    cache={};arrays={};result=[];rows=[]
    def at(index):
        if index in cache:return cache[index]
        stop=(gain<=0)|(np.log(np.maximum(gain,1e-30))<=d["grid"][index]+lc)
        stop[:,:,-1]=True;n=stop.argmax(2);k=kmat[qi,si,n]
        # Independent threshold formulation, as used by the published E25 evaluator.
        thresholds=np.where(gain>0,np.log(np.maximum(gain,1e-30))-lc,-np.inf)
        thresholds[:,:,-1]=-np.inf
        np.testing.assert_array_equal(n,(thresholds<=d["grid"][index]).argmax(2))
        tokens=tok[qi,si,n].astype(float);correct=corr[qi,si,n,k]
        vals=dict(n=n+1,k=k,tokens=tokens,R=correct.mean(1),C=np.exp(GAMMA*tokens).mean(1),M=tokens.mean(1),N=(n+1).mean(1))
        for key in ["R","C","M","N"]:np.testing.assert_allclose(vals[key],d[key][index],rtol=1e-11,atol=1e-10)
        cache[index]=vals
        return vals
    for b in [2000,8000,32000]:
        point=budget_point(d,b);a,z=at(point["lo"]),at(point["hi"]);w=point["theta"]
        avg={key:(1-w)*a[key]+w*z[key] for key in ["R","C","M","N"]}
        actual=float(np.log(avg["C"].mean())/GAMMA)
        np.testing.assert_allclose(actual,b,atol=1e-8)
        levels=[]
        for level in range(1,6):
            mask=raw["levels"]==level
            entry=dict(level=level,queries=int(mask.sum()))
            for key,kind in [("M","tokens"),("N","counts")]:
                v=avg[key][mask];p10,q1,median,q3,p90=np.percentile(v,[10,25,50,75,90])
                stats=dict(mean=float(v.mean()),median=float(median),q1=float(q1),q3=float(q3),p10=float(p10),p90=float(p90))
                entry[kind]=stats;rows.append(dict(model=tag,budget=b,level=level,queries=int(mask.sum()),metric=kind,**stats))
            levels.append(entry)
        result.append(dict(requested_risk=b,actual_risk=actual,accuracy_pct=100*float(avg["R"].mean()),
                           mean_tokens=float(avg["M"].mean()),mean_count=float(avg["N"].mean()),
                           mixture=point,log_prices=[float(d["grid"][point[key]]) for key in ["lo","hi"]],levels=levels))
        arrays.update({f"b{b}__{key}":val for key,val in avg.items()})
        log(tag,"allocation",b,"L1/L5 median n",round(levels[0]["counts"]["median"],3),round(levels[-1]["counts"]["median"],3),
            "L1/L5 mean tokens",round(levels[0]["tokens"]["mean"]),round(levels[-1]["tokens"]["mean"]))
    source_paths=[source/"test/regression_w4.npz",source/"test_states.npz",source/"test.npz",source/"models.pkl"]
    saved=dict(model=tag,primary="regression_w4",points=result,source_sha256={str(p):sha(p) for p in source_paths},
               verification=dict(status="passed",endpoints_checked=len(cache),paths_per_query=SN,queried_without_level_features=True))
    dump(OUT/tag/"allocation_results.json",saved)
    np.savez_compressed(OUT/tag/"allocation_query_means.npz",qid=raw["qid"],levels=raw["levels"],**arrays)
    pd.DataFrame(rows).to_csv(OUT/tag/"allocation_levels.csv",index=False)
    return saved


sys.path.insert(0, str(PAPER / "scripts"))
import ptrue_figures  # house-style figures (2026-09-17); they read the saved E26 arrays


def style():
    pass  # paperstyle is applied inside ptrue_figures


def plot_rates(tag, result):
    ptrue_figures.rate_validation(tag, ROOT)


def plot_allocation(tag, result):
    ptrue_figures.allocation(tag, ROOT)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--reuse-tables",action="store_true")
    parser.add_argument("--cell",choices=["qwen","llama","both"],default="both")
    args=parser.parse_args();OUT.mkdir(exist_ok=True);style()
    for tag in LABELS if args.cell=="both" else [args.cell]:
        raw,data=prepare(tag,args.reuse_tables)
        rr=run_rates(tag,data);aa=run_allocation(tag,raw)
        plot_rates(tag,rr);plot_allocation(tag,aa)
        dump(OUT/tag/"verification.json",dict(status="passed",rate_queries=len(data["qid"]),
             joint_argmax_checked=True,reward_kernel_independently_checked=True,allocation=aa["verification"],
             script_sha256=sha(__file__),protocol_sha256=sha(OUT/"PROTOCOL.md")))
        log(tag,"complete")
    # The manuscript table is generated from the same saved sensitivity records.
    if all((OUT/tag/"rate_results.json").exists() for tag in LABELS):
        lines=[r"\begin{table}[!ht]",r"\centering\small",
            r"\begin{tabular}{@{}lrrrrrr@{}}",r"\toprule",
            r"Generator & $N$ & proxy & fitted & slope & slowest rate & slope percentile \\",r"\midrule"]
        for tag,label in LABELS.items():
            for d in json.loads((OUT/tag/"rate_results.json").read_text())["sensitivity"]:
                if d["threshold"] != .999:continue
                lines.append(f"{label if d['N']==32 else ''} & {d['N']} & {d['proxy_queries']} & {d['fitted_queries']} & {d['terminal_slope']*1e4:.2f} & {d['minimum_rate']*1e4:.2f} & {d['terminal_percentile']:.2f}"+r" \\")
            if tag=="qwen":lines.append(r"\midrule")
        lines.extend([r"\bottomrule",r"\end{tabular}",
            r"\caption{Horizon sensitivity of the rate diagnostic, with consistency threshold $0.999$. Slope and slowest fitted rate are in units of $10^{-4}$ per risk-adjusted token. The final column is the percentage of fitted query rates at or below the population terminal slope.}",
            r"\label{tab:rate-sens}",r"\end{table}"])
        (PAPER/"tables/tab_ptrue_rate_sensitivity.tex").write_text("\n".join(lines)+"\n")


if __name__=="__main__":main()
