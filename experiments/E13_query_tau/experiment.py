"""Calibration-only selection of query voting rules; frozen MATH-500 replay evaluation.

Run with python3.11 experiment.py --cell qwen --phase train, then --phase evaluate.
The train phase never opens MATH-500 labels or replay data. See PROTOCOL.md.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
import argparse, hashlib, json, pickle, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C
CP = np.array([1, 2, 4, 8, 16, 32, 48, 64])
EVAL_N = np.array([2, 4, 8, 16, 32, 64])
PENALTIES = [0., .002, .005, .01]
ALPHAS = [10., 100., 1000.]
FREEZES = [0, 4, 8, 16]  # 0 means continue updating at checkpoints
START = time.time()

def log(s):
    print(f"[{time.time()-START:7.1f}s] {s}", flush=True)

def dump_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

def embedding(cell, qids):
    z = np.load(Path(C.EXP)/'outputs/cells'/cell/'pools/embeddings.npz')
    row = {int(q): i for i, q in enumerate(z['query_id'])}
    h = z['H'][[row[int(q)] for q in qids]].astype(np.float32)
    return h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-20)

def prefix_features(phi, clu, ell, none, permutations, etas):
    """Only observable inputs. No target/correctness argument. All statistics are causal.

    Returns checkpoint features, winner IDs at every count, token totals.
    """
    s, t = permutations.shape
    ids = np.unique(clu)
    pc = np.searchsorted(ids, clu[permutations])
    ph, le = phi[permutations], ell[permutations]
    valid = clu[permutations] != none
    cn = len(ids)
    oh = np.zeros((s, t, cn))
    oh[np.arange(s)[:, None], np.arange(t)[None, :], pc] = valid
    counts = np.cumsum(oh, axis=1)
    psum = np.cumsum(oh * ph[:, :, None], axis=1)
    mx = np.maximum.accumulate(np.where(oh > 0, ph[:, :, None], -1.), axis=1)
    ar = np.arange(1, t+1)[None, :]
    nv = counts.sum(2)
    pl = counts.argmax(2)
    get = lambda a, k: np.take_along_axis(a, k[..., None], axis=2)[..., 0]
    cnt = get(counts, pl)
    pmean = np.cumsum(ph, 1)/ar
    plmean = np.where(nv > 0, get(psum, pl)/np.maximum(cnt, 1), pmean)
    othermean = np.where(nv > cnt, (psum.sum(2)-get(psum, pl))/np.maximum(nv-cnt, 1), plmean)
    tok = np.cumsum(le, 1)
    glob = np.stack([cnt/ar, (counts > 0).sum(2)/ar, nv/ar, pmean,
        np.maximum.accumulate(ph, 1), np.minimum.accumulate(ph, 1),
        np.sqrt(np.maximum(np.cumsum(ph**2, 1)/ar-pmean**2, 0)),
        np.log(tok/ar+1), np.log(np.maximum.accumulate(le, 1)+1),
        cnt == 1, cnt == ar, plmean, othermean, plmean-othermean,
        (mx.argmax(2) == pl) & (nv > 0)], axis=-1)
    wins = np.empty((s, t, len(etas)), np.int16)
    cand = np.empty((s, t, len(etas), 8), np.float32)
    for k, eta in enumerate(etas):
        if np.isinf(eta):
            win = mx.argmax(2)
            share = np.where(nv > 0, 1., 0.)
            gap, ess = share, np.where(nv > 0, 1/ar, 0.)
        else:
            ww = np.exp(eta*ph)*valid
            cw = np.cumsum(oh*ww[:, :, None], 1)
            win = (cw + 1e-9*(mx+1)).argmax(2)
            total = cw.sum(2)
            top = get(cw, win)
            second = np.partition(cw, -2, axis=2)[:, :, -2] if cn > 1 else np.zeros_like(top)
            share = top/np.maximum(total, 1e-30)
            gap = (top-second)/np.maximum(total, 1e-30)
            ess = total**2/np.maximum(np.cumsum(ww**2, 1), 1e-30)/ar
        wc = get(counts, win)
        wm = get(psum, win)/np.maximum(wc, 1)
        wx = get(mx, win)
        wins[:, :, k] = np.where(nv > 0, ids[win], -32768)
        cand[:, :, k] = np.stack([share, gap, ess, wc/ar, wm, wx,
            win == pl, win == mx.argmax(2)], axis=-1)
    checkpoints=CP[CP<=t]
    feat = np.concatenate([glob[:, checkpoints-1], cand[:, checkpoints-1].reshape(s, len(checkpoints), -1)], axis=-1)
    return feat.astype(np.float32), wins, tok.astype(np.int32)

def empirical_rates(phi, clu, none, target, etas):
    """Exact Chernoff exponents for the empirical iid law at finite grid temperatures.

    The minimum ranges over every observed wrong answer. Missing correct answers and
    nonpositive true margins get zero. A pool with no wrong valid answer still has
    the abstention error exponent. Finite pools cannot establish true consistency.
    """
    r = len(phi)
    good = clu == target
    rivals = np.unique(clu[(clu != target) & (clu != none)])
    ans = np.zeros(len(etas)-1)
    if not good.any():
        return ans
    if not len(rivals):
        ans[:] = -np.log(max(1-good.mean(), .5/(r+1)))
        return ans
    for k, eta in enumerate(etas[:-1]):
        w = np.exp(eta*(phi-phi.max()))
        pos = w[good]
        rs = []
        for a in rivals:
            neg = w[clu == a]
            if pos.sum() <= neg.sum() + 1e-14:
                rs.append(0.)
                break
            def derivative_balance(x):
                return logsumexp(np.log(pos)-x*pos)-logsumexp(np.log(neg)+x*neg)
            hi = 1.
            while derivative_balance(hi) > 0:
                hi *= 2
            root = brentq(derivative_balance, 0., hi)
            nz = r-len(pos)-len(neg)
            terms = [-root*pos, root*neg]
            if nz:
                terms.append(np.array([np.log(nz)]))
            rate = np.log(r)-logsumexp(np.concatenate(terms))
            rs.append(max(float(rate), 0.))
        ans[k] = min(rs)
    return ans

def causal_check():
    eta = np.r_[np.linspace(0, 20, 12), np.inf]
    rng = np.random.default_rng(12)
    phi = rng.random(64); cl = rng.integers(0, 5, 64); le = rng.integers(50, 500, 64)
    p = np.arange(64)[None]
    a, wa, _ = prefix_features(phi, cl, le, 4, p, eta)
    phi[8:] = rng.random(56); cl[8:] = rng.integers(0, 9, 56); le[8:] *= 2
    b, wb, _ = prefix_features(phi, cl, le, 4, p, eta)
    assert np.allclose(a[:, :4], b[:, :4], atol=1e-7)
    assert np.array_equal(wa[:, :8], wb[:, :8])
    # Binary iid SC rate has the closed form -log(2 sqrt(p(1-p))).
    r = empirical_rates(np.ones(10), np.r_[np.ones(7), np.zeros(3)], 2, 1, eta)
    assert np.max(np.abs(r + np.log(2*np.sqrt(.7*.3)))) < 1e-10

def prepare_train(tag, out):
    path = out/'train_cache.npz'
    if path.exists():
        log('loading calibration cache')
        return dict(np.load(path))
    w = C.CELLS[tag]
    z = np.load(Path(C.EXP)/'outputs/cells'/w['train']/'lp3/prep_ov__phi_deepconf2.npz')
    ok = z['target'] != z['none_cluster']
    qid = z['query_id'][ok]; etas = z['etas']; q = len(qid)
    phi, clu, ell = (z[n][ok] for n in ['phi_probe','clu_probe','ell_probe'])
    target, none = z['target'][ok], z['none_cluster'][ok]
    # Four replay paths per query; query, not replay, is the statistical unit.
    feat = np.empty((q, 4, len(CP), 119), np.float32)
    corr = np.empty((q, 4, 64, 13), bool)
    tok = np.empty((q, 4, 64), np.int32)
    rate = np.empty((q, 12), np.float32)
    rng = np.random.default_rng(20260916)
    for i in range(q):
        p = np.stack([rng.permutation(64) for _ in range(4)])
        feat[i], wins, tok[i] = prefix_features(phi[i], clu[i], ell[i], none[i], p, etas)
        corr[i] = wins == target[i]
        rate[i] = empirical_rates(phi[i].astype(float), clu[i], none[i], target[i], etas)
        if i % 500 == 0:
            log(f'calibration features/rates {i}/{q}')
    d = dict(feat=feat, corr=corr, tok=tok, rate=rate, qid=qid, etas=etas,
             V=z['V_tgt'][ok].astype(np.float32), H=embedding(w['train'], qid))
    np.savez_compressed(path, **d)
    return d

def unify_reward_targets(tag,out,d):
    """Rebuild calibration targets with deployed ties on independent replay paths.

    The inherited V table breaks SC ties lexicographically. That mismatch is large
    relative to the vote-selection effects. The fixed baseline still uses its original
    calibration-selected rung; new regressions use these common-rule curves.
    """
    path=out/'unified_reward_targets.npz'
    if path.exists():
        return np.load(path)['V']
    w=C.CELLS[tag]
    z=np.load(Path(C.EXP)/'outputs/cells'/w['train']/'lp3/prep_ov__phi_deepconf2.npz')
    ok=z['target']!=z['none_cluster']; qid=z['query_id'][ok]
    assert np.array_equal(qid,d['qid'])
    phi,clu=(z[n][ok] for n in ['phi_probe','clu_probe'])
    target,none=z['target'][ok],z['none_cluster'][ok]
    v=np.empty_like(d['V']); etas=d['etas']; sn=32
    for i in range(len(qid)):
        rng=np.random.default_rng([20260917,int(qid[i])])
        p=np.stack([rng.permutation(64) for _ in range(sn)])
        ph=phi[i][p].astype(float); cl=clu[i][p]; valid=cl!=none[i]
        ids=np.unique(clu[i]); pc=np.searchsorted(ids,cl)
        oh=np.zeros((sn,64,len(ids)))
        oh[np.arange(sn)[:,None],np.arange(64)[None,:],pc]=valid
        mx=np.maximum.accumulate(np.where(oh>0,ph[:,:,None],-1.),axis=1)
        goodprefix=np.cumsum(valid,axis=1)>0
        for k,eta in enumerate(etas):
            if np.isinf(eta):
                win=mx.argmax(2)
            else:
                cw=np.cumsum(oh*np.exp(eta*ph)[:,:,None],axis=1)
                win=(cw+1e-9*(mx+1)).argmax(2)
            v[i,k]=((ids[win]==target[i])&goodprefix).mean(0)
        if i%1000==0: log(f'unified calibration reward curves {i}/{len(qid)}')
    np.savez_compressed(path,V=v,query_id=qid,replay_paths=sn,seed=20260917)
    return v

def stage_index(freeze=0):
    m = np.minimum(np.arange(1,65), freeze) if freeze else np.arange(1,65)
    return np.searchsorted(CP, m, side='right')-1

def choose(pred, k0, penalty):
    p = pred.copy()
    p[..., k0] += penalty+1e-10  # deterministic tie fallback
    return p.argmax(-1).astype(np.int8)

def score_choice(kstage, corr, freeze=0):
    ks = kstage[:, :, stage_index(freeze)[EVAL_N-1]]
    cc = np.take_along_axis(corr[:, :, EVAL_N-1], ks[..., None], axis=-1)[..., 0]
    return float(cc.mean()), cc.mean((0,1)).tolist()

def design(feat, h, pca, scaler=None, fit=False, query_only=False):
    e = pca.transform(h).astype(np.float32)
    if query_only:
        x = e
    else:
        q,s,m,f = feat.shape
        fl = feat.reshape(-1,f)
        poly = PolynomialFeatures(2, include_bias=False).fit_transform(fl[:, :15])
        extra = np.tile(np.c_[CP/64, np.log(CP), 1/np.sqrt(CP)], (q*s,1))
        x = np.c_[poly, fl[:,15:], np.repeat(e, s*m, axis=0), extra].astype(np.float32)
    if fit:
        scaler = StandardScaler().fit(x)
    return scaler.transform(x).astype(np.float32), scaler

def fit_ridge_bank(x, y, alpha):
    # One multi-output solve, independent of sklearn's multi-output memory choices.
    xm, ym = x.mean(0, dtype=np.float64), y.mean(0, dtype=np.float64)
    xx = x.T@x - len(x)*np.outer(xm,xm)
    xy = x.T@y - len(x)*np.outer(xm,ym)
    coef = np.linalg.solve(xx.astype(float)+alpha*np.eye(x.shape[1]), xy).astype(np.float32)
    return dict(coef=coef, intercept=(ym-xm@coef).astype(np.float32))

def ridge_predict(x, model):
    return x@model['coef']+model['intercept']

def targets(d, idx, sequential):
    v = d['V'][idx]
    rate = d['rate'][idx]
    rate = rate/(rate+.05)  # bounded monotone transform; preserves per-query rate argmax
    y = np.c_[v.reshape(len(idx),-1), rate].astype(np.float32)
    return np.repeat(y, d['feat'].shape[1]*len(CP), axis=0) if sequential else y

def utility(pred, objective):
    if objective == 'rate':
        return pred[..., 13*64:]
    v = pred[..., :13*64].reshape(*pred.shape[:-1],13,64)
    return v.mean(-1) if objective == 'average' else v[...,32:].mean(-1)

def classifier_design(feat, h, pca):
    q,s,m,_ = feat.shape
    fl = feat.reshape(-1,119)
    g, c = fl[:,:15], fl[:,15:].reshape(-1,13,8)
    emb = np.repeat(pca.transform(h)[:,:8], s*m, axis=0)
    n = np.tile(np.c_[CP/64, np.log(CP)], (q*s,1))
    base = np.c_[g, emb, n]
    return np.concatenate([np.repeat(base[:,None,:],13,axis=1), c,
        np.broadcast_to(np.arange(13)[None,:,None]/12, (len(fl),13,1))],axis=-1).reshape(-1,34).astype(np.float32)

def train(tag, out):
    causal_check()
    d = prepare_train(tag,out)
    q = len(d['qid'])
    split = np.random.default_rng(20260916).permutation(q)
    a,b = np.sort(split[:int(.8*q)]), np.sort(split[int(.8*q):])
    k0 = int(d['V'].mean((0,2)).argmax())
    d['V']=unify_reward_targets(tag,out,d)
    # Keep the published calibration-selected baseline; its global selection uses all
    # calibration labels, including this holdout. It has no test-label access.
    pca = PCA(32, random_state=0).fit(d['H'][a])
    base = float(d['corr'][b][:,:,EVAL_N-1,k0].mean())
    scores = []
    selections = {}
    for query_only in [True,False]:
        family = 'query' if query_only else 'sequential'
        xa, sc = design(d['feat'][a], d['H'][a], pca, fit=True, query_only=query_only)
        xb,_ = design(d['feat'][b],d['H'][b],pca,sc,query_only=query_only)
        y = targets(d,a,not query_only)
        for alpha in ALPHAS:
            model = fit_ridge_bank(xa,y,alpha)
            pr = ridge_predict(xb,model)
            if query_only:
                pr = np.broadcast_to(pr[:,None,None,:], (len(b),4,len(CP),pr.shape[-1]))
            else:
                pr = pr.reshape(len(b),4,len(CP),-1)
            for obj in ['average','tail','rate']:
                u = utility(pr,obj)
                for penalty in PENALTIES:
                    ks = choose(u,k0,penalty)
                    for freeze in ([0] if query_only else FREEZES):
                        score, bycount = score_choice(ks,d['corr'][b],freeze)
                        rec = dict(family=family, objective=obj, alpha=alpha, penalty=penalty,
                                   freeze=freeze, score=score, delta_pp=100*(score-base), bycount=bycount)
                        scores.append(rec)
                        key = f'{family}_{obj}'
                        if key not in selections or score > selections[key]['score']:
                            selections[key] = rec
            log(f'{family} ridge alpha={alpha}: best calibration delta {max(r["delta_pp"] for r in scores if r["family"]==family):+.3f} pp')
        del xa,xb,y,pr
    # A finite-count comparator: estimate correctness of each currently returned vote.
    xa = classifier_design(d['feat'][a],d['H'][a],pca)
    xb = classifier_design(d['feat'][b],d['H'][b],pca)
    ya = d['corr'][a][:,:,CP-1].reshape(-1)
    for leaves in [7,15]:
        clf = HistGradientBoostingClassifier(max_iter=100,learning_rate=.06,max_leaf_nodes=leaves,
            min_samples_leaf=200,l2_regularization=10,early_stopping=False,random_state=0)
        clf.fit(xa,ya)
        pr = clf.predict_proba(xb)[:,1].reshape(len(b),4,len(CP),13)
        for penalty in PENALTIES:
            ks = choose(pr,k0,penalty)
            for freeze in FREEZES:
                score,bycount = score_choice(ks,d['corr'][b],freeze)
                rec = dict(family='conditional',objective='current_correctness',leaves=leaves,
                           penalty=penalty,freeze=freeze,score=score,delta_pp=100*(score-base),bycount=bycount)
                scores.append(rec)
                if 'conditional' not in selections or score > selections['conditional']['score']:
                    selections['conditional'] = rec
        log(f'conditional classifier leaves={leaves}: calibration delta {selections["conditional"]["delta_pp"]:+.3f} pp')
    del xa,xb,ya,pr,clf
    chosen = max(selections,key=lambda k:selections[k]['score'])
    if selections[chosen]['score'] <= base:
        chosen = 'fixed'
    config = dict(cell=tag,k_fixed=k0,tau_fixed=float(d['etas'][k0]),calibration_queries=q,
        train_qids=d['qid'][a].tolist(),validation_qids=d['qid'][b].tolist(),validation_fixed_accuracy=base,
        selections=selections,primary_selected=chosen,all_candidates=scores,
        score_counts=EVAL_N.tolist(),checkpoints=CP.tolist(),protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
        rate_target='empirical iid Chernoff exponent, I/(I+.05), finite temperatures only',
        calibration_score_transform='existing MATH-train rank transform',
        test_score_transform='E02 CDF frozen on MATH-train',selection_uses_test_labels=False)
    dump_json(out/'selection.json',config)
    log(f'FROZEN family selection: {chosen}; now refitting on all calibration queries')
    allidx=np.arange(q)
    pca = PCA(32,random_state=0).fit(d['H'])
    bundle = dict(pca=pca,config=config,models={})
    for query_only in [True,False]:
        fam='query' if query_only else 'sequential'
        x,sc=design(d['feat'],d['H'],pca,fit=True,query_only=query_only)
        y=targets(d,allidx,not query_only)
        for alpha in sorted({r['alpha'] for r in selections.values() if r['family']==fam}):
            bundle['models'][(fam,alpha)] = dict(scaler=sc,reg=fit_ridge_bank(x,y,alpha))
        del x,y
    x=classifier_design(d['feat'],d['H'],pca)
    clf=HistGradientBoostingClassifier(max_iter=100,learning_rate=.06,max_leaf_nodes=selections['conditional']['leaves'],
        min_samples_leaf=200,l2_regularization=10,early_stopping=False,random_state=0)
    clf.fit(x,d['corr'][:,:,CP-1].reshape(-1))
    bundle['classifier']=clf
    with open(out/'models.pkl','wb') as f:
        pickle.dump(bundle,f)
    log('calibration training complete; test labels have not been loaded')

def prepare_test(tag,out):
    path=out/'test_cache.npz'
    if path.exists():
        return dict(np.load(path))
    w=C.CELLS[tag]
    cell=Path(C.EXP)/'outputs/cells'/w['cell']
    z=np.load(HERE.parent/'E02_frozen'/tag/'replay.npz')
    qid,etas,p=z['qid'],z['etas'],z['perms']
    pool=pd.read_parquet(cell/'pools/pool.parquet',columns=['query_id','rollout_id','phi_deepconf2','ell_tokens','correct'])
    raw=pd.read_parquet(Path(C.EXP)/'outputs/cells'/w['train']/'pools/pool.parquet',columns=['phi_deepconf2'])
    srt=np.sort(np.nan_to_num(raw['phi_deepconf2'].to_numpy(float),nan=0.))
    pool['phi']=np.searchsorted(srt,np.nan_to_num(pool['phi_deepconf2'].to_numpy(float),nan=0.),side='right')/(len(srt)+1.)
    clusters=pd.read_parquet(cell/'lp/clusters.parquet')
    groups=dict(tuple(pool.merge(clusters,on=['query_id','rollout_id']).sort_values(['query_id','rollout_id']).groupby('query_id')))
    feat=np.empty((len(qid),64,len(CP),119),np.float32)
    # Labels below are used exclusively for reconstruction checks and final scoring.
    # prefix_features itself accepts no target and computes all policy information.
    saved_corr=z['corr']; saved_feat=z['feats']; saved_tok=z['tok']
    maxdev=0.; maxdev_other=0.; absent=np.zeros(len(qid),bool); false_states=[]
    for i,q in enumerate(qid):
        g=groups[int(q)]
        feat[i],wins,tok=prefix_features(g['phi'].to_numpy(),g['cluster'].to_numpy(),
            g['ell_tokens'].to_numpy(),int(g['none_cluster'].iloc[0]),p[i],etas)
        target=int(g['target_cluster'].iloc[0])
        absent[i]=target not in g['cluster'].to_numpy()
        expected=saved_corr[i].copy()
        if absent[i]:
            assert not g['correct'].any(), f'metadata/grading disagreement on missing target {q}'
            false_states.append(dict(query_id=int(q),incorrectly_positive_vote_states=int(expected.sum())))
            expected[:]=False
        assert np.array_equal(wins == target, expected), f'vote replay mismatch query {q}'
        assert np.array_equal(tok,saved_tok[i])
        maxdev=max(maxdev,float(np.max(np.abs(feat[i,:,:,:15]-saved_feat[i][:,CP-1]))))
        keep=[j for j in range(15) if j!=6]
        maxdev_other=max(maxdev_other,float(np.max(np.abs(feat[i][:,:,keep]-saved_feat[i][:,CP-1][:,:,keep]))))
        if i % 100 == 0:
            log(f'test causal feature reconstruction {i}/{len(qid)}')
    # The old replay rounded the running score mean to float32 before subtracting
    # its square, producing small spurious standard deviations near zero.
    assert maxdev < 1e-3 and maxdev_other < 1e-5,(maxdev,maxdev_other)
    d=dict(feat=feat,qid=qid,etas=etas,H=embedding(w['cell'],qid),absent_target=absent)
    np.savez_compressed(path,**d)
    dump_json(out/'reconstruction.json',dict(exact_votes_after_absent_target_fix=True,exact_tokens=True,
        max_original_feature_difference=maxdev,max_other_feature_difference=maxdev_other,
        no_target_argument_to_feature_function=True,prefix_invariance_check=True,
        absent_target_queries=false_states,absent_target_query_count=int(absent.sum()),
        bug='searchsorted(target) was treated as a matching cluster index without checking membership'))
    return d

def projected_reg(reg, objective):
    return dict(coef=utility(reg['coef'],objective),intercept=utility(reg['intercept'],objective))

def predict_rules(bundle,d):
    q,s,m,_=d['feat'].shape
    config=bundle['config']; k0=config['k_fixed']; pca=bundle['pca']
    stages={k:np.empty((q,s,m),np.int8) for k in config['selections']}
    # Curve predictions for joint stopping come from the full-calibration sequential
    # average-reward model, chosen before test evaluation; same estimator for all arms.
    curve_alpha=config['selections']['sequential_average']['alpha']
    curve_model=bundle['models'][('sequential',curve_alpha)]
    gainbank=np.empty((q,s,64,13),np.float32)
    vcoef=curve_model['reg']['coef'][:,:13*64].reshape(-1,13,64)
    vint=curve_model['reg']['intercept'][:13*64].reshape(13,64)
    dc=np.diff(vcoef,axis=2); di=np.diff(vint,axis=1)
    for lo in range(0,q,24):
        hi=min(lo+24,q); f=d['feat'][lo:hi]; h=d['H'][lo:hi]
        designs={}
        for key,sel in config['selections'].items():
            if sel['family']=='conditional':
                x=classifier_design(f,h,pca)
                pred=bundle['classifier'].predict_proba(x)[:,1].reshape(hi-lo,s,m,13)
            else:
                fam,alpha=sel['family'],sel['alpha']
                if (fam,alpha) not in designs:
                    mod=bundle['models'][(fam,alpha)]
                    designs[(fam,alpha)]=design(f,h,pca,mod['scaler'],query_only=fam=='query')[0]
                x=designs[(fam,alpha)]
                pred=ridge_predict(x,projected_reg(bundle['models'][(fam,alpha)]['reg'],sel['objective']))
                if fam=='query':
                    pred=np.broadcast_to(pred[:,None,None],(hi-lo,s,m,pred.shape[-1]))
                else:
                    pred=pred.reshape(hi-lo,s,m,-1)
            stages[key][lo:hi]=choose(pred,k0,sel['penalty'])
        x=designs[('sequential',curve_alpha)].reshape(hi-lo,s,m,-1)
        # The latest observed checkpoint may estimate a curve at later counts, but
        # never consumes features from those future counts.
        for j,start in enumerate(CP):
            end=CP[j+1]-1 if j+1<len(CP) else 64
            ns=np.arange(start,min(end,63)+1)-1
            if not len(ns):
                continue
            pred=x[:,:,j].reshape(-1,x.shape[-1])@dc[:,:,ns].reshape(x.shape[-1],-1)
            pred+=di[:,ns].reshape(-1)
            gainbank[lo:hi,:,ns,:]=pred.reshape(hi-lo,s,13,len(ns)).transpose(0,1,3,2)
        gainbank[lo:hi,:,63,:]=-np.inf
        log(f'predict frozen rules and curve gains {hi}/{q}')
    kmats={k:stages[k][:,:,stage_index(sel['freeze'])] for k,sel in config['selections'].items()}
    return stages,kmats,gainbank

def bootstrap_hulls(recs,cols,gamma,sn,nboot=800,seed=20260916):
    """Vectorized equivalent of common.hull_at, with paired query weights."""
    q=len(recs[0][0])
    rng=np.random.default_rng(seed)
    counts=np.array([np.bincount(rng.integers(q,size=q),minlength=q) for _ in range(nboot)],float).T
    cc=np.c_[np.ones(q),counts]
    lg=np.stack([r[0] for r in recs])
    rr=np.stack([r[1] for r in recs]); tt=np.stack([r[2] for r in recs])
    mx=lg.max(1,keepdims=True)
    risk=(mx+np.log(np.exp(lg-mx)@cc)-np.log(q*sn))/gamma
    tokens=tt@cc/q; acc=rr@cc/q
    result={}
    for axis,x in [('mgf',risk),('lin',tokens)]:
        z=np.empty((nboot+1,len(cols[axis])))
        for i in range(nboot+1):
            order=np.argsort(x[:,i])
            z[i]=np.interp(cols[axis],x[order,i],np.maximum.accumulate(acc[order,i]),left=np.nan)
        result[axis]=z
    return result

def evaluate(tag,out):
    causal_check()
    bundle=pickle.load(open(out/'models.pkl','rb'))
    selhash=hashlib.sha256((out/'selection.json').read_bytes()).hexdigest()
    d=prepare_test(tag,out)
    stages,kmats,gainbank=predict_rules(bundle,d)
    np.savez_compressed(out/'selected_temperatures.npz',**kmats)
    cfg=bundle['config']; k0=cfg['k_fixed']
    old=HERE.parent/'E02_frozen'/tag
    z=np.load(old/'replay.npz')
    shared=dict(np.load(old/'states_shared.npz'))
    corr,tok=z['corr'],z['tok'].astype(np.float32)
    rec0=pickle.load(open(old/'records_cal.pkl','rb'))
    gamma=float(rec0['gamma']); q,s=corr.shape[:2]
    cols=C.table3_columns(tag)
    F=dict(corr=corr,tok=tok,gamma=gamma,Q=q,Sn=s,**shared)
    saved={a:pickle.load(open(old/f'records_{a}.pkl','rb'))['recs'] for a in ['cal','adaptive']}
    states={a:dict(np.load(old/f'states_{a}.npz')) for a in ['cal','adaptive']}
    base_k=np.full((q,s,64),k0,np.int8)
    records={}; summary={}; stop_indices={}
    # Saved stopping paths are computed once, then reused for every vote-only arm.
    for base in ['cal','adaptive']:
        g=states[base]['gain']
        for axis,grid in [('mgf',C.LOGLAMS),('lin',C.LAMS)]:
            lg=np.where(g>0,np.log(np.maximum(g,1e-30)),-np.inf)
            stop_indices[(base,axis)]=[C.stops_at(lg,F['logcost'],g,F['Lte'],v,axis) for v in grid]
    qi=np.arange(q)[:,None]; si=np.arange(s)[None,:]
    def fixed_path_rows(kmat,base,axis):
        rows=[]
        for ms in stop_indices[(base,axis)]:
            kk=kmat[qi,si,ms-1]
            tk=tok[qi,si,ms-1]; cc=corr[qi,si,ms-1,kk]
            rows.append((logsumexp(gamma*tk,axis=1),cc.mean(1),tk.mean(1),float(ms.mean()),float((ms%2==1).mean())))
        return rows
    for base,kmat in [('cal',base_k),('adaptive',states['adaptive']['kmat'])]:
        rr={axis:fixed_path_rows(kmat,base,axis) for axis in ['mgf','lin']}
        dev=max(float(np.max(np.abs(np.asarray(rn[j])-np.asarray(ro[j]))))
            for axis in rr for rn,ro in zip(rr[axis],saved[base][axis]) for j in range(3))
        assert dev==0,dev
        records[base]=rr
    historical={base:{axis:(100*C.hull_at(records[base][axis],cols[axis],C.xf_mgf(gamma,s) if axis=='mgf' else C.xf_lin(),np.arange(q))).tolist()
                     for axis in ['mgf','lin']} for base in ['cal','adaptive']}
    corr[d['absent_target']]=False
    for base,kmat in [('cal',base_k),('adaptive',states['adaptive']['kmat'])]:
        records[base]={axis:fixed_path_rows(kmat,base,axis) for axis in ['mgf','lin']}
    for key,kmat in kmats.items():
        for base in ['cal','adaptive']:
            name=f'{key}__{base}_stops'
            records[name]={axis:fixed_path_rows(kmat,base,axis) for axis in ['mgf','lin']}
        g=np.take_along_axis(gainbank,kmat[...,None],axis=3)[...,0]
        FF=dict(F,gain=g,lgain=np.where(g>0,np.log(np.maximum(g,1e-30)),-np.inf),kmat=kmat)
        records[f'{key}__joint']={axis:C.sweep_rows(FF,cost_kind=axis) for axis in ['mgf','lin']}
        log(f'swept {key}: fixed paths and joint controller')
    for warm in [4,8,16]:
        for key in ['cal','sequential_tail','sequential_rate']:
            kmat=base_k if key=='cal' else stages[key][:,:,stage_index(warm)]
            # Same original fixed-temperature stopping surrogate on both sides;
            # only the minimum count and returned vote change in this ablation.
            g=states['cal']['gain'].copy(); g[:,:,:warm-1]=np.inf
            FF=dict(F,gain=g,lgain=np.where(g>0,np.log(np.maximum(g,1e-30)),-np.inf),kmat=kmat)
            records[f'{key}__warm{warm}']={axis:C.sweep_rows(FF,cost_kind=axis) for axis in ['mgf','lin']}
    idx=np.arange(q)
    for name,recs in records.items():
        point={axis:(100*C.hull_at(recs[axis],cols[axis],C.xf_mgf(gamma,s) if axis=='mgf' else C.xf_lin(),idx)).tolist()
               for axis in ['mgf','lin']}
        summary[name]=point
    # Confidence intervals for the calibration-selected family, including all its
    # three controller integrations, irrespective of their observed test performance.
    primary=cfg['primary_selected']
    ci={}; boot={}
    wanted=['cal','adaptive']+([] if primary=='fixed' else [f'{primary}__{x}' for x in ['cal_stops','adaptive_stops','joint']])
    for name in wanted:
        boot[name]={}
        for axis in ['mgf','lin']:
            boot[name][axis]=bootstrap_hulls(records[name][axis],cols,gamma,s)[axis]
        log(f'paired query bootstrap {name}')
    for name in wanted[2:]:
        ci[name]={}
        for base in ['cal','adaptive']:
            ci[name][base]={}
            for axis in ['mgf','lin']:
                delta=100*(boot[name][axis]-boot[base][axis])
                ci[name][base][axis]=dict(delta_pp=delta[0].tolist(),
                    lo=np.nanquantile(delta[1:],.025,axis=0).tolist(),hi=np.nanquantile(delta[1:],.975,axis=0).tolist(),
                    defined_share=np.isfinite(delta[1:]).mean(0).tolist())
    diagnostics={}
    for name,kmat in kmats.items():
        cc=np.take_along_axis(corr,kmat[...,None],axis=-1)[...,0]
        ks=stages[name]
        diagnostics[name]=dict(fixed_count_accuracy_pct=(100*cc.mean((0,1))).tolist(),
            fixed_count_delta_pp=(100*(cc.astype(float)-corr[:,:,:,k0]).mean((0,1))).tolist(),
            fraction_other_than_global=float((kmat!=k0).mean()),
            tau_index_frequencies=(np.bincount(kmat.ravel(),minlength=13)/kmat.size).tolist(),
            prefix8_vs64_agreement=float((ks[:,:,3]==ks[:,:,-1]).mean()),
            prefix16_vs64_agreement=float((ks[:,:,4]==ks[:,:,-1]).mean()),
            note='stability compares unfrozen estimates, even when the chosen controller freezes earlier')
    result=dict(cell=tag,primary_selected=primary,budgets={a:cols[a] for a in ['mgf','lin']},
        point_accuracy_pct=summary,primary_confidence_intervals=ci,diagnostics=diagnostics,
        fixed_count_baseline_pct=(100*corr[:,:,:,k0].mean((0,1))).tolist(),
        selection_sha256=selhash,exact_historical_baseline_reproduction_before_label_fix=True,bootstrap_queries=800,
        absent_target_query_count=int(d['absent_target'].sum()),historical_accuracy_before_label_fix_pct=historical,
        score_transform='calibration-frozen E02',test_queries=q,replay_paths_per_query=s,
        evaluation='retrospective replay, no test-set selection; uncertainty conditions on fitted calibration models')
    # JSON uses null for unsupported budget columns rather than non-standard NaN.
    def clean(o):
        if isinstance(o,float) and not np.isfinite(o): return None
        if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
        if isinstance(o,list): return [clean(v) for v in o]
        return o
    dump_json(out/'results.json',clean(result))
    pickle.dump(dict(records=records,gamma=gamma,Q=q,Sn=s),open(out/'records.pkl','wb'))
    log('evaluation complete')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--cell',choices=['qwen','llama'],required=True)
    ap.add_argument('--phase',choices=['train','evaluate'],required=True)
    args=ap.parse_args()
    out=HERE/args.cell; out.mkdir(exist_ok=True)
    if args.phase=='train':
        train(args.cell,out)
    else:
        evaluate(args.cell,out)
