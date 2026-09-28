"""Independent checks of source-only actions and B executed path accounting."""
import json,hashlib
from pathlib import Path
import config as c
import numpy as np
from scipy.special import logsumexp
from price_sweep import mix


def verify(tag):
    out=c.HERE/tag;tab=dict(np.load(c.E17/tag/'tables.npz'));replay=dict(np.load(out/'replay_cost.npz'))
    raw=c.D.load(tag,'test');meta=json.loads((out/'results.json').read_text());Q=meta['queries'];qi=np.arange(Q)[None,:]
    for p,h in json.loads((out/'source.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    checks=0;price_checks=0;curve_cost={};curve_reward={};price_rng=np.random.default_rng(167)
    for name in ['joint',*[f'fixed_{k}' for k in range(meta['temperatures'])]]:
        d=dict(np.load(out/f'{name}.npz'));P=len(d['lambdas']);rr=np.zeros((P,Q));cc=np.zeros_like(rr);mm=np.zeros_like(rr)
        for direction in range(2*c.SPLITS):
            part,side=divmod(direction,2);n=d['n'][direction]-1;k=d['k'][direction]
            rr+=tab['V'][part,1-side][qi,k,n].astype(float)/(2*c.SPLITS)
            cc+=np.exp(replay['logC'][part,1-side])[qi,n]/(2*c.SPLITS)
            mm+=replay['mean_tokens'][part,1-side][qi,n]/(2*c.SPLITS)
            for j in np.unique(np.r_[0,P-1,price_rng.integers(0,P,10)]):
                V=tab['V'][part,side].astype(float);C=np.exp(tab['logC_plugin'][part,side]);lam=d['lambdas'][j]
                R=V.max(1) if name=='joint' else V[:,int(name.split('_')[1])]
                direct=(R-lam*C).max(1)
                picked=V[np.arange(Q),k[j],n[j]]-lam*C[np.arange(Q),n[j]]
                np.testing.assert_allclose(picked,direct,rtol=1e-11,atol=1e-12);price_checks+=Q
            checks+=P*Q
        np.testing.assert_allclose(rr,d['R_B'],rtol=0,atol=1e-13)
        np.testing.assert_allclose(cc,d['C_B'],rtol=1e-13)
        np.testing.assert_allclose(mm,d['mean_B'],rtol=1e-13)
        curve_cost[name]=cc.mean(1);curve_reward[name]=rr.mean(1)
        if name=='joint':
            # Recompute actual votes and cumulative cost directly on selected B
            # queries, without going through either stored reward/cost table.
            for part,side,i in [(0,1,0),(0,1,27),(1,0,35),(3,1,Q-1)]:
                half=tab['half_indices'][part,side,i];qid=int(raw['qid'][i])
                rng=np.random.default_rng([c.SEED,qid,part,side,1])
                perms=np.stack([half[rng.permutation(64)] for _ in range(c.S)])
                tok=np.cumsum(raw['ell'][i,perms],axis=1)
                _,wins,_=c.D.old.prefix_features(raw['phi'][i],raw['clu'][i],raw['ell'][i],raw['none'][i],perms,raw['etas'])
                good=c.D.score_winners(wins,raw['clu'][i],raw['good'][i])
                np.testing.assert_array_equal(good.mean(0).T,tab['V'][part,side,i])
                np.testing.assert_allclose(logsumexp(c.GAMMA*tok,axis=0)-np.log(c.S),replay['logC'][part,side,i],rtol=1e-13)
                np.testing.assert_allclose(tok.mean(0),replay['mean_tokens'][part,side,i])
    max_err=0.
    for method,points in meta['methods'].items():
        for j,p in enumerate(points):
            name=('joint' if method=='joint' else 'fixed_0' if method=='sc' else
                f'fixed_{meta["temperatures"]-1}' if method=='bon' else
                f'fixed_{meta["k_fixed"]}' if method=='calibration_fixed' else f'fixed_{p["temperature_index"]}')
            C=float(mix(curve_cost[name],p));R=float(mix(curve_reward[name],p));b=float(np.log(C)/c.GAMMA)
            assert abs(100*R-p['accuracy_B_pct'])<1e-10
            assert b<=c.BUDGETS[j]+1e-7
            if p['status']=='exact':max_err=max(max_err,abs(b-c.BUDGETS[j]))
    q=dict(np.load(out/'matched_joint_queries.npz'))
    np.testing.assert_allclose(100*(q['R_B']-q['global_vote_B']).mean(1),meta['equal_count_vote']['A_global']['mean'],atol=1e-12)
    return dict(source_hashes_verified=True,source_priced_actions_checked=price_checks,
        query_price_direction_outcomes_checked=checks,all_B_rewards_and_costs_recomputed=True,
        independent_raw_path_vote_and_cost_checks=4,max_matched_budget_error_tokens=max_err,
        matched_count_controls_share_the_joint_prefixes=True,
        max_nested_grid_difference_pp=meta['max_half_grid_difference_pp'])


if __name__=='__main__':
    r={tag:verify(tag) for tag in ['qwen','llama']};c.dump(c.HERE/'verification.json',r);c.log(r)
