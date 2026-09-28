from pathlib import Path
import json,hashlib
import numpy as np
from scipy.special import logsumexp
import setup20 as s


def analyze(tag,score):
    path=s.HERE/tag/score;tab=dict(np.load(path/'tables.npz'));Q=len(tab['qid'])
    weights=np.random.default_rng(s.SEED).multinomial(Q,np.full(Q,1/Q),size=1000)/Q
    qi=np.arange(Q)[:,None];ni=np.arange(64)[None,:]
    all_modes={};saved={}
    for mode,name in enumerate(s.MODES):
        A=tab['V_A'][mode].astype(float);B=tab['V_B'][mode].astype(float)
        assert A.shape==B.shape==(4,2,Q,13,64)
        rows={k:[] for k in ['adaptive','A_global','sc','bon','adaptive_A','global_A','B_per_query','adaptive_strict','sc_strict']}
        kdirs=[];gdirs=[]
        for part in range(4):
            for side in [0,1]:
                a=A[part,side];b=B[part,side];bs=tab['V_strict_B'][mode,part,side]
                ka=a.argmax(1);kg=a.mean(0).argmax(0)
                vals=dict(adaptive=b[qi,ka,ni],A_global=b[qi,kg[None,:],ni],sc=b[:,0],bon=b[:,-1],
                    adaptive_A=a[qi,ka,ni],global_A=a[qi,kg[None,:],ni],B_per_query=b.max(1),
                    adaptive_strict=bs[qi,ka,ni],sc_strict=bs[:,0])
                assert np.all(vals['adaptive_A']>=vals['global_A'])
                for k,v in vals.items():rows[k].append(v)
                kdirs.append(ka);gdirs.append(kg)
        rows={k:np.stack(v).mean(0) for k,v in rows.items()}
        kb=B.mean(axis=(0,1,2)).argmax(0)
        rows['B_global']=B.mean(axis=(0,1))[qi,kb[None,:],ni]
        assert np.all(rows['B_global'].mean(0)>=rows['sc'].mean(0)-1e-12)
        for k in ['adaptive','sc','bon','A_global','B_global']:
            np.testing.assert_allclose(rows[k][:,0],rows['sc'][:,0],atol=1e-12,rtol=0)
            assert np.all(rows['B_per_query']>=rows[k]-1e-12)
        contrasts={}
        for base in ['sc','A_global','B_global']:
            d=100*(rows['adaptive']-rows[base]);boot=weights@d
            contrasts[base]=dict(mean=d.mean(0).tolist(),lo=np.quantile(boot,.025,axis=0).tolist(),hi=np.quantile(boot,.975,axis=0).tolist())
        all_modes[name]=dict(accuracy_pct={k:(100*v.mean(0)).tolist() for k,v in rows.items()},contrasts=contrasts,
            B_global_temperature_indices=kb.tolist(),
            strict_gain_vs_sc=(100*(rows['adaptive_strict']-rows['sc_strict']).mean(0)).tolist(),
            conflict_free_gain_vs_sc=(100*(rows['adaptive']-rows['sc'])[~tab['conflict']].mean(0)).tolist())
        saved.update({name+'__'+k:v for k,v in rows.items()})
        saved[name+'__k']=np.stack(kdirs);saved[name+'__global_k']=np.stack(gdirs)
    costs=dict(np.load(s.E18/tag/'replay_cost.npz'))
    risk=(logsumexp(costs['logC'],axis=(0,1,2))-np.log(8*Q))/s.GAMMA
    result=dict(model=tag,score=score,queries=Q,counts=list(range(1,65)),summary_counts=s.COUNTS,
        modes=all_modes,B_risk_cost=risk.tolist(),B_mean_tokens=costs['mean_tokens'].mean(axis=(0,1,2)).tolist(),
        temperatures=[('infinity' if np.isinf(x) else float(x)) for x in tab['etas']],
        normalization='Pooled A-only empirical CDF, refit per source direction and frozen for A/B',
        table_sha256=hashlib.sha256((path/'tables.npz').read_bytes()).hexdigest(),bootstrap_replicates=1000)
    s.dump(path/'results.json',result);np.savez_compressed(path/'outcomes.npz',**saved,qid=tab['qid'])
    s.log(tag,score,'n32 primary', {k:round(v[31],3) for k,v in all_modes['common_tie']['accuracy_pct'].items()})
    return result


def compare():
    summary={}
    for tag in ['qwen','llama']:
        ref=dict(np.load(s.HERE/tag/'deepconf/outcomes.npz'));Q=len(ref['qid'])
        weights=np.random.default_rng(s.SEED).multinomial(Q,np.full(Q,1/Q),size=1000)/Q
        summary[tag]={}
        for score in s.SCORES:
            out=dict(np.load(s.HERE/tag/score/'outcomes.npz'))
            np.testing.assert_array_equal(ref['qid'],out['qid'])
            np.testing.assert_array_equal(ref['common_tie__sc'],out['common_tie__sc'])
            summary[tag][score]={}
            for mode in s.MODES:
                delta=100*(out[mode+'__adaptive']-ref[mode+'__adaptive']);boot=weights@delta
                summary[tag][score][mode]=dict(adaptive_minus_deepconf_pp=delta.mean(0).tolist(),
                    lo=np.quantile(boot,.025,axis=0).tolist(),hi=np.quantile(boot,.975,axis=0).tolist())
    s.dump(s.HERE/'cross_score_comparison.json',summary)
    s.log('All common-tie SC query-level outcomes are exactly equal across scores.')


if __name__=='__main__':
    for tag in ['qwen','llama']:
        for score in s.SCORES:analyze(tag,score)
    compare()
