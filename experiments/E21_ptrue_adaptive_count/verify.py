"""Verify price decisions, B rewards/costs, and fixed-budget readout independently."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
import setup21 as s


def verify(tag,mode):
    out=s.HERE/tag/mode;t=s.load(tag,mode);meta=json.loads((out/'results.json').read_text())
    for p,h in json.loads((out/'source.json').read_text()).items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    Q=len(t['qid']);K=meta['temperatures'];qi=np.arange(Q)[None,:]
    rng=np.random.default_rng(1721);price_checks=0;outcome_checks=0;max_error=0.
    for name in ['joint',*[f'fixed_{k}' for k in range(K)]]:
        d=dict(np.load(out/f'{name}.npz'));L=len(d['lambdas'])
        reward=np.zeros((L,Q));cost=np.zeros_like(reward);mean=np.zeros_like(reward)
        if name=='joint':glob=np.zeros_like(reward)
        for dr in range(8):
            part,side=divmod(dr,2);n=d['n'][dr].astype(int)-1;k=d['k'][dr]
            A=t['A'][part,side].astype(float);B=t['B'][part,side];C=np.exp(t['logC_A'][part,side])
            reward+=B[qi,k,n].astype(float)/8
            cost+=np.exp(t['logC_B'][part,1-side])[qi,n]/8
            mean+=t['mean'][part,1-side][qi,n]/8
            for j in np.unique(np.r_[0,L-1,rng.integers(0,L,6)]):
                lam=d['lambdas'][j]
                if name=='joint':R=A.max(1)
                else:
                    fixed=int(name.split('_')[1]);R=A[:,fixed]
                    np.testing.assert_array_equal(k[j],np.full(Q,fixed))
                chosen=A[np.arange(Q),k[j],n[j]]-lam*C[np.arange(Q),n[j]]
                direct=(R-lam*C).max(1)
                np.testing.assert_allclose(chosen,direct,rtol=1e-11,atol=1e-12);price_checks+=Q
                if name=='joint':
                    expected=A[np.arange(Q)[:,None],np.arange(K)[None,:],n[j,:,None]].mean(0).argmax()
                    assert expected==d['global_k'][dr,j]
            if name=='joint':glob+=B[qi,d['global_k'][dr,:,None],n].astype(float)/8
            outcome_checks+=L*Q
        np.testing.assert_allclose(reward,d['R_B'],rtol=0,atol=1e-13)
        np.testing.assert_allclose(cost,d['C_B'],rtol=1e-13)
        np.testing.assert_allclose(mean,d['mean_B'],rtol=1e-13)
        if name=='joint':np.testing.assert_allclose(glob,d['global_vote_B'],rtol=0,atol=1e-13)
        for j,p in enumerate(meta['all_fixed_readouts'][name]):
            # Use direct scalar mixture arithmetic, independently of read_point/mix.
            lo,hi,w=p['lo'],p['hi'],p['theta'];assert 0<=w<=1
            Cb=(1-w)*cost[lo].mean()+w*cost[hi].mean()
            rb=(1-w)*reward[lo].mean()+w*reward[hi].mean()
            b=np.log(Cb)/s.GAMMA
            assert abs(100*rb-p['accuracy_B_pct'])<1e-10
            assert b<=s.BUDGETS[j]+1e-7
            if p['status']=='exact':max_error=max(max_error,abs(b-s.BUDGETS[j]))
    q=dict(np.load(out/'matched_queries.npz'));clean=~t['conflict']
    np.testing.assert_allclose(100*(q['joint__R_B']-q['fixed__R_B']).mean(1),meta['contrasts']['best_fixed']['mean'],atol=1e-12)
    np.testing.assert_allclose(100*(q['joint__R_B']-q['fixed__R_B'])[:,clean].mean(1),meta['conflict_free_joint_minus_best_fixed_pp'],atol=1e-12)
    for j,p in enumerate(meta['methods']['best_fixed_hindsight']):
        expected=np.argmax([meta['all_fixed_readouts'][f'fixed_{k}'][j]['R'] for k in range(K)])
        assert expected==p['temperature_index']
    # Match primary E20 n=32 source-table results before variable-count selection.
    k=t['A'].argmax(3);b=np.take_along_axis(t['B'],k[:,:,:,None,:],axis=3).squeeze(3)
    previous=json.loads((s.E20/tag/'ptrue/results.json').read_text())
    np.testing.assert_allclose(100*b[:,:,:,31].mean(dtype=float),previous['modes'][mode]['accuracy_pct']['adaptive'][31],atol=1e-12)
    # Raw A lengths reproduce the entire cost primitive; independently replay B paths.
    raw=s.D.load(tag,'test');halves=np.load(s.E17/tag/'tables.npz')['half_indices']
    for part in range(4):
        for side in [0,1]:
            ell=raw['ell'][np.arange(Q)[:,None],halves[part,side]]
            logM=logsumexp(s.GAMMA*ell,axis=1)-np.log(64)
            np.testing.assert_allclose(logM[:,None]*np.arange(1,65),t['logC_A'][part,side],rtol=1e-13)
    for part,side,idx in [(0,1,0),(1,0,37),(2,1,Q//2),(3,0,Q-1)]:
        half=halves[part,side,idx]
        rng=np.random.default_rng([s.SEED,int(raw['qid'][idx]),part,side,1])
        perms=np.stack([half[rng.permutation(64)] for _ in range(256)])
        tokens=raw['ell'][idx][perms].cumsum(1)
        np.testing.assert_allclose(logsumexp(s.GAMMA*tokens,axis=0)-np.log(256),t['logC_B'][part,side,idx],rtol=1e-13)
        np.testing.assert_allclose(tokens.mean(0),t['mean'][part,side,idx])
    assert meta['max_half_grid_difference_pp']<=.03
    return dict(status='passed',source_hashes_verified=True,A_only_priced_actions_checked=price_checks,
                B_query_direction_outcomes_checked=outcome_checks,B_rewards_and_path_costs_reconciled=True,
                all_A_cost_primitives_reconstructed=True,raw_B_cost_contexts=4,E20_n32_result_reconciled=True,
                equal_count_A_global_actions_reconciled=True,frontier_temperature_hindsight_selection_checked=True,
                max_budget_error_tokens=max_error,max_half_grid_difference_pp=meta['max_half_grid_difference_pp'])


if __name__=='__main__':
    results={}
    for tag in ['qwen','llama']:
        results[tag]={}
        for mode in s.MODES:
            results[tag][mode]=verify(tag,mode);s.log(tag,mode,results[tag][mode])
    s.dump(s.HERE/'verification.json',results)
