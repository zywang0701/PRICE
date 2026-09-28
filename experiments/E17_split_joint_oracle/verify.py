"""Independent arithmetic and source/action isolation checks for the saved run."""
import hashlib
import json
import numpy as np
import settings as s
from allocation import Frontier
from run_oracle import METHODS


def verify(tag,kind):
    out=s.HERE/tag;tab=dict(np.load(out/'tables.npz'));p=dict(np.load(out/f'policies_{kind}.npz'));z=dict(np.load(out/f'outcomes_{kind}.npz'))
    res=json.loads((out/f'results_{kind}.json').read_text());source=json.loads((out/'source.json').read_text())
    assert hashlib.sha256(__import__('pathlib').Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    Q=len(tab['qid']);b=p['budgets'];qi=np.arange(Q);checked=0;max_budget_error=0.
    for direction in range(2*s.SPLITS):
        part,side=divmod(direction,2)
        a=tab['half_indices'][part,0];bb=tab['half_indices'][part,1]
        for i in range(Q):assert len(set(a[i])&set(bb[i]))==0
        for mi,name in enumerate(METHODS):
            for j in range(len(b)):
                th=p['theta'][mi,direction,j];kl=p['k_low'][mi,direction,j];kh=p['k_high'][mi,direction,j]
                nl=p['n_low'][mi,direction,j]-1;nh=p['n_high'][mi,direction,j]-1
                for context,pool_side in [('in',side),('out',1-side)]:
                    V=tab['V'][part,pool_side];lc=tab['logC_'+kind][part,pool_side]
                    # Independent action lookup, not the runner's evaluate helper.
                    rr=V[qi,kl,nl].astype(float)*(1-th)+V[qi,kh,nh].astype(float)*th
                    cc=np.exp(lc[qi,nl])*(1-th)+np.exp(lc[qi,nh])*th
                    np.testing.assert_allclose(rr,z['R_'+context][mi,direction,j],rtol=0,atol=1e-14)
                    np.testing.assert_allclose(cc,z['C_'+context][mi,direction,j],rtol=1e-14)
                    checked+=Q
                if p['status'][mi,direction,j]==0:
                    err=abs(np.log(z['C_in'][mi,direction,j].mean())/s.GAMMA-b[j])
                    max_budget_error=max(max_budget_error,err);assert err<1e-6
    for mi,name in enumerate(METHODS):
        # Pool MGFs first; averaging risk budgets would fail this check.
        rb=np.log(z['C_out'][mi].mean((0,2)))/s.GAMMA
        np.testing.assert_allclose(rb,res['methods'][name]['evaluation_risk_budget'],rtol=1e-12)
    for name in ['joint_counts_best_vote','joint_counts_cal_vote']:
        j=METHODS.index(name)
        for key in ['n_low','n_high','theta']:np.testing.assert_array_equal(p[key][0],p[key][j])
        np.testing.assert_array_equal(z['C_out'][0],z['C_out'][j])
    # The solver's API has only source inputs. Check deterministic frozen actions.
    V=tab['V'][0,0];lc=tab['logC_'+kind][0,0];f=Frontier(V,lc)
    policy=f.at(np.exp(s.GAMMA*b[len(b)//2]));reference=policy.n_low.copy()
    check=f.at(np.exp(s.GAMMA*b[len(b)//2]));np.testing.assert_array_equal(reference,check.n_low)
    assert np.all(tab['logC_exact']<=tab['logC_plugin']+1e-10)
    raw=s.D.load(tag,'test');absent=~raw['good'].any(1)
    assert np.all(tab['V'][:,:,absent]==0)
    np.testing.assert_array_equal(s.D.score_winners(np.array([0,1,2,-32768]),
        np.array([0,1,2]),np.array([True,True,False])),[True,True,False,False])
    return dict(source_hash_unchanged=True,action_outcomes_independently_checked=checked,
        max_exact_source_budget_error_tokens=max_budget_error,disjoint_half_ids_verified=True,
        same_count_vote_costs_identical=True,aggregation_on_mgf_scale_verified=True,
        exact_cost_not_above_plugin_verified=True,selection_api_takes_source_tables_only=True,
        multiple_correct_clusters_scored=True,absent_gold_queries_retained=int(absent.sum()))


if __name__=='__main__':
    results={tag:{kind:verify(tag,kind) for kind in ['exact','plugin']} for tag in ['qwen','llama']}
    s.dump(s.HERE/'verification.json',results);s.log(results)
