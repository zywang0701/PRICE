import argparse,json,pickle
import setup25 as c
from setup25 import np,s
from analyze import sequence_actions


def verify(tag):
    root=c.HERE/tag;result=json.loads((root/'results.json').read_text());config=result['policy_config']
    assert not config['fixed_temperature_fallback_allowed']
    for path,digest in config['source_models'].items():assert c.sha(path)==digest
    sel23=json.loads((c.E23/tag/'selection.json').read_text())
    assert config['calibration_best_adaptive']==sel23['adaptive_candidate']
    cols=c.columns(tag);tune_values={}
    for name in ['regression_w1','regression_w4','conditional_w1','conditional_w4']:
        d=np.load(c.E22/tag/'tune'/(name+'.npz'))
        tune_values[name]=np.mean([s.E21.read_point(d['C'].mean(1),d['R'].mean(1),b,c.GAMMA)['R'] for b in cols])
    assert config['calibration_best_adaptive']==max(tune_values,key=tune_values.get)
    assert config['sequential_primary']==max(['regression_w1','regression_w4'],key=tune_values.get)
    old23=json.loads((c.E23/tag/'results.json').read_text());old24=json.loads((c.E24/tag/'results.json').read_text())
    np.testing.assert_allclose(result['policies']['sequential_calibration_best']['accuracy_pct'],old23['adaptive_candidate_accuracy_pct'],atol=1e-12)
    np.testing.assert_allclose(result['policies']['committed_joint']['accuracy_pct'],old24['committed_joint_accuracy_pct'],atol=1e-12)
    np.testing.assert_allclose(result['fixed_sequential_accuracy_pct'],old23['price_accuracy_pct'],atol=1e-12)
    np.testing.assert_allclose(result['fixed_committed_accuracy_pct'],old24['committed_accuracy_pct'],atol=1e-12)
    raw=np.load(c.E22/tag/'test.npz');qid=raw['qid'];Q=len(qid)
    for family in c.old.FAMILIES:
        d=np.load(c.E23/tag/'baselines'/(family+'.npz'))
        np.testing.assert_array_equal(qid,d['qid'])
        np.testing.assert_allclose(result['baseline_accuracy_pct'][family],old23['baseline_accuracy_pct'][family],atol=1e-12)
    with (c.E24/tag/'models.pkl').open('rb') as f:bundle=pickle.load(f)
    checks=0;maxgap=0.;rng=np.random.default_rng(20260926)
    for phase in ['tune','audit','test']:
        inp=np.load(c.E24/tag/phase/'inputs.npz');d=np.load(root/phase/'query_locked.npz')
        V,logM=c.predict(bundle,inp['H']);k=c.query_temperature(V)
        np.testing.assert_array_equal(d['k'],np.broadcast_to(k,d['k'].shape))
        for q in rng.choice(len(k),8,replace=False):
            # Single-query inference and batched inference give the same fixed vote.
            vv,ll=c.predict(bundle,inp['H'][q:q+1]);assert c.query_temperature(vv)[0]==k[q]
            for i in np.unique(np.linspace(0,len(d['n'])-1,17).astype(int)):
                objective=V[q,k[q]]-np.exp(d['grid'][i])*np.exp(logM[q]*np.arange(1,65))
                gap=float(objective.max()-objective[d['n'][i,q]-1]);maxgap=max(maxgap,gap)
                assert gap<=1e-8*max(1,abs(float(objective.max())))
                checks+=1
        if phase=='test':np.testing.assert_array_equal(qid,d['qid'])
    for name,policy in result['policies'].items():
        assert all(len([x for x in u['temperature_share'] if x>1e-8])>=2 for u in policy['usage']),name
        assert all(u['unique_temperatures_at_same_path'][1]>=2 for u in policy['usage']),name
        if name!='sequential_calibration_best':assert all(u['unique_temperatures_at_same_path'][0]>=2 for u in policy['usage']),name
        for j,b in enumerate(cols):
            assert policy['actual_risk'][j]<=b+1e-7
            np.testing.assert_allclose(policy['gain_vs_best_pp'][j],policy['accuracy_pct'][j]-result['best_baseline_accuracy_pct'][j],atol=1e-12)
    locked_usage=result['policies']['committed_query']['usage']
    for u in locked_usage:np.testing.assert_allclose(u['temperature_share'],np.array(result['query_locked_temperature_counts'])/Q,atol=1e-12)
    # Execute audit prices directly, without relying on compressed test-grid lookup.
    states=dict(np.load(c.E22/tag/'test_states.npz'))
    inp=np.load(c.E24/tag/'test/inputs.npz');qi=np.arange(Q)[:,None];si=np.arange(64)[None,:]
    for name,records in result['prospective'].items():
        for p in records:
            R=np.zeros(Q);C=np.zeros(Q)
            for ix,w in zip(p['grid_indices'],[1-p['theta'],p['theta']]):
                if name=='sequential_adaptive':
                    grid=np.r_[-np.inf,np.linspace(-450,6,769)]
                    n,k=sequence_actions(states,config['sequential_primary'],grid[ix])
                else:
                    if name=='committed_joint':n0,k0=s.E21.actions(inp['V'],inp['logM'][:,None]*np.arange(1,65),c.old.GRID[[ix]])
                    else:n0,k0=c.query_locked_actions(inp['V'],inp['logM'],c.old.GRID[[ix]])
                    n=np.broadcast_to(n0[0,:,None]-1,(Q,64));k=np.broadcast_to(k0[0,:,None],(Q,64))
                R+=w*raw['corr'][qi,si,n,k].mean(1)
                C+=w*np.exp(c.GAMMA*raw['tok'][qi,si,n].astype(float)).mean(1)
            np.testing.assert_allclose(100*R.mean(),p['accuracy_pct'],atol=1e-10)
            np.testing.assert_allclose(np.log(C.mean())/c.GAMMA,p['actual_risk'],atol=1e-8)
    report=dict(status='passed',model_hashes_unchanged=True,no_fixed_fallback=True,
        sequential_window_is_calibration_best_within_envelope_family=True,old_adaptive_results_exact=True,
        same_query_cohort_and_baselines=True,nonconstant_temperature_at_every_budget=True,
        across_query_variation_at_same_path=True,query_only_temperature_budget_invariant=True,
        query_locked_argmax_checks=checks,max_objective_gap=maxgap,prospective_prices_directly_executed=True,
        own_count_temperature_usage_and_raw_path_costs_checked_in_analysis=True)
    c.dump(root/'verification.json',report);print(tag,report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True,choices=['qwen','llama']);verify(p.parse_args().cell)
