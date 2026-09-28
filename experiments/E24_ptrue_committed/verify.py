import argparse,json,pickle,hashlib,inspect
import setup24 as c
from setup24 import np,s
from models import predict
from controller import CommittedController
from run import load_arm


def verify(tag):
    root=c.HERE/tag;sel=json.loads((root/'selection.json').read_text());training=json.loads((root/'training.json').read_text())
    assert hashlib.sha256((root/'models.pkl').read_bytes()).hexdigest()==sel['model_sha256']==training['model_sha256']
    assert training['selected']==min(training['candidates'],key=lambda z:z['mse'])['name']
    tune=json.loads((root/'tune/readouts.json').read_text())
    fixed=[max([n for n in tune if n.startswith('fixed_')],key=lambda n:tune[n][j]['R']) for j in range(6)]
    gain=100*np.mean([tune['joint'][j]['R']-tune[n][j]['R'] for j,n in enumerate(fixed)])
    assert fixed==sel['fixed_arms'] and sel['primary_arms']==(['joint']*6 if gain>0 else fixed)
    with (root/'models.pkl').open('rb') as f:bundle=pickle.load(f)
    assert list(inspect.signature(predict).parameters)==['bundle','H']
    rng=np.random.default_rng(20260925);checked=0;runtime_checks=0;maxgap=0.
    for phase in ['tune','audit','test']:
        inp=np.load(root/phase/'inputs.npz');V,logM=predict(bundle,inp['H'])
        np.testing.assert_allclose(V,inp['V'],atol=1e-12);np.testing.assert_allclose(logM,inp['logM'],atol=1e-12)
        for path in (root/phase).glob('*.npz'):
            if path.stem=='inputs':continue
            a=np.load(path);indices=np.unique(np.r_[0,len(a['n'])-1,np.linspace(0,len(a['n'])-1,15).astype(int)])
            qs=rng.choice(len(logM),min(8,len(logM)),replace=False)
            for i in indices:
                price=np.exp(a['log_grid'][i])
                for q in qs:
                    objective=V[q].T-price*np.exp(logM[q]*np.arange(1,65))[:,None]
                    k=int(a['k'][i,q]);n=int(a['n'][i,q])-1
                    optimum=float(objective.max()) if path.stem=='joint' else float(objective[:,int(path.stem.split('_')[1])].max())
                    gap=optimum-float(objective[n,k]);maxgap=max(maxgap,gap)
                    assert gap<=1e-8*max(1,abs(optimum)),(phase,path.stem,q,i,gap)
                    checked+=1
    raw=np.load(c.E22/tag/'test.npz');inp=np.load(root/'test/inputs.npz')
    controller=CommittedController(root,raw['etas'])
    for name in sorted(set(sel['primary_arms']+['joint'])):
        a=load_arm(tag,'test',name)
        for i in np.unique(np.linspace(0,len(a['n'])-1,7).astype(int)):
            for q in [0,17,100]:
                k=int(a['k'][i,q]);n=int(a['n'][i,q])-1
                reward=raw['corr'][q,:,n,k].mean();cost=np.exp(c.GAMMA*raw['tok'][q,:,n].astype(float)).mean()
                np.testing.assert_allclose(reward,a['R'][i,q],atol=1e-12);np.testing.assert_allclose(cost,a['C'][i,q],rtol=1e-12)
                decision=controller.decide(inp['H'][q],a['log_grid'][i],None if name=='joint' else int(name.split('_')[1]))
                assert decision['count']==n+1 and decision['temperature_index']==k
                runtime_checks+=1
    # Independently execute the audit-calibrated prices without test-frontier interpolation.
    result=json.loads((root/'results.json').read_text());arrays=np.load(root/'matched_queries.npz')
    for j,p in enumerate(result['prospective']['policies']):
        V=inp['V'];logC=inp['logM'][:,None]*np.arange(1,65)
        n,k=s.E21.actions(V,logC,c.GRID[p['grid_indices']],None if p['arm']=='joint' else int(p['arm'].split('_')[1]))
        qi=np.arange(len(inp['qid']))[None,:]
        for key,values in [('R',inp['R'][qi,k,n-1]),('C',inp['C'][qi,n-1]),('M',inp['M'][qi,n-1])]:
            expected=(1-p['theta'])*values[0]+p['theta']*values[1]
            np.testing.assert_allclose(expected,arrays['prospective__'+key][j],rtol=1e-12,atol=1e-10)
    maxerr=0.
    for j,b in enumerate(c.columns(tag)):
        risk=np.log(arrays['committed__C'][j].mean())/c.GAMMA;assert risk<=b+1e-7;maxerr=max(maxerr,abs(risk-b))
    old=json.loads((c.E23/tag/'results.json').read_text())
    np.testing.assert_allclose(result['sequential_accuracy_pct'],old['price_accuracy_pct'],atol=1e-12)
    np.testing.assert_allclose(result['best_baseline_accuracy_pct'],old['best_baseline_accuracy_pct'],atol=1e-12)
    report=dict(status='passed',frozen_model_and_calibration_choices=True,query_embedding_only_runtime=True,
        per_query_count_shared_by_all_replay_paths=True,direct_argmax_checks=checked,max_objective_gap=maxgap,
        runtime_and_raw_path_checks=runtime_checks,audit_prices_independently_executed=True,
        E23_baselines_and_sequential_exact=True,matched_budget_max_error_tokens=float(maxerr),
        max_half_grid_change_pp=max(abs(x) for v in result['half_grid_change_pp'].values() for x in v))
    c.dump(root/'verification.json',report);c.log(tag,report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);verify(p.parse_args().cell)
