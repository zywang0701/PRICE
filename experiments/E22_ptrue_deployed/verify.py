import argparse,json,pickle,hashlib
import numpy as np
from scipy.special import logsumexp
import setup22 as s
from models import infer
from prepare import features
from evaluation import sweep,arms,GRID


def direct(g,lc,k,corr,tok,price):
    q,paths,_=g.shape
    threshold=np.where(g>0,np.log(np.maximum(g,1e-30))-lc,-np.inf);threshold[:,:,-1]=-np.inf
    n=(threshold<=price).argmax(2);qi=np.arange(q)[:,None];si=np.arange(paths)[None,:]
    kk=k[qi,si,n];tokens=tok[qi,si,n]
    return dict(R=corr[qi,si,n,kk].mean(1),C=np.exp(s.GAMMA*tokens).mean(1),M=tokens.mean(1),N=(n+1).mean(1))


def verify(tag):
    root=s.HERE/tag;sel=json.loads((root/'selection.json').read_text());result=json.loads((root/'results.json').read_text())
    assert hashlib.sha256((root/'models.pkl').read_bytes()).hexdigest()==sel['model_sha256']
    assert hashlib.sha256((root/'cdf.npy').read_bytes()).hexdigest()==sel['cdf_sha256']
    sets=json.loads((root/'query_splits.json').read_text())
    assert not(set(sets['fit'])&set(sets['tune'])) and not(set(sets['fit'])&set(sets['audit'])) and not(set(sets['tune'])&set(sets['audit']))
    raw=dict(np.load(root/'test.npz'));states=dict(np.load(root/'test_states.npz'));checked=0
    for name,g,k in arms(states):
        saved=dict(np.load(root/'test'/(name+'.npz')))
        for idx in np.unique(np.linspace(0,len(saved['grid'])-1,7).astype(int)):
            got=direct(g,states['logcost'],k,raw['corr'],raw['tok'],saved['grid'][idx])
            for key,v in got.items():np.testing.assert_allclose(v,saved[key][idx],rtol=1e-12,atol=1e-10)
            checked+=len(raw['qid'])*raw['corr'].shape[1]
    # Small randomized independent comparison exercises zero/negative gain and horizon.
    rng=np.random.default_rng(166);g=rng.uniform(-.01,.1,(4,3,64));lc=rng.normal(4,1,g.shape)
    kk=rng.integers(0,13,g.shape).astype(np.int8);corr=rng.integers(0,2,(*g.shape,13)).astype(bool)
    tok=np.cumsum(rng.integers(1,15,g.shape),axis=2);grid=np.r_[-np.inf,np.linspace(-15,3,25)]
    a=sweep(g,lc,kk,corr,tok,grid,s.GAMMA)
    for i,p in enumerate(grid):
        b=direct(g,lc,kk,corr,tok,p)
        for value,key in zip(a,['R','C','M','N']):np.testing.assert_allclose(value[i],b[key],rtol=1e-12,atol=1e-12)
    # Runtime inference succeeds without any label, gold, target-curve or true-cost field.
    with (root/'models.pkl').open('rb') as f:bundle=pickle.load(f)
    allowed={k:raw[k][:2] for k in ['feat','H','ell','tok','qid']}
    predicted=infer(bundle,allowed)
    for key in ['gains','envg']:np.testing.assert_array_equal(predicted[key],states[key][:,:2])
    for key in ['regression_k','conditional_k','logcost']:np.testing.assert_array_equal(predicted[key],states[key][:2])
    # Raw-prefix invariance, then invariance of runtime policy states.
    phi=rng.random(64);rawphi=phi**2;cl=rng.integers(0,5,64);ell=rng.integers(10,100,64);perm=np.arange(64)[None,:]
    a,w,t=features(phi,rawphi,cl,ell,-1,perm,raw['etas'])
    phi[8:]=rng.random(56);rawphi[8:]=rng.random(56);cl[8:]=rng.integers(0,10,56);ell[8:]*=4
    b,v,u=features(phi,rawphi,cl,ell,-1,perm,raw['etas'])
    np.testing.assert_array_equal(a[:,:4],b[:,:4]);np.testing.assert_array_equal(w[:,:8],v[:,:8])
    left={k:val.copy() for k,val in allowed.items()};right={k:val.copy() for k,val in allowed.items()}
    right['feat'][:,:,4:]=rng.normal(size=right['feat'][:,:,4:].shape)
    right['ell'][:,:,8:]*=3;right['tok']=np.cumsum(right['ell'],axis=2)
    p=infer(bundle,right)
    for key in ['gains','envg']:np.testing.assert_array_equal(predicted[key][:,:,:,:8],p[key][:,:,:,:8])
    for key in ['regression_k','conditional_k','logcost']:np.testing.assert_array_equal(predicted[key][:,:,:8],p[key][:,:,:8])
    # Primary comparison and all matched-budget identities from query-level saved values.
    mq=np.load(root/'matched_queries.npz');maxerr=0.
    for name,c in result['candidates'].items():
        delta=100*(mq[name+'__R']-mq['fixed__R'])
        np.testing.assert_allclose(delta.mean(1),c['gain']['mean'],atol=1e-12)
        risk=np.log(mq[name+'__C'].mean(1))/s.GAMMA
        assert np.all(risk<=s.BUDGETS+1e-7)
        for j,status in enumerate(c['statuses']):
            if status=='exact':maxerr=max(maxerr,abs(risk[j]-s.BUDGETS[j]))
    # Prospective prices are calibration-chosen; independently apply them to test states.
    policies={name:(g,k) for name,g,k in arms(states)}
    for label,v in result['prospective'].items():
        for j,p in enumerate(v['policies']):
            g,k=policies[p['arm']];prices=[-np.inf if x=='-infinity' else x for x in p['log_prices']]
            a,b=[direct(g,states['logcost'],k,raw['corr'],raw['tok'],price) for price in prices]
            for key in a:
                expected=(1-p['theta'])*a[key]+p['theta']*b[key]
                np.testing.assert_allclose(expected,mq['prospective_'+label+'__'+key][j],rtol=1e-12,atol=1e-10)
    report=dict(status='passed',frozen_model_and_cdf_hashes=True,calibration_query_folds_disjoint=True,
                question_text_overlap=json.loads((s.HERE/'question_disjointness.json').read_text())[tag]['question_hash_overlap'],
                direct_stop_path_checks=checked,synthetic_sweep_checks=len(grid),runtime_requires_no_labels=True,
                future_rollout_invariance=True,matched_budget_max_error_tokens=maxerr,prospective_prices_replayed=True,
                scoring='E16 repaired labels; native deployed ties',bootstrap='conditional paired query')
    s.dump(root/'verification.json',report);s.log(tag,report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);a=p.parse_args();verify(a.cell)
