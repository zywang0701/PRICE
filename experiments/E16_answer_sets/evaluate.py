"""All candidates are frozen before any new test scoring; costs are identical."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
import data as d
import features as f
import model as m
from split_pool import bootstrap


def gather(corr,k,counts):
    q,s,t,_=corr.shape
    qi=np.arange(q)[:,None,None];si=np.arange(s)[None,:,None]
    if np.ndim(counts)==1:counts=np.broadcast_to(counts[None,None,:],(q,s,len(counts)))
    kk=k[qi,si,counts-1]
    return corr[qi,si,counts-1,kk]


def run(tag):
    root=d.HERE/tag;selpath=root/'selection.json'
    selection_hash=hashlib.sha256(selpath.read_bytes()).hexdigest();selection=json.loads(selpath.read_text())
    assert all((root/f'{family}.pt').exists() for family in m.FAMILIES)
    if not (root/'test_data.npz').exists():d.prepare(tag,'test')
    raw=d.load(tag,'test');replay=dict(np.load(d.HERE.parent/'E02_frozen'/tag/'replay.npz'))
    q,s,t,k=replay['corr'].shape
    assert np.array_equal(raw['qid'],replay['qid']) and np.array_equal(raw['etas'],replay['etas'])
    nets={}
    for family in m.FAMILIES:
        bundle=torch.load(root/f'{family}.pt',map_location='cpu',weights_only=False)
        net=m.AnswerNet(family);net.load_state_dict(bundle['state']);nets[family]=net.eval()
    prediction={fam:np.empty((q,s,len(d.CP),13),np.float32) for fam in m.FAMILIES}
    corr=np.zeros_like(replay['corr']);strict=np.zeros_like(corr)
    seconds={fam:0. for fam in m.FAMILIES};feature_seconds=0.
    for i in range(q):
        perms=replay['perms'][i]
        _,winners,tokens=d.old.prefix_features(raw['phi'][i],raw['clu'][i],raw['ell'][i],raw['none'][i],perms,raw['etas'])
        old_good=(raw['clu'][i]==raw['old_target'][i])&(raw['clu'][i]!=raw['none'][i])
        old_corr=d.score_winners(winners,raw['clu'][i],old_good)
        expected=replay['corr'][i].copy()
        if not old_good.any():expected[:]=False
        assert np.array_equal(old_corr,expected),f'Old replay mismatch: {tag} {raw["qid"][i]}'
        assert np.array_equal(tokens,replay['tok'][i]),'Token mismatch'
        corr[i]=d.score_winners(winners,raw['clu'][i],raw['good'][i])
        strict[i]=d.score_winners(winners,raw['clu'][i],raw['strict_good'][i])
        started=time.perf_counter();records=[]
        for path in perms:
            for n in d.CP:
                p=path[:n]
                r=f.state(raw['phi'][i,p],raw['clu'][i,p],raw['ell'][i,p],raw['none'][i],raw['etas'])
                ans=np.where(r[2]>=0,r[3][np.maximum(r[2],0)],-32768)
                assert np.array_equal(ans,winners[len(records)//len(d.CP),n-1]),'Feature/vote mismatch'
                records.append(r)
        z=f.pack(records);feature_seconds+=time.perf_counter()-started
        for family,net in nets.items():
            started=time.perf_counter()
            v=m.scores(net,z,np.arange(len(records))).reshape(s,len(d.CP),13)
            if family=='cross_pool':v[:,6:]=v[:,5:6]
            prediction[family][i]=v;seconds[family]+=time.perf_counter()-started
        if i%50==0:d.log(tag,'test replay',i,'/',q)
    stage=np.searchsorted(d.CP,np.arange(1,65),side='right')-1;k0=selection['k_fixed']
    mats={family:m.choose(v,k0,selection['selections'][family]['threshold'])[:,:,stage] for family,v in prediction.items()}
    mats['fixed']=np.full((q,s,t),k0,np.int8)
    mats['original_fixed']=np.full((q,s,t),selection['original_k_fixed'],np.int8)
    states=np.load(d.HERE.parent/'E02_frozen'/tag/'states_adaptive.npz')
    mats['old_deployed']=states['kmat']
    e14=json.loads((d.HERE.parent/'E14_neural_voting/combined_selection.json').read_text())['cells'][tag]['selected']
    loc=d.HERE.parent/'E14_neural_voting'/('' if e14['scope']=='main' else e14['scope'])/tag
    mats['previous_neural']=np.load(loc/'selected_temperatures.npz')[e14['family']]
    operating=np.load(d.HERE.parent/'E15_query_diagnostics'/tag/'diagnostic_paths.npz')['stops']
    meta=json.loads((d.HERE.parent/'E15_query_diagnostics'/tag/'summary.json').read_text())
    contexts={'stops':operating,'counts':d.EVAL};results={};per_query={};outcomes={}
    for context,counts in contexts.items():
        scores={name:gather(corr,kk,counts) for name,kk in mats.items()}
        strict_scores={name:gather(strict,kk,counts) for name,kk in mats.items()}
        results[context]={};per_query[context]={}
        for name,cc in scores.items():
            pq=cc.mean(1);base=scores['fixed'];old=scores['old_deployed']
            results[context][name]=dict(accuracy_pct=(100*pq.mean(0)).tolist(),
                versus_fixed=bootstrap(pq-base.mean(1)),versus_old_deployed=bootstrap(pq-old.mean(1)),
                versus_previous_neural=bootstrap(pq-scores['previous_neural'].mean(1)),
                strict_versus_fixed=bootstrap(strict_scores[name].mean(1)-strict_scores['fixed'].mean(1)),
                clean_versus_fixed=bootstrap((pq-base.mean(1))[~raw['conflict']]),
                rescue_pp=(100*(cc&~base).mean((0,1))).tolist(),harm_pp=(100*(~cc&base).mean((0,1))).tolist())
            per_query[context][name]=pq;outcomes[context+'__'+name]=cc
    cases=[]
    ids=[58,234,385] if tag=='qwen' else [29,141,448,345]
    for qid in ids:
        i=int(np.flatnonzero(raw['qid']==qid)[0])
        cases.append(dict(query_id=qid,accuracy_pct={name:float(100*p[i].mean()) for name,p in per_query['stops'].items()}))
    summary=dict(primary=selection['primary'],scoring='All correct clusters; conflict-query representatives regraded',
        queries=q,paths=s,actual_risk_budgets=meta['actual_risk_budgets'],nominal_risk_budgets=meta['nominal_risk_budgets'],
        results=results,cases=cases,inference_seconds=seconds,feature_seconds=feature_seconds,
        inference_states=q*s*len(d.CP),same_tokens_verified=True,exact_old_votes_verified=True,
        selector_has_no_gold_or_budget_inputs=True,selection_sha256=selection_hash,
        temperature_usage={name:np.bincount(kk.ravel(),minlength=13).tolist() for name,kk in mats.items()})
    assert selection_hash==hashlib.sha256(selpath.read_bytes()).hexdigest()
    d.dump(root/'results.json',summary)
    np.savez_compressed(root/'predicted_utilities.npz',**prediction)
    np.savez_compressed(root/'temperatures.npz',**mats)
    np.savez_compressed(root/'outcomes.npz',**outcomes,qid=raw['qid'],conflict=raw['conflict'])
    d.log(tag,'DONE; primary',summary['primary'],results['stops'][summary['primary']]['versus_fixed'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);args=p.parse_args();run(args.cell)
