import argparse,json,hashlib,pickle
import numpy as np
import setup22 as s
from evaluation import arms


def outputs_at(root,phase,name,point):
    d=np.load(root/phase/(name+'.npz'))
    return {k:s.E21.mix(d[k],point) for k in ['R','C','M','N']}


def at_price(d,price):
    i=max(0,int(np.searchsorted(d['grid'],price,side='right')-1))
    return {k:d[k][i] for k in ['R','C','M','N']}


def analyze(tag):
    root=s.HERE/tag;sel=json.loads((root/'selection.json').read_text())
    assert hashlib.sha256((root/'models.pkl').read_bytes()).hexdigest()==sel['model_sha256']
    assert hashlib.sha256((root/'cdf.npy').read_bytes()).hexdigest()==sel['cdf_sha256']
    test=json.loads((root/'test/readouts.json').read_text());audit=json.loads((root/'audit/readouts.json').read_text())
    raw=dict(np.load(root/'test.npz'));Q=len(raw['qid']);Sn=raw['corr'].shape[1]
    weights=np.random.default_rng(s.SEED).multinomial(Q,np.full(Q,1/Q),size=1000)/Q
    def ci(d):
        b=weights@d.T
        return dict(mean=d.mean(1).tolist(),lo=np.quantile(b,.025,axis=0).tolist(),hi=np.quantile(b,.975,axis=0).tolist())
    fixed={k:np.stack([outputs_at(root,'test',name,test[name][j])[k] for j,name in enumerate(sel['fixed_arms_by_budget'])]) for k in ['R','C','M','N']}
    candidates={};arrays={'fixed__'+k:v for k,v in fixed.items()}
    for name in sel['all_adaptive_candidates']:
        new={k:np.stack([outputs_at(root,'test',name,p)[k] for p in test[name]]) for k in ['R','C','M','N']}
        candidates[name]=dict(accuracy_pct=(100*new['R'].mean(1)).tolist(),gain=ci(100*(new['R']-fixed['R'])),
            mean_tokens=new['M'].mean(1).tolist(),mean_n=new['N'].mean(1).tolist(),
            actual_risk=(np.log(new['C'].mean(1))/s.GAMMA).tolist(),statuses=[p['status'] for p in test[name]])
        arrays.update({name+'__'+k:v for k,v in new.items()})
    selected=sel['adaptive_candidate'];states=dict(np.load(root/'test_states.npz'))
    policies={name:(g,k) for name,g,k in arms(states)};g,k=policies[selected]
    # Preserve the fixed baseline's own counts/endpoints, changing only its vote.
    same=[];same_strict=[];fixed_strict=[];usage=np.zeros((7,13))
    for j,fixed_name in enumerate(sel['fixed_arms_by_budget']):
        point=test[fixed_name][j];fd=np.load(root/'test'/(fixed_name+'.npz'));fg,fk=policies[fixed_name]
        reward=np.zeros(Q);strict=np.zeros(Q);fs=np.zeros(Q)
        for endpoint,w in [(point['lo'],1-point['theta']),(point['hi'],point['theta'])]:
            price=fd['grid'][endpoint];threshold=np.where(fg>0,np.log(np.maximum(fg,1e-30))-states['logcost'],-np.inf)
            threshold[:,:,-1]=-np.inf;n=(threshold<=price).argmax(2)
            qi=np.arange(Q)[:,None];si=np.arange(Sn)[None,:];kk=k[qi,si,n];basek=fk[qi,si,n]
            reward+=w*raw['corr'][qi,si,n,kk].mean(1)
            strict+=w*raw['strict'][qi,si,n,kk].mean(1);fs+=w*raw['strict'][qi,si,n,basek].mean(1)
            usage[j]+=w*np.bincount(kk.ravel(),minlength=13)/(Q*Sn)
        same.append(reward);same_strict.append(strict);fixed_strict.append(fs)
    same=np.stack(same);same_strict=np.stack(same_strict);fixed_strict=np.stack(fixed_strict)
    arrays['same_fixed_stops__R']=same
    # Audit chooses only prices/mixtures; apply those exact prices to test trajectories.
    prospective={}
    for label in ['adaptive_candidate','fixed']:
        pq={k:[] for k in ['R','C','M','N']};details=[]
        for j in range(7):
            name=selected if label=='adaptive_candidate' else sel['fixed_arms_by_budget'][j]
            p=audit[name][j];ad=np.load(root/'audit'/(name+'.npz'));td=np.load(root/'test'/(name+'.npz'))
            prices=[float(ad['grid'][p['lo']]),float(ad['grid'][p['hi']])]
            a,b=[at_price(td,price) for price in prices]
            for key in pq:pq[key].append((1-p['theta'])*a[key]+p['theta']*b[key])
            details.append(dict(arm=name,log_prices=[x if np.isfinite(x) else '-infinity' for x in prices],theta=p['theta']))
        pq={k:np.stack(v) for k,v in pq.items()};arrays.update({'prospective_'+label+'__'+k:v for k,v in pq.items()})
        prospective[label]=dict(accuracy_pct=(100*pq['R'].mean(1)).tolist(),actual_risk=(np.log(pq['C'].mean(1))/s.GAMMA).tolist(),
                               mean_tokens=pq['M'].mean(1).tolist(),policies=details)
    counts=[2,4,8,16,32,64];qi=np.arange(Q)[:,None,None];si=np.arange(Sn)[None,:,None];n=np.array(counts)[None,None,:]-1
    chosen=k[qi,si,n];cc=raw['corr'][qi,si,n,chosen].mean(1)
    k0=json.loads((root/'model_training.json').read_text())['reference_temperature_index']
    ff=raw['corr'][:,:,np.array(counts)-1,k0].mean(1)
    fixed_count=dict(counts=counts,adaptive_accuracy_pct=(100*cc.mean(0)).tolist(),reference_fixed_index=k0,
                     reference_accuracy_pct=(100*ff.mean(0)).tolist(),gain=ci(100*(cc-ff).T))
    result=dict(model=tag,queries=Q,paths=Sn,selection=sel,budgets=s.BUDGETS.tolist(),
        fixed_accuracy_pct=(100*fixed['R'].mean(1)).tolist(),fixed_actual_risk=(np.log(fixed['C'].mean(1))/s.GAMMA).tolist(),
        fixed_mean_tokens=fixed['M'].mean(1).tolist(),fixed_mean_n=fixed['N'].mean(1).tolist(),candidates=candidates,
        selected_deployment_gain_pp=([0.]*7 if sel['primary']=='fixed_fallback' else candidates[selected]['gain']['mean']),
        same_fixed_stops=dict(gain=ci(100*(same-fixed['R'])),strict_gain=ci(100*(same_strict-fixed_strict)),
            conflict_free_gain=(100*(same-fixed['R'])[:,~raw['conflict']].mean(1)).tolist(),temperature_usage=usage.tolist()),
        prospective=prospective,prospective_gain=ci(100*(arrays['prospective_adaptive_candidate__R']-arrays['prospective_fixed__R'])),
        fixed_count=fixed_count,bootstrap='1000 paired query draws, all policies and readout choices frozen')
    s.dump(root/'results.json',result);np.savez_compressed(root/'matched_queries.npz',**arrays,qid=raw['qid'])
    s.log(tag,'selected',sel['primary'],'adaptive candidate',selected,'gain',candidates[selected]['gain'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);a=p.parse_args();analyze(a.cell)
