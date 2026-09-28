import argparse,json
import setup25 as c
from setup25 import np,s


def load_seq(tag,phase,name):return dict(np.load(c.E22/tag/phase/(name+'.npz')))


def locked(tag,phase):
    inp=np.load(c.E24/tag/phase/'inputs.npz')
    n,k=c.query_locked_actions(inp['V'],inp['logM'],c.old.GRID)
    changed=np.r_[True,np.any(n[1:]!=n[:-1],axis=1)]
    keep=s.E21.compact_indices(changed);n=n[keep];k=k[keep];qi=np.arange(len(inp['qid']))[None,:]
    d=dict(n=n,k=k,grid_indices=keep,grid=c.old.LOG_GRID[keep],qid=inp['qid'],
           R=inp['R'][qi,k,n-1],C=inp['C'][qi,n-1],M=inp['M'][qi,n-1])
    out=c.HERE/tag/phase;out.mkdir(exist_ok=True)
    np.savez_compressed(out/'query_locked.npz',**d)
    return d


def points(d,cols):return [s.E21.read_point(d['C'].mean(1),d['R'].mean(1),b,c.GAMMA) for b in cols]


def read_many(C,R,cols):
    h=s.E21.frontier(C,R)
    return np.interp(np.exp(c.GAMMA*cols),C[h],R[h],left=np.nan)


def sequence_actions(states,name,log_price):
    kind,window=name.split('_w');wi=[1,4].index(int(window))
    if kind=='regression':k=states['regression_k'];gain=states['envg'][wi]
    else:
        k=states['conditional_k']
        gain=np.take_along_axis(states['gains'][wi],k[...,None],axis=3)[...,0]
    thresholds=np.where(gain>0,np.log(np.maximum(gain,1e-30))-states['logcost'],-np.inf)
    thresholds[:,:,-1]=-np.inf
    n=(thresholds<=log_price).argmax(2)
    qi=np.arange(len(n))[:,None];pi=np.arange(n.shape[1])[None,:]
    return n,k[qi,pi,n]


def usage(tag,name,d,pts,fixed_arms,raw,states):
    Q,S=raw['corr'].shape[:2];out=[]
    qi=np.arange(Q)[:,None];pi=np.arange(S)[None,:]
    for j,p in enumerate(pts):
        hist=np.zeros(13);modal=np.zeros(13);R=np.zeros(Q);C=np.zeros(Q);M=np.zeros(Q);alt=np.zeros(Q)
        differing=0.;min_path_unique=13;max_path_unique=0
        fixed_k=int(fixed_arms[j].split('_')[1])
        for endpoint,w in [(p['lo'],1-p['theta']),(p['hi'],p['theta'])]:
            if not w:continue
            if name.startswith('sequential'):
                n,k=sequence_actions(states,d['arm'],d['grid'][endpoint])
            else:
                n=np.broadcast_to(d['n'][endpoint,:,None]-1,(Q,S))
                k=np.broadcast_to(d['k'][endpoint,:,None],(Q,S))
            assert np.isfinite(n).all()
            hist+=w*np.bincount(k.ravel(),minlength=13)/(Q*S)
            modes=np.array([np.bincount(row,minlength=13).argmax() for row in k])
            modal+=w*np.bincount(modes,minlength=13)/Q
            unique=[len(np.unique(k[:,i])) for i in range(S)]
            min_path_unique=min(min_path_unique,min(unique));max_path_unique=max(max_path_unique,max(unique))
            rr=raw['corr'][qi,pi,n,k];aa=raw['corr'][qi,pi,n,fixed_k]
            R+=w*rr.mean(1);alt+=w*aa.mean(1);differing+=w*float((rr!=aa).mean())
            tokens=raw['tok'][qi,pi,n].astype(float)
            C+=w*np.exp(c.GAMMA*tokens).mean(1);M+=w*tokens.mean(1)
        for key,value in [('R',R),('C',C),('M',M)]:np.testing.assert_allclose(value,s.E21.mix(d[key],p),rtol=1e-11,atol=1e-10)
        out.append(dict(temperature_share=hist.tolist(),modal_temperature_share_across_queries=modal.tolist(),
            nonmodal_terminal_pct=float(100*(1-hist.max())),nonmodal_query_mode_pct=float(100*(1-modal.max())),
            unique_temperatures_at_same_path=[min_path_unique,max_path_unique],
            same_count_vote_gain_vs_fixed_pp=float(100*(R-alt).mean()),
            vote_correctness_diff_pct=100*differing,fixed_temperature_index=fixed_k))
    return out


def prospective(audit,test,cols):
    out=[]
    for b,p in zip(cols,points(audit,cols)):
        ix=[]
        for e in ['lo','hi']:
            grid_index=audit['grid_indices'][p[e]]
            ix.append(max(0,int(np.searchsorted(test['grid_indices'],grid_index,side='right')-1)))
        vals={key:(1-p['theta'])*test[key][ix[0]]+p['theta']*test[key][ix[1]] for key in ['R','C','M']}
        out.append(dict(requested_risk=float(b),actual_risk=float(np.log(vals['C'].mean())/c.GAMMA),
            accuracy_pct=float(100*vals['R'].mean()),mean_tokens=float(vals['M'].mean()),
            grid_indices=[int(audit['grid_indices'][p[e]]) for e in ['lo','hi']],theta=p['theta']))
    return out


def analyze(tag):
    root=c.HERE/tag;root.mkdir(exist_ok=True);cols=c.columns(tag)
    sel23=json.loads((c.E23/tag/'selection.json').read_text());sel24=json.loads((c.E24/tag/'selection.json').read_text())
    # Freeze policy class and existing calibration choices before new test readouts.
    tune_values={}
    for name in ['regression_w1','regression_w4']:
        td=load_seq(tag,'tune',name)
        tune_values[name]=float(np.mean([p['R'] for p in points(td,cols)]))
    seqname=max(tune_values,key=tune_values.get)
    spec=dict(sequential_primary=seqname,calibration_best_adaptive=sel23['adaptive_candidate'],
        committed_primary='joint',query_locked_rule='argmax mean predicted V over n=1..64',
        fixed_temperature_fallback_allowed=False,window_selection_uses_test=False,
        primary_family_chosen_after_temperature_usage_audit=True,window_tune_values=tune_values,
        source_models={str(c.E22/tag/'models.pkl'):c.sha(c.E22/tag/'models.pkl'),str(c.E24/tag/'models.pkl'):c.sha(c.E24/tag/'models.pkl')})
    c.dump(root/'deployment_config.json',spec)
    lock={phase:locked(tag,phase) for phase in ['tune','audit','test']}
    data={f'base_{n}':dict(np.load(c.E23/tag/'baselines'/(n+'.npz'))) for n in c.old.FAMILIES}
    for n in set(sel23['fixed_arms']):data['sf_'+n]=load_seq(tag,'test',n)
    for n in set(sel24['fixed_arms']):data['cf_'+n]=c.load_arm(tag,'test',n)
    data.update(sequential_adaptive=dict(load_seq(tag,'test',seqname),arm=seqname),
                sequential_calibration_best=dict(load_seq(tag,'test',sel23['adaptive_candidate']),arm=sel23['adaptive_candidate']),
                committed_joint=c.load_arm(tag,'test','joint'),committed_query=lock['test'])
    policies=['sequential_adaptive','sequential_calibration_best','committed_joint','committed_query']
    Q=len(data['base_sc']['qid']);pts={n:points(d,cols) for n,d in data.items()}
    rows={n:100*read_many(d['C'].mean(1),d['R'].mean(1),cols) for n,d in data.items()}
    best=np.stack([rows['base_'+n] for n in c.old.FAMILIES]).max(0)
    fixed_seq=np.array([rows['sf_'+n][j] for j,n in enumerate(sel23['fixed_arms'])])
    fixed_com=np.array([rows['cf_'+n][j] for j,n in enumerate(sel24['fixed_arms'])])
    weights=np.random.default_rng(20260924).multinomial(Q,np.full(Q,1/Q),800)/Q
    boot={}
    for name,d in data.items():
        rr=weights@d['R'].T;cc=weights@d['C'].T
        boot[name]=100*np.stack([read_many(cc[i],rr[i],cols) for i in range(len(weights))])
    bbase=np.nanmax(np.stack([boot['base_'+n] for n in c.old.FAMILIES]),axis=0)
    bfs=np.stack([boot['sf_'+n][:,j] for j,n in enumerate(sel23['fixed_arms'])],axis=1)
    bfc=np.stack([boot['cf_'+n][:,j] for j,n in enumerate(sel24['fixed_arms'])],axis=1)
    def interval(delta):
        assert np.isfinite(delta).all()
        return dict(lo=np.quantile(delta,.025,axis=0).tolist(),hi=np.quantile(delta,.975,axis=0).tolist())
    raw=np.load(c.E22/tag/'test.npz');states=dict(np.load(c.E22/tag/'test_states.npz'))
    result=dict(model=tag,queries=Q,paths=64,budgets=cols.tolist(),headers=c.s.D.C.table3_columns(tag)['mgf_hdr'],
        baseline_accuracy_pct={n:rows['base_'+n].tolist() for n in c.old.FAMILIES},
        best_baseline_accuracy_pct=best.tolist(),fixed_sequential_accuracy_pct=fixed_seq.tolist(),
        fixed_committed_accuracy_pct=fixed_com.tolist(),policy_config=spec,policies={},bootstrap_draws=800,
        temperature_grid=[float(x) if np.isfinite(x) else 'infinity' for x in raw['etas']])
    for name in policies:
        fixed=fixed_seq if name.startswith('sequential') else fixed_com
        bf=bfs if name.startswith('sequential') else bfc
        d=data[name]
        result['policies'][name]=dict(accuracy_pct=rows[name].tolist(),gain_vs_best_pp=(rows[name]-best).tolist(),
            gain_vs_best_ci=interval(boot[name]-bbase),gain_vs_fixed_pp=(rows[name]-fixed).tolist(),
            gain_vs_fixed_ci=interval(boot[name]-bf),
            actual_risk=[p['b_actual'] for p in pts[name]],
            mean_tokens=[float(s.E21.mix(d['M'],p).mean()) for p in pts[name]],
            usage=usage(tag,name,d,pts[name],sel23['fixed_arms'] if name.startswith('sequential') else sel24['fixed_arms'],raw,states))
    result['prospective']={
        'sequential_adaptive':prospective(load_seq(tag,'audit',seqname),data['sequential_adaptive'],cols),
        'committed_joint':prospective(c.load_arm(tag,'audit','joint'),data['committed_joint'],cols),
        'committed_query':prospective(lock['audit'],lock['test'],cols)}
    result['query_locked_temperature_counts']=np.bincount(lock['test']['k'][0],minlength=13).tolist()
    c.dump(root/'results.json',result);c.dump(root/'readout_points.json',pts)
    np.savez_compressed(root/'bootstrap.npz',**{n+'__vs_best_pp':boot[n]-bbase for n in policies},
                        **{n+'__vs_fixed_pp':boot[n]-(bfs if n.startswith('sequential') else bfc) for n in policies})
    for name in policies:print(tag,name,'accuracy',np.round(rows[name],3),'vs baseline',np.round(rows[name]-best,3),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True,choices=['qwen','llama']);analyze(p.parse_args().cell)
