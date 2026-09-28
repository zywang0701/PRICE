import argparse,json
import setup24 as c
from setup24 import np,s
from run import load_arm


def read_many(C,R,cols):
    h=s.E21.frontier(C,R)
    return np.interp(np.exp(c.GAMMA*cols),C[h],R[h],left=np.nan)


def matched(d,cols):
    pts=[s.E21.read_point(d['C'].mean(1),d['R'].mean(1),b,c.GAMMA) for b in cols]
    assert all(p['status']!='below_minimum' for p in pts)
    return {key:np.stack([s.E21.mix(d[key],p) for p in pts]) for key in ['R','C','M']},pts


def prospective_committed(tag,sel,cols):
    outputs={key:[] for key in ['R','C','M']};policies=[]
    for j,name in enumerate(sel['primary_arms']):
        audit=load_arm(tag,'audit',name);test=load_arm(tag,'test',name)
        p=s.E21.read_point(audit['C'].mean(1),audit['R'].mean(1),cols[j],c.GAMMA)
        indices=[int(audit['grid_indices'][p[e]]) for e in ['lo','hi']]
        at=[max(0,int(np.searchsorted(test['grid_indices'],i,side='right')-1)) for i in indices]
        for key in outputs:outputs[key].append((1-p['theta'])*test[key][at[0]]+p['theta']*test[key][at[1]])
        policies.append(dict(arm=name,grid_indices=indices,theta=p['theta']))
    return {key:np.stack(v) for key,v in outputs.items()},policies


def analyze(tag):
    root=c.HERE/tag;sel=json.loads((root/'selection.json').read_text());seqsel=json.loads((c.E23/tag/'selection.json').read_text());cols=c.columns(tag)
    names=sorted(set(sel['primary_arms']+sel['fixed_arms']+['joint']))
    data={n:load_arm(tag,'test',n) for n in names}
    for family in c.FAMILIES:data['base_'+family]=dict(np.load(c.E23/tag/'baselines'/(family+'.npz')))
    for name in set(seqsel['fixed_arms']):data['seq_'+name]=dict(np.load(c.E22/tag/'test'/(name+'.npz')))
    m={};points={}
    for name,d in data.items():m[name],points[name]=matched(d,cols)
    def selected(names):return {key:np.stack([m[n][key][j] for j,n in enumerate(names)]) for key in ['R','C','M']}
    committed=selected(sel['primary_arms']);seq=selected(['seq_'+n for n in seqsel['fixed_arms']]);fixed=selected(sel['fixed_arms'])
    base=np.stack([m['base_'+n]['R'].mean(1) for n in c.FAMILIES]);Q=committed['R'].shape[1]
    weights=np.random.default_rng(20260924).multinomial(Q,np.full(Q,1/Q),800)/Q;br={}
    for name,d in data.items():
        R=weights@d['R'].T;C=weights@d['C'].T
        br[name]=np.stack([read_many(C[i],R[i],cols) for i in range(800)])
    bc=np.stack([br[n][:,j] for j,n in enumerate(sel['primary_arms'])],axis=1)
    bs=np.stack([br['seq_'+n][:,j] for j,n in enumerate(seqsel['fixed_arms'])],axis=1)
    bb=np.nanmax(np.stack([br['base_'+n] for n in c.FAMILIES]),axis=0)
    def ci(delta):
        delta=100*delta;assert np.isfinite(delta).all()
        return dict(lo=np.quantile(delta,.025,axis=0).tolist(),hi=np.quantile(delta,.975,axis=0).tolist(),p_positive=(delta>0).mean(0).tolist())
    coarse={}
    for name in names:
        d=data[name];keep=(d['grid_indices']==0)|(d['grid_indices']%2==1)
        coarse[name]=(100*(read_many(d['C'].mean(1),d['R'].mean(1),cols)-read_many(d['C'][keep].mean(1),d['R'][keep].mean(1),cols))).tolist()
    pros,pros_policies=prospective_committed(tag,sel,cols)
    terminal={}
    for j,name in enumerate(sel['primary_arms']):
        d=data[name];p=points[name][j]
        terminal[str(j)]=dict(mean_n=float(s.E21.mix(d['n'],p).mean()),n_range=[int(min(d['n'][p['lo']].min(),d['n'][p['hi']].min())),int(max(d['n'][p['lo']].max(),d['n'][p['hi']].max()))],
            temperature_histogram=((1-p['theta'])*np.bincount(d['k'][p['lo']],minlength=13)/Q+p['theta']*np.bincount(d['k'][p['hi']],minlength=13)/Q).tolist())
    result=dict(model=tag,queries=Q,paths=64,budgets=cols.tolist(),headers=s.D.C.table3_columns(tag)['mgf_hdr'],
        committed_accuracy_pct=(100*committed['R'].mean(1)).tolist(),sequential_accuracy_pct=(100*seq['R'].mean(1)).tolist(),
        best_baseline_accuracy_pct=(100*base.max(0)).tolist(),baseline_accuracy_pct={n:(100*m['base_'+n]['R'].mean(1)).tolist() for n in c.FAMILIES},
        best_baseline=[c.FAMILIES[i] for i in base.argmax(0)],
        committed_minus_best_pp=(100*(committed['R'].mean(1)-base.max(0))).tolist(),committed_minus_best_ci=ci(bc-bb),
        committed_minus_sequential_pp=(100*(committed['R'].mean(1)-seq['R'].mean(1))).tolist(),committed_minus_sequential_ci=ci(bc-bs),
        committed_joint_accuracy_pct=(100*m['joint']['R'].mean(1)).tolist(),committed_fixed_accuracy_pct=(100*fixed['R'].mean(1)).tolist(),
        committed_actual_risk=(np.log(committed['C'].mean(1))/c.GAMMA).tolist(),committed_mean_tokens=committed['M'].mean(1).tolist(),
        sequential_mean_tokens=seq['M'].mean(1).tolist(),terminal_actions=terminal,
        half_grid_change_pp=coarse,prospective=dict(accuracy_pct=(100*pros['R'].mean(1)).tolist(),
            actual_risk=(np.log(pros['C'].mean(1))/c.GAMMA).tolist(),mean_tokens=pros['M'].mean(1).tolist(),policies=pros_policies),
        selection=sel,training=json.loads((root/'training.json').read_text()),bootstrap_draws=800,frontiers_refit=True)
    c.dump(root/'results.json',result);c.dump(root/'readout_points.json',points)
    arrays={name+'__'+key:value for name,dd in m.items() for key,value in dd.items()}
    for prefix,dd in [('committed',committed),('sequential',seq),('prospective',pros)]:arrays.update({prefix+'__'+key:value for key,value in dd.items()})
    np.savez_compressed(root/'matched_queries.npz',**arrays,bootstrap_vs_best_pp=100*(bc-bb),bootstrap_vs_sequential_pp=100*(bc-bs))
    c.log(tag,'committed',result['committed_accuracy_pct'],'gain over best',result['committed_minus_best_pp'],'CI',result['committed_minus_best_ci'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);analyze(p.parse_args().cell)
