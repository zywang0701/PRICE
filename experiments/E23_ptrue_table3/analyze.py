import argparse,json
import setup23 as c
from setup23 import np,s

FAMILIES=['sc','bon','cisc','ac','esc','dcoff']
NAMES={'sc':'Self-consistency','bon':'Best-of-n','cisc':'CISC','ac':'Adaptive-Consistency','esc':'ESC','dcoff':'DeepConf (offline)'}


def read_many(C,R,cols):
    h=s.E21.frontier(C,R);values=np.interp(np.exp(c.GAMMA*cols),C[h],R[h],left=np.nan)
    return values


def analyze(tag):
    root=c.HERE/tag;sel=json.loads((root/'selection.json').read_text());cols=c.columns(tag)
    data={n:dict(np.load(root/'baselines'/(n+'.npz'))) for n in FAMILIES}
    for n in set(sel['fixed_arms']+[sel['adaptive_candidate']]):data[n]=dict(np.load(c.E22/tag/'test'/(n+'.npz')))
    Q=len(data['sc']['qid']);rows={};points={};legacy={};matched={}
    for name,d in data.items():
        R=d['R'].mean(1);C=d['C'].mean(1);points[name]=[s.E21.read_point(C,R,b,c.GAMMA) for b in cols]
        assert all(p['status']!='below_minimum' for p in points[name]),name
        rows[name]=100*read_many(C,R,cols)
        order=np.argsort(C);legacy[name]=100*np.interp(cols,np.log(C[order])/c.GAMMA,np.maximum.accumulate(R[order]),left=np.nan)
        matched[name]={key:np.stack([s.E21.mix(d[key],p) for p in points[name]]) for key in ['R','C','M']}
    ours=np.array([rows[n][j] for j,n in enumerate(sel['fixed_arms'])]);base=np.stack([rows[n] for n in FAMILIES]);best=base.max(0)
    weights=np.random.default_rng(20260924).multinomial(Q,np.full(Q,1/Q),800)/Q
    br={}
    for name,d in data.items():
        r=weights@d['R'].T;cost=weights@d['C'].T
        br[name]=100*np.stack([read_many(cost[i],r[i],cols) for i in range(800)])
    new=np.stack([br[n][:,j] for j,n in enumerate(sel['fixed_arms'])],axis=1)
    baseboot=np.nanmax(np.stack([br[n] for n in FAMILIES]),axis=0);delta=new-baseboot
    assert np.isfinite(delta).all()
    output=dict(model=tag,queries=Q,paths=64,budgets=cols.tolist(),headers=s.D.C.table3_columns(tag)['mgf_hdr'],
        baseline_accuracy_pct={n:rows[n].tolist() for n in FAMILIES},price_accuracy_pct=ours.tolist(),
        adaptive_candidate_accuracy_pct=rows[sel['adaptive_candidate']].tolist(),
        best_baseline=[FAMILIES[i] for i in base.argmax(0)],best_baseline_accuracy_pct=best.tolist(),gain_pp=(ours-best).tolist(),
        gain_ci_lo_pp=np.quantile(delta,.025,axis=0).tolist(),gain_ci_hi_pp=np.quantile(delta,.975,axis=0).tolist(),
        p_gain_positive=(delta>0).mean(0).tolist(),nboot=800,frontiers_refit_in_bootstrap=True,
        price_mean_tokens=[float(matched[n]['M'][j].mean()) for j,n in enumerate(sel['fixed_arms'])],
        price_actual_risk=[float(np.log(matched[n]['C'][j].mean())/c.GAMMA) for j,n in enumerate(sel['fixed_arms'])],
        statuses=[points[n][j]['status'] for j,n in enumerate(sel['fixed_arms'])],selection=sel,
        legacy_risk_axis_interpolation_pct=dict(baselines={n:legacy[n].tolist() for n in FAMILIES},
            price=[float(legacy[n][j]) for j,n in enumerate(sel['fixed_arms'])]))
    for key in ['R','C','M']:matched['price__'+key]=np.stack([matched[n][key][j] for j,n in enumerate(sel['fixed_arms'])])
    flat={name+'__'+key:value for name,dd in matched.items() if isinstance(dd,dict) for key,value in dd.items()}
    flat.update({k:v for k,v in matched.items() if not isinstance(v,dict)})
    np.savez_compressed(root/'matched_queries.npz',**flat,qid=data['sc']['qid'],bootstrap_gain_pp=delta)
    c.dump(root/'readout_points.json',points);c.dump(root/'results.json',output)
    c.log(tag,'Table3 PRICE',ours,'best baseline',best,'gain',ours-best)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);analyze(p.parse_args().cell)
