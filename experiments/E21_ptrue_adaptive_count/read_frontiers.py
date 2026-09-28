import argparse,json
import numpy as np
import setup21 as s


def run(tag,mode):
    out=s.HERE/tag/mode;meta=json.loads((out/'run.json').read_text());t=s.load(tag,mode)
    Q=meta['queries'];K=meta['temperatures'];qi=np.arange(Q);weights=np.random.default_rng(s.SEED).multinomial(Q,np.full(Q,1/Q),size=1000)/Q
    readouts={};curves={};queries={};max_grid=0.
    for name in ['joint',*[f'fixed_{k}' for k in range(K)]]:
        d=dict(np.load(out/f'{name}.npz'));R=d['R_B'].mean(1);C=d['C_B'].mean(1)
        coarse=np.flatnonzero((d['grid_indices']==0)|(d['grid_indices']%2==1))
        pts=[s.read_point(C,R,b,s.GAMMA) for b in s.BUDGETS]
        cp=[s.read_point(C,R,b,s.GAMMA,coarse) for b in s.BUDGETS]
        assert all(p['status']!='below_minimum' for p in pts)
        for p,b in zip(pts,s.BUDGETS):
            assert p['b_actual']<=b+1e-7
            if p['status']=='exact':assert abs(p['b_actual']-b)<1e-7
            p.update(lambda_low=float(d['lambdas'][p['lo']]),lambda_high=float(d['lambdas'][p['hi']]),
                     mean_tokens=float(s.mix(d['mean_B'],p).mean()),mean_n=float(s.mix(d['mean_n'],p).mean()),
                     accuracy_A_pct=float(100*s.mix(d['R_A'],p).mean()),accuracy_B_pct=float(100*p['R']))
        grid=max(abs(100*(p['R']-z['R'])) for p,z in zip(pts,cp));max_grid=max(max_grid,grid)
        readouts[name]=pts
        curves[name]=dict(B_risk=(np.log(C)/s.GAMMA).tolist(),B_accuracy=(100*R).tolist(),
            B_mean_tokens=d['mean_B'].mean(1).tolist(),frontier_indices=s.frontier(C,R).tolist(),
            half_grid_max_accuracy_difference_pp=grid)
        keys=['R_A','R_B','C_B','mean_B','strict_B','mean_n']
        if name=='joint':keys+=['global_vote_A','global_vote_B','global_strict_B','sc_vote_B']
        queries[name]={key:np.stack([s.mix(d[key],p) for p in pts]) for key in keys}
        if name=='joint':
            all_global=[]
            for p in pts:
                v=np.zeros((Q,K))
                for endpoint,w in [(p['lo'],1-p['theta']),(p['hi'],p['theta'])]:
                    for direction in range(8):
                        part,side=divmod(direction,2);n=d['n'][direction,endpoint].astype(int)-1
                        v+=w*t['B'][part,side][qi[:,None],np.arange(K)[None,:],n[:,None]]/8
                all_global.append(v)
            all_global=np.stack(all_global);kg=all_global.mean(1).argmax(1)
            queries[name]['B_global_same_count']=all_global[np.arange(len(pts)),:,kg]
            same_count_best_k=kg.tolist()
    best_k=[int(np.argmax([readouts[f'fixed_{k}'][j]['R'] for k in range(K)])) for j in range(len(s.BUDGETS))]
    best=[dict(readouts[f'fixed_{k}'][j],temperature_index=k,temperature=meta['etas'][k]) for j,k in enumerate(best_k)]
    bestq={key:np.stack([queries[f'fixed_{k}'][key][j] for j,k in enumerate(best_k)]) for key in ['R_A','R_B','C_B','mean_B','strict_B','mean_n']}
    def interval(d):
        boot=weights@d.T
        return dict(mean=d.mean(1).tolist(),lo=np.quantile(boot,.025,axis=0).tolist(),hi=np.quantile(boot,.975,axis=0).tolist())
    j=queries['joint'];contrasts={}
    for name,other in [('best_fixed',bestq['R_B']),('sc',queries['fixed_0']['R_B']),('same_count_A_global',j['global_vote_B']),
                       ('same_count_B_global',j['B_global_same_count']),('same_count_sc',j['sc_vote_B'])]:
        contrasts[name]=interval(100*(j['R_B']-other))
    result=dict(**meta,methods=dict(joint=readouts['joint'],best_fixed_hindsight=best,sc=readouts['fixed_0'],bon=readouts[f'fixed_{K-1}']),
        all_fixed_readouts=readouts,contrasts=contrasts,max_half_grid_difference_pp=max_grid,
        same_count_B_global_k=same_count_best_k,
        strict_joint_minus_best_fixed_pp=(100*(j['strict_B']-bestq['strict_B']).mean(1)).tolist(),
        conflict_free_joint_minus_best_fixed_pp=(100*(j['R_B']-bestq['R_B'])[:,~t['conflict']].mean(1)).tolist(),
        bootstrap='1000 paired query resamples, policies/global tau/frontier endpoints/mixing weights frozen')
    s.dump(out/'results.json',result);s.dump(out/'curves.json',curves)
    np.savez_compressed(out/'matched_queries.npz',**{'joint__'+k:v for k,v in j.items()},
                        **{'fixed__'+k:v for k,v in bestq.items()},qid=t['qid'])
    s.log(tag,mode,'joint',[round(p['accuracy_B_pct'],3) for p in readouts['joint']],
          'gain',[round(x,3) for x in contrasts['best_fixed']['mean']],
          'same-count',[round(x,3) for x in contrasts['same_count_A_global']['mean']], 'grid',max_grid)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--mode',choices=s.MODES,required=True)
    a=p.parse_args();run(a.cell,a.mode)
