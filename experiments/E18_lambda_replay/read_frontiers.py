"""Read all methods at matched B realized risk costs; retain operating policies."""
import argparse,json
import config as c
import numpy as np
from price_sweep import frontier,read_point,mix


def run(tag):
    out=c.HERE/tag;meta=json.loads((out/'run.json').read_text());K=meta['temperatures'];Q=meta['queries']
    weights=np.random.default_rng(c.SEED).multinomial(Q,np.full(Q,1/Q),size=1000)/Q
    curves={};readouts={};coarse_readouts={};joint_query={};max_grid=0.
    for name in ['joint',*[f'fixed_{k}' for k in range(K)]]:
        d=dict(np.load(out/f'{name}.npz'));R=d['R_B'].mean(1);C=d['C_B'].mean(1)
        coarse=np.flatnonzero((d['grid_indices']==0)|(d['grid_indices']%2==1))
        pts=[read_point(C,R,b,c.GAMMA) for b in c.BUDGETS]
        cp=[read_point(C,R,b,c.GAMMA,coarse) for b in c.BUDGETS]
        assert all(p['status']!='below_minimum' for p in pts)
        for p,b in zip(pts,c.BUDGETS):
            assert p['b_actual']<=b+1e-7
            if p['status']=='exact':assert abs(p['b_actual']-b)<1e-7
            p.update(lambda_low=float(d['lambdas'][p['lo']]),lambda_high=float(d['lambdas'][p['hi']]),
                mean_tokens=float(mix(d['mean_B'],p).mean()),accuracy_A_pct=float(100*mix(d['R_A'],p).mean()),
                accuracy_B_pct=float(100*p['R']))
        grid=max(abs(100*(p['R']-z['R'])) for p,z in zip(pts,cp));max_grid=max(max_grid,grid)
        readouts[name]=pts;coarse_readouts[name]=cp
        curves[name]=dict(lambdas=d['lambdas'].tolist(),grid_indices=d['grid_indices'].tolist(),
            B_risk=(np.log(C)/c.GAMMA).tolist(),B_accuracy=(100*R).tolist(),
            A_risk_replay=(np.log(d['C_A_replay'].mean(1))/c.GAMMA).tolist(),
            A_accuracy=(100*d['R_A'].mean(1)).tolist(),B_mean_tokens=d['mean_B'].mean(1).tolist(),
            frontier_indices=frontier(C,R).tolist(),
            A_frontier_indices=frontier(d['C_A_replay'].mean(1),d['R_A'].mean(1)).tolist(),
            half_grid_max_accuracy_difference_pp=grid)
        if name=='joint':
            joint_query={key:np.stack([mix(d[key],p) for p in pts]) for key in
                ['R_A','R_B','C_B','mean_B','strict_B','global_vote_A','global_vote_B',
                 'global_vote_strict_B','cal_vote_B','cal_vote_strict_B']}
    best_ks=[int(np.argmax([readouts[f'fixed_{k}'][j]['R'] for k in range(K)])) for j in range(len(c.BUDGETS))]
    best=[dict(readouts[f'fixed_{k}'][j],temperature_index=k) for j,k in enumerate(best_ks)]
    methods=dict(sc=readouts['fixed_0'],bon=readouts[f'fixed_{K-1}'],
        calibration_fixed=readouts[f'fixed_{meta["k_fixed"]}'],best_fixed_hindsight=best,joint=readouts['joint'])
    vote={}
    for name,key in [('A_global','global_vote_B'),('calibration','cal_vote_B')]:
        delta=100*(joint_query['R_B']-joint_query[key]);boot=weights@delta.T
        vote[name]=dict(mean=delta.mean(1).tolist(),lo=np.quantile(boot,.025,axis=0).tolist(),
            hi=np.quantile(boot,.975,axis=0).tolist())
    vote['in_sample_A_global']=(100*(joint_query['R_A']-joint_query['global_vote_A']).mean(1)).tolist()
    vote['strict_A_global']=(100*(joint_query['strict_B']-joint_query['global_vote_strict_B']).mean(1)).tolist()
    raw=c.D.load(tag,'test');clean=~raw['conflict']
    vote['clean_A_global']=(100*(joint_query['R_B']-joint_query['global_vote_B'])[:,clean].mean(1)).tolist()
    best_coarse=[max(coarse_readouts[f'fixed_{k}'][j]['R'] for k in range(K)) for j in range(len(c.BUDGETS))]
    best_grid=max(abs(100*(p['R']-z)) for p,z in zip(best,best_coarse))
    c.dump(out/'results.json',dict(**meta,methods=methods,equal_count_vote=vote,
        joint_minus_best_fixed_pp=[100*(a['R']-b['R']) for a,b in zip(methods['joint'],best)],
        max_half_grid_difference_pp=max_grid,best_fixed_half_grid_difference_pp=best_grid,
        uncertainty='paired query bootstrap conditional on A-fitted policies and selected B operating points',
        bootstrap_replicates=1000))
    c.dump(out/'curves.json',curves);np.savez_compressed(out/'matched_joint_queries.npz',**joint_query,budgets=c.BUDGETS)
    c.log(tag,'joint B accuracy',[round(p['accuracy_B_pct'],3) for p in methods['joint']],
        'best fixed',[round(p['accuracy_B_pct'],3) for p in best],
        'same-count voting',[round(x,3) for x in vote['A_global']['mean']],
        'half-grid max difference pp',max_grid)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);run(p.parse_args().cell)
