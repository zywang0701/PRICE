import json,pickle,hashlib
import numpy as np
from numba import njit
import setup22 as s
from models import infer

GRID=np.r_[-np.inf,np.linspace(-450,6,769)]


@njit(cache=True)
def sweep(gain,logcost,kmat,corr,tok,grid,gamma):
    Q,S,T=gain.shape;L=len(grid);R=np.zeros((L,Q));C=np.zeros_like(R);M=np.zeros_like(R);N=np.zeros_like(R)
    for q in range(Q):
        for path in range(S):
            previous=L
            for t in range(T):
                threshold=np.log(gain[q,path,t])-logcost[q,path,t] if gain[q,path,t]>0 and t<T-1 else -np.inf
                first=np.searchsorted(grid,threshold)
                if first<previous:
                    reward=1.0 if corr[q,path,t,kmat[q,path,t]] else 0.0
                    tokens=float(tok[q,path,t]);cost=np.exp(gamma*tokens)
                    for j in range(first,previous):
                        R[j,q]+=reward/S;C[j,q]+=cost/S;M[j,q]+=tokens/S;N[j,q]+=(t+1)/S
                    previous=first
                if previous==0:break
    return R,C,M,N


def arms(states):
    shape=states['regression_k'].shape
    for wi,window in enumerate([1,4]):
        yield f'regression_w{window}',states['envg'][wi],states['regression_k']
        k=states['conditional_k'];g=np.take_along_axis(states['gains'][wi],k[...,None],axis=3)[...,0]
        yield f'conditional_w{window}',g,k
        for k in range(13):yield f'fixed_{k}_w{window}',states['gains'][wi,...,k],np.full(shape,k,np.int8)


def read(R,C,M,N):
    pts=[]
    for b in s.BUDGETS:
        p=s.E21.read_point(C.mean(1),R.mean(1),b,s.GAMMA)
        assert p['status']!='below_minimum',f'Budget {b} below attainable initial-rollout cost'
        p.update(accuracy_pct=100*p['R'],mean_tokens=float(s.E21.mix(M,p).mean()),mean_n=float(s.E21.mix(N,p).mean()))
        pts.append(p)
    return pts


def evaluate_phase(tag,phase):
    root=s.HERE/tag;d=dict(np.load(root/f'{phase}.npz'))
    with (root/'models.pkl').open('rb') as f:bundle=pickle.load(f)
    states=infer(bundle,d);np.savez_compressed(root/f'{phase}_states.npz',**states)
    out=root/phase;out.mkdir(exist_ok=True);summary={}
    for name,g,k in arms(states):
        R,C,M,N=sweep(g,states['logcost'],k,d['corr'],d['tok'],GRID,s.GAMMA)
        assert np.isfinite(C).all()
        # Compact only duplicate output states; preserve raw grid indices.
        change=np.r_[True,np.any(R[1:]!=R[:-1],axis=1)|np.any(C[1:]!=C[:-1],axis=1)
                     |np.any(M[1:]!=M[:-1],axis=1)|np.any(N[1:]!=N[:-1],axis=1)]
        keep=np.flatnonzero(change);R,C,M,N=[v[keep] for v in [R,C,M,N]]
        np.savez_compressed(out/f'{name}.npz',R=R,C=C,M=M,N=N,grid=GRID[keep],grid_indices=keep,qid=d['qid'])
        summary[name]=read(R,C,M,N)
        s.log(tag,phase,name,[round(p['accuracy_pct'],2) for p in summary[name]])
    s.dump(out/'readouts.json',summary)
    if phase=='tune':
        fixed=[max([name for name in summary if name.startswith('fixed_')],key=lambda name:summary[name][j]['R']) for j in range(7)]
        adaptive=[name for name in summary if not name.startswith('fixed_')]
        best=max(adaptive,key=lambda name:np.mean([p['R'] for p in summary[name]]))
        fixed_avg=np.mean([summary[name][j]['R'] for j,name in enumerate(fixed)])
        adaptive_avg=np.mean([p['R'] for p in summary[best]])
        s.dump(root/'selection.json',dict(primary=best if adaptive_avg>fixed_avg else 'fixed_fallback',adaptive_candidate=best,
            fixed_arms_by_budget=fixed,tune_adaptive_mean_pct=100*adaptive_avg,tune_fixed_mean_pct=100*fixed_avg,
            tune_gain_pp=100*(adaptive_avg-fixed_avg),model_sha256=hashlib.sha256((root/'models.pkl').read_bytes()).hexdigest(),
            cdf_sha256=hashlib.sha256((root/'cdf.npy').read_bytes()).hexdigest(),budgets=s.BUDGETS.tolist(),
            all_adaptive_candidates=adaptive,selection_uses_MATH500=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--phase',choices=['tune','audit','test'],required=True)
    a=p.parse_args();evaluate_phase(a.cell,a.phase)
