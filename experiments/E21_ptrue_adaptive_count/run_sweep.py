import argparse,hashlib
import numpy as np
import setup21 as s


def run(tag,mode,points):
    out=s.HERE/tag/mode;out.mkdir(parents=True,exist_ok=True)
    t=s.load(tag,mode);Q=len(t['qid']);K=t['A'].shape[3]
    lambdas=np.r_[0.,np.geomspace(1e-180,1e6,points)];L=len(lambdas);qi=np.arange(Q)[None,:]
    actual=np.exp(t['logC_B']);source=np.exp(t['logC_A'])
    for fixed in [None,*range(K)]:
        name='joint' if fixed is None else f'fixed_{fixed}'
        sums={key:np.zeros((L,Q)) for key in ['R_A','R_B','C_B','mean_B','strict_B','mean_n']}
        if fixed is None:sums.update({key:np.zeros((L,Q)) for key in ['global_vote_A','global_vote_B','global_strict_B','sc_vote_B']})
        ns=[];ks=[];gks=[];changed=np.zeros(L,bool);changed[0]=True
        for part in range(4):
            for side in [0,1]:
                V=t['A'][part,side];W=t['B'][part,side];Ws=t['strict'][part,side]
                n,k=s.actions(V,t['logC_A'][part,side],lambdas,fixed);nn=n-1
                ns.append(n);ks.append(k)
                changed[1:]|=np.any(n[1:]!=n[:-1],axis=1)|np.any(k[1:]!=k[:-1],axis=1)
                for key,z in [('R_A',V[qi,k,nn]),('R_B',W[qi,k,nn]),('strict_B',Ws[qi,k,nn]),
                    ('C_B',actual[part,1-side][qi,nn]),('mean_B',t['mean'][part,1-side][qi,nn]),('mean_n',n)]:
                    sums[key]+=z.astype(float)/8
                if fixed is None:
                    kg=np.stack([V[qi,kk,nn].mean(1,dtype=float) for kk in range(K)]).argmax(0);gks.append(kg)
                    for key,z in [('global_vote_A',V[qi,kg[:,None],nn]),('global_vote_B',W[qi,kg[:,None],nn]),
                        ('global_strict_B',Ws[qi,kg[:,None],nn]),('sc_vote_B',W[qi,0,nn])]:sums[key]+=z.astype(float)/8
        keep=s.compact_indices(changed)
        np.savez_compressed(out/f'{name}.npz',**{key:x[keep] for key,x in sums.items()},
            n=np.stack(ns)[:,keep],k=np.stack(ks)[:,keep],lambdas=lambdas[keep],grid_indices=keep,
            global_k=np.stack(gks)[:,keep] if fixed is None else np.empty((0,0),int),qid=t['qid'])
        s.log(tag,mode,name,'saved',len(keep),'states')
    sources=[s.E20/tag/'ptrue/tables.npz',s.E20/tag/'ptrue/transforms.json',s.E17/tag/'tables.npz',s.E18/tag/'replay_cost.npz']
    s.dump(out/'source.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    s.dump(out/'run.json',dict(model=tag,tie_mode=mode,queries=Q,temperatures=K,
        etas=[float(x) if np.isfinite(x) else 'infinity' for x in t['etas']],
        gamma=s.GAMMA,grid_points=L,budgets=s.BUDGETS.tolist()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--mode',choices=s.MODES,required=True)
    p.add_argument('--points',type=int,default=16385);a=p.parse_args();run(a.cell,a.mode,a.points)
