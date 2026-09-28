import argparse,json,hashlib
from pathlib import Path
import numpy as np
import config as c
from price_sweep import actions,compact_indices


def run(tag):
    out=c.HERE/tag;tab=dict(np.load(c.E17/tag/'tables.npz'));cost=dict(np.load(out/'replay_cost.npz'))
    for p,digest in json.loads((out/'source.json').read_text()).items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest
    Q=len(tab['qid']);K=tab['V'].shape[3];L=len(c.LAM);qi=np.arange(Q)[None,:]
    replayC=np.exp(cost['logC']);sourceC=np.exp(tab['logC_plugin'])
    k0=c.D.k_fixed(tag)
    for fixed in [None,*range(K)]:
        name='joint' if fixed is None else f'fixed_{fixed}'
        sums={key:np.zeros((L,Q)) for key in ['R_A','R_B','C_A_replay','C_A_plugin','C_B','mean_B','strict_B']}
        if fixed is None:
            sums.update({key:np.zeros((L,Q)) for key in ['global_vote_A','global_vote_B','global_vote_strict_B',
                'cal_vote_B','cal_vote_strict_B']})
        ndirs=[];kdirs=[];global_ks=[];changed=np.zeros(L,bool);changed[0]=True
        for part in range(c.SPLITS):
            for side in [0,1]:
                V=tab['V'][part,side];W=tab['V'][part,1-side];Vs=tab['V_strict'][part,1-side]
                n,k=actions(V,tab['logC_plugin'][part,side],c.LAM,fixed);nn=n-1
                ndirs.append(n);kdirs.append(k)
                changed[1:]|=np.any(n[1:]!=n[:-1],axis=1)|np.any(k[1:]!=k[:-1],axis=1)
                for key,z in [('R_A',V[qi,k,nn]),('R_B',W[qi,k,nn]),('strict_B',Vs[qi,k,nn]),
                    ('C_A_replay',replayC[part,side][qi,nn]),('C_A_plugin',sourceC[part,side][qi,nn]),
                    ('C_B',replayC[part,1-side][qi,nn]),('mean_B',cost['mean_tokens'][part,1-side][qi,nn])]:
                    sums[key]+=z.astype(float)/(2*c.SPLITS)
                if fixed is None:
                    global_reward=np.stack([V[qi,kk,nn].mean(1,dtype=np.float64) for kk in range(K)])
                    kg=global_reward.argmax(0);global_ks.append(kg)
                    for key,z in [('global_vote_A',V[qi,kg[:,None],nn]),
                        ('global_vote_B',W[qi,kg[:,None],nn]),('global_vote_strict_B',Vs[qi,kg[:,None],nn]),
                        ('cal_vote_B',W[qi,k0,nn]),('cal_vote_strict_B',Vs[qi,k0,nn])]:
                        sums[key]+=z.astype(float)/(2*c.SPLITS)
        keep=compact_indices(changed)
        np.savez_compressed(out/f'{name}.npz',**{key:x[keep] for key,x in sums.items()},
            n=np.stack(ndirs)[:,keep],k=np.stack(kdirs)[:,keep],lambdas=c.LAM[keep],grid_indices=keep,
            global_k=np.array(global_ks)[:,keep] if fixed is None else np.empty((0,0),int),qid=tab['qid'])
        c.log(tag,name,'saved',len(keep),'distinct-state/coarse-grid representatives')
    c.dump(out/'run.json',dict(gamma=c.GAMMA,queries=Q,temperatures=K,k_fixed=k0,
        grid_points=len(c.LAM),grid_min_positive=float(c.LAM[1]),grid_max=float(c.LAM[-1]),
        splits=c.SPLITS,directions=2*c.SPLITS,permutations=c.S,budgets=c.BUDGETS.tolist()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);run(p.parse_args().cell)
