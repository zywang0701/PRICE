"""Actual path costs on precisely the reward-table permutations cached by E17."""
import argparse,hashlib
import numpy as np
from scipy.special import logsumexp
import config as c

def build(tag):
    out=c.HERE/tag;out.mkdir(exist_ok=True)
    tab=dict(np.load(c.E17/tag/'tables.npz'));raw=c.D.load(tag,'test');Q=len(raw['qid'])
    logC=np.empty((c.SPLITS,2,Q,c.T));mean=np.empty_like(logC)
    for part in range(c.SPLITS):
        for side in [0,1]:
            for i,qid in enumerate(raw['qid']):
                half=tab['half_indices'][part,side,i]
                rng=np.random.default_rng([c.SEED,int(qid),part,side,1])
                perms=np.stack([half[rng.permutation(len(half))][:c.T] for _ in range(c.S)])
                tok=np.cumsum(raw['ell'][i,perms].astype(float),axis=1)
                assert np.all(tok[:,-1]==raw['ell'][i,half].sum())
                logC[part,side,i]=logsumexp(c.GAMMA*tok,axis=0)-np.log(c.S)
                mean[part,side,i]=tok.mean(0)
            c.log(tag,'actual prefix costs',part,side)
    np.savez_compressed(out/'replay_cost.npz',logC=logC,mean_tokens=mean,qid=raw['qid'])
    c.dump(out/'source.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [c.E17/tag/'tables.npz',c.D.HERE/tag/'test_data.npz']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);build(p.parse_args().cell)
