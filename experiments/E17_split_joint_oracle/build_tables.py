import argparse
import hashlib
import numpy as np
from scipy.special import logsumexp
import settings as s
from allocation import finite_pool_logcost


def one(raw,i,pool_indices,seed):
    rng=np.random.default_rng(seed)
    perms=np.stack([pool_indices[rng.permutation(len(pool_indices))][:s.T] for _ in range(s.PERMUTATIONS)])
    _,wins,_=s.D.old.prefix_features(raw['phi'][i],raw['clu'][i],raw['ell'][i],raw['none'][i],perms,raw['etas'])
    good=s.D.score_winners(wins,raw['clu'][i],raw['good'][i])
    strict=s.D.score_winners(wins,raw['clu'][i],raw['strict_good'][i])
    ell=raw['ell'][i,pool_indices].astype(float)
    return dict(V=good.mean(0).T.astype(np.float32),V_strict=strict.mean(0).T.astype(np.float32),
        logC_exact=finite_pool_logcost(ell,s.GAMMA,s.T),
        logC_plugin=np.arange(1,s.T+1)*(logsumexp(s.GAMMA*ell)-np.log(len(ell))),mean_length=ell.mean())


def build(tag):
    out=s.HERE/tag;out.mkdir(exist_ok=True)
    source=s.D.HERE/tag/'test_data.npz';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    raw=s.D.load(tag,'test');Q=len(raw['qid']);K=len(raw['etas'])
    V=np.zeros((s.SPLITS,2,Q,K,s.T),np.float32);strict=np.zeros_like(V)
    exact=np.zeros((s.SPLITS,2,Q,s.T));plugin=np.zeros_like(exact);length=np.zeros((s.SPLITS,2,Q))
    halves=np.zeros((s.SPLITS,2,Q,64),np.int16)
    full=[]
    for i,qid in enumerate(raw['qid']):
        for part in range(s.SPLITS):
            rng=np.random.default_rng([s.SEED,int(qid),part]);p=rng.permutation(128)
            a,b=p[:64],p[64:]
            assert not set(raw['rid'][i,a])&set(raw['rid'][i,b])
            for side,half in enumerate([a,b]):
                z=one(raw,i,half,[s.SEED,int(qid),part,side,1])
                V[part,side,i]=z['V'];strict[part,side,i]=z['V_strict']
                exact[part,side,i]=z['logC_exact'];plugin[part,side,i]=z['logC_plugin'];length[part,side,i]=z['mean_length']
                halves[part,side,i]=half
        full.append(one(raw,i,np.arange(128),[s.SEED,int(qid),99]))
        if i%50==0:s.log(tag,'tables',i,'/',Q)
    np.savez_compressed(out/'tables.npz',V=V,V_strict=strict,logC_exact=exact,logC_plugin=plugin,
        mean_length=length,half_indices=halves,qid=raw['qid'],etas=raw['etas'],conflict=raw['conflict'])
    np.savez_compressed(out/'full_tables.npz',**{k:np.stack([r[k] for r in full]) for k in full[0]},qid=raw['qid'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    s.dump(out/'source.json',dict(path=str(source),sha256=digest,queries=Q,split_seed=s.SEED,
        splits=s.SPLITS,permutations=s.PERMUTATIONS,horizon=s.T,gamma=s.GAMMA,disjoint_ids_verified=True))
    s.log(tag,'tables saved')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);args=p.parse_args();build(args.cell)
