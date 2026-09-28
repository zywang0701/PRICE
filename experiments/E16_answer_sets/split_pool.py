"""Disjoint rollout-ID transfer diagnostic and future-pool supervision."""
import argparse
import json
import numpy as np
import data as d
import features as f


def bootstrap(x,seed=d.SEED,nboot=1000):
    x=np.asarray(x);rng=np.random.default_rng(seed)
    vals=np.stack([x[rng.integers(len(x),size=len(x))].mean(0) for _ in range(nboot)])
    return dict(mean_pp=(100*x.mean(0)).tolist(),lo_pp=(100*np.quantile(vals,.025,axis=0)).tolist(),
        hi_pp=(100*np.quantile(vals,.975,axis=0)).tolist())


def closest_best(u,k0):
    priority=np.argsort(np.abs(np.arange(u.shape[-1])-k0),kind='stable')
    return priority[np.argmax(u[...,priority],axis=-1)]


def run(tag):
    raw=d.load(tag,'train');q=len(raw['qid']);etas=raw['etas']
    k0=json.loads((d.HERE.parent/'E13_query_tau'/tag/'selection.json').read_text())['k_fixed']
    counts=np.array([2,4,8,16,32]);stages=d.CP[d.CP<=32]
    utilities=np.empty((q,4,2,13),np.float32)
    records=[];targets=[];state_q=[];state_stage=[];disjoint=True
    for i in range(q):
        for part in range(4):
            rng=np.random.default_rng([d.SEED,int(raw['qid'][i]),20,part])
            p=rng.permutation(64);halves=[p[:32],p[32:]]
            assert not set(raw['rid'][i,halves[0]]) & set(raw['rid'][i,halves[1]])
            paths=[];us=[]
            for half in halves:
                permutations=np.stack([half[rng.permutation(32)] for _ in range(16)])
                w=d.votes(raw['phi'][i],raw['clu'][i],raw['none'][i],etas,permutations)
                c=d.score_winners(w,raw['clu'][i],raw['good'][i])
                us.append(c[:,counts-1].mean((0,1)));paths.append(permutations[0])
            utilities[i,part]=us
            if part<2:
                for direction in range(2):
                    for j,n in enumerate(stages):
                        p=paths[direction][:n]
                        records.append(f.state(raw['phi'][i,p],raw['clu'][i,p],raw['ell'][i,p],raw['none'][i],etas))
                        targets.append(us[1-direction]);state_q.append(i);state_stage.append(j)
        if i%500==0:d.log(tag,'split-pool',i,'/',q)
    same=[];transfer=[]
    for side in [0,1]:
        ka=closest_best(utilities[:,:,side],k0)
        same.append(np.take_along_axis(utilities[:,:,side],ka[...,None],axis=-1)[...,0]-utilities[:,:,side,k0])
        transfer.append(np.take_along_axis(utilities[:,:,1-side],ka[...,None],axis=-1)[...,0]-utilities[:,:,1-side,k0])
    same=np.stack(same,axis=-1).mean((1,2));transfer=np.stack(transfer,axis=-1).mean((1,2))
    summary=dict(queries=q,half_size=32,partitions=4,permutations_per_half=16,counts=counts.tolist(),
        fixed_temperature=float(etas[k0]),same_half_gain=bootstrap(same),cross_half_gain=bootstrap(transfer),
        optimism_gap=bootstrap(same-transfer),disjoint_ids_verified=True,
        caution='Disjoint partitions of the same finite pool, not fresh independent generation; no test data used.')
    clean=~raw['conflict']
    summary['without_original_label_conflicts']=dict(queries=int(clean.sum()),same_half_gain=bootstrap(same[clean]),
        cross_half_gain=bootstrap(transfer[clean]))
    d.dump(d.HERE/tag/'split_pool_summary.json',summary)
    np.savez_compressed(d.HERE/tag/'split_pool_utilities.npz',utilities=utilities,same_gain=same,transfer_gain=transfer,qid=raw['qid'])
    z=f.pack(records);z.update(utility=np.array(targets,np.float32),state_q=np.array(state_q),
        state_stage=np.array(state_stage),qid=raw['qid'],etas=etas)
    np.savez_compressed(d.HERE/tag/'cross_features.npz',**z)
    d.log(tag,'split-pool ready',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);args=p.parse_args();run(args.cell)
