"""Post-evaluation stability diagnostic; never used for model selection."""
import argparse
import hashlib
import json
import numpy as np
import data as d
from split_pool import closest_best,bootstrap


def run(tag):
    root=d.HERE/tag
    assert (root/'results.json').exists(),'Run only after frozen evaluation is complete'
    digest=hashlib.sha256((root/'selection.json').read_bytes()).hexdigest()
    raw=d.load(tag,'test');q=len(raw['qid']);assert raw['phi'].shape[1]==128
    sel=json.loads((root/'selection.json').read_text());k0=sel['k_fixed']
    u=np.empty((q,4,2,13),np.float32)
    for i in range(q):
        for part in range(4):
            rng=np.random.default_rng([d.SEED,int(raw['qid'][i]),30,part]);p=rng.permutation(128)
            halves=[p[:64],p[64:]]
            assert not set(raw['rid'][i,halves[0]])&set(raw['rid'][i,halves[1]])
            for side,half in enumerate(halves):
                perms=np.stack([half[rng.permutation(64)] for _ in range(16)])
                w=d.votes(raw['phi'][i],raw['clu'][i],raw['none'][i],raw['etas'],perms)
                c=d.score_winners(w,raw['clu'][i],raw['good'][i]);u[i,part,side]=c[:,d.EVAL-1].mean((0,1))
    same=[];transfer=[]
    for side in [0,1]:
        k=closest_best(u[:,:,side],k0)
        same.append(np.take_along_axis(u[:,:,side],k[...,None],-1)[...,0]-u[:,:,side,k0])
        transfer.append(np.take_along_axis(u[:,:,1-side],k[...,None],-1)[...,0]-u[:,:,1-side,k0])
    same=np.stack(same,-1).mean((1,2));transfer=np.stack(transfer,-1).mean((1,2))
    clean=~raw['conflict']
    summary=dict(queries=q,half_size=64,partitions=4,permutations_per_half=16,counts=d.EVAL.tolist(),
        same_half_gain=bootstrap(same),cross_half_gain=bootstrap(transfer),
        without_original_label_conflicts=dict(queries=int(clean.sum()),same_half_gain=bootstrap(same[clean]),
            cross_half_gain=bootstrap(transfer[clean])),
        status='Post-test explanatory diagnostic; not used for training or selection. Uses existing 128-rollout pool.',
        selection_hash=digest)
    assert hashlib.sha256((root/'selection.json').read_bytes()).hexdigest()==digest
    d.dump(root/'test_transfer_summary.json',summary)
    np.savez_compressed(root/'test_transfer_utilities.npz',utilities=u,qid=raw['qid'],same_gain=same,transfer_gain=transfer)
    d.log(tag,summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);args=p.parse_args();run(args.cell)
