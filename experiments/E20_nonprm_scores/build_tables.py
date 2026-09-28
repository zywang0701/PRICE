import argparse,json,hashlib
from pathlib import Path
import numpy as np
import setup20 as s
from kernel import reward_tables


def build(tag,score):
    root=s.HERE/tag;out=root/score;out.mkdir(exist_ok=True)
    raw=s.D.load(tag,'test');tab=dict(np.load(s.E17/tag/'tables.npz'))
    src=np.load(root/'raw_scores.npz');phi_raw=src['values'][list(s.SCORES).index(score)]
    np.testing.assert_array_equal(src['qid'],raw['qid']);Q=len(raw['qid'])
    V=np.empty((2,4,2,Q,s.K,s.N),np.float32);strict=np.empty_like(V)
    summaries=[]
    for part in range(4):
        for side in [0,1]:
            source_indices=tab['half_indices'][part,side]
            cdf=np.sort(np.take_along_axis(phi_raw,source_indices,axis=1).reshape(-1))
            phi=np.searchsorted(cdf,phi_raw,side='right')/(len(cdf)+1.)
            for q,qid in enumerate(raw['qid']):
                ids=np.unique(raw['clu'][q]);cl=np.searchsorted(ids,raw['clu'][q]).astype(np.int64)
                no=int(np.searchsorted(ids,raw['none'][q])) if raw['none'][q] in ids else -1
                for target_side in [side,1-side]:
                    half=tab['half_indices'][part,target_side,q]
                    rng=np.random.default_rng([s.SEED,int(qid),part,target_side,1])
                    perms=np.stack([half[rng.permutation(64)] for _ in range(s.S)]).astype(np.int64)
                    # Save A and B with an explicit extra evaluation context: the
                    # transform is refitted for each source direction, never on B.
                    values,values_strict=reward_tables(phi[q],cl,no,
                        raw['good'][q],raw['strict_good'][q],perms,raw['etas'])
                    if target_side==side:
                        V[:,part,side,q]=values;strict[:,part,side,q]=values_strict
                    else:
                        if q==0:
                            cross=np.empty((2,Q,s.K,s.N),np.float32);cross_strict=np.empty_like(cross)
                        cross[:,q]=values;cross_strict[:,q]=values_strict
            np.savez_compressed(out/f'B_{part}_{side}.npz',V=cross,V_strict=cross_strict)
            summaries.append(dict(part=part,source_side=side,cdf_observations=len(cdf),
                distinct_source_values=int(len(np.unique(cdf))),cdf_sha256=hashlib.sha256(cdf.tobytes()).hexdigest()))
            s.log(tag,score,'direction',part,side,'done')
    np.savez_compressed(out/'tables.npz',V_A=V,V_strict_A=strict,
        V_B=np.stack([np.stack([np.load(out/f'B_{p}_{d}.npz')['V'] for d in [0,1]],axis=1) for p in range(4)],axis=1),
        V_strict_B=np.stack([np.stack([np.load(out/f'B_{p}_{d}.npz')['V_strict'] for d in [0,1]],axis=1) for p in range(4)],axis=1),
        qid=raw['qid'],etas=raw['etas'],conflict=raw['conflict'])
    s.dump(out/'transforms.json',summaries)
    s.log(tag,score,'all tables saved')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--score',choices=list(s.SCORES),required=True)
    a=p.parse_args();build(a.cell,a.score)
