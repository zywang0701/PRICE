import argparse,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import setup22 as s


def align(tag,phase,raw):
    w=s.D.C.CELLS[tag];cell=w['train'] if phase=='train' else w['cell']
    p=Path(s.D.C.EXP)/'outputs/cells'/cell/'pools/pool.parquet'
    df=pd.read_parquet(p,columns=['query_id','rollout_id','phi_conf','ell_tokens'])
    override=s.HERE/tag/('calibration_ptrue.parquet' if phase=='train' else 'evaluation_ptrue.parquet')
    if override.exists():
        z=pd.read_parquet(override)
        assert not z.duplicated(['query_id','rollout_id']).any()
        df=df.drop(columns='phi_conf').merge(z[['query_id','rollout_id','phi_conf']],on=['query_id','rollout_id'],validate='one_to_one')
    assert not df.duplicated(['query_id','rollout_id']).any()
    keys=pd.MultiIndex.from_arrays([np.repeat(raw['qid'],raw['rid'].shape[1]),raw['rid'].ravel()])
    a=df.set_index(['query_id','rollout_id']).loc[keys]
    np.testing.assert_array_equal(a.ell_tokens.to_numpy().reshape(raw['ell'].shape),raw['ell'])
    scores=a.phi_conf.to_numpy(float).reshape(raw['ell'].shape)
    assert np.isfinite(scores).all(),f'{tag} {phase}: missing P(True) scores'
    return scores


def features(phi,raw_phi,clu,ell,none,perms,etas):
    f,w,t=s.D.old.prefix_features(phi,clu,ell,none,perms,etas)
    rp=raw_phi[perms];n=np.arange(1,perms.shape[1]+1)
    mean=np.cumsum(rp,1)/n
    extra=np.stack([mean,np.sqrt(np.maximum(np.cumsum(rp**2,1)/n-mean**2,0)),
                    np.maximum.accumulate(rp,1),np.minimum.accumulate(rp,1)],axis=-1)[:,s.CP-1]
    return np.concatenate([f,extra],axis=-1).astype(np.float32),w,t


def prepare(tag,phase):
    out=s.HERE/tag;out.mkdir(exist_ok=True)
    train=s.D.load(tag,'train');sp=s.splits(len(train['qid']))
    # Keep repeated question texts in one calibration fold, even if IDs differ.
    meta=pd.read_parquet(Path(s.D.C.EXP)/'outputs/cells'/s.D.C.CELLS['llama']['train']/'pools/queries.parquet').set_index('query_id')
    texts=[' '.join(str(meta.loc[int(q),'question']).split()) for q in train['qid']]
    owner={int(i):name for name,idx in sp.items() for i in idx};seen={}
    for i,text in enumerate(texts):
        if text in seen:owner[i]=owner[seen[text]]
        else:seen[text]=i
    sp={name:np.array([i for i in range(len(texts)) if owner[i]==name]) for name in sp}
    if phase=='train':
        raw=train;scores=align(tag,phase,raw);cdf=np.sort(scores[sp['fit']].ravel())
        np.save(out/'cdf.npy',cdf)
        s.dump(out/'query_splits.json',{k:raw['qid'][v].tolist() for k,v in sp.items()})
        groups=sp
    else:
        assert (out/'selection.json').exists(),'Freeze calibration choices before building test labels'
        raw=s.D.load(tag,'test');scores=align(tag,phase,raw);cdf=np.load(out/'cdf.npy');groups={'test':np.arange(len(raw['qid']))}
    phi=np.searchsorted(cdf,scores,side='right')/(len(cdf)+1.)
    cell=s.D.C.CELLS[tag]['train' if phase=='train' else 'cell']
    H=s.D.old.embedding(cell,raw['qid'])
    for name,indices in groups.items():
        q=len(indices);paths=2 if name=='fit' else 64 if name=='test' else 16
        feat=np.empty((q,paths,8,123),np.float32);corr=np.empty((q,paths,64,13),bool);strict=np.empty_like(corr)
        tok=np.empty((q,paths,64),np.int32);ell_draw=np.empty_like(tok)
        V=np.empty((q,13,64),np.float32) if name!='test' else np.empty((0,13,64),np.float32)
        for j,i in enumerate(indices):
            rng=np.random.default_rng([s.SEED,int(raw['qid'][i]),1])
            if name=='test':
                if j==0:replay=dict(np.load(s.BASE/'E02_frozen'/tag/'replay.npz'));np.testing.assert_array_equal(replay['qid'],raw['qid'])
                perms=replay['perms'][i]
            else:perms=np.stack([rng.permutation(64) for _ in range(paths)])
            feat[j],wins,tok[j]=features(phi[i],scores[i],raw['clu'][i],raw['ell'][i],raw['none'][i],perms,raw['etas'])
            corr[j]=s.D.score_winners(wins,raw['clu'][i],raw['good'][i]);strict[j]=s.D.score_winners(wins,raw['clu'][i],raw['strict_good'][i])
            ell_draw[j]=raw['ell'][i][perms]
            if name!='test':
                ids=np.unique(raw['clu'][i]);cl=np.searchsorted(ids,raw['clu'][i]).astype(np.int64)
                none=int(np.searchsorted(ids,raw['none'][i])) if raw['none'][i] in ids else -1
                rng=np.random.default_rng([s.SEED,int(raw['qid'][i]),2]);pp=np.stack([rng.permutation(64) for _ in range(32)])
                values,_=s.reward_tables(phi[i],cl,none,raw['good'][i],raw['strict_good'][i],pp,raw['etas']);V[j]=values[1]
            if j%500==0:s.log(tag,name,'features',j,'/',q)
        np.savez_compressed(out/f'{name}.npz',feat=feat,corr=corr,strict=strict,tok=tok,ell=ell_draw,V=V,
            qid=raw['qid'][indices],H=H[indices],etas=raw['etas'],conflict=raw['conflict'][indices],
                            logM=logsumexp(s.GAMMA*raw['ell'][indices],axis=1)-np.log(raw['ell'].shape[1]))
        s.log(tag,name,'saved',feat.shape)
    if phase=='train':
        s.dump(out/'preparation.json',dict(fit_queries=len(sp['fit']),tune_queries=len(sp['tune']),audit_queries=len(sp['audit']),
            cdf_fit_rollouts=len(cdf),cdf_sha256=hashlib.sha256(cdf.tobytes()).hexdigest(),score='phi_conf',
            score_native_ties=True,source=str(Path(s.D.C.EXP)/'outputs/cells'/cell/'pools/pool.parquet')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--phase',choices=['train','test'],required=True)
    a=p.parse_args();prepare(a.cell,a.phase)
