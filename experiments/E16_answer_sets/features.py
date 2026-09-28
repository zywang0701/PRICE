"""Causal answer-set features. No reference answers or correctness inputs."""
import argparse
import numpy as np
import data as d


def state(phi,clu,ell,none,etas):
    phi=np.asarray(phi,float);clu=np.asarray(clu);n=len(phi)
    valid=clu!=none;ids=np.unique(clu[valid]);nv=valid.sum()
    if not len(ids):
        return np.zeros((1,57),np.float32),np.zeros(12,np.float32),np.full(13,-1,np.int16),np.array([-32768])
    membership=(clu[:,None]==ids[None,:])&valid[:,None]
    counts=membership.sum(0);mx=np.where(membership,phi[:,None],-1).max(0)
    psum=(membership*phi[:,None]).sum(0);mean=psum/counts
    weights=np.exp(phi[:,None]*etas[None,:-1])*valid[:,None]
    weighted=membership.T@weights
    wins=np.r_[(weighted+1e-9*(mx[:,None]+1)).argmax(0),mx.argmax()].astype(np.int16)
    shares=np.c_[weighted/np.maximum(weighted.sum(0),1e-30),np.eye(len(ids))[mx.argmax()]]
    top=int(np.argmax(np.where(valid,phi,-np.inf)));removed=weighted.copy()
    removed[clu[top]==ids]-=weights[top]
    removed_mx=np.where(membership & (np.arange(n)[:,None]!=top),phi[:,None],-1).max(0)
    removed_wins=np.r_[(removed+1e-9*(removed_mx[:,None]+1)).argmax(0),removed_mx.argmax()]
    surviving=np.arange(len(ids))[:,None]==removed_wins[None,:]
    if nv==1:surviving[:]=False
    mid=max(n//2,1);rows=[]
    for i,a in enumerate(ids):
        x=phi[clu==a];sx=np.sort(x)[::-1]
        top3=np.pad(sx[:3],(0,max(0,3-len(sx))),constant_values=0)
        quant=np.quantile(x,[.25,.5,.75,.9])
        early=np.sum(clu[:mid]==a)/mid
        recent=np.sum(clu[mid:]==a)/max(n-mid,1)
        base=[counts[i]/n,counts[i]/nv,mean[i],x.std(),x.min(),*top3,*quant,
              np.sum(x>=.6)/n,np.sum(x>=.8)/n,np.sum(x>=.9)/n,
              len(x)>=2,len(x)>=3,top3[0]-top3[1] if len(x)>=2 else top3[0],
              np.mean(x>=phi[valid].mean()),early,recent,recent-early,
              np.log1p(counts[i])/np.log(65),np.sum(clu[-min(4,n):]==a)/min(4,n),
              counts[i]==counts.max(),mx[i]==mx.max(),np.mean(wins==i),
              np.sum(x)/max(psum.sum(),1e-9),np.mean(np.log1p(ell[clu==a]))/10,
              np.max(ell[clu==a])/max(np.max(ell),1),np.min(ell[clu==a])/max(np.max(ell),1)]
        rows.append(np.r_[base,shares[i],surviving[i]])
    p=counts/nv
    context=np.array([np.log1p(n)/np.log(65),nv/n,len(ids)/n,counts.max()/n,
        phi[valid].mean(),phi[valid].std(),phi[valid].max(),np.median(phi[valid]),
        (1-nv/n),np.log1p(np.mean(ell))/10,-np.sum(p*np.log(p))/np.log(max(n,2)),
        len(np.unique(wins))/13],np.float32)
    features=np.array(rows,np.float32)
    assert features.shape[1]==57 and np.isfinite(features).all()
    return features,context,wins,ids


def pack(records):
    offsets=np.r_[0,np.cumsum([len(r[0]) for r in records])]
    return dict(x=np.concatenate([r[0] for r in records]),context=np.stack([r[1] for r in records]),
        winners=np.stack([r[2] for r in records]),ids=np.concatenate([r[3] for r in records]),offsets=offsets)


def prepare(tag):
    raw=d.load(tag,'train');q=len(raw['qid']);records=[];ys=[]
    corr=np.zeros((q,2,64,13),bool)
    for i in range(q):
        rng=np.random.default_rng([d.SEED,int(raw['qid'][i]),10])
        perms=np.stack([rng.permutation(64) for _ in range(2)])
        wins=d.votes(raw['phi'][i],raw['clu'][i],raw['none'][i],raw['etas'],perms)
        corr[i]=d.score_winners(wins,raw['clu'][i],raw['good'][i])
        good_ids=np.unique(raw['clu'][i,raw['good'][i]])
        for path in perms:
            for n in d.CP:
                p=path[:n]
                r=state(raw['phi'][i,p],raw['clu'][i,p],raw['ell'][i,p],raw['none'][i],raw['etas'])
                records.append(r);ys.append(np.isin(r[3],good_ids))
        if i%500==0:d.log(tag,'answer features',i,'/',q)
    result=pack(records);result.update(y=np.concatenate(ys),corr=corr,qid=raw['qid'],etas=raw['etas'],
        state_q=np.repeat(np.arange(q),2*len(d.CP)),state_stage=np.tile(np.arange(len(d.CP)),q*2))
    np.savez_compressed(d.HERE/tag/'train_features.npz',**result)
    d.log(tag,'features ready',result['x'].shape)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);args=p.parse_args();prepare(args.cell)
