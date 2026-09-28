import time,json
import numpy as np
import setup20 as s
from kernel import reward_tables


def reference(phi,cl,none,good,perms,etas):
    P,T=perms.shape;C=cl.max()+1
    pc=cl[perms];ph=phi[perms];valid=pc!=none
    oh=np.eye(C)[pc]*valid[:,:,None]
    mx=np.maximum.accumulate(np.where(oh>0,ph[:,:,None],-1.),axis=1)
    result=np.empty((2,len(etas),T))
    cg=np.unique(cl[good])
    for k,eta in enumerate(etas):
        if np.isinf(eta):
            w=mx.argmax(2)
            wins=[w,w]
        else:
            sums=np.cumsum(oh*np.exp(eta*ph)[:,:,None],axis=1)
            wins=[sums.argmax(2),(sums+1e-9*(mx+1)).argmax(2)]
        for mode,w in enumerate(wins):
            ok=np.isin(w,cg)&(np.cumsum(valid,axis=1)>0)
            result[mode,k]=ok.mean(0)
    return result


def main():
    rng=np.random.default_rng(427);etas=np.r_[0.,.3,1.,2.,4.,8.,16.,np.inf]
    checks=0
    for kind in ['continuous','tied','all_equal','abstentions']:
        phi=rng.random(80)
        if kind=='tied':phi=np.round(phi*3)/3
        if kind=='all_equal':phi[:]=.5
        cl=rng.integers(0,9,len(phi));none=8
        if kind=='abstentions':cl[:40]=8
        good=np.isin(cl,[1,3]);strict=cl==1
        perms=np.stack([rng.permutation(80)[:64] for _ in range(16)])
        got,gs=reward_tables(phi,cl,none,good,strict,perms,etas)
        np.testing.assert_array_equal(got,reference(phi,cl,none,good,perms,etas))
        np.testing.assert_array_equal(gs,reference(phi,cl,none,strict,perms,etas));checks+=1
    raw=s.D.load('qwen','test');elapsed=[]
    for q in [0,37,210,488]:
        ids=np.unique(raw['clu'][q]);cl=np.searchsorted(ids,raw['clu'][q]).astype(np.int64)
        no=int(np.searchsorted(ids,raw['none'][q])) if raw['none'][q] in ids else -1
        perms=np.stack([rng.permutation(128)[:64] for _ in range(256)])
        t=time.perf_counter()
        got,gs=reward_tables(raw['phi'][q],cl,no,raw['good'][q],raw['strict_good'][q],perms,raw['etas'])
        elapsed.append(time.perf_counter()-t)
        np.testing.assert_array_equal(got,reference(raw['phi'][q],cl,no,raw['good'][q],perms,raw['etas']))
        _,wins,_=s.D.old.prefix_features(raw['phi'][q],raw['clu'][q],raw['ell'][q],raw['none'][q],perms,raw['etas'])
        np.testing.assert_array_equal(got[1],s.D.score_winners(wins,raw['clu'][q],raw['good'][q]).mean(0).T)
        checks+=1
    result=dict(cases_checked=checks,primary_and_native_match_vector_reference=True,
        native_matches_existing_replay=True,seconds_per_query_half=elapsed)
    s.dump(s.HERE/'kernel_validation.json',result);print(result)


if __name__=='__main__':main()
