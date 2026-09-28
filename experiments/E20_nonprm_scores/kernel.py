"""Incremental votes, retaining both common and legacy score-native tie rules."""
import numpy as np
from numba import njit


@njit(cache=True)
def reward_tables(phi,clu,none,good,strict,perms,etas):
    P,T=perms.shape;K=len(etas);C=int(np.max(clu))+1
    correct=np.zeros(C,np.bool_);strict_correct=np.zeros(C,np.bool_)
    for i in range(len(clu)):
        if good[i]:correct[clu[i]]=True
        if strict[i]:strict_correct[clu[i]]=True
    result=np.zeros((2,K,T),np.float64);strict_result=np.zeros_like(result)
    weights=np.zeros((len(phi),K-1),np.float64)
    for i in range(len(phi)):
        for k in range(K-1):weights[i,k]=np.exp(etas[k]*phi[i])
    for p in range(P):
        sums=np.zeros((K-1,C),np.float64);mx=np.full(C,-1.,np.float64)
        winner=np.full((2,K),-1,np.int64)
        for t in range(T):
            i=perms[p,t];a=clu[i]
            if a!=none:
                mx[a]=max(mx[a],phi[i])
                for k in range(K):
                    if k<K-1:
                        sums[k,a]+=weights[i,k]
                    for mode in range(2):
                        w=winner[mode,k]
                        if w<0:
                            winner[mode,k]=a
                        else:
                            if k==K-1:
                                v=mx[a];old=mx[w]
                            elif mode==0:
                                v=sums[k,a];old=sums[k,w]
                            else:
                                v=sums[k,a]+1e-9*(mx[a]+1);old=sums[k,w]+1e-9*(mx[w]+1)
                            if v>old or (v==old and a<w):winner[mode,k]=a
            for mode in range(2):
                for k in range(K):
                    w=winner[mode,k]
                    if w>=0:
                        result[mode,k,t]+=correct[w]
                        strict_result[mode,k,t]+=strict_correct[w]
    return result/P,strict_result/P
