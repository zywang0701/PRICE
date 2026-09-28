"""Committed actions at fixed prices and retrospective MGF-cost envelopes."""
import numpy as np
from allocation import upper_hull


def actions(V,logC,lambdas,k_fixed=None):
    V=np.asarray(V,float);Q,K,T=V.shape;C=np.exp(logC)
    ks=V.argmax(1) if k_fixed is None else np.full((Q,T),k_fixed,int)
    R=np.take_along_axis(V,ks[:,None,:],axis=1)[:,0,:]
    n=np.empty((len(lambdas),Q),np.uint8);k=np.empty_like(n)
    for q in range(Q):
        h=upper_hull(C[q],R[q])
        slopes=np.diff(R[q,h])/np.diff(C[q,h])
        # Equal priced values resolve to the smaller count.
        at=np.searchsorted(-slopes,-np.asarray(lambdas),side='left')
        nn=h[at];n[:,q]=nn+1;k[:,q]=ks[q,nn]
    return n,k


def frontier(cost,reward):
    cost=np.asarray(cost,float);reward=np.asarray(reward,float)
    # At each measured cost retain the best raw point, then discard dominated
    # points and form the upper concave hull on the MGF (not log-budget) scale.
    order=np.lexsort((np.arange(len(cost)),-reward,cost))
    cs=cost[order];order=order[np.r_[True,cs[1:]!=cs[:-1]]]
    return order[upper_hull(cost[order],reward[order])]


def read_point(cost,reward,budget,gamma,allowed=None):
    allowed=np.arange(len(cost)) if allowed is None else np.asarray(allowed)
    h=allowed[frontier(cost[allowed],reward[allowed])]
    target=np.exp(gamma*budget);at=int(np.searchsorted(cost[h],target,side='left'))
    if at==0:
        lo=hi=int(h[0]);theta=0.;status='below_minimum' if target<cost[lo]*(1-1e-12) else 'exact'
    elif at>=len(h):
        lo=hi=int(h[-1]);theta=0.;status='saturated'
    else:
        lo=int(h[at-1]);hi=int(h[at]);theta=float((target-cost[lo])/(cost[hi]-cost[lo]));status='exact'
    C=(1-theta)*cost[lo]+theta*cost[hi];R=(1-theta)*reward[lo]+theta*reward[hi]
    return dict(lo=lo,hi=hi,theta=theta,status=status,b_actual=float(np.log(C)/gamma),R=float(R))


def mix(x,p):return (1-p['theta'])*x[p['lo']]+p['theta']*x[p['hi']]


def compact_indices(changed):
    """Keep every action state and a representative from the nested half grid."""
    starts=np.flatnonzero(changed);ends=np.r_[starts[1:],len(changed)];keep=set(starts.tolist())
    for start,end in zip(starts,ends):
        coarse=0 if start==0 else start if start%2==1 else start+1
        if coarse<end:keep.add(int(coarse))
    return np.array(sorted(keep),int)

