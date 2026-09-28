"""Exact finite-grid allocation using query-wise concave hulls and their slopes."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp,gammaln


def finite_pool_logcost(lengths,gamma,horizon):
    """Expected exp(gamma * sum of n distinct uniform draws), for n=1..horizon."""
    x=gamma*np.asarray(lengths,float);m=len(x);assert horizon<=m
    coef=np.full(horizon+1,-np.inf);coef[0]=0.
    for j,a in enumerate(x):
        hi=min(j+1,horizon)
        coef[1:hi+1]=np.logaddexp(coef[1:hi+1],a+coef[:hi])
    n=np.arange(1,horizon+1)
    return coef[1:]-(gammaln(m+1)-gammaln(n+1)-gammaln(m-n+1))


def upper_hull(cost,reward):
    """Indices of the increasing, nondominated upper concave chain."""
    keep=[];best=-np.inf
    for j,(c,r) in enumerate(zip(cost,reward)):
        if r<=best:continue
        if keep and c==cost[keep[-1]]:keep.pop()
        while len(keep)>=2:
            a,b=keep[-2:]
            left=(reward[b]-reward[a])/(cost[b]-cost[a])
            right=(r-reward[b])/(c-cost[b])
            if left>right:break
            keep.pop()
        keep.append(j);best=r
    return np.array(keep,int)


@dataclass
class Policy:
    k_low:np.ndarray
    n_low:np.ndarray
    k_high:np.ndarray
    n_high:np.ndarray
    theta:float  # probability of high-cost policy
    source_reward:float
    source_cost:float
    status:str


class Frontier:
    def __init__(self,V,logcost,temperature=None):
        self.V=np.asarray(V,float);self.Q,self.K,self.T=self.V.shape
        assert np.all(np.isfinite(logcost)) and np.max(logcost)<650
        self.C=np.exp(logcost)
        assert np.all(np.diff(self.C,axis=1)>0)
        self.k=(self.V.argmax(1) if temperature is None else np.full((self.Q,self.T),temperature,int))
        self.R=np.take_along_axis(self.V,self.k[:,None,:],axis=1)[:,0,:]
        self.initial=np.zeros(self.Q,int);edges=[]
        for q in range(self.Q):
            h=upper_hull(self.C[q],self.R[q]);self.initial[q]=h[0]
            for a,b in zip(h[:-1],h[1:]):
                dc=self.C[q,b]-self.C[q,a];dr=self.R[q,b]-self.R[q,a]
                edges.append((dr/dc,q,int(a),int(b),dc/self.Q,dr/self.Q))
        edges.sort(key=lambda e:(-e[0],e[1],e[3]))
        self.edges=edges
        r0=self.R[np.arange(self.Q),self.initial].mean()
        c0=self.C[np.arange(self.Q),self.initial].mean()
        self.costs=np.r_[c0,c0+np.cumsum([e[4] for e in edges])]
        self.rewards=np.r_[r0,r0+np.cumsum([e[5] for e in edges])]
        assert np.all(np.diff(self.costs)>0)

    def at(self,B):
        q=np.arange(self.Q);j=int(np.searchsorted(self.costs,B,side='left'))
        n=self.initial.copy()
        used=min(max(j-1,0),len(self.edges))
        for e in self.edges[:used]:n[e[1]]=e[3]
        if j==0:
            lo=hi=n;theta=0.;status='below_minimum' if B<self.costs[0]*(1-1e-12) else 'exact'
        elif j>=len(self.costs):
            for e in self.edges[used:]:n[e[1]]=e[3]
            lo=hi=n;theta=0.;status='saturated'
        else:
            lo=n;hi=n.copy();edge=self.edges[j-1];assert hi[edge[1]]==edge[2]
            hi[edge[1]]=edge[3]
            theta=float(np.clip((B-self.costs[j-1])/(self.costs[j]-self.costs[j-1]),0,1));status='exact'
        kl=self.k[q,lo];kh=self.k[q,hi]
        R=((1-theta)*self.R[q,lo]+theta*self.R[q,hi]).mean()
        cost=((1-theta)*self.C[q,lo]+theta*self.C[q,hi]).mean()
        return Policy(kl.copy(),lo+1,kh.copy(),hi+1,theta,float(R),float(cost),status)


def evaluate(policy,V,logcost,mean_length,override=None):
    q=np.arange(len(V));kl=policy.k_low if override is None else np.full(len(q),override)
    kh=policy.k_high if override is None else np.full(len(q),override)
    lo=policy.n_low-1;hi=policy.n_high-1;theta=policy.theta
    R=(1-theta)*V[q,kl,lo].astype(float)+theta*V[q,kh,hi].astype(float)
    cost=(1-theta)*np.exp(logcost[q,lo])+theta*np.exp(logcost[q,hi])
    mean=(1-theta)*policy.n_low*mean_length+theta*policy.n_high*mean_length
    return R,cost,mean


def common_frontier(V,logcost,temperature):
    c=logsumexp(logcost,axis=0,keepdims=True)-np.log(len(V))
    return Frontier(np.asarray(V,float).mean(0,keepdims=True),c,temperature)


def expand_common(p,Q):
    return Policy(*(np.full(Q,int(x[0]),int) for x in [p.k_low,p.n_low,p.k_high,p.n_high]),
        p.theta,p.source_reward,p.source_cost,p.status)
