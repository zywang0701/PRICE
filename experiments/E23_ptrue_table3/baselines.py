"""Vectorized original baseline sweeps; return label-blind winner cluster IDs."""
import numpy as np
from scipy.special import betainc

SC_N=[1,2,3,4,5,6,7,8,9,10,12,14,16,19,22,27,32,39,48,56,64]
BON_N=[1,2,4,8,16,32,48,64]
CISC_N=[1,2,3,5,7,10,15,20,30]
CISC_T_GRID=np.geomspace(1e-4,1e4,80)
AC_THRESH=[.5,.7,.75,.8,.85,.9,.95,.99,.995,.999]
ESC_WL=([(8,L) for L in [16,24,32,40,48,56,64]]+[(w,40) for w in [3,4,5,6,10]]+
        [(3,6),(3,9),(3,12),(4,8),(5,10),(5,15),(2,2),(2,4),(2,6),(2,8),(3,3)])
DC_OFF_N=[2,4,8,16,24,32,48,64];DC_OFF_ETA=[.1,.9];DC_OFF_STATS=['lgc','bot10','tail']
STAB=1-betainc(np.arange(132)[:,None]+1.,np.arange(132)[None,:]+1.,.5)


def cisc_winners(clu,none,perms,ptrue,temperatures,counts=(5,10)):
    ids=np.unique(clu);cid=np.searchsorted(ids,clu[perms]);valid=clu[perms]!=none;S=perms.shape[0]
    out=[]
    for n in counts:
        f=ptrue[perms[:,:n]];f=np.where(valid[:,:n],f,-np.inf)
        mx=f.max(1);mx=np.where(np.isfinite(mx),mx,0.)
        weights=np.exp((f-mx[:,None])[...,None]/np.asarray(temperatures)[None,None,:])
        scores=np.zeros((S,len(ids),len(temperatures)))
        np.add.at(scores,(np.arange(S)[:,None],cid[:,:n]),weights)
        winners=ids[scores.argmax(1)];winners=np.where(valid[:,:n].any(1)[:,None],winners,-32768)
        out.append(winners)
    return np.stack(out,axis=1)


def one_query(clu,ell,none,perms,ptrue,lgc,bot10,tail,Tstar):
    ids=np.unique(clu);cp=clu[perms];cid=np.searchsorted(ids,cp);valid=cp!=none;S,T=cp.shape
    oh=np.zeros((S,T,len(ids)));oh[np.arange(S)[:,None],np.arange(T)[None,:],cid]=valid
    counts=np.cumsum(oh,axis=1);cum=np.cumsum(ell[perms],axis=1);out={}
    order=np.argsort(counts,axis=2)[:,:,::-1];top=order[:,:,0];m1=np.take_along_axis(counts,top[:,:,None],axis=2)[:,:,0]
    m2=np.take_along_axis(counts,order[:,:,1,None],axis=2)[:,:,0] if len(ids)>1 else np.zeros_like(m1)
    stability=STAB[m1.astype(int),m2.astype(int)]
    for threshold,cap in [(x,64) for x in AC_THRESH]+[(.95,40)]:
        hit=stability[:,:cap]>=threshold;hit[:,-1]=True;n=hit.argmax(1);rows=np.arange(S)
        winner=np.where(m1[rows,n]>0,ids[top[rows,n]],-32768)
        out[('ac' if cap==64 else 'ac_pub',str(threshold))]=(winner,cum[rows,n])
    for w,L in ESC_WL:
        blocks=cp[:,:L//w*w].reshape(S,L//w,w);hit=np.all(blocks==blocks[:,:,0,None],axis=2)&(blocks[:,:,0]!=none)
        first=hit.argmax(1);has=hit.any(1);n=np.where(has,(first+1)*w,L)
        fallback=np.where(counts[:,L-1].sum(1)>0,ids[counts[:,L-1].argmax(1)],-32768)
        winner=np.where(has,blocks[np.arange(S),first,0],fallback)
        out[('esc',f'{w}x{L}')]=(winner,cum[np.arange(S),n-1])
    cw=cisc_winners(clu,none,perms,ptrue,[Tstar],CISC_N)[:,:,0]
    for j,n in enumerate(CISC_N):out[('cisc',str(n))]=(cw[:,j],cum[:,n-1])
    for stat,raw in [('lgc',lgc),('bot10',bot10),('tail',tail)]:
        for n in DC_OFF_N:
            v=raw[perms[:,:n]]
            for eta in DC_OFF_ETA:
                keep=(v>=np.quantile(v,1-eta,axis=1)[:,None])&valid[:,:n]
                weights=np.maximum(v,0)*keep;scores=np.zeros((S,len(ids)))
                np.add.at(scores,(np.arange(S)[:,None],cid[:,:n]),weights)
                winner=np.where(keep.any(1),ids[scores.argmax(1)],-32768)
                out[('dcoff',f'{stat}|{eta}|{n}')]=(winner,cum[:,n-1])
    return out
