import pickle
import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
import setup22 as s


def design(d,pca,scope):
    f=d['feat'];q,paths,m,_=f.shape
    global_features=np.concatenate([f[...,:15],f[...,119:]],axis=-1).reshape(-1,19)
    poly=PolynomialFeatures(2,include_bias=False).fit_transform(global_features)
    x=np.c_[poly,f[...,15:119].reshape(-1,104),np.tile(np.c_[s.CP/64,np.log(s.CP),1/np.sqrt(s.CP)],(q*paths,1))]
    if scope=='query_prefix':x=np.c_[x,np.repeat(pca.transform(d['H']),paths*m,axis=0)]
    return x.astype(np.float32)


def candidate_design(d,pca):
    f=d['feat'];q,paths,m,_=f.shape
    global_features=np.concatenate([f[...,:15],f[...,119:]],axis=-1).reshape(-1,19)
    z=np.c_[global_features,np.repeat(pca.transform(d['H'])[:,:8],paths*m,axis=0),
            np.tile(np.c_[s.CP/64,np.log(s.CP),1/np.sqrt(s.CP)],(q*paths,1))]
    cand=f[...,15:119].reshape(-1,13,8)
    x=np.c_[np.repeat(z,13,axis=0),cand.reshape(-1,8),np.tile(np.arange(13)/12,len(z))]
    return x.astype(np.float32)


def choose(prob,k0,penalty):
    z=prob.copy();z[...,k0]+=penalty+1e-10
    return z.argmax(-1).astype(np.int8)


def train(tag):
    root=s.HERE/tag;fit=dict(np.load(root/'fit.npz'));tune=dict(np.load(root/'tune.npz'))
    pca=PCA(n_components=32,random_state=s.SEED).fit(fit['H'])
    models={};scores=[];y=fit['V'].reshape(len(fit['qid']),-1).astype(float);ymean=y.mean(0)
    for scope in ['prefix','query_prefix']:
        x=design(fit,pca,scope);scaler=StandardScaler().fit(x);x=scaler.transform(x).astype(float)
        xtx=x.T@x
        summed=x.reshape(len(fit['qid']),-1,x.shape[-1]).sum(1)
        xty=summed.T@(y-ymean)
        xt=scaler.transform(design(tune,pca,scope)).astype(float)
        states_per_q=tune['feat'].shape[1]*8
        for alpha in [100.,1000.,10000.]:
            coef=np.linalg.solve(xtx+alpha*np.eye(len(xtx)),xty);sse=0.;num=0
            for lo in range(0,len(xt),2048):
                pred=xt[lo:lo+2048]@coef+ymean
                target=tune['V'][np.arange(lo,min(lo+2048,len(xt)))//states_per_q].reshape(len(pred),-1)
                sse+=float(np.square(pred-target).sum());num+=pred.size
            key=scope+'_'+str(int(alpha));models[key]=dict(scope=scope,scaler=scaler,coef=coef.astype(np.float32),intercept=ymean.astype(np.float32))
            scores.append(dict(name=key,mse=sse/num));s.log(tag,'curve',key,'tune MSE',sse/num)
        del x,xt
    best=min(scores,key=lambda z:z['mse'])['name'];curve=models[best]
    prior=Ridge(alpha=100.).fit(pca.transform(fit['H']),fit['logM'])
    k0=int(fit['V'].mean((0,2)).argmax())
    X=candidate_design(fit,pca);Xt=candidate_design(tune,pca)
    y=fit['corr'][:,:,s.CP-1].reshape(-1)
    cand=fit['feat'][...,15:119].reshape(-1,13,8)
    disagreement=np.any(np.ptp(cand[:,:,[3,4,5]],axis=1)>1e-7,axis=1)
    use=np.repeat(disagreement | (np.arange(len(cand))%10==0),13)
    ck=[];classifiers={};tune_probs={}
    for leaves in [7,15]:
        clf=HistGradientBoostingClassifier(max_leaf_nodes=leaves,max_iter=120,learning_rate=.05,min_samples_leaf=100,
                                           l2_regularization=1.,early_stopping=False,random_state=s.SEED).fit(X[use],y[use])
        p=clf.predict_proba(Xt)[:,1].reshape(*tune['feat'].shape[:3],13)
        for penalty in [0.,.01,.03,.05]:
            k=choose(p,k0,penalty);c=np.take_along_axis(tune['corr'][:,:,s.CP-1],k[...,None],axis=-1)[...,0]
            accuracy=float(c[:,:,[1,2,3,4,5,7]].mean())
            ck.append(dict(leaves=leaves,penalty=penalty,accuracy=accuracy))
        classifiers[leaves]=clf;tune_probs[leaves]=p
        s.log(tag,'conditional vote classifier',leaves,'trained',int(use.sum()),'rows')
    bestc=max(ck,key=lambda z:(z['accuracy'],z['penalty'],-z['leaves']))
    bundle=dict(pca=pca,curve=curve,prior=prior,k0=k0,classifier=classifiers[bestc['leaves']],
                classifier_penalty=bestc['penalty'],curve_candidates=scores,classifier_candidates=ck,
                curve_selected=best,classifier_selected=bestc)
    with (root/'models.pkl').open('wb') as f:pickle.dump(bundle,f)
    s.dump(root/'model_training.json',dict(curve_candidates=scores,curve_selected=best,classifier_candidates=ck,
                                          classifier_selected=bestc,reference_temperature_index=k0))
    s.log(tag,'models frozen',best,bestc)


def infer(bundle,d):
    """Label-blind runtime: only H, feat, ell and tok are read."""
    q,paths,_,_=d['feat'].shape;curve=bundle['curve'];pca=bundle['pca']
    x=curve['scaler'].transform(design(d,pca,curve['scope']))
    gains=np.zeros((2,q,paths,64,13),np.float32);envg=np.zeros((2,q,paths,64),np.float32)
    kr=np.zeros((q,paths,64),np.int8)
    for lo in range(0,len(x),1024):
        hi=min(lo+1024,len(x));pred=np.clip(x[lo:hi]@curve['coef']+curve['intercept'],0,1).reshape(-1,13,64)
        ix=np.arange(lo,hi);qq=ix//(paths*8);ss=(ix//8)%paths;cp=ix%8
        for stage,first in enumerate(s.CP):
            sel=np.flatnonzero(cp==stage)
            if not len(sel):continue
            pp=pred[sel];E=pp.max(1);qa,sa=qq[sel],ss[sel]
            end=s.CP[stage+1]-1 if stage<7 else 64
            for n in range(first,end+1):
                kr[qa,sa,n-1]=pp[:,:,n-1].argmax(1)
                if n==64:continue
                for wi,window in enumerate([1,4]):
                    future=np.arange(n,min(n+window,64));den=future-(n-1)
                    gains[wi,qa,sa,n-1]=np.max((pp[:,:,future]-pp[:,:,n-1,None])/den,axis=2)
                    envg[wi,qa,sa,n-1]=np.max((E[:,future]-E[:,n-1,None])/den,axis=1)
    prob=bundle['classifier'].predict_proba(candidate_design(d,pca))[:,1].reshape(q,paths,8,13)
    kc=choose(prob,bundle['k0'],bundle['classifier_penalty'])
    cp=np.searchsorted(s.CP,np.arange(1,65),side='right')-1;kc=kc[:,:,cp]
    ell=d['ell'].astype(float);prior=np.maximum(bundle['prior'].predict(pca.transform(d['H'])),1e-6)
    observed=np.logaddexp.accumulate(s.GAMMA*ell,axis=2)-np.log(np.arange(1,65))
    weight=np.arange(1,65)/(np.arange(1,65)+8.)
    logM=(1-weight)*prior[:,None,None]+weight*observed
    logcost=s.GAMMA*d['tok'].astype(float)+np.log(np.expm1(np.maximum(logM,1e-6)))
    return dict(gains=gains,envg=envg,regression_k=kr,conditional_k=kc,logcost=logcost,qid=d['qid'])
