import pickle,hashlib
import setup24 as c
from setup24 import np,PolynomialFeatures,StandardScaler


def neighbors(H,fitH,fitY,ks):
    result={k:np.empty((len(H),fitY.shape[1]),np.float32) for k in ks};maximum=max(ks)
    for lo in range(0,len(H),32):
        sim=H[lo:lo+32]@fitH.T
        idx=np.argpartition(-sim,maximum-1,axis=1)[:,:maximum]
        order=np.argsort(-np.take_along_axis(sim,idx,axis=1),axis=1);idx=np.take_along_axis(idx,order,axis=1)
        for k in ks:result[k][lo:lo+len(sim)]=fitY[idx[:,:k]].mean(1)
    return result


def predict(bundle,H):
    """Runtime signature accepts query embeddings only; no rollout or label fields."""
    H=np.asarray(H,np.float32)
    if bundle['kind']=='constant':V=np.broadcast_to(bundle['mean'],(len(H),832)).copy()
    elif bundle['kind']=='knn':V=neighbors(H,bundle['fitH'],bundle['fitY'],[bundle['k']])[bundle['k']]
    else:
        x=bundle['poly'].transform(bundle['pca'].transform(H));x=bundle['scaler'].transform(x)
        V=x@bundle['coef']+bundle['mean']
    logM=np.maximum(bundle['prior'].predict(bundle['pca'].transform(H)),1e-6)
    return np.clip(V,0,1).reshape(len(H),13,64),logM


def train(tag):
    root=c.HERE/tag;root.mkdir(exist_ok=True)
    fit=np.load(c.E22/tag/'fit.npz');tune=np.load(c.E22/tag/'tune.npz')
    with (c.E22/tag/'models.pkl').open('rb') as f:old=pickle.load(f)
    H=fit['H'];Ht=tune['H'];Y=fit['V'].reshape(len(H),-1);Yt=tune['V'].reshape(len(Ht),-1)
    mean=Y.mean(0);pca=old['pca'];common=dict(pca=pca,prior=old['prior'],mean=mean)
    candidates={};scores=[]
    def retain(name,bundle,pred):
        mse=float(np.mean(np.square(np.clip(pred,0,1)-Yt)))
        candidates[name]=bundle;scores.append(dict(name=name,mse=mse));c.log(tag,name,'tune MSE',mse)
    retain('constant',dict(common,kind='constant'),np.broadcast_to(mean,Yt.shape))
    for degree in [1,2]:
        poly=PolynomialFeatures(degree,include_bias=False);x=poly.fit_transform(pca.transform(H)).astype(float)
        scaler=StandardScaler().fit(x);x=scaler.transform(x);xt=scaler.transform(poly.transform(pca.transform(Ht)))
        xtx=x.T@x;xty=x.T@(Y-mean)
        for alpha in [1.,10.,100.,1000.]:
            coef=np.linalg.solve(xtx+alpha*np.eye(x.shape[1]),xty)
            bundle=dict(common,kind='ridge',poly=poly,scaler=scaler,coef=coef,alpha=alpha,degree=degree)
            retain(f'ridge_d{degree}_a{int(alpha)}',bundle,xt@coef+mean)
    near=neighbors(Ht,H,Y,[32,128,512])
    for k,pred in near.items():retain(f'knn_{k}',dict(common,kind='knn',fitH=H,fitY=Y,k=k),pred)
    best=min(scores,key=lambda r:r['mse'])['name'];bundle=candidates[best]
    with (root/'models.pkl').open('wb') as f:pickle.dump(bundle,f)
    c.dump(root/'training.json',dict(selected=best,candidates=scores,fit_queries=len(H),tune_queries=len(Ht),
        runtime_inputs=['query_embedding'],cost_prior='E22 frozen query-only log-MGF ridge',
        model_sha256=hashlib.sha256((root/'models.pkl').read_bytes()).hexdigest(),
        E22_model_sha256=hashlib.sha256((c.E22/tag/'models.pkl').read_bytes()).hexdigest()))
    c.log(tag,'model frozen',best)
