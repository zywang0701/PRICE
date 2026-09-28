"""Streaming P(True) controller. Runtime inputs contain no correctness labels."""
import pickle
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
import setup22 as s
from models import design,candidate_design,choose


class PTrueController:
    def __init__(self,model_directory,query_embedding,etas,arm,abstention_cluster=-1):
        root=Path(model_directory)
        with (root/'models.pkl').open('rb') as f:self.bundle=pickle.load(f)
        self.cdf=np.load(root/'cdf.npy');self.H=np.asarray(query_embedding,np.float32).reshape(1,-1)
        self.H/=np.maximum(np.linalg.norm(self.H,axis=1,keepdims=True),1e-20)
        self.etas=np.asarray(etas,float);self.arm=arm;self.none=abstention_cluster
        self.window=int(arm.rsplit('_w',1)[1]);assert self.window in [1,4]
        self.fixed=int(arm.split('_')[1]) if arm.startswith('fixed_') else None
        self.prior=max(float(self.bundle['prior'].predict(self.bundle['pca'].transform(self.H))[0]),1e-6)
        self.raw=[];self.phi=[];self.clu=[];self.ell=[];self.curve=None;self.conditional_k=None

    def observe(self,ptrue,answer_cluster,generated_tokens):
        if len(self.raw)>=64:raise ValueError('Horizon is 64 rollouts.')
        if not np.isfinite(ptrue) or not 0<=ptrue<=1:raise ValueError('P(True) must be in [0,1].')
        if generated_tokens<0:raise ValueError('Token count must be nonnegative.')
        self.raw.append(float(ptrue));self.clu.append(int(answer_cluster));self.ell.append(int(generated_tokens))
        self.phi.append(np.searchsorted(self.cdf,ptrue,side='right')/(len(self.cdf)+1.))
        n=len(self.raw);perms=np.arange(n)[None,:]
        feat,winners,_=s.D.old.prefix_features(np.array(self.phi),np.array(self.clu),np.array(self.ell),self.none,perms,self.etas)
        if n in s.CP:
            extra=np.array([np.mean(self.raw),np.std(self.raw),np.max(self.raw),np.min(self.raw)])
            row=np.r_[feat[0,-1],extra].astype(np.float32)
            # Only the current checkpoint row is consumed; placeholders contain no future information.
            inputs=dict(feat=np.broadcast_to(row,(1,1,8,123)).copy(),H=self.H)
            cp=int(np.flatnonzero(s.CP==n)[0]);reg=self.bundle['curve']
            x=reg['scaler'].transform(design(inputs,self.bundle['pca'],reg['scope']))[cp:cp+1]
            self.curve=np.clip(x@reg['coef']+reg['intercept'],0,1).reshape(13,64)
            xc=candidate_design(inputs,self.bundle['pca'])[cp*13:(cp+1)*13]
            prob=self.bundle['classifier'].predict_proba(xc)[:,1][None,:]
            self.conditional_k=int(choose(prob,self.bundle['k0'],self.bundle['classifier_penalty'])[0])
        if self.fixed is not None:k=self.fixed
        elif self.arm.startswith('conditional'):k=self.conditional_k
        else:k=int(self.curve[:,n-1].argmax())
        reward_curve=self.curve.max(0) if self.arm.startswith('regression') else self.curve[k]
        gain=-np.inf
        if n<64:
            future=np.arange(n,min(n+self.window,64))
            gain=float(np.max((reward_curve[future]-reward_curve[n-1])/(future-n+1)))
        observed=float(logsumexp(s.GAMMA*np.array(self.ell))-np.log(n));weight=n/(n+8.)
        logM=(1-weight)*self.prior+weight*observed
        logcost=s.GAMMA*sum(self.ell)+np.log(np.expm1(max(logM,1e-6)))
        return dict(count=n,temperature_index=k,temperature=float(self.etas[k]),answer_cluster=int(winners[0,n-1,k]),
                    predicted_gain=gain,log_marginal_cost=float(logcost),generated_tokens=sum(self.ell))

    @staticmethod
    def should_stop(state,log_price):
        return state['count']>=64 or state['predicted_gain']<=0 or np.log(state['predicted_gain'])<=log_price+state['log_marginal_cost']
