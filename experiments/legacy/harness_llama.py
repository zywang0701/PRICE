"""Shared harness for the m<=4 probe policy search (MATH-train -> MATH-500, Qwen2.5-1.5B, phi = deepconf2).

RULES (do not violate):
  * calibrate/train ONLY on `tr` (MATH-train). `te` labels/curves are for scoring only.
  * at test time a policy may use: te['H'] (embedding), the FIRST <=4 probe rollouts of each query
    (te['phi'][:, :4] deepconf2 score, te['clu'][:, :4] answer-cluster id, te['ell'][:, :4] token length,
    te['okp'][:, :4] non-abstention flag, te['none'] the abstention cluster id), and any calibration artifact from tr.
  * NEVER use te['V'], te['cor'], te['L'] to make decisions (they define the reward/cost).
  * probe rollouts are paid for and reused in the vote: if a policy looks at m probe rollouts, its committed count is max(n, m).
Reward for a decision (tau index k, count n) on test query i is te['V'][i, k, n-1]; token cost is te['L'][i] * n.
Score = accuracy at equal mean tokens/query, read off the (tokens, accuracy) curve traced over the price grid LAMS.
"""
import numpy as np, sys, os, warnings
warnings.filterwarnings("ignore")
SP=os.path.dirname(os.path.abspath(__file__))
ROOT=os.environ.get("PRICE_ROOT", os.path.dirname(os.path.dirname(SP)))+"/"   # repo root
sys.path.insert(0, ROOT+"generation/prep")
from lp_lib import apply_lcm
K=13; T=64
BUDGETS=[1500,2200,3000,4500,6000,12000]
LAMS=np.logspace(-6.5,-1.5,80)

def _load(cell,sub,ov=False):
    p=np.load(f"{ROOT}outputs/cells/{cell}/{sub}/prep{'_ov' if ov else ''}__phi_deepconf2.npz",allow_pickle=True); ok=p["target"]!=p["none_cluster"]
    emb=np.load(f"{ROOT}outputs/cells/{cell}/pools/embeddings.npz"); H=emb["H"]/np.linalg.norm(emb["H"],axis=1,keepdims=True); row={int(q):i for i,q in enumerate(emb["query_id"])}
    H=H[[row[int(q)] for q in p["query_id"]]]
    d=dict(H=H[ok],V=p["V_tgt"].astype(float)[ok,:,:T],L=p["ell_tgt_mean"][ok],phi=p["phi_probe"][ok].astype(float),clu=p["clu_probe"][ok],
           ell=p["ell_probe"][ok].astype(float),okp=p["ok_probe"][ok],cor=p["cor_probe"][ok],none=p["none_cluster"][ok],etas=p["etas"],qid=p["query_id"][ok])
    d["Vproj"]=d["V"].copy()   # E003: RAW curves — no cummax, no concave projection
    return d
_cache={}
def load_all():
    if "tr" not in _cache:
        _cache["tr"]=_load("llama32-3b_mathtrain","lp3",ov=True); _cache["te"]=_load("llama32-3b_math500","lp")
    return _cache["tr"],_cache["te"]

def difficulty_index():
    """Embedding -> single-rollout accuracy. Returns (oof_train_pred, test_pred), clipped to [0,1]. Cached on disk."""
    f=SP+"/difficulty_mlp_llama.npz"
    if os.path.exists(f):
        z=np.load(f); return z["oof"],z["te"]
    from sklearn.neural_network import MLPRegressor; from sklearn.decomposition import PCA; from sklearn.model_selection import KFold
    tr,te=load_all(); v1=tr["V"][:,0,0]
    pca=PCA(256,random_state=0).fit(tr["H"]); Ptr=pca.transform(tr["H"]); Pte=pca.transform(te["H"])
    mk=lambda s: MLPRegressor(hidden_layer_sizes=(256,64),alpha=1e-3,max_iter=300,early_stopping=True,random_state=s)
    oof=np.zeros(len(Ptr))
    for i,(a,b) in enumerate(KFold(5,shuffle=True,random_state=0).split(Ptr)): oof[b]=mk(i).fit(Ptr[a],v1[a]).predict(Ptr[b])
    oof=np.clip(oof,0,1); tep=np.clip(mk(0).fit(Ptr,v1).predict(Pte),0,1)
    np.savez(f,oof=oof,te=tep); return oof,tep

def probe_feats(d,m=4):
    """Label-free features of the first m probe rollouts (deepconf2 score, answer agreement, length)."""
    F=[]
    for i in range(len(d["V"])):
        cl=d["clu"][i,:m]; ph=d["phi"][i,:m]; el=d["ell"][i,:m]; ok_=d["okp"][i,:m]&(cl!=d["none"][i])
        if ok_.any(): ids,cnt=np.unique(cl[ok_],return_counts=True)
        else: ids,cnt=np.array([]),np.array([0])
        top=cnt.max()
        # score of the plurality cluster vs others
        if ok_.any():
            plur=ids[cnt.argmax()]; ph_pl=ph[ok_][cl[ok_]==plur].mean(); oth=ph[ok_][cl[ok_]!=plur]; ph_ot=oth.mean() if len(oth) else ph_pl
            top_is_plur=float(cl[ok_][ph[ok_].argmax()]==plur)
        else: ph_pl=ph_ot=ph.mean(); top_is_plur=0.0
        F.append([top/m, len(ids)/m, ok_.mean(), ph.mean(), ph.max(), ph.min(), ph.std(), np.log(el.mean()+1), np.log(el.max()+1),
                  float(top==1), float(top==m), ph_pl, ph_ot, ph_pl-ph_ot, top_is_plur])
    return np.array(F)
PROBE_FEAT_NAMES=["agree_share","n_distinct/m","valid_share","phi_mean","phi_max","phi_min","phi_std","log_len_mean","log_len_max","all_distinct","all_agree","phi_plur","phi_other","phi_gap","top_is_plur"]

def family(idx_tr,idx_te,h=0.05,tr=None):
    """Kernel-smoothed curve family (Q_te,K,T) and mean length keyed on a scalar index (train side must be OOF)."""
    tr=tr or load_all()[0]
    W=np.exp(-0.5*((idx_te[:,None]-idx_tr[None,:])/h)**2); W/=W.sum(1,keepdims=True)
    return np.einsum("ij,jkt->ikt",W,tr["Vproj"]), W@tr["L"]

def monotone_best(Mg):
    """Mg (G,K): weighted mean V per bin (hard->easy) per tau. Non-increasing tau index across bins maximizing sum."""
    G=len(Mg); best=np.full((G,K),-np.inf); arg=np.zeros((G,K),int); best[0]=Mg[0]
    for g in range(1,G):
        for k in range(K):
            j=np.argmax(best[g-1,k:])+k; best[g,k]=best[g-1,j]+Mg[g,k]; arg[g,k]=j
    ks=[int(np.argmax(best[G-1]))]
    for g in range(G-1,0,-1): ks.append(arg[g,ks[-1]])
    return np.array(ks[::-1])

def mono_tau_mask(idx_tr,idx_te,n=16,nb=20,tr=None):
    tr=tr or load_all()[0]
    e=np.quantile(idx_tr,np.linspace(0,1,nb+1)[1:-1]); b=np.digitize(idx_tr,e); M=tr["V"][:,:,n-1]
    ks=monotone_best(np.array([M[b==g].mean(0)*np.sum(b==g) for g in range(nb)]))
    m=np.zeros((len(idx_te),K),bool); m[np.arange(len(idx_te)),ks[np.digitize(idx_te,e)]]=True; return m

def first_crossing(Vhat,Lhat,lam,allowed=None,nmin=1):
    """Per-query first-crossing on each allowed tau curve, then argmax net value. Returns (n, k)."""
    Q=len(Vhat); allowed=np.ones((Q,K),bool) if allowed is None else allowed
    best=np.full(Q,-np.inf); bn=np.ones(Q,int); bk=np.zeros(Q,int)
    for k in range(K):
        g=Vhat[:,k,1:]-Vhat[:,k,:-1].copy(); g=g.copy()
        if nmin>1: g[:,:nmin-1]=np.inf
        hit=g<=lam*Lhat[:,None]; nn=np.where(hit.any(1),hit.argmax(1)+1,T)
        val=np.where(allowed[:,k],Vhat[np.arange(Q),k,nn-1]-lam*Lhat*nn,-np.inf)
        take=val>best; best=np.where(take,val,best); bn=np.where(take,nn,bn); bk=np.where(take,k,bk)
    return bn,bk

def score_policy(decide,te=None):
    """decide(lam) -> (n array (Q,), k array (Q,)) using only allowed info. Returns dict budget->accuracy and the raw curve."""
    te=te or load_all()[1]; Q=len(te["V"]); pts=[]
    for lam in LAMS:
        n,k=decide(lam); n=np.clip(np.asarray(n,int),1,T); k=np.asarray(k,int)
        pts.append(((te["L"]*n).mean(), te["V"][np.arange(Q),k,n-1].mean()))
    c=np.array(pts); x,y=c[:,0],c[:,1]; o=np.argsort(x); x,y=x[o],np.maximum.accumulate(y[o])
    return {b:float(np.interp(b,x,y,left=np.nan)) for b in BUDGETS}, c

def fmt(name,res): return f"{name:60s}"+"".join(f"{res[b]:8.4f}" if not np.isnan(res[b]) else "     n/a" for b in BUDGETS)
def header(): return f"{'policy':60s}"+"".join(f"{b:>8d}" for b in BUDGETS)

def baselines():
    """Population single-tau (m=0) and zero-probe continuous difficulty index with monotone tau (m=0)."""
    tr,te=load_all(); oof,tep=difficulty_index(); Q=len(te["V"])
    ks=tr["V"][:,:,15].mean(0).argmax(); single=np.zeros((Q,K),bool); single[:,ks]=True
    pop=np.repeat(tr["Vproj"].mean(0,keepdims=True),Q,0); Lp=np.full(Q,tr["L"].mean())
    Vh,Lh=family(oof,tep); mm=mono_tau_mask(oof,tep)
    out={}
    out["BASE population, single tau (m=0)"]=score_policy(lambda lam: first_crossing(pop,Lp,lam,single))[0]
    out["BASE zero-probe difficulty index, monotone tau (m=0)"]=score_policy(lambda lam: first_crossing(Vh,Lh,lam,mm))[0]
    Porc=te["Vproj"]
    out["BOUND oracle per-query true curve"]=score_policy(lambda lam: first_crossing(Porc,te["L"],lam))[0]
    return out

if __name__=="__main__":
    print(header())
    for k,v in baselines().items(): print(fmt(k,v))
