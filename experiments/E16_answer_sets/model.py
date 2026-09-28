"""Shared answer scoring, Deep Sets comparisons and cross-pool utility learning."""
import argparse
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import data as d
from split_pool import closest_best
torch.set_num_threads(3)
FAMILIES=['answer_mlp','answer_set','cross_pool']
CHECKPOINTS=[15,30,60]
THRESHOLDS=[0.,.02,.05,.1,.2]
DEVICE='cpu'


class AnswerNet(nn.Module):
    def __init__(self,family,width=48):
        super().__init__();self.family=family;self.width=width
        self.enc=nn.Sequential(nn.Linear(69,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        size=width+12 if family=='answer_mlp' else (2*width+12 if family=='cross_pool' else 3*width+12)
        self.head=nn.Sequential(nn.Linear(size,width),nn.SiLU(),nn.Linear(width,13 if family=='cross_pool' else 1))

    def forward(self,x,context,mask):
        ctx=context[:,None].expand(-1,x.shape[1],-1)
        z=self.enc(torch.cat([x,ctx],-1))
        mean=(z*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
        mx=z.masked_fill(~mask[:,:,None],-1e4).max(1).values
        mx=torch.where(mask.any(1)[:,None],mx,torch.zeros_like(mx))
        if self.family=='cross_pool':return self.head(torch.cat([mean,mx,context],-1))
        if self.family=='answer_mlp':h=torch.cat([z,ctx],-1)
        else:h=torch.cat([z,mean[:,None].expand_as(z),mx[:,None].expand_as(z),ctx],-1)
        return self.head(h).squeeze(-1)


def batch(z,indices):
    indices=np.asarray(indices);starts=z['offsets'][indices];sizes=z['offsets'][indices+1]-starts
    j=np.arange(int(sizes.max()))[None,:];exists=j<sizes[:,None]
    rows=np.minimum(starts[:,None]+j,len(z['x'])-1)
    x=z['x'][rows];mask=exists&(z['ids'][rows]>=0)
    args=[torch.as_tensor(x,dtype=torch.float32,device=DEVICE),
          torch.as_tensor(z['context'][indices],dtype=torch.float32,device=DEVICE),
          torch.as_tensor(mask,dtype=torch.bool,device=DEVICE)]
    return args,rows,exists


def scores(net,z,indices,batch_size=512):
    net.eval();out=np.zeros((len(indices),13),np.float32)
    with torch.no_grad():
        for lo in range(0,len(indices),batch_size):
            ix=indices[lo:lo+batch_size];args,_,_=batch(z,ix);pred=net(*args)
            if net.family=='cross_pool':v=pred/10
            else:
                w=torch.as_tensor(z['winners'][ix].astype(np.int64),device=DEVICE)
                v=torch.gather(pred.sigmoid(),1,w.clamp_min(0))
                v=torch.where(w>=0,v,torch.zeros_like(v))
            out[lo:lo+len(ix)]=v.cpu().numpy()
    return out


def choose(values,k0,threshold):
    k=closest_best(values,k0)
    benefit=np.take_along_axis(values,k[...,None],axis=-1)[...,0]-values[...,k0]
    return np.where(benefit>threshold+1e-10,k,k0).astype(np.int8)


def evaluate_scores(values,z,qidx,k0,threshold,family):
    values=values.reshape(len(qidx),2,len(d.CP),13)
    if family=='cross_pool':values[:,:,6:]=values[:,:,5:6]
    k=choose(values,k0,threshold)
    stages=np.searchsorted(d.CP,d.EVAL)
    correct=z['corr'][qidx][:,:,d.EVAL-1]
    result=np.take_along_axis(correct,k[:,:,stages,None],axis=-1)[...,0]
    return float(result.mean()),result.mean((0,1)).tolist()


def train_net(train,fit_idx,family,k0,epochs,checkpoint=None):
    torch.manual_seed(d.SEED);rng=np.random.default_rng(d.SEED)
    net=AnswerNet(family).to(DEVICE);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.003)
    per_query=len(train['state_q'])//len(train['qid'])
    stages=len(d.CP) if family!='cross_pool' else 6
    allowed=np.flatnonzero(np.arange(per_query)%stages!=0)
    started=time.perf_counter()
    for epoch in range(1,epochs+1):
        net.train();qs=np.tile(fit_idx,2);rng.shuffle(qs)
        ix=qs*per_query+rng.choice(allowed,size=len(qs));losses=[]
        for lo in range(0,len(ix),384):
            ii=ix[lo:lo+384];args,rows,exists=batch(train,ii);mask=args[2]
            out=net(*args)
            if family=='cross_pool':
                y=train['utility'][ii]-train['utility'][ii,k0,None]
                loss=F.mse_loss(out,torch.as_tensor(y*10,dtype=torch.float32,device=DEVICE))
            else:
                y=torch.as_tensor(train['y'][rows],dtype=torch.float32,device=DEVICE)
                bce=F.binary_cross_entropy_with_logits(out,y,reduction='none')
                loss=((bce*mask).sum(1)/mask.sum(1).clamp_min(1)).mean()
                reachable=np.zeros_like(exists)
                for r,ww in enumerate(train['winners'][ii]):reachable[r,ww[ww>=0]]=True
                reach=torch.as_tensor(reachable,device=DEVICE)&mask
                pairs=(y[:,:,None]>y[:,None,:])&reach[:,:,None]&reach[:,None,:]
                rank=F.softplus(out[:,None,:]-out[:,:,None])
                active=pairs.sum((1,2))>0
                loss=loss+.3*((rank*pairs).sum((1,2))/pairs.sum((1,2)).clamp_min(1)*active).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(float(loss.detach()))
        if epoch in CHECKPOINTS or epoch==epochs:
            record=dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.perf_counter()-started)
            d.log(family,record)
            if checkpoint:checkpoint(net,record)
    return net


def train(tag):
    out=d.HERE/tag
    z=dict(np.load(out/'train_features.npz'));cross=dict(np.load(out/'cross_features.npz'))
    splits=d.split_queries(z['qid']);fit,tune,audit=[splits[x] for x in ['fit','tune','audit']]
    fit_utility=z['corr'][fit][:,:,d.EVAL-1].mean((0,1,2));k0=int(fit_utility.argmax())
    old_k0=json.loads((d.HERE.parent/'E13_query_tau'/tag/'selection.json').read_text())['k_fixed']
    tune_rows=(tune[:,None]*16+np.arange(16)).reshape(-1)
    audit_rows=(audit[:,None]*16+np.arange(16)).reshape(-1)
    base=float(z['corr'][audit][:,:,d.EVAL-1,k0].mean());picks={};records=[]
    for family in FAMILIES:
        selected=None;state=None
        def checkpoint(net,rec):
            nonlocal selected,state
            pred=scores(net,z,tune_rows)
            for threshold in THRESHOLDS:
                acc,bycount=evaluate_scores(pred,z,tune,k0,threshold,family)
                row=dict(family=family,threshold=threshold,tune_accuracy=acc,bycount=bycount,**rec)
                records.append(row)
                if selected is None or acc>selected['tune_accuracy']:
                    selected=row.copy();state=copy.deepcopy(net.state_dict())
        train_net(cross if family=='cross_pool' else z,fit,family,k0,60,checkpoint)
        net=AnswerNet(family);net.load_state_dict(state)
        pred=scores(net,z,audit_rows)
        acc,bycount=evaluate_scores(pred,z,audit,k0,selected['threshold'],family)
        selected.update(audit_accuracy=acc,audit_delta_pp=100*(acc-base),audit_bycount=bycount)
        picks[family]=selected
        torch.save(dict(family=family,state=state,k0=k0,selection=selected),out/f'{family}_fit.pt')
        d.log(tag,'AUDIT',selected)
    primary=max(picks,key=lambda f:picks[f]['audit_accuracy'])
    if picks[primary]['audit_accuracy']<=base:primary='fixed'
    selection=dict(primary=primary,k_fixed=k0,original_k_fixed=old_k0,tau_fixed=float(z['etas'][k0]),
        etas=[float(x) if np.isfinite(x) else 'inf' for x in z['etas']],audit_fixed_accuracy=base,
        selections=picks,candidates=records,seed=d.SEED,query_counts={k:len(v) for k,v in splits.items()},
        test_seen_during_selection=False,device=DEVICE)
    d.dump(out/'selection.json',selection)
    for family,sel in picks.items():
        net=train_net(cross if family=='cross_pool' else z,np.arange(len(z['qid'])),family,k0,sel['epoch'])
        torch.save(dict(family=family,state=net.state_dict(),k0=k0,selection=sel),out/f'{family}.pt')
    d.log(tag,'SELECTED',primary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);args=p.parse_args();train(args.cell)
