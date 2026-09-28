"""Incremental voting selector; the caller retains its existing stopping rule.

Inputs are observed answer-cluster IDs and scores transformed with the training
CDF. No query gold, future samples, target budget, or reference labels are used.
Cluster assignment itself remains the caller's responsibility.
"""
import json
import numpy as np
import torch
import data as d
import features as f
import model as m


class AnswerSetSelector:
    def __init__(self,tag,family=None,none_cluster=-1):
        root=d.HERE/tag;self.selection=json.loads((root/'selection.json').read_text())
        self.family=family or self.selection['primary'];self.none=none_cluster
        self.etas=np.array([np.inf if x=='inf' else x for x in self.selection['etas']])
        self.k0=self.selection['k_fixed'];self.net=None
        if self.family!='fixed':
            pack=torch.load(root/f'{self.family}.pt',map_location='cpu',weights_only=False)
            self.net=m.AnswerNet(self.family).eval();self.net.load_state_dict(pack['state'])
            self.threshold=self.selection['selections'][self.family]['threshold']
        self.reset()

    def reset(self):
        self.phi=[];self.clu=[];self.ell=[];self.k=self.k0

    def observe(self,score,answer_cluster,token_count):
        if len(self.phi)>=64:raise ValueError('This selector was calibrated for at most 64 rollouts.')
        if not np.isfinite(score) or not 0<=score<=1:raise ValueError('Expected a finite training-CDF score in [0,1].')
        if token_count<0:raise ValueError('token_count must be nonnegative.')
        self.phi.append(float(score));self.clu.append(int(answer_cluster));self.ell.append(int(token_count))
        n=len(self.phi)
        if self.net is not None and n in d.CP and not (self.family=='cross_pool' and n>32):
            r=f.state(np.array(self.phi),np.array(self.clu),np.array(self.ell),self.none,self.etas)
            z=f.pack([r]);v=m.scores(self.net,z,np.array([0]))
            self.k=int(m.choose(v,self.k0,self.threshold)[0])
        return dict(index=self.k,temperature=float(self.etas[self.k]),rollouts=n)
