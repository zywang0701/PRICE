import argparse
import hashlib
import json
import numpy as np
import settings as s
from allocation import Frontier,Policy,evaluate,common_frontier,expand_common

METHODS=['joint','best_fixed','cal_fixed_count','common_cal_fixed','common_sc',
         'joint_counts_best_vote','joint_counts_cal_vote']


def select_policies(V,logC,length,k0,bs):
    joint=Frontier(V,logC);fixed=[Frontier(V,logC,k) for k in range(V.shape[1])]
    common=common_frontier(V,logC,k0);sc=common_frontier(V,logC,0)
    selected={name:[] for name in METHODS};best_ks=[];joint_ks=[]
    for b in bs:
        B=np.exp(s.GAMMA*b);p=joint.at(B);per_k=[f.at(B) for f in fixed]
        kb=int(np.argmax([z.source_reward for z in per_k]));bf=per_k[kb]
        pc=expand_common(common.at(B),len(V));psc=expand_common(sc.at(B),len(V))
        assert p.source_reward+1e-10>=bf.source_reward
        assert bf.source_reward+1e-10>=per_k[k0].source_reward
        assert per_k[k0].source_reward+1e-10>=pc.source_reward
        rr=[evaluate(p,V,logC,length,override=k)[0].mean() for k in range(V.shape[1])]
        kg=int(np.argmax(rr));best_ks.append(kb);joint_ks.append(kg)
        def override(k):
            ks=np.full(len(V),k,int)
            return Policy(ks,p.n_low.copy(),ks.copy(),p.n_high.copy(),p.theta,float(rr[k]),p.source_cost,p.status)
        row=dict(joint=p,best_fixed=bf,cal_fixed_count=per_k[k0],common_cal_fixed=pc,common_sc=psc,
            joint_counts_best_vote=override(kg),joint_counts_cal_vote=override(k0))
        for name,policy in row.items():
            if policy.status=='exact':assert abs(policy.source_cost/B-1)<1e-9
            selected[name].append(policy)
    return selected,np.array(best_ks),np.array(joint_ks)


def bootstrap_samples(q,seed=s.SEED,n=1000):
    rng=np.random.default_rng(seed)
    return rng.multinomial(q,np.full(q,1/q),size=n)/q


def ci(x,weights):
    # x is budget x query; folds have already been averaged within each query.
    v=weights@x.T
    return dict(mean=(x.mean(-1)).tolist(),lo=np.quantile(v,.025,axis=0).tolist(),hi=np.quantile(v,.975,axis=0).tolist())


def summarize(Rin,Rout,Cin,Cout,mean_in,mean_out,strict,raw,bs,columns,names):
    Q=Rin.shape[-1];weights=bootstrap_samples(Q);clean=~raw['conflict'];wclean=bootstrap_samples(int(clean.sum()))
    result={};base=names.index('joint_counts_best_vote');joint=names.index('joint');cal=names.index('joint_counts_cal_vote')
    for j,name in enumerate(names):
        ri=Rin[j].mean(0);ro=Rout[j].mean(0);ca=Cin[j].mean(0);cb=Cout[j].mean(0)
        in_ci=ci(ri,weights);out_ci=ci(ro,weights)
        rb=np.log(cb.mean(-1))/s.GAMMA;ra=np.log(ca.mean(-1))/s.GAMMA
        boot_b=np.log(weights@cb.T)/s.GAMMA
        result[name]=dict(accuracy_in_pct=(100*ri.mean(-1)).tolist(),accuracy_out_pct=(100*ro.mean(-1)).tolist(),
            out_ci_lo_pct=(100*np.array(out_ci['lo'])).tolist(),out_ci_hi_pct=(100*np.array(out_ci['hi'])).tolist(),
            source_risk_budget=ra.tolist(),evaluation_risk_budget=rb.tolist(),
            evaluation_risk_ci_lo=np.quantile(boot_b,.025,axis=0).tolist(),evaluation_risk_ci_hi=np.quantile(boot_b,.975,axis=0).tolist(),
            source_mean_tokens=mean_in[j].mean((0,2)).tolist(),evaluation_mean_tokens=mean_out[j].mean((0,2)).tolist(),
            optimism_pp={k:(100*np.array(v)).tolist() for k,v in ci(ri-ro,weights).items()},
            risk_ratio=(rb/ra).tolist())
    voting={}
    for label,j in [('best_global_vote',base),('calibration_fixed_vote',cal)]:
        np.testing.assert_allclose(Cin[joint],Cin[j],rtol=0,atol=0)
        np.testing.assert_allclose(Cout[joint],Cout[j],rtol=0,atol=0)
        np.testing.assert_allclose(mean_out[joint],mean_out[j],rtol=0,atol=0)
        delta_in=(Rin[joint]-Rin[j]).mean(0);delta_out=(Rout[joint]-Rout[j]).mean(0)
        assert delta_in.mean(-1).min()>-1e-10
        voting[label]=dict(in_sample_pp={k:(100*np.array(v)).tolist() for k,v in ci(delta_in,weights).items()},
            out_of_sample_pp={k:(100*np.array(v)).tolist() for k,v in ci(delta_out,weights).items()},
            clean_out_of_sample_pp={k:(100*np.array(v)).tolist() for k,v in ci(delta_out[:,clean],wclean).items()},
            strict_out_of_sample_pp={k:(100*np.array(v)).tolist() for k,v in ci((strict[joint]-strict[j]).mean(0),weights).items()})
    return dict(budgets=bs.tolist(),column_indices=[int(np.flatnonzero(bs==b)[0]) for b in columns],
        paper_columns=columns.tolist(),methods=result,equal_count_voting=voting,
        queries=Q,clean_queries=int(clean.sum()),bootstrap_replicates=1000)


def run(tag,cost_kind):
    out=s.HERE/tag;raw=s.D.load(tag,'test');table=dict(np.load(out/'tables.npz'))
    source=json.loads((out/'source.json').read_text())
    assert hashlib.sha256(__import__('pathlib').Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    k0=s.D.k_fixed(tag)
    bs,columns=s.budgets(tag);F=2*s.SPLITS;Q=len(raw['qid']);M=len(METHODS);N=len(bs)
    shape=(M,F,N,Q)
    Rin=np.empty(shape);Rout=np.empty(shape);Cin=np.empty(shape);Cout=np.empty(shape)
    mean_in=np.empty(shape);mean_out=np.empty(shape);strict=np.empty(shape)
    actions={key:np.empty(shape,dtype=np.int16) for key in ['k_low','n_low','k_high','n_high']}
    theta=np.empty((M,F,N));status=np.empty((M,F,N),np.int8)
    best_ks=[];joint_ks=[]
    for part in range(s.SPLITS):
        for side in [0,1]:
            direction=2*part+side;V=table['V'][part,side];Ve=table['V'][part,1-side]
            Vs=table['V_strict'][part,1-side]
            C=table['logC_'+cost_kind][part,side];Ce=table['logC_'+cost_kind][part,1-side]
            L=table['mean_length'][part,side];Le=table['mean_length'][part,1-side]
            selected,ks,jks=select_policies(V,C,L,k0,bs);best_ks.append(ks);joint_ks.append(jks)
            for mi,name in enumerate(METHODS):
                for b,policy in enumerate(selected[name]):
                    Rin[mi,direction,b],Cin[mi,direction,b],mean_in[mi,direction,b]=evaluate(policy,V,C,L)
                    Rout[mi,direction,b],Cout[mi,direction,b],mean_out[mi,direction,b]=evaluate(policy,Ve,Ce,Le)
                    strict[mi,direction,b]=evaluate(policy,Vs,Ce,Le)[0]
                    for key in actions:actions[key][mi,direction,b]=getattr(policy,key)
                    theta[mi,direction,b]=policy.theta
                    status[mi,direction,b]={'exact':0,'below_minimum':-1,'saturated':1}[policy.status]
            s.log(tag,cost_kind,'evaluated frozen direction',direction+1,'/',F)
    result=summarize(Rin,Rout,Cin,Cout,mean_in,mean_out,strict,raw,bs,columns,METHODS)
    full=dict(np.load(out/'full_tables.npz'));fsel,_,_=select_policies(full['V'],full['logC_'+cost_kind],full['mean_length'],k0,bs)
    result['full_pool']={}
    for name in ['joint','best_fixed','cal_fixed_count','common_cal_fixed','common_sc']:
        rr=[evaluate(p,full['V'],full['logC_'+cost_kind],full['mean_length']) for p in fsel[name]]
        result['full_pool'][name]=dict(accuracy_pct=[float(100*r[0].mean()) for r in rr],
            risk_budget=[float(np.log(r[1].mean())/s.GAMMA) for r in rr],mean_tokens=[float(r[2].mean()) for r in rr])
    result.update(cost_kind=cost_kind,k_fixed=k0,gamma=s.GAMMA,split_count=s.SPLITS,directions=F,permutations=s.PERMUTATIONS,
        status_counts={name:{str(v):int((status[i]==v).sum()) for v in [-1,0,1]} for i,name in enumerate(METHODS)},
        best_fixed_indices=np.array(best_ks).tolist(),best_same_count_vote_indices=np.array(joint_ks).tolist(),
        source_sha256=source['sha256'])
    s.dump(out/f'results_{cost_kind}.json',result)
    np.savez_compressed(out/f'outcomes_{cost_kind}.npz',R_in=Rin,R_out=Rout,C_in=Cin,C_out=Cout,
        mean_in=mean_in,mean_out=mean_out,R_strict=strict,budgets=bs,methods=np.array(METHODS),qid=raw['qid'])
    np.savez_compressed(out/f'policies_{cost_kind}.npz',**actions,theta=theta,status=status,budgets=bs,methods=np.array(METHODS),qid=raw['qid'])
    cols=result['column_indices'];joint=result['methods']['joint'];v=result['equal_count_voting']['best_global_vote']['out_of_sample_pp']['mean']
    s.log(tag,cost_kind,'JOINT out',[round(joint['accuracy_out_pct'][i],2) for i in cols],
        'B budgets',[round(joint['evaluation_risk_budget'][i]) for i in cols],
        'same-count voting pp',[round(v[i],3) for i in cols])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--cost',choices=['exact','plugin'],default='exact')
    a=p.parse_args();run(a.cell,a.cost)
