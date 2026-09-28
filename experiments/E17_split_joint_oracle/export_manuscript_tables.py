"""Table 2 and its uncertainty appendix on one split-pool, iid-plugin protocol.

The seven original Table 2 budget columns are retained. Source JSONs and cached
tables are read only; no deployment or manuscript file is changed by this script.
"""
import hashlib
import json
import numpy as np
import settings as s
from allocation import Frontier,evaluate

BS=np.array([2000,3000,5000,8000,12000,20000,30000])
TAGS=['qwen','llama']


def prepare():
    data={};sources={}
    for tag in TAGS:
        base=s.HERE/tag;result=json.loads((base/'results_plugin.json').read_text())
        ii=np.array([result['budgets'].index(float(b)) for b in BS]);tab=dict(np.load(base/'tables.npz'))
        r={name:{key:np.array(v)[ii].tolist() for key,v in obj.items() if isinstance(v,list)}
           for name,obj in result['methods'].items()}
        voting={name:{kind:{stat:np.array(v)[ii].tolist() for stat,v in ci.items()}
                      for kind,ci in obj.items()} for name,obj in result['equal_count_voting'].items()}
        exact=json.loads((base/'results_exact.json').read_text())
        exact_v={stat:np.array(v)[ii].tolist() for stat,v in
                 exact['equal_count_voting']['best_global_vote']['out_of_sample_pp'].items()}
        saved={}
        for name,k in [('sc',0),('bon',tab['V'].shape[3]-1)]:
            rin=[];rout=[];cin=[];cout=[]
            for part in range(s.SPLITS):
                for side in [0,1]:
                    V=tab['V'][part,side];lc=tab['logC_plugin'][part,side]
                    f=Frontier(V,lc,k);ri=[];ro=[];ca=[];cb=[]
                    for b in BS:
                        p=f.at(np.exp(s.GAMMA*b));assert p.status=='exact'
                        x,c,_=evaluate(p,V,lc,tab['mean_length'][part,side]);ri.append(x);ca.append(c)
                        x,c,_=evaluate(p,tab['V'][part,1-side],tab['logC_plugin'][part,1-side],
                                      tab['mean_length'][part,1-side]);ro.append(x);cb.append(c)
                    rin.append(ri);rout.append(ro);cin.append(ca);cout.append(cb)
            rin=np.array(rin);rout=np.array(rout);cin=np.array(cin);cout=np.array(cout)
            r[name]=dict(accuracy_in_pct=(100*rin.mean((0,2))).tolist(),
                accuracy_out_pct=(100*rout.mean((0,2))).tolist(),
                source_risk_budget=(np.log(cin.mean((0,2)))/s.GAMMA).tolist(),
                evaluation_risk_budget=(np.log(cout.mean((0,2)))/s.GAMMA).tolist())
            np.testing.assert_allclose(r[name]['source_risk_budget'],BS,rtol=1e-12)
            assert np.all(np.array(r[name]['accuracy_in_pct'])<=np.array(r['best_fixed']['accuracy_in_pct'])+1e-9)
            saved.update({name+'_'+key:val for key,val in dict(R_in=rin,R_out=rout,C_in=cin,C_out=cout).items()})
        np.savez_compressed(base/'table2_fixed_rule_outcomes.npz',**saved,budgets=BS)
        data[tag]=dict(methods=r,voting=voting,exact_same_count_gain=exact_v,
            queries=result['queries'],clean_queries=result['clean_queries'])
        sources[tag]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [base/'tables.npz',base/'results_plugin.json',base/'results_exact.json']}
    s.dump(s.HERE/'table2_data.json',dict(budgets=BS.tolist(),cost='iid_plugin',
        splits=s.SPLITS,directions=2*s.SPLITS,permutations=s.PERMUTATIONS,data=data,sources=sources))
    return data


def row(label,values,digits=1,bold=False,signed=False):
    def fmt(x):
        z=f'{x:+.{digits}f}' if signed else f'{x:.{digits}f}'
        return r'\textbf{'+z+'}' if bold else z
    return label+' & '+' & '.join(fmt(x) for x in values)+r' \\'


def main_table(data):
    def values(method,key,scale=1):return [x*scale for t in TAGS for x in data[t]['methods'][method][key]]
    def gain(kind,stat='mean'):return [x for t in TAGS for x in data[t]['voting']['best_global_vote'][kind][stat]]
    lines=[r'\begin{table}[!ht]',r'\centering',r'\resizebox{0.98\textwidth}{!}{%',
        r'\color{revisionblue}\setlength{\tabcolsep}{3.5pt}\begin{tabular}{@{}lccccccc@{\hspace{12pt}}ccccccc@{}}',
        r'\toprule',r' & \multicolumn{7}{c}{Qwen2.5-1.5B} & \multicolumn{7}{c}{Llama-3.2-3B} \\',
        r'\cmidrule(lr){2-8}\cmidrule(lr){9-15}',
        r'Readout\hfill selection budget $b_A$ $\rightarrow$ & 2k & 3k & 5k & 8k & 12k & 20k & 30k & 2k & 3k & 5k & 8k & 12k & 20k & 30k \\',
        r'\midrule',r'\multicolumn{15}{@{}l}{\emph{Selected and evaluated on A: accuracy (\%)}}\\[1pt]',
        row(r'SC ($\tau=0$)',values('sc','accuracy_in_pct')),
        row(r'BoN ($\tau=\infty$)',values('bon','accuracy_in_pct')),
        row(r'Best fixed $\tau$ (selected on A)',values('best_fixed','accuracy_in_pct')),
        row(r'\rowcolor{ourshl} PRICE-oracle',values('joint','accuracy_in_pct'),bold=True),
        r'\midrule',r'\multicolumn{15}{@{}l}{\emph{Selected on A and frozen on B: accuracy (\%) and actual risk cost}}\\[1pt]',
        row(r'Best fixed $\tau$: accuracy',values('best_fixed','accuracy_out_pct')),
        row(r'\quad actual $b_B$ (k tokens)',values('best_fixed','evaluation_risk_budget',.001)),
        row(r'\rowcolor{ourshl} PRICE-oracle: accuracy',values('joint','accuracy_out_pct')),
        row(r'\quad actual $b_B$ (k tokens)',values('joint','evaluation_risk_budget',.001)),
        r'\midrule',r'\multicolumn{15}{@{}l}{\emph{Voting gain at identical query-level counts (pp)}}\\[1pt]',
        row(r'PRICE $-$ best global vote, on A',gain('in_sample_pp'),digits=2,signed=True),
        row(r'PRICE $-$ best global vote, on B',gain('out_of_sample_pp'),digits=2,signed=True),
        r'\bottomrule',r'\end{tabular}}',
        r'\caption{\rev{PRICE-oracle within and across rollout pools (MATH-500, DeepConf score). Each query has disjoint halves A and B of $64$ rollouts; we average four splits in both directions. All methods choose the count per query on A. The best fixed temperature is one global value chosen on A separately at each budget. The column budget is the selection-side iid plug-in cost; the actual B costs can differ between methods. The last two rows instead keep PRICE-oracle\textquotesingle s exact count distribution and replace only its vote by the best global temperature at those counts, selected on A, so these voting gains have identical costs. Appendix~\ref{app:exp-split-pool} gives the protocol and confidence intervals.}}',
        r'\label{tab:oracle-vs-fixed}',r'\end{table}']
    # Standard LaTeX apostrophe avoids depending on a text-symbol package.
    return '\n'.join(lines).replace(r'\textquotesingle s',"'s")+'\n'


def appendix_table(data):
    lines=[r'\begin{table}[!ht]',r'\centering',r'\resizebox{0.98\textwidth}{!}{%',
        r'\color{revisionblue}\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}llrrrrrrr@{}}',
        r'\toprule',r' & & \multicolumn{7}{c}{selection-side plug-in budget $b_A$} \\',
        r'\cmidrule(lr){3-9}',r'Generator & readout & 2k & 3k & 5k & 8k & 12k & 20k & 30k \\',r'\midrule']
    for ti,tag in enumerate(TAGS):
        v=data[tag]['voting']['best_global_vote'];j=data[tag]['methods']['joint']
        label=['Qwen2.5-1.5B','Llama-3.2-3B'][ti]
        lines.extend([row(label+' & same-count voting gain on B (pp)',v['out_of_sample_pp']['mean'],3,False,True),
            row(r' & $95\%$ interval: lower',v['out_of_sample_pp']['lo'],3,False,True),
            row(r' & $95\%$ interval: upper',v['out_of_sample_pp']['hi'],3,False,True),
            row(r' & stricter grading: gain (pp)',v['strict_out_of_sample_pp']['mean'],3,False,True),
            row(r' & conflict-free queries: gain (pp)',v['clean_out_of_sample_pp']['mean'],3,False,True),
            row(r' & PRICE-oracle mean tokens on B (k)',np.array(j['evaluation_mean_tokens'])/1000,2),
            row(r' & PRICE-oracle risk-cost ratio $b_B/b_A$',j['risk_ratio'],2)])
        if ti==0:lines.append(r'\midrule')
    lines.extend([r'\bottomrule',r'\end{tabular}}',
        r'\caption{\rev{Uncertainty and cost diagnostics for Table~\ref{tab:oracle-vs-fixed}. Voting gains compare identical count distributions; the global vote is selected on A. Intervals use $1{,}000$ paired query-bootstrap resamples and are conditional on the fitted policies. Stricter grading rechecks the marked grading conflicts; the conflict-free subset excludes those queries. These checks do not change the selected actions. Mean tokens and entropic-risk costs are different readouts.}}',
        r'\label{tab:split-pool}',r'\end{table}'])
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    data=prepare()
    (s.HERE/'table2.tex').write_text(main_table(data))
    (s.HERE/'split_pool_appendix_table.tex').write_text(appendix_table(data))
    for tag in TAGS:
        j=data[tag]['methods']['joint'];v=data[tag]['voting']['best_global_vote']['out_of_sample_pp']
        print(tag,'optimism',np.array(j['accuracy_in_pct'])-j['accuracy_out_pct'],
              'cost ratio',j['risk_ratio'],'same-count vote',v,
              'exact-cost sensitivity',data[tag]['exact_same_count_gain'])
