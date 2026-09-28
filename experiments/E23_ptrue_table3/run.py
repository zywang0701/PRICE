import argparse,json,pickle,hashlib
import pandas as pd
from scipy.special import logsumexp
import setup23 as c
from setup23 import np,s
from baselines import one_query,cisc_winners,CISC_T_GRID,SC_N,BON_N


def select(tag):
    root=c.HERE/tag;root.mkdir(exist_ok=True);src=c.E22/tag
    cols=c.columns(tag);names=[p.stem for p in (src/'tune').glob('fixed_*.npz')]
    values={}
    for name in sorted(names):
        d=np.load(src/'tune'/(name+'.npz'))
        values[name]=[s.E21.read_point(d['C'].mean(1),d['R'].mean(1),b,c.GAMMA)['R'] for b in cols]
    fixed=[max(values,key=lambda n:values[n][j]) for j in range(6)]
    previous=json.loads((src/'selection.json').read_text())
    assert previous['primary']=='fixed_fallback'
    raw=s.D.load(tag,'train');scores=c.align(tag,'train',raw)
    tune_ids=set(json.loads((src/'query_splits.json').read_text())['tune']);curve=np.zeros(80);queries=0
    for i,qid in enumerate(raw['qid']):
        if int(qid) not in tune_ids:continue
        rng=np.random.default_rng([s.SEED,int(qid),1]);perms=np.stack([rng.permutation(64) for _ in range(16)])
        wins=cisc_winners(raw['clu'][i],raw['none'][i],perms,scores[i],CISC_T_GRID)
        curve+=s.D.score_winners(wins,raw['clu'][i],raw['good'][i]).mean((0,1));queries+=1
    curve/=queries;T=float(CISC_T_GRID[curve.argmax()])
    record=dict(budgets=cols.tolist(),fixed_arms=fixed,primary='fixed_fallback',adaptive_candidate=previous['adaptive_candidate'],
                fixed_tune_values=values,cisc_temperature=T,cisc_tune_curve=curve.tolist(),cisc_tune_queries=queries,
                model_sha256=previous['model_sha256'],cdf_sha256=previous['cdf_sha256'],test_used_for_selection=False)
    c.dump(root/'selection.json',record);c.log(tag,'calibration choices frozen',fixed,'CISC T',T)


def load_scores(tag,raw):
    cell=c.s.D.C.CELLS[tag]['cell'];base=pd.read_parquet(c.s.D.C.EXP+'/outputs/cells/'+cell+'/pools/pool.parquet')
    trace=pd.read_parquet(c.s.D.C.EXP+'/outputs/cells/'+cell+'/pools/conf_trace2.parquet')
    np.testing.assert_array_equal(base.query_id,trace.query_id)
    assert len(base)==64000 and not base.duplicated(['query_id','rollout_id']).any()
    base['lgc']=trace.lgc.to_numpy();base['bot10']=trace.c_bot10.to_numpy();base['tail']=-base.conf_tail
    keys=pd.MultiIndex.from_arrays([np.repeat(raw['qid'],128),raw['rid'].ravel()])
    aligned=base.set_index(['query_id','rollout_id']).loc[keys]
    np.testing.assert_array_equal(aligned.ell_tokens.to_numpy().reshape(raw['ell'].shape),raw['ell'])
    out={k:aligned[k].to_numpy(float).reshape(raw['ell'].shape) for k in ['lgc','bot10','tail','phi_conf']}
    for k in out:assert np.isfinite(out[k]).all(),k
    return out


def evaluate(tag):
    root=c.HERE/tag;src=c.E22/tag;sel=json.loads((root/'selection.json').read_text())
    assert hashlib.sha256((src/'models.pkl').read_bytes()).hexdigest()==sel['model_sha256']
    raw=s.D.load(tag,'test');scores=c.align(tag,'test',raw);native=dict(np.load(src/'test.npz'))
    replay=np.load(c.BASE/'E02_frozen'/tag/'replay.npz');np.testing.assert_array_equal(raw['qid'],replay['qid'])
    aux=load_scores(tag,raw);Q=len(raw['qid']);cor={};tok={};legacy_mismatch={};unexplained=0;legacy_bug_matches=0
    legacy_path=Path(c.FINAL)/tag/'baselines_replay.pkl'   # pre-P(True) baseline replay; cross-check only, skipped when absent
    legacy=pickle.load(legacy_path.open('rb')) if legacy_path.exists() else None
    if legacy is None:c.log(tag,'legacy baseline replay not found; skipping the cross-check against it')
    for i in range(Q):
        out=one_query(raw['clu'][i],raw['ell'][i],raw['none'][i],replay['perms'][i],scores[i],
                      aux['lgc'][i],aux['bot10'][i],aux['tail'][i],sel['cisc_temperature'])
        for key,(win,cost) in out.items():
            if key not in cor:cor[key]=np.empty((Q,64),bool);tok[key]=np.empty((Q,64))
            cor[key][i]=s.D.score_winners(win,raw['clu'][i],raw['good'][i]);tok[key][i]=cost
            if legacy is not None and key[0] in ['ac','ac_pub','esc','dcoff']:
                oldkey=(key[0],float(key[1]) if key[0] in ['ac','ac_pub'] else key[1])
                target=raw['old_target'][i];old=(win==target)&(target!=raw['none'][i])&(win!=-32768)
                mismatch=int(np.count_nonzero(old!=legacy['COR'][oldkey][i]));legacy_mismatch[key]=legacy_mismatch.get(key,0)+mismatch
                ids=np.unique(raw['clu'][i]);insertion=np.searchsorted(ids,target)
                if target in ids:unexplained+=mismatch
                elif insertion<len(ids):
                    buggy=(win==ids[insertion])&(win!=-32768)
                    legacy_bug_matches+=int(np.count_nonzero((old!=legacy['COR'][oldkey][i])&(buggy==legacy['COR'][oldkey][i])))
                np.testing.assert_array_equal(cost,legacy['TOK'][oldkey][i])
        if i%100==0:c.log(tag,'baselines',i,'/',Q)
    for family,ns,k in [('sc',SC_N,0),('bon',BON_N,12)]:
        for n in ns:cor[(family,str(n))]=native['corr'][:,:,n-1,k];tok[(family,str(n))]=native['tok'][:,:,n-1]
    families=['sc','bon','cisc','ac','esc','dcoff'];outdir=root/'baselines';outdir.mkdir(exist_ok=True)
    for family in families:
        keys=[k for k in cor if k[0]==family or (family=='ac' and k[0]=='ac_pub')]
        R=np.stack([cor[k].mean(1) for k in keys]);C=np.stack([np.exp(c.GAMMA*tok[k]).mean(1) for k in keys]);M=np.stack([tok[k].mean(1) for k in keys])
        assert np.isfinite(C).all()
        np.savez_compressed(outdir/(family+'.npz'),R=R,C=C,M=M,qid=raw['qid'],parameters=np.array([str(k) for k in keys]))
    assert unexplained==0 and legacy_bug_matches==sum(legacy_mismatch.values())
    c.dump(root/'baseline_replay_checks.json',dict(queries=Q,paths=64,original_score_independent_costs_exact=True,
        legacy_cross_check_performed=legacy is not None,original_score_independent_legacy_grading_mismatches=sum(legacy_mismatch.values()),
        all_legacy_mismatches_explained_by_absent_target_searchsorted_bug=True,
        mismatches_by_parameter={str(k):v for k,v in legacy_mismatch.items() if v},sc_bon_use_E22_frozen_replay=True))
    c.log(tag,'baseline replay done; legacy grading differences',sum(legacy_mismatch.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--phase',choices=['select','evaluate','all'],default='all');a=p.parse_args()
    if a.phase in ['select','all']:select(a.cell)
    if a.phase in ['evaluate','all']:evaluate(a.cell)
