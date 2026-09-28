import ast,argparse,json
from scipy.special import betainc
import setup23 as c
from setup23 import np,s
import baselines as B
from run import load_scores
from analyze import FAMILIES,read_many


def original_module(tag):
    path=c.BASE.parent/s.D.C.CELLS[tag]['lab']/'anytime/baselines64.py';tree=ast.parse(path.read_text());nodes=[]
    constants={'T','SC_N','BON_N','AC_THRESH','AC_CAP','AC_PUB','ESC_WL','ESC_PUB','CISC_N','CISC_T_GRID',
               'CISC_HOLD_FRAC','CISC_HOLD_SEED','DC_ETA','DC_TAU','N_INIT','DC_COMBOS','DC_KEY','DC_OFF_N','DC_OFF_ETA','DC_OFF_STATS','_M','STAB'}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            names={n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name)}
            if names<=constants:nodes.append(node)
        elif isinstance(node,ast.FunctionDef) and node.name in ['_top2','_abort_tokens','one_query']:nodes.append(node)
    module=ast.Module(body=nodes,type_ignores=[]);namespace={'np':np,'betainc':betainc};exec(compile(module,str(path),'exec'),namespace)
    return namespace


def verify(tag):
    root=c.HERE/tag;sel=json.loads((root/'selection.json').read_text());raw=s.D.load(tag,'test')
    scores=c.align(tag,'test',raw);aux=load_scores(tag,raw);perms=np.load(c.BASE/'E02_frozen'/tag/'replay.npz')['perms']
    reference=original_module(tag)
    for name in ['SC_N','BON_N','CISC_N','CISC_T_GRID','AC_THRESH','ESC_WL','DC_OFF_N','DC_OFF_ETA','DC_OFF_STATS']:
        np.testing.assert_array_equal(getattr(B,name),reference[name])
    missing=next(i for i in range(len(raw['qid'])) if raw['old_target'][i] not in raw['clu'][i]);checked=0
    for i in sorted(set([0,17,100,missing])):
        ids=np.unique(raw['clu'][i]);cid=np.searchsorted(ids,raw['clu'][i]);none=int(np.searchsorted(ids,raw['none'][i])) if raw['none'][i] in ids else -1
        tgt=int(np.searchsorted(ids,raw['old_target'][i])) if raw['old_target'][i] in ids else -1
        pp=perms[i,[0,3,41]]
        old=reference['one_query'](cid,raw['ell'][i],raw['phi'][i],scores[i],aux['lgc'][i],aux['bot10'][i],aux['tail'][i],
            [[] for _ in range(128)],[[] for _ in range(128)],tgt,none,len(ids),pp,sel['cisc_temperature'])
        new=B.one_query(raw['clu'][i],raw['ell'][i],raw['none'][i],pp,scores[i],aux['lgc'][i],aux['bot10'][i],aux['tail'][i],sel['cisc_temperature'])
        for (family,param),(win,tokens) in new.items():
            key=(family,float(param) if family in ['ac','ac_pub'] else int(param) if family=='cisc' else param)
            cor=(win==raw['old_target'][i])&(tgt>=0)&(win!=-32768)
            np.testing.assert_array_equal(cor,old[key][0]);np.testing.assert_array_equal(tokens,old[key][1]);checked+=len(pp)
    d=json.loads((root/'results.json').read_text());saved=np.load(root/'matched_queries.npz');points=json.loads((root/'readout_points.json').read_text());maxerr=0.
    for name,readouts in points.items():
        z=np.load(root/'baselines'/(name+'.npz')) if name in FAMILIES else np.load(c.E22/tag/'test'/(name+'.npz'))
        R=z['R'].mean(1);C=z['C'].mean(1)
        for j,p in enumerate(readouts):
            value=s.E21.mix(z['R'],p)
            np.testing.assert_allclose(value,saved[name+'__R'][j],atol=1e-12)
            np.testing.assert_allclose(value.mean(),read_many(C,R,c.columns(tag))[j],atol=1e-12)
            cost=s.E21.mix(z['C'],p).mean();risk=np.log(cost)/c.GAMMA
            assert risk<=c.columns(tag)[j]+1e-7
            if p['status']=='exact':maxerr=max(maxerr,abs(risk-c.columns(tag)[j]))
    base=np.stack([saved[n+'__R'].mean(1) for n in FAMILIES]);ours=saved['price__R'].mean(1)
    np.testing.assert_allclose(100*(ours-base.max(0)),d['gain_pp'],atol=1e-12)
    expected_CISC=float(B.CISC_T_GRID[np.argmax(sel['cisc_tune_curve'])]);assert sel['cisc_temperature']==expected_CISC
    expected_fixed=[max(sel['fixed_tune_values'],key=lambda n:sel['fixed_tune_values'][n][j]) for j in range(6)]
    assert sel['fixed_arms']==expected_fixed
    legacy=json.loads((root/'baseline_replay_checks.json').read_text());assert legacy['all_legacy_mismatches_explained_by_absent_target_searchsorted_bug']
    result=dict(status='passed',baseline_reference_path_checks=checked,original_parameter_grids_exact=True,
        source_legacy_cache_discrepancy_explained=True,selection_reconstructed_from_calibration=True,
        all_saved_frontier_mixtures_reconciled=True,matched_budget_max_error_tokens=float(maxerr),
        primary_gain_recomputed_from_query_arrays=True)
    c.dump(root/'verification.json',result);c.log(tag,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);verify(p.parse_args().cell)
