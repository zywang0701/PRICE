"""Independent calibration-choice and published-table reconciliation."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import setup22 as s

rows=list(csv.DictReader((s.HERE/'summary.csv').open()));assert len(rows)==14
checked=0
for tag in ['qwen','llama']:
    root=s.HERE/tag;sel=json.loads((root/'selection.json').read_text())
    tune=json.loads((root/'tune/readouts.json').read_text());model=json.loads((root/'model_training.json').read_text())
    names=[n for n in tune if not n.startswith('fixed_')]
    best=max(names,key=lambda n:np.mean([p['R'] for p in tune[n]]))
    fixed=[max([n for n in tune if n.startswith('fixed_')],key=lambda n:tune[n][j]['R']) for j in range(7)]
    delta=100*(np.mean([p['R'] for p in tune[best]])-np.mean([tune[n][j]['R'] for j,n in enumerate(fixed)]))
    assert best==sel['adaptive_candidate'] and fixed==sel['fixed_arms_by_budget']
    assert sel['primary']==(best if delta>0 else 'fixed_fallback')
    np.testing.assert_allclose(delta,sel['tune_gain_pp'],atol=1e-12)
    assert model['curve_selected']==min(model['curve_candidates'],key=lambda z:z['mse'])['name']
    assert model['classifier_selected']==max(model['classifier_candidates'],key=lambda z:(z['accuracy'],z['penalty'],-z['leaves']))
    arrays=np.load(root/'matched_queries.npz');d=json.loads((root/'results.json').read_text())
    for j,row in enumerate([r for r in rows if r['model']==tag]):
        gain=100*(arrays[best+'__R'][j]-arrays['fixed__R'][j]).mean()
        same=100*(arrays['same_fixed_stops__R'][j]-arrays['fixed__R'][j]).mean()
        np.testing.assert_allclose(float(row['gain_pp']),gain,atol=1e-12)
        np.testing.assert_allclose(float(row['same_stop_gain_pp']),same,atol=1e-12)
        np.testing.assert_allclose(float(row['candidate_accuracy_pct']),100*arrays[best+'__R'][j].mean(),atol=1e-12)
        np.testing.assert_allclose(float(row['fixed_accuracy_pct']),100*arrays['fixed__R'][j].mean(),atol=1e-12)
        assert row['selected_deployment']==sel['primary'] and float(row['risk_budget'])==s.BUDGETS[j]
        checked+=1
paper=Path(s.D.C.PAPER)
for ext in ['png','pdf']:
    assert (s.HERE/f'ptrue_deployed_gain.{ext}').read_bytes()==(paper/'images'/f'fig_ptrue_deployed_gain.{ext}').read_bytes()
manifest=json.loads((s.HERE/'artifact_manifest.json').read_text())
for name,digest in manifest.items():assert hashlib.sha256((s.HERE/name).read_bytes()).hexdigest()==digest
report=dict(status='passed',published_rows_reconciled=checked,all_selections_reconstructed_from_calibration=True,
            figure_copies_identical=True,manifest_hashes_verified=True)
s.dump(s.HERE/'artifact_validation.json',report);print(json.dumps(report,indent=2))
