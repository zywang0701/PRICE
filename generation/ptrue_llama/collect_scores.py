"""Validate completed Amarel artifacts and create isolated keyed score overrides."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
remote=ROOT/'remote';out=remote/'results_v3'
checks=[json.loads((out/f'complete_{i}.json').read_text()) for i in range(4)]
for c in checks:
    assert c['batch_check_mean_abs']<.005 and c['batch_check_max_abs']<.03
    assert c['dtype']=='float16' and c['train_and_test_rescored_with_same_implementation']
for field in ['model','revision','prompt_sha256','dtype','padding','logit','torch']:
    assert len({c[field] for c in checks})==1,field
reference=pd.read_parquet(out/'check_0.parquet').new_phi_conf.to_numpy();cross=[]
for i in range(4):
    diff=np.abs(pd.read_parquet(out/f'check_{i}.parquet').new_phi_conf.to_numpy()-reference)
    assert diff.mean()<.005 and diff.max()<.03
    cross.append(dict(shard=i,reference_shard=0,mean_abs=float(diff.mean()),max_abs=float(diff.max())))
parts=[pd.read_parquet(out/f'shard_{i}_complete.parquet') for i in range(4)]
for i,d in enumerate(parts):
    assert (d.unique_id%4==i).all() and len(d)==checks[i]['scored_unique_rows']
scores=pd.concat(parts,ignore_index=True)
unique=pd.read_parquet(remote/'llama_ptrue_unique_v2.parquet')
keys=pd.read_parquet(remote/'llama_ptrue_keys_v2.parquet')
assert len(scores)==len(unique)==117571 and not scores.unique_id.duplicated().any()
assert set(scores.unique_id)==set(unique.unique_id)
assert scores.phi_conf.between(0,1).all() and np.isfinite(scores.phi_conf).all()
assert not keys.duplicated(['corpus','query_id','rollout_id']).any()
joined=keys.merge(scores,on='unique_id',validate='many_to_one',how='left')
assert len(joined)==539200 and joined.phi_conf.notna().all()
dest=ROOT/'llama';dest.mkdir(exist_ok=True);counts={};hashes={}
(dest/'cross_gpu_verification.json').write_text(json.dumps(dict(status='passed',checks=cross),indent=2)+'\n')
for corpus,name,expected in [('train','calibration',475200),('test','evaluation',64000)]:
    d=joined.loc[joined.corpus==corpus,['query_id','rollout_id','phi_conf']].sort_values(['query_id','rollout_id'])
    assert len(d)==expected
    path=dest/f'{name}_ptrue.parquet';d.to_parquet(path,index=False)
    counts[name]=len(d);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
audit=dict(status='passed',score_version='E22 Amarel FP16 v3',job_id='61657780',unique_prompts=len(scores),
           keyed_rollouts=counts,sha256=hashes,checks=checks,
           legacy_cache_reproduced=False,legacy_pool_files_modified=False,
           cpu_gpu_scoring_seconds_sum=sum(c['elapsed_seconds'] for c in checks))
(dest/'score_provenance.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k!='checks'},indent=2))
