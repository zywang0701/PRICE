import argparse,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import setup20 as s


def prepare(tag):
    raw=s.D.load(tag,'test');cell=s.D.C.CELLS[tag]['cell']
    path=Path(s.D.C.EXP)/'outputs/cells'/cell/'pools/pool.parquet'
    pool=pd.read_parquet(path,columns=['query_id','rollout_id','ell_tokens',*s.SCORES.values()])
    assert not pool.duplicated(['query_id','rollout_id']).any()
    indexed=pool.set_index(['query_id','rollout_id'])
    keys=pd.MultiIndex.from_arrays([np.repeat(raw['qid'],128),raw['rid'].reshape(-1)],names=['query_id','rollout_id'])
    aligned=indexed.loc[keys]
    np.testing.assert_array_equal(aligned.ell_tokens.to_numpy().reshape(-1,128),raw['ell'])
    scores=np.stack([aligned[col].to_numpy(float).reshape(-1,128) for col in s.SCORES.values()])
    assert np.isfinite(scores).all()
    out=s.HERE/tag;out.mkdir(exist_ok=True)
    np.savez_compressed(out/'raw_scores.npz',values=scores,qid=raw['qid'],rid=raw['rid'],names=np.array(list(s.SCORES)))
    s.dump(out/'source.json',dict(pool_path=str(path),pool_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cached_labels=str(s.D.HERE/tag/'test_data.npz'),labels_sha256=hashlib.sha256((s.D.HERE/tag/'test_data.npz').read_bytes()).hexdigest(),
        queries=len(raw['qid']),rollouts_per_query=128,score_columns=list(s.SCORES.values()),
        missing_values=0,query_rollout_alignment=True,token_lengths_match=True))
    s.log(tag,'prepared',scores.shape)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);prepare(p.parse_args().cell)
