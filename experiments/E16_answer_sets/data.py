"""Isolated scoring repair; no gold-dependent changes to voting clusters."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '3')
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C
sys.path.insert(0, str(HERE.parent/'E13_query_tau'))
import experiment as old
# Keep this experiment's public modules ahead of older controllers with the same name.
sys.path.insert(0, str(HERE))
SEED = 20260921
CP = np.array([1, 2, 4, 8, 16, 32, 48, 64])
EVAL = np.array([2, 4, 8, 16, 32, 64])
START = time.time()


def log(*args):
    print(f'[{time.time()-START:.1f}s]', *args, flush=True)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


@lru_cache(maxsize=100000)
def parsed(s):
    from math_verify import parse
    return parse('$'+str(s).strip().strip('$')+'$', parsing_timeout=1)


@lru_cache(maxsize=100000)
def grade(answer, gold, digits=6):
    from math_verify import verify
    if answer == '<none>':
        return False
    try:
        result = bool(verify(parsed(gold), parsed(answer), float_rounding=digits, timeout_seconds=1))
    except Exception:
        result = False
    from awv.answers import canonicalize
    return result or canonicalize(answer) == canonicalize(gold)


def split_queries(qids):
    order = np.random.default_rng(SEED).permutation(len(qids))
    return dict(zip(['fit', 'tune', 'audit'], [np.sort(x) for x in
        np.split(order, [int(.7*len(qids)), int(.85*len(qids))])]))


def load(tag, phase):
    return dict(np.load(HERE/tag/f'{phase}_data.npz'))


def k_fixed(tag):
    """Calibration-fixed temperature index: this experiment's selection when it was run, else the recorded constant."""
    sel = HERE/tag/'selection.json'
    if sel.exists():
        return json.loads(sel.read_text())['k_fixed']
    return json.loads((Path(C.EXP)/'config'/'calibration_fixed_temperature.json').read_text())[tag]['k_fixed']


def prepare(tag, phase):
    out = HERE/tag; out.mkdir(exist_ok=True)
    w = C.CELLS[tag]; cell = w['train'] if phase == 'train' else w['cell']
    root = Path(C.EXP)/'outputs/cells'/cell
    columns = ['query_id', 'rollout_id', 'answer_canonical', 'correct', 'phi_deepconf2', 'ell_tokens']
    pool = pd.read_parquet(root/'pools/pool.parquet', columns=columns)
    clusters = pd.read_parquet(root/('lp3' if phase == 'train' else 'lp')/'clusters.parquet')
    pool = pool.merge(clusters, on=['query_id', 'rollout_id'], validate='one_to_one')
    meta_path = root/'pools/queries.parquet'
    if not meta_path.exists():
        assert phase == 'train' and tag == 'qwen'
        meta_path = Path(C.EXP)/'outputs/cells'/C.CELLS['llama']['train']/'pools/queries.parquet'
    meta = pd.read_parquet(meta_path).set_index('query_id')
    if phase == 'train':
        prep = np.load(root/'lp3/prep_ov__phi_deepconf2.npz')
        qids = prep['query_id']; etas = prep['etas']
    else:
        replay = np.load(HERE.parent/'E02_frozen'/tag/'replay.npz')
        qids = replay['qid']; etas = replay['etas']
    assert len(set(qids)-set(meta.index)) == 0
    train_pool = pd.read_parquet(Path(C.EXP)/'outputs/cells'/w['train']/'pools/pool.parquet', columns=['phi_deepconf2'])
    cdf = np.sort(np.nan_to_num(train_pool.phi_deepconf2.to_numpy(float), nan=0))
    groups = dict(tuple(pool.sort_values(['query_id','rollout_id']).groupby('query_id')))
    arrays = {k: [] for k in ['phi','clu','ell','rid','good','strict_good','raw_good']}
    none, target, conflict, changes = [], [], [], []
    alignment = []
    for i,qid in enumerate(qids):
        g=groups[int(qid)]; gold=str(meta.loc[int(qid),'gold_answer'])
        cl=g.cluster.to_numpy(int); raw=g.correct.to_numpy(bool)
        tg=int(g.target_cluster.iloc[0]); no=int(g.none_cluster.iloc[0])
        mismatch=bool(np.any(raw != ((cl==tg)&(cl!=no))))
        good=np.zeros(len(g),bool); strict=np.zeros(len(g),bool)
        details=[]
        for a,gg in g.groupby('cluster'):
            rep=min(gg.answer_canonical.astype(str)); mask=cl==a
            is_good = grade(rep,gold) if mismatch else bool(gg.correct.iloc[0])
            is_strict = grade(rep,gold,12) if mismatch else is_good
            is_good = is_good and a != no; is_strict = is_strict and a != no
            good[mask]=is_good; strict[mask]=is_strict
            if mismatch:
                details.append(dict(cluster=int(a), representative=rep, raw_correct=int(gg.correct.sum()),
                    count=len(gg), repaired_correct=is_good, stricter_correct=is_strict))
        if phase=='train' and i%31==0 and raw.any():
            ans=str(g.loc[g.correct,'answer_canonical'].iloc[0])
            alignment.append(dict(query_id=int(qid), answer=ans,gold=gold,matches=grade(ans,gold)))
        if mismatch:
            changes.append(dict(query_id=int(qid),gold=gold,old_target=tg,clusters=details,
                changed_rollouts=int(np.sum(good != ((cl==tg)&(cl!=no)))),
                differs_from_raw=int(np.sum(good != raw))))
        for key,value in dict(phi=np.searchsorted(cdf,np.nan_to_num(g.phi_deepconf2.to_numpy(float),nan=0),side='right')/(len(cdf)+1),
                clu=cl,ell=g.ell_tokens.to_numpy(int),rid=g.rollout_id.to_numpy(int),good=good,
                strict_good=strict,raw_good=raw).items():
            arrays[key].append(value)
        none.append(no);target.append(tg);conflict.append(mismatch)
        if i%1000==0:log(tag,phase,'scoring',i,'/',len(qids))
    if alignment:
        assert np.mean([a['matches'] for a in alignment])>.98, 'Shared query metadata fails alignment check'
    d={k:np.stack(v) for k,v in arrays.items()}
    d.update(qid=qids,etas=etas,none=np.array(none),old_target=np.array(target),conflict=np.array(conflict))
    d['clu']=d['clu'].astype(np.int16);d['ell']=d['ell'].astype(np.int32)
    np.savez_compressed(out/f'{phase}_data.npz',**d)
    summary=dict(queries=len(qids),rollouts_per_query=d['phi'].shape[1],conflicted_queries=int(d['conflict'].sum()),
        queries_with_multiple_correct_clusters=int(sum(len(np.unique(c[g]))>1 for c,g in zip(d['clu'],d['good']))),
        old_correct_rollouts=int(np.sum((d['clu']==d['old_target'][:,None])&(d['clu']!=d['none'][:,None]))),
        repaired_correct_rollouts=int(d['good'].sum()),changed_vs_raw=int(np.sum(d['good']!=d['raw_good'])),
        stricter_changed_rollouts=int(np.sum(d['good']!=d['strict_good'])),
        absent_correct_queries=int(np.sum(~d['good'].any(1))),metadata_source=str(meta_path),
        metadata_alignment=alignment,changes=changes)
    dump(out/f'{phase}_scoring_audit.json',summary)
    if phase=='train':dump(out/'query_splits.json',{k:qids[v].tolist() for k,v in split_queries(qids).items()})
    log(tag,phase,'ready',len(qids),'queries; conflicts',summary['conflicted_queries'])


def votes(phi,clu,none,etas,perms):
    """Existing tie behavior, with label-free local cluster IDs."""
    _,winners,tokens=old.prefix_features(phi,clu,np.ones(len(phi),int),none,perms,etas)
    return winners


def score_winners(winners,clu,good):
    return np.isin(winners,np.unique(clu[good])) & (winners!=-32768)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',choices=['qwen','llama'],required=True)
    p.add_argument('--phase',choices=['train','test'],default='train');args=p.parse_args()
    prepare(args.cell,args.phase)
