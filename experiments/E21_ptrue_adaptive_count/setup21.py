import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
E20=BASE/'E20_nonprm_scores'
sys.path.insert(0,str(E20))
import setup20 as old
D=old.D;E17=old.E17;E18=old.E18
sys.path.insert(0,str(E18))
from price_sweep import actions,compact_indices,frontier,read_point,mix
GAMMA=old.GAMMA;SEED=old.SEED;BUDGETS=np.array([2000.,3000.,5000.,8000.,12000.,20000.,30000.])
MODES=old.MODES;START=time.time()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def log(*a):print(f'[{time.time()-START:.1f}s]',*a,flush=True)


def load(tag,mode):
    tab=dict(np.load(E20/tag/'ptrue/tables.npz'))
    original=dict(np.load(E17/tag/'tables.npz'))
    cost=dict(np.load(E18/tag/'replay_cost.npz'))
    np.testing.assert_array_equal(tab['qid'],original['qid'])
    np.testing.assert_array_equal(tab['qid'],cost['qid'])
    i=MODES.index(mode)
    return dict(A=tab['V_A'][i],B=tab['V_B'][i],strict=tab['V_strict_B'][i],
                logC_A=original['logC_plugin'],logC_B=cost['logC'],mean=cost['mean_tokens'],
                qid=tab['qid'],etas=tab['etas'],conflict=tab['conflict'])
