import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/'E16_answer_sets'))
import data as D
sys.path.insert(0,str(BASE/'E20_nonprm_scores'))
from kernel import reward_tables
sys.path.insert(0,str(BASE/'E21_ptrue_adaptive_count'))
import setup21 as E21
sys.path.insert(0,str(HERE))
CP=np.array([1,2,4,8,16,32,48,64]);SEED=20260923;GAMMA=.0029262
BUDGETS=E21.BUDGETS;START=time.time()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def log(*a):print(f'[{time.time()-START:.1f}s]',*a,flush=True)
def splits(n):
    o=np.random.default_rng(SEED).permutation(n)
    return dict(zip(['fit','tune','audit'],[np.sort(a) for a in np.split(o,[int(.7*n),int(.85*n)])]))
