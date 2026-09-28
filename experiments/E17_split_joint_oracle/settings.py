import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import json
import sys
import time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'E16_answer_sets'))
import data as D
sys.path.insert(0,str(HERE))
C=D.C
GAMMA=.0029262
SEED=20260922
SPLITS=4
PERMUTATIONS=256
T=64
START=time.time()


def log(*args):print(f'[{time.time()-START:.1f}s]',*args,flush=True)


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def budgets(tag):
    columns=np.array(C.table3_columns(tag)['mgf'])
    return np.unique(np.r_[np.geomspace(1500,60000,61),columns,[2000,3000,5000,8000,12000,20000,30000]]),columns
