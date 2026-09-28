import os,sys,json,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;E22=BASE/'E22_ptrue_deployed'
sys.path.insert(0,str(E22))
import setup22 as s
from prepare import align
sys.path.insert(0,str(HERE))
import numpy as np
GAMMA=s.GAMMA;START=time.time()
def log(*x):print(f'[{time.time()-START:.1f}s]',*x,flush=True)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def columns(tag):return np.array(s.D.C.table3_columns(tag)['mgf'])
