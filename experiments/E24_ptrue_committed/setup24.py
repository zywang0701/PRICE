import os,sys,json,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;E22=BASE/'E22_ptrue_deployed';E23=BASE/'E23_ptrue_table3'
sys.path.insert(0,str(E22))
import setup22 as s
sys.path.insert(0,str(HERE))
import numpy as np
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
GAMMA=s.GAMMA;START=time.time();LOG_GRID=np.r_[-np.inf,np.linspace(-450,6,16385)]
GRID=np.exp(LOG_GRID)
FAMILIES=['sc','bon','cisc','ac','esc','dcoff']
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def log(*x):print(f'[{time.time()-START:.1f}s]',*x,flush=True)
def columns(tag):return np.array(s.D.C.table3_columns(tag)['mgf'])
