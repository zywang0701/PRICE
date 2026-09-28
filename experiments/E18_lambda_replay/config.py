import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
E17=HERE.parent/'E17_split_joint_oracle'
sys.path.insert(0,str(E17))
import settings as old
sys.path.insert(0,str(HERE))
D=old.D
GAMMA=old.GAMMA
SEED=old.SEED
SPLITS=old.SPLITS
S=old.PERMUTATIONS
T=old.T
LAM=np.r_[0.,np.geomspace(1e-180,1e6,16385)]
BUDGETS=np.array([2000.,3000.,5000.,8000.,12000.,20000.,30000.])
START=time.time()
def log(*a):print(f'[{time.time()-START:.1f}s]',*a,flush=True)
def dump(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

