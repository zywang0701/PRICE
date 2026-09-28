import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE/'E18_lambda_replay'))
import config as old
sys.path.insert(0,str(HERE))
D=old.D
E17=BASE/'E17_split_joint_oracle'
E18=BASE/'E18_lambda_replay'
SEED=old.SEED;S=256;N=64;K=13;GAMMA=old.GAMMA
SCORES={'deepconf':'phi_deepconf2','ptrue':'phi_conf','selfcert':'phi_selfcert','likelihood':'phi_lik'}
NAMES={'deepconf':'DeepConf','ptrue':'P(True)','selfcert':'Self-certainty','likelihood':'Likelihood'}
MODES=['common_tie','score_native_tie']
COUNTS=[4,8,16,32,64]
START=time.time()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def log(*a):print(f'[{time.time()-START:.1f}s]',*a,flush=True)
