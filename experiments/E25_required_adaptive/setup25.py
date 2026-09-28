import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'E24_ptrue_committed'))
import setup24 as old
from setup24 import np,s
from run import load_arm
from models import predict
sys.path.insert(0,str(HERE))
E22=old.E22;E23=old.E23;E24=old.HERE;GAMMA=old.GAMMA
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def columns(tag):return old.columns(tag)

def query_temperature(V):
    """One query-dependent temperature, independent of price/budget/rollouts."""
    return np.asarray(V).mean(axis=2).argmax(axis=1).astype(np.uint8)

def query_locked_actions(V,logM,lambdas):
    k=query_temperature(V);chosen=V[np.arange(len(V)),k][:,None,:]
    n,_=s.E21.actions(chosen,logM[:,None]*np.arange(1,65),lambdas)
    return n,np.broadcast_to(k,n.shape).copy()
