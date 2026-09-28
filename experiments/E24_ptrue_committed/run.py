import argparse,json,pickle,hashlib
import setup24 as c
from setup24 import np,s
from models import train,predict


def phase_data(tag,phase):
    d=np.load(c.E22/tag/(phase+'.npz'))
    R=d['corr'].mean(1).transpose(0,2,1)
    C=np.exp(c.GAMMA*d['tok'].astype(float)).mean(1);M=d['tok'].mean(1)
    return dict(H=d['H'],R=R,C=C,M=M,qid=d['qid'])


def evaluate_phase(tag,phase,names=None):
    root=c.HERE/tag;out=root/phase;out.mkdir(exist_ok=True)
    with (root/'models.pkl').open('rb') as f:bundle=pickle.load(f)
    d=phase_data(tag,phase);V,logM=predict(bundle,d['H']);Q=len(logM);logC=logM[:,None]*np.arange(1,65)
    assert np.isfinite(np.exp(logC)).all()
    np.savez_compressed(out/'inputs.npz',V=V,logM=logM,R=d['R'],C=d['C'],M=d['M'],qid=d['qid'],H=d['H'])
    names=names or ['joint']+[f'fixed_{k}' for k in range(13)];summary={};cols=c.columns(tag)
    for name in names:
        n,k=s.E21.actions(V,logC,c.GRID,None if name=='joint' else int(name.split('_')[1]))
        changed=np.r_[True,np.any(n[1:]!=n[:-1],axis=1)|np.any(k[1:]!=k[:-1],axis=1)]
        keep=s.E21.compact_indices(changed);n=n[keep];k=k[keep];qi=np.arange(Q)[None,:]
        R=d['R'][qi,k,n-1];C=d['C'][qi,n-1];M=d['M'][qi,n-1]
        points=[s.E21.read_point(C.mean(1),R.mean(1),b,c.GAMMA) for b in cols]
        assert all(p['status']!='below_minimum' for p in points)
        summary[name]=points
        np.savez_compressed(out/(name+'.npz'),n=n,k=k,grid_indices=keep,log_grid=c.LOG_GRID[keep],
                            Rbar=R.mean(1),Cbar=C.mean(1),Mbar=M.mean(1),Nbar=n.mean(1),qid=d['qid'])
        c.log(tag,phase,name,[round(100*p['R'],3) for p in points])
    c.dump(out/'readouts.json',summary)
    if phase=='tune':
        fixed=[max([n for n in summary if n.startswith('fixed_')],key=lambda n:summary[n][j]['R']) for j in range(6)]
        gain=100*np.mean([summary['joint'][j]['R']-summary[n][j]['R'] for j,n in enumerate(fixed)])
        primary=['joint']*6 if gain>0 else fixed
        c.dump(root/'selection.json',dict(primary_arms=primary,fixed_arms=fixed,joint_mean_tune_gain_pp=gain,
            budgets=cols.tolist(),model_sha256=hashlib.sha256((root/'models.pkl').read_bytes()).hexdigest(),
            selection_uses_test=False,query_only=True))
        c.log(tag,'policy frozen',primary)


def load_arm(tag,phase,name):
    root=c.HERE/tag/phase;d=np.load(root/'inputs.npz');a=dict(np.load(root/(name+'.npz')));qi=np.arange(len(d['qid']))[None,:]
    return dict(a,R=d['R'][qi,a['k'],a['n']-1],C=d['C'][qi,a['n']-1],M=d['M'][qi,a['n']-1])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);a=p.parse_args();tag=a.cell
    train(tag);evaluate_phase(tag,'tune')
    selected=json.loads((c.HERE/tag/'selection.json').read_text());names=sorted(set(selected['primary_arms']+selected['fixed_arms']+['joint']))
    evaluate_phase(tag,'audit',names);evaluate_phase(tag,'test',names)
