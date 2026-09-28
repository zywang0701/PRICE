"""Compare the actual observe-one-rollout API against frozen batch replay."""
import argparse
import numpy as np
import setup22 as s
from prepare import align
from controller import PTrueController
from evaluation import arms


def verify(tag):
    root=s.HERE/tag;raw=s.D.load(tag,'test');test=dict(np.load(root/'test.npz'))
    states=dict(np.load(root/'test_states.npz'));scores=align(tag,'test',raw)
    replay=np.load(s.BASE/'E02_frozen'/tag/'replay.npz');np.testing.assert_array_equal(replay['qid'],raw['qid'])
    policies={name:(g,k) for name,g,k in arms(states)}
    checked=0;max_gain_error=0.;max_cost_error=0.
    for i,path in [(0,0),(17,3)]:
        for name in ['regression_w1','regression_w4','conditional_w4','fixed_3_w4']:
            ctrl=PTrueController(root,test['H'][i],raw['etas'],name,raw['none'][i])
            g,k=policies[name]
            for n,r in enumerate(replay['perms'][i,path]):
                z=ctrl.observe(scores[i,r],raw['clu'][i,r],raw['ell'][i,r])
                assert z['temperature_index']==int(k[i,path,n]),(name,i,n,z['temperature_index'],int(k[i,path,n]))
                assert z['generated_tokens']==int(test['tok'][i,path,n])
                good=s.D.score_winners(np.array([z['answer_cluster']]),raw['clu'][i],raw['good'][i])[0]
                assert good==test['corr'][i,path,n,z['temperature_index']]
                if n<63:
                    err=abs(z['predicted_gain']-g[i,path,n]);max_gain_error=max(max_gain_error,err)
                    np.testing.assert_allclose(z['predicted_gain'],g[i,path,n],atol=2e-6,rtol=1e-4)
                err=abs(z['log_marginal_cost']-states['logcost'][i,path,n]);max_cost_error=max(max_cost_error,err)
                np.testing.assert_allclose(z['log_marginal_cost'],states['logcost'][i,path,n],atol=2e-6,rtol=1e-6)
                checked+=1
    report=dict(status='passed',observed_rollout_states=checked,temperature_and_reward_exact=True,
                max_gain_abs_error=float(max_gain_error),max_log_cost_abs_error=float(max_cost_error))
    s.dump(root/'streaming_verification.json',report);s.log(tag,report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);verify(p.parse_args().cell)
