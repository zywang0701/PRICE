import unittest
import numpy as np
import config as c
from price_sweep import actions,frontier,read_point,compact_indices

class PriceTests(unittest.TestCase):
    def test_direct_priced_argmax(self):
        rng=np.random.default_rng(196)
        for _ in range(12):
            V=rng.random((7,4,8));C=np.cumsum(rng.uniform(.1,2,(7,8)),axis=1)
            lam=np.r_[0.,np.geomspace(.0001,100,40)]
            for fixed in [None,0,2]:
                n,k=actions(V,np.log(C),lam,fixed)
                for j,l in enumerate(lam):
                    r=V.max(1) if fixed is None else V[:,fixed]
                    direct=(r-l*C).argmax(1)
                    np.testing.assert_array_equal(n[j],direct+1)
                    np.testing.assert_array_equal(k[j],V.argmax(1)[np.arange(7),direct] if fixed is None else np.full(7,fixed))

    def test_mgf_mixture_not_log_interpolation(self):
        costs=np.array([2.,4.,8.,12.,16.]);rewards=np.array([.2,.5,.4,.8,.7])
        np.testing.assert_array_equal(frontier(costs,rewards),[0,1,3])
        p=read_point(costs,rewards,np.log(8),1.)
        self.assertAlmostEqual(p['theta'],.5);self.assertAlmostEqual(p['R'],.65)
        self.assertAlmostEqual(p['b_actual'],np.log(8))

    def test_compaction_keeps_nested_grid_states(self):
        states=np.array([0,0,0,1,2,2,2,3,3,4,5,5,5,5])
        keep=compact_indices(np.r_[True,states[1:]!=states[:-1]])
        full_coarse=np.r_[0,np.arange(1,len(states),2)]
        kept_coarse=keep[(keep==0)|(keep%2==1)]
        np.testing.assert_array_equal(np.unique(states[full_coarse]),np.unique(states[kept_coarse]))
        np.testing.assert_array_equal(np.unique(states),np.unique(states[keep]))

if __name__=='__main__':unittest.main()
