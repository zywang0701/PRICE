import itertools
import unittest
import numpy as np
from scipy.optimize import linprog
from allocation import Frontier,evaluate,finite_pool_logcost,common_frontier,expand_common


class AllocationTests(unittest.TestCase):
    def test_exact_without_replacement_cost(self):
        lengths=np.array([2,3,5,11,17]);gamma=.08
        actual=np.exp(finite_pool_logcost(lengths,gamma,len(lengths)))
        expected=[np.mean([np.exp(gamma*lengths[list(c)].sum()) for c in itertools.combinations(range(5),n)]) for n in range(1,6)]
        np.testing.assert_allclose(actual,expected,rtol=1e-13)
        self.assertAlmostEqual(np.log(actual[-1]),gamma*lengths.sum(),places=12)
        plugin=np.exp(np.arange(1,6)*np.log(np.exp(gamma*lengths).mean()))
        self.assertTrue(np.all(actual<=plugin*(1+1e-12)))

    def test_against_independent_linear_program(self):
        rng=np.random.default_rng(26)
        for case in range(12):
            Q,K,T=4,3,6
            V=rng.random((Q,K,T)) # deliberately nonmonotonic reward curves
            logc=np.log(np.cumsum(rng.uniform(.3,2,(Q,T)),axis=1)+1)
            for temperature in [None,0,2]:
                obj=V if temperature is None else V[:,temperature:temperature+1]
                k=obj.shape[1];frontier=Frontier(V,logc,temperature)
                cost=np.broadcast_to(np.exp(logc)[:,None],obj.shape).reshape(-1)/Q
                eq=np.kron(np.eye(Q),np.ones((1,k*T)))
                for fraction in [.01,.25,.75,.99]:
                    B=(1-fraction)*np.exp(logc[:,0]).mean()+fraction*np.exp(logc[:,-1]).mean()
                    lp=linprog(-obj.reshape(-1)/Q,A_ub=cost[None],b_ub=[B],A_eq=eq,b_eq=np.ones(Q),bounds=(0,None),method='highs')
                    self.assertTrue(lp.success)
                    p=frontier.at(B);R,C,_=evaluate(p,V,logc,np.ones(Q))
                    self.assertAlmostEqual(R.mean(),-lp.fun,places=9)
                    self.assertLessEqual(C.mean(),B*(1+1e-10))
                    if p.status=='exact':self.assertAlmostEqual(C.mean(),B,places=9)

    def test_dominance_common_count_and_cost_identity(self):
        rng=np.random.default_rng(12);V=rng.random((6,4,8));logc=np.log(np.arange(1,9)[None,:]*np.arange(1,7)[:,None])
        joint=Frontier(V,logc);fixed=Frontier(V,logc,2);common=common_frontier(V,logc,2)
        for B in [4,7,15,25]:
            pj,pf,pc=joint.at(B),fixed.at(B),expand_common(common.at(B),6)
            self.assertGreaterEqual(pj.source_reward+1e-12,pf.source_reward)
            self.assertGreaterEqual(pf.source_reward+1e-12,pc.source_reward)
            _,cj,_=evaluate(pj,V,logc,np.ones(6));_,cv,_=evaluate(pj,V,logc,np.ones(6),override=2)
            np.testing.assert_array_equal(cj,cv)

    def test_endpoints_absent_reward_and_extreme_cost(self):
        V=np.zeros((3,2,64));logc=np.arange(1,65)[None,:]*np.array([.1,2.,5.])[:,None]
        f=Frontier(V,logc)
        low=f.at(1);high=f.at(1e140)
        self.assertEqual(low.status,'below_minimum');self.assertEqual(high.status,'saturated')
        np.testing.assert_array_equal(high.n_low,np.ones(3))
        self.assertTrue(np.isfinite(high.source_cost))


if __name__=='__main__':unittest.main()
