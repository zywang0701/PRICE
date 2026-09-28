"""Checks that guard against answer-label leakage and replay/evaluation errors."""
import unittest
import numpy as np
import torch
import data as d
import features as f
import model as m


class Invariants(unittest.TestCase):
    def test_multi_correct_and_absence(self):
        self.assertEqual(d.score_winners(np.array([0,1,2,-32768]),np.array([0,1,2]),
            np.array([True,True,False])).tolist(),[True,True,False,False])
        self.assertFalse(d.score_winners(np.array([0,1,-32768]),np.array([0,1]),np.array([False,False])).any())

    def test_grading_format_variants(self):
        self.assertTrue(d.grade('78','78'));self.assertTrue(d.grade('$78','78'))
        self.assertTrue(d.grade('5','5'));self.assertFalse(d.grade('49','34'))

    def test_state_votes_match_replay(self):
        rng=np.random.default_rng(42);etas=np.r_[np.linspace(0,30,12),np.inf]
        for case in range(8):
            ph=rng.random(64);cl=rng.integers(0,8,64);ell=rng.integers(10,100,64)
            if case==1:cl[:]=7
            if case==2:ph[:]=.5
            if case==3:cl=np.arange(64)%2;ph[:]=.5
            w=d.votes(ph,cl,7,etas,np.arange(64)[None])[0]
            for n in d.CP:
                x,ctx,win,ids=f.state(ph[:n],cl[:n],ell[:n],7,etas)
                ans=np.where(win>=0,ids[np.maximum(win,0)],-32768)
                np.testing.assert_array_equal(ans,w[n-1])
                self.assertTrue(np.isfinite(x).all());self.assertTrue(np.isfinite(ctx).all())

    def test_future_does_not_enter_state(self):
        rng=np.random.default_rng(40);ph=rng.random(64);cl=rng.integers(0,5,64);ell=np.ones(64)
        eta=np.r_[np.linspace(0,25,12),np.inf]
        before=f.state(ph[:8],cl[:8],ell[:8],4,eta)
        ph[8:]=100;cl[8:]=99;ell[8:]=100000
        after=f.state(ph[:8],cl[:8],ell[:8],4,eta)
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)

    def test_answer_order_equivariance(self):
        torch.manual_seed(2);x=torch.rand(3,5,57);ctx=torch.rand(3,12)
        mask=torch.tensor([[True]*5,[True,True,False,False,False],[False]*5])
        perm=torch.tensor([3,0,4,1,2])
        for family in m.FAMILIES:
            net=m.AnswerNet(family).eval();a=net(x,ctx,mask);b=net(x[:,perm],ctx,mask[:,perm])
            expected=a if family=='cross_pool' else a[:,perm]
            torch.testing.assert_close(b,expected,atol=1e-6,rtol=1e-5)

    def test_query_splits_and_fallback(self):
        split=d.split_queries(np.arange(7425))
        self.assertEqual(len(set(np.concatenate(list(split.values())))),7425)
        self.assertFalse(set(split['fit'])&set(split['audit']))
        values=np.ones((5,13));np.testing.assert_array_equal(m.choose(values,4,0),np.full(5,4))


if __name__=='__main__':unittest.main()
