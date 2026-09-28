"""Independent source-prefix reconstruction and headline reconciliation."""
import hashlib
import json
import numpy as np
import setup20 as s
from check_kernel import reference


def main():
    checks = []
    details = {}
    for tag in ['qwen', 'llama']:
        raw = s.D.load(tag, 'test')
        halves = np.load(s.E17/tag/'tables.npz')['half_indices']
        source = np.load(s.HERE/tag/'raw_scores.npz')
        Q = len(raw['qid'])
        assert halves.shape == (4, 2, Q, 64)
        for p in range(4):
            np.testing.assert_array_equal(np.sort(halves[p].transpose(1,0,2).reshape(Q,128),axis=1),
                                          np.broadcast_to(np.arange(128),(Q,128)))
        ref_sc = None
        ref_cost = None
        details[tag] = {}
        for si, score in enumerate(s.SCORES):
            root = s.HERE/tag/score
            tables = np.load(root/'tables.npz')
            outcomes = np.load(root/'outcomes.npz')
            results = json.loads((root/'results.json').read_text())
            transforms = json.loads((root/'transforms.json').read_text())
            phi_raw = source['values'][si]
            for p in range(4):
                for side in [0,1]:
                    Aids = halves[p,side]
                    cdf = np.sort(phi_raw[np.arange(Q)[:,None], Aids].ravel())
                    assert transforms[2*p+side]['cdf_sha256'] == hashlib.sha256(cdf.tobytes()).hexdigest()
            # Include score quantization, early/middle/last queries, both source directions.
            for p,side,q in [(0,0,0),(1,1,37),(2,0,Q//2),(3,1,Q-1)]:
                cdf = np.sort(phi_raw[np.arange(Q)[:,None], halves[p,side]].ravel())
                phi = np.searchsorted(cdf,phi_raw[q],side='right')/(cdf.size+1.)
                ids = np.unique(raw['clu'][q])
                cl = np.searchsorted(ids,raw['clu'][q])
                none = int(np.searchsorted(ids,raw['none'][q])) if raw['none'][q] in ids else -1
                refs = []
                for target, key in [(side,'A'),(1-side,'B')]:
                    half = halves[p,target,q]
                    rng = np.random.default_rng([s.SEED,int(raw['qid'][q]),p,target,1])
                    perms = np.stack([half[rng.permutation(64)] for _ in range(s.S)])
                    expected = reference(phi,cl,none,raw['good'][q],perms,raw['etas'])
                    np.testing.assert_array_equal(expected,tables['V_'+key][:,p,side,q])
                    strict = reference(phi,cl,none,raw['strict_good'][q],perms,raw['etas'])
                    np.testing.assert_array_equal(strict,tables['V_strict_'+key][:,p,side,q])
                    if key == 'B':
                        # Same B path costs, independently reconstructed from original lengths.
                        costs = np.load(s.E18/tag/'replay_cost.npz')
                        np.testing.assert_allclose(raw['ell'][q][perms].cumsum(1).mean(0),
                                                   costs['mean_tokens'][p,target,q],rtol=1e-7)
                    refs.append(expected)
                    checks.append(dict(model=tag,score=score,part=p,side=side,query=int(raw['qid'][q]),target=key))
                for mi,mode in enumerate(s.MODES):
                    k = refs[0][mi].argmax(0)
                    np.testing.assert_array_equal(k,outcomes[mode+'__k'][2*p+side,q])
                    np.testing.assert_array_equal(refs[1][mi,k,np.arange(64)],
                        tables['V_B'][mi,p,side,q,k,np.arange(64)])
            for mi,mode in enumerate(s.MODES):
                # Independent reduce of all query/direction arms, not saved means.
                a = tables['V_A'][mi].astype(float)
                b = tables['V_B'][mi].astype(float)
                ki = np.argmax(a,axis=3)
                selected = np.take_along_axis(b,ki[:,:,:,None,:],axis=3).squeeze(3)
                np.testing.assert_array_equal(selected.mean((0,1)),outcomes[mode+'__adaptive'])
                kg = a.mean(2).argmax(2)
                glob = np.take_along_axis(b, np.broadcast_to(kg[:,:,None,None,:],(4,2,Q,1,64)),axis=3).squeeze(3)
                gap = 100*(selected-glob).mean((0,1,2))
                np.testing.assert_allclose(gap,results['modes'][mode]['contrasts']['A_global']['mean'],atol=1e-12)
            sc = outcomes['common_tie__sc']
            costs = results['B_mean_tokens']
            if ref_sc is not None:
                np.testing.assert_array_equal(sc,ref_sc)
                np.testing.assert_array_equal(costs,ref_cost)
            ref_sc, ref_cost = sc, costs
            for mode in s.MODES:
                for arm in ['adaptive','A_global','bon','B_global']:
                    np.testing.assert_array_equal(outcomes[mode+'__'+arm][:,0],sc[:,0])
            m = results['modes']['common_tie']
            k = outcomes['common_tie__k'][:,:,31]
            d = outcomes['common_tie__adaptive'][:,31]-outcomes['common_tie__sc'][:,31]
            details[tag][score] = dict(n32=dict(
                strict_gain_vs_sc_pp=m['strict_gain_vs_sc'][31],
                conflict_free_gain_vs_sc_pp=m['conflict_free_gain_vs_sc'][31],
                source_policy_sc_fraction=float((k==0).mean()),
                source_policy_bon_fraction=float((k==12).mean()),
                query_delta_positive=int((d>0).sum()),query_delta_negative=int((d<0).sum()),
                query_delta_zero=int((d==0).sum())))
    s.dump(s.HERE/'verification.json',dict(status='passed',
        independent_raw_prefix_reconstructions=len(checks),
        normalization_hashes_verified=64,source_and_target_disjoint=True,
        primary_sc_identical_across_scores=True,n1_all_arms_equal=True,
        chosen_temperatures_and_B_rewards_reconstructed=True,
        original_B_token_costs_reconstructed=True,all_headline_means_reconciled=True,
        checked_contexts=checks,diagnostics=details))
    print(json.dumps(details,indent=2))
    print('PASS:',len(checks),'raw-prefix contexts; 64 A-only transforms; full-arm reconciliation.')


if __name__ == '__main__':
    main()
