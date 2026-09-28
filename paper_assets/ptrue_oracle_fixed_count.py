"""Add fixed-count SC/BoN controls on the exact E21 split-pool replay.

Run before ptrue_paper_assets.py. No score, pool, tie or adaptive-policy changes.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'experiments'
OUT = ROOT / 'E27_ptrue_oracle_fixed_count'
sys.path.insert(0, str(ROOT / 'E21_ptrue_adaptive_count'))
import setup21 as s


def hull(c, r):
    keep = []
    for i in sorted(range(len(c)), key=lambda i: (c[i], -r[i])):
        if keep and (c[i] == c[keep[-1]] or r[i] <= r[keep[-1]]):
            continue
        while len(keep) > 1:
            a, b = keep[-2:]
            if (r[b]-r[a])/(c[b]-c[a]) > (r[i]-r[b])/(c[i]-c[b]):
                break
            keep.pop()
        keep.append(i)
    return np.array(keep)


def readout(raw, budget):
    c, r = raw['C_B'].mean(1), raw['R_B'].mean(1)
    h = hull(c, r); target = np.exp(s.GAMMA*budget)
    assert target >= c[h[0]]
    j = np.searchsorted(c[h], target)
    if j == len(h):
        lo = hi = int(h[-1]); theta = 0.
    else:
        lo, hi = int(h[max(0,j-1)]), int(h[j])
        theta = float((target-c[lo])/(c[hi]-c[lo])) if lo != hi else 0.
    risk = float(np.log((1-theta)*c[lo]+theta*c[hi])/s.GAMMA)
    assert risk <= budget+1e-8 and 0 <= theta <= 1
    accuracy = float(100*((1-theta)*r[lo]+theta*r[hi]))
    # Independent interpolation check against the existing E21 frontier utility.
    old = s.read_point(c, r, budget, s.GAMMA)
    np.testing.assert_allclose([accuracy, risk], [100*old['R'], old['b_actual']], atol=1e-9)
    return dict(budget=float(budget), b_actual=risk, accuracy_B_pct=accuracy,
                lo=lo, hi=hi, theta=theta, count_lo=lo+1, count_hi=hi+1,
                mean_n=float((1-theta)*(lo+1)+theta*(hi+1)))


def main():
    OUT.mkdir(exist_ok=True)
    for tag in ['qwen', 'llama']:
        t = s.load(tag, 'common_tie'); dest = OUT/tag; dest.mkdir(exist_ok=True)
        oldpath = ROOT/'E21_ptrue_adaptive_count'/tag/'common_tie/results.json'
        old = json.loads(oldpath.read_text())
        # B rewards already reverse the side index; aggregate both directions.
        c = np.exp(t['logC_B']).mean((0, 1)).T
        m = t['mean'].mean((0, 1)).T
        assert c.shape == (64, len(t['qid']))
        methods, contrasts = {}, {}
        for method, k in [('sc',0), ('bon',12)]:
            r = t['B'][:,:,:,k,:].mean((0,1), dtype=np.float64).T
            # Loop independently over the eight A->B directions.
            rr = np.zeros_like(r); cc = np.zeros_like(c)
            for part in range(4):
                for side in [0,1]:
                    rr += t['B'][part,side,:,k,:].T.astype(float)/8
                    cc += np.exp(t['logC_B'][part,1-side]).T/8
            np.testing.assert_allclose(r, rr, atol=1e-12)
            np.testing.assert_allclose(c, cc, rtol=1e-12)
            raw = dict(R_B=r, C_B=c, mean_B=m, qid=t['qid'], counts=np.arange(1,65))
            np.savez_compressed(dest/(method+'_fixed_count.npz'), **raw)
            points = [readout(raw, b) for b in old['budgets']]
            methods[method] = points
            adaptive = np.array([p['accuracy_B_pct'] for p in old['methods'][method]])
            joint = np.array([p['accuracy_B_pct'] for p in old['methods']['joint']])
            fixed = np.array([p['accuracy_B_pct'] for p in points])
            contrasts[method+'_count_gain_pp'] = (adaptive-fixed).tolist()
            contrasts[method+'_rule_gain_pp'] = (joint-adaptive).tolist()
            np.testing.assert_allclose((adaptive-fixed)+(joint-adaptive),joint-fixed)
            print(tag, method, 'fixed', fixed.round(2), 'count gain', (adaptive-fixed).round(2), flush=True)
        # 2026-09-17: fixed-count sweeps of every temperature, for the hindsight-best fixed-tau row
        # of Table 2 (read as swept in ptrue_paper_assets.py, no hull).
        R_all = np.stack([t['B'][:, :, :, k, :].mean((0, 1), dtype=np.float64).T for k in range(t['B'].shape[3])])
        np.savez_compressed(dest/'all_fixed_count.npz', R_B=R_all, C_B=c, qid=t['qid'], counts=np.arange(1, 65))
        print(tag, 'all_fixed_count.npz', R_all.shape, flush=True)
        sources = [s.E20/tag/'ptrue/tables.npz', s.E17/tag/'tables.npz', s.E18/tag/'replay_cost.npz', oldpath]
        data = dict(model=tag, queries=len(t['qid']), gamma=s.GAMMA, budgets=old['budgets'],
                    protocol='E21 common_tie; four 64+64 partitions, both directions; counts 1..64; retrospective MGF mixtures',
                    methods=methods, contrasts=contrasts, status='passed',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    protocol_sha256=hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest(),
                    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        (dest/'results.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__ == '__main__':
    main()
