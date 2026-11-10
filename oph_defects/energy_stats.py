"""Energy bookkeeping of the bare law (tests T1 and T2 of the theory session).

    python -m oph_defects.energy_stats --level 4

T1  On a canonical (balanced) background every attempt has contrast |d| <= 1. An attempt with |d| = 1 moves the
    unit with probability 1/2 and releases nothing; attempts with |d| >= 2 are relaxation events releasing
    floor(d^2 / 4). Reported: the joint counts of (|d|, moved) and the release by |d|, for the balanced state and
    along a quench from independent loads 0..5.
T2  At integer filling a raised port (+1) and a lowered port (-1) are the excitations. Their number can only fall,
    a +1 and a -1 on one seam annihilate, and the total release equals the number of annihilated pairs exactly.
    Reported: pairs against sweeps, cumulative release, and the decay exponent of the pair density.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import fedgeom as FG
from .universe import COSTS, Engine, prepare_loads, schedule_seed


def run(geo, loads, seed, sweeps, base=None):
    eng = Engine(geo, loads, seed, COSTS['binding'], 5, False, slots=2, register=False, fast_rng=True)
    rows = []
    for s in range(sweeps):
        seq, coin = eng.draw()
        hist = np.zeros((6, 2), np.int64); rel = np.zeros(6, np.int64)
        for lo in range(0, eng.S, eng.chunk):
            n = eng.run_chunk(seq[lo:lo + eng.chunk], coin[lo:lo + eng.chunk], lo, s)
            con = np.abs(eng.o_con[:n].astype(np.int64)); cur = eng.o_cur[:n].astype(np.int64)
            c5 = np.minimum(con, 5)
            np.add.at(hist, (c5, (cur != 0).astype(np.int64)), 1)
            np.add.at(rel, c5, cur * (eng.o_con[:n].astype(np.int64) - cur))
        row = {'sweep': s + 1, 'hist': hist.tolist(), 'release_by_contrast': rel.tolist()}
        if base is not None:
            x = eng.x.astype(np.int64) - base
            row['plus'] = int((x > 0).sum()); row['minus'] = int((x < 0).sum()); row['max_abs'] = int(np.abs(x).max())
        rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--level', type=int, default=4)
    p.add_argument('--sweeps', type=int, default=400)
    p.add_argument('--json', type=Path, default=None)
    a = p.parse_args(argv)
    geo = FG.build(a.level)
    out = {'level': a.level}
    # T1 balanced
    loads, _ = prepare_loads('canonical:1/3:1', a.level, geo.ports)
    rows = run(geo, loads, schedule_seed(a.level, 71), 40)
    H = np.sum([r['hist'] for r in rows], axis=0); R = np.sum([r['release_by_contrast'] for r in rows], axis=0)
    out['balanced'] = {'hist': H.tolist(), 'release_by_contrast': R.tolist()}
    print(f"T1 balanced state, filling 1/3, 40 sweeps: attempts with |d| = 0: {H[0].sum():,}; |d| = 1: {H[1].sum():,} of which moved {H[1, 1]:,} "
          f"(fraction {H[1, 1] / H[1].sum():.5f}); |d| >= 2: {H[2:].sum()}; total release {R.sum()}")
    # T1 quench
    loads, _ = prepare_loads('exact06', a.level, geo.ports)
    rows = run(geo, loads, schedule_seed(a.level, 0), 60)
    out['quench'] = rows
    print('T1 quench from independent loads 0..5: per sweep, attempts by contrast |d| = 1, 2, 3, 4, 5+ and the moved fraction at |d| = 1; release / sum floor(d^2/4)')
    for r in rows[:6] + rows[9::10]:
        H = np.array(r['hist']); R = np.array(r['release_by_contrast'])
        pred = int(sum(H[d].sum() * (d * d // 4) for d in range(2, 5)))
        print(f"   sweep {r['sweep']:3d}: {H[1].sum():>9,} {H[2].sum():>9,} {H[3].sum():>8,} {H[4].sum():>7,} {H[5].sum():>6,}   moved at |d|=1: {H[1, 1] / max(1, H[1].sum()):.4f}   "
              f"release {int(R.sum()):>9,} (contrasts 2..4 alone predict {pred:,}; all moved at |d|>=2: {bool((H[2:, 0] == 0).all())})")
    # T2 integer filling
    rng = np.random.default_rng(5)
    P = geo.ports
    k = P // 5
    idx = rng.permutation(P)
    loads = np.full(P, 2, np.int8); loads[idx[:k]] = 3; loads[idx[k:2 * k]] = 1
    rows = run(geo, loads, schedule_seed(a.level, 72), a.sweeps, base=2)
    rel = np.cumsum([sum(r['release_by_contrast']) for r in rows])
    pairs = np.array([r['plus'] for r in rows])
    ok = all(r['plus'] == r['minus'] and r['max_abs'] <= 1 for r in rows)
    out['integer_filling'] = {'initial_pairs': k, 'pairs': pairs.tolist(), 'release_cumulative': rel.tolist()}
    t = np.arange(1, a.sweeps + 1)
    late = (t >= 20) & (pairs > 50)
    slope = float(np.polyfit(np.log(t[late]), np.log(pairs[late] / P), 1)[0]) if late.sum() > 5 else float('nan')
    print(f"T2 integer filling, {k:,} raised and {k:,} lowered ports on {P:,}: equal numbers and no load beyond +-1 at every sweep: {ok}; "
          f"pairs left after 1, 10, 100, {a.sweeps} sweeps: {pairs[0]:,}, {pairs[9]:,}, {pairs[min(99, a.sweeps - 1)]:,}, {pairs[-1]:,}")
    print(f"   annihilated pairs {k - pairs[-1]:,}, total release {int(rel[-1]):,}: ratio {rel[-1] / max(1, k - pairs[-1]):.6f}; "
          f"every sweep exact: {bool((rel == k - pairs).all())}; pair density decays as t^{slope:.3f} for t >= 20")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out))


if __name__ == '__main__':
    main()
