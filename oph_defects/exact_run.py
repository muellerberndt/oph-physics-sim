"""Law C on the exact federation, recorded sweep by sweep.

    python -m oph_defects.exact_run --level 8 --sweeps 300 --out runs/lawC_L8

This is ``oph_exact.federation`` as it is: the twelve-port carriers on the
geodesic tower, the declared ``port_pair`` gluing, the declared initial
loads in ``{0..5}``, the uniform asynchronous seam schedule, and the integer
nearest-agreement law with a fair tie coin (law C: coin + current). The only
additions are recording: the descent functional ``V`` after every sweep, the
mismatch potential, the number of seams whose endpoints still differ, and a
per-carrier mean-load snapshot every ``--snapshot-every`` sweeps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from oph_exact import federation as F


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--level', type=int, default=8)
    p.add_argument('--sweeps', type=int, default=300)
    p.add_argument('--schedule', type=int, default=0, help='schedule index (seed per oph_exact convention)')
    p.add_argument('--snapshot-every', type=int, default=10)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fed = F.build_federation(a.level, 'port_pair')
    loads = F.initial_loads(a.level, fed.ports)
    seed = F.schedule_seed(a.level, 'port_pair', a.schedule)
    print(f'level {a.level}: {fed.carriers} carriers, {fed.ports} ports, {fed.seams} seams, built in {time.time() - t0:.0f}s', flush=True)
    engine = F.resolve_engine('native', fed.ports)
    move = F.INTEGER_ENGINES[engine]
    x = np.asarray(loads, dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    v = int(np.dot(x, x))
    v_min = fed.integer_minimum_descent(loads)
    tally = F.IntegerTally()
    sa, sb = fed.seam_a, fed.seam_b
    series = {'V': [v], 'phi': [], 'unsettled_seams': [], 'descents': [], 'swaps': [], 'waits': []}
    snaps = []
    d = x[sa] - x[sb]
    series['phi'].append(int(np.dot(d, d)))
    series['unsettled_seams'].append(int(np.count_nonzero(d)))
    for s in range(a.sweeps):
        seq = rng.integers(0, fed.seams, size=fed.seams, dtype=np.int64)
        coin = rng.integers(0, 2, size=fed.seams, dtype=np.int64)
        before = (tally.descents, tally.swaps, tally.waits)
        v, first = move(x, fed.graph, seq, coin, v, v_min, tally)
        d = x[sa] - x[sb]
        series['V'].append(int(v))
        series['phi'].append(int(np.dot(d, d)))
        series['unsettled_seams'].append(int(np.count_nonzero(d)))
        series['descents'].append(tally.descents - before[0])
        series['swaps'].append(tally.swaps - before[1])
        series['waits'].append(tally.waits - before[2])
        if s % a.snapshot_every == 0 or s == a.sweeps - 1:
            mean = x.reshape(fed.carriers, 12).mean(axis=1).astype(np.float32)
            spread = (x.reshape(fed.carriers, 12).max(axis=1) - x.reshape(fed.carriers, 12).min(axis=1)).astype(np.int8)
            name = f'carrier_{s:05d}.npz'
            np.savez_compressed(a.out / name, mean=mean, spread=spread)
            snaps.append({'sweep': s, 'file': name})
        if s % 10 == 0 or s == a.sweeps - 1:
            print(f'sweep {s}: V={v} (min {v_min}) phi={series["phi"][-1]} unsettled={series["unsettled_seams"][-1]} '
                  f'descents={series["descents"][-1]} swaps={series["swaps"][-1]} elapsed={time.time() - t0:.0f}s', flush=True)
        if first >= 0:
            print(f'balanced class reached at sweep {s}', flush=True)
            break
    np.save(a.out / 'final_loads.npy', x.astype(np.int8))
    (a.out / 'run.json').write_text(json.dumps({
        'law': 'integer nearest agreement with fair tie coin (law C), oph_exact.federation native kernel',
        'level': a.level, 'carriers': fed.carriers, 'ports': fed.ports, 'seams': fed.seams,
        'inter_seams': fed.inter_count, 'schedule': 'uniform over seams, one sweep = |S| attempts',
        'seed': seed, 'initial_loads': 'oph_exact.federation.initial_loads (seeded {0..5})',
        'sweeps': len(series['descents']), 'V_min': v_min, 'series': series, 'snapshots': snaps,
        'runtime_seconds': round(time.time() - t0, 1)}))
    print('done', flush=True)


if __name__ == '__main__':
    main()
