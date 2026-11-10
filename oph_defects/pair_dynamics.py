"""Phase 2 of the closed-loop law: charges that move (declared extension).

    python -m oph_defects.pair_dynamics --level 5 --seeds 200 --sweeps 2000 --out runs/closed_loop/L5_pairs --workers 30

Law C with transported contrast and the unit-twist rule of the theory session: at an attempt on a glue seam with
contrast d != 0, the seam offset moves by sign(d) when the seam potential it releases, |d| - 1/2, exceeds the change
of the cost of the two plaquette holonomies; otherwise the ordinary repair runs. A charge is a plaquette of nonzero
holonomy; a twist moves, creates, fuses or annihilates charges.

Each run prepares charges on the density-1/3 canonical background at the rounded stationary heights of the pinned
configuration, holds them pinned for a short burn-in, then releases them and follows the two marked charges:

    opposite   +1 at A, -1 at B, separation r0;
    equal      +1 at A and +1 at B, separation r0, with two -1 compensators near the antipode;
    single     +1 at A with its -1 compensator near the antipode (the diffusion control).

Recorded per sweep: the separation of the marked charges in mesh-edge units, until one of them ends
(annihilation, fusion, split or exchange) or the run ends.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from . import fedgeom as FG
from . import kernels as K
from .closed_loop import Mesh, dipole_offsets, rounded_heights
from .universe import COSTS, Engine, prepare_loads, schedule_seed

_CACHE = {}


def _setup(level: int):
    """Per-process cache: geometry, mesh, incidence and the factorised port Laplacian."""
    if level not in _CACHE:
        geo = FG.build(level)
        mesh = Mesh(geo)
        a, b = geo.all_seam_ends()
        S, P = geo.seams, geo.ports
        B = sp.csr_matrix((np.concatenate([np.ones(S), -np.ones(S)]), (np.concatenate([np.arange(S), np.arange(S)]), np.concatenate([a, b]))), shape=(S, P))
        lu = spl.splu((B.T @ B).tocsc()[1:, 1:])
        v_plus = int(geo.tri[4 ** level - 1].max())
        d_plus = mesh.distances(v_plus)
        anti = int(np.argmin(geo.mesh_xyz @ geo.mesh_xyz[v_plus]))
        d_anti = mesh.distances(anti)
        nb = [w for w in mesh.nbr[v_plus]]
        edge = float(np.mean([np.arccos(np.clip(geo.mesh_xyz[v_plus] @ geo.mesh_xyz[w], -1, 1)) for w in nb]))
        _CACHE[level] = dict(geo=geo, mesh=mesh, B=B, lu=lu, v_plus=v_plus, d_plus=d_plus, anti=anti, d_anti=d_anti, edge=edge)
    return _CACHE[level]


def heights(env, off: np.ndarray) -> np.ndarray:
    rhs = env['B'].T @ off.astype(np.float64)
    x = np.zeros(env['geo'].ports)
    x[1:] = env['lu'].solve(rhs[1:])
    return x - x.mean()


def run_one(spec: dict) -> dict:
    t0 = time.time()
    level, kind, r0, seed = spec['level'], spec['kind'], spec['r0'], spec['seed']
    env = _setup(level)
    geo, mesh = env['geo'], env['mesh']
    A = env['v_plus']
    ring = np.flatnonzero(env['d_plus'] == r0)
    Bv = int(ring[seed % ring.size])                              # the partner direction varies with the seed
    far = np.flatnonzero(env['d_anti'] <= 2)
    off = np.zeros(geo.seams, np.int64)
    if kind == 'opposite':
        o, _ = dipole_offsets(mesh, A, Bv, 1); off += o
        marked = [(A, 1), (Bv, -1)]
    elif kind == 'equal':
        o, _ = dipole_offsets(mesh, A, int(far[0]), 1); off += o
        o, _ = dipole_offsets(mesh, Bv, int(far[-1]), 1); off += o
        marked = [(A, 1), (Bv, 1)]
    else:
        o, _ = dipole_offsets(mesh, A, int(far[0]), 1); off += o
        marked = [(A, 1)]
    off = off.astype(np.int8)
    loads, _ = prepare_loads(f'canonical:1/3:{seed}', level, geo.ports)
    loads = (loads.astype(np.int64) + rounded_heights(heights(env, off))).astype(np.int8)
    eng = Engine(geo, loads, schedule_seed(level, 1000 + seed), COSTS[spec.get('cost', 'binding')], spec.get('kappa', 5), False, slots=2,
                 register=False, offsets=off, twists=True, fast_rng=True)
    FI = geo.intra_faces
    ids = [int(eng.tracker.face_id[v]) for v, _ in marked]
    assert all(i >= 0 for i in ids) and [int(eng.tracker.cols['cls'][i].astype(np.int8)) for i in ids] == [q for _, q in marked]
    flags_on = eng.flags
    eng.flags = flags_on & ~K.F_TWIST; eng.ctx.flags = eng.flags
    for s in range(spec['burn']):
        seq, coin = eng.draw()
        for lo in range(0, eng.S, eng.chunk):
            eng.run_chunk(seq[lo:lo + eng.chunk], coin[lo:lo + eng.chunk], lo, s)
    eng.flags = flags_on; eng.ctx.flags = flags_on
    xyz = geo.mesh_xyz
    cols = eng.tracker.cols
    sweeps = spec['sweeps']
    sep = np.full(sweeps + 1, np.nan, np.float32)
    disp = np.full(sweeps + 1, np.nan, np.float32)                # displacement of charge A from its start, edge units
    def positions():
        return [int(cols['last_face'][i]) - FI for i in ids]
    def measure(t):
        pos = positions()
        disp[t] = np.arccos(np.clip(xyz[pos[0]] @ xyz[A], -1, 1)) / env['edge']
        if len(pos) == 2:
            sep[t] = np.arccos(np.clip(xyz[pos[0]] @ xyz[pos[1]], -1, 1)) / env['edge']
    measure(0)
    end, twists = None, 0
    for s in range(sweeps):
        seq, coin = eng.draw()
        for lo in range(0, eng.S, eng.chunk):
            n = eng.run_chunk(seq[lo:lo + eng.chunk], coin[lo:lo + eng.chunk], lo, spec['burn'] + s)
            twists += int(np.count_nonzero(eng.o_twist[:n]))
        dead = [i for i in ids if cols['death_sweep'][i] >= 0]
        if dead:
            end = {'sweep': s + 1, 'cause': ['alive', 'annihilation', 'fusion', 'split', 'exchange'][int(cols['cause'][dead[0]])],
                   'with_marked_partner': bool(len(ids) == 2 and int(cols['partner'][dead[0]]) in ids)}
            break
        measure(s + 1)
    hops = [int(cols['hops'][i]) for i in ids]
    out = Path(spec['out'])
    row = {'level': level, 'kind': kind, 'r0': r0, 'seed': seed, 'A': A, 'B': Bv if kind != 'single' else None,
           'separation_initial': None if kind == 'single' else float(sep[0]), 'burn': spec['burn'], 'sweeps': sweeps,
           'end': end, 'twists': twists, 'hops': hops, 'identities': int(eng.tracker.n),
           'charges_alive': int(np.count_nonzero(eng.cls[FI:])), 'seconds': round(time.time() - t0, 1)}
    return row, sep, disp


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--level', type=int, default=5)
    p.add_argument('--seeds', type=int, default=200)
    p.add_argument('--sweeps', type=int, default=2000)
    p.add_argument('--burn', type=int, default=100)
    p.add_argument('--separations', default='4,8,16,32')
    p.add_argument('--kinds', default='opposite,equal,single')
    p.add_argument('--cost', default='binding')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    specs = []
    for kind in a.kinds.split(','):
        for r0 in ([0] if kind == 'single' else [int(v) for v in a.separations.split(',')]):
            for seed in range(a.seeds):
                specs.append(dict(level=a.level, kind=kind, r0=r0 or 4, seed=seed, sweeps=a.sweeps, burn=a.burn, cost=a.cost, out=str(a.out)))
    rows, seps, disps = [], {}, {}
    t0 = time.time()
    with Pool(a.workers) as pool:
        for n, (row, sep, disp) in enumerate(pool.imap_unordered(run_one, specs, chunksize=2)):
            key = (row['kind'], row['r0'] if row['kind'] != 'single' else 0)
            rows.append(row); seps.setdefault(key, []).append(sep); disps.setdefault(key, []).append(disp)
            if (n + 1) % 50 == 0:
                print(f'{n + 1} of {len(specs)} runs, {time.time() - t0:.0f}s', flush=True)
    summary = []
    times = [t for t in (10, 20, 50, 100, 200, 500, 1000, 2000) if t <= a.sweeps]
    for key in sorted(seps):
        kind, r0 = key
        sep = np.stack(seps[key]); disp = np.stack(disps[key])
        sub = [r for r in rows if (r['kind'], r['r0'] if r['kind'] != 'single' else 0) == key]
        ends = [r['end'] for r in sub if r['end']]
        item = {'kind': kind, 'r0': r0, 'runs': len(sub), 'ended': len(ends),
                'ended_by': {c: sum(1 for e in ends if e['cause'] == c) for c in ('annihilation', 'fusion', 'split', 'exchange')},
                'median_end_sweep': float(np.median([e['sweep'] for e in ends])) if ends else None, 'at': {}}
        end_sweep = np.array([r['end']['sweep'] if r['end'] else 10 ** 9 for r in sub])
        met = np.array([bool(r['end'] and r['end']['with_marked_partner']) for r in sub])     # the two marked charges ended on each other
        for t in times:
            if kind != 'single':
                d = sep[:, t] - sep[:, 0]
                ok = ~np.isnan(d)
                # all runs: a pair that ended on each other by t has separation 0; runs that ended otherwise are left out
                full = np.where(ok, sep[:, t], np.where(met & (end_sweep <= t), 0.0, np.nan)) - sep[:, 0]
                okf = ~np.isnan(full)
                item['at'][str(t)] = {'surviving': int(ok.sum()), 'mean_change_of_separation': float(np.nanmean(d)) if ok.any() else None,
                                      'standard_error': float(np.nanstd(d) / np.sqrt(max(1, ok.sum()))) if ok.sum() > 1 else None,
                                      'met_by_then': int((met & (end_sweep <= t)).sum()),
                                      'mean_change_counting_met_pairs_as_zero': float(np.nanmean(full)) if okf.any() else None,
                                      'standard_error_counting_met': float(np.nanstd(full) / np.sqrt(max(1, okf.sum()))) if okf.sum() > 1 else None}
            else:
                d = disp[:, t]
                ok = ~np.isnan(d)
                item['at'][str(t)] = {'surviving': int(ok.sum()), 'mean_square_displacement': float(np.nanmean(d ** 2)) if ok.any() else None}
        summary.append(item)
        np.savez_compressed(a.out / f'{kind}_r{r0}.npz', separation=sep, displacement_of_A=disp)
        line = f'{kind:9} r0={r0:2d} runs={len(sub)} ended={len(ends)} {item["ended_by"]} median end {item["median_end_sweep"]}'
        for t in times:
            v = item['at'][str(t)]
            line += (f' | t={t}: {v["mean_change_counting_met_pairs_as_zero"]:+.2f}±{v["standard_error_counting_met"]:.2f} (met {v["met_by_then"]}, alive {v["surviving"]})' if kind != 'single' and v['mean_change_counting_met_pairs_as_zero'] is not None and v['standard_error_counting_met'] is not None
                     else f' | t={t}: msd {v.get("mean_square_displacement")}' if kind == 'single' else f' | t={t}: n={v["surviving"]}')
        print(line, flush=True)
    (a.out / 'summary.json').write_text(json.dumps({'law': 'law C with transported contrast and unit twists (phase 2)', 'cost': a.cost,
                                                   'level': a.level, 'summary': summary, 'rows': rows}))


if __name__ == '__main__':
    main()
