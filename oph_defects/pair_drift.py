"""Drift of loop charges in the closed-loop law: mean displacement vectors in the frame of the pair axis.

    python -m oph_defects.pair_drift --level 5 --seeds 200 --sweeps 40 --separations 8,16,32 \
        --backgrounds frozen,1/10,1/4,1/3,1/2,3/4 --workers 26 --out runs/closed_loop/L5_drift

Test of the law reported by the theory session for a plain triangular lattice: every unit loop charge moves at one
speed v0 in a direction set by the filling phase and the bearings of the other charges,
    zdot_j = v0 exp(-i q_j [2 pi f + pi + sum_k q_k arg(z_j - z_k)]),
so that an opposite pair has dR/dt = -2 v0 cos(2 pi f) and its centre moves sideways at v0 sin(2 pi f).

Each run prepares an opposite unit pair (+1 at A, -1 at B, B at mesh distance r0 in a direction that varies with the
seed) on a background of filling f at the rounded stationary heights, holds it pinned for a burn-in, releases it and
records the plaquette of each marked charge after every sweep. Reported per (background, r0), averaged over runs in
which both charges are still alive: the displacement of each charge in its own frame,
    toward   component along the geodesic to the initial position of its partner,
    left     component along n x toward, n the outward normal (left of that direction, seen from outside),
in mesh edges, with standard errors, at the listed sweeps; the mean and mean square of the step per sweep; and the
mean square displacement per hop.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

from . import kernels as K
from .closed_loop import dipole_offsets, rounded_heights
from .pair_dynamics import _setup, heights
from .universe import COSTS, Engine, prepare_loads, schedule_seed


def run_one(spec: dict):
    level, r0, seed, bg = spec['level'], spec['r0'], spec['seed'], spec['background']
    env = _setup(level)
    geo, mesh = env['geo'], env['mesh']
    A = env['v_plus']
    ring = np.flatnonzero(env['d_plus'] == r0)
    Bv = int(ring[seed % ring.size])
    if spec['kind'] == 'opposite':
        off, _ = dipole_offsets(mesh, A, Bv, 1)
        marked = [(A, 1), (Bv, -1)]
    else:                                                        # single +1 with its compensator at the antipode
        far = np.flatnonzero(env['d_anti'] <= 2)
        off, _ = dipole_offsets(mesh, A, int(far[seed % far.size]), 1)
        marked = [(A, 1)]
        Bv = int(far[seed % far.size])
    off = off.astype(np.int8)
    if bg == 'frozen':
        loads = np.zeros(geo.ports, np.int8)
    else:
        loads, _ = prepare_loads(f'canonical:{bg}:{seed}', level, geo.ports)
    loads = (loads.astype(np.int64) + rounded_heights(heights(env, off.astype(np.int64)))).astype(np.int8)
    eng = Engine(geo, loads, schedule_seed(level, 3000 + seed), COSTS['binding'], spec.get('kappa', 5), False, slots=2,
                 register=False, offsets=off, twists=True, fast_rng=True)
    FI = geo.intra_faces
    ids = [int(eng.tracker.face_id[v]) for v, _ in marked]
    flags_on = eng.flags
    eng.flags = flags_on & ~K.F_TWIST; eng.ctx.flags = eng.flags
    for s in range(spec['burn']):
        seq, coin = eng.draw()
        for lo in range(0, eng.S, eng.chunk):
            eng.run_chunk(seq[lo:lo + eng.chunk], coin[lo:lo + eng.chunk], lo, s)
    eng.flags = flags_on; eng.ctx.flags = flags_on
    cols = eng.tracker.cols
    sweeps = spec['sweeps']
    pos = np.full((sweeps + 1, 2), -1, np.int32)
    hops = np.zeros((sweeps + 1, 2), np.int32)
    pos[0, :len(ids)] = [int(cols['last_face'][i]) - FI for i in ids]
    for s in range(sweeps):
        seq, coin = eng.draw()
        for lo in range(0, eng.S, eng.chunk):
            eng.run_chunk(seq[lo:lo + eng.chunk], coin[lo:lo + eng.chunk], lo, spec['burn'] + s)
        if any(cols['death_sweep'][i] >= 0 for i in ids):
            break
        pos[s + 1, :len(ids)] = [int(cols['last_face'][i]) - FI for i in ids]
        hops[s + 1, :len(ids)] = [int(cols['hops'][i]) for i in ids]
    return spec['kind'], bg, r0, seed, A, Bv, pos, hops


def tangent_components(xyz: np.ndarray, start: int, partner: int, where: np.ndarray, edge: float) -> np.ndarray:
    """Log map of the vertices ``where`` at vertex ``start`` in the frame (toward partner, left), in mesh edges."""
    n = xyz[start]
    t = xyz[partner] - (xyz[partner] @ n) * n
    t /= np.linalg.norm(t)
    left = np.cross(n, t)
    p = xyz[where]
    cos = np.clip(p @ n, -1, 1)
    ang = np.arccos(cos)
    tan = p - cos[:, None] * n
    norm = np.maximum(np.linalg.norm(tan, axis=1), 1e-300)
    u = tan / norm[:, None] * (ang / edge)[:, None]
    return np.stack([u @ t, u @ left], axis=1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--level', type=int, default=5)
    p.add_argument('--seeds', type=int, default=200)
    p.add_argument('--sweeps', type=int, default=40)
    p.add_argument('--burn', type=int, default=100)
    p.add_argument('--separations', default='8,16,32')
    p.add_argument('--backgrounds', default='frozen,1/10,1/4,1/3,1/2,3/4')
    p.add_argument('--kinds', default='opposite')
    p.add_argument('--kappa', type=int, default=5)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    specs = [dict(level=a.level, kind=kind, r0=int(r0), seed=seed, background=bg, sweeps=a.sweeps, burn=a.burn, kappa=a.kappa)
             for kind in a.kinds.split(',') for bg in a.backgrounds.split(',')
             for r0 in (a.separations.split(',') if kind == 'opposite' else ['0']) for seed in range(a.seeds)]
    for s in specs:
        if s['kind'] == 'single':
            s['r0'] = 4
    env = _setup(a.level)
    xyz, edge = env['geo'].mesh_xyz, env['edge']
    got = {}
    t0 = time.time()
    with Pool(a.workers) as pool:
        for n, (kind, bg, r0, seed, A, B, pos, hops) in enumerate(pool.imap_unordered(run_one, specs, chunksize=4)):
            got.setdefault((kind, bg, r0 if kind == 'opposite' else 0), []).append((seed, A, B, pos, hops))
            if (n + 1) % 200 == 0:
                print(f'{n + 1} of {len(specs)} runs, {time.time() - t0:.0f}s', flush=True)
    times = [t for t in (5, 10, 20, 40, 80, 160) if t <= a.sweeps]
    rows = []
    for key in sorted(got, key=lambda k: (k[0], k[1], k[2])):
        kind, bg, r0 = key
        runs = got[key]
        ncharge = 2 if kind == 'opposite' else 1
        D = np.full((len(runs), a.sweeps + 1, ncharge, 2), np.nan)
        H = np.full((len(runs), a.sweeps + 1, ncharge), np.nan)
        for i, (seed, A, B, pos, hops) in enumerate(runs):
            alive = (pos[:, :ncharge] >= 0).all(axis=1)
            last = int(np.flatnonzero(alive)[-1])
            for c, (start, partner) in enumerate([(A, B), (B, A)][:ncharge]):
                D[i, :last + 1, c] = tangent_components(xyz, start, partner, pos[:last + 1, c], edge)
                H[i, :last + 1, c] = hops[:last + 1, c]
        item = {'kind': kind, 'background': bg, 'r0': r0, 'runs': len(runs), 'at': {}}
        line = f'{kind:8} f={bg:6} r0={r0:2d}:'
        for t in times:
            ok = ~np.isnan(D[:, t, 0, 0])
            if ok.sum() < 5:
                continue
            m = np.nanmean(D[ok, t], axis=0); se = np.nanstd(D[ok, t], axis=0) / np.sqrt(ok.sum())
            msd = np.nanmean((D[ok, t] ** 2).sum(axis=2), axis=0)
            hp = np.nanmean(H[ok, t], axis=0)
            item['at'][str(t)] = {'alive': int(ok.sum()), 'mean': m.tolist(), 'standard_error': se.tolist(),
                                  'mean_square_displacement': msd.tolist(), 'mean_hops': hp.tolist()}
            line += f' | t={t}: +({m[0, 0]:+.2f},{m[0, 1]:+.2f})' + (f' -({m[1, 0]:+.2f},{m[1, 1]:+.2f})' if ncharge == 2 else '') + f' se {se.max():.2f} msd {msd[0]:.1f} hops {hp[0]:.1f}'
        rows.append(item)
        print(line, flush=True)
        np.savez_compressed(a.out / f"{kind}_f{bg.replace('/', '-')}_r{r0}.npz", displacement=D.astype(np.float32), hops=H.astype(np.float32),
                            seeds=np.array([r[0] for r in runs]), start=np.array([[r[1], r[2]] for r in runs]),
                            positions=np.stack([r[3] for r in runs]))
    (a.out / 'summary.json').write_text(json.dumps({'law': 'law C with transported contrast and unit twists', 'level': a.level, 'kappa': a.kappa,
                                                   'frame': 'toward = along the geodesic to the initial position of the partner; left = n x toward, n outward',
                                                   'rows': rows}))


if __name__ == '__main__':
    main()
