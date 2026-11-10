"""Radar distances: can a signal define a length?

    python -m oph_defects.radar --level 4 --distances 2,4,8 --json runs/light/radar_L4.json

An observer measures a distance as half the round-trip time of a signal on its own tick counter (about 33 ticks
per sweep for a carrier). Round-trip times of a signal that travels at a fixed speed add up along a line:
T(A, C) = T(A, B) + T(B, C) for B half-way. For a signal that spreads by diffusion the time grows with the square of
the distance and T(A, C) = 2 [T(A, B) + T(B, C)]. Three collinear observers A, B, C at mesh distances r and 2r from A
are timed with four signals:

  front    the influence front of the native repair (first passage, unit-rate seams)
  load     a load pulse placed on A: the time at which the expected excess load on the target carrier peaks, under
           the exact mean law of repair, and the size of that peak for one quantum
  flow     a wave on a regular clock (the scalar wave psi'' = -L psi): the time at which the signal on the target
           is largest; times in units of the wave clock T
  events   the same wave driven seam by seam by independent random draws with pulse angle phi: the mean signal
           (average over realisations) on the target; its peak size against the flow tells how much of the wave
           survives the schedule

Reported per signal: one-way times at r and 2r, the exponent log2(T(2r) / T(r)) and the additivity ratio
T(A, C) / (T(A, B) + T(B, C)).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.sparse.csgraph import dijkstra

from . import light as LT
from .observer_lab import Lab


def targets(lab: Lab, centre: int, r: float):
    xy = lab.chart(centre)
    b = int(np.argmin(((xy - np.array([r, 0.0])) ** 2).sum(axis=1)))
    c = int(np.argmin(((xy - np.array([2 * r, 0.0])) ** 2).sum(axis=1)))
    return b, c, xy


def front_times(lab: Lab, pairs, realisations: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(pairs))
    sources = sorted({a for a, _ in pairs})
    for _ in range(realisations):
        w = rng.exponential(1.0, lab.M)
        A = sp.coo_matrix((w, (lab.a, lab.b)), shape=(lab.P, lab.P)).tocsr()
        for s in sources:
            t = dijkstra(A, directed=False, indices=lab.ports_of([s]), min_only=True).reshape(lab.N, 12).min(axis=1)
            for i, (a, b) in enumerate(pairs):
                if a == s:
                    acc[i] += t[b]
    return acc / realisations


def load_peak(lab: Lab, src: int, dst: int, guess: float):
    """Peak time (sweeps) and size of the expected excess on carrier dst from one quantum spread over carrier src."""
    p = np.zeros(lab.P); p[lab.ports_of([src])] = 1.0 / 12
    grid = np.geomspace(0.05 * guess, 4.0 * guess, 48)
    vals = spl.expm_multiply(-0.5 * lab.L, p, start=grid[0], stop=grid[-1], num=96, endpoint=True)
    times = np.linspace(grid[0], grid[-1], 96)
    sig = vals[:, lab.ports_of([dst])].sum(axis=1)
    i = int(np.argmax(sig))
    if 0 < i < 95:                                    # parabolic refinement
        y0, y1, y2 = sig[i - 1], sig[i], sig[i + 1]
        shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        return float(times[i] + shift * (times[1] - times[0])), float(y1)
    return float(times[i]), float(sig[i])


def flow_series(lab: Lab, src: int, dsts, T: float, step: float = 0.25, width: float = 1.0):
    xy = lab.chart(src)
    u = np.exp(-(xy ** 2).sum(axis=1) / (2 * width ** 2))[lab.carrier_of]
    u /= np.linalg.norm(u)
    prev = u.copy(); h = step * step
    cur = u - 0.5 * h * (lab.L @ u)                   # starts at rest
    n = int(T / step)
    out = np.zeros((n + 1, len(dsts)))
    ports = [lab.ports_of([d]) for d in dsts]
    for k in range(n + 1):
        for j, pp in enumerate(ports):
            out[k, j] = cur[pp].mean() if k else u[pp].mean()
        if k:
            prev, cur = cur, 2 * cur - h * (lab.L @ cur) - prev
    return step * np.arange(1, n + 2), out, u


def event_series(lab: Lab, u0: np.ndarray, dsts, phi: float, sweeps: int, realisations: int, seed: int = 0):
    """Mean signal on the targets for the wave driven by independent seam draws (scalar sector rotations)."""
    lib = LT._lib()
    M, P = lab.M, lab.P
    indptr = (2 * np.arange(M + 1)).astype(np.int64)
    idx = np.stack([lab.a, lab.b], axis=1).ravel().astype(np.int32)
    coef = np.tile(np.array([2 ** -0.5, -(2 ** -0.5)]), M)
    ports = [lab.ports_of([d]) for d in dsts]
    acc = np.zeros((sweeps + 1, len(dsts)))
    c, s = float(np.cos(phi)), float(np.sin(phi))
    for r in range(realisations):
        rng = np.random.default_rng(seed + r)
        X = u0.copy(); E = np.zeros(M)
        for t in range(sweeps + 1):
            for j, pp in enumerate(ports):
                acc[t, j] += X[pp].mean()
            if t < sweeps:
                seq = rng.integers(0, M, size=M, dtype=np.int32)
                lib.oph_rotate(indptr.ctypes.data, idx.ctypes.data, coef.ctypes.data, seq.ctypes.data, M, c, s, E.ctypes.data, X.ctypes.data)
    return acc / realisations


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--level', type=int, default=4)
    p.add_argument('--distances', default='2,4,8')
    p.add_argument('--phi', type=float, default=0.4)
    p.add_argument('--realisations', type=int, default=60)
    p.add_argument('--json', type=Path, default=None)
    a = p.parse_args(argv)
    lab = Lab(a.level, gluing='face_axis')
    A = 4 ** a.level - 1
    sigma = lab.conductivity()
    D = 0.5 * sigma / (12 / (3 ** 0.5 / 4))
    c_flow = np.sqrt(2 * D)
    rows = []
    for r in [float(v) for v in a.distances.split(',')]:
        B, C, xy = targets(lab, A, r)
        rAB, rAC = float(np.hypot(*xy[B])), float(np.hypot(*xy[C]))
        ft = front_times(lab, [(A, B), (B, C), (A, C)], a.realisations)
        lAB, sAB = load_peak(lab, A, B, rAB ** 2 / (4 * D)); lBC, _ = load_peak(lab, B, C, rAB ** 2 / (4 * D)); lAC, sAC = load_peak(lab, A, C, rAC ** 2 / (4 * D))
        T, sig, u0 = flow_series(lab, A, [B, C], 1.6 * rAC / c_flow)
        fAB, fAC = float(T[np.argmax(np.abs(sig[:, 0]))]), float(T[np.argmax(np.abs(sig[:, 1]))])
        _, sig2, _ = flow_series(lab, B, [C], 1.6 * rAB / c_flow)
        fBC = float(T[np.argmax(np.abs(sig2[:, 0]))])
        sweeps = int(1.6 * rAC / (c_flow * a.phi * 2 ** -0.5))
        ev = event_series(lab, u0, [B, C], a.phi, sweeps, max(8, a.realisations // 4))
        row = {'r': rAB, 'two_r': rAC,
               'front': {'AB': float(ft[0]), 'BC': float(ft[1]), 'AC': float(ft[2])},
               'load': {'AB': lAB, 'BC': lBC, 'AC': lAC, 'peak_excess_per_quantum_AB': sAB, 'peak_excess_per_quantum_AC': sAC},
               'flow': {'AB': fAB, 'BC': fBC, 'AC': fAC, 'peak_AB': float(np.abs(sig[:, 0]).max()), 'peak_AC': float(np.abs(sig[:, 1]).max())},
               'events': {'phi': a.phi, 'sweeps': sweeps, 'peak_AB': float(np.abs(ev[:, 0]).max()), 'peak_AC': float(np.abs(ev[:, 1]).max()),
                          'peak_sweep_AB': int(np.argmax(np.abs(ev[:, 0]))), 'peak_sweep_AC': int(np.argmax(np.abs(ev[:, 1])))}}
        for key in ('front', 'load', 'flow'):
            s = row[key]
            s['exponent'] = float(np.log2(s['AC'] / s['AB']) / np.log2(rAC / rAB)); s['additivity'] = float(s['AC'] / (s['AB'] + s['BC']))
        rows.append(row)
        print(f"r = {rAB:.2f}, 2r = {rAC:.2f} edges:")
        print(f"   front   {ft[0]:8.2f} {ft[1]:8.2f} {ft[2]:8.2f} sweeps   exponent {row['front']['exponent']:.2f}   additivity {row['front']['additivity']:.3f}")
        print(f"   load    {lAB:8.1f} {lBC:8.1f} {lAC:8.1f} sweeps   exponent {row['load']['exponent']:.2f}   additivity {row['load']['additivity']:.3f}   peak excess per quantum {sAB:.2e}, {sAC:.2e}")
        print(f"   flow    {fAB:8.2f} {fBC:8.2f} {fAC:8.2f} clock T  exponent {row['flow']['exponent']:.2f}   additivity {row['flow']['additivity']:.3f}   peak signal {row['flow']['peak_AB']:.3e}, {row['flow']['peak_AC']:.3e}")
        print(f"   events  phi {a.phi}: mean signal peak {row['events']['peak_AB']:.3e} at sweep {row['events']['peak_sweep_AB']}, {row['events']['peak_AC']:.3e} at sweep {row['events']['peak_sweep_AC']} "
              f"(fraction of the flow peak {row['events']['peak_AB'] / row['flow']['peak_AB']:.3f}, {row['events']['peak_AC'] / row['flow']['peak_AC']:.3f})", flush=True)
    out = {'level': a.level, 'gluing': 'face_axis', 'diffusion_constant': D, 'flow_speed_edges_per_T': float(c_flow), 'rows': rows}
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out))


if __name__ == '__main__':
    main()
