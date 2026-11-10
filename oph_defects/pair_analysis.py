"""Coupling of the charge gas from the pair matrix of ``pair_dynamics``.

    python -m oph_defects.pair_analysis runs/closed_loop/L5_pairs

For a logarithmic pair potential the relative coordinate of two unit charges obeys, at early times,
d<r^2>/dt = 8 D +- 4 mu kappa (equal charges +, opposite -), independent of r in the plane. On the sphere of
radius R (in edges) the diffusion part carries the factor c_D = (1 + theta cot theta) / 2 and the drift part
c_F = (theta / 2) cot(theta / 2), theta = r / R. With the slopes S_eq and S_op of <r^2>:

    D = (S_eq + S_op) / (16 c_D),   mu kappa = (S_eq - S_op) / (8 c_F),   Gamma = mu kappa / D.

The single-charge control gives D independently from its mean square displacement, 4 D t.
Slopes are fitted on the window where fewer than 5 percent of the pairs have met.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def slope(t: np.ndarray, y: np.ndarray, err: np.ndarray):
    """Weighted straight-line slope through (t, y) and its standard error."""
    w = 1 / np.maximum(err, 1e-9) ** 2
    tm, ym = (w * t).sum() / w.sum(), (w * y).sum() / w.sum()
    sxx = (w * (t - tm) ** 2).sum()
    return float((w * (t - tm) * (y - ym)).sum() / sxx), float(np.sqrt(1 / sxx))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('path', type=Path)
    p.add_argument('--tmin', type=int, default=2)
    p.add_argument('--tmax', type=int, default=200, help='upper end of the fit window if fewer than 5 percent have met by then')
    p.add_argument('--json', type=Path, default=None)
    a = p.parse_args(argv)
    meta = json.loads((a.path / 'summary.json').read_text())
    level = meta['level']
    V = 10 * 4 ** level + 2
    R = np.sqrt(V * np.sqrt(3) / 2 / (4 * np.pi))                       # sphere radius in mesh edges (hexagonal cells of area sqrt(3)/2)
    rows = meta['rows']
    out = {'level': level, 'sphere_radius_in_edges': float(R), 'sets': [], 'coupling': []}
    slopes = {}
    z = np.load(a.path / 'single_r0.npz')
    disp = z['displacement_of_A'].astype(np.float64)
    msd = np.nanmean(disp ** 2, axis=0); n = (~np.isnan(disp)).sum(axis=0)
    err = np.nanstd(disp ** 2, axis=0) / np.sqrt(np.maximum(n, 1))
    alive = n / disp.shape[0]
    T = max(a.tmin + 5, int(np.argmax(alive < 0.95)) if (alive < 0.95).any() else a.tmax)
    T = min(T, a.tmax)
    t = np.arange(a.tmin, T + 1)
    s1, e1 = slope(t.astype(float), msd[t], err[t])
    D_single = s1 / 4
    print(f'level {level}, sphere radius {R:.1f} edges')
    print(f'single charge: msd slope {s1:.4f} +- {e1:.4f} per sweep on t = {a.tmin}..{T} -> D = {D_single:.4f} +- {e1 / 4:.4f} edge^2 per sweep ({disp.shape[0]} runs)')
    out['single'] = {'msd_slope': s1, 'msd_slope_error': e1, 'D': D_single, 'window': [a.tmin, int(T)], 'runs': int(disp.shape[0])}
    for kind in ('opposite', 'equal'):
        for f in sorted(a.path.glob(f'{kind}_r*.npz'), key=lambda q: int(q.stem.split('_r')[1])):
            r0 = int(f.stem.split('_r')[1])
            sep = np.load(f)['separation'].astype(np.float64)
            sub = [r for r in rows if r['kind'] == kind and r['r0'] == r0]
            runs = sep.shape[0]
            end = np.array([r['end']['sweep'] if r['end'] else 10 ** 9 for r in sub])
            met_by = np.array([(end <= tt).mean() for tt in range(sep.shape[1])])
            T = int(np.argmax(met_by >= 0.05)) - 1 if (met_by >= 0.05).any() else a.tmax
            T = max(a.tmin + 5, min(T, a.tmax))
            t = np.arange(a.tmin, T + 1)
            r2 = np.nanmean(sep ** 2, axis=0); nn = (~np.isnan(sep)).sum(axis=0)
            er = np.nanstd(sep ** 2, axis=0) / np.sqrt(np.maximum(nn, 1))
            s, e = slope(t.astype(float), r2[t], er[t])
            r_init = float(np.nanmean(sep[:, 0]))
            slopes[(kind, r0)] = (s, e, r_init)
            item = {'kind': kind, 'r0': r0, 'runs': runs, 'separation_initial': r_init, 'slope_r2': s, 'slope_r2_error': e, 'window': [a.tmin, T],
                    'met_fraction': {str(tt): float(met_by[min(tt, len(met_by) - 1)]) for tt in (10, 20, 50, 100, 200, 500, 1000, 2000) if tt < sep.shape[1]},
                    'median_meeting_sweep': float(np.median(end[end < 10 ** 9])) if (end < 10 ** 9).any() else None}
            out['sets'].append(item)
            print(f"{kind:9} r0 = {r0:2d} (measured {r_init:5.2f}): d<r^2>/dt = {s:+.4f} +- {e:.4f} on t = {a.tmin}..{T}; met by 10/50/200/1000 sweeps: "
                  + ' / '.join(f"{item['met_fraction'].get(str(tt), float('nan')):.2f}" for tt in (10, 50, 200, 1000)))
    for r0 in sorted({k[1] for k in slopes}):
        if ('equal', r0) in slopes and ('opposite', r0) in slopes:
            se, ee, re_ = slopes[('equal', r0)]; so, eo, ro = slopes[('opposite', r0)]
            th = 0.5 * (re_ + ro) / R
            cD, cF = (1 + th / np.tan(th)) / 2, (th / 2) / np.tan(th / 2)
            D = (se + so) / (16 * cD); mk = (se - so) / (8 * cF)
            dD = np.hypot(ee, eo) / (16 * cD); dmk = np.hypot(ee, eo) / (8 * cF)
            gamma = mk / D
            dg = abs(gamma) * np.hypot(dmk / max(abs(mk), 1e-12), dD / max(abs(D), 1e-12))
            out['coupling'].append({'r0': r0, 'theta': float(th), 'c_D': float(cD), 'c_F': float(cF), 'D': float(D), 'D_error': float(dD),
                                    'mu_kappa': float(mk), 'mu_kappa_error': float(dmk), 'Gamma': float(gamma), 'Gamma_error': float(dg),
                                    'D_over_single': float(D / D_single)})
            print(f'r0 = {r0:2d}: D = {D:.4f} +- {dD:.4f} ({D / D_single:.2f} of the single-charge value), mu kappa = {mk:+.4f} +- {dmk:.4f}, Gamma = {gamma:+.3f} +- {dg:.3f}')
    if a.json:
        a.json.write_text(json.dumps(out))


if __name__ == '__main__':
    main()
