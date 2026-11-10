"""Stationary charge gas of a closed-loop run: densities and the charge structure factor.

    python -m oph_defects.plasma_analysis runs/twist/L5_plasma_k20 --skip 0.5

Reads the plaquette charges stored in the density series (``--series-every`` with ``--twists``). Reports the
mean number of charges by size over the second part of the run and the charge structure factor
S_l = <|sum_i q_i Y_lm(x_i)|^2> averaged over m, divided by the mean number of charged plaquettes, against
k^2 = l (l + 1) / R^2 (R the sphere radius in mesh edges). A screened (Debye) gas has
S(k) = k^2 / (k^2 + m^2): S -> 0 at small k and the fitted m is the inverse screening length.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import sph_harm_y

from . import fedgeom as FG
from .dataset import Dataset


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('--skip', type=float, default=0.5, help='fraction of the run discarded as transient')
    p.add_argument('--lmax', type=int, default=48)
    p.add_argument('--json', type=Path, default=None)
    a = p.parse_args(argv)
    ds = Dataset(a.run)
    geo = FG.build(ds.level)
    xyz = geo.mesh_xyz
    V = xyz.shape[0]
    R = np.sqrt(V * np.sqrt(3) / 2 / (4 * np.pi))
    theta = np.arccos(np.clip(xyz[:, 2], -1, 1)); phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    Q, sweeps = [], []
    for block in ds.series():
        Q.append(block['plaquette_charge'].astype(np.float64)); sweeps.append(block['sweep'])
    Q = np.concatenate(Q); sweeps = np.concatenate(sweeps)
    keep = sweeps > a.skip * sweeps.max()
    Qs = Q[keep]
    n_by = {k: float((np.abs(Qs) == k).sum(axis=1).mean()) for k in (1, 2, 3)}
    n_charged = float((Qs != 0).sum(axis=1).mean())
    q2 = float((Qs ** 2).sum(axis=1).mean())
    S = np.zeros(a.lmax + 1)
    for l in range(1, a.lmax + 1):
        acc = 0.0
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, m, theta, phi)
            acc += float((np.abs(Qs @ Y) ** 2).mean())
        S[l] = 4 * np.pi * acc / (2 * l + 1) / max(q2, 1e-12)
    ell = np.arange(1, a.lmax + 1)
    k2 = ell * (ell + 1) / R ** 2
    # Debye fit on the lower half of the range: S = A k^2 / (k^2 + m^2)
    lo = ell <= a.lmax // 2
    best = None
    for m2 in np.geomspace(1e-4, 10, 400):
        f = k2[lo] / (k2[lo] + m2)
        A = (S[1:][lo] * f).sum() / (f * f).sum()
        err = ((S[1:][lo] - A * f) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, m2, A)
    out = {'run': str(a.run), 'kappa': ds.meta['config']['kappa'], 'J': ds.meta['config']['cost_table'][1] / ds.meta['config']['kappa'],
           'plaquettes': int(V), 'snapshots_used': int(keep.sum()), 'mean_charges_by_size': n_by, 'mean_charged_plaquettes': n_charged,
           'charged_fraction': n_charged / V, 'mean_square_charge_sum': q2, 'sphere_radius_edges': float(R),
           'structure_factor': S[1:].tolist(), 'k2': k2.tolist(), 'debye_m2': float(best[1]), 'debye_amplitude': float(best[2]),
           'screening_length_edges': float(1 / np.sqrt(best[1])),
           'stats_tail': {k: float(np.asarray(ds.stats()[k])[-200:].mean()) for k in ('descents', 'potential_released', 'twists_accepted', 'swaps')}}
    print(f"kappa {out['kappa']} (J = {out['J']:.3f}): charges of size 1/2/3 = {n_by[1]:.1f} / {n_by[2]:.1f} / {n_by[3]:.1f} on {V} plaquettes "
          f"({100 * n_charged / V:.2f} percent charged); twists per sweep {out['stats_tail']['twists_accepted']:.0f}; released per sweep {out['stats_tail']['potential_released']:.0f}")
    print('   S_l for l = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48: ' + ' '.join(f'{S[l]:.3f}' for l in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48) if l <= a.lmax))
    print(f"   Debye fit S = A k^2/(k^2 + m^2): m^2 = {best[1]:.4f} per edge^2, screening length {out['screening_length_edges']:.2f} edges, A = {best[2]:.3f}")
    if a.json:
        a.json.write_text(json.dumps(out))


if __name__ == '__main__':
    main()
