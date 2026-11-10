"""Charge density and pair correlations of a closed-loop (twist) universe, slice by slice.

    python -m oph_defects.charge_correlations runs/twist/L5 --rmax 16

For every stored slice of synchronised time: the number of charges by size, the pair correlations of
unit charges g+-(r) and g++(r) (opposite and equal sign) against the angular distance in mesh-edge units,
normalised by the count expected for uncorrelated points of the same densities on the sphere, and the
variance of the net charge inside discs of radius R around random centres. For uncorrelated unit charges
that variance equals the mean number of charges in the disc (it grows with the area); a screened gas keeps
it proportional to the perimeter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .dataset import Dataset


def pair_counts(xyz_a, xyz_b, same: bool, edges_rad: np.ndarray) -> np.ndarray:
    """Pairs (a, b) with angular distance in each bin; chord = 2 sin(theta / 2)."""
    if len(xyz_a) == 0 or len(xyz_b) == 0:
        return np.zeros(len(edges_rad) - 1)
    chords = 2 * np.sin(edges_rad / 2)
    ta, tb = cKDTree(xyz_a), cKDTree(xyz_b)
    cum = ta.count_neighbors(tb, chords).astype(np.float64)
    if same:
        cum = (cum - len(xyz_a)) / 2                      # drop self pairs, count each pair once
    return np.diff(cum)


def disc_variance(xyz: np.ndarray, q: np.ndarray, centres: np.ndarray, radii_rad: np.ndarray) -> list[dict]:
    """Variance over the centres of the net charge inside each disc radius, with the mean number of charges inside."""
    out = []
    if len(xyz) == 0:
        return [{'variance_of_net_charge': 0.0, 'mean_charges_inside': 0.0} for _ in radii_rad]
    chord = 2 * np.sin(radii_rad / 2)
    m = cKDTree(centres).sparse_distance_matrix(cKDTree(xyz), chord.max(), output_type='coo_matrix')
    for c in chord:
        inside = m.data <= c
        net = np.bincount(m.row[inside], weights=q[m.col[inside]].astype(np.float64), minlength=len(centres))
        cnt = np.bincount(m.row[inside], weights=np.abs(q[m.col[inside]]).astype(np.float64), minlength=len(centres))
        out.append({'variance_of_net_charge': float(net.var()), 'mean_charges_inside': float(cnt.mean())})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('--rmax', type=float, default=16.0)
    p.add_argument('--bins', type=int, default=16)
    p.add_argument('--json', type=Path, default=None)
    p.add_argument('--discs', default='2,4,8,16', help='disc radii in mesh edges')
    p.add_argument('--centres', type=int, default=4000)
    a = p.parse_args(argv)
    ds = Dataset(a.run)
    edge = ds.meta['geometry']['cell_spacing'] * np.sqrt(3)       # mesh edge angle from the cell-centre spacing of a triangular tiling
    edges = np.linspace(0, a.rmax, a.bins + 1) * edge
    area = 2 * np.pi * (np.cos(edges[:-1]) - np.cos(edges[1:])) / (4 * np.pi)      # fraction of the sphere in each ring
    stats = ds.stats()
    radii = np.array([float(v) for v in a.discs.split(',')])
    rng = np.random.default_rng(12345)
    centres = rng.normal(size=(a.centres, 3))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    out = []
    print(f"{'depth':>8}{'sweep':>8}{'n1':>8}{'n2':>7}{'n3':>6}  g+-(r) and g++(r) for r bins of {a.rmax / a.bins:.1f} edges (first 6 bins)")
    for cut in ds.cuts():
        z = ds.cut(cut['k'])
        q = z['screen_charge'].astype(int)
        xyz = z['screen_xyz'].astype(np.float64)
        plus, minus = xyz[q == 1], xyz[q == -1]
        gpm = pair_counts(plus, minus, False, edges) / np.maximum(len(plus) * len(minus) * area, 1e-12)
        gpp = (pair_counts(plus, plus, True, edges) + pair_counts(minus, minus, True, edges)) / np.maximum(
            (len(plus) * (len(plus) - 1) / 2 + len(minus) * (len(minus) - 1) / 2) * area, 1e-12)
        depth = cut['depth']
        sweep = float(np.searchsorted(stats['depth_mean'], depth))
        discs = disc_variance(xyz, q, centres, radii * edge)
        row = {'depth': depth, 'sweep': sweep, 'n1': int((np.abs(q) == 1).sum()), 'n2': int((np.abs(q) == 2).sum()),
               'n3': int((np.abs(q) >= 3).sum()), 'g_opposite': gpm.tolist(), 'g_equal': gpp.tolist(),
               'discs': {str(R): d for R, d in zip(a.discs.split(','), discs)}}
        out.append(row)
        if cut['k'] % max(1, len(ds.cuts()) // 24) == 0:
            print(f"{depth:>8}{sweep:>8.0f}{row['n1']:>8}{row['n2']:>7}{row['n3']:>6}  " + ' '.join(f'{v:5.2f}' for v in gpm[:6]) + '  |  ' + ' '.join(f'{v:5.2f}' for v in gpp[:6])
                  + '  | disc var/mean: ' + ' '.join(f"{d['variance_of_net_charge'] / max(d['mean_charges_inside'], 1e-9):.2f}" for d in discs))
    result = {'run': str(a.run), 'edge_angle': edge, 'bin_edges_in_edges': np.linspace(0, a.rmax, a.bins + 1).tolist(), 'slices': out,
              'released_per_sweep': stats['potential_released'].tolist(), 'descents_per_sweep': stats['descents'].tolist()}
    if a.json:
        a.json.write_text(json.dumps(result))


if __name__ == '__main__':
    main()
