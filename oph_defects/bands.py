"""Band structure of the wave operators of a federation: which excitations are massless, massive or immobile.

    python -m oph_defects.bands --level 3 --wiring cells:face_axis --json runs/light/bands_L3_face_axis.json

Under a regular clock (continuous flow) the scalar sector has frequencies nu sqrt(lambda), lambda the eigenvalues of
the port Laplacian L = B^T B. A band whose lower edge is at lambda_0 > 0 and rises as lambda_0 + c_b^2 k^2 is a
massive wave with rest frequency nu sqrt(lambda_0) and limiting speed c_b; it shares the light cone of the
massless band only if c_b equals the speed c of the band that starts at zero. A band of zero width is an
excitation that does not move at all.

On a sphere of radius R a band with internal multiplicity g has (l + 1)^2 g states up to angular momentum l, so
near an edge the eigenvalue rises linearly with the state index n: lambda = lambda_0 + c_b^2 n / (g R^2). The
slope against the index, in units of the slope of the massless band, is c_b^2 / (g c^2). The multiplicity g is
read from the degeneracy of the edge level (2l + 1 for orbital l, times the internal dimension).

Wirings: cells:port_pair (declared), cells:face_axis (symmetric control), vertices:port_pair (carriers on the mesh
vertices, six glued ports, declared router).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from oph_exact import carrier as C
from oph_fpe.core.icosahedral import geodesic_icosahedral_patch_arrays
from oph_fpe.core.screen_ports import assign_echosahedral_ports

from . import fedgeom as FG


def port_graph(level: int, wiring: str):
    basis, gluing = wiring.split(':')
    if basis == 'cells' and gluing == 'face_axis':
        geo = FG.build(level, gluing='face_axis')
        a, b = geo.all_seam_ends()
        return np.asarray(a, np.int64), np.asarray(b, np.int64), geo.carriers
    points, left, right = geodesic_icosahedral_patch_arrays(level, patch_basis=basis)
    N = points.shape[0]
    pm = assign_echosahedral_ports(left, right, N, points=points)
    if pm.overflow_count:
        raise RuntimeError('router overflow')
    T = np.asarray(C.seams(), np.int64)
    ia = (12 * np.arange(N)[:, None] + T[None, :, 0]).ravel(); ib = (12 * np.arange(N)[:, None] + T[None, :, 1]).ravel()
    ga = left.astype(np.int64) * 12 + pm.left_port.astype(np.int64); gb = right.astype(np.int64) * 12 + pm.right_port.astype(np.int64)
    return np.concatenate([ia, ga]), np.concatenate([ib, gb]), N


def levels_of(values: np.ndarray, tol: float = 2e-6):
    """Group sorted eigenvalues into levels: list of (value, multiplicity)."""
    out = []
    start = 0
    for i in range(1, values.size + 1):
        if i == values.size or values[i] - values[i - 1] > tol * max(1.0, abs(values[i])):
            out.append((float(values[start:i].mean()), i - start))
            start = i
    return out


def analyse(level: int, wiring: str, gap: float = 0.05) -> dict:
    t0 = time.time()
    a, b, N = port_graph(level, wiring)
    P = 12 * N
    A = sp.coo_matrix((np.ones(a.size), (a, b)), shape=(P, P)); A = (A + A.T).toarray()
    L = np.diag(A.sum(axis=1)) - A
    del A
    ev = np.linalg.eigvalsh(L)
    del L
    cuts = np.flatnonzero(np.diff(ev) > gap)
    edges = np.concatenate([[0], cuts + 1, [P]])
    bands = []
    acoustic = None
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = ev[lo:hi]
        n = band.size
        flat = bool(band[-1] - band[0] < 1e-6)
        item = {'first_state': int(lo), 'states_per_carrier': n / N, 'lower_edge': float(band[0]), 'upper_edge': float(band[-1]),
                'width': float(band[-1] - band[0]), 'flat': flat}
        if not flat and n >= 40:
            m = max(12, min(n // 8, 4 * int(np.sqrt(N))))
            item['slope_bottom_times_N'] = float(np.polyfit(np.arange(m), band[:m], 1)[0] * N)
            item['slope_top_times_N'] = float(-np.polyfit(np.arange(m), band[::-1][:m], 1)[0] * N)
            item['levels_bottom'] = [(round(v, 6), g) for v, g in levels_of(band[:60])[:8]]
            item['levels_top'] = [(round(v, 6), g) for v, g in levels_of(band[::-1][:60][::-1])[::-1][:8]]
        bands.append(item)
        if lo == 0:
            acoustic = item
    for item in bands:
        if 'slope_bottom_times_N' in item and acoustic and acoustic.get('slope_bottom_times_N'):
            item['bottom_slope_over_massless'] = item['slope_bottom_times_N'] / acoustic['slope_bottom_times_N']
            item['top_slope_over_massless'] = item['slope_top_times_N'] / acoustic['slope_bottom_times_N']
    tri = ev[1:4]
    return {'level': level, 'wiring': wiring, 'carriers': int(N), 'ports': int(P), 'seams': int(a.size),
            'glued_ports_per_carrier': float(2 * (a.size - 30 * N) / N), 'triplet_split_percent': float(100 * (tri.max() - tri.min()) / tri.mean()),
            'lowest_scaled_by_carriers_over_20': (ev[1:10] * N / 20).tolist(), 'bands': bands, 'seconds': round(time.time() - t0, 1)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--level', type=int, default=3)
    p.add_argument('--wiring', default='cells:face_axis')
    p.add_argument('--json', type=Path, default=None)
    a = p.parse_args(argv)
    r = analyse(a.level, a.wiring)
    print(f"{r['wiring']} level {r['level']}: {r['carriers']} carriers, {r['glued_ports_per_carrier']:.1f} glued ports per carrier, lowest triplet split {r['triplet_split_percent']:.2f} %  ({r['seconds']}s)")
    for bnd in r['bands']:
        line = f"  {bnd['states_per_carrier']:5.2f} states per carrier  [{bnd['lower_edge']:.5f}, {bnd['upper_edge']:.5f}]"
        if bnd['flat']:
            line += '  flat: does not move'
        elif 'bottom_slope_over_massless' in bnd:
            line += f"  slope bottom {bnd['bottom_slope_over_massless']:.3f}, top {bnd['top_slope_over_massless']:.3f} of the massless band"
            line += f"\n        bottom levels {bnd['levels_bottom'][:6]}\n        top levels    {bnd['levels_top'][:6]}"
        print(line)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(r))


if __name__ == '__main__':
    main()
