"""The accumulated seam-current record of the bare law on a federation: is it a persistent record, and of what?

    python -m oph_defects.current_record --level 3 --sweeps 4000 --every 50 --seeds 8 --workers 8

R_e(t) is the sum of the recorded currents of seam e up to time t (a native record: every attempt writes its
current). Its boundary is x(t) - x(0) identically, bounded by the loads. Its cycle part is not restored by anything.
Per carrier the cycle space of the thirty internal seams has dimension 19 and splits under the rotation group
into 3 + 3' + 4 + 4 + 5 (as the face circulations: 20 faces with one relation). Between carriers the cycle space is
spanned by the circulations around the screen plaquettes. Measured on canonical (balanced) runs:

  face      growth per sweep of the variance of the circulation of R around one internal face
  axial     the vector A_c = sum over the internal seams of carrier c of R_e (u_a x u_b): growth per sweep of
            <|A_c(t) - A_c(0)|^2>, per component and in the local frame (normal against tangent)
  ring      growth per sweep of the variance of the circulation of R around a screen plaquette
  shared    the correlation of the increments of A between carriers against their mesh distance (do neighbouring
            carriers share the record), and of the plaquette circulations between plaquettes
  spectrum  eigenvalues of the covariance of the face circulations of one carrier over time, by rotation multiplet:
            which components grow and which stay bounded
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import fedgeom as FG
from .universe import COSTS, Engine, prepare_loads, schedule_seed


def run_one(spec):
    level, seed, sweeps, every, filling, gluing = spec
    geo = FG.build(level, gluing=gluing)
    N = geo.carriers
    loads, _ = prepare_loads(f'canonical:{filling}:{seed}', level, geo.ports)
    eng = Engine(geo, loads, schedule_seed(level, 8800 + seed), COSTS['binding'], 5, False, slots=2, register=False,
                 currents=True, fast_rng=True)
    U = FG.PORT_UNIT
    T = FG.TEMPLATE_SEAMS
    cross = np.cross(U[T[:, 0]], U[T[:, 1]])                              # (30, 3) template axial weights
    frames = geo.frames(np.arange(N))                                     # (N, 3, 3): columns tangent_x, tangent_y, normal
    plus, minus = FG.TEMPLATE_PLUS, FG.TEMPLATE_MINUS
    snaps = []
    for s in range(sweeps + 1):
        if s % every == 0:
            R = eng.jsum.astype(np.int64)
            Ri = R[:geo.intra_seams].reshape(N, 30)
            face = np.zeros((N, 20), np.int64)
            np.add.at(face, (slice(None), plus), Ri); np.add.at(face, (slice(None), minus), -Ri)
            A_t = Ri @ cross                                              # template coordinates
            ring = np.zeros(geo.vertices, np.int64)
            Rg = R[geo.intra_seams:]
            np.add.at(ring, geo.glue_fp - geo.intra_faces, Rg); np.add.at(ring, geo.glue_fm - geo.intra_faces, -Rg)
            snaps.append((face.astype(np.int32), A_t.astype(np.float32), ring.astype(np.int32)))
        if s < sweeps:
            seq, coin = eng.draw()
            for lo in range(0, eng.S, eng.chunk):
                eng.run_chunk(seq[lo:lo + eng.chunk], coin[lo:lo + eng.chunk], lo, s)
    return {'face': np.stack([a for a, _, _ in snaps]), 'axial': np.stack([b for _, b, _ in snaps]), 'ring': np.stack([c for _, _, c in snaps])}


def multiplets():
    """Orthonormal bases of the rotation multiplets of the face space of one carrier (20 = 1 + 3 + 3' + 4 + 4 + 5),
    from the dual Laplacian (dodecahedron graph): eigenvalues 0, 3 - sqrt5, 2, 3, 5, 3 + sqrt5 with multiplicities
    1, 3, 5, 4, 4, 3."""
    D = np.zeros((20, 20))
    for k in range(30):
        p, m = int(FG.TEMPLATE_PLUS[k]), int(FG.TEMPLATE_MINUS[k])
        D[p, p] += 1; D[m, m] += 1; D[p, m] -= 1; D[m, p] -= 1
    ev, V = np.linalg.eigh(D)
    out, start = [], 0
    for i in range(1, 21):
        if i == 20 or ev[i] - ev[i - 1] > 1e-6:
            out.append((float(ev[start:i].mean()), V[:, start:i])); start = i
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--level', type=int, default=3)
    p.add_argument('--sweeps', type=int, default=4000)
    p.add_argument('--every', type=int, default=50)
    p.add_argument('--seeds', type=int, default=8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--filling', default='1/3')
    p.add_argument('--gluing', default='face_axis')
    p.add_argument('--json', type=Path, default=None)
    a = p.parse_args(argv)
    with Pool(a.workers) as pool:
        runs = pool.map(run_one, [(a.level, seed, a.sweeps, a.every, a.filling, a.gluing) for seed in range(a.seeds)])
    geo = FG.build(a.level, gluing=a.gluing)
    N = geo.carriers
    t = np.arange(0, a.sweeps + 1, a.every).astype(float)
    late = t >= t[-1] / 4
    def growth(msd):                                                      # slope of a mean square against sweeps
        return float(np.polyfit(t[late], msd[late], 1)[0]), float(msd[-1])
    face = np.stack([r['face'] for r in runs]).astype(np.float64)         # (seeds, T, N, 20)
    face -= face[:, :1]
    out = {'level': a.level, 'gluing': a.gluing, 'filling': a.filling, 'sweeps': a.sweeps, 'seeds': a.seeds, 'carriers': N}
    g, last = growth((face ** 2).mean(axis=(0, 2, 3)))
    out['face_circulation'] = {'variance_growth_per_sweep': g, 'variance_at_end': last}
    rows = []
    for lam, V in multiplets():
        comp = face @ V                                                   # (seeds, T, N, dim)
        g, last = growth((comp ** 2).mean(axis=(0, 2, 3)))
        rows.append({'dual_eigenvalue': lam, 'dimension': int(V.shape[1]), 'variance_growth_per_sweep_per_component': g, 'variance_at_end': last})
    out['face_multiplets'] = rows
    ax = np.stack([r['axial'] for r in runs]).astype(np.float64); ax -= ax[:, :1]
    g, last = growth((ax ** 2).sum(axis=3).mean(axis=(0, 2)))
    out['axial'] = {'growth_per_sweep_of_mean_square': g, 'mean_square_at_end': last}
    ring = np.stack([r['ring'] for r in runs]).astype(np.float64); ring -= ring[:, :1]
    g, last = growth((ring ** 2).mean(axis=(0, 2)))
    out['ring'] = {'variance_growth_per_sweep': g, 'variance_at_end': last}
    # shared: correlation of the final increments between carriers at glue distance 1, 2, 3 and far
    nb = geo.neighbours()                                                 # (N, 3)
    A = ax[:, -1]                                                         # (seeds, N, 3) in template coordinates
    self_ms = float((A ** 2).sum(axis=2).mean())
    c1 = float(np.mean([(A[:, :, :] * A[:, nb[:, k], :]).sum(axis=2).mean() for k in range(3)]))
    F = face[:, -1]
    f_self = float((F ** 2).mean())
    out['shared'] = {'axial_self_mean_square': self_ms, 'axial_neighbour_dot_mean': c1, 'axial_neighbour_correlation_template_frame': c1 / self_ms,
                     'face_self': f_self}
    tri = geo.tri.astype(np.int64)
    rg = ring[:, -1]
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]])
    out['shared']['ring_self'] = float((rg ** 2).mean()); out['shared']['ring_neighbour_correlation'] = float((rg[:, edges[:, 0]] * rg[:, edges[:, 1]]).mean() / (rg ** 2).mean())
    # ring against the carrier face circulations it touches: is the total circulation around a mesh vertex shared
    print(f"level {a.level} {a.gluing} filling {a.filling}: {a.seeds} seeds x {a.sweeps} sweeps, {N} carriers")
    print(f"   face circulation of R: variance grows {out['face_circulation']['variance_growth_per_sweep']:.5f} per sweep (at end {out['face_circulation']['variance_at_end']:.1f})")
    for r in rows:
        print(f"      multiplet of dimension {r['dimension']} (dual eigenvalue {r['dual_eigenvalue']:.4f}): growth {r['variance_growth_per_sweep_per_component']:.5f} per sweep per component, at end {r['variance_at_end']:.2f}")
    print(f"   axial vector: <|A|^2> grows {out['axial']['growth_per_sweep_of_mean_square']:.5f} per sweep (at end {out['axial']['mean_square_at_end']:.1f}); neighbour correlation of the increments {out['shared']['axial_neighbour_correlation_template_frame']:+.4f}")
    print(f"   plaquette ring: variance grows {out['ring']['variance_growth_per_sweep']:.5f} per sweep (at end {out['ring']['variance_at_end']:.1f}); neighbour correlation {out['shared']['ring_neighbour_correlation']:+.4f}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out))


if __name__ == '__main__':
    main()
