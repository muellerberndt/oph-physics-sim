"""Law C with transported contrast: pinned offset dipoles on the exact federation (declared extension).

    python -m oph_defects.closed_loop --level 3 --sweeps 20000 --out runs/closed_loop/L3 --workers 16

State: loads x (signed) and one integer offset b_e per seam, oriented a -> b. An attempt on seam (a, b)
with coin c takes s = x_a + x_b + b_e, sets x_a to ceil(s/2) (c = 1) or floor(s/2) (c = 0) and x_b to
s - x_a - b_e. With all b_e = 0 this is law C. The seam compares x_a with the transported neighbour
x_b + b_e; the transported contrast is d_e = x_a - x_b - b_e and the current j_e = old x_a - new x_a has
E[j_e | e drawn] = d_e / 2. Around every closed cycle of the port graph the contrasts sum to minus the
holonomy of b, so the expected net current around the cycle is -hol / (2 M) per attempt, -hol / 2 per sweep.

Phase 1 pins a dipole: offsets on the glue seams dual to a mesh path between two screen vertices, so that
every port cycle around one end has holonomy +q or -q. Measured on the unchanged schedule:

* cumulative current around explicit port cycles (rings) at several radii, against -hol / 2 per sweep;
* the release R = j (d - j) per attempt and the work of the offsets -sum_e b_e J_e, against the exact
  stationary value (1/2) |dbar|^2 per sweep, where dbar = B x* - b is the divergence-free part of -b
  (x* the least-squares heights, B the seam-by-port incidence);
* the time-averaged contrast and the cumulative current of every seam, against dbar and dbar / 2 per sweep;
* release, descent and nonzero-current events per carrier, against the distance from the cores.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from . import fedgeom as FG
from .universe import COSTS, Engine, prepare_loads, schedule_seed


# ---------------------------------------------------------------------------- mesh helpers
class Mesh:
    def __init__(self, geo: FG.Geometry):
        self.geo = geo
        N = geo.carriers
        tri = geo.tri.astype(np.int64)
        V = geo.vertices
        nbr = [set() for _ in range(V)]
        for a, b, c in tri.tolist():
            nbr[a].update((b, c)); nbr[b].update((a, c)); nbr[c].update((a, b))
        self.nbr = [sorted(s) for s in nbr]
        vp = geo.glue_fp.astype(np.int64) - 20 * N
        vm = geo.glue_fm.astype(np.int64) - 20 * N
        self.edge_glue = {(int(min(u, w)), int(max(u, w))): g for g, (u, w) in enumerate(zip(vp, vm))}
        self.vp, self.vm = vp, vm
        # template port graph for paths inside a carrier
        self.tnbr = [[] for _ in range(12)]
        self.tseam = {}
        for k, (a, b) in enumerate(FG.TEMPLATE_SEAMS.tolist()):
            self.tnbr[a].append(b); self.tnbr[b].append(a)
            self.tseam[(a, b)] = (k, 1); self.tseam[(b, a)] = (k, -1)

    def distances(self, v: int, limit: int | None = None) -> np.ndarray:
        dist = np.full(self.geo.vertices, -1, np.int32)
        dist[v] = 0
        q = deque([v])
        while q:
            u = q.popleft()
            if limit is not None and dist[u] >= limit:
                continue
            for w in self.nbr[u]:
                if dist[w] < 0:
                    dist[w] = dist[u] + 1
                    q.append(w)
        return dist

    def path(self, v0: int, v1: int) -> list[int]:
        dist = self.distances(v1)
        out = [v0]
        while out[-1] != v1:
            u = out[-1]
            out.append(min((w for w in self.nbr[u] if dist[w] == dist[u] - 1)))
        return out

    def template_path(self, p: int, q: int) -> list[tuple[int, int]]:
        """Shortest path of template seams from port p to port q: list of (template seam, sign)."""
        if p == q:
            return []
        prev = {p: None}
        dq = deque([p])
        while dq:
            u = dq.popleft()
            if u == q:
                break
            for w in self.tnbr[u]:
                if w not in prev:
                    prev[w] = u
                    dq.append(w)
        nodes = [q]
        while prev[nodes[-1]] is not None:
            nodes.append(prev[nodes[-1]])
        nodes.reverse()
        return [self.tseam[(a, b)] for a, b in zip(nodes[:-1], nodes[1:])]


def dipole_offsets(mesh: Mesh, v_plus: int, v_minus: int, q: int) -> tuple[np.ndarray, list[int]]:
    """Offsets on the glue seams dual to a mesh path, giving plaquette curl +q at v_plus and -q at v_minus."""
    geo = mesh.geo
    off = np.zeros(geo.seams, np.int8)
    path = mesh.path(v_plus, v_minus)
    for u, w in zip(path[:-1], path[1:]):
        g = mesh.edge_glue[(min(u, w), max(u, w))]
        off[geo.intra_seams + g] = q if mesh.vp[g] == u else -q
    return off, path


def ring_cycle(mesh: Mesh, v: int, R: int) -> list[tuple[int, int]] | None:
    """A closed cycle of the port graph around mesh vertex v at radius R: list of (seam, sign); None if the strip is not a simple cycle."""
    geo = mesh.geo
    dist = mesh.distances(v, R + 1)
    td = dist[geo.tri.astype(np.int64)]
    strip = np.flatnonzero((td.min(axis=1) == R - 1) & (td.max(axis=1) == R))
    if strip.size < 3:
        return None
    in_strip = set(strip.tolist())
    nb = geo.neighbours()
    cg = geo.carrier_glue_seams()
    links = {}
    for c in strip.tolist():
        out = [(int(w), int(g)) for w, g in zip(nb[c], cg[c]) if int(w) in in_strip]
        if len(out) != 2:
            return None
        links[c] = out
    order, glues = [int(strip[0])], []
    prev = None
    while True:
        c = order[-1]
        nxt = [(w, g) for w, g in links[c] if w != prev]
        w, g = nxt[0]
        glues.append(g)
        if w == order[0]:
            break
        prev = c
        order.append(w)
        if len(order) > strip.size:
            return None
    if len(order) != strip.size:
        return None
    cycle = []
    n = len(order)
    for i in range(n):
        c, g_in, g_out = order[i], glues[i - 1], glues[i]
        port_in = int(geo.glue_a[g_in]) if int(geo.glue_a[g_in]) // 12 == c else int(geo.glue_b[g_in])
        a_out, b_out = int(geo.glue_a[g_out]), int(geo.glue_b[g_out])
        port_out = a_out if a_out // 12 == c else b_out
        for k, sign in mesh.template_path(port_in - 12 * c, port_out - 12 * c):
            cycle.append((30 * c + k, sign))
        cycle.append((geo.intra_seams + g_out, 1 if a_out // 12 == c else -1))
    return cycle


def cycle_is_closed(geo: FG.Geometry, cycle) -> bool:
    seams = np.array([s for s, _ in cycle], np.int64)
    sign = np.array([g for _, g in cycle], np.int64)
    a, b = geo.seam_ends(seams)
    head = np.where(sign > 0, a, b)
    tail = np.where(sign > 0, b, a)
    return bool((tail == np.roll(head, -1)).all())


# ---------------------------------------------------------------------------- exact stationary field
def stationary_field(geo: FG.Geometry, off: np.ndarray) -> dict:
    """dbar = B x* - b with x* = argmin |B x - b|^2: the mean transported contrast of every stationary state."""
    a, b = geo.all_seam_ends()
    S, P = geo.seams, geo.ports
    B = sp.csr_matrix((np.concatenate([np.ones(S), -np.ones(S)]), (np.concatenate([np.arange(S), np.arange(S)]), np.concatenate([a, b]))), shape=(S, P))
    rhs = B.T @ off.astype(np.float64)
    if not np.any(rhs):
        return {'dbar': np.zeros(S), 'x': np.zeros(P), 'release_per_sweep': 0.0, 'residual': 0.0}
    L = (B.T @ B).tocsc()
    if P <= 400_000:
        x = np.zeros(P)
        x[1:] = spl.splu(L[1:, 1:]).solve(rhs[1:])
    else:
        dinv = 1.0 / L.diagonal()
        x, info = spl.cg(L, rhs, rtol=1e-12, atol=0.0, maxiter=200_000, M=spl.LinearOperator((P, P), matvec=lambda v: dinv * v))
        if info:
            raise RuntimeError('conjugate gradients did not converge')
    dbar = B @ x - off
    return {'dbar': dbar, 'x': x - x.mean(), 'release_per_sweep': float(0.5 * dbar @ dbar), 'residual': float(np.abs(B.T @ dbar).max())}


def predicted_nonzero(geo: FG.Geometry, off: np.ndarray, x_star: np.ndarray, rho: float) -> np.ndarray:
    """Parameter-free estimate of the nonzero-current probability per draw of every seam (theory session).

    A port's stationary mean load is rho + x*_p; it holds the upper of the two adjacent integers a fraction
    f_p (its fractional part) of the time. Treating the two ends of a seam as independent two-level variables,
    the transported contrast d = X_a - (X_b + b_e) is odd with a probability fixed by f_a, f_b and the level
    difference; a tie (|d| = 1) moves a unit on half of its draws and a contrast of two or more always does.
    """
    a, b = geo.all_seam_ends()
    ma = rho + x_star[a]
    mb = rho + x_star[b] + off.astype(np.float64)
    na, nb = np.floor(ma), np.floor(mb)
    fa, fb = ma - na, mb - nb
    lv = (na - nb).astype(np.int64)                       # level difference; d = lv + u_a - u_b with u in {0, 1}
    p = np.zeros(geo.seams)
    for ua, pa in ((0, 1 - fa), (1, fa)):
        for ub, pb in ((0, 1 - fb), (1, fb)):
            d = np.abs(lv + ua - ub)
            p += pa * pb * np.where(d == 1, 0.5, np.where(d >= 2, 1.0, 0.0))
    return p


def rounded_heights(x: np.ndarray) -> np.ndarray:
    """Integer heights nearest to the mean-zero least-squares heights, with total exactly zero."""
    r = np.rint(x).astype(np.int64)
    excess = int(r.sum())
    if excess:
        order = np.argsort(x - r, kind='stable')
        if excess > 0:
            r[order[:excess]] -= 1
        else:
            r[order[excess:]] += 1
    assert int(r.sum()) == 0
    return r


# ---------------------------------------------------------------------------- one run
def run_one(spec: dict) -> dict:
    t0 = time.time()
    level, q, r = spec['level'], spec['q'], spec['r']
    geo = FG.build(level, gluing=spec.get('gluing', 'port_pair'))
    mesh = Mesh(geo)
    N = geo.carriers
    v_plus = spec.get('v_plus')
    if v_plus is None:
        v_plus = int(geo.tri[4 ** level - 1].max())          # a six-valent vertex of the root cell
    dplus = mesh.distances(v_plus)
    if q and r:
        v_minus = int(np.flatnonzero(dplus == r)[0])
        off, path = dipole_offsets(mesh, v_plus, v_minus, q)
    else:
        v_minus, path = -1, [v_plus]
        off = np.zeros(geo.seams, np.int8)
    dminus = mesh.distances(v_minus) if v_minus >= 0 else np.full(geo.vertices, 10 ** 6, np.int32)
    if spec['background'] == 'frozen':
        loads = np.zeros(geo.ports, np.int8)
    else:
        loads, _ = prepare_loads(f'canonical:{spec["background"]}:{spec.get("prep_seed", 0)}', level, geo.ports)
    exact = stationary_field(geo, off)
    rho = float(loads.astype(np.float64).mean())
    p_nz = predicted_nonzero(geo, off, exact['x'], rho)
    if spec.get('start', 'relaxed') == 'relaxed':             # start at the rounded stationary heights: no mean-field transient
        loads = (loads.astype(np.int64) + rounded_heights(exact['x'])).astype(np.int8)
    eng = Engine(geo, loads, schedule_seed(level, spec.get('schedule', 0)), COSTS['binding'], 5, False, slots=2,
                 register=False, offsets=off, currents=True, counters=True, fast_rng=True)
    rings = []
    for centre, v in (('plus', v_plus), ('minus', v_minus)):
        if v < 0:
            continue
        for R in spec['radii']:
            cyc = ring_cycle(mesh, v, R)
            if cyc is None or not cycle_is_closed(geo, cyc):
                continue
            seams = np.array([s_ for s_, _ in cyc], np.int64)
            sign = np.array([g for _, g in cyc], np.int64)
            rings.append({'centre': centre, 'R': R, 'seams': seams, 'sign': sign, 'holonomy': int((sign * off[seams]).sum())})
    a_all, b_all = geo.all_seam_ends()
    off64 = off.astype(np.int64)
    string = np.flatnonzero(off)
    carrier_a = (a_all // 12).astype(np.int64)
    carrier_b = (b_all // 12).astype(np.int64)
    sweeps, half = spec['sweeps'], spec['sweeps'] // 2
    ser = {k: [] for k in ('sweep', 'strain_twice', 'odd_seams', 'release', 'offset_work', 'descents', 'nonzero')}
    zser = []
    dsum = np.zeros(geo.seams, np.int64)
    nsamp = 0
    rel = desc = nz = 0
    release2 = np.zeros(N, np.float64)                        # twice the release attributed to each carrier, second half
    draws = np.zeros(geo.seams, np.int64)                     # realized attempts per seam, second half
    rel_seam = np.zeros(geo.seams, np.float64)                # release per seam, second half
    jhalf = None
    for s in range(sweeps):
        if s == half:
            jhalf = eng.jsum.copy(); eng.nzc[:] = 0; eng.dsc[:] = 0; release2[:] = 0
        seq, coin = eng.draw()
        for lo in range(0, eng.S, eng.chunk):
            sq = seq[lo:lo + eng.chunk]
            n = eng.run_chunk(sq, coin[lo:lo + eng.chunk], lo, s)
            cur, con = eng.o_cur[:n].astype(np.int32), eng.o_con[:n].astype(np.int32)
            R_att = cur * (con - cur)
            desc += int(np.count_nonzero((con >= 2) | (con <= -2))); nz += int(np.count_nonzero(cur)); rel += int(R_att.sum(dtype=np.int64))
            if s >= half:
                draws += np.bincount(sq, minlength=geo.seams)
                rel_seam += np.bincount(sq, weights=R_att.astype(np.float64), minlength=geo.seams)
                hit = np.flatnonzero(R_att)
                if hit.size:
                    w = R_att[hit].astype(np.float64)
                    release2 += np.bincount(carrier_a[sq[hit]], weights=w, minlength=N) + np.bincount(carrier_b[sq[hit]], weights=w, minlength=N)
        t = s + 1
        if t <= 1024 or t % 16 == 0:
            x = eng.x.astype(np.int64)
            d = x[a_all] - x[b_all] - off64
            if t > half:
                dsum += d; nsamp += 1
            ser['sweep'].append(t); ser['strain_twice'].append(int((d * d).sum())); ser['odd_seams'].append(int(np.count_nonzero(d & 1)))
            ser['release'].append(rel); ser['offset_work'].append(-int((off64[string] * eng.jsum[string]).sum()))
            ser['descents'].append(desc); ser['nonzero'].append(nz)
            zser.append([int((ring['sign'] * eng.jsum[ring['seams']]).sum()) for ring in rings])
            if int(np.abs(eng.x).max()) > 100:
                raise RuntimeError('loads left the int8 guard range')
    ser = {k: np.asarray(v, np.int64) for k, v in ser.items()}
    zser = np.asarray(zser, np.int64).reshape(ser['sweep'].size, len(rings))
    late = ser['sweep'] > half
    span = sweeps - half
    i_half = int(np.flatnonzero(ser['sweep'] == half)[0]) if half in ser['sweep'] else int(np.searchsorted(ser['sweep'], half))
    def rate(col):                                             # stationary rate per sweep over the second half
        return float((col[-1] - col[i_half]) / (ser['sweep'][-1] - ser['sweep'][i_half]))
    ring_rows = []
    for i, ring in enumerate(rings):
        z = zser[:, i]
        tt, zz = ser['sweep'][late], z[late]
        inc = np.diff(zz) / np.diff(tt)                                         # per-sweep current in each late interval
        mean = float((zz[-1] - zz[0]) / (tt[-1] - tt[0]))
        se = float(inc.std(ddof=1) / np.sqrt(inc.size)) if inc.size > 2 else None
        predicted = float((ring['sign'] * exact['dbar'][ring['seams']]).sum() / 2)
        nd = np.maximum(draws[ring['seams']], 1)
        per_draw = float((ring['sign'] * (eng.jsum[ring['seams']].astype(np.int64) - jhalf[ring['seams']]) / nd).sum())
        ring_rows.append({'centre': ring['centre'], 'R': ring['R'], 'length': int(ring['seams'].size), 'holonomy': ring['holonomy'],
                          'expected_per_sweep': -ring['holonomy'] / 2, 'from_dbar_per_sweep': predicted, 'measured_per_sweep': mean,
                          'measured_per_draw': per_draw,
                          'standard_error': se, 'z': None if not se else (mean + ring['holonomy'] / 2) / se})
    tri = geo.tri.astype(np.int64)
    cd_plus = dplus[tri].min(axis=1)
    cd_minus = dminus[tri].min(axis=1)
    name = f"L{level}_{spec['background'].replace('/', '-')}_q{q}_r{r}" + ('' if geo.gluing == 'port_pair' else '_' + geo.gluing)
    out = Path(spec['out']) / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'offsets.npy', off)
    mean_d = dsum / max(1, nsamp)
    jdiff = eng.jsum.astype(np.int64) - jhalf
    jlate = jdiff / span
    nd_all = np.maximum(draws, 1)
    j_draw = jdiff / nd_all                                  # mean current per realized draw: estimates dbar / 2
    pred_carrier = np.bincount(carrier_a, weights=p_nz, minlength=N)
    glue = np.arange(geo.seams) >= geo.intra_seams
    pred_carrier += np.bincount(carrier_b[glue], weights=p_nz[glue], minlength=N)        # a glue seam counts for both carriers
    meas_carrier = eng.nzc.astype(np.float64) / span
    release_per_draw = float((rel_seam / nd_all).sum())
    work_per_draw = -float((off64[string] * j_draw[string]).sum())
    arrays = {'sweep': ser['sweep'], 'strain_twice': ser['strain_twice'], 'odd_seams': ser['odd_seams'],
              'release_cumulative': ser['release'], 'offset_work_cumulative': ser['offset_work'],
              'descents_cumulative': ser['descents'], 'nonzero_cumulative': ser['nonzero'], 'ring_current_cumulative': zser,
              'descent_events_per_carrier_late': eng.dsc.copy(), 'nonzero_events_per_carrier_late': eng.nzc.copy(),
              'release_twice_per_carrier_late': release2, 'carrier_distance_plus': cd_plus.astype(np.int32),
              'carrier_distance_minus': cd_minus.astype(np.int32), 'mean_contrast_per_seam_late': mean_d.astype(np.float32),
              'current_per_sweep_per_seam_late': jlate.astype(np.float32), 'cumulative_current_per_seam': eng.jsum.copy(),
              'current_per_draw_per_seam_late': j_draw.astype(np.float32), 'draws_per_seam_late': draws.astype(np.int32),
              'release_per_seam_late': rel_seam.astype(np.float32), 'initial_loads': loads.copy(),
              'nonzero_probability_predicted_per_seam': p_nz.astype(np.float32), 'x_star': exact['x'].astype(np.float32),
              'nonzero_rate_predicted_per_carrier': pred_carrier.astype(np.float32),
              'dbar_exact': exact['dbar'].astype(np.float64), 'string_path': np.asarray(path), 'final_loads': eng.x.copy()}
    for i, ring in enumerate(rings):
        arrays[f'ring{i}_seams'] = ring['seams'].astype(np.int32); arrays[f'ring{i}_sign'] = ring['sign'].astype(np.int8)
    np.savez_compressed(out / 'arrays.npz', **arrays)
    strain = ser['strain_twice'][late] / 2
    pred = exact['release_per_sweep']
    row = {'name': name, 'level': level, 'background': spec['background'], 'q': q, 'r': r, 'v_plus': v_plus, 'v_minus': v_minus,
           'seams': int(geo.seams), 'sweeps': sweeps, 'stationary_window': [half, sweeps],
           'schedule': spec.get('schedule', 0), 'schedule_seed': schedule_seed(level, spec.get('schedule', 0)),
           'preparation_seed': spec.get('prep_seed', 0),
           'release_exact_per_sweep': pred, 'release_measured_per_sweep': rate(ser['release']),
           'offset_work_measured_per_sweep': rate(ser['offset_work']),
           'release_relative_error': None if not pred else rate(ser['release']) / pred - 1,
           'release_per_draw': release_per_draw, 'offset_work_per_draw': work_per_draw,
           'release_per_draw_relative_error': None if not pred else release_per_draw / pred - 1,
           'offset_work_per_draw_relative_error': None if not pred else work_per_draw / pred - 1,
           'current_field_per_draw_rms_error': float(np.sqrt(((j_draw - exact['dbar'] / 2) ** 2).mean())),
           'start': spec.get('start', 'relaxed'), 'density': rho, 'gluing': geo.gluing,
           'nonzero_predicted_per_sweep': float(p_nz.sum()),
           'nonzero_map_correlation': float(np.corrcoef(pred_carrier, meas_carrier)[0, 1]) if q else None,
           'nonzero_map_rms_relative_error': float(np.sqrt((((meas_carrier - pred_carrier) / np.maximum(pred_carrier, 1e-9)) ** 2).mean())),
           'odd_fraction': float(ser['odd_seams'][late].mean() / geo.seams),
           'descents_per_sweep': rate(ser['descents']), 'nonzero_per_sweep': rate(ser['nonzero']),
           'strain_mean': float(strain.mean()), 'odd_seams_mean': float(ser['odd_seams'][late].mean()),
           'strain_identity_rhs': 2 * pred + float(ser['odd_seams'][late].mean()) if True else None,
           'strain_twice_mean': float(ser['strain_twice'][late].mean()),
           'contrast_field_max_error': float(np.abs(mean_d - exact['dbar']).max()),
           'contrast_field_rms_error': float(np.sqrt(((mean_d - exact['dbar']) ** 2).mean())),
           'current_field_rms_error': float(np.sqrt(((jlate - exact['dbar'] / 2) ** 2).mean())),
           'load_min': int(eng.x.min()), 'load_max': int(eng.x.max()), 'rings': ring_rows,
           'files': ['offsets.npy', 'arrays.npz'], 'seconds': round(time.time() - t0, 1)}
    (out / 'summary.json').write_text(json.dumps(row))
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--level', type=int, required=True)
    p.add_argument('--sweeps', type=int, default=20000)
    p.add_argument('--frozen-sweeps', type=int, default=4000)
    p.add_argument('--charges', default='1,2,3')
    p.add_argument('--separations', default='1,2,4,8,16')
    p.add_argument('--backgrounds', default='frozen,1/3')
    p.add_argument('--radii', default='1,2,4,8')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--schedule-base', type=int, default=0, help='run i uses schedule and preparation seed base + i')
    p.add_argument('--gluing', default='port_pair', choices=('port_pair', 'face_axis'), help='face_axis is the frame-free control gluing')
    p.add_argument('--start', default='relaxed', choices=('relaxed', 'background'),
                   help='relaxed: background plus the rounded stationary heights; background: the bare background')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    radii = [int(v) for v in a.radii.split(',')]
    specs = []
    for bg in a.backgrounds.split(','):
        n = a.frozen_sweeps if bg == 'frozen' else a.sweeps
        specs.append(dict(level=a.level, q=0, r=0, background=bg, radii=radii, sweeps=n, out=str(a.out)))
        for q in (int(v) for v in a.charges.split(',')):
            for r in (int(v) for v in a.separations.split(',')):
                specs.append(dict(level=a.level, q=q, r=r, background=bg, radii=sorted(set(radii + [r + 2, 2 * r + 2])), sweeps=n, out=str(a.out)))
    for i, spec in enumerate(specs):                      # independent schedules and preparations per run
        spec['schedule'] = a.schedule_base + i
        spec['prep_seed'] = a.schedule_base + i
        spec['start'] = a.start
        spec['gluing'] = a.gluing
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(a.workers) as pool:
        for row in pool.imap_unordered(run_one, specs):
            rows.append(row)
            zs = [abs(g['z']) for g in row['rings'] if g['z'] is not None]
            err = row['release_relative_error']
            e2 = row['release_per_draw_relative_error']
            print(f"{row['name']}: release {row['release_measured_per_sweep']:.4f} exact {row['release_exact_per_sweep']:.4f}"
                  f" ({'%+.2f%%' % (100 * err) if err is not None else 'n/a'}; per draw {'%+.2f%%' % (100 * e2) if e2 is not None else 'n/a'})"
                  f" offset work {row['offset_work_measured_per_sweep']:.4f}"
                  f" odd {row['odd_seams_mean']:.1f} rings {len(row['rings'])} max|z| {max(zs) if zs else 0:.2f} {row['seconds']}s", flush=True)
    rows.sort(key=lambda z: (z['background'], z['q'], z['r']))
    (a.out / 'summary.json').write_text(json.dumps({'law': 'law C with transported contrast, static offsets (phase 1)', 'rows': rows}))


if __name__ == '__main__':
    main()
