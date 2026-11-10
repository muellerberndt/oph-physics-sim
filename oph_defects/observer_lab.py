"""Detector lab: one load quantum on a federation, under the native record law and two coherent continuations.

    python -m oph_defects.observer_lab --level 4 --json runs/lab/L4_report.json

The sector. Total load one on the binary shell (a single hole on a full shell is the same after x -> 1 - x).
The basis is |p>, p a port, and a state is a density operator on l2(ports): this is the Hilbert space on which
an observer describes one quantum.

The native law. The current instrument of the executable-observer paper (its equation for the Kraus operators
L_(c,j)) on seam (a, b), a the first end, with a fair coin c, restricted to this sector:

    c = 0:  L_(0,+1) = |b><a| / sqrt 2      (the quantum was at a and moved to b)
            L_(0, 0) = (1 - |a><a|) / sqrt 2  (it was not at a; nothing moved)
    c = 1:  L_(1,-1) = |a><b| / sqrt 2      (the quantum was at b and moved to a)
            L_(1, 0) = (1 - |b><b|) / sqrt 2  (it was not at b; nothing moved)

So an attempt tests one end of its seam for the quantum, the coin choosing the end. Every branch maps basis
vectors to basis vectors. Hence (i) every effect of every composition of native instruments is diagonal in the
load basis: the statistics of native records depend on the diagonal of the state alone, and no composition of
native instruments is a momentum measurement; (ii) with M seams and one uniformly drawn seam per attempt the
unread channel is exactly

    diag(rho)(n + 1) = K diag(rho)(n),                      K = 1 - L / (2 M),
    rho_pq(n + 1) = (1 - (d_p + d_q) / (2 M)) rho_pq(n)     (p != q),

with L the port Laplacian and d the degree: coherence never moves and is never created. Per sweep (M attempts)
the diagonal follows exp(-L / 2) and a coherence decays by exp(-(d_p + d_q) / 2), with d = 5 or 6.

An observer as a detector. Carrier O reads the records of its own 33 seams. A quantum outside O enters O only
through one of the three glue seams of O, and that hop is a nonzero current on the tape of O: a click. Until
then the law of the quantum given the tape of O is the absorbing evolution with generator -L / 2 on the ports
outside O (per sweep), and the click rate is half the probability on the three outside ends of those seams.

Two coherent continuations, both declared, for comparison with the native law:

    'transfer'  psi(s) = exp(-i s L / 2) psi(0), s in sweeps. The unitary group with the generator of the native
                transfer exp(-s L / 2). Dispersion E = lambda / 2: for long waves E = k^2 / (2 m) with
                m = n / sigma (n ports per unit area, sigma the homogenised conductivity).
    'wave'      psi(T) = exp(-i T sqrt(L)) psi(0). The positive-frequency part of the admitted coherent walk
                cos(t sqrt(A)), A = 4^level L, of the paper: T = 2^level t, N = T sqrt(M) walk steps, and
                N = d s for a register at a port of degree d after s sweeps. Dispersion omega = sqrt(lambda):
                for long waves omega = c |k| with c = sqrt(sigma / n).

Supplied detectors (Born readout, a declared service of the paper): a position detector is the projector on a set
of ports; a momentum analyser is the plane-wave transform over the chart of a patch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from . import fedgeom as FG

SQRT3 = 3.0 ** 0.5
PORT_DENSITY = 12.0 / (SQRT3 / 4.0)          # ports per unit area in units of the mesh edge


class Lab:
    """The port graph of one federation with the operators of the one-quantum sector."""

    def __init__(self, level: int, gluing: str = 'port_pair'):
        self.level = level
        self.geo = geo = FG.build(level, gluing=gluing)
        a, b = geo.all_seam_ends()
        self.a = np.asarray(a, np.int64); self.b = np.asarray(b, np.int64)
        self.P, self.M, self.N = geo.ports, geo.seams, geo.carriers
        ones = np.ones(self.M)
        A = sp.coo_matrix((ones, (self.a, self.b)), shape=(self.P, self.P))
        A = (A + A.T).tocsr()
        self.deg = np.asarray(A.sum(axis=1)).ravel()
        self.L = (sp.diags(self.deg) - A).tocsr()
        self.radius = float(np.sqrt(geo.vertices * SQRT3 / 2 / (4 * np.pi)))      # sphere radius in mesh edges
        self.carrier_of = np.arange(self.P) // 12
        self._sigma = None

    # ------------------------------------------------------------------ geometry
    def chart(self, centre: int) -> np.ndarray:
        """Azimuthal equidistant coordinates of every carrier about carrier ``centre``, in mesh edges."""
        X = self.geo.carrier_xyz / np.linalg.norm(self.geo.carrier_xyz, axis=1, keepdims=True)
        n = X[centre]
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(ref, n); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        cos = np.clip(X @ n, -1, 1)
        theta = np.arccos(cos)
        t1, t2 = X @ e1, X @ e2
        norm = np.maximum(np.hypot(t1, t2), 1e-300)
        return self.radius * theta[:, None] * np.stack([t1 / norm, t2 / norm], axis=1)

    def carriers_within(self, centre: int, edges: float) -> np.ndarray:
        X = self.geo.carrier_xyz / np.linalg.norm(self.geo.carrier_xyz, axis=1, keepdims=True)
        return np.flatnonzero(self.radius * np.arccos(np.clip(X @ X[centre], -1, 1)) <= edges)

    def ports_of(self, carriers) -> np.ndarray:
        return (12 * np.asarray(carriers, np.int64)[:, None] + np.arange(12)[None, :]).ravel()

    def conductivity(self) -> float:
        """Homogenised conductivity sigma from the lowest triplet: lambda = sigma l (l + 1) pi / (6 V) at l = 1."""
        if self._sigma is None:
            vals = spl.eigsh(self.L, k=4, sigma=-1e-4 / 4 ** self.level, which='LM', return_eigenvectors=False)
            lam1 = float(np.sort(vals)[1:4].mean())
            self._sigma = lam1 * 6 * self.geo.vertices / (2 * np.pi)
        return self._sigma

    # ------------------------------------------------------------------ preparations
    def packet(self, centre: int, width: float, k=(0.0, 0.0), xy: np.ndarray | None = None) -> np.ndarray:
        """Gaussian packet of position spread ``width`` (per axis, mesh edges) and wave vector k (per mesh edge),
        constant on the twelve ports of each carrier, normalised."""
        xy = self.chart(centre) if xy is None else xy
        amp = np.exp(-(xy ** 2).sum(axis=1) / (4 * width ** 2)) * np.exp(1j * (xy @ np.asarray(k, float)))
        psi = amp[self.carrier_of].astype(np.complex128)
        return psi / np.linalg.norm(psi)

    # ------------------------------------------------------------------ native law (exact unread channel)
    def native_diagonal(self, p0: np.ndarray, sweeps: float) -> np.ndarray:
        return spl.expm_multiply(-0.5 * sweeps * self.L, np.asarray(p0, float))

    def native_coherent_part(self, psi0: np.ndarray, sweeps: float) -> np.ndarray:
        """psi~ with rho(s) = diag(p(s)) + |psi~><psi~| - diag(|psi~|^2) for the pure preparation psi0."""
        return psi0 * np.exp(-0.5 * sweeps * self.deg)

    # ------------------------------------------------------------------ coherent continuations
    def transfer_unitary(self, psi0: np.ndarray, sweeps: float) -> np.ndarray:
        return spl.expm_multiply(-0.5j * sweeps * self.L, psi0.astype(np.complex128))

    def lanczos_apply(self, f, v: np.ndarray, m: int = 80) -> np.ndarray:
        """f(L) v from an m-dimensional Krylov space with full reorthogonalisation."""
        v = v.astype(np.complex128)
        beta0 = np.linalg.norm(v)
        Q = np.zeros((m, v.size), np.complex128)
        alpha = np.zeros(m); beta = np.zeros(m)
        q = v / beta0
        k = m
        for j in range(m):
            Q[j] = q
            w = self.L @ q
            alpha[j] = float(np.vdot(q, w).real)
            w = w - alpha[j] * q - (beta[j - 1] * Q[j - 1] if j else 0)
            w = w - Q[:j + 1].T @ (Q[:j + 1].conj() @ w)
            beta[j] = np.linalg.norm(w)
            if beta[j] < 1e-12:
                k = j + 1
                break
            q = w / beta[j]
        T = np.diag(alpha[:k]) + np.diag(beta[:k - 1], 1) + np.diag(beta[:k - 1], -1)
        ev, U = np.linalg.eigh(T)
        coef = U @ (f(np.maximum(ev, 0.0)) * U[0].conj())
        return beta0 * (Q[:k].T @ coef)

    def wave_unitary(self, psi0: np.ndarray, T: float, step: float = 0.25, m: int = 80) -> np.ndarray:
        """exp(-i T sqrt(L)) psi0 by the Chebyshev walk psi(n+1) = 2 K psi(n) - psi(n-1), K = 1 - step^2 L / 2,
        started with one Krylov step. With step = 1 / sqrt(M) this is the walk of the paper; a larger step is the
        same continuum wave with a relative frequency error step^2 lambda / 24."""
        n = max(1, int(round(T / step)))
        h = (T / n) ** 2
        if h * float(self.deg.max()) > 1.0:
            raise ValueError('wave step too long for a stable walk')
        prev = psi0.astype(np.complex128)
        cur = self.lanczos_apply(lambda x: np.exp(-1j * np.arccos(np.clip(1 - h * x / 2, -1, 1))), prev, m)
        for _ in range(n - 1):
            prev, cur = cur, 2 * cur - h * (self.L @ cur) - prev
        return cur

    # ------------------------------------------------------------------ observer as detector
    def click_law(self, observer: int, p0: np.ndarray, sweeps: np.ndarray):
        """Survival and click rate at the given sweep counts for the quantum law p0 (zero on the observer)."""
        own = self.ports_of([observer])
        outside = np.ones(self.P, bool); outside[own] = False
        if float(np.asarray(p0)[own].sum()) > 1e-12:
            raise ValueError('the quantum must start outside the observer')
        G = (-0.5 * self.L)[outside][:, outside].tocsc()
        idx = np.full(self.P, -1); idx[outside] = np.arange(int(outside.sum()))
        glue = np.arange(self.geo.intra_seams, self.M)
        ga, gb = self.a[glue], self.b[glue]
        ends = np.concatenate([gb[self.carrier_of[ga] == observer], ga[self.carrier_of[gb] == observer]])
        sweeps = np.asarray(sweeps, float)
        p = np.asarray(p0, float)[outside]
        out_s, out_r = [], []
        last = 0.0
        for s in sweeps:
            if s > last:
                p = spl.expm_multiply(G * (s - last), p)
                last = s
            out_s.append(float(p.sum())); out_r.append(0.5 * float(p[idx[ends]].sum()))
        return np.array(out_s), np.array(out_r)

    # ------------------------------------------------------------------ readouts
    def cell_probability(self, rho_diag: np.ndarray) -> np.ndarray:
        return np.asarray(rho_diag, float).reshape(self.N, 12).sum(axis=1)

    def position_moments(self, rho_diag: np.ndarray, xy: np.ndarray):
        w = self.cell_probability(rho_diag); w = w / w.sum()
        mean = w @ xy
        var = float(w @ ((xy - mean) ** 2).sum(axis=1))
        return mean, var

    def fisher(self, rho_diag: np.ndarray) -> float:
        """Fisher information of the cell law in the continuum normalisation, 16 sum over glue seams of
        (sqrt P_c - sqrt P_c')^2 (per mesh edge squared); (Var_x + Var_y) I / 4 >= 1."""
        f = np.sqrt(np.maximum(self.cell_probability(rho_diag), 0))
        glue = np.arange(self.geo.intra_seams, self.M)
        return 16.0 * float(((f[self.carrier_of[self.a[glue]]] - f[self.carrier_of[self.b[glue]]]) ** 2).sum())

    def momentum_law(self, psi: np.ndarray, xy: np.ndarray, patch: np.ndarray, kmax: float = 2.5, n: int = 61,
                     flat: float = 0.0):
        """Born law of the plane-wave analyser over the carriers ``patch``: |sum_c psi_c exp(-i k x_c)|^2 on an
        n x n grid of wave vectors, psi_c the cell amplitude. ``flat`` adds the incoherent weight of a state whose
        diagonal carries that probability without coherence (uniform over the grid). Returns grid, law, mean, var."""
        cell = psi.reshape(self.N, 12).sum(axis=1)[patch] / np.sqrt(12.0)
        ks = np.linspace(-kmax, kmax, n)
        ex = np.exp(-1j * np.outer(ks, xy[patch, 0])); ey = np.exp(-1j * np.outer(ks, xy[patch, 1]))
        amp = (ex[:, None, :] * ey[None, :, :]) @ cell
        law = np.abs(amp) ** 2
        coherent = float(law.sum())
        if coherent > 0:
            law = law / coherent * (1.0 - flat)
        law = law + flat / law.size
        KX, KY = np.meshgrid(ks, ks, indexing='ij')
        mean = np.array([float((law * KX).sum()), float((law * KY).sum())])
        var = float((law * ((KX - mean[0]) ** 2 + (KY - mean[1]) ** 2)).sum())
        return ks, law, mean, var


# ---------------------------------------------------------------------------- exact Kraus trajectories (tests, small graphs)
def kraus_attempt(psi: np.ndarray, a: int, b: int, coin: int, u: float):
    """One native attempt on the conditional pure state psi. Returns (new psi, current j)."""
    tested, other, j = (a, b, 1) if coin == 0 else (b, a, -1)
    hit = float(abs(psi[tested]) ** 2)
    if u < hit:
        out = np.zeros_like(psi); out[other] = 1.0
        return out, j
    out = psi.copy(); out[tested] = 0.0
    return out / np.linalg.norm(out), 0


def trajectory_coherence(lab: Lab, p: int, q: int, attempts: int, runs: int, seed: int = 0) -> np.ndarray:
    """Mean of rho_pq over record histories after 0..attempts native attempts, for the preparation
    (|p> + |q>) / sqrt 2. The unread channel predicts 0.5 (1 - (d_p + d_q) / (2 M))^n."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(attempts + 1)
    for _ in range(runs):
        psi = np.zeros(lab.P, np.complex128); psi[p] = psi[q] = 2 ** -0.5
        acc[0] += 0.5
        seams = rng.integers(0, lab.M, size=attempts); coins = rng.integers(0, 2, size=attempts); us = rng.random(attempts)
        alive = True
        for n in range(attempts):
            if alive:
                s = int(seams[n]); a, b = int(lab.a[s]), int(lab.b[s])
                if a in (p, q) or b in (p, q):
                    psi, _ = kraus_attempt(psi, a, b, int(coins[n]), float(us[n]))
                    alive = abs(psi[p]) > 0 and abs(psi[q]) > 0
            if alive:
                acc[n + 1] += float((psi[p] * np.conj(psi[q])).real)
    return acc / runs


# ---------------------------------------------------------------------------- the report
def report(level: int, width: float = 2.0, k0: float = 1.0, travel: float = 8.0) -> dict:
    lab = Lab(level)
    sigma = lab.conductivity()
    c_wave = float(np.sqrt(sigma / PORT_DENSITY))             # mesh edges per unit T
    mass = PORT_DENSITY / sigma                               # E = k^2 / (2 m) per sweep, k per mesh edge
    centre = 4 ** level - 1
    xy = lab.chart(centre)
    patch = lab.carriers_within(centre, min(0.6 * np.pi * lab.radius, 4 * width + travel + 4))
    k = np.array([k0, 0.0])
    psi0 = lab.packet(centre, width, k, xy)
    out = {'level': level, 'ports': lab.P, 'seams': lab.M, 'sphere_radius_edges': lab.radius, 'conductivity': sigma,
           'wave_speed_edges_per_unit_T': c_wave, 'transfer_mass': mass, 'packet_width_edges': width, 'packet_k_per_edge': k0}

    def state_row(name, psi, diag, flat, clock):
        mean, var = lab.position_moments(diag, xy)
        _, law, kmean, kvar = lab.momentum_law(psi, xy, patch, flat=flat)
        fisher = lab.fisher(diag)
        return {'law': name, **clock, 'position_mean': mean.tolist(), 'position_variance': var, 'k_mean': kmean.tolist(),
                'k_variance': kvar, 'uncertainty_product': float(np.sqrt(var * kvar)), 'fisher_product': float(var * fisher / 4)}

    rows = [state_row('preparation', psi0, np.abs(psi0) ** 2, 0.0, {'sweeps': 0.0})]
    # the admitted wave: travel `travel` mesh edges
    T = travel / c_wave
    sweeps_wave = T * np.sqrt(lab.M) / 6.0                   # N = T sqrt(M) walk steps, N = 6 s at a glued port
    psi = lab.wave_unitary(psi0, T)
    rows.append(state_row('wave', psi, np.abs(psi) ** 2, 0.0, {'T': T, 'walk_steps': T * np.sqrt(lab.M), 'sweeps': sweeps_wave}))
    # the transfer unitary: travel the same distance at group velocity k / m
    s_tr = travel * mass / k0
    psi = lab.transfer_unitary(psi0, s_tr)
    rows.append(state_row('transfer', psi, np.abs(psi) ** 2, 0.0, {'sweeps': s_tr}))
    # the native law at three times: one coherence time, the transfer time and the wave time
    for s in (0.2, 1.0, s_tr, sweeps_wave):
        diag = lab.native_diagonal(np.abs(psi0) ** 2, s)
        tilde = lab.native_coherent_part(psi0, s)
        rows.append(state_row('native', tilde, diag, 1.0 - float((np.abs(tilde) ** 2).sum()), {'sweeps': s}))
    out['states'] = rows
    # two paths: packets at -sep/2 and +sep/2 moving towards each other
    sep = travel
    left = int(np.argmin(((xy - np.array([-sep / 2, 0.0])) ** 2).sum(axis=1)))
    right = int(np.argmin(((xy - np.array([sep / 2, 0.0])) ** 2).sum(axis=1)))
    amp = (np.exp(-((xy - xy[left]) ** 2).sum(axis=1) / (4 * width ** 2)) * np.exp(1j * xy[:, 0] * k0)
           + np.exp(-((xy - xy[right]) ** 2).sum(axis=1) / (4 * width ** 2)) * np.exp(-1j * xy[:, 0] * k0))
    two = amp[lab.carrier_of].astype(np.complex128); two /= np.linalg.norm(two)

    def visibility(diag):
        """Fringe contrast of the cell law along x in the overlap strip |y| < width, |x| < width."""
        w = lab.cell_probability(diag)
        strip = np.flatnonzero((np.abs(xy[:, 1]) < width) & (np.abs(xy[:, 0]) < 1.5 * width))
        x = xy[strip, 0]; pw = w[strip]
        basis = np.stack([np.ones_like(x), np.cos(2 * k0 * x), np.sin(2 * k0 * x)], axis=1)
        env = np.exp(-x ** 2 / (2 * width ** 2))
        coef, *_ = np.linalg.lstsq(basis * env[:, None], pw, rcond=None)
        return float(np.hypot(coef[1], coef[2]) / abs(coef[0])) if coef[0] else 0.0

    T2 = (sep / 2) / c_wave
    s2 = T2 * np.sqrt(lab.M) / 6.0
    out['two_paths'] = {'separation_edges': sep, 'fringe_period_edges': float(np.pi / k0),
                        'wave': {'T': T2, 'sweeps': s2, 'visibility': visibility(np.abs(lab.wave_unitary(two, T2)) ** 2)},
                        'transfer': {'sweeps': (sep / 2) * mass / k0,
                                     'visibility': visibility(np.abs(lab.transfer_unitary(two, (sep / 2) * mass / k0)) ** 2)},
                        'native': {'sweeps': s2, 'visibility': visibility(lab.native_diagonal(np.abs(two) ** 2, s2))}}
    # coherence lifetime
    d = lab.deg
    out['coherence'] = {'decay_rate_per_sweep_min': float(d.min()), 'decay_rate_per_sweep_max': float(d.max()),
                        'lifetime_sweeps': [float(1 / d.max()), float(1 / d.min())],
                        'packet_coherent_weight_after_one_sweep': float((np.abs(lab.native_coherent_part(psi0, 1.0)) ** 2).sum())}
    # the observer as a detector: a packet `travel` edges from the observer
    src = int(np.argmin(((xy - np.array([travel, 0.0])) ** 2).sum(axis=1)))
    xs = lab.chart(src)
    p0 = np.abs(lab.packet(src, width, (0.0, 0.0), xs)) ** 2
    own = lab.ports_of([centre]); p0[own] = 0; p0 /= p0.sum()
    grid = np.geomspace(1.0, 64 * travel ** 2 / (4 * 0.5 * sigma / PORT_DENSITY), 60)
    surv, rate = lab.click_law(centre, p0, grid)
    half = float(np.interp(0.5, 1 - surv, grid)) if surv[-1] < 0.5 else None
    out['observer_detector'] = {'distance_edges': travel, 'sweeps': grid.tolist(), 'no_click_probability': surv.tolist(),
                                'click_rate_per_sweep': rate.tolist(), 'median_click_sweep': half,
                                'light_time_sweeps_wave_reading': travel / c_wave * np.sqrt(lab.M) / 6.0}
    return out


# ---------------------------------------------------------------------------- viewer export
def export(level: int, path: Path | None = None) -> Path:
    """Write the arrays the Detector tab of the viewer needs for one federation: seams, degrees, a chart about the
    root observer, the ports of that observer and the outside ends of its three glue seams."""
    lab = Lab(level)
    observer = 4 ** level - 1
    xy = lab.chart(observer)
    glue = np.arange(lab.geo.intra_seams, lab.M)
    ga, gb = lab.a[glue], lab.b[glue]
    ends = np.concatenate([gb[lab.carrier_of[ga] == observer], ga[lab.carrier_of[gb] == observer]])
    if path is None:
        root = Path.home() / '.cache' / 'oph_universe'
        root.mkdir(parents=True, exist_ok=True)
        path = root / f'lab_L{level}.npz'
    sigma = lab.conductivity()
    # the same chart for the mesh vertices, so that the viewer can draw every carrier as its triangle
    Xc = lab.geo.carrier_xyz / np.linalg.norm(lab.geo.carrier_xyz, axis=1, keepdims=True)
    n = Xc[observer]
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ref, n); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    V = lab.geo.mesh_xyz
    th = np.arccos(np.clip(V @ n, -1, 1)); t1, t2 = V @ e1, V @ e2
    nm = np.maximum(np.hypot(t1, t2), 1e-300)
    vxy = lab.radius * th[:, None] * np.stack([t1 / nm, t2 / nm], axis=1)
    np.savez_compressed(path, a=lab.a.astype(np.int32), b=lab.b.astype(np.int32), deg=lab.deg.astype(np.uint8),
                        chart=xy.astype(np.float32), triangles=vxy[lab.geo.tri].reshape(lab.N, 6).astype(np.float32), observer_ports=lab.ports_of([observer]).astype(np.int32),
                        observer_outside_ends=ends.astype(np.int32),
                        numbers=np.array([level, lab.P, lab.M, lab.N, observer, lab.radius, sigma, PORT_DENSITY], np.float64))
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--level', type=int, default=4)
    p.add_argument('--width', type=float, default=2.0)
    p.add_argument('--k', type=float, default=1.0)
    p.add_argument('--travel', type=float, default=8.0)
    p.add_argument('--json', type=Path, default=None)
    p.add_argument('--export', action='store_true', help='write the viewer file for this level and stop')
    a = p.parse_args(argv)
    if a.export:
        print(export(a.level))
        return
    r = report(a.level, a.width, a.k, a.travel)
    print(f"level {r['level']}: {r['ports']} ports, sphere radius {r['sphere_radius_edges']:.1f} edges, conductivity {r['conductivity']:.4f}, "
          f"wave speed {r['wave_speed_edges_per_unit_T']:.4f} edges per unit T, transfer mass {r['transfer_mass']:.1f}")
    print(f"{'law':>12}{'sweeps':>12}{'<x>':>8}{'<y>':>8}{'Var x':>9}{'<kx>':>8}{'Var k':>9}{'dx dk':>8}{'Fisher':>8}")
    for s in r['states']:
        print(f"{s['law']:>12}{s['sweeps']:12.1f}{s['position_mean'][0]:8.2f}{s['position_mean'][1]:8.2f}{s['position_variance']:9.2f}"
              f"{s['k_mean'][0]:8.3f}{s['k_variance']:9.3f}{s['uncertainty_product']:8.2f}{s['fisher_product']:8.2f}")
    t = r['two_paths']
    print(f"two paths {t['separation_edges']} edges apart, fringe period {t['fringe_period_edges']:.2f} edges: visibility wave {t['wave']['visibility']:.3f}, "
          f"transfer {t['transfer']['visibility']:.3f}, native {t['native']['visibility']:.4f} (after {t['native']['sweeps']:.0f} sweeps)")
    c = r['coherence']
    print(f"native coherence between two ports decays at {c['decay_rate_per_sweep_min']:.0f} to {c['decay_rate_per_sweep_max']:.0f} per sweep; "
          f"coherent weight of the packet after one sweep {c['packet_coherent_weight_after_one_sweep']:.2e}")
    o = r['observer_detector']
    print(f"observer as detector, packet {o['distance_edges']} edges away: median click at sweep {o['median_click_sweep']}, "
          f"no-click probability at the last time {o['no_click_probability'][-1]:.3f}; light time in the wave reading {o['light_time_sweeps_wave_reading']:.0f} sweeps")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(r))


if __name__ == '__main__':
    main()
