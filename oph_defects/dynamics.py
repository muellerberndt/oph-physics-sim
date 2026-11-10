"""Z6 link register with single-seam repair proposals and complete defect tracking.

State: one Z6 element per seam. A face is a defect when the loop sum of its
three seams is nonzero; the loop sum is its class ``k`` in 1..5. A proposal
changes one seam by a Z6 element ``delta`` (``+-1`` most often). Because every
seam is traversed once in each direction by its two faces, the change adds
``delta`` to one face and ``-delta`` to the other, so the total class over the
closed surface stays ``0 mod 6``: defects are born in conjugate pairs
``(k, 6-k)``, move one face per accepted proposal (a class-``k`` defect moves
when its seam changes by ``-k``), fuse additively and annihilate.

Each defect that ever exists receives an identity. Births, moves, fusions,
splits, exchanges and annihilations are recorded per identity, so the
catalogue is complete rather than reconstructed from snapshots.

A proposal is accepted when it lowers the declared cost
``sum_f cost[class_f]`` (cost 0,1,2,3,2,1 for classes 0..5) and otherwise
with probability ``exp(-dE / T)`` at the scheduled temperature ``T``. The
proposal fraction, the cost table, the schedule and the seed are declared
inputs of the run; the carrier is the refined certified twelve-port carrier.
Nothing here is the canonical scalar repair law of ``oph_exact``: the link
register is the declared curvature-defect extension.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math

import numpy as np

from .geometry import Surface, face_curvature

COST = np.array([0, 1, 2, 3, 2, 1], dtype=np.int64)
# A proposal adds one of these Z6 elements to a seam: +-1 most often, +-2 and 3 less often.
# The weights are symmetric under negation, so Metropolis acceptance keeps detailed balance.
DELTAS = np.array([1, 2, 3, 4, 5], dtype=np.int64)
DELTA_WEIGHTS = np.array([0.35, 0.10, 0.10, 0.10, 0.35])
# exact sampling of those weights: twenty equiprobable slots
DELTA_TABLE = np.array([1] * 7 + [2] * 2 + [3] * 2 + [4] * 2 + [5] * 7, dtype=np.int64)
CAUSE = {'alive': 0, 'annihilation': 1, 'fusion': 2, 'split': 3, 'exchange': 4}


def temperature_schedule(knots: list[tuple[int, float]], ticks: int) -> np.ndarray:
    """Piecewise-linear temperature per tick from (tick, T) knots."""
    xs = np.array([k[0] for k in knots], dtype=np.float64)
    ys = np.array([k[1] for k in knots], dtype=np.float64)
    return np.interp(np.arange(ticks, dtype=np.float64), xs, ys)


class Catalogue:
    """Growable per-identity columns."""
    COLUMNS = (('cls', np.int8), ('birth', np.int32), ('death', np.int32), ('bface', np.int32),
               ('lface', np.int32), ('hops', np.int32), ('partner', np.int32), ('p1', np.int32),
               ('p2', np.int32), ('cause', np.int8), ('c1', np.int32), ('c2', np.int32))

    def __init__(self, capacity: int = 1 << 16):
        self.n = 0
        self.cols = {name: np.full(capacity, -1, dtype=dt) for name, dt in self.COLUMNS}
        self.cols['hops'][:] = 0
        self.cols['cause'][:] = 0

    def _grow(self, extra: int):
        need = self.n + extra
        cap = len(self.cols['cls'])
        if need <= cap:
            return
        while cap < need:
            cap *= 2
        for name, dt in self.COLUMNS:
            new = np.full(cap, -1 if name not in ('hops', 'cause') else 0, dtype=dt)
            new[:self.n] = self.cols[name][:self.n]
            self.cols[name] = new

    def allocate(self, count: int) -> np.ndarray:
        self._grow(count)
        ids = np.arange(self.n, self.n + count, dtype=np.int32)
        self.n += count
        return ids

    def view(self) -> dict[str, np.ndarray]:
        return {name: self.cols[name][:self.n] for name, _ in self.COLUMNS}


@dataclass
class RunStats:
    temperature: list = field(default_factory=list)
    proposals: list = field(default_factory=list)
    kept: list = field(default_factory=list)
    accepted: list = field(default_factory=list)
    births: list = field(default_factory=list)
    moves: list = field(default_factory=list)
    annihilations: list = field(default_factory=list)
    fusions: list = field(default_factory=list)
    splits: list = field(default_factory=list)
    exchanges: list = field(default_factory=list)
    counts: list = field(default_factory=list)   # per tick: classes 0..5
    cost: list = field(default_factory=list)


class Z6Run:
    def __init__(self, surface: Surface, seed: int, proposal_fraction: float, cost: np.ndarray = COST):
        self.s = surface
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.seed = seed
        self.p = proposal_fraction
        self.cost = np.asarray(cost, dtype=np.int64)
        self.links = np.zeros(surface.edge_count, dtype=np.int8)
        self.curv = np.zeros(surface.face_count, dtype=np.int8)
        self.face_id = np.full(surface.face_count, -1, dtype=np.int32)
        self.cat = Catalogue()
        self.stats = RunStats()
        self.tick = 0
        self.snapshots: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []

    # ------------------------------------------------------------ bookkeeping
    def _birth(self, faces: np.ndarray, classes: np.ndarray, partner_of_pair: bool, p1=None):
        ids = self.cat.allocate(len(faces))
        c = self.cat.cols
        c['cls'][ids] = classes
        c['birth'][ids] = self.tick
        c['bface'][ids] = faces
        c['lface'][ids] = faces
        if p1 is not None:
            c['p1'][ids] = p1
        self.face_id[faces] = ids
        return ids

    def _die(self, ids: np.ndarray, cause: int, partner=None):
        c = self.cat.cols
        c['death'][ids] = self.tick
        c['cause'][ids] = cause
        if partner is not None:
            c['partner'][ids] = partner

    def step(self, T: float):
        s, rng = self.s, self.rng
        m = int(round(self.p * s.edge_count))
        e = rng.integers(0, s.edge_count, m, dtype=np.int64)
        delta = DELTA_TABLE[rng.integers(0, len(DELTA_TABLE), m)]
        fp = s.edge_plus[e].astype(np.int64)
        fm = s.edge_minus[e].astype(np.int64)
        touched = np.bincount(np.concatenate([fp, fm]), minlength=s.face_count)
        keep = (touched[fp] == 1) & (touched[fm] == 1)
        e, delta, fp, fm = e[keep], delta[keep], fp[keep], fm[keep]
        k1 = self.curv[fp].astype(np.int64)
        k2 = self.curv[fm].astype(np.int64)
        n1 = (k1 + delta) % 6
        n2 = (k2 - delta) % 6
        dE = self.cost[n1] + self.cost[n2] - self.cost[k1] - self.cost[k2]
        u = rng.random(len(e))
        accept = dE <= 0
        if T > 0:
            uphill = ~accept
            accept = accept | (uphill & (u < np.exp(-np.where(uphill, dE, 0) / T)))
        e, delta, fp, fm, k1, k2, n1, n2 = (x[accept] for x in (e, delta, fp, fm, k1, k2, n1, n2))
        # apply
        self.links[e] = ((self.links[e].astype(np.int64) + delta) % 6).astype(np.int8)
        self.curv[fp] = n1.astype(np.int8)
        self.curv[fm] = n2.astype(np.int8)
        # categories
        o1, o2, w1, w2 = k1 != 0, k2 != 0, n1 != 0, n2 != 0
        birth = ~o1 & ~o2
        move12 = o1 & ~o2 & ~w1
        move21 = ~o1 & o2 & ~w2
        split1 = o1 & ~o2 & w1
        split2 = ~o1 & o2 & w2
        both = o1 & o2
        annih = both & ~w1 & ~w2
        merge2 = both & ~w1 & w2
        merge1 = both & w1 & ~w2
        exch = both & w1 & w2
        c = self.cat.cols
        nb = int(birth.sum())
        if nb:
            ids = self.cat.allocate(2 * nb)
            a, b = ids[:nb], ids[nb:]
            for idx, faces, classes in ((a, fp[birth], n1[birth]), (b, fm[birth], n2[birth])):
                c['cls'][idx] = classes
                c['birth'][idx] = self.tick
                c['bface'][idx] = faces
                c['lface'][idx] = faces
                self.face_id[faces] = idx
            c['partner'][a] = b
            c['partner'][b] = a
        for mask, src, dst in ((move12, fp, fm), (move21, fm, fp)):
            if mask.any():
                ids = self.face_id[src[mask]]
                self.face_id[dst[mask]] = ids
                self.face_id[src[mask]] = -1
                c['hops'][ids] += 1
                c['lface'][ids] = dst[mask]
        for mask, home, away, nh, na in ((split1, fp, fm, n1, n2), (split2, fm, fp, n2, n1)):
            if mask.any():
                old = self.face_id[home[mask]]
                self._die(old, CAUSE['split'])
                kids = self.cat.allocate(2 * int(mask.sum()))
                h, w = kids[:len(old)], kids[len(old):]
                for idx, faces, classes in ((h, home[mask], nh[mask]), (w, away[mask], na[mask])):
                    c['cls'][idx] = classes
                    c['birth'][idx] = self.tick
                    c['bface'][idx] = faces
                    c['lface'][idx] = faces
                    c['p1'][idx] = old
                    self.face_id[faces] = idx
                c['partner'][h] = w
                c['partner'][w] = h
                c['c1'][old] = h
                c['c2'][old] = w
        if annih.any():
            i1, i2 = self.face_id[fp[annih]], self.face_id[fm[annih]]
            self._die(i1, CAUSE['annihilation'], i2)
            self._die(i2, CAUSE['annihilation'], i1)
            self.face_id[fp[annih]] = -1
            self.face_id[fm[annih]] = -1
        for mask, dst, src, nd in ((merge2, fm, fp, n2), (merge1, fp, fm, n1)):
            if mask.any():
                i1, i2 = self.face_id[src[mask]], self.face_id[dst[mask]]
                self._die(i1, CAUSE['fusion'], i2)
                self._die(i2, CAUSE['fusion'], i1)
                new = self._birth(dst[mask], nd[mask], False, p1=i1)
                c['p2'][new] = i2
                c['c1'][i1] = new
                c['c1'][i2] = new
                self.face_id[src[mask]] = -1
        if exch.any():
            i1, i2 = self.face_id[fp[exch]], self.face_id[fm[exch]]
            self._die(i1, CAUSE['exchange'], i2)
            self._die(i2, CAUSE['exchange'], i1)
            new1 = self._birth(fp[exch], n1[exch], False, p1=i1)
            new2 = self._birth(fm[exch], n2[exch], False, p1=i2)
            c['p2'][new1] = i2
            c['p2'][new2] = i1
            c['c1'][i1] = new1
            c['c2'][i1] = new2
            c['c1'][i2] = new1
            c['c2'][i2] = new2
        st = self.stats
        st.temperature.append(float(T))
        st.proposals.append(m)
        st.kept.append(int(keep.sum()))
        st.accepted.append(int(accept.sum()))
        st.births.append(nb)
        st.moves.append(int(move12.sum() + move21.sum()))
        st.annihilations.append(int(annih.sum()))
        st.fusions.append(int(merge1.sum() + merge2.sum()))
        st.splits.append(int(split1.sum() + split2.sum()))
        st.exchanges.append(int(exch.sum()))
        counts = np.bincount(self.curv, minlength=6)
        st.counts.append(counts.tolist())
        st.cost.append(int((self.cost[counts.nonzero()[0]] * counts[counts.nonzero()[0]]).sum()))
        self.tick += 1

    def snapshot(self):
        faces = np.flatnonzero(self.curv).astype(np.int32)
        self.snapshots.append((self.tick, faces, self.curv[faces].copy(), self.face_id[faces].copy()))

    # ------------------------------------------------------------ invariants
    def check(self) -> dict:
        """Exact invariants: recomputed curvature, zero total class, identity consistency."""
        recomputed = face_curvature(self.s, self.links)
        assert (recomputed == self.curv).all(), 'curvature register drifted from the links'
        assert int(self.curv.astype(np.int64).sum()) % 6 == 0, 'total class is not zero mod 6'
        live = self.face_id >= 0
        assert (live == (self.curv != 0)).all(), 'identity map disagrees with the curvature'
        c = self.cat.view()
        ids = self.face_id[live]
        assert (c['death'][ids] == -1).all(), 'a dead identity sits on a face'
        assert (c['cls'][ids] == self.curv[live]).all(), 'identity class disagrees with the face'
        assert (c['lface'][ids] == np.flatnonzero(live)).all(), 'last face is stale'
        dead = c['death'] >= 0
        assert int(dead.sum()) + int(live.sum()) == self.cat.n, 'identities lost'
        return {'live': int(live.sum()), 'identities': int(self.cat.n), 'cost': int(self.stats.cost[-1]) if self.stats.cost else 0}


def region_check(run: Z6Run, rng: np.random.Generator, trials: int = 8) -> None:
    """The boundary loop sum of a random region equals the sum of its classes mod 6."""
    s = run.s
    for _ in range(trials):
        seed_face = int(rng.integers(0, s.face_count))
        region = {seed_face}
        frontier = [seed_face]
        for _ in range(int(rng.integers(1, 6))):
            new = []
            for f in frontier:
                for e in s.face_edges[f]:
                    g = int(s.edge_plus[e]) if int(s.edge_minus[e]) == f else int(s.edge_minus[e])
                    if g not in region:
                        region.add(g)
                        new.append(g)
            frontier = new
        faces = np.array(sorted(region))
        inside = int(run.curv[faces].astype(np.int64).sum()) % 6
        edges = s.face_edges[faces].ravel()
        signs = s.face_signs[faces].ravel().astype(np.int64)
        counts = np.bincount(edges, minlength=s.edge_count)
        boundary = counts[edges] == 1
        loop = int((signs[boundary] * run.links[edges[boundary]].astype(np.int64)).sum()) % 6
        assert loop == inside, (loop, inside)


def run_config(level: int, ticks: int, seed: int, p: float, knots, snapshot_every: int) -> dict:
    return {
        'law': 'Z6 link register, single-seam +-1 proposals, Metropolis acceptance on the declared cost',
        'cost': COST.tolist(), 'level': level, 'ticks': ticks, 'seed': seed,
        'proposal_fraction': p, 'temperature_knots': [list(k) for k in knots],
        'proposal_deltas': DELTAS.tolist(), 'proposal_weights': DELTA_WEIGHTS.tolist(),
        'snapshot_every': snapshot_every,
        'carrier': 'certified twelve-port icosahedral carrier, refined by edge midpoints',
        'status': 'declared curvature-defect extension; the canonical scalar repair law carries no loop invariant',
    }
