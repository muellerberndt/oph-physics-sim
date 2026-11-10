"""Defect identities from the ordered register events of a run.

A defect is a face of nonzero class. Every accepted twist of a seam changes
the classes of its two faces; the event (pre and post classes of both faces)
decides what happened to the identities on them:

    both flat before            a conjugate pair is born
    one occupied, it empties    the defect moves to the other face
    one occupied, it stays      it splits into two new identities
    both occupied, both empty   annihilation
    both occupied, one empties  fusion into one new identity
    both occupied, both stay    exchange: two new identities

The loop runs in C (``kernels.oph_track``) in the order of the schedule.
For every open consistent cut it also copies the class and the identity of
the two faces of each event whose depth does not exceed the cut depth.
"""
from __future__ import annotations

import ctypes

import numpy as np

from . import kernels as K

COLUMNS = (('cls', np.uint8), ('birth_sweep', np.int32), ('birth_depth', np.int32), ('death_sweep', np.int32),
           ('death_depth', np.int32), ('birth_face', np.int32), ('last_face', np.int32), ('hops', np.int32),
           ('partner', np.int32), ('parent1', np.int32), ('parent2', np.int32), ('cause', np.uint8),
           ('child1', np.int32), ('child2', np.int32))
CAUSES = ('alive', 'annihilation', 'fusion', 'split', 'exchange')
STATS = ('births', 'moves', 'annihilations', 'fusions', 'splits', 'exchanges')


class Tracker:
    def __init__(self, first_face: int, tracked_faces: int, slots: int, capacity: int = 1 << 16):
        self.F0 = int(first_face)
        self.FT = int(tracked_faces)
        self.face_id = np.full(self.FT, -1, np.int32)
        self.n = 0
        self.cap = int(capacity)
        self.cols = {name: np.zeros(self.cap, dt) for name, dt in COLUMNS}
        self.stats = np.zeros(6, np.int64)
        self.slots = int(slots)
        self.cut_cls = np.zeros((self.slots, self.FT), np.int8)
        self.cut_fid = np.full((self.slots, self.FT), -1, np.int32)
        self._n = np.zeros(1, np.int64)
        self.lib = K.library()

    def _grow(self):
        self.cap *= 2
        for name, dt in COLUMNS:
            new = np.zeros(self.cap, dt)
            new[:self.n] = self.cols[name][:self.n]
            self.cols[name] = new

    def process(self, events: np.ndarray, sweep: int, cutT: np.ndarray, cut_slot: np.ndarray, ncut: int):
        """events: int32 [m, 8] rows (pos, fp, fm, k1, k2, n1, n2, depth) in schedule order."""
        m = int(events.shape[0])
        if m == 0:
            return
        ev = np.ascontiguousarray(events, dtype=np.int32)
        done = 0
        while done < m:
            self._n[0] = self.n
            c = self.cols
            got = self.lib.oph_track(
                ev.ctypes.data + 32 * done, m - done, ctypes.c_int32(sweep), self.F0, K.ptr(self.face_id),
                K.ptr(c['cls']), K.ptr(c['birth_sweep']), K.ptr(c['birth_depth']), K.ptr(c['death_sweep']),
                K.ptr(c['death_depth']), K.ptr(c['birth_face']), K.ptr(c['last_face']), K.ptr(c['hops']),
                K.ptr(c['partner']), K.ptr(c['parent1']), K.ptr(c['parent2']), K.ptr(c['cause']),
                K.ptr(c['child1']), K.ptr(c['child2']), K.ptr(self._n), self.cap,
                ctypes.c_int32(ncut), K.ptr(cutT), K.ptr(cut_slot), K.ptr(self.cut_cls), K.ptr(self.cut_fid),
                self.FT, K.ptr(self.stats))
            self.n = int(self._n[0])
            done += int(got)
            if done < m:
                self._grow()

    def seed(self, faces: np.ndarray, classes: np.ndarray):
        """Identities for defects that exist in the preparation (birth sweep -1, depth 0, no parents)."""
        for f, k in zip(np.asarray(faces).tolist(), np.asarray(classes).tolist()):
            if self.n >= self.cap:
                self._grow()
            i = self.n
            self.n += 1
            c = self.cols
            c['cls'][i] = np.uint8(k & 0xFF); c['birth_sweep'][i] = -1; c['birth_depth'][i] = 0
            c['death_sweep'][i] = -1; c['death_depth'][i] = -1; c['birth_face'][i] = f; c['last_face'][i] = f
            c['hops'][i] = 0; c['cause'][i] = 0
            for name in ('partner', 'parent1', 'parent2', 'child1', 'child2'):
                c[name][i] = -1
            self.face_id[f - self.F0] = i

    def open_cut(self, slot: int, classes_tracked: np.ndarray):
        self.cut_cls[slot] = classes_tracked
        self.cut_fid[slot] = self.face_id

    def view(self) -> dict[str, np.ndarray]:
        return {name: self.cols[name][:self.n] for name, _ in COLUMNS}

    # ------------------------------------------------------------ persistence
    def state(self) -> dict[str, np.ndarray]:
        out = {'cat_' + name: self.cols[name][:self.n].copy() for name, _ in COLUMNS}
        out['face_id'] = self.face_id.copy()
        out['tracker_stats'] = self.stats.copy()
        return out

    def restore(self, payload):
        n = int(payload['cat_cls'].shape[0])
        while self.cap < n + 2:
            self.cap *= 2
        self.cols = {name: np.zeros(self.cap, dt) for name, dt in COLUMNS}
        for name, _ in COLUMNS:
            self.cols[name][:n] = payload['cat_' + name]
        self.n = n
        self.face_id[:] = payload['face_id']
        self.stats[:] = payload['tracker_stats']
