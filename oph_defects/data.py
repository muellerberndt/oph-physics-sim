"""Readers and exact replay for federation runs (schema in SCHEMA.md).

    from oph_defects import data
    run = data.Run('runs/lawC_defects_L8')
    cat = run.catalogue()                       # one row per defect identity
    snap = run.snapshot(10, tier='macro')       # faces, classes, ids, xyz at a snapshot
    ck = run.checkpoint(25)                     # loads, register, classes at sweep 25
    for sweep, rec in run.replay(25, 30):       # exact per-attempt records of sweeps 25..29
        ...
    tape = run.observer_tape(carrier=4711, sweeps=range(25, 30))
"""
from __future__ import annotations

import ctypes
import gzip
import json
from pathlib import Path
import struct

import numpy as np

from oph_exact import federation as F
from . import federation_geometry as G
from .federation_run import COST, kernel

CAT_COLUMNS = (('cls', 'u1'), ('birth', '<i4'), ('death', '<i4'), ('bface', '<i4'), ('lface', '<i4'), ('hops', '<i4'),
               ('partner', '<i4'), ('p1', '<i4'), ('p2', '<i4'), ('cause', 'u1'), ('c1', '<i4'), ('c2', '<i4'))


class Run:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / 'run.json').read_text())
        self.level = self.meta['geometry']['level']
        self._geom = None

    # ---------------------------------------------------------------- geometry
    @property
    def geometry(self):
        """(ExactFederation, FederationFaces): seams, carriers, faces, positions. Built on first use."""
        if self._geom is None:
            self._geom = G.build(self.level)
        return self._geom

    # ---------------------------------------------------------------- catalogue
    def catalogue(self) -> dict[str, np.ndarray]:
        raw = gzip.open(self.path / 'catalogue.bin.gz', 'rb').read()
        assert raw[:4] == b'Z6CT'
        n = struct.unpack('<I', raw[4:8])[0]
        o, cols = 8, {}
        for name, dtype in CAT_COLUMNS:
            size = np.dtype(dtype).itemsize
            cols[name] = np.frombuffer(raw[o:o + size * n], dtype)
            o += size * n
        xyz = self.path / 'catalogue_xyz.bin.gz'
        if xyz.exists():
            raw = gzip.open(xyz, 'rb').read()
            cols['birth_xyz'] = np.frombuffer(raw[8:8 + 12 * n], '<f4').reshape(n, 3)
            cols['last_xyz'] = np.frombuffer(raw[8 + 12 * n:8 + 24 * n], '<f4').reshape(n, 3)
        intra = self.meta['geometry']['intra_faces']
        cols['tier'] = np.where(cols['bface'] < intra, 'internal', 'screen')
        return cols

    # ---------------------------------------------------------------- snapshots
    def snapshots(self):
        return self.meta['snapshots']

    def snapshot(self, index: int, tier: str = 'macro') -> dict:
        entry = self.meta['snapshots'][index]
        name = entry['file'] if tier == 'macro' else entry.get('intra_file')
        if name is None:
            raise ValueError(f'snapshot {index} has no {tier} file')
        raw = gzip.open(self.path / name, 'rb').read()
        assert raw[:4] == b'Z6SX'
        tick, n = struct.unpack('<II', raw[4:12])
        o = 12
        faces = np.frombuffer(raw[o:o + 4 * n], '<i4'); o += 4 * n
        classes = np.frombuffer(raw[o:o + n], 'u1'); o += n
        ids = np.frombuffer(raw[o:o + 4 * n], '<i4'); o += 4 * n
        xyz = np.frombuffer(raw[o:o + 12 * n], '<f4').reshape(n, 3)
        return {'tick': tick, 'sweep': tick // self.meta['config']['chunks_per_sweep'], 'faces': faces,
                'classes': classes, 'ids': ids, 'xyz': xyz}

    # ---------------------------------------------------------------- checkpoints
    def checkpoints(self):
        return self.meta['checkpoints']

    def checkpoint(self, sweep: int):
        entry = next(c for c in self.meta['checkpoints'] if c['sweep'] == sweep)
        ck = np.load(self.path / entry['file'], allow_pickle=False)
        rng = json.loads((self.path / entry['file']).with_suffix('.json').read_text())
        return {'sweep': sweep, 'loads': ck['loads'], 'register': ck['register'], 'classes': ck['classes'],
                'activity': ck['activity'], 'face_id': ck['face_id'], 'law_rng': rng['law_rng'],
                'register_rng': rng['register_rng']}

    # ---------------------------------------------------------------- replay
    def replay(self, start: int, stop: int, with_register: bool = True):
        """Yield (sweep, records) for sweeps start..stop-1, replayed exactly from the nearest checkpoint.

        records: dict with 'seam' (int64, attempt order), 'coin' (int64), 'current' (int8, the units
        moved from seam end a to end b, 0 for a hold), and 'loads_after' (the loads at the end of the sweep).
        The register temperature and acceptance are replayed too, so classes stay in step.
        """
        fed, fc = self.geometry
        cfg = self.meta['config']
        base = max(c['sweep'] for c in self.meta['checkpoints'] if c['sweep'] <= start)
        ck = self.checkpoint(base)
        x = ck['loads'].astype(np.int64)
        reg = ck['register'].copy()
        cls = ck['classes'].copy()
        activity = ck['activity'].astype(np.float64)
        rng = np.random.default_rng(0); rng.bit_generator.state = ck['law_rng']
        reg_rng = np.random.default_rng(0); reg_rng.bit_generator.state = ck['register_rng']
        lib = kernel()
        S = fed.seams
        seam_carrier = np.ascontiguousarray(fc.seam_carrier, dtype=np.int32)
        attempts = np.bincount(seam_carrier, minlength=fed.carriers).astype(np.float64)
        fplus = np.ascontiguousarray(fc.seam_plus, dtype=np.int64)
        fminus = np.ascontiguousarray(fc.seam_minus, dtype=np.int64)
        a_arr = np.ascontiguousarray(fed.seam_a, dtype=np.int64)
        b_arr = np.ascontiguousarray(fed.seam_b, dtype=np.int64)
        cost = np.ascontiguousarray(np.array(cfg['cost']), dtype=np.int64)
        temp = np.empty(S, np.float32)
        carrier_desc = np.zeros(fed.carriers, np.int32)
        events = np.zeros((1 << 20, 6), np.int32)
        counts = np.zeros(7, np.int64)
        cur = np.zeros(S, np.int8)
        chunks = cfg['chunks_per_sweep']
        T0 = cfg['T0']
        nI = fc.intra_seams
        i64p, i8p, i32p, f32p = (ctypes.POINTER(t) for t in (ctypes.c_int64, ctypes.c_int8, ctypes.c_int32, ctypes.c_float))
        for s in range(base, stop):
            ta = activity[seam_carrier]
            ta[nI:] = 0.5 * (activity[fed.inter_pairs[:, 0]] + activity[fed.inter_pairs[:, 2]])
            temp[:] = (T0 * ta).astype(np.float32)
            seq = rng.integers(0, S, size=S, dtype=np.int64)
            coin = rng.integers(0, 2, size=S, dtype=np.int64)
            carrier_desc[:] = 0
            step = (S + chunks - 1) // chunks
            for k in range(chunks):
                lo, hi = k * step, min(S, (k + 1) * step)
                if hi <= lo:
                    break
                counts[5] = 0; counts[6] = 0
                lib.oph_reg_integer(x.ctypes.data_as(i64p), a_arr.ctypes.data_as(i64p), b_arr.ctypes.data_as(i64p),
                                    seq[lo:hi].ctypes.data_as(i64p), coin[lo:hi].ctypes.data_as(i64p), hi - lo,
                                    reg.ctypes.data_as(i8p), cls.ctypes.data_as(i8p), fplus.ctypes.data_as(i64p),
                                    fminus.ctypes.data_as(i64p), cost.ctypes.data_as(i64p), temp.ctypes.data_as(f32p),
                                    seam_carrier.ctypes.data_as(i32p), carrier_desc.ctypes.data_as(i32p),
                                    ctypes.c_uint64(int(reg_rng.integers(0, 2 ** 63))), events.ctypes.data_as(i32p),
                                    events.shape[0], counts.ctypes.data_as(i64p), cur[lo:hi].ctypes.data_as(i8p))
            activity = carrier_desc / attempts
            if s >= start:
                yield s, {'seam': seq, 'coin': coin, 'current': cur.copy(), 'loads_after': x.copy(),
                          'register_after': reg.copy(), 'classes_after': cls.copy()}

    def records(self, sweep: int):
        """Dense per-attempt records stored for a window sweep: seam, coin, current."""
        entry = next(r for r in self.meta['record_windows'] if r['sweep'] == sweep)
        rng_state = json.loads((self.path / entry['rng']).read_text())
        rng = np.random.default_rng(0); rng.bit_generator.state = rng_state['law_rng']
        n = entry['attempts']
        seam = rng.integers(0, n, size=n, dtype=np.int64)
        coin = rng.integers(0, 2, size=n, dtype=np.int64)
        with gzip.open(self.path / entry['currents'], 'rb') as fh:
            current = np.load(fh)
        return {'seam': seam, 'coin': coin, 'current': current}

    def observer_tape(self, carrier: int, sweeps):
        """Records on the seams of one carrier, in attempt order: (sweep, attempt, seam, coin, current)."""
        fed, fc = self.geometry
        own = np.flatnonzero(fc.seam_carrier == carrier)
        inter_right = fed.inter_pairs[:, 2] == carrier
        own = np.union1d(own, fc.intra_seams + np.flatnonzero(inter_right))
        mask = np.zeros(fed.seams, bool); mask[own] = True
        rows = []
        stored = {r['sweep'] for r in self.meta['record_windows']}
        for s in sweeps:
            rec = self.records(s) if s in stored else None
            if rec is None:
                rec = next(r for _, r in self.replay(s, s + 1))
            hit = np.flatnonzero(mask[rec['seam']])
            rows.append(np.stack([np.full(hit.size, s), hit, rec['seam'][hit], rec['coin'][hit], rec['current'][hit]], axis=1))
        return np.concatenate(rows) if rows else np.zeros((0, 5), np.int64)
