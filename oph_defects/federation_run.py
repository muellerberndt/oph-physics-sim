"""Law C on the exact federation with a guarded transport register.

    python -m oph_defects.federation_run --level 8 --sweeps 300 --out runs/lawC_defects_L8

Loads follow law C exactly as ``oph_exact.federation`` runs it (same carrier,
gluing, initial loads, schedule, coin draws and move arithmetic; the final
loads of this program equal those of ``oph_exact.federation.run_integer_law``
for the same seed and sweep count). The register is a declared addition:
every seam carries a Z6 element, and each unit that law C moves across a
seam proposes to add its signed current to that seam's element. The
proposal changes the two faces of the seam by ``+delta`` and ``-delta`` in
each tier (twenty template faces per carrier; one dual face per screen
vertex). It is accepted when the declared face cost does not rise and
otherwise with probability ``exp(-dE/T)``, where ``T`` is ``T0`` times the
descent activity of the seam's carriers in the previous sweep: hot while
the universe settles, zero once only tie swaps remain. Every accepted
register change is logged and tracked into defect identities.
"""
from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

from oph_exact import federation as F
from . import federation_geometry as G

COST = np.array([0, 5, 8, 10, 8, 5], dtype=np.int64)
CAUSE = {'alive': 0, 'annihilation': 1, 'fusion': 2, 'split': 3, 'exchange': 4}

KERNEL = r"""
#include <stdint.h>
#include <math.h>
static inline uint64_t sm64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
/* counts = {descents, swaps, waits, register accepted, register rejected, events, overflow} */
void oph_reg_integer(int64_t *x, const int64_t *a, const int64_t *b, const int64_t *seq, const int64_t *coin,
                     int64_t n, int8_t *reg, int8_t *cls, const int64_t *fplus, const int64_t *fminus,
                     const int64_t *cost, const float *temp, const int32_t *seam_carrier, int32_t *carrier_desc,
                     uint64_t rng_seed, int32_t *events, int64_t ev_cap, int64_t *counts, int8_t *cur_out)
{
    uint64_t rs = rng_seed;
    int64_t descents = 0, swaps = 0, waits = 0, acc = 0, rej = 0, ev = counts[5];
    for (int64_t t = 0; t < n; ++t) {
        const int64_t s = seq[t];
        const int64_t i = a[s], j = b[s];
        const int64_t xi = x[i], xj = x[j];
        const int64_t tot = xi + xj;
        const int64_t lo = tot >> 1;
        const int64_t hi = tot - lo;
        const int64_t ni = coin[t] ? hi : lo;
        const int64_t nj = tot - ni;
        const int64_t d = xi - xj;
        if (d == 0) { ++waits; if (cur_out) cur_out[t] = 0; continue; }
        if (d == 1 || d == -1) {
            if (ni == xi) { ++waits; if (cur_out) cur_out[t] = 0; continue; }
            ++swaps;
        } else { ++descents; carrier_desc[seam_carrier[s]] += 1; }
        x[i] = ni; x[j] = nj;
        /* transport register: the current moved from a to b */
        const int64_t cur = xi - ni;
        if (cur_out) cur_out[t] = (int8_t)(cur > 127 ? 127 : (cur < -127 ? -127 : cur));
        int64_t delta = ((cur % 6) + 6) % 6;
        if (delta == 0) continue;
        const int64_t fp = fplus[s], fm = fminus[s];
        const int64_t k1 = cls[fp], k2 = cls[fm];
        const int64_t n1 = (k1 + delta) % 6, n2 = ((k2 - delta) % 6 + 6) % 6;
        const int64_t dE = cost[n1] + cost[n2] - cost[k1] - cost[k2];
        int ok = dE <= 0;
        if (!ok) {
            const float T = temp[s];
            if (T > 0.0f) {
                const double u = (double)(sm64(&rs) >> 11) * (1.0 / 9007199254740992.0);
                ok = u < exp(-(double)dE / (double)T);
            }
        }
        if (!ok) { ++rej; continue; }
        ++acc;
        reg[s] = (int8_t)((reg[s] + delta) % 6);
        cls[fp] = (int8_t)n1; cls[fm] = (int8_t)n2;
        if (ev < ev_cap) {
            int32_t *e = events + 6 * ev;
            e[0] = (int32_t)fp; e[1] = (int32_t)fm; e[2] = (int32_t)k1; e[3] = (int32_t)k2; e[4] = (int32_t)n1; e[5] = (int32_t)n2;
            ++ev;
        } else { counts[6] = 1; }
    }
    counts[0] += descents; counts[1] += swaps; counts[2] += waits; counts[3] += acc; counts[4] += rej; counts[5] = ev;
}
"""

_LIB = {'lib': None}


def kernel():
    if _LIB['lib'] is not None:
        return _LIB['lib']
    digest = hashlib.sha256(KERNEL.encode()).hexdigest()[:16]
    directory = Path(os.environ.get('OPH_EXACT_KERNEL_DIR', Path.home() / '.cache' / 'oph_exact_kernels'))
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / f'oph_reg_kernel_{digest}.so'
    if not library.exists():
        compiler = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
        source = directory / f'oph_reg_kernel_{digest}.c'
        source.write_text(KERNEL)
        staging = directory / f'oph_reg_kernel_{digest}.{os.getpid()}.so'
        subprocess.run([compiler, '-O2', '-ffp-contract=off', '-shared', '-fPIC', '-o', str(staging), str(source)],
                       check=True, capture_output=True)
        os.replace(staging, library)
    lib = ctypes.CDLL(str(library))
    i64p = ctypes.POINTER(ctypes.c_int64)
    i8p = ctypes.POINTER(ctypes.c_int8)
    i32p = ctypes.POINTER(ctypes.c_int32)
    f32p = ctypes.POINTER(ctypes.c_float)
    lib.oph_reg_integer.argtypes = [i64p, i64p, i64p, i64p, i64p, ctypes.c_int64, i8p, i8p, i64p, i64p, i64p, f32p,
                                    i32p, i32p, ctypes.c_uint64, i32p, ctypes.c_int64, i64p, i8p]
    lib.oph_reg_integer.restype = None
    _LIB['lib'] = lib
    return lib


class Tracker:
    """Defect identities from the ordered register events of each chunk."""
    COLS = (('cls', np.int8), ('birth', np.int32), ('death', np.int32), ('bface', np.int32), ('lface', np.int32),
            ('hops', np.int32), ('partner', np.int32), ('p1', np.int32), ('p2', np.int32), ('cause', np.int8),
            ('c1', np.int32), ('c2', np.int32))

    def __init__(self, face_count: int):
        self.face_id = np.full(face_count, -1, np.int32)
        self.n = 0
        self.cap = 1 << 16
        self.cols = {k: np.full(self.cap, -1 if k not in ('hops', 'cause') else 0, dt) for k, dt in self.COLS}
        self.stats = dict(births=0, moves=0, annihilations=0, fusions=0, splits=0, exchanges=0)

    def _new(self, cls, face, tick, p1=-1, p2=-1):
        if self.n >= self.cap:
            self.cap *= 2
            for k, dt in self.COLS:
                new = np.full(self.cap, -1 if k not in ('hops', 'cause') else 0, dt)
                new[:self.n] = self.cols[k][:self.n]
                self.cols[k] = new
        i = self.n
        self.n += 1
        c = self.cols
        c['cls'][i] = cls; c['birth'][i] = tick; c['bface'][i] = face; c['lface'][i] = face
        c['p1'][i] = p1; c['p2'][i] = p2
        self.face_id[face] = i
        return i

    def _die(self, i, cause, partner, tick):
        self.cols['death'][i] = tick
        self.cols['cause'][i] = cause
        self.cols['partner'][i] = partner

    def process(self, events: np.ndarray, tick: int):
        c, fid, st = self.cols, self.face_id, self.stats
        for fp, fm, k1, k2, n1, n2 in events.tolist():
            o1, o2, w1, w2 = k1 != 0, k2 != 0, n1 != 0, n2 != 0
            if not o1 and not o2:
                a = self._new(n1, fp, tick); b = self._new(n2, fm, tick)
                c['partner'][a] = b; c['partner'][b] = a; st['births'] += 1
            elif o1 and not o2 and not w1:
                i = fid[fp]; fid[fm] = i; fid[fp] = -1; c['hops'][i] += 1; c['lface'][i] = fm; st['moves'] += 1
            elif not o1 and o2 and not w2:
                i = fid[fm]; fid[fp] = i; fid[fm] = -1; c['hops'][i] += 1; c['lface'][i] = fp; st['moves'] += 1
            elif o1 and not o2:
                old = fid[fp]; self._die(old, CAUSE['split'], -1, tick)
                h = self._new(n1, fp, tick, old); w = self._new(n2, fm, tick, old)
                c['partner'][h] = w; c['partner'][w] = h; c['c1'][old] = h; c['c2'][old] = w; st['splits'] += 1
            elif not o1 and o2:
                old = fid[fm]; self._die(old, CAUSE['split'], -1, tick)
                h = self._new(n2, fm, tick, old); w = self._new(n1, fp, tick, old)
                c['partner'][h] = w; c['partner'][w] = h; c['c1'][old] = h; c['c2'][old] = w; st['splits'] += 1
            else:
                i1, i2 = fid[fp], fid[fm]
                if not w1 and not w2:
                    self._die(i1, CAUSE['annihilation'], i2, tick); self._die(i2, CAUSE['annihilation'], i1, tick)
                    fid[fp] = -1; fid[fm] = -1; st['annihilations'] += 1
                elif not w1:
                    self._die(i1, CAUSE['fusion'], i2, tick); self._die(i2, CAUSE['fusion'], i1, tick)
                    new = self._new(n2, fm, tick, i1, i2); c['c1'][i1] = new; c['c1'][i2] = new; fid[fp] = -1; st['fusions'] += 1
                elif not w2:
                    self._die(i1, CAUSE['fusion'], i2, tick); self._die(i2, CAUSE['fusion'], i1, tick)
                    new = self._new(n1, fp, tick, i1, i2); c['c1'][i1] = new; c['c1'][i2] = new; fid[fm] = -1; st['fusions'] += 1
                else:
                    self._die(i1, CAUSE['exchange'], i2, tick); self._die(i2, CAUSE['exchange'], i1, tick)
                    a = self._new(n1, fp, tick, i1, i2); b = self._new(n2, fm, tick, i2, i1)
                    c['c1'][i1] = a; c['c2'][i1] = b; c['c1'][i2] = a; c['c2'][i2] = b; st['exchanges'] += 1

    def view(self):
        return {k: self.cols[k][:self.n] for k, _ in self.COLS}

    def state(self):
        return {'n': self.n, 'face_id': self.face_id, 'stats': dict(self.stats), **{k: v for k, v in self.view().items()}}

    @classmethod
    def restore(cls, face_count, payload):
        t = cls(face_count)
        t.face_id = np.array(payload['face_id'], dtype=np.int32)
        t.n = int(payload['n'])
        t.cap = max(t.cap, 1 << int(np.ceil(np.log2(max(2, t.n + 1)))))
        for k, dt in cls.COLS:
            arr = np.full(t.cap, -1 if k not in ('hops', 'cause') else 0, dt)
            arr[:t.n] = payload[k]
            t.cols[k] = arr
        t.stats = {k: int(v) for k, v in payload['stats'].items()}
        return t


def write_snapshot(path, tick, faces, classes, ids, xyz):
    body = (b'Z6SX' + struct.pack('<II', tick, len(faces)) + faces.astype('<i4').tobytes()
            + classes.astype('u1').tobytes() + ids.astype('<i4').tobytes() + xyz.astype('<f4').tobytes())
    with gzip.open(path, 'wb', compresslevel=6) as fh:
        fh.write(body)


def write_catalogue(path, cols):
    n = len(cols['cls'])
    body = b'Z6CT' + struct.pack('<I', n)
    for name, dtype in (('cls', 'u1'), ('birth', '<i4'), ('death', '<i4'), ('bface', '<i4'), ('lface', '<i4'),
                        ('hops', '<i4'), ('partner', '<i4'), ('p1', '<i4'), ('p2', '<i4'), ('cause', 'u1'),
                        ('c1', '<i4'), ('c2', '<i4')):
        body += cols[name].astype(dtype).tobytes()
    with gzip.open(path, 'wb', compresslevel=6) as fh:
        fh.write(body)


def parse_windows(text):
    windows = []
    for part in (text or '').split(','):
        if not part:
            continue
        a, b = part.split(':')
        windows.append((int(a), int(b)))
    return windows


def save_checkpoint(path: Path, sweep, x, reg, cls, activity, rng, reg_rng, tracker):
    ts = tracker.state()
    np.savez_compressed(path, sweep=np.int64(sweep), loads=x.astype(np.int8), register=reg, classes=cls,
                        activity=activity.astype(np.float32), face_id=ts['face_id'],
                        **{'cat_' + k: ts[k] for k, _ in Tracker.COLS}, cat_n=np.int64(ts['n']))
    (path.with_suffix('.json')).write_text(json.dumps({'sweep': int(sweep), 'law_rng': rng.bit_generator.state,
                                                        'register_rng': reg_rng.bit_generator.state,
                                                        'tracker_stats': ts['stats']}))


def run(level, sweeps, out: Path, schedule=0, T0=3.0, chunks=8, snapshot_every=2, loads_check=False, quiet=False,
        intra_every=10, checkpoint_every=25, record_windows='', resume: Path | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for sub in ('snapshots', 'checkpoints', 'records'):
        (out / sub).mkdir(exist_ok=True)
    t0 = time.time()
    fed, fc = G.build(level)
    loads = F.initial_loads(level, fed.ports)
    seed = F.schedule_seed(level, 'port_pair', schedule)
    log = lambda *a: None if quiet else print(*a, flush=True)
    log(f'level {level}: {fed.carriers} carriers, {fed.ports} ports, {fed.seams} seams, {fc.face_count} faces, built {time.time() - t0:.0f}s')
    lib = kernel()
    x = np.asarray(loads, dtype=np.int64).copy()
    rng = np.random.default_rng(seed)              # law C draws: identical to oph_exact
    reg_rng = np.random.default_rng(seed + 1)      # register acceptance seeds, a separate stream
    S = fed.seams
    ref = F.run_integer_law(fed, loads, seed, engine='native', max_sweeps=sweeps) if loads_check else None
    reg = np.zeros(S, np.int8)
    cls = np.zeros(fc.face_count, np.int8)
    windows = parse_windows(record_windows)
    first_sweep = 0
    parent = None
    fplus = np.ascontiguousarray(fc.seam_plus, dtype=np.int64)
    fminus = np.ascontiguousarray(fc.seam_minus, dtype=np.int64)
    seam_carrier = np.ascontiguousarray(fc.seam_carrier, dtype=np.int32)
    attempts_per_carrier = np.bincount(seam_carrier, minlength=fed.carriers).astype(np.float64)
    carrier_desc = np.zeros(fed.carriers, np.int32)
    activity = np.ones(fed.carriers, np.float64)
    temp = np.empty(S, np.float32)
    ev_cap = 1 << 22
    events = np.zeros((ev_cap, 6), np.int32)
    counts = np.zeros(7, np.int64)
    tracker = Tracker(fc.face_count)
    if resume is not None:
        ck = np.load(resume, allow_pickle=False)
        meta_ck = json.loads(resume.with_suffix('.json').read_text())
        x = ck['loads'].astype(np.int64); reg = ck['register'].copy(); cls = ck['classes'].copy()
        activity = ck['activity'].astype(np.float64)
        rng.bit_generator.state = meta_ck['law_rng']; reg_rng.bit_generator.state = meta_ck['register_rng']
        payload = {'n': int(ck['cat_n']), 'face_id': ck['face_id'], 'stats': meta_ck['tracker_stats'],
                   **{k: ck['cat_' + k] for k, _ in Tracker.COLS}}
        tracker = Tracker.restore(fc.face_count, payload)
        first_sweep = int(ck['sweep'])
        parent = str(resume)
        log(f'resumed from {resume} at sweep {first_sweep} with {tracker.n} identities')
    series = {k: [] for k in ('V', 'phi', 'unsettled', 'descents', 'swaps', 'reg_accepted', 'reg_rejected',
                              'births', 'moves', 'annihilations', 'fusions', 'splits', 'exchanges', 'T_mean',
                              'counts_intra', 'counts_macro')}
    snaps = []
    checkpoints = []
    records = []
    cur_out = np.zeros(S, np.int8)
    i64p = ctypes.POINTER(ctypes.c_int64)
    i8p = ctypes.POINTER(ctypes.c_int8)
    i32p = ctypes.POINTER(ctypes.c_int32)
    f32p = ctypes.POINTER(ctypes.c_float)
    a_arr = np.ascontiguousarray(fed.seam_a, dtype=np.int64)
    b_arr = np.ascontiguousarray(fed.seam_b, dtype=np.int64)
    cost = np.ascontiguousarray(COST, dtype=np.int64)
    nI = fc.intra_seams
    for s in range(first_sweep, first_sweep + sweeps):
        in_window = any(a <= s < b for a, b in windows)
        if s % checkpoint_every == 0 or in_window:
            # RNG states at the start of this sweep: enough to replay it exactly
            state_json = {'sweep': s, 'law_rng': rng.bit_generator.state, 'register_rng': reg_rng.bit_generator.state}
        if s % checkpoint_every == 0:
            name = f'checkpoints/ckpt_{s:06d}.npz'
            save_checkpoint(out / name, s, x, reg, cls, activity, rng, reg_rng, tracker)
            checkpoints.append({'sweep': s, 'file': name})
        # the register temperature for this sweep: T0 times last sweep's descent activity
        ta = activity[seam_carrier]
        inter_b = fed.inter_pairs[:, 2]
        ta[nI:] = 0.5 * (activity[fed.inter_pairs[:, 0]] + activity[inter_b])
        temp[:] = (T0 * ta).astype(np.float32)
        seq = rng.integers(0, S, size=S, dtype=np.int64)
        coin = rng.integers(0, 2, size=S, dtype=np.int64)
        carrier_desc[:] = 0
        before = counts.copy()
        tr_before = dict(tracker.stats)
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
                                ev_cap, counts.ctypes.data_as(i64p),
                                cur_out[lo:hi].ctypes.data_as(i8p) if in_window else ctypes.cast(0, i8p))
            if counts[6]:
                raise RuntimeError('event buffer overflow; raise ev_cap or chunks')
            tracker.process(events[:counts[5]], s * chunks + k)
        activity = carrier_desc / attempts_per_carrier
        if in_window:
            name = f'records/cur_{s:06d}.npy.gz'
            with gzip.open(out / name, 'wb', compresslevel=6) as fh:
                np.save(fh, cur_out)
            (out / f'records/rng_{s:06d}.json').write_text(json.dumps(state_json))
            records.append({'sweep': s, 'currents': name, 'rng': f'records/rng_{s:06d}.json', 'attempts': int(S)})
        if ref is not None and s == ref['sweeps'] - 1:
            assert (ref['state'] == x).all(), 'loads differ from oph_exact law C'
            log(f'loads equal oph_exact law C after {ref["sweeps"]} sweeps')
        d = x[fed.seam_a] - x[fed.seam_b]
        series['V'].append(int(np.dot(x, x)))
        series['phi'].append(int(np.dot(d, d)))
        series['unsettled'].append(int(np.count_nonzero(d)))
        series['descents'].append(int(counts[0] - before[0]))
        series['swaps'].append(int(counts[1] - before[1]))
        series['reg_accepted'].append(int(counts[3] - before[3]))
        series['reg_rejected'].append(int(counts[4] - before[4]))
        for k in ('births', 'moves', 'annihilations', 'fusions', 'splits', 'exchanges'):
            series[k].append(tracker.stats[k] - tr_before[k])
        series['T_mean'].append(float(temp.mean()))
        series['counts_intra'].append(np.bincount(cls[:20 * fed.carriers], minlength=6).tolist())
        series['counts_macro'].append(np.bincount(cls[20 * fed.carriers:], minlength=6).tolist())
        if s % snapshot_every == 0 or s == sweeps - 1:
            nIF = 20 * fed.carriers
            macro = (nIF + np.flatnonzero(cls[nIF:])).astype(np.int32)
            name = f'snapshots/macro_{s:06d}.bin.gz'
            write_snapshot(out / name, s * chunks, macro, cls[macro], tracker.face_id[macro], fc.face_xyz[macro])
            entry = {'tick': s * chunks, 'file': name, 'count': int(len(macro)), 'intra_file': None, 'intra_count': 0}
            if s % intra_every == 0 or s == sweeps - 1:
                intra = np.flatnonzero(cls[:nIF]).astype(np.int32)
                iname = f'snapshots/intra_{s:06d}.bin.gz'
                write_snapshot(out / iname, s * chunks, intra, cls[intra], tracker.face_id[intra], fc.face_xyz[intra])
                entry['intra_file'] = iname
                entry['intra_count'] = int(len(intra))
            snaps.append(entry)
        if s % 5 == 0 or s == sweeps - 1:
            ci, cm = series['counts_intra'][-1], series['counts_macro'][-1]
            log(f'sweep {s}: T={series["T_mean"][-1]:.3f} descents={series["descents"][-1]} swaps={series["swaps"][-1]} '
                f'reg acc/rej={series["reg_accepted"][-1]}/{series["reg_rejected"][-1]} intra={ci[1:]} macro={cm[1:]} '
                f'ids={tracker.n} elapsed={time.time() - t0:.0f}s')
    final_sweep = first_sweep + sweeps
    name = f'checkpoints/ckpt_{final_sweep:06d}.npz'
    save_checkpoint(out / name, final_sweep, x, reg, cls, activity, rng, reg_rng, tracker)
    checkpoints.append({'sweep': final_sweep, 'file': name})
    # exact checks
    recomputed = G.face_classes(fc, reg)
    assert (recomputed == cls).all(), 'class register drifted'
    assert int(cls[:20 * fed.carriers].astype(np.int64).sum()) % 6 == 0 and int(cls[20 * fed.carriers:].astype(np.int64).sum()) % 6 == 0
    live = tracker.face_id >= 0
    assert (live == (cls != 0)).all(), 'identity map disagrees with classes'
    cols = tracker.view()
    write_catalogue(out / 'catalogue.bin.gz', cols)
    with gzip.open(out / 'catalogue_xyz.bin.gz', 'wb', compresslevel=6) as fh:
        fh.write(b'Z6XY' + struct.pack('<I', tracker.n) + fc.face_xyz[cols['bface']].astype('<f4').tobytes()
                 + fc.face_xyz[cols['lface']].astype('<f4').tobytes())
    np.save(out / 'final_loads.npy', x.astype(np.int8))
    np.save(out / 'final_register.npy', reg)
    meta = {
        'schema': 'oph.defects.federation_run.v1',
        'schema_doc': 'oph_defects/SCHEMA.md',
        'parent_checkpoint': parent,
        'first_sweep': first_sweep,
        'checkpoints': checkpoints,
        'record_windows': records,
        'config': {'law': 'law C (integer nearest agreement, fair tie coin) exactly as oph_exact.federation; '
                          'guarded Z6 transport register driven by the moved current',
                   'cost': COST.tolist(), 'T0': T0, 'temperature': 'T0 times the previous sweep descent activity of the seam carriers',
                   'level': level, 'ticks': sweeps * chunks, 'sweeps': sweeps, 'chunks_per_sweep': chunks, 'seed': seed,
                   'schedule': 'uniform over seams, one sweep = |S| attempts', 'snapshot_every': snapshot_every * chunks,
                   'intra_every': intra_every * chunks, 'tiers': 'macro = screen vertices (between carriers); intra = template faces inside carriers',
                   'checkpoint_every': checkpoint_every, 'kernel_sha256': hashlib.sha256(KERNEL.encode()).hexdigest(),
                   'initial_loads': 'oph_exact.federation.initial_loads(level, ports): default_rng(20260909 + level).integers(0, 6)',
                   'status': 'loads: the executable spec exactly; register, cost table and temperature law: declared extension'},
        'geometry': {'level': level, 'observers': fed.carriers, 'carriers': fed.carriers, 'ports': fed.ports,
                     'seams': fed.seams, 'inter_seams': fed.inter_count, 'screen_vertices': fc.vertices,
                     'faces': fc.face_count, 'intra_faces': 20 * fed.carriers, 'coords': True,
                     'base_faces': G._template_faces().tolist(), 'base_vertices': G._template_vertices().tolist()},
        'stats': series, 'snapshots': snaps, 'identities': int(tracker.n),
        'runtime_seconds': round(time.time() - t0, 1), 'numpy': np.__version__, 'python': sys.version.split()[0],
    }
    (out / 'run.json').write_text(json.dumps(meta))
    log(f'done: {tracker.n} identities, {len(snaps)} snapshots, {time.time() - t0:.0f}s')
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--level', type=int, default=8)
    p.add_argument('--sweeps', type=int, default=300)
    p.add_argument('--schedule', type=int, default=0)
    p.add_argument('--T0', type=float, default=3.0)
    p.add_argument('--intra-every', type=int, default=10, help='sweeps between snapshots of the internal tier')
    p.add_argument('--chunks', type=int, default=8)
    p.add_argument('--snapshot-every', type=int, default=2)
    p.add_argument('--loads-check', action='store_true')
    p.add_argument('--checkpoint-every', type=int, default=25)
    p.add_argument('--record-windows', default='', help='sweep ranges with dense per-attempt records, e.g. 0:4,296:300')
    p.add_argument('--resume', type=Path, default=None, help='checkpoint .npz to continue from')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    run(a.level, a.sweeps, a.out, a.schedule, a.T0, a.chunks, a.snapshot_every, a.loads_check, intra_every=a.intra_every,
        checkpoint_every=a.checkpoint_every, record_windows=a.record_windows, resume=a.resume)


if __name__ == '__main__':
    main()
