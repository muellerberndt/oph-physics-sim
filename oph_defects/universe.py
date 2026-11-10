"""Law C on the exact federation at universe scale, with a documented dataset as output.

    python -m oph_defects.universe --level 5 --sweeps 400 --records all --out runs/universe_L5

Exact layer (the executable specification, nothing added): the federation of
``oph_exact.federation.build_federation(level, 'port_pair')``, the declared
preparation, the uniform seam schedule with fair tie coins drawn exactly as
``run_integer_law`` draws them, and the integer law (law C). The loads this
program produces are those of ``run_integer_law`` for the same seed.

Declared layers on top of the unchanged loads, each a deterministic function
of the exact event stream (no further randomness):

* causal depth of every attempt and consistent cuts of equal depth (the
  clock-synchronised slices; the sweep index is a solver counter);
* incident clocks: attempts on the 33 seams of every carrier;
* the energy-accounted Z6 link register and the identities of its defects.

The run directory is described in ``oph_defects/DATASET.md``.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

import numpy as np

from . import fedgeom as FG
from . import kernels as K
from .tracking import COLUMNS, STATS, Tracker

SCHEMA = 'oph.universe.run.v2'
LOAD_SEED_BASE = 20260909          # oph_exact.federation.LOAD_SEED_BASE
SCHEDULE_SEED_BASE = 909000        # oph_exact.federation.SCHEDULE_SEED_BASE
LOAD_MAX = 5                       # oph_exact.federation.LOAD_MAX
GLUING_INDEX = 1                   # 'port_pair' in oph_exact.federation.GLUINGS
COSTS = {'binding': (0, 5, 8, 10, 8, 5), 'additive': (0, 5, 10, 15, 10, 5), 'flat': (0, 5, 5, 5, 5, 5),
         'wilson': (0, 5, 15, 20, 15, 5)}
# literal source hashes pinned by the executable model (reverse-engineering-reality/model/federated_observer/source.py)
MODEL_PINS = {
    'oph_exact/carrier.py': '83354f120a0d40d9fd3c04721bbbb3c6140cc3c5bf10655d899e9aa4325804bb',
    'oph_exact/federation.py': '29b0bb7a1e4d8a2077f16dbb508b7e08e794a7191a4a9f51e6c1b471e726ca1d',
    'oph_fpe/core/icosahedral.py': '89fac4f2d24bb4d85c4a84624b3c826e99cb717df68060dd23a4b26c6cfe9adc',
    'oph_fpe/core/screen_ports.py': '37b79a8a949fad2d18f17e850533c6ef33581f64c8f1a72dd94e45da93cef68e',
}
DEFINING = ('oph_exact/carrier.py', 'oph_fpe/core/icosahedral.py', 'oph_fpe/core/screen_ports.py')
REPO = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    level: int
    sweeps: int = 300
    schedule: int = 0
    prep: str = 'exact06'            # exact06 | canonical:<num>/<den>:<seed>
    cost: str = 'binding'
    kappa: int = 5
    backend: str = 'sequential'      # sequential | layers | cuda
    track_intra: bool | None = None
    cut_ratio: float = 1.5
    cut_step: int = 64
    slots: int = 24
    checkpoint_every: int = 50
    records: str = ''                # '' | 'all' | 'a:b,c:d'
    patch_radius: int = 16
    full_cut_level: int = 6          # store full loads and clocks on every cut up to this level
    geometry_level: int = 7          # store the geometry arrays in the run directory up to this level
    display_level: int = 7
    chunk: int = 0
    fast_rng: bool | None = None
    reverse_layers: bool = False
    register: bool = True            # the declared Z6 register and its identities
    observers: bool = True           # tapes and patches of the featured observers
    series_every: int = 0            # sweeps between density snapshots (0: none)
    series_ports: bool | None = None  # per-port loads in the snapshots (default: level <= 5)
    twists: bool = False             # declared extension: transported contrast with unit twists (charges act back on the loads)
    offsets: str = ''                # with twists: '' (all zero) or 'iid:<seed>' (independent glue offsets in {-1, 0, +1})
    track: bool = True               # identities of defects or charges; switch off for dense stationary charge gases


# ---------------------------------------------------------------------------- declared inputs
def schedule_seed(level: int, schedule: int) -> int:
    return SCHEDULE_SEED_BASE + 1000 * level + 100 * GLUING_INDEX + schedule


def prepare_loads(prep: str, level: int, ports: int) -> tuple[np.ndarray, dict]:
    """The declared preparation. ``exact06`` is ``oph_exact.federation.initial_loads``."""
    if prep == 'exact06':
        rng = np.random.default_rng(LOAD_SEED_BASE + level)
        loads = rng.integers(0, LOAD_MAX + 1, size=ports).astype(np.int8)
        return loads, {'name': 'exact06', 'rule': f'numpy.random.default_rng({LOAD_SEED_BASE} + level).integers(0, {LOAD_MAX + 1}, size=ports)',
                       'source': 'oph_exact.federation.initial_loads'}
    if prep.startswith('canonical:'):
        _, frac, seed = prep.split(':')
        num, den = (int(v) for v in frac.split('/'))
        occupied = ports * num // den
        if occupied * den != ports * num:
            raise ValueError('the occupied number must be an integer')
        rng = random.Random(int(seed))
        left = occupied
        state = bytearray(ports)
        for port in range(ports):                      # model/observer_time_fields/canonical_sector.draw_sector
            state[port] = int(rng.randrange(ports - port) < left)
            left -= state[port]
        assert left == 0
        return np.frombuffer(bytes(state), dtype=np.int8).copy(), {
            'name': 'canonical', 'occupied': occupied, 'population': ports, 'seed': int(seed),
            'rule': 'uniform subset of fixed size, sequential exact integer draws random.Random(seed).randrange(population - port) < left',
            'source': 'model/observer_time_fields/canonical_sector.draw_sector'}
    raise ValueError(f'unknown preparation {prep!r}')


def cut_depths(ratio: float, step: int):
    """Depths of the consistent cuts: 0, then geometric from 1, then every ``step``."""
    yield 0
    t = 1
    while t < 2 * step:
        yield t
        t = max(t + 1, int(round(t * ratio)))
    t = ((t + step - 1) // step) * step
    while True:
        yield t
        t += step


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance() -> dict:
    pins = {}
    for rel, pinned in MODEL_PINS.items():
        local = REPO / rel
        digest = sha256_file(local) if local.exists() else None
        pins[rel] = {'sha256': digest, 'model_pin': pinned, 'equal': digest == pinned, 'defines_the_run': rel in DEFINING}
    def git(*args):
        try:
            return subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ''
    return {'source_pins': pins, 'defining_sources_equal_model_pins': all(pins[r]['equal'] for r in DEFINING),
            'kernel_core_sha256': K.CORE_SHA256, 'kernel_c_sha256': K.C_SHA256,
            'simulator_commit': git('rev-parse', 'HEAD'), 'simulator_dirty': bool(git('status', '--porcelain', '--', 'oph_defects')),
            'numpy': np.__version__, 'python': sys.version.split()[0], 'platform': platform.platform(),
            'machine': platform.machine()}


# ---------------------------------------------------------------------------- engine
class Engine:
    """State of one universe and the execution of schedule chunks (CPU backends)."""
    xp = np

    def __init__(self, geo: FG.Geometry, loads: np.ndarray, seed: int, cost, kappa: int, track_intra: bool,
                 backend: str = 'sequential', slots: int = 24, chunk: int = 0, fast_rng: bool = False,
                 reverse_layers: bool = False, register: bool = True, offsets: np.ndarray | None = None,
                 currents: bool = False, counters: bool = False, twists: bool = False, track: bool = True):
        self.geo = geo
        self.track = bool(track)
        if twists and (register or offsets is None or track_intra):
            raise ValueError('twists need seam offsets, no Z6 register and screen-tier tracking only')
        self.flags = (K.F_REG if register else 0) | (K.F_OFF if offsets is not None else 0) \
            | (K.F_JSUM if currents else 0) | (K.F_COUNT if counters else 0) | (K.F_TWIST if twists else 0)
        self.offsets0 = None if offsets is None else np.ascontiguousarray(offsets, dtype=np.int8)
        g = geo
        self.N, self.P, self.S, self.V = g.carriers, g.ports, g.seams, g.vertices
        self.SI, self.FI, self.F = g.intra_seams, g.intra_faces, g.faces
        self.F0 = 0 if track_intra else self.FI
        if not track:
            self.F0 = self.F                       # no tracked face: no event is ever logged
        self.FT = self.F - self.F0
        self.backend = backend
        self.slots = slots
        self.chunk = int(chunk) if chunk else (max(1024, self.P // 16) if backend != 'sequential' else min(self.S, 1 << 22))
        self.reverse_layers = reverse_layers
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.fast = K.FastPCG64(self.rng) if fast_rng else None
        self.cost = np.asarray(cost, dtype=np.int32)
        self.kappa = int(kappa)
        self._alloc(loads)
        self.tracker = Tracker(self.F0, self.FT, slots)
        if twists:                          # the holonomies of the prepared offsets are the initial charges
            curl = geo.face_curl(self.offsets0)
            if np.abs(curl).max() > 100 or curl[:self.FI].any():
                raise ValueError('prepared offsets must have small holonomies on screen plaquettes only')
            self.cls[:] = self._dev(curl.astype(np.int8))
            faces = np.flatnonzero(curl)
            if self.track:
                self.tracker.seed(faces, curl[faces])
        self.open: list[dict] = []          # open cuts, ascending depth: {'k', 'T', 'slot', 'opened'}
        self.free = list(range(slots - 1, -1, -1))
        self._sync_cut_arrays()

    # ---- memory
    def _alloc(self, loads):
        g = self.geo
        n = self.chunk
        self.x = np.ascontiguousarray(loads, dtype=np.int8).copy()
        self.reg = np.zeros(self.S, np.int8)
        self.cls = np.zeros(self.F, np.int8)
        self.dnat = np.zeros(self.P, np.int32)
        self.dsyn = np.zeros(self.P + self.V, np.int32)
        self.ticks = np.zeros(self.N, np.int32)
        self.cut_x = np.zeros((self.slots, self.P), np.int8)
        self.cut_ticks = np.zeros((self.slots, self.N), np.int32)
        self.cutT = np.zeros(self.slots, np.int32)
        self.cut_slot = np.zeros(self.slots, np.int32)
        self.o_cur = np.zeros(n, np.int8); self.o_con = np.zeros(n, np.int8)
        self.o_twist = np.zeros(n, np.int8); self.o_de = np.zeros(n, np.int8); self.o_tot = np.zeros(n, np.int8)
        self.o_dep = np.zeros(n, np.int32)
        self.off = self.offsets0.copy() if self.flags & K.F_OFF else np.zeros(1, np.int8)
        self.jsum = np.zeros(self.S if self.flags & K.F_JSUM else 1, np.int32)
        self.nzc = np.zeros(self.N if self.flags & K.F_COUNT else 1, np.int32)
        self.dsc = np.zeros(self.N if self.flags & K.F_COUNT else 1, np.int32)
        self.events = np.zeros((n, 8), np.int32)
        self.ev_count = np.zeros(2, np.int64)
        self.ta = np.ascontiguousarray(FG.TEMPLATE_SEAMS[:, 0], dtype=np.int8)
        self.tb = np.ascontiguousarray(FG.TEMPLATE_SEAMS[:, 1], dtype=np.int8)
        self.tp = np.ascontiguousarray(FG.TEMPLATE_PLUS, dtype=np.int8)
        self.tm = np.ascontiguousarray(FG.TEMPLATE_MINUS, dtype=np.int8)
        self.ga = np.ascontiguousarray(g.glue_a, dtype=np.int32); self.gb = np.ascontiguousarray(g.glue_b, dtype=np.int32)
        self.gfp = np.ascontiguousarray(g.glue_fp, dtype=np.int32); self.gfm = np.ascontiguousarray(g.glue_fm, dtype=np.int32)
        if self.backend == 'layers':
            self.pend = np.zeros(n, np.int32)
            self.nxt = np.full(self.P + self.V, K.INF, np.int32)
        self.lib = K.library()
        c = K.Ctx()
        c.N, c.P, c.SI, c.FI, c.F0 = self.N, self.P, self.SI, self.FI, self.F0
        for name in ('ta', 'tb', 'tp', 'tm', 'ga', 'gb', 'gfp', 'gfm', 'x', 'reg', 'cls', 'dnat', 'dsyn', 'ticks', 'cost',
                     'cutT', 'cut_slot', 'cut_x', 'cut_ticks', 'o_cur', 'o_con', 'o_twist', 'o_de', 'o_tot', 'o_dep', 'off',
                     'jsum', 'nzc', 'dsc', 'events', 'ev_count'):
            setattr(c, name, K.ptr(getattr(self, name)))
        c.kappa = self.kappa
        c.flags = self.flags
        c.ncut = 0
        c.ev_cap = n
        self.ctx = c

    def host(self, a):
        return a

    def hist6(self, a) -> np.ndarray:
        """Counts of the values 0..5 of an int8 array. With twists the entries are signed: charges are counted by
        their residue mod 6 (so -1 and 5 share a bin) and loads are clipped to 0..5."""
        xp = self.xp
        if self.flags & K.F_TWIST:
            v = a.astype(xp.int16)
            v = xp.clip(v, 0, 5) if a is self.x or a.shape[-1] == self.P else (v % 6 + 6) % 6
            return np.asarray(self.host(xp.bincount(v.astype(xp.int64).ravel(), minlength=6)))[:6].astype(np.int64)
        return np.asarray(self.host(xp.bincount(a.view(xp.uint8), minlength=6)))[:6].astype(np.int64)

    # ---- schedule draws, exactly as oph_exact.federation.run_integer_law
    def draw(self) -> tuple[np.ndarray, np.ndarray]:
        S = self.S
        if self.fast is not None:
            state = self.rng.bit_generator.state
            seq = self.fast.integers32(S, S)
            coin = self.fast.bits8(S)
            m = min(S, 1 << 16)                       # the first draws of both kinds, checked against numpy itself
            ref = np.random.default_rng(0); ref.bit_generator.state = state
            if not (ref.integers(0, S, size=m, dtype=np.int64) == seq[:m]).all():
                raise RuntimeError('fast generator differs from numpy on the seam draws')
            return seq, coin
        piece = 1 << 24
        seq = np.empty(S, np.int32); coin = np.empty(S, np.uint8)
        for lo in range(0, S, piece):
            hi = min(S, lo + piece)
            seq[lo:hi] = self.rng.integers(0, S, size=hi - lo, dtype=np.int64)
        for lo in range(0, S, piece):
            hi = min(S, lo + piece)
            coin[lo:hi] = self.rng.integers(0, 2, size=hi - lo, dtype=np.int64)
        return seq, coin

    # ---- one chunk of the schedule
    def run_chunk(self, seq: np.ndarray, coin: np.ndarray, t_base: int, sweep: int) -> int:
        """Execute attempts seq[0:n] (schedule positions t_base..). Fills o_* [0:n]; returns n."""
        n = int(seq.shape[0])
        assert n <= self.chunk and seq.dtype == np.int32 and coin.dtype == np.uint8
        seq = np.ascontiguousarray(seq); coin = np.ascontiguousarray(coin)
        self.ev_count[:] = 0
        if self.backend == 'sequential':
            self.lib.oph_sequential(ctypes.byref(self.ctx), K.ptr(seq), K.ptr(coin), n, t_base)
        else:
            self.layers = getattr(self, 'layers', 0) + int(self.lib.oph_layers(
                ctypes.byref(self.ctx), K.ptr(seq), K.ptr(coin), n, t_base, K.ptr(self.pend), K.ptr(self.nxt),
                int(self.reverse_layers)))
        if self.ev_count[1]:
            raise RuntimeError('event buffer overflow')
        m = int(self.ev_count[0])
        ev = self.events[:m]
        if self.backend != 'sequential' and m:
            ev = ev[np.argsort(ev[:, 0], kind='stable')]
        self.last_events = ev
        self.tracker.process(ev, sweep, self.cutT, self.cut_slot, len(self.open))
        return n

    # ---- consistent cuts
    def _sync_cut_arrays(self):
        for q, cut in enumerate(self.open):
            self.cutT[q] = cut['T']; self.cut_slot[q] = cut['slot']
        self.ctx.ncut = len(self.open)

    def open_cut(self, k: int, T: int, sweep: int):
        if not self.free:
            raise RuntimeError('no free cut slot: raise --slots or --cut-step')
        if self.open and T <= self.open[-1]['T']:
            raise RuntimeError('cuts must open in ascending depth')
        slot = self.free.pop()
        self.cut_x[slot] = self.x
        self.cut_ticks[slot] = self.ticks
        self.tracker.open_cut(slot, self.host(self.cls[self.F0:]))
        self.open.append({'k': int(k), 'T': int(T), 'slot': slot, 'opened': int(sweep)})
        self._sync_cut_arrays()

    def close_cut(self) -> dict:
        cut = self.open.pop(0)
        self.free.append(cut['slot'])
        self._sync_cut_arrays()
        return cut

    def depth_range(self) -> tuple[int, int, float]:
        d = self.dsyn[:self.P]
        return int(d.min()), int(d.max()), float(d.mean())

    # ---- persistence
    def state(self) -> dict:
        out = {'loads': self.host(self.x).copy(), 'register': self.host(self.reg).copy(), 'classes': self.host(self.cls).copy(),
               'offsets': self.host(self.off).copy(),
               'depth_native': self.host(self.dnat).copy(), 'depth': self.host(self.dsyn).copy(), 'ticks': self.host(self.ticks).copy()}
        out.update(self.tracker.state())
        slots = [c['slot'] for c in self.open]
        out['open_cut_index'] = np.array([c['k'] for c in self.open], np.int32)
        out['open_cut_depth'] = np.array([c['T'] for c in self.open], np.int32)
        out['open_cut_opened'] = np.array([c['opened'] for c in self.open], np.int32)
        out['open_cut_loads'] = self.host(self.cut_x[slots]) if slots else np.zeros((0, self.P), np.int8)
        out['open_cut_ticks'] = self.host(self.cut_ticks[slots]) if slots else np.zeros((0, self.N), np.int32)
        out['open_cut_classes'] = self.tracker.cut_cls[slots] if slots else np.zeros((0, self.FT), np.int8)
        out['open_cut_ids'] = self.tracker.cut_fid[slots] if slots else np.zeros((0, self.FT), np.int32)
        return out

    def restore(self, z, rng_state: dict):
        self.x[:] = self._dev(z['loads']); self.reg[:] = self._dev(z['register']); self.cls[:] = self._dev(z['classes'])
        if self.flags & K.F_OFF:
            self.off[:] = self._dev(z['offsets'])
        self.dnat[:] = self._dev(z['depth_native']); self.dsyn[:] = self._dev(z['depth']); self.ticks[:] = self._dev(z['ticks'])
        self.tracker.restore(z)
        self.rng.bit_generator.state = rng_state
        self.open = []
        self.free = list(range(self.slots - 1, -1, -1))
        for q in range(len(z['open_cut_index'])):
            slot = self.free.pop()
            self.cut_x[slot] = self._dev(z['open_cut_loads'][q]); self.cut_ticks[slot] = self._dev(z['open_cut_ticks'][q])
            self.tracker.cut_cls[slot] = z['open_cut_classes'][q]; self.tracker.cut_fid[slot] = z['open_cut_ids'][q]
            self.open.append({'k': int(z['open_cut_index'][q]), 'T': int(z['open_cut_depth'][q]), 'slot': slot,
                              'opened': int(z['open_cut_opened'][q])})
        self._sync_cut_arrays()

    def _dev(self, a):
        return a


# ---------------------------------------------------------------------------- featured observers
def featured_observers(geo: FG.Geometry) -> list[dict]:
    """The root observer of the model, the central carrier of every base face and one carrier at every base vertex."""
    L, N = geo.level, geo.carriers
    per = 4 ** L
    out, seen = [], set()
    for f in range(20):
        c = f * per + per - 1
        if c not in seen:
            seen.add(c)
            out.append({'carrier': int(c), 'role': 'root observer (central cell of base face 0)' if f == 0 else f'central cell of base face {f}'})
    for v in range(12):
        c = int(np.flatnonzero((geo.tri == v).any(axis=1))[0])
        if c not in seen:
            seen.add(c)
            out.append({'carrier': c, 'role': f'pentagonal cell at base vertex {v}'})
    return out


def patch_of(geo: FG.Geometry, carrier: int, radius: int) -> dict:
    cells, dist = geo.ball(carrier, radius)
    local = {int(c): i for i, c in enumerate(cells.tolist())}
    m = cells.size
    a = (np.arange(m)[:, None] * 12 + FG.TEMPLATE_SEAMS[None, :, 0]).ravel()
    b = (np.arange(m)[:, None] * 12 + FG.TEMPLATE_SEAMS[None, :, 1]).ravel()
    s = (cells[:, None] * 30 + np.arange(30)[None, :]).ravel()
    cg = geo.carrier_glue_seams()[cells]
    glue = np.unique(cg.ravel())
    ga, gb = geo.glue_a[glue].astype(np.int64), geo.glue_b[glue].astype(np.int64)
    inside = np.array([(int(p) // 12 in local) and (int(q) // 12 in local) for p, q in zip(ga, gb)], bool)
    la = np.array([local[int(p) // 12] * 12 + int(p) % 12 for p in ga[inside]], np.int64)
    lb = np.array([local[int(q) // 12] * 12 + int(q) % 12 for q in gb[inside]], np.int64)
    seams = np.stack([np.concatenate([a, la]), np.concatenate([b, lb]),
                      np.concatenate([s, geo.intra_seams + glue[inside]])], axis=1).astype(np.int32)
    ports = (cells[:, None] * 12 + np.arange(12)[None, :]).ravel()
    return {'cells': cells.astype(np.int32), 'dist': dist.astype(np.int16), 'seams': seams, 'ports': ports,
            'open_glue': int((~inside).sum()), 'cell_xyz': geo.carrier_xyz[cells].astype(np.float32),
            'frames': geo.frames(cells).astype(np.float32)}


# ---------------------------------------------------------------------------- the run
class Run:
    def __init__(self, cfg: Config, out: Path, resume: Path | None = None, quiet: bool = False):
        self.cfg, self.out = cfg, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        for sub in ('checkpoints', 'cuts', 'records', 'observers', 'series'):
            (self.out / sub).mkdir(exist_ok=True)
        self.quiet = quiet
        self.t0 = time.time()
        geo = self.geo = FG.build(cfg.level)
        track_intra = cfg.track_intra if cfg.track_intra is not None else cfg.level <= 6
        self.track_intra = bool(track_intra)
        loads, self.prep = prepare_loads(cfg.prep, cfg.level, geo.ports)
        self.loads0 = loads.copy()
        self.seed = schedule_seed(cfg.level, cfg.schedule)
        fast = cfg.fast_rng if cfg.fast_rng is not None else cfg.level >= 9
        extra = {} if cfg.track else {'track': False}
        if cfg.twists:
            self.track_intra = False
            off0 = np.zeros(geo.seams, np.int8)
            if cfg.offsets.startswith('iid:'):
                off0[geo.intra_seams:] = np.random.default_rng(int(cfg.offsets[4:])).integers(-1, 2, size=geo.glue_seams).astype(np.int8)
            elif cfg.offsets:
                raise ValueError(f'unknown offset preparation {cfg.offsets!r}')
            self.offsets0 = off0
            extra.update(offsets=off0, twists=True)
        if cfg.backend == 'cuda':
            from .cuda import CudaEngine
            self.eng = CudaEngine(geo, loads, self.seed, COSTS[cfg.cost], cfg.kappa, self.track_intra, slots=cfg.slots,
                                  chunk=cfg.chunk, fast_rng=bool(fast), register=cfg.register and not cfg.twists,
                                  counters=cfg.series_every > 0, **extra)
        else:
            self.eng = Engine(geo, loads, self.seed, COSTS[cfg.cost], cfg.kappa, self.track_intra, backend=cfg.backend,
                              slots=cfg.slots, chunk=cfg.chunk, fast_rng=bool(fast), reverse_layers=cfg.reverse_layers,
                              register=cfg.register and not cfg.twists, counters=cfg.series_every > 0, **extra)
        self.xp = self.eng.xp
        total = int(loads.astype(np.int64).sum())
        q, r = divmod(total, geo.ports)
        self.total, self.q, self.r = total, q, r
        self.v_min = (geo.ports - r) * q * q + r * (q + 1) * (q + 1)
        self.v = int((loads.astype(np.int64) ** 2).sum())
        self.balanced_at = 0 if self.v == self.v_min else -1
        self.windows = self._windows(cfg.records)
        self.cut_iter = cut_depths(cfg.cut_ratio, cfg.cut_step)
        self.next_cut = (0, next(self.cut_iter))
        self.max_growth = 0
        self.series: dict[str, list] = {}
        self.cuts: list[dict] = []
        self.checkpoints: list[dict] = []
        self.records: list[dict] = []
        self.first_sweep = 0
        self.sweep = 0
        self.display_level = min(cfg.level, cfg.display_level)
        self.use_cuts = cfg.cut_step > 0
        self.series_ports = cfg.series_ports if cfg.series_ports is not None else cfg.level <= 5
        self.series_buf = {}
        self.series_bytes = 0
        self.series_blocks: list[dict] = []
        self._observers()
        if resume is not None:
            self._resume(Path(resume))
        self.log(f'level {cfg.level}: {geo.carriers} carriers, {geo.ports} ports, {geo.seams} seams, {geo.faces} faces; '
                 f'backend {cfg.backend}; seed {self.seed}; prep {self.prep["name"]}; total load {total} (band {q},{q + 1}); '
                 f'ready {time.time() - self.t0:.0f}s')

    def log(self, *a):
        if not self.quiet:
            print(*a, flush=True)

    def _windows(self, text):
        if text == 'all':
            return 'all'
        out = []
        for part in (text or '').split(','):
            if part:
                a, b = part.split(':')
                out.append((int(a), int(b)))
        return out

    def _in_window(self, s):
        return self.windows == 'all' or any(a <= s < b for a, b in self.windows)

    # ---- observers
    def _observers(self):
        geo, cfg = self.geo, self.cfg
        self.feat = featured_observers(geo) if cfg.observers else []
        self.feat_index = np.full(geo.carriers, -1, np.int32)
        seam_feat = np.zeros(geo.seams, np.uint8)
        self.patches = []
        cg = geo.carrier_glue_seams()
        for i, ob in enumerate(self.feat):
            c = ob['carrier']
            self.feat_index[c] = i
            seam_feat[geo.carrier_incident_seams(c)] = 1
            patch = patch_of(geo, c, cfg.patch_radius)
            ob['patch_cells'] = int(patch['cells'].size)
            ob['glue_seams'] = cg[c].astype(int).tolist()
            self.patches.append(patch)
        self.seam_feat = self.eng._dev(seam_feat)
        self.patch_ports = [self.eng._dev(p['ports']) for p in self.patches]
        self.tape = {k: [] for k in ('observer', 'sweep', 'pos', 'seam', 'slot', 'end', 'coin', 'current', 'twist', 'depth')}
        self.feat_cut = [{'depth': [], 'ticks': [], 'loads': []} for _ in self.feat]
        self.glue_slot = {(ob['carrier'], g): j for ob in self.feat for j, g in enumerate(ob['glue_seams'])}

    def _tape_rows(self, seq, coin, n, t_base, s):
        xp, e, geo = self.xp, self.eng, self.geo
        if not self.feat:
            return
        hit = xp.flatnonzero(self.seam_feat[seq[:n]])
        if hit.size == 0:
            return
        hit_h = e.host(hit)
        seam = e.host(seq[:n][hit]).astype(np.int64)
        co = e.host(coin[:n][hit]); cur = e.host(e.o_cur[:n][hit]); tw = e.host(e.o_twist[:n][hit]); dep = e.host(e.o_dep[:n][hit])
        T = self.tape
        for j in range(seam.size):
            sm = int(seam[j])
            if sm < geo.intra_seams:
                c = sm // 30
                rows = ((int(self.feat_index[c]), sm - 30 * c, 0),)
            else:
                g = sm - geo.intra_seams
                ca, cb = int(geo.glue_a[g]) // 12, int(geo.glue_b[g]) // 12
                rows = tuple((int(self.feat_index[c]), 30 + self.glue_slot[(c, g)], end)
                             for c, end in ((ca, 0), (cb, 1)) if self.feat_index[c] >= 0)
            for ob, slot, end in rows:
                T['observer'].append(ob); T['sweep'].append(s); T['pos'].append(t_base + int(hit_h[j])); T['seam'].append(sm)
                T['slot'].append(slot); T['end'].append(end); T['coin'].append(int(co[j])); T['current'].append(int(cur[j]))
                T['twist'].append(int(tw[j])); T['depth'].append(int(dep[j]))

    # ---- cuts
    def _open_cuts(self, s):
        if not self.use_cuts:
            return
        _, dmax, _ = self.eng.depth_range()
        margin = 256 if self.max_growth == 0 else max(64, 3 * self.max_growth)
        while self.next_cut[1] <= dmax + margin:
            k, T = self.next_cut
            if T < dmax:
                raise RuntimeError(f'cut {k} at depth {T} would open after depth {dmax} was reached')
            self.eng.open_cut(k, T, s)
            self.next_cut = (k + 1, next(self.cut_iter))

    def _close_cuts(self, s, final=False):
        dmin, dmax, _ = self.eng.depth_range()
        while self.eng.open and self.eng.open[0]['T'] <= dmin:
            slot = self.eng.open[0]['slot']
            cut = self.eng.open[0]
            entry = self._write_frame(f'cuts/cut_{cut["k"]:05d}.npz', cut['T'], self.eng.cut_x[slot], self.eng.cut_ticks[slot],
                                      self.eng.tracker.cut_cls[slot], self.eng.tracker.cut_fid[slot])
            entry.update({'k': cut['k'], 'opened_sweep': cut['opened'], 'closed_sweep': s})
            self.cuts.append(entry)
            self.eng.close_cut()

    def _write_frame(self, name, depth, x, ticks, cls_t, fid_t, end=False) -> dict:
        """One consistent slice: defects, identities, loads summary, clocks, observer patches."""
        e, geo, cfg, xp = self.eng, self.geo, self.cfg, self.xp
        tr = e.tracker
        cat = tr.cols
        idx = np.flatnonzero(cls_t)
        faces = (idx + e.F0).astype(np.int32)
        charge = cls_t[idx].astype(np.int8)
        classes = ((charge.astype(np.int16) % 6 + 6) % 6).astype(np.uint8) if cfg.twists else charge.astype(np.uint8)
        ids = fid_t[idx]
        assert (ids >= 0).all(), 'a defect face without identity'
        assert (cat['cls'][ids].view(np.int8) == charge).all(), 'identity class differs from the face class'
        if not end:
            bd, dd = cat['birth_depth'][ids], cat['death_depth'][ids]
            assert (bd <= depth).all() and ((dd < 0) | (dd > depth)).all(), 'identity outside its lifetime on the cut'
        screen = faces >= geo.intra_faces
        if cfg.twists:
            assert int(charge[screen].astype(np.int64).sum()) == 0, 'screen charges do not sum to zero'
        else:
            assert int(classes[screen].astype(np.int64).sum()) % 6 == 0, 'screen tier does not sum to zero mod 6'
        arrays = {'depth': np.int64(-1 if end else depth)}
        if cfg.twists:
            arrays['screen_charge'] = charge[screen]
        counts_screen = np.bincount(classes[screen], minlength=6)
        counts_intra = None
        arrays.update(screen_face=faces[screen], screen_cls=classes[screen], screen_id=ids[screen].astype(np.int32),
                      screen_birth_depth=cat['birth_depth'][ids[screen]], screen_xyz=geo.face_xyz(faces[screen]))
        if self.track_intra:
            fi, ci = faces[~screen], classes[~screen]
            per = np.bincount(fi // 20, weights=ci.astype(np.float64), minlength=geo.carriers)
            assert not (per.astype(np.int64) % 6).any(), 'an internal tier does not sum to zero mod 6'
            counts_intra = np.bincount(ci, minlength=6)
            arrays.update(intra_face=fi, intra_cls=ci, intra_id=ids[~screen].astype(np.int32),
                          intra_birth_depth=cat['birth_depth'][ids[~screen]], intra_xyz=geo.face_xyz(fi))
        else:
            full = e.hist6(e.cls[:geo.intra_faces]) if end else None
            counts_intra = full
        hist = e.hist6(x)
        v = int((x.astype(xp.int32) ** 2).sum(dtype=xp.int64)) if cfg.twists else int((hist * np.arange(hist.size) ** 2).sum())
        per_cell = x.reshape(geo.carriers, 12).astype(xp.int32).sum(axis=1)
        if cfg.twists:
            per_cell = xp.maximum(per_cell, 0)
        k = 4 ** (cfg.level - self.display_level)
        load_map = e.host(per_cell.reshape(-1, k).sum(axis=1)).astype(np.uint16)
        tk = e.host(xp.stack([ticks.min(), ticks.max()])).astype(np.int64)
        tmean = float(ticks.mean())
        feat_ticks = (e.host(ticks[e._dev(np.array([o['carrier'] for o in self.feat], np.int64))]).astype(np.int32)
                      if self.feat else np.zeros(0, np.int32))
        arrays.update(load_hist=hist, load_map=load_map, feat_ticks=feat_ticks)
        for i in range(len(self.feat)):
            self.feat_cut[i]['depth'].append(-1 if end else depth)
            self.feat_cut[i]['ticks'].append(int(feat_ticks[i]))
            self.feat_cut[i]['loads'].append(e.host(x[self.patch_ports[i]]).astype(np.uint8))
        if cfg.level <= cfg.full_cut_level:
            arrays.update(loads=e.host(x).astype(np.int8), ticks=e.host(ticks).astype(np.int32))
        np.savez_compressed(self.out / name, **arrays)
        return {'depth': None if end else int(depth), 'file': name, 'screen': int(screen.sum()), 'intra': int((~screen).sum()),
                'counts_screen': counts_screen.tolist(), 'counts_intra': None if counts_intra is None else np.asarray(counts_intra).tolist(),
                'V': v, 'load_hist': hist.tolist(), 'ticks_min': int(tk[0]), 'ticks_max': int(tk[1]), 'ticks_mean': tmean,
                'unsettled_ports': int(xp.count_nonzero((x != self.q) & (x != self.q + 1)))}

    # ---- checkpoints
    def _checkpoint(self, s):
        name = f'checkpoints/ckpt_{s:06d}.npz'
        np.savez_compressed(self.out / name, sweep=np.int64(s), **self.eng.state())
        meta = {'sweep': s, 'rng': self.eng.rng.bit_generator.state, 'next_cut': list(self.next_cut), 'max_growth': self.max_growth,
                'v': self.v, 'balanced_at': self.balanced_at, 'cuts': self.cuts, 'series_blocks': self.series_blocks,
                'records': self.records, 'checkpoints': self.checkpoints + [{'sweep': s, 'file': name}],
                'feat_cut': [{'depth': f['depth'], 'ticks': f['ticks']} for f in self.feat_cut]}
        (self.out / name).with_suffix('.json').write_text(json.dumps(meta))
        self._series_flush()
        np.savez_compressed(self.out / f'checkpoints/aux_{s:06d}.npz',
                            **{'stats_' + k: np.asarray(v) for k, v in self.series.items()},
                            **{'tape_' + k: np.asarray(v, dtype=np.int32) for k, v in self.tape.items()},
                            **{f'featcut_{i}': (np.stack(f['loads']) if f['loads'] else np.zeros((0, self.patches[i]['ports'].size), np.uint8))
                               for i, f in enumerate(self.feat_cut)})
        self.checkpoints.append({'sweep': s, 'file': name})

    def _resume(self, path: Path):
        z = np.load(path)
        meta = json.loads(path.with_suffix('.json').read_text())
        s = int(z['sweep'])
        self.eng.restore(z, meta['rng'])
        self.next_cut = tuple(meta['next_cut'])
        self.cut_iter = cut_depths(self.cfg.cut_ratio, self.cfg.cut_step)
        while next(self.cut_iter) != self.next_cut[1]:
            pass
        self.max_growth, self.v, self.balanced_at = meta['max_growth'], meta['v'], meta['balanced_at']
        self.cuts, self.records = meta['cuts'], meta['records']
        self.series_blocks = meta.get('series_blocks', [])
        self.checkpoints = [c for c in meta['checkpoints'] if c['sweep'] < s]
        aux = np.load(path.parent / f'aux_{s:06d}.npz')
        self.series = {k[6:]: aux[k].tolist() for k in aux.files if k.startswith('stats_')}
        self.tape = {k: aux['tape_' + k].tolist() for k in self.tape}
        for i, f in enumerate(self.feat_cut):
            f['depth'] = list(meta['feat_cut'][i]['depth']); f['ticks'] = list(meta['feat_cut'][i]['ticks'])
            f['loads'] = [row for row in aux[f'featcut_{i}']]
        if Path(path).resolve().parent.parent != self.out.resolve():
            for entry in self.cuts + self.records + self.series_blocks:   # continue in a new directory: bring the finished files along
                src, dst = path.parent.parent / entry['file'], self.out / entry['file']
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
        self.first_sweep = self.sweep = s
        self.resumed_from = str(path)

    # ---- one sweep
    def step(self):
        e, cfg, geo, xp = self.eng, self.cfg, self.geo, self.xp
        s = self.sweep
        if s % cfg.checkpoint_every == 0 and s != self.first_sweep or (s == 0 and self.first_sweep == 0):
            self._checkpoint(s)
        self._open_cuts(s)
        rng_state = e.rng.bit_generator.state
        seq, coin = e.draw()
        window = self._in_window(s)
        rec = {k: [] for k in ('current', 'contrast', 'total', 'twist', 'energy', 'depth')} if window else None
        acc = dict(descents=0, swaps=0, drop=0, accepted=0, blocked=0, accepted_screen=0, blocked_screen=0, work_used=0, released=0)
        d0 = e.depth_range()
        tr0 = e.tracker.stats.copy()
        S = geo.seams
        for lo in range(0, S, e.chunk):
            hi = min(S, lo + e.chunk)
            sq, cn = e.to_device(seq[lo:hi], coin[lo:hi]) if hasattr(e, 'to_device') else (seq[lo:hi], coin[lo:hi])
            n = e.run_chunk(sq, cn, lo, s)
            cur, con, tw, de = e.o_cur[:n], e.o_con[:n], e.o_twist[:n], e.o_de[:n]
            big = (con >= 2) | (con <= -2)
            nd = int(xp.count_nonzero(big)); nm = int(xp.count_nonzero(cur))
            dropv = cur.astype(xp.int32) * (con.astype(xp.int32) - cur)
            drop = int(dropv.sum(dtype=xp.int64))
            if self.balanced_at < 0 and not cfg.twists and self.v - 2 * drop == self.v_min:
                csum = xp.cumsum(dropv, dtype=xp.int64)
                first = int(xp.flatnonzero(self.v - 2 * csum == self.v_min)[0])
                self.balanced_at = s * S + lo + first + 1
            self.v -= 2 * drop
            ok = (tw != 0) if cfg.twists else (tw > 0)
            glue = sq >= geo.intra_seams
            acc['descents'] += nd; acc['swaps'] += nm - nd; acc['drop'] += drop
            blocked = (tw < 0) & ~ok
            acc['accepted'] += int(xp.count_nonzero(ok)); acc['blocked'] += int(xp.count_nonzero(blocked))
            acc['accepted_screen'] += int(xp.count_nonzero(ok & glue)); acc['blocked_screen'] += int(xp.count_nonzero(blocked & glue))
            dei = de.astype(xp.int32)
            acc['work_used'] += int(dei[ok & (de > 0)].sum(dtype=xp.int64)); acc['released'] += int(-dei[ok & (de < 0)].sum(dtype=xp.int64))
            self._tape_rows(sq, cn, n, lo, s)
            if rec is not None:
                rec['current'].append(e.host(cur).copy()); rec['contrast'].append(e.host(con).copy()); rec['twist'].append(e.host(tw).copy())
                rec['total'].append(e.host(e.o_tot[:n]).copy())
                rec['energy'].append(e.host(de).copy()); rec['depth'].append(e.host(e.o_dep[:n]).copy())
        if rec is not None:
            name = f'records/sweep_{s:06d}.npz'
            np.savez_compressed(self.out / name, seam=seq, coin=coin, **{k: np.concatenate(v) for k, v in rec.items()})
            (self.out / f'records/sweep_{s:06d}.rng.json').write_text(json.dumps(rng_state))
            self.records.append({'sweep': s, 'file': name, 'rng': f'records/sweep_{s:06d}.rng.json', 'attempts': int(S)})
        d1 = e.depth_range()
        self.max_growth = max(self.max_growth, d1[1] - d0[1])
        if self.use_cuts and self.next_cut[1] < d1[1]:
            raise RuntimeError(f'depth grew past unopened cut {self.next_cut}; rerun with a larger opening margin')
        self.sweep = s + 1
        self._close_cuts(self.sweep)
        nat = e.dnat
        hist = e.hist6(e.x)
        if cfg.twists:                                    # with offsets the sum of squares is not a ledger of the releases
            self.v = int((e.x.astype(xp.int32) ** 2).sum(dtype=xp.int64))
        else:
            assert int((hist * np.arange(hist.size) ** 2).sum()) == self.v, 'descent functional ledger differs from the loads'
        row = dict(V=self.v, descents=acc['descents'], swaps=acc['swaps'], waits=S - acc['descents'] - acc['swaps'],
                   potential_released=acc['drop'], twists_accepted=acc['accepted'], twists_blocked=acc['blocked'],
                   twists_accepted_screen=acc['accepted_screen'], twists_blocked_screen=acc['blocked_screen'],
                   work_offered=cfg.kappa * acc['drop'], work_used=acc['work_used'], energy_released=acc['released'],
                   unsettled_ports=int(xp.count_nonzero((e.x != self.q) & (e.x != self.q + 1))),
                   depth_min=d1[0], depth_max=d1[1], depth_mean=round(d1[2], 3),
                   depth_native_min=int(nat.min()), depth_native_max=int(nat.max()), depth_native_mean=round(float(nat.mean()), 3),
                   ticks_min=int(e.ticks.min()), ticks_max=int(e.ticks.max()),
                   counts_screen=e.hist6(e.cls[geo.intra_faces:]).tolist(),
                   counts_intra=e.hist6(e.cls[:geo.intra_faces]).tolist(),
                   identities=int(e.tracker.n), open_cuts=len(e.open), seconds=round(time.time() - self.t0, 2))
        for i, name in enumerate(STATS):
            row[name] = int(e.tracker.stats[i] - tr0[i])
        for k, v in row.items():
            self.series.setdefault(k, []).append(v)
        if cfg.series_every and self.sweep % cfg.series_every == 0:
            self._series_snapshot()
        return row

    # ---- density series on a fixed sweep cadence
    def _series_snapshot(self):
        e, geo, cfg, xp = self.eng, self.geo, self.cfg, self.xp
        cells = e.x.reshape(geo.carriers, 12).astype(xp.int32).sum(axis=1)
        snap = {'sweep': np.int64(self.sweep), 'cells': e.host(cells).astype(np.int16 if int(cells.min()) < 0 or int(cells.max()) > 255 else np.uint8),
                'nonzero_current_events': np.minimum(e.host(e.nzc), 65535).astype(np.uint16),
                'descent_events': np.minimum(e.host(e.dsc), 65535).astype(np.uint16)}
        if cfg.level > 5:
            snap['level5_sums'] = e.host(cells.reshape(20 * 4 ** 5, -1).sum(axis=1)).astype(np.int32)
        if self.series_ports:
            snap['ports'] = e.host(e.x).copy()
        if cfg.twists:
            snap['plaquette_charge'] = e.host(e.cls[geo.intra_faces:]).copy()
        e.nzc[:] = 0; e.dsc[:] = 0
        for k, v in snap.items():
            self.series_buf.setdefault(k, []).append(v)
            self.series_bytes += int(np.asarray(v).nbytes)
        if self.series_bytes >= (1 << 28):
            self._series_flush()

    def _series_flush(self):
        if not self.series_buf:
            return
        k = len(self.series_blocks)
        name = f'series/block_{k:05d}.npz'
        arrays = {key: np.stack(v) for key, v in self.series_buf.items()}
        np.savez_compressed(self.out / name, **arrays)
        self.series_blocks.append({'file': name, 'first_sweep': int(arrays['sweep'][0]), 'last_sweep': int(arrays['sweep'][-1]),
                                   'snapshots': int(arrays['sweep'].size)})
        self.series_buf, self.series_bytes = {}, 0

    # ---- driver
    def run(self):
        cfg = self.cfg
        stop = self.first_sweep + cfg.sweeps
        while self.sweep < stop:
            row = self.step()
            s = self.sweep - 1
            if s < 5 or s % 10 == 0 or self.sweep == stop:
                cs = row['counts_screen']
                self.log(f'sweep {s}: descents={row["descents"]} swaps={row["swaps"]} unsettled={row["unsettled_ports"]} '
                         f'depth={row["depth_min"]}..{row["depth_max"]} twists ok/blocked={row["twists_accepted"]}/{row["twists_blocked"]} '
                         f'screen A/B/C={cs[1] + cs[5]}/{cs[2] + cs[4]}/{cs[3]} ids={row["identities"]} cuts={len(self.cuts)}+{row["open_cuts"]} '
                         f't={row["seconds"]:.0f}s')
        return self.finish()

    def finish(self) -> dict:
        e, geo, cfg = self.eng, self.geo, self.cfg
        s = self.sweep
        self._checkpoint(s)
        # exact end checks
        reg_h, cls_h = e.host(e.reg), e.host(e.cls)
        if cfg.twists:
            assert (geo.face_curl(e.host(e.off)) == cls_h).all() if geo.level <= 8 else True, 'charges differ from the holonomies of the offsets'
            np.save(self.out / 'final_offsets.npy', e.host(e.off).astype(np.int8))
            np.save(self.out / 'initial_offsets.npy', self.offsets0)
        else:
            assert (geo.face_classes(reg_h) == cls_h).all() if geo.level <= 8 else True, 'classes differ from the loop sums of the register'
        live = e.tracker.face_id >= 0
        assert (live == (cls_h[e.F0:] != 0)).all(), 'identity map disagrees with the classes'
        end = self._write_frame('cuts/end.npz', -1, e.x, e.ticks, cls_h[e.F0:], e.tracker.face_id, end=True)
        end.update({'k': None, 'sweep': s, 'kind': 'end_of_run'})
        self._write_catalogue()
        self._write_observers()
        self._write_geometry()
        np.save(self.out / 'final_loads.npy', e.host(e.x).astype(np.int8))
        self._series_flush()
        np.savez_compressed(self.out / 'stats.npz', **{k: np.asarray(v) for k, v in self.series.items()})
        regime = ('quench: the seeded loads carry variance 35/12 per port against 1/4 in the balanced band; block totals are conserved, '
                  'so after the balanced class is reached the long density modes are still far from the canonical equilibrium and relax '
                  'at rate 2 lambda_k in scaled time tau = sweeps / (2 * 4^level); statistics of this run are quench statistics'
                  if self.prep['name'] == 'exact06' else
                  'stationary: uniform fixed-number preparation, the canonical equilibrium of the law; no settling phase')
        prov = provenance()
        meta = {
            'schema': SCHEMA, 'schema_doc': 'DATASET.md',
            'config': {**asdict(cfg), 'track_intra': self.track_intra, 'seed': self.seed, 'cost_table': list(COSTS[cfg.cost]),
                       'chunk': e.chunk, 'cut_depth_rule': f'0, then geometric ratio {cfg.cut_ratio} from 1 below {2 * cfg.cut_step}, then every {cfg.cut_step}'},
            'law': {'name': 'law C with transported contrast and unit twists (declared extension, not the exact layer)' if cfg.twists else 'law C: integer nearest agreement with a fair tie coin',
                    'formula': 'attempt on seam (a, b) with loads x_a, x_b and coin c: t = x_a + x_b; x_a becomes ceil(t/2) if c = 1 else floor(t/2); x_b = t - x_a; current = old x_a - new x_a',
                    'schedule': 'per sweep seq = default_rng(seed).integers(0, S, size=S, dtype=int64), then coin = integers(0, 2, size=S, dtype=int64); attempt i acts on seam seq[i] with coin[i]',
                    'seed_rule': f'{SCHEDULE_SEED_BASE} + 1000 * level + 100 * {GLUING_INDEX} + schedule',
                    'source': 'oph_exact.federation.run_integer_law (native kernel oph_fed_integer)'},
            'preparation': dict(self.prep, offsets=(cfg.offsets or 'all zero') if cfg.twists else None), 'regime': regime,
            'scaled_time': {'tau_per_sweep': 1.0 / (2 * 4 ** cfg.level), 'rule': 'tau = attempts / (1260 * 16^level) = sweeps / (2 * 4^level)'},
            'layers': {
                'depth': 'causal depth of an attempt = 1 + max depth of the earlier attempts on its resources; resources are its two ports and, for a glue seam, its two screen plaquettes; depth_native uses ports only',
                'cuts': 'cut T = all attempts of depth <= T, a down-set of the causal order; the state on the cut is the state after exactly those attempts',
                'clocks': 'ticks[c] = attempts on the 33 seams incident to carrier c',
                'register': ({'kind': 'declared extension: law C with transported contrast and unit twists; the loads of this run are NOT those of plain law C',
                              'rule': 'every seam carries an integer offset b_e (zero at the start) and compares x_a with x_b + b_e; at an attempt on a glue seam with contrast d = x_a - x_b - b_e != 0 the offset moves by sign(d) when the seam potential |d| - 1/2 exceeds the change of the cost of the two plaquette holonomies (cost in units of kappa per unit of potential, additive beyond 3), with the coin deciding exact ties; otherwise the transported repair runs. A plaquette of nonzero holonomy is a charge',
                              'cost': list(COSTS[cfg.cost]), 'kappa': cfg.kappa, 'randomness': 'the tie coin of the attempt'}
                             if cfg.twists else
                             {'kind': 'declared extension', 'rule': 'a repair with current j proposes delta = j mod 6 on its seam link; the faces that traverse the seam a->b and b->a change by +delta and -delta; it is applied when cost[new] - cost[old] <= kappa * j * (d - j), the mismatch potential 1/2 sum x^2 released by the repair in cost units',
                              'cost': list(COSTS[cfg.cost]), 'kappa': cfg.kappa, 'randomness': 'none'})},
            'geometry': {'level': cfg.level, 'carriers': geo.carriers, 'ports': geo.ports, 'seams': geo.seams,
                         'intra_seams': geo.intra_seams, 'glue_seams': geo.glue_seams, 'faces': geo.faces,
                         'intra_faces': geo.intra_faces, 'screen_plaquettes': geo.vertices, 'display_level': self.display_level,
                         'template_seams': FG.TEMPLATE_SEAMS.tolist(), 'template_faces': FG.TEMPLATE_FACES.tolist(),
                         'template_face_plus': FG.TEMPLATE_PLUS.tolist(), 'template_face_minus': FG.TEMPLATE_MINUS.tolist(),
                         'port_vectors': FG.PORT_VECTORS.tolist(), 'base_vertices': FG.PORT_UNIT.tolist(),
                         'base_faces': FG.TEMPLATE_FACES.tolist(), 'cell_spacing': geo.cell_spacing(),
                         'geometry_source_sha256': geo.source_sha256,
                         'gluing': 'port_pair (oph_fpe.core.screen_ports.assign_echosahedral_ports)',
                         'gluing_note': 'the routing reads port directions in a tangent frame built from the fixed axis (0,0,1), '
                                        'switching to (1,0,0) where |normal_z| > 0.9; the port graph therefore has no icosahedral '
                                        'rotation symmetry and the three slowest modes of the port Laplacian are split by about 3 percent '
                                        '(4^l lambda = 0.0408, 0.0416, 0.0419 at level 5). A property of the declared gluing, not of law C'},
            'totals': {'total_load': self.total, 'band': [self.q, self.q + 1], 'V_initial': int((self.loads0.astype(np.int64) ** 2).sum()),
                       'V_minimum': self.v_min, 'V_final': self.v, 'attempts_to_balanced_class': self.balanced_at,
                       'balanced': self.v == self.v_min},
            'sweeps': s, 'first_sweep': self.first_sweep, 'resumed_from': getattr(self, 'resumed_from', None),
            'stats_file': 'stats.npz', 'stats_fields': sorted(self.series), 'cuts': self.cuts, 'end': end,
            'checkpoints': self.checkpoints, 'records': self.records, 'series': self.series_blocks,
            'observers': [{k: v for k, v in ob.items()} | {'file': f'observers/obs_{ob["carrier"]}.npz'} for ob in self.feat],
            'identities': int(e.tracker.n), 'tracker_totals': {n: int(v) for n, v in zip(STATS, e.tracker.stats)},
            'provenance': prov, 'runtime_seconds': round(time.time() - self.t0, 1),
        }
        (self.out / 'run.json').write_text(json.dumps(meta))
        meta = dict(meta, stats=self.series)
        from . import dataset
        dataset.build_view(self.out)
        dataset.write_card(self.out, meta)
        self.log(f'done: {s} sweeps, {e.tracker.n} identities, {len(self.cuts)} cuts, balanced={meta["totals"]["balanced"]} '
                 f'(attempt {self.balanced_at}), {time.time() - self.t0:.0f}s')
        return meta

    def _write_catalogue(self):
        e, geo = self.eng, self.geo
        cols = {k: v.copy() for k, v in e.tracker.view().items()}
        cols['tier'] = (cols['birth_face'] >= geo.intra_faces).astype(np.uint8)
        if e.tracker.n <= 8_000_000:
            cols['birth_xyz'] = geo.face_xyz(cols['birth_face'])
            cols['last_xyz'] = geo.face_xyz(cols['last_face'])
        np.savez_compressed(self.out / 'catalogue.npz', **cols)

    def _write_observers(self):
        e, geo = self.eng, self.geo
        T = {k: np.asarray(v) for k, v in self.tape.items()}
        x_end = e.host(e.x)
        W = np.array([np.cross(FG.PORT_UNIT[a], FG.PORT_UNIT[b]) for a, b in FG.TEMPLATE_SEAMS])    # [30, 3]
        for i, ob in enumerate(self.feat):
            c, patch = ob['carrier'], self.patches[i]
            m = T['observer'] == i
            rows = {k: T[k][m] for k in T if k != 'observer'}
            own0 = self.loads0[12 * c:12 * c + 12].astype(np.int64)
            x = own0.copy()
            n = int(m.sum())
            disp = np.zeros((n + 1, 3)); axial = np.zeros((n + 1, 3)); loads_t = np.zeros((n + 1, 12), np.int8)
            disp[0] = (x[:, None] * FG.PORT_VECTORS).sum(axis=0); loads_t[0] = x
            z = np.zeros(3)
            for j in range(n):
                slot, cur, end, sm = int(rows['slot'][j]), int(rows['current'][j]), int(rows['end'][j]), int(rows['seam'][j])
                if slot < 30:
                    a, b = FG.TEMPLATE_SEAMS[slot]
                    x[a] -= cur; x[b] += cur
                    z = z + cur * W[slot]
                else:
                    g = sm - geo.intra_seams
                    p = int(geo.glue_a[g] if end == 0 else geo.glue_b[g]) - 12 * c
                    x[p] += -cur if end == 0 else cur
                disp[j + 1] = (x[:, None] * FG.PORT_VECTORS).sum(axis=0); axial[j + 1] = z; loads_t[j + 1] = x
            assert (x == x_end[12 * c:12 * c + 12]).all(), 'an observer tape does not reproduce the observer loads'
            fc = self.feat_cut[i]
            np.savez_compressed(
                self.out / f'observers/obs_{c}.npz', carrier=np.int64(c), xyz=geo.carrier_xyz[c].astype(np.float32),
                frame=geo.frames(np.array([c]))[0].astype(np.float32), loads_initial=own0.astype(np.int8),
                loads_final=x.astype(np.int8), patch_cells=patch['cells'], patch_dist=patch['dist'], patch_seams=patch['seams'],
                patch_cell_xyz=patch['cell_xyz'], patch_frames=patch['frames'],
                patch_loads_initial=self.loads0[patch['ports']].astype(np.uint8), patch_loads_final=x_end[patch['ports']].astype(np.uint8),
                tape_sweep=rows['sweep'].astype(np.int32), tape_pos=rows['pos'].astype(np.int32), tape_seam=rows['seam'].astype(np.int32),
                tape_slot=rows['slot'].astype(np.int8), tape_end=rows['end'].astype(np.int8), tape_coin=rows['coin'].astype(np.uint8),
                tape_current=rows['current'].astype(np.int8), tape_twist=rows['twist'].astype(np.int8), tape_depth=rows['depth'].astype(np.int32),
                loads_by_tick=loads_t, displacement=disp.astype(np.float32), axial=axial.astype(np.float32),
                cut_depth=np.asarray(fc['depth'], np.int32), cut_ticks=np.asarray(fc['ticks'], np.int32),
                cut_patch_loads=np.stack(fc['loads']) if fc['loads'] else np.zeros((0, patch['ports'].size), np.uint8))

    def _write_geometry(self):
        geo, cfg = self.geo, self.cfg
        if cfg.level > cfg.geometry_level:
            return
        np.savez_compressed(self.out / 'geometry.npz', glue_a=geo.glue_a, glue_b=geo.glue_b, glue_fp=geo.glue_fp, glue_fm=geo.glue_fm,
                            tri=geo.tri, mesh_xyz=geo.mesh_xyz.astype(np.float32), carrier_xyz=geo.carrier_xyz.astype(np.float32),
                            neighbours=geo.neighbours(), loads_initial=self.loads0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--level', type=int, required=True)
    p.add_argument('--sweeps', type=int, default=300)
    p.add_argument('--schedule', type=int, default=0)
    p.add_argument('--prep', default='exact06')
    p.add_argument('--cost', default='binding', choices=sorted(COSTS))
    p.add_argument('--kappa', type=int, default=5)
    p.add_argument('--backend', default='sequential', choices=('sequential', 'layers', 'cuda'))
    p.add_argument('--track-intra', dest='track_intra', action='store_true', default=None)
    p.add_argument('--no-track-intra', dest='track_intra', action='store_false')
    p.add_argument('--cut-ratio', type=float, default=1.5)
    p.add_argument('--cut-step', type=int, default=64)
    p.add_argument('--slots', type=int, default=24)
    p.add_argument('--checkpoint-every', type=int, default=50)
    p.add_argument('--records', default='', help="'all' or sweep ranges a:b,c:d with complete per-attempt records")
    p.add_argument('--patch-radius', type=int, default=16)
    p.add_argument('--full-cut-level', type=int, default=6)
    p.add_argument('--chunk', type=int, default=0)
    p.add_argument('--fast-rng', dest='fast_rng', action='store_true', default=None)
    p.add_argument('--no-register', dest='register', action='store_false')
    p.add_argument('--twists', action='store_true', help='declared extension: transported contrast with unit twists (charges act back on the loads)')
    p.add_argument('--offsets', default='', help="with --twists: 'iid:<seed>' prepares independent glue offsets in {-1, 0, +1}")
    p.add_argument('--no-track', dest='track', action='store_false', help='no identities (dense stationary charge gases)')
    p.add_argument('--no-observers', dest='observers', action='store_false')
    p.add_argument('--series-every', type=int, default=0, help='sweeps between density snapshots (cell totals, event counts)')
    p.add_argument('--resume', type=Path, default=None, help='checkpoint .npz to continue from')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    cfg = Config(level=a.level, sweeps=a.sweeps, schedule=a.schedule, prep=a.prep, cost=a.cost, kappa=a.kappa, backend=a.backend,
                 track_intra=a.track_intra, cut_ratio=a.cut_ratio, cut_step=a.cut_step, slots=a.slots,
                 checkpoint_every=a.checkpoint_every, records=a.records, patch_radius=a.patch_radius,
                 full_cut_level=a.full_cut_level, chunk=a.chunk, fast_rng=a.fast_rng, register=a.register,
                 observers=a.observers, series_every=a.series_every, twists=a.twists, offsets=a.offsets, track=a.track)
    Run(cfg, a.out, resume=a.resume).run()


if __name__ == '__main__':
    main()
