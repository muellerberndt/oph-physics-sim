"""GPU execution of the universe kernel: the causal layers of each schedule chunk in parallel.

The attempt body is the same C text as the CPU kernels (``kernels.CORE``).
A chunk of the schedule is executed in rounds: every pending attempt marks
its resources with its schedule position (atomic minimum); an attempt whose
resources all carry its own position has no earlier pending attempt on any
of them and runs; the rest wait for the next round. Attempts of one round
are disjoint, so the result equals the sequential order bit for bit. The
tests compare both orders on the CPU; ``selfcheck`` compares GPU and CPU.
"""
from __future__ import annotations

import numpy as np

from . import fedgeom as FG
from . import kernels as K
from .universe import Engine

CUDA_TYPES = r"""
#define int8_t signed char
#define uint8_t unsigned char
#define int32_t int
#define int64_t long long
"""


class CudaEngine(Engine):
    def __init__(self, *args, **kw):
        import cupy
        self.cp = cupy
        self.xp = cupy
        kw['backend'] = 'cuda'
        super().__init__(*args, **kw)

    def _alloc(self, loads):
        cp, g = self.cp, self.geo
        n = self.chunk
        self.x = cp.asarray(np.ascontiguousarray(loads, dtype=np.int8))
        self.reg = cp.zeros(self.S, cp.int8)
        self.cls = cp.zeros(self.F, cp.int8)
        self.dnat = cp.zeros(self.P, cp.int32)
        self.dsyn = cp.zeros(self.P + self.V, cp.int32)
        self.ticks = cp.zeros(self.N, cp.int32)
        self.cut_x = cp.zeros((self.slots, self.P), cp.int8)
        self.cut_ticks = cp.zeros((self.slots, self.N), cp.int32)
        self.cutT = cp.zeros(self.slots, cp.int32)
        self.cut_slot = cp.zeros(self.slots, cp.int32)
        self.cutT_host = np.zeros(self.slots, np.int32)
        self.cut_slot_host = np.zeros(self.slots, np.int32)
        self.o_cur = cp.zeros(n, cp.int8); self.o_con = cp.zeros(n, cp.int8)
        self.o_twist = cp.zeros(n, cp.int8); self.o_de = cp.zeros(n, cp.int8); self.o_tot = cp.zeros(n, cp.int8)
        self.o_dep = cp.zeros(n, cp.int32)
        self.off = cp.asarray(self.offsets0) if self.flags & K.F_OFF else cp.zeros(1, cp.int8)
        self.jsum = cp.zeros(self.S if self.flags & K.F_JSUM else 1, cp.int32)
        self.nzc = cp.zeros(self.N if self.flags & K.F_COUNT else 1, cp.int32)
        self.dsc = cp.zeros(self.N if self.flags & K.F_COUNT else 1, cp.int32)
        self.events = cp.zeros((n, 8), cp.int32)
        self.ev_count = cp.zeros(2, cp.int64)
        self.ta = cp.asarray(FG.TEMPLATE_SEAMS[:, 0].astype(np.int8)); self.tb = cp.asarray(FG.TEMPLATE_SEAMS[:, 1].astype(np.int8))
        self.tp = cp.asarray(FG.TEMPLATE_PLUS.astype(np.int8)); self.tm = cp.asarray(FG.TEMPLATE_MINUS.astype(np.int8))
        self.ga = cp.asarray(g.glue_a.astype(np.int32)); self.gb = cp.asarray(g.glue_b.astype(np.int32))
        self.gfp = cp.asarray(g.glue_fp.astype(np.int32)); self.gfm = cp.asarray(g.glue_fm.astype(np.int32))
        self.cost_d = cp.asarray(self.cost.astype(np.int32))
        self.nxt = cp.full(self.P + self.V, K.INF, cp.int32)
        self.done = cp.zeros(n, cp.uint8)
        src = CUDA_TYPES + K.CUDA_SOURCE.replace('typedef signed char int8_t;', '').replace('typedef unsigned char uint8_t;', '') \
            .replace('typedef int int32_t;', '').replace('typedef long long int64_t;', '')
        self.module = cp.RawModule(code=src, options=('--std=c++11',))
        self.k_mark = self.module.get_function('oph_mark')
        self.k_exec = self.module.get_function('oph_exec')
        self.k_reset = self.module.get_function('oph_reset')
        self.ncut = 0
        self.rounds = 0
        self.lib = K.library()

    def host(self, a):
        return self.cp.asnumpy(a) if isinstance(a, self.cp.ndarray) else a

    def _dev(self, a):
        return self.cp.asarray(a)

    def to_device(self, seq, coin):
        return self.cp.asarray(seq), self.cp.asarray(coin)

    def _ctx_args(self):
        i64, i32 = np.int64, np.int32
        return (i64(self.N), i64(self.P), i64(self.SI), i64(self.FI), i64(self.F0), self.ta, self.tb, self.tp, self.tm,
                self.ga, self.gb, self.gfp, self.gfm, self.x, self.reg, self.cls, self.dnat, self.dsyn, self.ticks,
                self.cost_d, i32(self.kappa), i32(self.ncut), self.cutT, self.cut_slot, self.cut_x, self.cut_ticks,
                self.o_cur, self.o_con, self.o_twist, self.o_de, self.o_dep, self.events, i64(self.chunk), self.ev_count,
                self.o_tot, i32(self.flags), self.off, self.jsum, self.nzc, self.dsc)

    def run_chunk(self, seq, coin, t_base: int, sweep: int) -> int:
        cp = self.cp
        n = int(seq.shape[0])
        assert n <= self.chunk and seq.dtype == cp.int32 and coin.dtype == cp.uint8
        self.ev_count[:] = 0
        args = self._ctx_args()
        pend = cp.arange(n, dtype=cp.int32)
        m, block = n, 256
        while m > 0:
            grid = ((m + block - 1) // block,)
            self.k_mark(grid, (block,), (pend, np.int64(m), seq, self.nxt) + args)
            self.k_exec(grid, (block,), (pend, np.int64(m), seq, coin, np.int64(t_base), self.nxt, self.done) + args)
            self.k_reset(grid, (block,), (pend, np.int64(m), seq, self.nxt) + args)
            pend = pend[self.done[:m] == 0]
            m = int(pend.shape[0])
            self.rounds += 1
        count = cp.asnumpy(self.ev_count)
        if count[1] or count[0] > self.chunk:
            raise RuntimeError('event buffer overflow')
        ev = cp.asnumpy(self.events[:int(count[0])])
        if ev.shape[0]:
            ev = ev[np.argsort(ev[:, 0], kind='stable')]
        self.last_events = ev
        self.tracker.process(ev, sweep, self.cutT_host, self.cut_slot_host, len(self.open))
        return n

    def _sync_cut_arrays(self):
        for q, cut in enumerate(self.open):
            self.cutT_host[q] = cut['T']; self.cut_slot_host[q] = cut['slot']
        self.cutT[:] = self.cp.asarray(self.cutT_host)
        self.cut_slot[:] = self.cp.asarray(self.cut_slot_host)
        self.ncut = len(self.open)


def selfcheck(level: int = 5, sweeps: int = 6, chunk: int = 0, twists: bool = False) -> dict:
    """Run the same sweeps on the GPU and with the sequential CPU kernel; every array must be equal."""
    from .universe import COSTS, prepare_loads, schedule_seed
    geo = FG.build(level)
    loads, _ = prepare_loads('exact06', level, geo.ports)
    seed = schedule_seed(level, 0)
    track = level <= 6 and not twists
    kw = dict(register=False, offsets=np.zeros(geo.seams, np.int8), twists=True, currents=True, counters=True) if twists else {}
    cpu = Engine(geo, loads, seed, COSTS['binding'], 5, track, **kw)
    gpu = CudaEngine(geo, loads, seed, COSTS['binding'], 5, track, chunk=chunk, **kw)
    for e in (cpu, gpu):
        e.open_cut(1, 5, 0); e.open_cut(2, 40, 0); e.open_cut(3, 1 << 30, 0)
    import time
    t_cpu = t_gpu = 0.0
    for s in range(sweeps):
        seq, coin = cpu.draw()
        seq2, coin2 = gpu.draw()
        assert (seq == seq2).all() and (coin == coin2).all()
        t = time.time()
        rec_cpu = {k: [] for k in ('cur', 'con', 'twist', 'de', 'dep')}
        for lo in range(0, cpu.S, cpu.chunk):
            n = cpu.run_chunk(seq[lo:lo + cpu.chunk], coin[lo:lo + cpu.chunk], lo, s)
            for k in rec_cpu:
                rec_cpu[k].append(getattr(cpu, 'o_' + k)[:n].copy())
        t_cpu += time.time() - t
        t = time.time()
        rec_gpu = {k: [] for k in rec_cpu}
        for lo in range(0, gpu.S, gpu.chunk):
            sq, cn = gpu.to_device(seq[lo:lo + gpu.chunk], coin[lo:lo + gpu.chunk])
            n = gpu.run_chunk(sq, cn, lo, s)
            for k in rec_gpu:
                rec_gpu[k].append(gpu.host(getattr(gpu, 'o_' + k)[:n]))
        gpu.cp.cuda.Stream.null.synchronize()
        t_gpu += time.time() - t
        for k in rec_cpu:
            assert (np.concatenate(rec_cpu[k]) == np.concatenate(rec_gpu[k])).all(), ('per-attempt output differs', k, s)
    a, b = cpu.state(), gpu.state()
    for k in a:
        assert a[k].shape == b[k].shape and (a[k] == b[k]).all(), ('state differs', k)
    if twists:
        for k in ('off', 'jsum', 'nzc', 'dsc'):
            assert (cpu.host(getattr(cpu, k)) == gpu.host(getattr(gpu, k))).all(), ('state differs', k)
    return {'level': level, 'sweeps': sweeps, 'attempts': sweeps * geo.seams, 'equal': True, 'twists': twists, 'identities': int(cpu.tracker.n),
            'cpu_seconds_per_sweep': t_cpu / sweeps, 'gpu_seconds_per_sweep': t_gpu / sweeps,
            'gpu_rounds_per_chunk': gpu.rounds / max(1, sweeps * -(-geo.seams // gpu.chunk)), 'gpu_chunk': gpu.chunk}


if __name__ == '__main__':
    import json
    import sys
    lv = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(json.dumps(selfcheck(lv, int(sys.argv[2]) if len(sys.argv) > 2 else 6, twists='twists' in sys.argv)))
