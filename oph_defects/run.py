"""Run a Z6 defect catalogue on the refined carrier and export it for the viewer.

    python -m oph_defects.run --level 10 --ticks 4000 --seed 7 --out runs/z6_L10_seed7

The run is deterministic for a fixed seed, numpy version and platform. Every
snapshot, the complete identity catalogue and per-tick statistics are written
as small gzip files that the viewer streams.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np

from . import geometry
from .dynamics import COST, Z6Run, region_check, run_config, temperature_schedule

BINDING = np.array([0, 5, 8, 10, 8, 5], dtype=np.int64)
ADDITIVE = np.array([0, 5, 10, 15, 10, 5], dtype=np.int64)
SNAP_MAGIC = b'Z6SN'
CAT_MAGIC = b'Z6CT'


def parse_knots(text: str) -> list[tuple[int, float]]:
    knots = []
    for part in text.split(','):
        t, v = part.split(':')
        knots.append((int(t), float(v)))
    return knots


def write_snapshot(path: Path, tick: int, faces: np.ndarray, classes: np.ndarray, ids: np.ndarray) -> int:
    body = (SNAP_MAGIC + struct.pack('<II', tick, len(faces))
            + faces.astype('<i4').tobytes() + classes.astype('u1').tobytes() + ids.astype('<i4').tobytes())
    with gzip.open(path, 'wb', compresslevel=6) as fh:
        fh.write(body)
    return len(faces)


def write_catalogue(path: Path, cols: dict[str, np.ndarray]) -> int:
    n = len(cols['cls'])
    body = CAT_MAGIC + struct.pack('<I', n)
    for name, dtype in (('cls', 'u1'), ('birth', '<i4'), ('death', '<i4'), ('bface', '<i4'), ('lface', '<i4'),
                        ('hops', '<i4'), ('partner', '<i4'), ('p1', '<i4'), ('p2', '<i4'), ('cause', 'u1'),
                        ('c1', '<i4'), ('c2', '<i4')):
        body += cols[name].astype(dtype).tobytes()
    with gzip.open(path, 'wb', compresslevel=6) as fh:
        fh.write(body)
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--level', type=int, default=7)
    p.add_argument('--ticks', type=int, default=4000)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--proposal-fraction', type=float, default=0.1)
    p.add_argument('--schedule', default='0:1.1,600:1.1,1800:0.0,4000:0.0',
                   help='tick:temperature knots, linear in between (cost units of J/5)')
    p.add_argument('--cost', choices=('binding', 'additive'), default='binding')
    p.add_argument('--snapshot-every', type=int, default=20)
    p.add_argument('--check-every', type=int, default=200)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    out = a.out
    (out / 'snapshots').mkdir(parents=True, exist_ok=True)
    cost = BINDING if a.cost == 'binding' else ADDITIVE
    knots = parse_knots(a.schedule)
    schedule = temperature_schedule(knots, a.ticks)
    t0 = time.time()
    surface = geometry.build(a.level)
    print(f'surface level {a.level}: {surface.vertex_count} observers, {surface.edge_count} seams, '
          f'{surface.face_count} faces ({time.time() - t0:.1f}s)', flush=True)
    run = Z6Run(surface, a.seed, a.proposal_fraction, cost)
    check_rng = np.random.default_rng(a.seed + 1)
    snaps = []
    log = out / 'progress.log'
    with np.errstate(over='ignore'):
        for t in range(a.ticks):
            run.step(float(schedule[t]))
            if t % a.snapshot_every == 0 or t == a.ticks - 1:
                faces = np.flatnonzero(run.curv).astype(np.int32)
                name = f'snapshots/snap_{t:06d}.bin.gz'
                n = write_snapshot(out / name, t, faces, run.curv[faces], run.face_id[faces])
                snaps.append({'tick': t, 'file': name, 'count': n})
            if t % a.check_every == 0 or t == a.ticks - 1:
                info = run.check()
                region_check(run, check_rng)
                c = run.stats.counts[-1]
                line = (f'tick {t} T={schedule[t]:.3f} classes={c[1:]} live={info["live"]} ids={info["identities"]} '
                        f'births={run.stats.births[-1]} moves={run.stats.moves[-1]} '
                        f'annih={run.stats.annihilations[-1]} fus={run.stats.fusions[-1]} '
                        f'elapsed={time.time() - t0:.0f}s')
                print(line, flush=True)
                with open(log, 'a') as fh:
                    fh.write(line + '\n')
    cols = run.cat.view()
    n_ids = write_catalogue(out / 'catalogue.bin.gz', cols)
    st = run.stats
    meta = {
        'config': run_config(a.level, a.ticks, a.seed, a.proposal_fraction, knots, a.snapshot_every)
        | {'cost': cost.tolist(), 'cost_name': a.cost},
        'geometry': {'level': a.level, 'observers': surface.vertex_count, 'seams': surface.edge_count,
                     'faces': surface.face_count, 'base_faces': geometry.BASE_FACES.tolist(),
                     'base_vertices': geometry.BASE_VERTICES.tolist()},
        'stats': {'temperature': st.temperature, 'counts': st.counts, 'births': st.births, 'moves': st.moves,
                  'annihilations': st.annihilations, 'fusions': st.fusions, 'splits': st.splits,
                  'exchanges': st.exchanges, 'accepted': st.accepted, 'kept': st.kept, 'cost': st.cost},
        'snapshots': snaps,
        'identities': n_ids,
        'runtime_seconds': round(time.time() - t0, 1),
        'numpy': np.__version__, 'python': sys.version.split()[0],
    }
    (out / 'run.json').write_text(json.dumps(meta))
    digest = hashlib.sha256()
    for path in sorted(out.rglob('*.bin.gz')):
        digest.update(path.read_bytes())
    (out / 'DIGEST').write_text(digest.hexdigest() + '\n')
    print(f'done: {n_ids} identities, {len(snaps)} snapshots, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
