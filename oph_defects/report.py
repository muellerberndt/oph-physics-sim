"""Catalogue statistics of a finished run.

    python -m oph_defects.report runs/z6_L10_seed7 [--json out.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import struct

import numpy as np

from .geometry import face_positions

FAMILY = {1: 'A', 5: 'A', 2: 'B', 4: 'B', 3: 'C'}
CAUSE = ['alive', 'annihilation', 'fusion', 'split', 'exchange']


def load_catalogue(path: Path) -> dict[str, np.ndarray]:
    raw = gzip.open(path, 'rb').read()
    n = struct.unpack('<I', raw[4:8])[0]
    o = 8
    cols = {}
    for name, dtype, size in (('cls', 'u1', 1), ('birth', '<i4', 4), ('death', '<i4', 4), ('bface', '<i4', 4),
                              ('lface', '<i4', 4), ('hops', '<i4', 4), ('partner', '<i4', 4), ('p1', '<i4', 4),
                              ('p2', '<i4', 4), ('cause', 'u1', 1), ('c1', '<i4', 4), ('c2', '<i4', 4)):
        cols[name] = np.frombuffer(raw[o:o + size * n], dtype)
        o += size * n
    return cols


def summarize(run_dir: Path) -> dict:
    meta = json.loads((run_dir / 'run.json').read_text())
    c = load_catalogue(run_dir / 'catalogue.bin.gz')
    level = meta['geometry']['level']
    ticks = meta['config']['ticks']
    end = np.where(c['death'] < 0, ticks, c['death'])
    life = end - c['birth']
    alive = c['death'] < 0
    federation = bool(meta['geometry'].get('coords'))
    if federation:
        raw = gzip.open(run_dir / 'catalogue_xyz.bin.gz', 'rb').read()
        n = struct.unpack('<I', raw[4:8])[0]
        pb = np.frombuffer(raw[8:8 + 12 * n], '<f4').reshape(n, 3).astype(np.float64)
        pl = np.frombuffer(raw[8 + 12 * n:8 + 24 * n], '<f4').reshape(n, 3).astype(np.float64)
        counts = [[a + b for a, b in zip(x, y)] for x, y in zip(meta['stats']['counts_intra'], meta['stats']['counts_macro'])]
        cells = meta['geometry']['screen_vertices']
    else:
        pb = face_positions(c['bface'], level)
        pl = face_positions(c['lface'], level)
        counts = meta['stats']['counts']
        cells = meta['geometry']['faces']
    arc = np.degrees(np.arccos(np.clip((pb * pl).sum(1), -1, 1)))
    mean_face_arc = 360.0 / (2 * np.pi * np.sqrt(cells / (4 * np.pi)))  # degrees per cell width
    out = {'run': str(run_dir), 'ticks': ticks, 'identities': int(len(life)), 'alive_at_end': int(alive.sum()),
           'final_counts_by_class': {str(k): int(v) for k, v in enumerate(counts[-1]) if k},
           'peak_defects': int(max(sum(x[1:]) for x in counts)),
           'births_by_mechanism': {'conjugate_pair': int(((c['p1'] < 0)).sum()),
                                   'split_child': int(((c['p1'] >= 0) & (c['p2'] < 0)).sum()),
                                   'fusion_or_exchange': int((c['p2'] >= 0).sum())},
           'deaths_by_cause': {CAUSE[k]: int(v) for k, v in enumerate(np.bincount(c['cause'], minlength=5))},
           'families': {}}
    if federation:
        intra = c['bface'] < meta['geometry']['intra_faces']
        out['tiers'] = {}
        for name, m in (('internal', intra), ('screen', ~intra)):
            out['tiers'][name] = {'ever': int(m.sum()), 'alive_at_end': int((m & alive).sum()),
                                  'lifetime_median': float(np.median(life[m])) if m.any() else 0.0,
                                  'hops_per_tick_mean': float((c['hops'][m] / np.maximum(1, life[m])).mean()) if m.any() else 0.0,
                                  'final_by_class': {str(k): int(v) for k, v in enumerate(meta['stats']['counts_intra' if name == 'internal' else 'counts_macro'][-1]) if k}}
    for fam in ('A', 'B', 'C'):
        m = np.isin(c['cls'], [k for k, f in FAMILY.items() if f == fam])
        if not m.any():
            out['families'][fam] = {'ever': 0}
            continue
        hops_rate = c['hops'][m] / np.maximum(1, life[m])
        out['families'][fam] = {
            'ever': int(m.sum()), 'alive_at_end': int((m & alive).sum()),
            'lifetime_median': float(np.median(life[m])), 'lifetime_mean': float(life[m].mean()),
            'lifetime_max': int(life[m].max()),
            'hops_per_tick_mean': float(hops_rate.mean()),
            'net_displacement_deg_median': float(np.median(arc[m])),
            'net_displacement_faces_median': float(np.median(arc[m]) / mean_face_arc),
        }
    # did annihilating partners die with their birth partner?
    ann = c['cause'] == 1
    if ann.any():
        birth_partner = np.full(len(c['cls']), -1, np.int64)
        sib = c['p1'] < 0
        birth_partner[sib] = c['partner'][sib]
        out['annihilation_with_birth_partner_fraction'] = float(
            (birth_partner[ann] == c['partner'][ann]).mean())
    top = np.argsort(-life)[:10]
    out['longest_lived'] = [{'id': int(i), 'class': int(c['cls'][i]), 'family': FAMILY[int(c['cls'][i])],
                             'lifetime': int(life[i]), 'hops': int(c['hops'][i]), 'alive': bool(alive[i])} for i in top]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--json', type=Path)
    a = p.parse_args(argv)
    out = summarize(a.run)
    text = json.dumps(out, indent=2)
    if a.json:
        a.json.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
