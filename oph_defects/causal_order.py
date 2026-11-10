"""Ordering fraction of the provenance order of a native history.

    python -m oph_defects.causal_order --level 5 --durations 8,16,24,32 --centres 6

Events are the attempts of one exact run of law C on a balanced background. Two conventions:
  all      every attempt is an event; an attempt precedes the next attempt on each of its two ports
  moves    only attempts with a nonzero current are events, linked through shared ports in the same way
The order is the transitive closure. For two related events p < q the interval is the set of events between them.
Its ordering fraction is the number of related pairs over the number of pairs. For a sprinkling of d-dimensional
Minkowski space the fraction is Gamma(d + 1) Gamma(d / 2) / (2 Gamma(3 d / 2)): 1/2, 8/35, 1/10 for d = 2, 3, 4
(Myrheim, Meyer). p and q are attempts on one carrier, `duration` sweeps apart.
"""
from __future__ import annotations

import argparse
import json
from math import gamma
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from . import fedgeom as FG
from .universe import COSTS, Engine, prepare_loads, schedule_seed


def fraction_of(d: float) -> float:
    return gamma(d + 1) * gamma(d / 2) / (2 * gamma(3 * d / 2))


def dimension_of(f: float) -> float:
    return float(brentq(lambda d: fraction_of(d) - f, 1.01, 12.0)) if fraction_of(12.0) < f < fraction_of(1.01) else float('nan')


def interval_stats(sa, sb, moved, t_p, t_q, moves_only: bool):
    """Events between attempt t_p and attempt t_q of the filtered history (ports sa[t], sb[t]); returns (n, relations)."""
    n_att = sa.size
    use = moved if moves_only else np.ones(n_att, bool)
    fut = np.zeros(n_att, bool)
    mark = {}
    mark[int(sa[t_p])] = True; mark[int(sb[t_p])] = True
    fut[t_p] = True
    for t in range(t_p + 1, t_q + 1):
        if not use[t]:
            continue
        a, b = int(sa[t]), int(sb[t])
        if a in mark or b in mark:
            fut[t] = True; mark[a] = True; mark[b] = True
    if not fut[t_q]:
        return 0, 0
    past = np.zeros(n_att, bool)
    mark = {int(sa[t_q]): True, int(sb[t_q]): True}
    past[t_q] = True
    for t in range(t_q - 1, t_p - 1, -1):
        if not use[t]:
            continue
        a, b = int(sa[t]), int(sb[t])
        if a in mark or b in mark:
            past[t] = True; mark[a] = True; mark[b] = True
    idx = np.flatnonzero(fut & past)
    n = idx.size
    last = {}                                   # port -> (bitset of that event and its predecessors)
    rel = 0
    for i, t in enumerate(idx):
        a, b = int(sa[t]), int(sb[t])
        pred = last.get(a, 0) | last.get(b, 0)
        rel += pred.bit_count()
        me = pred | (1 << i)
        last[a] = me; last[b] = me
    return n, rel


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--level', type=int, default=5)
    p.add_argument('--durations', default='8,16,24')
    p.add_argument('--centres', type=int, default=6)
    p.add_argument('--json', type=Path, default=None)
    a = p.parse_args(argv)
    geo = FG.build(a.level)
    durations = [int(v) for v in a.durations.split(',')]
    W = max(durations) + 3
    sa_all, sb_all = geo.all_seam_ends()
    sa_all = np.asarray(sa_all, np.int64); sb_all = np.asarray(sb_all, np.int64)
    X = geo.carrier_xyz / np.linalg.norm(geo.carrier_xyz, axis=1, keepdims=True)
    R = np.sqrt(geo.vertices * 3 ** 0.5 / 2 / (4 * np.pi))
    rng = np.random.default_rng(11)
    centres = [4 ** a.level - 1] + [int(v) for v in rng.choice(geo.carriers, size=a.centres - 1, replace=False)]
    reach = 0.5 * 0.37 * max(durations) + 2.5                      # the double cone stays inside this radius
    seam_patch = np.full(geo.seams, -1, np.int16)
    for ci, c in enumerate(centres):
        near = R * np.arccos(np.clip(X @ X[c], -1, 1)) <= reach
        m = near[sa_all // 12] & near[sb_all // 12]
        seam_patch[m & (seam_patch < 0)] = ci
    loads, _ = prepare_loads('canonical:1/3:3', a.level, geo.ports)
    eng = Engine(geo, loads, schedule_seed(a.level, 4242), COSTS['binding'], 5, False, slots=2, register=False, fast_rng=True)
    hist = [[] for _ in centres]
    for s in range(W):
        seq, coin = eng.draw()
        for lo in range(0, eng.S, eng.chunk):
            sq = seq[lo:lo + eng.chunk]
            n = eng.run_chunk(sq, coin[lo:lo + eng.chunk], lo, s)
            pc = seam_patch[sq]
            hit = np.flatnonzero(pc >= 0)
            cur = eng.o_cur[:n][hit]
            for ci in np.unique(pc[hit]):
                sel = pc[hit] == ci
                hist[ci].append((np.full(int(sel.sum()), s, np.int32), sq[hit][sel].astype(np.int64), cur[sel] != 0))
    rows = []
    for ci, c in enumerate(centres):
        sw = np.concatenate([h[0] for h in hist[ci]]); seam = np.concatenate([h[1] for h in hist[ci]]); moved = np.concatenate([h[2] for h in hist[ci]])
        sa, sb = sa_all[seam], sb_all[seam]
        own = (sa // 12 == c) | (sb // 12 == c)
        for dur in durations:
            for conv in ('all', 'moves'):
                ok = own & (moved if conv == 'moves' else True)
                start = np.flatnonzero(ok & (sw >= 2))
                if not start.size:
                    continue
                t_p = int(start[0])
                cand = np.flatnonzero(ok & (sw < sw[t_p] + dur))
                best = (0, 0, -1)
                for t_q in cand[::-1][:12]:                        # the latest own event that lies in the future of p
                    n, rel = interval_stats(sa, sb, moved, t_p, int(t_q), conv == 'moves')
                    if n > 2:
                        best = (n, rel, int(t_q)); break
                n, rel, t_q = best
                if n > 2:
                    f = rel / (n * (n - 1) / 2)
                    rows.append({'centre': int(c), 'duration_sweeps': dur, 'events': conv, 'interval_events': n, 'relations': rel,
                                 'ordering_fraction': f, 'myrheim_meyer_dimension': dimension_of(f)})
    print(f"level {a.level}: provenance order of one native history on a balanced background (filling 1/3); reference fractions 1/2, 8/35 = 0.2286, 1/10 for d = 2, 3, 4")
    print(f"{'duration':>9}{'events':>8}{'intervals':>10}{'mean size':>11}{'ordering fraction':>19}{'+-':>8}{'dimension':>11}")
    summary = []
    for dur in durations:
        for conv in ('all', 'moves'):
            sub = [r for r in rows if r['duration_sweeps'] == dur and r['events'] == conv]
            if not sub:
                continue
            f = np.array([r['ordering_fraction'] for r in sub])
            item = {'duration_sweeps': dur, 'events': conv, 'intervals': len(sub), 'mean_interval_events': float(np.mean([r['interval_events'] for r in sub])),
                    'ordering_fraction': float(f.mean()), 'standard_error': float(f.std(ddof=1) / np.sqrt(f.size)) if f.size > 1 else None,
                    'myrheim_meyer_dimension': dimension_of(float(f.mean()))}
            summary.append(item)
            print(f"{dur:>9}{conv:>8}{item['intervals']:>10}{item['mean_interval_events']:>11.0f}{item['ordering_fraction']:>19.4f}"
                  f"{(item['standard_error'] or 0):>8.4f}{item['myrheim_meyer_dimension']:>11.3f}", flush=True)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({'level': a.level, 'summary': summary, 'rows': rows}))


if __name__ == '__main__':
    main()
