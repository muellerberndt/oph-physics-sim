"""Reader, exact replay and dataset card for universe runs (layout in DATASET.md).

    from oph_defects.dataset import Dataset
    ds = Dataset('runs/universe_L5')
    ds.meta['totals']                      # totals, band, attempts to the balanced class
    cut = ds.cut(12)                       # one clock-synchronised slice: defects, loads, clocks
    cat = ds.catalogue()                   # one row per defect identity
    rec = ds.records(3)                    # every attempt of sweep 3 (stored window or exact replay)
    obs = ds.observer(ds.meta['observers'][0]['carrier'])   # tape, patch, readouts of one observer
    for sweep, rec in ds.replay(40, 45):   # exact replay from the nearest checkpoint
        ...

Every array file is a NumPy ``.npz`` archive; every table is JSON.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import fedgeom as FG


class Dataset:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / 'run.json').read_text())
        self.level = self.meta['geometry']['level']
        self._geo = None

    # ------------------------------------------------------------ static structure
    @property
    def geometry(self) -> FG.Geometry:
        """The federation (seams, faces, positions), rebuilt deterministically from the pinned sources."""
        if self._geo is None:
            self._geo = FG.build(self.level)
        return self._geo

    # ------------------------------------------------------------ files
    def cuts(self) -> list[dict]:
        return self.meta['cuts']

    def cut(self, k: int | None) -> dict:
        """Arrays of consistent cut ``k`` (``None`` or -1: the end of the run)."""
        name = 'cuts/end.npz' if k in (None, -1) else next(c['file'] for c in self.meta['cuts'] if c['k'] == k)
        with np.load(self.path / name) as z:
            return {key: z[key] for key in z.files}

    def catalogue(self) -> dict:
        with np.load(self.path / 'catalogue.npz') as z:
            return {key: z[key] for key in z.files}

    def checkpoint(self, sweep: int) -> dict:
        entry = next(c for c in self.meta['checkpoints'] if c['sweep'] == sweep)
        with np.load(self.path / entry['file']) as z:
            out = {key: z[key] for key in z.files}
        out['meta'] = json.loads((self.path / entry['file']).with_suffix('.json').read_text())
        return out

    def stats(self) -> dict:
        """Per-sweep tables (one row per sweep of the solver schedule)."""
        with np.load(self.path / self.meta['stats_file']) as z:
            return {key: z[key] for key in z.files}

    def series(self):
        """Yield the density-snapshot blocks: sweep, cells, event counts, optional ports and level-5 sums."""
        for block in self.meta.get('series', []):
            with np.load(self.path / block['file']) as z:
                yield {key: z[key] for key in z.files}

    def observer(self, carrier: int) -> dict:
        with np.load(self.path / f'observers/obs_{int(carrier)}.npz') as z:
            return {key: z[key] for key in z.files}

    def records(self, sweep: int) -> dict:
        """Every attempt of one sweep in schedule order: seam, coin, current, contrast, twist, energy, depth."""
        for r in self.meta['records']:
            if r['sweep'] == sweep:
                with np.load(self.path / r['file']) as z:
                    return {key: z[key] for key in z.files}
        return next(rec for _, rec in self.replay(sweep, sweep + 1))

    # ------------------------------------------------------------ exact replay
    def engine_at(self, sweep: int):
        """An engine restored at the latest checkpoint not after ``sweep`` and advanced to the start of ``sweep``."""
        from .universe import COSTS, Engine
        cfg = self.meta['config']
        base = max(c['sweep'] for c in self.meta['checkpoints'] if c['sweep'] <= sweep)
        ck = self.checkpoint(base)
        geo = self.geometry
        tw = bool(cfg.get('twists'))
        eng = Engine(geo, ck['loads'], cfg['seed'], COSTS[cfg['cost']], cfg['kappa'], cfg['track_intra'],
                     backend='sequential', slots=max(2, len(ck['open_cut_index']) + 1), register=cfg.get('register', True) and not tw,
                     offsets=ck['offsets'].astype(np.int8) if tw else None, twists=tw)
        eng.restore(ck, ck['meta']['rng'])
        for s in range(base, sweep):
            self._advance(eng, s)
        return eng

    @staticmethod
    def _advance(eng, s, keep=False):
        seq, coin = eng.draw()
        out = {k: [] for k in ('current', 'contrast', 'total', 'twist', 'energy', 'depth')} if keep else None
        for lo in range(0, eng.S, eng.chunk):
            hi = min(eng.S, lo + eng.chunk)
            n = eng.run_chunk(seq[lo:hi], coin[lo:hi], lo, s)
            if keep:
                out['current'].append(eng.o_cur[:n].copy()); out['contrast'].append(eng.o_con[:n].copy())
                out['total'].append(eng.o_tot[:n].copy())
                out['twist'].append(eng.o_twist[:n].copy()); out['energy'].append(eng.o_de[:n].copy())
                out['depth'].append(eng.o_dep[:n].copy())
        if keep:
            rec = {k: np.concatenate(v) for k, v in out.items()}
            rec.update(seam=seq, coin=coin)
            return rec

    def replay(self, start: int, stop: int):
        """Yield ``(sweep, records)`` for sweeps ``start..stop-1``; ``records`` also holds the state after the sweep."""
        eng = self.engine_at(start)
        for s in range(start, stop):
            rec = self._advance(eng, s, keep=True)
            rec.update(loads_after=eng.x.copy(), register_after=eng.reg.copy(), classes_after=eng.cls.copy(),
                       depth_after=eng.dsyn.copy(), ticks_after=eng.ticks.copy())
            yield s, rec

    def observer_tape(self, carrier: int, sweeps) -> dict:
        """Records on the 33 seams of any carrier for the given sweeps, in schedule order (stored or replayed)."""
        geo = self.geometry
        own = geo.carrier_incident_seams(int(carrier))
        mask = np.zeros(geo.seams, bool)
        mask[own] = True
        slot_of = {int(s): i for i, s in enumerate(own.tolist())}
        cols = {k: [] for k in ('sweep', 'pos', 'seam', 'slot', 'end', 'coin', 'current', 'twist', 'depth')}
        for s in sweeps:
            rec = self.records(s)
            hit = np.flatnonzero(mask[rec['seam']])
            seam = rec['seam'][hit].astype(np.int64)
            cols['sweep'].append(np.full(hit.size, s)); cols['pos'].append(hit); cols['seam'].append(seam)
            cols['slot'].append(np.array([slot_of[int(v)] for v in seam], np.int64))
            g = np.maximum(seam - geo.intra_seams, 0)
            cols['end'].append(np.where(seam >= geo.intra_seams, (geo.glue_b[g] // 12 == carrier).astype(np.int64), 0))
            for k in ('coin', 'current', 'twist', 'depth'):
                cols[k].append(rec[k][hit])
        return {k: np.concatenate(v) if v else np.zeros(0, np.int64) for k, v in cols.items()}


# ---------------------------------------------------------------------------- viewer files
FAMILY = {0: None, 1: 'A', 5: 'A', 2: 'B', 4: 'B', 3: 'C'}


def class_statistics(cat: dict, meta: dict) -> dict:
    """Per class and tier: how many identities, how long they live (in causal depth), how they move, how they end,
    and the reaction channels read off the parent and child links."""
    cls = cat['cls'].astype(np.int64)
    tier = cat['tier']
    end_depth = int(meta['stats_end_depth'])
    alive = cat['death_sweep'] < 0
    life = np.where(alive, end_depth, cat['death_depth']) - cat['birth_depth']
    out = {'tiers': {}, 'depth_at_end': end_depth}
    for tname, tval in (('screen', 1), ('internal', 0)):
        m_t = tier == tval
        if not m_t.any():
            continue
        rows = {}
        for k in range(1, 6):
            m = m_t & (cls == k)
            n = int(m.sum())
            if not n:
                rows[str(k)] = {'ever': 0}
                continue
            lf = life[m]
            hops = cat['hops'][m]
            born_pair = int(((cat['parent1'][m] < 0)).sum())
            rows[str(k)] = {
                'ever': n, 'alive_at_end': int((m & alive).sum()), 'lifetime_depth_median': float(np.median(lf)),
                'lifetime_depth_mean': float(lf.mean()), 'lifetime_depth_max': int(lf.max()),
                'hops_mean': float(hops.mean()), 'hops_per_depth': float(hops.sum() / max(1, lf.sum())),
                'moved_fraction': float((hops > 0).mean()),
                'born': {'conjugate_pair': born_pair, 'split_product': int(((cat['parent1'][m] >= 0) & (cat['parent2'][m] < 0)).sum()),
                         'fusion_or_exchange_product': int((cat['parent2'][m] >= 0).sum())},
                'ended': {name: int((cat['cause'][m] == i).sum()) for i, name in enumerate(('alive', 'annihilation', 'fusion', 'split', 'exchange'))}}
        # reaction channels
        chan = {}
        def bump(key):
            chan[key] = chan.get(key, 0) + 1
        ids = np.flatnonzero(m_t)
        p1, p2, c1, c2, cause, partner = (cat[k][ids] for k in ('parent1', 'parent2', 'child1', 'child2', 'cause', 'partner'))
        kk = cls[ids]
        # pair births: identities without parents come in conjugate pairs (the partner column is overwritten at death)
        pb = p1 < 0
        for a in np.unique(np.minimum(kk[pb], 6 - kk[pb])):
            n = int((pb & (np.minimum(kk, 6 - kk) == a)).sum()) // 2
            chan[f'0 -> {a} + {6 - a}'] = n
        fus = (cause == 2) & (c1 >= 0)
        if fus.any():
            a, b, c = kk[fus], cls[partner[fus]], cls[c1[fus]]
            keys = np.stack([np.minimum(a, b), np.maximum(a, b), c], axis=1)
            u, cnt = np.unique(keys, axis=0, return_counts=True)
            for (x, y, z), n in zip(u.tolist(), cnt.tolist()):
                chan[f'{x} + {y} -> {z}'] = n // 2
        ann = cause == 1
        if ann.any():
            a = np.minimum(kk[ann], 6 - kk[ann])
            u, cnt = np.unique(a, return_counts=True)
            for x, n in zip(u.tolist(), cnt.tolist()):
                chan[f'{x} + {6 - x} -> 0'] = n // 2
        spl = (cause == 3) & (c1 >= 0) & (c2 >= 0)
        if spl.any():
            a, b, c = kk[spl], cls[c1[spl]], cls[c2[spl]]
            keys = np.stack([a, np.minimum(b, c), np.maximum(b, c)], axis=1)
            u, cnt = np.unique(keys, axis=0, return_counts=True)
            for (x, y, z), n in zip(u.tolist(), cnt.tolist()):
                chan[f'{x} -> {y} + {z}'] = n
        exc = (cause == 4) & (c1 >= 0) & (c2 >= 0)
        if exc.any():
            a, b = kk[exc], cls[partner[exc]]
            c, d = cls[c1[exc]], cls[c2[exc]]
            keys = np.stack([np.minimum(a, b), np.maximum(a, b), np.minimum(c, d), np.maximum(c, d)], axis=1)
            u, cnt = np.unique(keys, axis=0, return_counts=True)
            for (w, x, y, z), n in zip(u.tolist(), cnt.tolist()):
                chan[f'{w} + {x} -> {y} + {z}'] = n // 2
        out['tiers'][tname] = {'classes': rows, 'channels': dict(sorted(chan.items(), key=lambda kv: -kv[1]))}
    return out


def build_view(path) -> dict:
    """Files the viewer reads: display mesh, class statistics, and the catalogue rows of every identity that
    appears on a stored slice (with partners, parents and children)."""
    path = Path(path)
    meta = json.loads((path / 'run.json').read_text())
    view = path / 'view'
    view.mkdir(exist_ok=True)
    level = meta['geometry']['level']
    dl = meta['geometry']['display_level']
    if not (view / 'display.npz').exists():
        geo = FG.build(dl)
        np.savez_compressed(view / 'display.npz', tri=geo.tri, mesh_xyz=geo.mesh_xyz.astype(np.float32),
                            cell_xyz=geo.carrier_xyz.astype(np.float32), level=np.int64(dl))
    with np.load(path / 'catalogue.npz') as z:
        cat = {k: z[k] for k in z.files}
    with np.load(path / meta['stats_file']) as z:
        end_depth = int(z['depth_max'][-1])
    stats = class_statistics(cat, dict(meta, stats_end_depth=end_depth))
    (view / 'class_stats.json').write_text(json.dumps(stats))
    n = int(cat['cls'].shape[0])
    keep = np.zeros(n, bool)
    # screen defects near each featured observer on every cut, so the observer view needs no cut file
    obs = []
    for ob in meta['observers']:
        with np.load(path / ob['file']) as z:
            lim = np.cos(min(1.5, (float(np.linalg.norm(z['patch_cell_xyz'] - z['xyz'], axis=1).max()) + meta['geometry']['cell_spacing'])))
            obs.append({'carrier': ob['carrier'], 'xyz': z['xyz'].astype(np.float64), 'lim': lim, 'cut': [], 'cls': [], 'id': [], 'xyz_d': []})
    for index, entry in enumerate(meta['cuts'] + [meta['end']]):
        with np.load(path / entry['file']) as z:
            for key in ('screen_id', 'intra_id'):
                if key in z.files:
                    keep[z[key]] = True
            if obs and index < len(meta['cuts']):
                xyz = z['screen_xyz'].astype(np.float64)
                cls = z['screen_charge'] if 'screen_charge' in z.files else z['screen_cls']
                for o in obs:
                    hit = np.flatnonzero(xyz @ o['xyz'] >= o['lim'])
                    if hit.size:
                        o['cut'].append(np.full(hit.size, index, np.int32)); o['cls'].append(cls[hit].astype(np.int8))
                        o['id'].append(z['screen_id'][hit]); o['xyz_d'].append(z['screen_xyz'][hit])
    if obs:
        arrays = {}
        for o in obs:
            c = o['carrier']
            arrays[f'o{c}_cut'] = np.concatenate(o['cut']) if o['cut'] else np.zeros(0, np.int32)
            arrays[f'o{c}_cls'] = np.concatenate(o['cls']) if o['cls'] else np.zeros(0, np.int8)
            arrays[f'o{c}_id'] = np.concatenate(o['id']) if o['id'] else np.zeros(0, np.int32)
            arrays[f'o{c}_xyz'] = np.concatenate(o['xyz_d']) if o['xyz_d'] else np.zeros((0, 3), np.float32)
        np.savez_compressed(view / 'observer_defects.npz', **arrays)
    base = np.flatnonzero(keep)
    for col in ('partner', 'parent1', 'parent2', 'child1', 'child2'):
        link = cat[col][base]
        keep[link[link >= 0]] = True
    ids = np.flatnonzero(keep).astype(np.int32)
    sub = {k: v[ids] for k, v in cat.items()}
    sub['id'] = ids
    np.savez_compressed(view / 'catalogue.npz', **sub)
    info = {'identities_total': n, 'identities_in_view': int(ids.size), 'display_level': dl, 'level': level}
    (view / 'view.json').write_text(json.dumps(info))
    return info


def lite(path, out, keep_one_in: int = 1) -> dict:
    """A copy of a run for the viewer without records, checkpoints and series. With ``keep_one_in`` = k > 1 the
    stored slices keep the defects whose identity number is a multiple of k (a fixed 1-in-k sample of identities,
    so a kept defect stays visible along its whole life); all counts and statistics remain those of the full run
    and ``display_sample`` in run.json records k."""
    import shutil
    path, out = Path(path), Path(out)
    meta = json.loads((path / 'run.json').read_text())
    if not (path / 'view' / 'view.json').is_file():
        build_view(path)
    (out / 'cuts').mkdir(parents=True, exist_ok=True); (out / 'view').mkdir(exist_ok=True)
    k = int(keep_one_in)
    for entry in meta['cuts'] + [meta['end']]:
        with np.load(path / entry['file']) as z:
            arrays = {key: z[key] for key in z.files}
        if k > 1:
            for tier in ('screen', 'intra'):
                if f'{tier}_id' in arrays:
                    m = arrays[f'{tier}_id'] % k == 0
                    for key in list(arrays):
                        if key.startswith(tier + '_') and arrays[key].shape[:1] == m.shape:
                            arrays[key] = arrays[key][m]
        np.savez_compressed(out / entry['file'], **arrays)
    with np.load(path / 'view' / 'catalogue.npz') as z:
        cat = {key: z[key] for key in z.files}
    if k > 1:
        ids = cat['id'].astype(np.int64)
        keep = ids % k == 0
        cat = {key: v[keep] for key, v in cat.items()}             # partners and parents outside the sample have no card
    np.savez_compressed(out / 'view' / 'catalogue.npz', **cat)
    for name in ('display.npz', 'class_stats.json', 'observer_defects.npz'):
        if (path / 'view' / name).is_file():
            shutil.copy2(path / 'view' / name, out / 'view' / name)
    info = json.loads((path / 'view' / 'view.json').read_text())
    info.update(identities_in_view=int(cat['id'].size), display_sample=k)
    (out / 'view' / 'view.json').write_text(json.dumps(info))
    for name in (meta['stats_file'], 'DATASET.md', 'particle_map.json', 'README.md', 'MANIFEST.sha256'):
        if (path / name).is_file():
            shutil.copy2(path / name, out / ('SOURCE_' + name if name in ('README.md', 'MANIFEST.sha256') else name))
    if (path / 'observers').is_dir():
        shutil.copytree(path / 'observers', out / 'observers', dirs_exist_ok=True)
    meta['display_sample'] = k
    meta['viewer_copy'] = {'of': str(path), 'keep_one_in': k, 'left_out': ['records', 'checkpoints', 'series', 'final loads'],
                           'note': 'counts and statistics are those of the full run; slices show a fixed 1-in-k sample of identities' if k > 1 else 'slices are complete'}
    (out / 'run.json').write_text(json.dumps(meta))
    return info


# ---------------------------------------------------------------------------- dataset card
def _size(n: float) -> str:
    for unit in ('B', 'kB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f'{n:.0f} {unit}' if unit == 'B' else f'{n:.1f} {unit}'
        n /= 1024


def write_card(out: Path, meta: dict):
    """README.md and MANIFEST.sha256 of a finished run, and a copy of the schema document."""
    out = Path(out)
    doc = Path(__file__).with_name('DATASET.md')
    if doc.exists():
        (out / 'DATASET.md').write_text(doc.read_text())
    pm = Path(__file__).with_name('particle_map.json')
    if pm.exists():
        (out / 'particle_map.json').write_text(pm.read_text())
    g, c, t = meta['geometry'], meta['config'], meta['totals']
    st = meta['stats']
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name not in ('README.md', 'MANIFEST.sha256'))
    total = sum(p.stat().st_size for p in files)
    groups = {}
    for p in files:
        key = p.relative_to(out).parts[0] if len(p.relative_to(out).parts) > 1 else p.name
        n, b = groups.get(key, (0, 0))
        groups[key] = (n + 1, b + p.stat().st_size)
    pins = meta['provenance']['source_pins']
    if c.get('twists'):
        intro = [
            f'# OPH closed-loop run (declared extension), level {g["level"]}: {g["carriers"]:,} observers',
            '',
            'NOT the exact layer. This run executes law C with transported contrast and unit twists on the exact',
            'federation: every seam carries an integer offset and compares one end with the transported other end,',
            'and an attempt on a glue seam may move the offset by one unit instead of moving load. With all offsets',
            'zero and twists off the rule is law C. A screen plaquette whose offsets have nonzero holonomy is a',
            'charge; it acts back on the loads. The loads of this run are therefore not those of plain law C.',
            '`DATASET.md` in this directory defines every file and field.',
            '']
    else:
        intro = [
            f'# OPH universe run, level {g["level"]}: {g["carriers"]:,} observers',
            '',
            'One execution of the OPH executable model at federation scale: law C (integer nearest agreement with a',
            'fair tie coin) on the exact federation of twelve-port carriers, from the declared preparation to the',
            'balanced class and beyond. The loads and the per-attempt records are the executable specification with',
            'nothing added. Depths, clocks, cuts and the Z6 register are declared layers computed from those records',
            'without further randomness. `DATASET.md` in this directory defines every file and field.',
            '']
    lines = intro + [
        '| Quantity | Value |',
        '| --- | --- |',
        f'| Carriers (observers) | {g["carriers"]:,} |',
        f'| Ports | {g["ports"]:,} |',
        f'| Seams | {g["seams"]:,} ({g["glue_seams"]:,} between carriers) |',
        f'| Faces | {g["faces"]:,} ({g["screen_plaquettes"]:,} screen plaquettes) |',
        f'| Preparation | {meta["preparation"]["name"]}: {meta["preparation"]["rule"]} |',
        f'| Schedule seed | {c["seed"]} (schedule {c["schedule"]}) |',
        f'| Sweeps | {meta["sweeps"]} ({meta["sweeps"] * g["seams"]:,} attempts) |',
        f'| Total load | {t["total_load"]:,}, balanced band {t["band"][0]} and {t["band"][1]} |',
        f'| Descent functional V | {t["V_initial"]:,} at the start, {t["V_final"]:,} at the end, minimum {t["V_minimum"]:,} |',
        f'| Balanced class | {"not defined for this law" if c.get("twists") else "reached after " + format(t["attempts_to_balanced_class"], ",") + " attempts" if t["balanced"] else "not reached in this run"} |',
        f'| Causal depth at the end | {st["depth_min"][-1]} to {st["depth_max"][-1]} |',
        f'| Consistent cuts | {len(meta["cuts"])}, depths {meta["cuts"][0]["depth"] if meta["cuts"] else "-"} to {meta["cuts"][-1]["depth"] if meta["cuts"] else "-"} |',
        f'| Register | cost {c["cost_table"]}, kappa {c["kappa"]}, internal tier tracked: {c["track_intra"]} |',
        f'| Defect identities | {meta["identities"]:,} |',
        f'| Featured observers | {len(meta["observers"])} |',
        f'| Backend | {c["backend"]} |',
        f'| Size | {_size(total)} in {len(files)} files |',
        '',
        '## Source custody',
        '',
        'The federation is built by the functions the executable model pins. Hashes of the local sources against',
        'the literal pins in `reverse-engineering-reality/model/federated_observer/source.py`:',
        '',
        '| Source | Defines this run | Equal to the model pin |',
        '| --- | --- | --- |',
    ]
    for rel, p in pins.items():
        lines.append(f'| `{rel}` | {"yes" if p["defines_the_run"] else "reference check only"} | {"yes" if p["equal"] else "no"} |')
    lines += ['', f'Kernel attempt body sha256 `{meta["provenance"]["kernel_core_sha256"]}`; numpy {meta["provenance"]["numpy"]}; '
              f'python {meta["provenance"]["python"]}; {meta["provenance"]["platform"]}.', '',
              '## Files', '', '| Path | Files | Size |', '| --- | --- | --- |']
    for key, (n, b) in sorted(groups.items()):
        lines.append(f'| `{key}` | {n} | {_size(b)} |')
    lines += ['', '## Reading it', '', '```python', 'from oph_defects.dataset import Dataset',
              "ds = Dataset('.')", 'cut = ds.cut(ds.cuts()[-1]["k"])      # the last clock-synchronised slice',
              'cat = ds.catalogue()                    # one row per defect identity',
              'obs = ds.observer(ds.meta["observers"][0]["carrier"])', 'rec = ds.records(0)                     # every attempt of sweep 0',
              '```', '', 'Without this package: every `.npz` is a plain NumPy archive and `run.json` is plain JSON.', '']
    (out / 'README.md').write_text('\n'.join(lines))
    manifest = []
    for p in files:
        h = hashlib.sha256()
        with open(p, 'rb') as fh:
            for block in iter(lambda: fh.read(1 << 22), b''):
                h.update(block)
        manifest.append(f'{h.hexdigest()}  {p.relative_to(out)}')
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')


def rewrite_card(path) -> None:
    """Regenerate README.md, MANIFEST.sha256 and the viewer files of a finished run directory."""
    path = Path(path)
    meta = json.loads((path / 'run.json').read_text())
    with np.load(path / meta['stats_file']) as z:
        meta['stats'] = {k: z[k].tolist() for k in ('depth_min', 'depth_max')}
    build_view(path)
    write_card(path, meta)


def verify_manifest(path) -> dict:
    """Recompute the sha256 of every file listed in MANIFEST.sha256."""
    path = Path(path)
    bad, n = [], 0
    for line in (path / 'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        h = hashlib.sha256()
        with open(path / name, 'rb') as fh:
            for block in iter(lambda: fh.read(1 << 22), b''):
                h.update(block)
        n += 1
        if h.hexdigest() != digest:
            bad.append(name)
    return {'files': n, 'mismatches': bad}


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description='Maintenance of universe run directories.')
    p.add_argument('command', choices=('card', 'view', 'verify', 'lite'))
    p.add_argument('run', type=Path)
    p.add_argument('out', type=Path, nargs='?', help='lite: target directory')
    p.add_argument('--keep-one-in', type=int, default=1, help='lite: show a fixed 1-in-k sample of identities on the slices')
    a = p.parse_args(argv)
    if a.command == 'lite':
        print(lite(a.run, a.out, a.keep_one_in))
        return
    if a.command == 'card':
        rewrite_card(a.run)
        print(f'{a.run}: card, manifest and viewer files rewritten')
    elif a.command == 'view':
        print(build_view(a.run))
    else:
        print(verify_manifest(a.run))


if __name__ == '__main__':
    main()
