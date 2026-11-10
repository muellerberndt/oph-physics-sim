"""Source-bound evidence for the ordinary Z6-register universe runs.

No particles, positions, missing identities or physical clocks are reconstructed.
The small JSON output is bundled with the viewer; no external research folder is needed. Full-run summaries
and the locally available (possibly sampled) catalogue are separate objects.

    python -m oph_defects.run_evidence runs/universe --out oph_defects/viewer2/run_evidence.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CAUSES = ('alive', 'annihilation', 'fusion', 'split', 'exchange')
PAIRS = ((1, 5), (2, 4), (3, 3))


def catalogue_census(cat: dict, *, complete: bool) -> dict:
    """Count supplied rows, explicitly separating seeded from event-born defects.

    A view catalogue cannot determine whole-run pair-event counts, even when its
    sampling factor is one: it need not contain every identity. Never multiply
    its counts by the display sample, or interpret global parent IDs as row IDs.
    This routine is for Z6 classes, not signed transported-offset holonomies.
    """
    cls = np.asarray(cat['cls'], dtype=np.int64)
    if np.any((cls < 1) | (cls > 5)):
        raise ValueError('Expected nonzero Z6 classes 1..5; signed-holonomy runs need a separate audit')
    seeded = np.asarray(cat['birth_sweep']) < 0
    parentless = np.asarray(cat['parent1']) < 0
    event_pair = parentless & ~seeded & (np.asarray(cat['birth_depth']) > 0)
    alive = np.asarray(cat['death_sweep']) < 0
    if np.any((cat['cause'] < 0) | (cat['cause'] >= len(CAUSES))):
        raise ValueError('Unsupported identity ending cause')
    if not np.array_equal(alive, np.asarray(cat['cause']) == 0):
        raise ValueError('Death sweep and cause disagree')
    out = {'scope': 'complete_catalogue' if complete else 'supplied_view_rows_only',
           'rows': int(cls.size), 'tiers': {}}
    for name, tier in (('screen', 1), ('internal', 0)):
        tm = np.asarray(cat['tier']) == tier
        if not tm.any():
            continue
        rows = {}
        for k in range(1, 6):
            m = tm & (cls == k)
            n = int(m.sum())
            rows[str(k)] = {
                'ever': n, 'alive_at_end': int((m & alive).sum()),
                'moved_identities': int(np.count_nonzero(np.asarray(cat['hops'])[m])),
                'total_hops': int(np.asarray(cat['hops'])[m].sum(dtype=np.int64)),
                'born': {'prepared': int((m & seeded).sum()),
                         'conjugate_pair': int((m & event_pair).sum()),
                         'split_product': int((m & (cat['parent1'] >= 0) & (cat['parent2'] < 0)).sum()),
                         'fusion_or_exchange_product': int((m & (cat['parent2'] >= 0)).sum())},
                'ended': {cause: int((m & (cat['cause'] == i)).sum()) for i, cause in enumerate(CAUSES)},
            }
            if sum(rows[str(k)]['born'].values()) != n:
                raise ValueError('Identity has an unsupported or inconsistent birth record')
        pairs = None
        if complete:
            pairs = {}
            for a, b in PAIRS:
                na, nb = (rows[str(k)]['born']['conjugate_pair'] for k in (a, b))
                if na != nb or (a == b and na % 2):
                    raise ValueError('Complete catalogue has unmatched conjugate pair births')
                pairs[f'0 -> {a} + {b}'] = na // 2 if a == b else na
        out['tiers'][name] = {'classes': rows, 'pair_birth_events': pairs}
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def audit_observer(cat: dict, meta: dict) -> dict:
    """Replay all own-load changes from the tape and the stored patch incidence."""
    carrier = int(cat['carrier'])
    own = int(np.flatnonzero(cat['patch_cells'] == carrier)[0])
    seams = {int(s): (int(a), int(b)) for a, b, s in cat['patch_seams']}
    x = cat['loads_initial'].astype(np.int64).copy()
    bad_rows = 0
    for i, (s, current) in enumerate(zip(cat['tape_seam'], cat['tape_current'])):
        a, b = seams[int(s)]
        if a // 12 == own:
            x[a % 12] -= int(current)
        if b // 12 == own:
            x[b % 12] += int(current)
        bad_rows += int(not np.array_equal(x, cat['loads_by_tick'][i + 1]))
    cut_errors = sum(int(int(tick) != (len(cat['tape_depth']) if depth < 0 else int(np.count_nonzero(cat['tape_depth'] <= depth))))
                     for depth, tick in zip(cat['cut_depth'], cat['cut_ticks']))
    frame = cat['frame'].astype(float)
    return {
        'carrier': carrier, 'incident_ticks': int(len(cat['tape_current'])),
        'hold_ticks': int(np.count_nonzero(cat['tape_current'] == 0)),
        'nonzero_current_ticks': int(np.count_nonzero(cat['tape_current'])),
        'load_rows_replayed': int(len(cat['tape_current'])), 'load_row_mismatches': bad_rows,
        'final_loads_match': bool(np.array_equal(x, cat['loads_final'])),
        'cut_tick_mismatches': cut_errors,
        'frame_orthogonality_error': float(np.max(np.abs(frame.T @ frame - np.eye(3)))),
        'frame_determinant': float(np.linalg.det(frame)),
        'physical_seconds_per_incident_tick': None,
        'clock_scope': 'Every incident seam attempt, including holds; distinct from causal depth and sweep.',
    }


def passive_register_control() -> dict:
    """Execute identical preparation/schedule with and without the passive register."""
    from . import fedgeom
    from .universe import COSTS, Engine, prepare_loads, schedule_seed
    level, sweeps = 1, 6
    geo = fedgeom.build(level)
    loads, _ = prepare_loads('exact06', level, geo.ports)
    engines = [Engine(geo, loads, schedule_seed(level, 0), COSTS['binding'], 5, True, register=r)
               for r in (True, False)]
    identical = True
    for sweep in range(sweeps):
        draws = [e.draw() for e in engines]
        identical &= all(np.array_equal(a, b) for a, b in zip(*draws))
        for lo in range(0, geo.seams, engines[0].chunk):
            n = None
            for e, (seq, coin) in zip(engines, draws):
                n = e.run_chunk(seq[lo:lo + e.chunk], coin[lo:lo + e.chunk], lo, sweep)
            for key in ('o_cur', 'o_con', 'o_tot', 'o_dep'):
                identical &= np.array_equal(getattr(engines[0], key)[:n], getattr(engines[1], key)[:n])
            identical &= np.array_equal(engines[0].x, engines[1].x)
    return {'level': level, 'sweeps': sweeps, 'attempts': sweeps * geo.seams,
            'same_preparation_schedule_and_coins': True,
            'all_load_current_contrast_total_depth_records_equal': bool(identical),
            'register_on_identity_count': engines[0].tracker.n,
            'register_off_identity_count': engines[1].tracker.n,
            'scope': 'Constructive finite control; source kernels.py shows load/current update independent of register.'}


def audit_run(path: Path, repository: Path) -> dict:
    path = path.resolve()
    sources = []

    def source(rel: str) -> Path:
        p = path / rel
        sources.append({'file': str(p.relative_to(repository)) if p.is_relative_to(repository) else 'external-run/'+path.name+'/'+rel, 'url': f'/runs/{path.name}/{rel}',
                        'sha256': sha256(p), 'bytes': p.stat().st_size})
        return p

    meta = json.loads(source('run.json').read_text())
    if meta['config'].get('twists') or not meta['config'].get('register', True):
        raise ValueError('This audit supports ordinary Z6-register runs only')
    quoted = json.loads(source('view/class_stats.json').read_text())
    stats = _npz(source(meta.get('stats_file', 'stats.npz')))
    full = (path / 'catalogue.npz').is_file()
    relcat = 'catalogue.npz' if full else 'view/catalogue.npz'
    cat = _npz(source(relcat))
    actual = catalogue_census(cat, complete=full)
    checks = []

    def check(name: str, ok: bool):
        checks.append({'check': name, 'passed': bool(ok)})

    for key, n in meta['tracker_totals'].items():
        check(f'stats.{key}.sum == tracker_totals.{key}', int(stats[key].sum(dtype=np.int64)) == n)
    check('stats identities final == metadata identities', int(stats['identities'][-1]) == meta['identities'])
    check('load potential ledger', int(stats['potential_released'].sum(dtype=np.int64)) * 2 == meta['totals']['V_initial'] - meta['totals']['V_final'])
    check('end load census conserves load', sum(i * n for i, n in enumerate(meta['end']['load_hist'])) == meta['totals']['total_load'])
    check('end census agrees with stats', np.array_equal(stats['counts_screen'][-1][1:], meta['end']['counts_screen'][1:]))
    sample = int(meta.get('display_sample', 1))
    if not full:
        check('view IDs obey declared fixed sample', bool(np.all(cat['id'] % sample == 0)))
    if full:
        check('full catalogue row count', actual['rows'] == meta['identities'])
        for tier, content in actual['tiers'].items():
            summary = quoted['tiers'][tier]
            for k, row in content['classes'].items():
                old = summary['classes'][k]
                for key in ('ever', 'alive_at_end', 'ended'):
                    check(f'{tier} class {k} {key} recomputed from complete catalogue', row[key] == old.get(key, 0))
                if row['ever']:
                    check(f'{tier} class {k} moved fraction recomputed', abs(row['moved_identities'] / row['ever'] - old['moved_fraction']) < 1e-14)
                # Seeded defects are not pair creation. This deliberately exposes stale summaries.
                check(f'{tier} class {k} pair-born identities exclude preparation', row['born']['conjugate_pair'] == old.get('born', {}).get('conjugate_pair', 0))
            for channel, n in content['pair_birth_events'].items():
                check(f'{tier} {channel} recomputed from complete catalogue', n == summary['channels'].get(channel, 0))
    first = next(c for c in meta['cuts'] if c['depth'] == 1)
    cut_checks = []
    # Verify actual IDs/classes on the first and last cut; never inflate a sample.
    ids = np.arange(len(cat['cls'])) if full else cat['id']
    for entry in (first, meta['end']):
        cut = _npz(source(entry['file']))
        cids = cut['screen_id']
        loc = np.searchsorted(ids, cids)
        present = (loc < len(ids))
        present[present] &= ids[loc[present]] == cids[present]
        check(f'{entry["file"]} all IDs available in selected catalogue', bool(present.all()))
        if not present.all():
            raise ValueError('A cut references IDs absent from its local catalogue')
        check(f'{entry["file"]} classes match catalogue', np.array_equal(cut['screen_cls'], cat['cls'][loc]))
        if entry['depth'] is not None:
            depth = entry['depth']
            check(f'{entry["file"]} rows alive on cut', bool(np.all((cat['birth_depth'][loc] <= depth) & ((cat['death_depth'][loc] > depth) | (cat['death_depth'][loc] < 0)))))
        else:
            check('end cut rows alive', bool(np.all(cat['death_sweep'][loc] < 0)))
        if sample == 1:
            check(f'{entry["file"]} full class census', np.array_equal(np.bincount(cut['screen_cls'], minlength=6)[1:], entry['counts_screen'][1:]))
        cut_checks.append({'url': f'/runs/{path.name}/{entry["file"]}', 'depth': entry['depth'],
                           'actual_stored_rows': int(len(cids)), 'full_summary_rows': entry['screen'],
                           'actual_positions_available': 'screen_xyz' in cut,
                           'stored_counts_screen': np.bincount(cut['screen_cls'], minlength=6).tolist()})
    ob = meta['observers'][0]
    observer = audit_observer(_npz(source(ob['file'])), meta)
    check('root observer tape replay', observer['load_row_mismatches'] == 0 and observer['final_loads_match'])
    check('root observer cut clocks', observer['cut_tick_mismatches'] == 0)
    screen = quoted['tiers']['screen']
    pairs = []
    for a, b in PAIRS:
        channel, end = f'0 -> {a} + {b}', f'{a} + {b} -> 0'
        pairs.append({'classes': [a, b], 'candidate_family': {(1, 5): 'qL', (2, 4): 'qR', (3, 3): 'lepton'}[(a, b)],
                      'first_cut_pair_births': first['counts_screen'][a] // (2 if a == b else 1),
                      'whole_run_pair_births': screen['channels'].get(channel, 0),
                      'whole_run_annihilations': screen['channels'].get(end, 0)})
    if not meta['config']['track_intra']:
        check('screen summary pair totals == tracked birth events', sum(p['whole_run_pair_births'] for p in pairs) == meta['tracker_totals']['births'])
        check('screen summary annihilations == tracked annihilation events', sum(p['whole_run_annihilations'] for p in pairs) == meta['tracker_totals']['annihilations'])
    check('class statistics final census', [screen['classes'][str(k)].get('alive_at_end', 0) for k in range(1, 6)] == meta['end']['counts_screen'][1:])
    balanced = int(meta['totals']['attempts_to_balanced_class'])
    # stats row s contains attempts [s*S,(s+1)*S); the containing row is partial.
    after = (balanced - 1) // int(meta['geometry']['seams']) + 1 if meta['totals']['balanced'] else None
    def after_sum(key):
        return int(stats[key][after:].sum()) if after is not None else None
    return {
        'name': path.name, 'level': meta['geometry']['level'], 'sweeps': meta['sweeps'],
        'preparation': meta['preparation'], 'regime': meta['regime'],
        'register': {'cost_table': meta['config']['cost_table'], 'kappa': meta['config']['kappa'],
                     'feeds_back_into_loads': False, 'initial_register': 'all zero'},
        'data_scope': {'full_catalogue_local': full, 'display_sample': sample,
                       'full_identity_count': meta['identities'], 'analysed_catalogue_rows': actual['rows'],
                       'catalogue_positions_available': 'birth_xyz' in cat,
                       'run_counts_authority': 'complete_catalogue_recomputed' if full else 'stored_full_run_summary_crosschecked_with_stats_and_sampled_cuts',
                       'missing_full_catalogue_counts_recomputed': full},
        'screen': {'pairs': pairs, 'classes': screen['classes'],
                   'first_cut': {'depth': 1, 'counts': first['counts_screen'],
                                 'incident_ticks_min': first['ticks_min'], 'incident_ticks_max': first['ticks_max']},
                   'end_counts': meta['end']['counts_screen'], 'cut_checks': cut_checks},
        'catalogue_census': actual,
        'balance': {'reached': meta['totals']['balanced'], 'attempt': balanced,
                    'first_wholly_after_balance_sweep_index': after,
                    'pair_birth_events_after_balance': after_sum('births'),
                    'potential_released_after_balance': after_sum('potential_released'),
                    'moves_after_balance': after_sum('moves'),
                    'tracked_tiers': ['screen', 'internal'] if meta['config']['track_intra'] else ['screen']},
        'observer': observer, 'checks': checks, 'all_checks_pass': all(x['passed'] for x in checks),
        'sources': sources,
    }


def export(root: Path, repository: Path) -> dict:
    source_paths = ['oph_defects/universe.py', 'oph_defects/kernels.py',
                    'oph_defects/tracking.py', 'oph_defects/dataset.py',
                    'oph_defects/DATASET.md', 'oph_defects/README.md',
                    'oph_defects/run_evidence.py', 'oph_defects/viewer2/cosmos.js',
                    'oph_defects/viewer2/cosmos_evidence.js',
                    'oph_defects/references/cosmology_comparison.json']
    comparison = json.loads((repository/'oph_defects/references/cosmology_comparison.json').read_text())
    runs = [audit_run(p, repository) for p in sorted(root.glob('L*'), key=lambda p: int(p.name[1:])) if p.name[1:].isdigit() and (p / 'run.json').is_file()]
    control = passive_register_control()
    return {'schema': 'oph.run_evidence.v1',
            'title': 'Executed register-defect census',
            'scope': 'Ordinary Law C load runs with the declared passive Z6 register; screen and internal tiers stay separate.',
            'interpretation': {
                'identity': 'qL, qR and lepton are candidate centre-class names. These rows do not identify Standard Model species, generation, spin statistics, mass or measured interaction amplitudes.',
                'birth': 'Defects are zero in the preparation, then born when current-driven register twists are accepted. This is event-born structure during an iid-load quench, not a simulation of vacuum pair production or reheating.',
                'freeze': 'Once loads balance, moved unit currents release zero mismatch potential. Positive-cost register proposals are rejected. Surviving defects can therefore be pinned by the declared cost barrier; this is not cosmological thermal freeze-out.',
                'clock': 'Causal depth defines a consistent cut, incident ticks count all local seam attempts, sweeps count solver work. None is calibrated to cosmological seconds by these run files.',
                'population': 'GEN lifetime modes mix short reactions, later reactions and right-censored survivors. GEN does not establish three particle generations.',
                'geometry': 'Use stored cut screen_xyz for actual positions. Birth-to-last interpolation is not a trajectory; random placement is an illustration. Sample counts must not be presented as reconstructed whole-run histories.',
            }, 'standard_cosmology': comparison, 'runs': runs, 'passive_register_control': control,
            'source_code': [{'file': str(p), 'sha256': sha256(repository / p)} for p in source_paths],
            'all_checks_pass': all(r['all_checks_pass'] for r in runs) and control['all_load_current_contrast_total_depth_records_equal']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    payload = export(args.runs.resolve(), repository)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.out), 'runs': len(payload['runs']), 'all_checks_pass': payload['all_checks_pass']}))
    if not payload['all_checks_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
