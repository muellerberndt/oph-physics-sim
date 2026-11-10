"""Exactness tests of the universe runner (oph_defects.universe)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oph_defects import fedgeom as FG
from oph_defects import kernels as K
from oph_defects.dataset import Dataset
from oph_defects.universe import COSTS, Config, Engine, Run, prepare_loads, schedule_seed

REPO = Path(__file__).resolve().parents[1]


def _run(tmp, level=2, sweeps=40, **kw):
    cfg = Config(level=level, sweeps=sweeps, checkpoint_every=kw.pop('checkpoint_every', 10), records=kw.pop('records', 'all'), **kw)
    return Run(cfg, tmp, quiet=True).run()


def _same_npz(a: Path, b: Path):
    with np.load(a) as x, np.load(b) as y:
        assert sorted(x.files) == sorted(y.files), (a.name, sorted(set(x.files) ^ set(y.files)))
        for k in x.files:
            assert x[k].dtype == y[k].dtype and x[k].shape == y[k].shape and (x[k] == y[k]).all(), (a.name, k)


def test_geometry_equals_the_pinned_builder():
    for level in (0, 1, 2, 3):
        assert FG.verify_against_pinned(FG.build(level, cache=False))['equal_to_pinned_builder']


def test_declared_inputs_equal_oph_exact():
    from oph_exact import federation as F
    for level in (0, 2, 4):
        geo = FG.build(level)
        loads, _ = prepare_loads('exact06', level, geo.ports)
        assert (F.initial_loads(level, geo.ports) == loads).all()
        for schedule in (0, 3, 15):
            assert F.schedule_seed(level, 'port_pair', schedule) == schedule_seed(level, schedule)


def test_fast_generator_equals_numpy():
    for seed, bound, n in ((914100, 645120, 300_001), (920100, 660602880, 400_000), (3, 2, 999), (9, 40320, 123_457)):
        assert K.verify_fast_rng(seed, bound, n)


@pytest.mark.parametrize('level', [1, 3])
def test_loads_equal_run_integer_law(level):
    """The loads are those of oph_exact.federation.run_integer_law: same terminal state, same tallies."""
    from oph_exact import federation as F
    fed = F.build_federation(level, 'port_pair')
    loads = F.initial_loads(level, fed.ports)
    seed = F.schedule_seed(level, 'port_pair', 0)
    ref = F.run_integer_law(fed, loads, seed, engine='python')
    geo = FG.build(level)
    eng = Engine(geo, loads, seed, COSTS['binding'], 5, True)
    descents = swaps = 0
    v = int((loads.astype(np.int64) ** 2).sum())
    first = -1
    for s in range(ref['sweeps']):
        seq, coin = eng.draw()
        for lo in range(0, eng.S, eng.chunk):
            n = eng.run_chunk(seq[lo:lo + eng.chunk], coin[lo:lo + eng.chunk], lo, s)
            cur, con = eng.o_cur[:n].astype(np.int64), eng.o_con[:n].astype(np.int64)
            descents += int((np.abs(con) >= 2).sum()); swaps += int(((cur != 0) & (np.abs(con) == 1)).sum())
            drop = np.cumsum(cur * (con - cur))
            if first < 0 and v - 2 * int(drop[-1]) == ref['descent_functional_minimum']:
                first = s * eng.S + lo + int(np.flatnonzero(v - 2 * drop == ref['descent_functional_minimum'])[0]) + 1
            v -= 2 * int(drop[-1])
    assert (eng.x == ref['state']).all()
    assert (descents, swaps) == (ref['descents'], ref['swaps'])
    assert first == ref['attempts_to_balanced_class']
    assert eng.rng.bit_generator.state is not None


def test_level5_matches_the_stored_september_receipt():
    """runs/exact_federation_L5/summary.json: seed 914100 reaches the balanced class after 33,820,528 attempts."""
    path = REPO / 'runs' / 'exact_federation_L5' / 'summary.json'
    if not path.exists():
        path = REPO / 'oph_defects' / 'references' / 'federation_level5.json'
    row = next(r for r in json.loads(path.read_text())['integer_law'] if r['seed'] == 914100)
    geo = FG.build(5)
    loads, _ = prepare_loads('exact06', 5, geo.ports)
    eng = Engine(geo, loads, schedule_seed(5, 0), COSTS['binding'], 5, False)
    descents = swaps = 0
    for s in range(row['sweeps']):
        seq, coin = eng.draw()
        for lo in range(0, eng.S, eng.chunk):
            n = eng.run_chunk(seq[lo:lo + eng.chunk], coin[lo:lo + eng.chunk], lo, s)
            cur, con = eng.o_cur[:n], eng.o_con[:n]
            descents += int((np.abs(con) >= 2).sum()); swaps += int(((cur != 0) & (np.abs(con) == 1)).sum())
    waits = row['sweeps'] * geo.seams - descents - swaps
    assert (descents, swaps, waits) == (row['descents'], row['swaps'], row['waits'])
    x = eng.x.astype(np.int64)
    assert int((x * x).sum()) == row['descent_functional_terminal'] == row['descent_functional_minimum']


@pytest.mark.parametrize('reverse', [False, True])
def test_causal_layers_equal_the_schedule_order(tmp_path, reverse):
    """Executing each causal layer of disjoint attempts in any order reproduces every output."""
    a = _run(tmp_path / 'seq', level=2, sweeps=30)
    b = _run(tmp_path / 'lay', level=2, sweeps=30, backend='layers', chunk=977, reverse_layers=reverse)
    assert a['stats'] == {k: v for k, v in b['stats'].items()} or all(a['stats'][k] == b['stats'][k] for k in a['stats'] if k != 'seconds')
    assert a['totals'] == b['totals'] and a['identities'] == b['identities']
    names = ['catalogue.npz', 'cuts/end.npz', 'final_loads.npy'] + [c['file'] for c in a['cuts']] + [r['file'] for r in a['records']] \
        + [c['file'] for c in a['checkpoints']] + [o['file'] for o in a['observers']]
    assert [c['file'] for c in a['cuts']] == [c['file'] for c in b['cuts']]
    for name in names:
        if name.endswith('.npy'):
            assert (np.load(tmp_path / 'seq' / name) == np.load(tmp_path / 'lay' / name)).all()
        else:
            _same_npz(tmp_path / 'seq' / name, tmp_path / 'lay' / name)


def test_resume_continues_the_same_history(tmp_path):
    full = _run(tmp_path / 'full', level=2, sweeps=30)
    _run(tmp_path / 'half', level=2, sweeps=20)
    cfg = Config(level=2, sweeps=10, checkpoint_every=10, records='all')
    cont = Run(cfg, tmp_path / 'cont', resume=tmp_path / 'half' / 'checkpoints' / 'ckpt_000020.npz', quiet=True).run()
    assert cont['sweeps'] == 30 and cont['totals'] == full['totals'] and cont['identities'] == full['identities']
    for k in full['stats']:
        if k != 'seconds':
            assert full['stats'][k] == cont['stats'][k], k
    assert [c['depth'] for c in full['cuts']] == [c['depth'] for c in cont['cuts']]
    for name in ['catalogue.npz', 'cuts/end.npz'] + [c['file'] for c in full['cuts']] + [o['file'] for o in full['observers']] \
            + ['checkpoints/ckpt_000030.npz']:
        _same_npz(tmp_path / 'full' / name, tmp_path / 'cont' / name)


def test_replay_and_tapes_equal_the_stored_records(tmp_path):
    meta = _run(tmp_path, level=2, sweeps=25, records='3:6,20:22')
    ds = Dataset(tmp_path)
    for s in (4, 21):
        stored = ds.records(s)
        (_, again), = list(ds.replay(s, s + 1))
        for k in stored:
            assert (stored[k] == again[k]).all(), k
        # the stored fields determine both loads before the attempt, and the law
        xa, xb = (stored['total'].astype(int) + stored['contrast']) // 2, (stored['total'].astype(int) - stored['contrast']) // 2
        assert ((xa - xb) == stored['contrast']).all()
        ya = np.where(stored['coin'] == 1, (xa + xb + 1) // 2, (xa + xb) // 2)
        assert ((xa - ya) == stored['current']).all()
    # a featured tape equals the tape rebuilt from the records, and it reproduces the observer loads
    ob = meta['observers'][0]['carrier']
    file_tape = ds.observer(ob)
    rebuilt = ds.observer_tape(ob, range(0, 25))
    for k in ('sweep', 'pos', 'seam', 'slot', 'end', 'coin', 'current', 'twist', 'depth'):
        assert (file_tape['tape_' + k].astype(np.int64) == rebuilt[k].astype(np.int64)).all(), k
    assert (file_tape['loads_by_tick'][-1] == np.load(tmp_path / 'final_loads.npy')[12 * ob:12 * ob + 12]).all()
    assert file_tape['tape_sweep'].size == file_tape['cut_ticks'][-1]


def test_a_cut_is_the_state_after_exactly_its_own_events(tmp_path):
    """Re-execute only the attempts of depth <= T, in schedule order: loads, classes and clocks equal the cut."""
    meta = _run(tmp_path, level=2, sweeps=30)
    ds = Dataset(tmp_path)
    geo = ds.geometry
    cfg = meta['config']
    loads, _ = prepare_loads('exact06', 2, geo.ports)
    recs = [ds.records(s) for s in range(30)]
    for entry in meta['cuts'][3::4]:
        T = entry['depth']
        cut = ds.cut(entry['k'])
        eng = Engine(geo, loads, cfg['seed'], COSTS[cfg['cost']], cfg['kappa'], True)
        for s, rec in enumerate(recs):
            keep = rec['depth'] <= T
            seq, coin = np.ascontiguousarray(rec['seam'][keep]), np.ascontiguousarray(rec['coin'][keep])
            for lo in range(0, seq.size, eng.chunk):
                n = eng.run_chunk(seq[lo:lo + eng.chunk], coin[lo:lo + eng.chunk], lo, s)
                assert (eng.o_dep[:n] == rec['depth'][keep][lo:lo + n]).all()
        assert (eng.x == cut['loads']).all()
        assert (eng.ticks == cut['ticks']).all()
        faces = np.flatnonzero(eng.cls)
        stored = np.concatenate([cut['intra_face'], cut['screen_face']])
        assert (np.sort(stored) == faces).all()
        assert (eng.cls[cut['screen_face']] == cut['screen_cls']).all() and (eng.cls[cut['intra_face']] == cut['intra_cls']).all()


def test_register_conserves_class_on_every_closed_surface(tmp_path):
    meta = _run(tmp_path, level=2, sweeps=12)
    ds = Dataset(tmp_path)
    geo = ds.geometry
    ck = ds.checkpoint(12)
    cls = geo.face_classes(ck['register'])
    assert (cls == ck['classes']).all()
    assert int(cls[geo.intra_faces:].astype(np.int64).sum()) % 6 == 0
    assert not (cls[:geo.intra_faces].reshape(-1, 20).astype(np.int64).sum(axis=1) % 6).any()
    cat = ds.catalogue()
    alive = cat['death_sweep'] < 0
    assert int(alive.sum()) == int((ck['classes'] != 0).sum())
    assert meta['provenance']['defining_sources_equal_model_pins']
