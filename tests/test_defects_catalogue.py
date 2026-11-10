"""Exact invariants of the Z6 defect run and its exported catalogue."""
import gzip
import json
import struct

import numpy as np
import pytest

from oph_defects import geometry, run as runner
from oph_defects.dynamics import Z6Run, region_check, temperature_schedule


@pytest.mark.parametrize('level', [0, 1, 2, 3])
def test_refined_carrier_is_a_closed_oriented_sphere(level):
    s = geometry.build(level)
    n = 1 << level
    assert (s.vertex_count, s.edge_count, s.face_count) == (10 * n * n + 2, 30 * n * n, 20 * n * n)
    assert s.euler_characteristic == 2
    # each seam is traversed once in each direction
    assert (s.edge_plus != s.edge_minus).all()
    # a vertex potential has zero loop sum on every face
    rng = np.random.default_rng(level)
    pot = rng.integers(0, 6, s.vertex_count)
    fv = s.face_vertices
    lo, hi = np.minimum(fv, np.roll(fv, -1, 1)), np.maximum(fv, np.roll(fv, -1, 1))
    links = np.zeros(s.edge_count, np.int64)
    links[s.face_edges.ravel()] = (pot[hi.ravel()] - pot[lo.ravel()]) % 6
    assert not geometry.face_curvature(s, links).any()


def test_level_zero_is_the_certified_carrier():
    s = geometry.build(0)
    assert (s.face_vertices == geometry.BASE_FACES).all()
    assert geometry.base_edges().shape == (30, 2)


def test_one_seam_change_twists_exactly_its_two_faces_as_conjugates():
    s = geometry.build(2)
    links = np.zeros(s.edge_count, np.int64)
    links[11] = 1
    c = geometry.face_curvature(s, links)
    nz = np.flatnonzero(c)
    assert sorted(nz.tolist()) == sorted([int(s.edge_plus[11]), int(s.edge_minus[11])])
    assert sorted(c[nz].tolist()) == [1, 5]


def test_face_index_round_trip_and_positions():
    level = 4
    s = geometry.build(level)
    f = np.arange(s.face_count)
    b, i, j, up = geometry.decode_faces(f, level)
    assert (geometry.encode_faces(b, i, j, up, level) == f).all()
    p = geometry.face_positions(f, level)
    assert np.allclose(np.linalg.norm(p, axis=1), 1)
    # centroids are distinct
    assert len(np.unique(np.round(p, 9), axis=0)) == s.face_count


def test_run_keeps_every_invariant_and_is_deterministic():
    s = geometry.build(3)
    T = temperature_schedule([(0, 1.1), (40, 1.1), (90, 0.0), (120, 0.0)], 120)
    digests = []
    for _ in range(2):
        run = Z6Run(s, seed=5, proposal_fraction=0.1, cost=runner.BINDING)
        rng = np.random.default_rng(11)
        with np.errstate(over='ignore'):
            for t in range(120):
                run.step(float(T[t]))
                if t % 10 == 0:
                    run.check()
                    region_check(run, rng, trials=4)
        c = run.cat.view()
        st = run.stats
        # births and deaths balance against the live count
        live = int((run.curv != 0).sum())
        assert live == int((c['death'] < 0).sum())
        assert sum(st.births) * 2 + 2 * sum(st.splits) + sum(st.fusions) + 2 * sum(st.exchanges) == run.cat.n
        # conjugate pairs at birth carry classes summing to zero
        pairs = c['partner'][c['p1'] < 0]
        born = np.flatnonzero(c['p1'] < 0)
        assert (((c['cls'][born].astype(int) + c['cls'][pairs].astype(int)) % 6) == 0).all()
        digests.append((run.links.tobytes(), c['birth'].tobytes(), c['death'].tobytes()))
    assert digests[0] == digests[1]


def test_export_round_trip(tmp_path):
    runner.main(['--level', '2', '--ticks', '30', '--seed', '3', '--schedule', '0:1.1,30:0.0',
                 '--snapshot-every', '10', '--check-every', '10', '--out', str(tmp_path)])
    meta = json.loads((tmp_path / 'run.json').read_text())
    assert meta['geometry']['faces'] == 320 and len(meta['snapshots']) == 4
    raw = gzip.open(tmp_path / meta['snapshots'][1]['file'], 'rb').read()
    assert raw[:4] == runner.SNAP_MAGIC
    tick, count = struct.unpack('<II', raw[4:12])
    assert tick == 10 and count == meta['snapshots'][1]['count']
    faces = np.frombuffer(raw[12:12 + 4 * count], '<i4')
    classes = np.frombuffer(raw[12 + 4 * count:12 + 5 * count], 'u1')
    assert (faces < 320).all() and set(np.unique(classes)) <= {1, 2, 3, 4, 5}
    cat = gzip.open(tmp_path / 'catalogue.bin.gz', 'rb').read()
    assert cat[:4] == runner.CAT_MAGIC
    assert struct.unpack('<I', cat[4:8])[0] == meta['identities']
    assert (tmp_path / 'DIGEST').read_text().strip()


def test_federation_faces_and_law_c_exactness(tmp_path):
    from oph_defects import federation_geometry as G
    from oph_defects import federation_run as R
    fed, fc = G.build(2)
    assert fc.face_count == 20 * fed.carriers + fc.vertices
    counts = np.bincount(np.concatenate([fc.seam_plus, fc.seam_minus]), minlength=fc.face_count)
    assert set(counts[:20 * fed.carriers].tolist()) == {3}
    assert set(counts[20 * fed.carriers:].tolist()) <= {5, 6}
    rng = np.random.default_rng(2)
    pot, potc = rng.integers(0, 6, fed.ports), rng.integers(0, 6, fed.carriers)
    reg = np.zeros(fed.seams, np.int64)
    nI = fc.intra_seams
    reg[:nI] = (pot[fed.seam_b[:nI]] - pot[fed.seam_a[:nI]]) % 6
    reg[nI:] = (potc[fed.inter_pairs[:, 2]] - potc[fed.inter_pairs[:, 0]]) % 6
    assert not G.face_classes(fc, reg).any()
    meta = R.run(2, 30, tmp_path, loads_check=True, quiet=True)
    assert meta['geometry']['coords'] and meta['identities'] > 0
    totals = np.array(meta['stats']['counts_intra'][-1]) @ np.arange(6)
    assert totals % 6 == 0


def test_federation_checkpoints_replay_exactly(tmp_path):
    from oph_defects import data
    from oph_defects import federation_run as R
    R.run(2, 12, tmp_path, checkpoint_every=5, record_windows='10:12', quiet=True)
    run = data.Run(tmp_path)
    ck10 = run.checkpoint(10)
    last = None
    for _, rec in run.replay(5, 10):
        last = rec
    assert (last['loads_after'] == ck10['loads']).all()
    assert (last['register_after'] == ck10['register']).all()
    assert (last['classes_after'] == ck10['classes']).all()
    stored = run.records(10)
    replayed = next(r for _, r in run.replay(10, 11))
    for key in ('seam', 'coin', 'current'):
        assert (stored[key] == replayed[key]).all()
    tape = run.observer_tape(3, range(10, 12))
    assert tape.shape[1] == 5 and tape.shape[0] > 0
    # a resumed run continues the same history
    R.run(2, 2, tmp_path / 'cont', resume=tmp_path / 'checkpoints' / 'ckpt_000010.npz', checkpoint_every=100, quiet=True)
    cont = data.Run(tmp_path / 'cont')
    assert (cont.checkpoint(12)['loads'] == run.checkpoint(12)['loads']).all()
