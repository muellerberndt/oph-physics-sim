"""Independent symmetry readouts and adversarial query custody checks."""
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pytest

from oph_fpe.physical_systems import WhitneySystem
from oph_fpe.physical_systems import whitney
from oph_fpe.physical_systems.__main__ import main

RER = Path(__file__).resolve().parents[2]/"reverse-engineering-reality"


@pytest.fixture(scope="module")
def system():
    if not RER.is_dir(): pytest.skip("requires the independently pinned sibling research checkout")
    return WhitneySystem.load(RER)


@pytest.mark.parametrize("frame", [0, 1, 20, 40, 80])
def test_dressed_fields_against_separate_symmetry_formula(system, frame):
    q = np.array([float(Fraction(system.observer(i, frame)["local_state"]["q_exact"])) for i in range(5)])
    v = np.array([float(Fraction(system.observer(i, frame)["local_state"]["v_exact"])) for i in range(5)])
    alpha, c, b = q[0], complex(*q[1:3]), complex(*q[3:5])
    cv, bv = complex(*v[1:3]), complex(*v[3:5])
    e = .25; mesh = system.mesh()
    for cell in range(20):
        lam = [.37, .11, .23, .29]; s = lam[0]
        actual = system.field(frame, cell, lam)
        points = np.array(mesh["vertices"])[mesh["tetrahedra"][cell]]
        grad_s = np.linalg.solve((points[1:]-points[0]), -np.ones(3))
        phase = np.exp(-1j*e*alpha*s); c_rot = np.exp(1j*e*alpha)*c
        scalar = phase*(s*c_rot+(1-s)*b)
        dt = phase*(s*np.exp(1j*e*alpha)*cv+(1-s)*bv+1j*e*s*(1-s)*v[0]*(c_rot-b))
        spatial = phase*(c_rot-b)*grad_s
        assert complex(*actual["scalar"]) == pytest.approx(scalar, abs=3e-12)
        assert complex(*actual["covariant_time_derivative"]) == pytest.approx(dt, abs=3e-12)
        observed = np.array(actual["covariant_spatial_derivative"])
        np.testing.assert_allclose(observed[:, 0]+1j*observed[:, 1], spatial, atol=3e-12)
        np.testing.assert_allclose(actual["electric"], v[0]*grad_s, atol=3e-12)
        np.testing.assert_allclose(actual["magnetic"], 0, atol=3e-12)


def test_all_frames_energy_and_copy_isolation(system):
    energies = [system.energy(i)["totals"]["total"] for i in range(81)]
    assert max(energies)-min(energies) < 1e-9
    mesh = system.mesh(); mesh["vertices"][0][0] = 123
    assert system.mesh()["vertices"][0][0] == 0
    assert system.quantum()["same_classical_history"] is False
    assert system.clock(80)["physical_clock_calibrated"] is False


@pytest.mark.parametrize("bad", [-1, 81, True, 1.5])
def test_invalid_frames(system, bad):
    with pytest.raises(ValueError): system.energy(bad)


@pytest.mark.parametrize("bad", [[1,0,0], [-.1,.1,.5,.5], [float("nan"),0,0,1], [True,0,0,0], [.2]*4])
def test_invalid_points(system, bad):
    with pytest.raises(ValueError): system.field(0, 0, bad)


def test_energy_mutation_rejected(system):
    frame = system._packets["charged_instrument"]["frames"][0]
    old = frame["energy"]
    try:
        frame["energy"] += 1
        with pytest.raises(ValueError, match="disagrees"): system.energy(0)
    finally: frame["energy"] = old


@pytest.mark.parametrize("target", ["whitney_charged_instrument_receipt.json", "verify_whitney_charged_dynamics.py"])
def test_stale_byte_pin_rejected(system, monkeypatch, target):
    read = whitney._read_bytes
    monkeypatch.setattr(whitney, "_read_bytes", lambda p: read(p)+b" " if p.name == target else read(p))
    with pytest.raises(ValueError, match="mismatch"): WhitneySystem.load(RER)


@pytest.mark.parametrize("raw", [b'{"x": 1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'])
def test_strict_json(raw):
    with pytest.raises(ValueError): whitney._strict_json(raw)


def test_cli(system, capsys):
    assert main(["--rer-root", str(RER), "observer", "2", "4"]) == 0
    assert json.loads(capsys.readouterr().out)["patch_id"] == 2
