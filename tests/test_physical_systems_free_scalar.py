"""Full FE oscillator matrix evolution checked without modal time formulas."""
import numpy as np
import pytest
from scipy.linalg import expm

from oph_fpe.physical_systems.free_scalar import FreeScalar

MESH = {"vertices": [[0,0,0], [1,0,0], [0,1,0], [0,0,1]], "tetrahedra": [[0,1,2,3]]}


def test_mass_matrix_and_all_mode_matrix_exponential():
    m = FreeScalar(MESH)
    np.testing.assert_allclose(m.mass, (np.ones((4,4))+np.eye(4))/120, atol=1e-16)
    u, v = np.array([1., -.3, .2, .7]), np.array([.1, .2, -.4, .5])
    state = m.prepare(u, v)
    generator = np.block([[np.zeros((4,4)), np.eye(4)], [-np.linalg.solve(m.mass,m.operator), np.zeros((4,4))]])
    expected_energy = (v@m.mass@v+u@m.operator@u)/2
    for time in [0, .125, 1, 10, -2]:
        expected = expm(time*generator)@np.r_[u,v]
        actual = state.state(time)
        np.testing.assert_allclose(actual["field_mean"]+actual["velocity_mean"], expected, atol=3e-13)
        assert actual["normal_ordered_energy"] == pytest.approx(expected_energy)
        assert actual["model"]["occupation_truncation"] is False


def test_constant_mode_exact_continuum_solution_and_smeared_covariance():
    m = FreeScalar(MESH, mass_squared=2., hbar=.7)
    state = m.prepare(np.ones(4), np.zeros(4))
    for t in [0., .3, 3.]:
        sample = state.smeared(t, np.ones(4))
        assert sample["mean"] == pytest.approx(np.cos(np.sqrt(2)*t)/6)
        assert sample["variance"] == pytest.approx(.7/(12*np.sqrt(2)))
        assert sample["commutator_with_smeared_velocity_over_i"] == pytest.approx(.7/6)


def test_integrated_local_energy_matches_quantum_energy():
    m = FreeScalar(MESH)
    state = m.prepare([1,0,-1,.2], [.1,.2,.3,.4])
    # Symmetric degree-two tetrahedron rule, independent of FE matrix assembly.
    a = (5+3*np.sqrt(5))/20; b = (5-np.sqrt(5))/20
    rows = [state.field(.73,0,[a if i == j else b for i in range(4)]) for j in range(4)]
    actual = state.state(.73)
    assert sum(x["normal_ordered_energy_density"] for x in rows)/24 == pytest.approx(actual["normal_ordered_energy"])
    assert sum(x["cutoff_vacuum_energy_density"] for x in rows)/24 == pytest.approx(actual["cutoff_vacuum_energy"])
    vacuum = m.prepare(np.zeros(4), np.zeros(4))
    sample = vacuum.field(0,0,[.25]*4)
    assert sample["normal_ordered_energy_density"] == 0
    assert sample["cutoff_field_variance"] > 0


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_bad_mass(bad):
    with pytest.raises(ValueError): FreeScalar(MESH,mass_squared=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_bad_time(bad):
    state = FreeScalar(MESH).prepare([0]*4,[0]*4)
    with pytest.raises(ValueError): state.state(bad)


def test_bad_mesh_and_state():
    with pytest.raises(ValueError): FreeScalar({**MESH,"tetrahedra":[[0,1,2,4]]})
    with pytest.raises(ValueError): FreeScalar({**MESH,"tetrahedra":[[0,1,2,2]]})
    with pytest.raises(ValueError): FreeScalar({**MESH,"tetrahedra":[[0,1,2,3]]*2})
    m=FreeScalar(MESH)
    with pytest.raises(ValueError): m.prepare([0]*3,[0]*4)
    with pytest.raises(ValueError): m.prepare([float("nan")]*4,[0]*4)
