"""Full FE oscillator matrix evolution checked without modal time formulas."""
import numpy as np
import pytest
from scipy.linalg import expm, sqrtm

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


@pytest.mark.parametrize("times", [(0., .31), (.73, -.2), (2., 2.)])
def test_two_point_against_independent_full_matrix_evolution(times):
    m = FreeScalar(MESH, mass_squared=1.7, hbar=.6)
    u, v = np.array([1., -.3, .2, .7]), np.array([.1, .2, -.4, .5])
    state = m.prepare(u, v)
    # Matrix square root in mass-whitened coordinates, without using the
    # simulator's eigenvectors or its modal propagator.
    inv_l = np.linalg.inv(np.linalg.cholesky(m.mass))
    omega = sqrtm(inv_l@m.operator@inv_l.T)
    cqq = .3*inv_l.T@np.linalg.inv(omega)@inv_l
    cvv = .3*inv_l.T@omega@inv_l
    cqv = .3j*np.linalg.inv(m.mass)
    vacuum = np.block([[cqq, cqv], [-cqv, cvv]])
    generator = np.block([[np.zeros((4,4)), np.eye(4)],
                          [-np.linalg.solve(m.mass,m.operator), np.zeros((4,4))]])
    lt, rt = (expm(t*generator)[:4] for t in times)
    connected_matrix = lt@vacuum@rt.T
    full_matrix = connected_matrix+np.outer(lt@np.r_[u,v], rt@np.r_[u,v])
    left, right = np.array([.1,.2,.3,.4]), np.array([.4,.1,.2,.3])
    result = state.two_point(times[0],0,left,times[1],0,right)
    assert complex(*result["wightman"]) == pytest.approx(left@full_matrix@right, abs=2e-12)
    assert complex(*result["connected_wightman"]) == pytest.approx(left@connected_matrix@right, abs=2e-12)
    swapped = state.two_point(times[1],0,right,times[0],0,left)
    assert complex(*result["wightman"]) == pytest.approx(complex(*swapped["wightman"]).conjugate())
    assert complex(*result["commutator"]) == pytest.approx(
        complex(*result["wightman"])-complex(*swapped["wightman"]), abs=2e-12)
    f, g = np.array([1.,-.4,.2,1.3]), np.array([.2,.6,-.3,.7])
    smeared = state.smeared_two_point(times[0],f,times[1],g)
    assert complex(*smeared["wightman"]) == pytest.approx(f@m.mass@full_matrix@m.mass@g, abs=2e-13)


def test_smeared_constant_two_point_exact_solution():
    m = FreeScalar(MESH,mass_squared=2.,hbar=.7)
    state = m.prepare(np.ones(4),np.zeros(4))
    t, s = .9, -.4
    connected = .7/(12*np.sqrt(2))*np.exp(-1j*np.sqrt(2)*(t-s))
    result = state.smeared_two_point(t,np.ones(4),s,np.ones(4))
    assert complex(*result["connected_wightman"]) == pytest.approx(connected)
    assert complex(*result["wightman"]) == pytest.approx(
        connected+np.cos(np.sqrt(2)*t)*np.cos(np.sqrt(2)*s)/36)


def test_two_point_positive_type_and_equal_point_variance():
    m = FreeScalar(MESH)
    state = m.prepare([1,.2,-.1,.3],[.1,.2,.3,.4])
    probes = [(t,[.25]*4) for t in [0,.1,.7]]+[(.3,np.eye(4)[i]) for i in range(3)]
    gram = np.array([[complex(*state.two_point(t,0,b,s,0,c)["wightman"])
                      for s,c in probes] for t,b in probes])
    np.testing.assert_allclose(gram,gram.conj().T,atol=1e-12)
    assert np.linalg.eigvalsh(gram).min() > -1e-12
    point = state.field(.7,0,[.25]*4)
    diagonal = state.two_point(.7,0,[.25]*4,.7,0,[.25]*4)
    assert diagonal["connected_wightman"] == pytest.approx([point["cutoff_field_variance"],0])
    assert diagonal["wightman"][0] == pytest.approx(point["field_mean"]**2+point["cutoff_field_variance"])


def test_two_point_invalid_probes_fail():
    state = FreeScalar(MESH).prepare([0]*4,[0]*4)
    with pytest.raises(ValueError): state.two_point(0,1,[.25]*4,0,0,[.25]*4)
    with pytest.raises(ValueError): state.two_point(0,0,[-1,0,1,1],0,0,[.25]*4)
    with pytest.raises(ValueError): state.smeared_two_point(0,[0]*3,0,[0]*4)
    with pytest.raises(ValueError): state.smeared_two_point(float("nan"),[0]*4,0,[0]*4)


@pytest.mark.parametrize("value", [np.complex128(1+2j), np.complex128(1), True])
def test_nonreal_scalar_inputs_are_not_coerced(value):
    with pytest.raises(ValueError): FreeScalar(MESH,mass_squared=value)
    with pytest.raises(ValueError): FreeScalar(MESH,hbar=value)
    state = FreeScalar(MESH).prepare([0]*4,[0]*4)
    with pytest.raises(ValueError): state.state(value)
    with pytest.raises(ValueError): state.smeared_two_point(value,[1]*4,0,[1]*4)


@pytest.mark.parametrize("vector", [np.ones(4,dtype=complex)*(1+2j),
                                     np.ones(4,dtype=complex), [True,0.,0.,0.],
                                     ["1",0.,0.,0.]])
def test_nonreal_vector_inputs_are_not_coerced(vector):
    m = FreeScalar(MESH)
    with pytest.raises(ValueError): m.prepare(vector,[0]*4)
    state = m.prepare([0]*4,[0]*4)
    with pytest.raises(ValueError): state.smeared(0,vector)
    with pytest.raises(ValueError): state.smeared_two_point(0,vector,.3,[1]*4)


def test_two_point_overflow_fails_explicitly():
    state = FreeScalar(MESH).prepare([0]*4,[0]*4)
    with pytest.raises(ValueError,match="overflow"):
        state.smeared_two_point(0,np.ones(4)*1e308,.3,np.ones(4)*1e308)


def test_complex_mesh_coordinates_are_not_coerced():
    with pytest.raises(ValueError):
        FreeScalar({**MESH,"vertices":np.asarray(MESH["vertices"],dtype=complex)+1j})


@pytest.mark.parametrize("bad", [False,np.bool_(False),0.,10**100])
def test_tetrahedron_indices_are_not_coerced(bad):
    with pytest.raises(ValueError): FreeScalar({**MESH,"tetrahedra":[[bad,1,2,3]]})
