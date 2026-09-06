"""Independent Gaussian quadrature, scope boundaries and source-custody checks."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad

from oph_fpe.physical_systems import bridges
from oph_fpe.physical_systems.bridges import ControlledBridges
from oph_fpe.physical_systems.__main__ import main

RER = Path(__file__).resolve().parents[2]/"reverse-engineering-reality"


@pytest.fixture(scope="module")
def system():
    if not RER.is_dir():
        pytest.skip("requires the byte-pinned sibling research checkout")
    return ControlledBridges.load(RER)


def rotate(x, theta):
    z = np.asarray(x[30:43])+1j*np.asarray(x[43:])
    z *= np.exp(1j*theta)
    return np.r_[x[:30], z.real, z.imag]


@pytest.mark.parametrize("sample", [0, 1])
@pytest.mark.parametrize("width", ["1/4", "1/2", "1"])
def test_all_modes_covariance_and_independent_circle_integrals(system, sample, width):
    packet = system.packet(sample, width)
    q, p = np.array(packet["seed_center"]), np.array(packet["seed_cotangent"])
    sigma = float(Fraction(width))
    cov = packet["seed_covariance"]
    assert cov["rank"] == 56
    np.testing.assert_array_equal(cov["position_diagonal"], np.full(56, sigma**2))
    np.testing.assert_array_equal(cov["momentum_diagonal"], np.full(56, 1/(4*sigma**2)))
    np.testing.assert_allclose(np.array(cov["momentum_diagonal"])*cov["position_diagonal"], .25)

    def integrand(theta, moment=False):
        rq, rp = rotate(q, theta), rotate(p, theta)
        overlap = np.exp(-np.sum((rq-q)**2)/(8*sigma**2)-sigma**2*np.sum((rp-p)**2)/2+
                         .5j*np.dot(p+rp, q-rq))
        if moment:
            mean = (q+rq)/2+1j*sigma**2*(rp-p)
            overlap *= 26*sigma**2+np.sum(mean[30:]**2)
        return float(overlap.real)

    norm = quad(integrand, -np.pi, np.pi, epsabs=1e-12, epsrel=1e-12)[0]/(2*np.pi)
    radius = quad(lambda t: integrand(t, True), -np.pi, np.pi, epsabs=1e-11)[0]/(2*np.pi*norm)
    neutral = packet["neutral_projection"]
    assert neutral["norm_squared"] == pytest.approx(norm, abs=2e-13)
    assert neutral["scalar_radius_numeric"] == pytest.approx(radius, abs=2e-10)
    assert abs(radius-(26*sigma**2+q[30:]@q[30:])) > .01
    assert neutral["full_projected_covariance_computed"] is False
    assert neutral["scalar_position_mean"] == [0.]*26
    assert packet["contract"]["classical_samples_are_quantum_means"] is False
    if sample == 0:
        assert norm > 1/64
        assert neutral["initial_exact_projection_norm_squared_lower"] == "1/64"
    else:
        assert neutral["initial_exact_projection_norm_squared_lower"] is None


@pytest.mark.parametrize("width", ["1/4", "1/2", "1"])
def test_origin_half_density_exact_gaussian_factor_and_projection(system, width):
    packet = system.packet(0, width)
    q, p = np.array(packet["seed_center"]), np.array(packet["seed_cotangent"])
    sigma = float(Fraction(width))
    expected = ((2*np.pi*sigma**2)**(-14)*np.exp(-q@q/(4*sigma**2)-1j*p@q)/
                math.sqrt(packet["neutral_projection"]["norm_squared"]))
    result = system.packet_half_density([0.]*56, width=width)
    observed = complex(*result["normalized_neutral_half_density"])
    assert observed == pytest.approx(expected, rel=3e-13, abs=1e-30)
    assert result["probability_density_relative_to_dq"] == pytest.approx(abs(expected)**2, rel=4e-13)
    assert result["certified_pointwise_error"] is None
    assert result["quantum_propagation"] is False
    assert abs(observed/(expected*math.sqrt(packet["neutral_projection"]["norm_squared"]))-1) > 1


def test_circle_invariance_keeps_radiative_directions(system):
    point = np.asarray(system.packet()["seed_center"])+np.linspace(-.04, .04, 56)
    a = system.packet_half_density(point)
    b = system.packet_half_density(rotate(point, .713))
    np.testing.assert_allclose(a["normalized_neutral_half_density"], b["normalized_neutral_half_density"], rtol=1e-12, atol=1e-14)
    changed = point.copy(); changed[0] += 1
    c = system.packet_half_density(changed)
    assert abs(complex(*c["normalized_neutral_half_density"])-complex(*a["normalized_neutral_half_density"])) > 1e-5


def test_overlap_at_pi_zero_and_periodic(system):
    zero = system.packet_overlap(0)
    assert zero["overlap"] == [1., 0.]
    pi = system.packet_overlap(np.pi)
    assert complex(*pi["overlap"]) == pytest.approx(math.exp(-2*pi["A"]), rel=2e-13)
    np.testing.assert_allclose(system.packet_overlap(.41)["overlap"], system.packet_overlap(.41+2*np.pi)["overlap"], atol=1e-14)


def test_coulomb_velocity_is_not_raw_temporal_gauge_velocity(system):
    packet = system.packet(0)
    assert packet["coulomb_velocity"][43] > 3.03
    assert abs(packet["coulomb_velocity"][44]+1) > .003
    assert abs(packet["coulomb_scalar_potential"][0]) > .15
    p = packet["seed_cotangent"]
    np.testing.assert_allclose(p[43:], [3+math.sqrt(5)]+[-.25-math.sqrt(5)/12]*12, atol=3e-14)
    assert abs(sum(p[43:])) < 3e-14
    assert packet["classical_center_model_time"] == 0
    assert system.packet(1)["classical_center_model_time"] == .025
    assert packet["contract"]["requested_propagation_target"]["status"] == "NOT_CERTIFIED"


def test_magnetic_curl_and_refinement_scope(system):
    x = np.array([.3, -.7, 1.2]); eps = .001
    jac = np.column_stack([(np.array(system.magnetic_field(x+eps*np.eye(3)[i])["vector_potential"])-
                            np.array(system.magnetic_field(x-eps*np.eye(3)[i])["vector_potential"]))/ (2*eps)
                           for i in range(3)])
    curl = [jac[2, 1]-jac[1, 2], jac[0, 2]-jac[2, 0], jac[1, 0]-jac[0, 1]]
    np.testing.assert_allclose(curl, system.magnetic_field(x)["magnetic"], atol=1e-12)
    result = system.magnetic_continuum(4, steps=64)
    assert len(result["refinements"]) == len(result["finite_pulses"]) == 1
    assert result["refinements"][0]["tetrahedra"] == 1280
    assert len(result["finite_pulses"][0]["final_state_real"]) == 110
    assert result["finite_pulses"][0]["n"] == 2
    assert "not Ritz" in result["finite_pulses"][0]["initialization"]
    assert result["finite_pulse_is_continuum_error_certificate"] is False
    assert result["computed_C_T"] is None
    assert result["self_consistent_maxwell_backreaction"] is False
    assert result["analytic_scope"]["formalized_in_lean"] is False


def test_query_results_are_independent_copies(system):
    first = system.packet(); first["seed_center"][30] = 123
    first["orthonormal_coulomb_frame"][0][0] = 123
    first["seed_covariance"]["position_diagonal"][0] = 123
    first["contract"]["quantum_propagation_certified"] = True
    first["provenance"]["receipt_sha256"]["quantum_packet"] = "fake"
    second = system.packet()
    assert second["seed_center"][30] == 1
    assert second["orthonormal_coulomb_frame"][0][0] != 123
    assert second["seed_covariance"]["position_diagonal"][0] == .25
    assert second["contract"]["quantum_propagation_certified"] is False
    data = system.magnetic_continuum(); data["finite_pulses"][0]["final_state_real"][0] = 123
    data["analytic_scope"]["self_consistent_maxwell_backreaction"] = True
    assert system.magnetic_continuum()["finite_pulses"][0]["final_state_real"][0] != 123
    assert system.magnetic_continuum()["analytic_scope"]["self_consistent_maxwell_backreaction"] is False


@pytest.mark.parametrize("bad", [-1, 2, True, 0., "0"])
def test_bad_preparation_index(system, bad):
    with pytest.raises(ValueError): system.packet(bad)


@pytest.mark.parametrize("bad", [0, .5, True, "0.5", "1/8", None])
def test_bad_width(system, bad):
    with pytest.raises(ValueError): system.packet(width=bad)


@pytest.mark.parametrize("bad", [[0.]*55, [0.]*57, [float("nan")]*56, [True]*56, [1j]*56, ["0"]*56, [1e308]*56])
def test_bad_amplitude_point(system, bad):
    with pytest.raises(ValueError): system.packet_half_density(bad)


@pytest.mark.parametrize("bad", [0, True, 128., 127, 1024])
def test_bad_circle_order(system, bad):
    with pytest.raises(ValueError): system.packet_half_density([0.]*56, nodes=bad)


@pytest.mark.parametrize("bad", [True, float("inf"), float("nan"), 1j, "0"])
def test_bad_angle(system, bad):
    with pytest.raises(ValueError): system.packet_overlap(bad)


@pytest.mark.parametrize("bad", [True, 0, 3, 1.0, "2"])
def test_bad_magnetic_refinement(system, bad):
    with pytest.raises(ValueError): system.magnetic_continuum(bad)


@pytest.mark.parametrize("bad", [True, 0, 32.0, 128])
def test_bad_magnetic_time_steps(system, bad):
    with pytest.raises(ValueError): system.magnetic_continuum(steps=bad)


@pytest.mark.parametrize("bad", [[1, 2], [1, 2, float("inf")], [True, 0, 0], [1j, 0, 0]])
def test_bad_magnetic_point(system, bad):
    with pytest.raises(ValueError): system.magnetic_field(bad)


@pytest.mark.parametrize("name", ["whitney_quantum_packet_receipt.json", "whitney_magnetic_continuum_receipt.json",
    "WHITNEY_QUANTUM_PACKET.tex", "MagneticMatterStability.lean", "whitney_charged_dynamics.py"])
def test_receipt_and_transitive_source_byte_mutations_rejected(system, monkeypatch, name):
    read = bridges._read_bytes
    monkeypatch.setattr(bridges, "_read_bytes", lambda p: read(p)+b" " if p.name == name else read(p))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ControlledBridges.load(RER)


def repin_packet(monkeypatch, mutate, extra_reads=None, stem="quantum_packet"):
    """Exercise defense-in-depth after deliberately overriding the test-only top pin."""
    name = f"whitney_{stem}_receipt.json"
    read = bridges._read_bytes
    packet = json.loads(read(RER/"code/electromagnetism/runtime"/name))
    mutate(packet)
    raw = json.dumps(packet, allow_nan=False).encode()
    monkeypatch.setitem(bridges.RECEIPT_PINS, stem, hashlib.sha256(raw).hexdigest())
    extra_reads = extra_reads or {}
    def altered(path):
        if path.name == name: return raw
        if path.name in extra_reads: return extra_reads[path.name]
        return read(path)
    monkeypatch.setattr(bridges, "_read_bytes", altered)


@pytest.mark.parametrize("mutation", [
    lambda d: d["contract"].__setitem__("quantum_propagation_certified", True),
    lambda d: d["contract"].__setitem__("configuration_dimension", 56.0),
    lambda d: d["contract"]["requested_propagation_target"].__setitem__("residual_upper_bound", .01),
    lambda d: d["samples"][0]["momentum"].__setitem__(43, 1),
    lambda d: d["samples"][0]["widths"][0].__setitem__("scalar_radius_numeric", 1),
    lambda d: d["source_pins"].__setitem__("../outside.py", "0"*64),
    lambda d: d["source_pins"].__setitem__("/tmp/outside.py", "0"*64),
    lambda d: d["source_pins"].__setitem__("a\\outside.py", "0"*64),
    lambda d: d["source_pins"].__setitem__("./outside.py", "0"*64),
])
def test_resealed_scope_algebra_and_path_mutations_rejected(system, monkeypatch, mutation):
    repin_packet(monkeypatch, mutation)
    with pytest.raises(ValueError): ControlledBridges.load(RER)


def test_authenticated_source_is_never_executed(system, monkeypatch):
    source_name = "whitney_quantum_packet.py"
    malicious = b'raise RuntimeError("query loader executed research code")\n'
    repin_packet(monkeypatch, lambda d: d["source_pins"].__setitem__(
        f"code/electromagnetism/{source_name}", hashlib.sha256(malicious).hexdigest()), {source_name: malicious})
    loaded = ControlledBridges.load(RER)
    assert loaded.describe()["provenance"]["source_code_executed"] is False
    assert "whitney_quantum_packet" not in sys.modules


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e9999}', b'{"x":Infinity}', b'[]'])
def test_strict_json_loader(raw):
    with pytest.raises(ValueError): bridges._strict_json(raw)


@pytest.mark.parametrize("arguments,key", [(["bridges"], "quantum_scope"),
    (["packet", "--sample", "1", "--width", "1/4"], "seed_cotangent"),
    (["packet-overlap", ".2"], "overlap"),
    (["magnetic-continuum", "--refinement", "2", "--steps", "32"], "refinements"),
    (["magnetic-field", "1", "2", "3"], "vector_potential"),
    (["charged-enclosure", "1/80"], "canonical_coordinate_bounds_exact")])
def test_new_cli_does_not_load_other_history(system, monkeypatch, capsys, arguments, key):
    from oph_fpe.physical_systems import __main__ as cli
    monkeypatch.setattr(cli.WhitneySystem, "load", lambda *_: pytest.fail("unrelated old history loaded"))
    assert main(["--rer-root", str(RER), *arguments]) == 0
    result = json.loads(capsys.readouterr().out)
    assert key in result
    assert result["provenance"]["physical_measurement_or_prediction"] is False


@pytest.mark.parametrize("time,index", [("0", 0), ("1/80", 0), ("1/40", 1), ("39/40", 39), ("2", 79)])
def test_exact_enclosure_evaluation_against_independent_power_sum(system, time, index):
    result = system.charged_enclosure(time)
    source = system._packets["charged_enclosure"]["states"][index]
    local = Fraction(time)-Fraction(index, 40)
    # Direct powers/summation are separate from the production Horner recurrence.
    expected = [sum((Fraction(int(c), 2**96)*local**k for k, c in enumerate(row)), Fraction(0))
                for row in source["polynomial"]]
    actual = list(map(Fraction, result["canonical_polynomial_exact"]))
    assert actual == expected
    assert result["piece_index"] == index
    error = Fraction(1, 10**20)
    for value, bounds in zip(expected, result["canonical_coordinate_bounds_exact"], strict=True):
        assert list(map(Fraction, bounds)) == [value-error, value+error]
    assert result["interval_proof_imported"] is True
    assert result["interval_proof_reexecuted"] is False
    assert result["same_coordinates_as_quantum_packet"] is False
    assert result["raw_observer_state_joined"] is False
    assert result["nonlinear_field_readout_error_certified"] is False
    assert result["interpretation"]["quantum_history"] is False
    assert result["interpretation"]["spatial_continuum_error_certified"] is False


def test_exact_initial_state_is_inside_imported_enclosure(system):
    initial = [0, 1, 0, 1, 0, 0, Fraction(3, 10), 0, -Fraction(3, 10)]
    bounds = system.charged_enclosure(0)["canonical_coordinate_bounds_exact"]
    assert all(Fraction(pair[0]) <= value <= Fraction(pair[1]) for value, pair in zip(initial, bounds, strict=True))
    assert system.charged_enclosure(Fraction(1, 80)) == system.charged_enclosure("1/80")


def test_both_polynomial_pieces_at_every_shared_boundary_fit_the_bound(system):
    states = system._packets["charged_enclosure"]["states"]
    step, error = Fraction(1, 40), Fraction(1, 10**20)
    for boundary in range(1, 80):
        for left, right in zip(states[boundary-1]["polynomial"], states[boundary]["polynomial"], strict=True):
            left_endpoint = sum((Fraction(int(c), 2**96)*step**k for k, c in enumerate(left)), Fraction(0))
            right_endpoint = Fraction(int(right[0]), 2**96)
            assert abs(left_endpoint-right_endpoint) <= 2*error


@pytest.mark.parametrize("bad", [True, .025, float("nan"), "01", "2/4", "1e-2", "-1", "3", "1/0", "0/1",
    Fraction(-1, 2), Fraction(2001, 1000), Fraction(1, 2**257), None])
def test_invalid_or_ambiguous_enclosure_time(system, bad):
    with pytest.raises(ValueError): system.charged_enclosure(bad)


@pytest.mark.parametrize("mutation", [
    lambda d: d["interpretation"].__setitem__("observer_history", True),
    lambda d: d["interpretation"].__setitem__("quantum_history", True),
    lambda d: d["bounds"].__setitem__("uniform_canonical_polynomial_error_upper", "1/1000000000000000000000"),
    lambda d: d["method"].__setitem__("polynomial_bits", 96.0),
    lambda d: d["states"][0]["polynomial"][0].__setitem__(0, "+1"),
    lambda d: d["states"][0]["polynomial"][0].__setitem__(0, 1),
    lambda d: d["states"][0]["polynomial"][0].pop(),
])
def test_resealed_enclosure_scope_and_encoding_mutations_rejected(system, monkeypatch, mutation):
    repin_packet(monkeypatch, mutation, stem="charged_enclosure")
    with pytest.raises(ValueError): ControlledBridges.load(RER)


def test_enclosure_query_copy_isolation(system):
    first = system.charged_enclosure("1/80")
    first["method"]["coordinate_order"][0] = "wrong"
    first["interpretation"]["observer_history"] = True
    first["canonical_coordinate_bounds_exact"][0][0] = "100"
    later = system.charged_enclosure("1/80")
    assert later["method"]["coordinate_order"][0] == "alpha"
    assert later["interpretation"]["observer_history"] is False
    assert Fraction(later["canonical_coordinate_bounds_exact"][0][0]) < 1
