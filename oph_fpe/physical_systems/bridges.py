"""Authenticated queries on controlled continuum and quantum preparation data.

Loading verifies receipt and transitive source hashes. It executes no RER source,
reruns no trajectory, and does not numerically prove the analytic theorems. The
local Gaussian formulas use all 56 configuration coordinates and the actual
recorded Coulomb cotangent; the interacting kinetic metric is not flattened.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import math
from numbers import Real
from pathlib import Path, PurePosixPath
import re

import numpy as np
from scipy.special import i0e, i1e

from .whitney import _exact, _index, _read_bytes, _require, _revision, _strict_json

# Immutable research commit containing the three byte-pinned bridge packages.
RER_COMMIT = "1443e0ad904ddd3181b5b910e764bcad2984909d"
RECEIPT_PINS = {
    "quantum_packet": "c4f8efdc1d17ce5a25d6196c8e98b426af5907fbf49b5450df457e01ea3c0b5a",
    "magnetic_continuum": "9458a6e50367f3643e3f87e8254f977073cafde492e0d846caaacb461cd678e7",
    "charged_enclosure": "88cdc380032628e57e4f452ec5e6762a0da156043b1b3db466742cff74694e54",
}
WIDTHS = ("1/4", "1/2", "1")


def _vector(value, size: int, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    _require(raw.shape == (size,) and all(isinstance(x, Real) and
        not isinstance(x, (bool, np.bool_)) for x in raw), f"{name} requires {size} real coordinates")
    result = np.asarray(raw, dtype=float)
    _require(np.isfinite(result).all(), f"{name} must be finite")
    return result


def _angle(value) -> float:
    _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)), "real angle required")
    result = float(value)
    _require(math.isfinite(result), "finite angle required")
    return result


def _rotate(vector: np.ndarray, angles: np.ndarray) -> np.ndarray:
    result = np.broadcast_to(vector, (len(angles), 56)).copy()
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    result[:, 30:43] = c*vector[30:43]-s*vector[43:]
    result[:, 43:] = s*vector[30:43]+c*vector[43:]
    return result


def _overlap_data(q: np.ndarray, p: np.ndarray, sigma: float) -> tuple[float, float, float, float]:
    """hbar=1: exact Gaussian identities evaluated in floating arithmetic."""
    A = float(q[30:]@q[30:]/(4*sigma**2)+sigma**2*(p[30:]@p[30:]))
    B = float(p[43:]@q[30:43]-p[30:43]@q[43:])
    _require(A >= abs(B)-1e-12 and math.isfinite(A), "Gaussian overlap parameters")
    z = math.sqrt(max(0., A*A-B*B))
    norm2 = math.exp(z-A)*float(i0e(z))
    radius = (26*sigma**2+float(q[30:]@q[30:])/2-
              2*sigma**4*float(p[30:]@p[30:])+2*sigma**2*z*float(i1e(z)/i0e(z)))
    _require(norm2 > 0 and all(math.isfinite(x) for x in (A, B, norm2, radius)), "finite Gaussian readout required")
    return A, B, norm2, radius


class ControlledBridges:
    """Read-only access to three supplied mathematical bridge packages.

    Use ``load``; query results contain independent copies. The packet samples
    are separate preparations centered on classical data, not quantum times.
    """

    @classmethod
    def load(cls, rer_root: str | Path | None = None) -> "ControlledBridges":
        root = (Path(rer_root) if rer_root is not None else
                Path(__file__).resolve().parents[3]/"reverse-engineering-reality").resolve()
        _revision(root)
        authenticated: dict[str, str] = {}
        packets = {}

        def authenticate(relative, digest):
            _require(type(relative) is str and type(digest) is str and
                re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "malformed source pin")
            path = PurePosixPath(relative)
            _require(not path.is_absolute() and str(path) == relative and
                ".." not in path.parts and "\\" not in relative, "unsafe source pin path")
            resolved = (root/relative).resolve()
            _require(resolved.is_relative_to(root), "source pin escapes checkout")
            if relative in authenticated:
                _require(authenticated[relative] == digest, "conflicting source pin")
                return None
            raw = _read_bytes(resolved)
            _require(hashlib.sha256(raw).hexdigest() == digest, f"SHA256 mismatch: {relative}")
            authenticated[relative] = digest
            if path.suffix == ".json":
                packet = _strict_json(raw)
                if "source_pins" in packet:
                    _require(type(packet["source_pins"]) is dict, "source pins must be an object")
                    for child, child_digest in packet["source_pins"].items():
                        authenticate(child, child_digest)
                return packet
            return None

        for name, digest in RECEIPT_PINS.items():
            packets[name] = authenticate(f"code/electromagnetism/runtime/whitney_{name}_receipt.json", digest)
        self = cls.__new__(cls)
        self._packets, self._source_hashes = packets, authenticated
        self._check_contracts()
        self._check_packet_algebra()
        self._check_enclosure()
        return self

    def _check_contracts(self):
        q, m = self._packets["quantum_packet"], self._packets["magnetic_continuum"]
        _exact(q["schema"], "oph.whitney_quantum_packet.v1", "packet schema")
        _exact(q["scope"], "FULL_56D_NEUTRAL_PACKET_PREPARATION__NO_QUANTUM_PROPAGATION_CERTIFICATE", "packet scope")
        _exact(q["parameters"], {"e": "1/4", "g": "1/4", "hbar": "1", "m_squared": "1/2"}, "packet parameters")
        for key, value in {"configuration_dimension": 56, "radiative_dimension": 30,
                "scalar_real_dimension": 26, "full_56D_preparation": True,
                "neutral_strong_domain_state": True, "quantum_propagation_certified": False,
                "covariance_evolution_computed": False, "classical_samples_are_quantum_means": False,
                "physical_comparison": False, "observer_preparation": False,
                "empirical_prediction": False}.items():
            _exact(q["contract"][key], value, f"packet contract: {key}")
        _exact(q["contract"]["requested_propagation_target"], {"T": "1/40", "hbar": "1", "norm_error": "1/10",
            "status": "NOT_CERTIFIED", "residual_upper_bound": None, "all_tail_bound": None}, "propagation boundary")
        _require(type(q["samples"]) is list and len(q["samples"]) == 2, "two packet preparations required")
        _exact(m["schema"], "oph.whitney_magnetic_continuum.v1", "magnetic schema")
        _exact(m["scope"], "FIXED_UNIFORM_MAGNETIC_BACKGROUND__CONDITIONAL_COMPLEX_MATTER_CONTINUUM__FINITE_NUMERICAL_CHECKS", "magnetic scope")
        _exact(m["parameters"], {"B": ["0", "0", "1"], "c": ["0", "0", "0"], "e": "1/4", "g": "1/4", "m2": "1/2"}, "magnetic parameters")
        _exact(m["analytic_scope"], {"conditional_complex_trajectory_bound": True,
            "conforming_magnetic_ritz_paper_proof": True, "continuum_reference_existence_assumed": True,
            "external_magnetic_background_fixed": True, "formalized_in_lean": False,
            "numerical_error_interval_certified": False, "physical_source_or_clock_selected": False,
            "pointwise_cubic_bounds_formalized_in_lean": True, "self_consistent_maxwell_backreaction": False,
            "uniform_mesh_paper_proof": True}, "magnetic assumptions and boundaries")
        _exact([row["n"] for row in m["mesh_checks"]], [1, 2, 4], "magnetic refinement census")
        _exact([row["steps"] for row in m["trajectories"]], [32, 64], "magnetic step census")
        for row in m["trajectories"]:
            _exact(row["initialization"], "nodal compact complex pulse; not Ritz data for a supplied continuum solution", "pulse initialization")
            _exact(row["time_interval"], ["0", "1/8"], "magnetic recorded interval")

    def _check_packet_algebra(self):
        for index, row in enumerate(self._packets["quantum_packet"]["samples"]):
            _exact(row["parent_sample_index"], index, "classical sample identity")
            _exact(row["model_time"], index/40, "classical sample time")
            q, p = _vector(row["q"], 56, "center"), _vector(row["momentum"], 56, "cotangent")
            _vector(row["velocity"], 56, "Coulomb velocity")
            _exact([w["sigma"] for w in row["widths"]], list(WIDTHS), "declared packet widths")
            for width in row["widths"]:
                actual = _overlap_data(q, p, float(Fraction(width["sigma"])))
                expected = [width[k] for k in ("A", "B", "norm_squared", "scalar_radius_numeric")]
                _require(np.allclose(actual, expected, rtol=3e-12, atol=3e-12), "Gaussian overlap or moment mismatch")

    def _check_enclosure(self):
        packet = self._packets["charged_enclosure"]
        _exact(packet["schema"], "oph.whitney_charged_enclosure.v1", "enclosure schema")
        _exact(packet["scope"], "RIGOROUS_FINITE_CHARGED_ACTION_TIME_TUBE__SUPPLIED_MODEL_PARAMETER", "enclosure scope")
        for key, value in {"steps": 80, "step": "1/40", "time_span": ["0", "2"],
                "taylor_order": 32, "interval_bits": 256, "polynomial_bits": 96,
                "coordinate_order": ["alpha", "Re(C)", "Im(C)", "Re(b)", "Im(b)",
                    "p_Re(C)", "p_Im(C)", "p_Re(b)", "p_Im(b)"],
                "error_norm": "maximum absolute canonical coordinate difference",
                "approximant": "piecewise polynomial; adjacent polynomial endpoints need not agree"}.items():
            _exact(packet["method"][key], value, f"enclosure method: {key}")
        _exact(packet["bounds"], {"uniform_canonical_polynomial_error_upper": "1/100000000000000000000",
            "historical_position_velocity_error_upper": "1/10000000000",
            "normalized_denominator_lower_bound": "2/5"}, "enclosure error bounds")
        for key, value in {"rigorous_time_enclosure": True, "whole_time_interval": True,
                "physical_clock_selected": False, "spatial_continuum_error_certified": False,
                "observer_history": False, "quantum_history": False, "empirical_comparison": False,
                "formalized_in_Lean": False}.items():
            _exact(packet["interpretation"][key], value, f"enclosure scope: {key}")
        _require(type(packet["states"]) is list and len(packet["states"]) == 81, "enclosure state census")
        for i, row in enumerate(packet["states"]):
            _require(type(row) is dict and type(row["state"]) is list and len(row["state"]) == 9, "enclosure state layout")
            for pair in row["state"]:
                _require(type(pair) is list and len(pair) == 2, "interval endpoint pair")
                lo, hi = (self._integer(x) for x in pair)
                _require(lo <= hi, "reversed enclosure interval")
            if i < 80:
                _require(type(row["polynomial"]) is list and len(row["polynomial"]) == 9, "polynomial coordinate census")
                for coefficients in row["polynomial"]:
                    _require(type(coefficients) is list and len(coefficients) == 33, "degree32 polynomial coefficient census")
                    for x in coefficients: self._integer(x)

    @staticmethod
    def _integer(value):
        _require(type(value) is str and len(value) <= 100 and
            re.fullmatch(r"0|-?[1-9][0-9]*", value) is not None, "canonical integer coefficient required")
        return int(value)

    def _provenance(self):
        return {"rer_commit": RER_COMMIT, "receipt_sha256": dict(RECEIPT_PINS),
            "authenticated_source_sha256": dict(self._source_hashes),
            "source_code_executed": False, "source_hashes_authenticated": True,
            "research_verifiers_reexecuted": False, "ode_reintegrated": False,
            "local_gaussian_algebra_checked": True, "analytic_theorems_proved_by_loader": False, "interval_certificate_reverified": False,
            "physical_measurement_or_prediction": False}

    def describe(self):
        return {"schema": "oph.controlled_bridge_queries.v1",
            "available_queries": ["packet", "packet_overlap", "packet_half_density", "magnetic_continuum", "magnetic_field", "charged_enclosure"],
            "quantum_scope": "two full56D neutral state preparations; no interacting quantum propagation certificate",
            "magnetic_scope": "prescribed magnetic background, conditional complex scalar continuum theorem and separate finite pulse checks",
            "enclosure_scope": "exact rational evaluation of an imported certified classical polynomial tube; supplied model time",
            "observer_relation": "packet centers come from a supplied charged model; its separate self-reading software patch records are queried through WhitneySystem",
            "same_quantum_classical_history": False, "physical_completion": False, "provenance": self._provenance()}

    def _packet(self, sample: int, width: str):
        rows = self._packets["quantum_packet"]["samples"]
        row = rows[_index(sample, len(rows), "preparation index")]
        _require(type(width) is str and width in WIDTHS, "width must be one of 1/4, 1/2, 1")
        return row, row["widths"][WIDTHS.index(width)], float(Fraction(width))

    def packet(self, sample: int = 0, width: str = "1/2"):
        row, selected, sigma = self._packet(sample, width)
        return {"schema": "oph.controlled_quantum_preparation.v1", "preparation_index": sample,
            "classical_center_model_time": row["model_time"], "classical_parent_sample_index": row["parent_sample_index"],
            "coordinate_order": "30 orthonormal Coulomb edge coordinates,13 Re(Psi),13 Im(Psi)",
            "orthonormal_coulomb_frame": deepcopy(self._packets["quantum_packet"]["orthonormal_coulomb_frame"]),
            "seed_center": deepcopy(row["q"]), "seed_cotangent": deepcopy(row["momentum"]),
            "coulomb_velocity": deepcopy(row["velocity"]),
            "coulomb_scalar_potential": deepcopy(row["transformed_scalar_potential"]),
            "seed_covariance": {"position_diagonal": [sigma**2]*56, "momentum_diagonal": [1/(4*sigma**2)]*56,
                "symmetrized_position_momentum": "zero", "off_diagonal": "zero", "rank": 56,
                "scope": "unprojected isotropic Gaussian; projected scalar covariance differs"},
            "neutral_projection": {**deepcopy(selected), "scalar_position_mean": [0.]*26,
                "scalar_canonical_momentum_mean": [0.]*26, "full_projected_covariance_computed": False,
                "scalar_radius_observable": "sum of 26 real nodal coordinate squares, not spatial field L2 norm",
                "numeric_values_are_interval_bounds": False,
                "initial_exact_projection_norm_squared_lower": "1/64" if sample == 0 else None},
            "state_measure": "L2(Q,rho dq), rho=sqrt(det(gamma)); query amplitudes use the equivalent Lebesgue half-density",
            "kinetic_operator": "declared variable-metric Laplace-Beltrami operator; no flat kinetic replacement",
            "contract": deepcopy(self._packets["quantum_packet"]["contract"]),
            "units": "supplied model coordinates and hbar=1; no physical preparation or calibration",
            "provenance": self._provenance()}

    def packet_overlap(self, angle: float, sample: int = 0, width: str = "1/2"):
        row, _, sigma = self._packet(sample, width)
        A, B, norm2, _ = _overlap_data(np.asarray(row["q"]), np.asarray(row["momentum"]), sigma)
        theta = _angle(angle)
        value = np.exp(-2*A*math.sin(theta/2)**2-1j*B*math.sin(theta))
        return {"angle": theta, "overlap": [float(value.real), float(value.imag)], "A": A, "B": B,
            "projection_norm_squared_numeric": norm2,
            "formula": "exp[-A(1-cos(theta))-i B sin(theta)]; squared projected norm is exp(-A) I0(sqrt(A^2-B^2))",
            "numerical_evaluation_not_interval_bound": True, "quantum_propagation": False,
            "provenance": self._provenance()}

    def packet_half_density(self, point, sample: int = 0, width: str = "1/2", nodes: int = 256):
        row, _, sigma = self._packet(sample, width)
        x = _vector(point, 56, "configuration point")
        _require(type(nodes) is int and nodes in (128, 256, 512), "circle nodes must be 128, 256 or 512")
        q, p = np.asarray(row["q"]), np.asarray(row["momentum"])
        _, _, norm2, _ = _overlap_data(q, p, sigma)

        def quadrature(count):
            theta = 2*np.pi*np.arange(count)/count
            centers, momenta = _rotate(q, theta), _rotate(p, theta)
            with np.errstate(over="ignore", invalid="ignore"):
                delta = x-centers
                exponent = -np.sum(delta*delta, axis=1)/(4*sigma**2)+1j*np.sum(momenta*delta, axis=1)
            _require(np.isfinite(exponent).all(), "configuration arithmetic overflow")
            return np.mean(np.exp(exponent))*(2*np.pi*sigma**2)**(-14)/math.sqrt(norm2)

        coarse, value = quadrature(nodes), quadrature(2*nodes)
        _require(np.isfinite(value), "nonfinite half-density amplitude")
        return {"preparation_index": sample, "width": width, "point": x.tolist(),
            "normalized_neutral_half_density": [float(value.real), float(value.imag)],
            "probability_density_relative_to_dq": float(abs(value)**2),
            "circle_nodes": 2*nodes, "absolute_quadrature_change": float(abs(value-coarse)),
            "certified_pointwise_error": None, "quantum_propagation": False,
            "representation": "sqrt(rho)*F in L2(R56,dq); rho and the interacting Hamiltonian are not replaced",
            "normalization_scope": "exact for the analytic circle integral; displayed finite quadrature is numerical",
            "provenance": self._provenance()}

    def magnetic_continuum(self, refinement: int | None = None, *, steps: int | None = None):
        packet = self._packets["magnetic_continuum"]
        if refinement is not None:
            _require(type(refinement) is int and refinement in (1, 2, 4), "refinement must be 1, 2 or 4")
        if steps is not None:
            _require(type(steps) is int and steps in (32, 64), "recorded time steps must be 32 or 64")
        return {"schema": "oph.controlled_magnetic_continuum_query.v1", "parameters": deepcopy(packet["parameters"]),
            "analytic_scope": deepcopy(packet["analytic_scope"]),
            "conditional_trajectory_estimate": "H1 field plus L2 velocity error <= C_T/n for supplied smooth magnetic-Neumann reference and magnetic Ritz initial data",
            "computed_C_T": None, "finite_pulse_is_continuum_error_certificate": False,
            "refinements": deepcopy([r for r in packet["mesh_checks"] if refinement is None or r["n"] == refinement]),
            "finite_pulses": deepcopy([r for r in packet["trajectories"] if steps is None or r["steps"] == steps]),
            "finite_pulse_layout": "each final_state_real/imag is concatenated nodal scalar and nodal velocity, in the source refinement ordering",
            "pulse_time_interpolation_available": False, "controls": deepcopy(packet["controls"]),
            "self_consistent_maxwell_backreaction": False, "units": "supplied model space, time, field and action units",
            "provenance": self._provenance()}

    def magnetic_field(self, position):
        x = _vector(position, 3, "position")
        params = self._packets["magnetic_continuum"]["parameters"]
        B = np.asarray([float(Fraction(v)) for v in params["B"]])
        c = np.asarray([float(Fraction(v)) for v in params["c"]])
        A = c+np.cross(B, x)/2
        _require(np.isfinite(A).all(), "field arithmetic overflow")
        return {"position": x.tolist(), "vector_potential": A.tolist(), "magnetic": B.tolist(),
            "electric": [0., 0., 0.], "scalar_potential": 0.,
            "scope": "prescribed affine-skew background; no matter field sampled or Maxwell equation solved",
            "provenance": self._provenance()}

    def charged_enclosure(self, time: str | int | Fraction = "0"):
        """Evaluate the authenticated polynomial exactly, importing its error proof.

        Floats are rejected: a requested rational time must be unambiguous.
        The output bounds the nine volume-normalized canonical coordinates,
        not arbitrary nonlinear field readouts or physical observations.
        """
        _require(type(time) in (str, int, Fraction), "exact rational time required; use a string or Fraction")
        if type(time) is str:
            _require(len(time) <= 128 and re.fullmatch(r"(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?", time) is not None,
                "canonical nonnegative rational time required")
            t = Fraction(time)
            _require(str(t) == time, "canonical rational time required")
        else:
            t = Fraction(time)
        _require(0 <= t <= 2, "certified model time lies in [0,2]")
        _require(t.denominator.bit_length() <= 256, "rational time denominator exceeds query budget")
        index = min(79, (40*t).numerator//(40*t).denominator)
        local = t-Fraction(index, 40)
        packet = self._packets["charged_enclosure"]
        error = Fraction(packet["bounds"]["uniform_canonical_polynomial_error_upper"])
        values = []
        for coefficients in packet["states"][index]["polynomial"]:
            value = Fraction(0)
            for coefficient in reversed(coefficients):
                value = value*local+Fraction(self._integer(coefficient), 1 << 96)
            values.append(value)
        return {"schema": "oph.controlled_charged_enclosure_query.v1", "model_time_exact": str(t),
            "piece_index": index, "piece_interval_exact": [str(Fraction(index, 40)), str(Fraction(index+1, 40))],
            "local_time_exact": str(local), "canonical_coordinate_order": deepcopy(packet["method"]["coordinate_order"]),
            "canonical_polynomial_exact": [str(x) for x in values],
            "canonical_coordinate_bounds_exact": [[str(x-error), str(x+error)] for x in values],
            "uniform_max_coordinate_error_upper_exact": str(error),
            "evaluation_arithmetic": "exact rational Horner evaluation; endpoints are exact rational strings",
            "boundary_convention": "right piece at interior boundaries, left piece at t=2; either adjacent piece has the imported bound",
            "canonical_map": packet["interpretation"]["canonical_map"],
            "method": deepcopy(packet["method"]), "interpretation": deepcopy(packet["interpretation"]),
            "interval_proof_imported": True, "interval_proof_reexecuted": False,
            "raw_observer_state_joined": False, "same_coordinates_as_quantum_packet": False,
            "nonlinear_field_readout_error_certified": False, "provenance": self._provenance()}
