"""Issue-573 producer: geometric modular normalization and cap-interior data.

The Einstein branch consumes a geometric modular normalization: the statement
that the cap-interior modular Hamiltonian of the physical cap state equals
``2*pi`` times the geometrically transported local generator.  A state that is
generated at a prescribed inverse temperature and then regressed against that
same temperature certifies nothing; issue #573 demands independently produced
sides, frozen before comparison, with named negative controls.

This module produces both sides from one frozen capture:

* Geometric side.  The capture's ``geometry_raw_primitives`` rows carry a
  declared flow parameter together with a held-out four-point cross ratio.
  The geometric rate is the fitted slope of ``log(cross_ratio)`` against the
  declared parameter; the ordered frame and orientation fields fix the framing
  and its sign.  No state data enters.
* State side.  Empirical cap states are built from the record-state snapshot
  rows: the sampled twelve-port state vectors are projected onto framed
  four-dimensional cap subspaces (one per icosahedral antipodal axis, framing
  fixed by port incidence alone), and the second-moment density matrix is
  normalized without any temperature parameter.  The cap-interior modular
  Hamiltonian is ``-log(rho)``.  No geometric flow value enters.

Both sides are hashed and frozen before the relative normalization is fitted.
The fitted normalization is compared against the Bisognano--Wichmann value
``2*pi`` with two preregistered wrong-normalization bands (``pi`` and
``4*pi``), on two refinement configurations, and under five negative controls
(wrong frame, reversed orientation, nonfaithful state, truncated extraction,
permuted cap/state pairing).  The verdict is recorded fail-closed: this
instrument can and does report ``NOT_ATTAINED`` when the source dynamics does
not thermalize the framed cap states at the geometric temperature.  No
physical promotion follows from any output of this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import logm

from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source

SCHEMA = "oph.modular-normalization-producer.v1"
PHYSICAL_PROMOTION_ALLOWED = False
PORTS = 12
TWO_PI = 2.0 * np.pi
ACCEPTANCE_HALF_WIDTH = 0.15  # relative band around 2*pi
WRONG_BANDS = ((np.pi, 0.15), (4.0 * np.pi, 0.15))
FAITHFULNESS_FLOOR = 1.0e-9

_NEIGHBORS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 4, 6),
    (0, 2, 3, 5, 7),
    (0, 1, 4, 5, 8),
    (0, 1, 6, 7, 9),
    (0, 2, 6, 8, 10),
    (1, 2, 7, 8, 11),
    (0, 3, 4, 9, 10),
    (1, 3, 5, 9, 11),
    (2, 4, 5, 10, 11),
    (3, 6, 7, 10, 11),
    (4, 6, 8, 9, 11),
    (5, 7, 8, 9, 10),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _axis_frame(axis: int, *, reversed_orientation: bool = False) -> np.ndarray:
    """Orthonormal 12x4 frame for one antipodal axis, from incidence alone.

    Columns: the axis port, its antipode, and the two orthogonal uniform
    neighbor-ring combinations whose relative sign carries the orientation.
    """

    antipode = 11 - axis
    ring = _NEIGHBORS[axis]
    ring_anti = _NEIGHBORS[antipode]
    e_axis = np.zeros(PORTS)
    e_axis[axis] = 1.0
    e_anti = np.zeros(PORTS)
    e_anti[antipode] = 1.0
    ring_vec = np.zeros(PORTS)
    for port in ring:
        ring_vec[port] = 1.0
    ring_anti_vec = np.zeros(PORTS)
    for port in ring_anti:
        ring_anti_vec[port] = 1.0
    sign = -1.0 if reversed_orientation else 1.0
    symmetric = (ring_vec + ring_anti_vec) / np.sqrt(10.0)
    antisymmetric = sign * (ring_vec - ring_anti_vec) / np.sqrt(10.0)
    frame = np.stack((e_axis, e_anti, symmetric, antisymmetric), axis=1)
    # Gram-Schmidt is unnecessary: the four columns are orthonormal by
    # construction (disjoint supports except the two ring combinations, which
    # are orthogonal to each other and to the axis pair).
    return frame


def _depth_generator(axis: int, *, orientation_sign: float = 1.0) -> np.ndarray:
    """Geometrically transported local generator on the framed subspace.

    The modular depth grading of the cap assigns graph distance from the axis
    port: 0 to the axis, 3 to the antipode, 1 to the near ring, 2 to the far
    ring.  The flow additionally transports the symmetric ring combination
    toward the antisymmetric one, so the generator carries an off-diagonal
    coupling whose sign is the declared orientation.  Incidence and
    orientation data only; no state input.
    """

    depths = np.asarray([0.0, 3.0, 1.0, 2.0])
    matrix = np.diag(depths)
    matrix[2, 3] = 0.5 * orientation_sign
    matrix[3, 2] = 0.5 * orientation_sign
    return matrix - (np.trace(matrix) / 4.0) * np.eye(4)


def geometric_flow_rate(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Fit the geometric rate from declared parameters and held-out ratios."""

    rows = capture["source_artifacts"]["geometry_raw_primitives"]
    parameters = np.asarray([float(row["geometric_parameter"]) for row in rows])
    ratios = np.asarray([float(row["cross_ratio"]) for row in rows])
    orientations = {str(row["orientation"]) for row in rows}
    if len(rows) < 3:
        return {"rate": None, "residual": None, "blocker": "too_few_geometry_rows"}
    if ratios.min() <= 0.0:
        return {"rate": None, "residual": None, "blocker": "nonpositive_cross_ratio"}
    design = np.stack((parameters, np.ones_like(parameters)), axis=1)
    solution, *_ = np.linalg.lstsq(design, np.log(ratios), rcond=None)
    fitted = design @ solution
    residual = float(np.max(np.abs(fitted - np.log(ratios))))
    return {
        "rate": float(solution[0]),
        "intercept": float(solution[1]),
        "residual": residual,
        "orientations": sorted(orientations),
        "row_count": len(rows),
        "blocker": None,
    }


def _snapshot_samples(capture: Mapping[str, Any]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for snapshot in capture["source_artifacts"]["dynamics"]["record_state_snapshots"]:
        for row in snapshot["carrier_rows"]:
            vectors.append(np.asarray(row["full_port_state"], dtype=float))
    return np.stack(vectors, axis=0)


def cap_interior_state(
    samples: np.ndarray,
    frame: np.ndarray,
    *,
    regularizer: float = 1.0e-6,
) -> dict[str, Any]:
    """Empirical faithful cap state and its interior modular Hamiltonian.

    The second-moment matrix of the framed samples is normalized to unit
    trace.  No temperature enters; faithfulness is checked, not assumed.
    """

    framed = samples @ frame
    moment = framed.T @ framed / framed.shape[0]
    moment = moment + regularizer * np.eye(4)
    rho = moment / np.trace(moment)
    eigenvalues = np.linalg.eigvalsh(rho)
    faithful = bool(eigenvalues.min() > FAITHFULNESS_FLOOR)
    if not faithful:
        return {"faithful": False, "rho": rho, "modular_hamiltonian": None}
    k_matrix = -np.real(logm(rho))
    k_matrix = k_matrix - (np.trace(k_matrix) / 4.0) * np.eye(4)
    return {"faithful": True, "rho": rho, "modular_hamiltonian": k_matrix}


def _alignment(k_matrix: np.ndarray, generator: np.ndarray) -> dict[str, float]:
    inner = float(np.tensordot(k_matrix, generator))
    norm_g = float(np.tensordot(generator, generator))
    coefficient = inner / norm_g
    residual = k_matrix - coefficient * generator
    return {
        "coefficient": coefficient,
        "relative_residual": float(
            np.linalg.norm(residual) / max(np.linalg.norm(k_matrix), 1.0e-30)
        ),
    }


def _band(center: float, half_width_rel: float) -> tuple[float, float]:
    return (center * (1.0 - half_width_rel), center * (1.0 + half_width_rel))


def _interval(values: np.ndarray) -> tuple[float, float]:
    return (float(values.min()), float(values.max()))


def _contains(interval: tuple[float, float], band: tuple[float, float]) -> bool:
    return band[0] <= interval[0] and interval[1] <= band[1]


def _disjoint(interval: tuple[float, float], band: tuple[float, float]) -> bool:
    return interval[1] < band[0] or band[1] < interval[0]


def _axis_coefficients(
    capture: Mapping[str, Any],
    *,
    reversed_orientation: bool = False,
    permuted_pairing: bool = False,
    truncate_snapshots: bool = False,
    wrong_frame: bool = False,
    zero_regularizer: bool = False,
) -> dict[str, Any]:
    samples = _snapshot_samples(capture)
    if truncate_snapshots:
        samples = samples[:2]
    axes = tuple(range(6))
    coefficients: list[float] = []
    residuals: list[float] = []
    nonfaithful = 0
    for axis in axes:
        frame_axis = axis
        if wrong_frame:
            # A non-antipodal pairing: replace the antipode column source by a
            # neighbor, breaking the icosahedral framing contract.
            frame = _axis_frame(axis)
            neighbor = _NEIGHBORS[axis][0]
            broken = frame.copy()
            broken[:, 1] = 0.0
            broken[neighbor, 1] = 1.0
            frame = broken
        else:
            # The reversed-orientation control desynchronizes the declared
            # orientation from the frame: the frame keeps the source
            # orientation while the transported generator below flips sign.
            # Reversing both together is a symmetry and would hide the error.
            frame = _axis_frame(frame_axis)
        state_samples = samples[:1] if zero_regularizer else samples
        state = cap_interior_state(
            state_samples,
            frame,
            regularizer=0.0 if zero_regularizer else 1.0e-6,
        )
        if not state["faithful"]:
            nonfaithful += 1
            continue
        generator_axis = (axis + 1) % 6 if permuted_pairing else axis
        del generator_axis  # the depth generator is axis-uniform by design;
        # permuted pairing instead permutes the framed state below.
        generator = _depth_generator(
            axis, orientation_sign=-1.0 if reversed_orientation else 1.0
        )
        k_matrix = state["modular_hamiltonian"]
        if permuted_pairing:
            permutation = np.asarray([1, 2, 3, 0])
            k_matrix = k_matrix[np.ix_(permutation, permutation)]
        result = _alignment(k_matrix, generator)
        coefficients.append(result["coefficient"])
        residuals.append(result["relative_residual"])
    return {
        "coefficients": np.asarray(coefficients),
        "relative_residuals": np.asarray(residuals),
        "nonfaithful_axes": nonfaithful,
        "axis_count": len(axes),
    }


def produce_modular_normalization_report(
    *,
    config: Mapping[str, Any] | None = None,
    refinement_config: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Produce the frozen two-sided normalization report with controls."""

    main_config = dict(
        {"carrier_count": 32, "cycles": 6, "seed": 20260751}
        if config is None
        else config
    )
    fine_config = dict(
        {"carrier_count": 64, "cycles": 6, "seed": 20260751}
        if refinement_config is None
        else refinement_config
    )
    capture = capture_physical_source(main_config)
    capture_fine = capture_physical_source(fine_config)

    geometry = geometric_flow_rate(capture)
    geometry_fine = geometric_flow_rate(capture_fine)
    samples = _snapshot_samples(capture)
    state_freeze = _sha256_value(samples.tolist())
    geometry_freeze = _sha256_value(
        [geometry.get("rate"), geometry.get("intercept"), geometry.get("residual")]
    )

    main = _axis_coefficients(capture)
    fine = _axis_coefficients(capture_fine)
    blockers: list[str] = []
    if geometry["blocker"] is not None:
        blockers.append(f"geometry:{geometry['blocker']}")
    if main["nonfaithful_axes"]:
        blockers.append("nonfaithful_axes_in_main_family")
    if main["coefficients"].size == 0:
        blockers.append("no_faithful_cap_states")

    verdict = "NOT_ATTAINED"
    normalization_interval = None
    wrong_band_exclusions = None
    refinement_stable = None
    if not blockers and geometry["rate"]:
        # Relative normalization: cap-interior coefficient divided by the
        # frozen geometric rate.  The Bisognano--Wichmann acceptance band and
        # both preregistered wrong bands are evaluated on the interval over
        # the nondegenerate axis family.
        normalized = main["coefficients"] / geometry["rate"]
        normalized_fine = (
            fine["coefficients"] / geometry_fine["rate"]
            if geometry_fine["blocker"] is None and geometry_fine["rate"]
            else None
        )
        normalization_interval = _interval(normalized)
        acceptance = _band(TWO_PI, ACCEPTANCE_HALF_WIDTH)
        wrong_band_exclusions = [
            _disjoint(normalization_interval, _band(center, width))
            for center, width in WRONG_BANDS
        ]
        contained = _contains(normalization_interval, acceptance)
        if normalized_fine is not None and normalized_fine.size:
            fine_interval = _interval(normalized_fine)
            refinement_stable = bool(
                _contains(fine_interval, acceptance) == contained
                and all(
                    _disjoint(fine_interval, _band(center, width)) == excluded
                    for (center, width), excluded in zip(
                        WRONG_BANDS, wrong_band_exclusions
                    )
                )
            )
        if contained and all(wrong_band_exclusions) and refinement_stable:
            verdict = "ATTAINED"
        else:
            if not contained:
                blockers.append("normalization_interval_outside_acceptance_band")
            if not all(wrong_band_exclusions):
                blockers.append("wrong_normalization_band_not_excluded")
            if refinement_stable is not True:
                blockers.append("refinement_stability_not_established")

    controls = {
        "wrong_frame": _axis_coefficients(capture, wrong_frame=True),
        "reversed_orientation": _axis_coefficients(
            capture, reversed_orientation=True
        ),
        "nonfaithful_state": _axis_coefficients(capture, zero_regularizer=True),
        "truncated_extraction": _axis_coefficients(
            capture, truncate_snapshots=True
        ),
        "permuted_cap_state": _axis_coefficients(capture, permuted_pairing=True),
    }
    control_rows: dict[str, dict[str, Any]] = {}
    controls_fail_closed = True
    for name, result in controls.items():
        if name == "nonfaithful_state":
            failed = result["nonfaithful_axes"] > 0
        elif result["coefficients"].size == 0:
            failed = True
        else:
            control_interval = _interval(
                result["coefficients"]
                / (geometry["rate"] if geometry["rate"] else 1.0)
            )
            deviated = (
                normalization_interval is None
                or not np.allclose(
                    control_interval, normalization_interval, rtol=1.0e-12
                )
            )
            worse_alignment = bool(
                result["relative_residuals"].size
                and main["relative_residuals"].size
                and result["relative_residuals"].mean()
                > main["relative_residuals"].mean() + 1.0e-12
            )
            failed = bool(deviated or worse_alignment)
        control_rows[name] = {
            "control_failure_detected": bool(failed),
            "nonfaithful_axes": int(result["nonfaithful_axes"]),
        }
        controls_fail_closed = controls_fail_closed and bool(failed)
    if not controls_fail_closed:
        blockers.append("negative_control_did_not_fail")
        verdict = "NOT_ATTAINED"

    report = {
        "schema": SCHEMA,
        "issue": 573,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "main_config": main_config,
        "refinement_config": fine_config,
        "capture_sha256": capture["capture_sha256"],
        "refinement_capture_sha256": capture_fine["capture_sha256"],
        "state_side_freeze_sha256": state_freeze,
        "geometry_side_freeze_sha256": geometry_freeze,
        "geometric_rate": geometry.get("rate"),
        "geometric_fit_residual": geometry.get("residual"),
        "cap_family_axis_count": int(main["axis_count"]),
        "faithful_axis_count": int(
            main["axis_count"] - main["nonfaithful_axes"]
        ),
        "alignment_relative_residuals": main["relative_residuals"].tolist(),
        "normalization_interval": normalization_interval,
        "bw_acceptance_band": _band(TWO_PI, ACCEPTANCE_HALF_WIDTH),
        "preregistered_wrong_bands": [
            _band(center, width) for center, width in WRONG_BANDS
        ],
        "wrong_band_exclusions": wrong_band_exclusions,
        "refinement_stable": refinement_stable,
        "negative_controls": control_rows,
        "controls_fail_closed": bool(controls_fail_closed),
        "verdict": verdict,
        "GEOMETRIC_MODULAR_NORMALIZATION_RECEIPT": bool(verdict == "ATTAINED"),
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "Finite issue-573 instrument on the framed icosahedral cap family. "
            "Both sides are produced independently and frozen before the fit; "
            "the acceptance and wrong-normalization bands are preregistered. "
            "A NOT_ATTAINED verdict is a fail-closed empirical result about "
            "this source dynamics at this cutoff, not a falsification of the "
            "conditional Einstein-branch theorem. An ATTAINED verdict on this "
            "family is a finite diagnostic and licenses no physical promotion; "
            "the continuum cap-interior tail, the null-net handoff, and the "
            "paper-grade cap algebra remain open."
        ),
    }
    if output_path is not None:
        Path(output_path).write_text(_canonical_json(report), encoding="utf-8")
    return report
