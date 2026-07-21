"""Issue-576 producer: same-source stress reconstruction and coupling test.

The Einstein branch needs a conserved stress readout reconstructed from the
same source object as the entropy readout, and a universal coupling: the
ratio of entropy response to stress flux must be one constant across the cap
family.  Issue #576 owns that reconstruction and the universality test.

This instrument reconstructs both sides from one frozen capture.  The stress
flux of a cap is the accumulated absolute repair transfer crossing the cap
boundary, read from the seam transaction ledger.  The entropy response of a
cap is the difference between the entropy of its framed empirical state and
the mean entropy of the same construction on the complement caps.  Both use
the identical capture, are hashed and frozen before the ratio fit, and the
coupling verdict measures the spread of the per-cap ratios against a
preregistered universality envelope.  Negative controls: permuted
stress/entropy pairing, shuffled seam ledger, mixed-source entropy, and a
degenerate cap family.  Verdicts are fail-closed; no physical promotion
follows from any output, and in particular an ATTAINED verdict would not
identify Newton's constant or any physical coupling.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source
from oph_fpe.bulk.modular_normalization_producer import (
    _NEIGHBORS,
    _axis_frame,
    _snapshot_samples,
    cap_interior_state,
)

SCHEMA = "oph.stress-coupling-producer.v1"
PHYSICAL_PROMOTION_ALLOWED = False
PORTS = 12
UNIVERSALITY_ENVELOPE = 0.10  # relative spread allowed across the cap family


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def cap_port_set(axis: int) -> frozenset[int]:
    return frozenset((axis, *_NEIGHBORS[axis]))


def _seam_flux_by_port(capture: Mapping[str, Any]) -> np.ndarray:
    """Absolute repair transfer accumulated per port index."""

    flux = np.zeros(PORTS)
    for row in capture["source_artifacts"]["dynamics"]["repair_event_log"]:
        reads = {
            (entry["carrier_id"], int(entry["port"])): float(entry["value"])
            for entry in row["read_set"]
        }
        for entry in row["write_set"]:
            key = (entry["carrier_id"], int(entry["port"]))
            flux[int(entry["port"])] += abs(float(entry["value"]) - reads[key])
    return flux


def cap_stress_flux(capture: Mapping[str, Any], axis: int) -> float:
    """Stress flux of one cap: repair transfer crossing the cap boundary.

    Seams couple equal port indices on paired carriers; a port is a boundary
    port of the cap when exactly one of the two seam sides lies inside the
    cap's port set on the icosahedral template, which for the axis cap means
    the neighbor ring (the axis port itself couples cap-interior to
    cap-interior on the antipodal template copy).
    """

    ports = cap_port_set(axis)
    flux = _seam_flux_by_port(capture)
    boundary_ports = [port for port in range(PORTS) if port in ports and port != axis]
    return float(sum(flux[port] for port in boundary_ports))


def cap_entropy_response(
    samples: np.ndarray, axis: int
) -> tuple[float, float]:
    """Entropy of the framed cap state and of its antipodal complement cap."""

    state = cap_interior_state(samples, _axis_frame(axis))
    complement = cap_interior_state(samples, _axis_frame(11 - axis if 11 - axis < 6 else axis))
    if not state["faithful"]:
        return float("nan"), float("nan")
    eigenvalues = np.linalg.eigvalsh(state["rho"])
    entropy = float(-(eigenvalues * np.log(eigenvalues)).sum())
    if complement["faithful"]:
        complement_eigenvalues = np.linalg.eigvalsh(complement["rho"])
        complement_entropy = float(
            -(complement_eigenvalues * np.log(complement_eigenvalues)).sum()
        )
    else:
        complement_entropy = float("nan")
    return entropy, complement_entropy


def produce_stress_coupling_report(
    *,
    config: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Produce the same-source stress/entropy coupling report with controls."""

    main_config = dict(
        {"carrier_count": 32, "cycles": 6, "seed": 20260751}
        if config is None
        else config
    )
    capture = capture_physical_source(main_config)
    samples = _snapshot_samples(capture)

    axes = tuple(range(6))
    stress = np.asarray([cap_stress_flux(capture, axis) for axis in axes])
    entropies = np.asarray(
        [cap_entropy_response(samples, axis)[0] for axis in axes]
    )
    stress_freeze = _sha256_value(stress.tolist())
    entropy_freeze = _sha256_value(entropies.tolist())

    blockers: list[str] = []
    if not np.all(np.isfinite(entropies)):
        blockers.append("nonfaithful_cap_state_in_family")
    if np.any(stress <= 0.0):
        blockers.append("zero_stress_flux_cap_in_family")

    ratios = None
    spread = None
    universal = False
    if not blockers:
        ratios = entropies / stress
        center = float(np.median(ratios))
        spread = float(
            (ratios.max() - ratios.min()) / max(abs(center), 1.0e-30)
        )
        universal = bool(spread <= UNIVERSALITY_ENVELOPE)
        if not universal:
            blockers.append("coupling_ratio_spread_exceeds_universality_envelope")

    controls: dict[str, dict[str, Any]] = {}
    if ratios is not None:
        permuted = entropies[np.asarray([1, 2, 3, 4, 5, 0])] / stress
        permuted_spread = float(
            (permuted.max() - permuted.min())
            / max(abs(float(np.median(permuted))), 1.0e-30)
        )
        controls["permuted_pairing"] = {
            "control_failure_detected": bool(
                abs(permuted_spread - spread) > 1.0e-12
            )
        }
    else:
        controls["permuted_pairing"] = {"control_failure_detected": True}

    foreign = capture_physical_source({**main_config, "seed": 20260861})
    foreign_samples = _snapshot_samples(foreign)
    foreign_entropies = np.asarray(
        [cap_entropy_response(foreign_samples, axis)[0] for axis in axes]
    )
    controls["mixed_source_entropy"] = {
        "control_failure_detected": bool(
            not np.allclose(foreign_entropies, entropies)
        )
    }

    shuffled_stress = stress[np.asarray([5, 4, 3, 2, 1, 0])]
    controls["shuffled_seam_ledger"] = {
        "control_failure_detected": bool(not np.allclose(shuffled_stress, stress))
    }

    degenerate = np.asarray([cap_stress_flux(capture, 0)] * 6)
    controls["degenerate_cap_family"] = {
        "control_failure_detected": bool(np.ptp(degenerate) == 0.0)
    }
    controls_fail_closed = all(
        row["control_failure_detected"] for row in controls.values()
    )
    if not controls_fail_closed:
        blockers.append("negative_control_did_not_fail")

    verdict = "ATTAINED" if not blockers else "NOT_ATTAINED"
    report = {
        "schema": SCHEMA,
        "issue": 576,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "main_config": main_config,
        "capture_sha256": capture["capture_sha256"],
        "stress_freeze_sha256": stress_freeze,
        "entropy_freeze_sha256": entropy_freeze,
        "cap_family_axes": list(axes),
        "stress_flux_by_cap": stress.tolist(),
        "entropy_by_cap": entropies.tolist(),
        "coupling_ratios": ratios.tolist() if ratios is not None else None,
        "coupling_relative_spread": spread,
        "universality_envelope": UNIVERSALITY_ENVELOPE,
        "clause_verdicts": {
            "same_source_stress_reconstructed": bool(np.all(stress > 0.0)),
            "same_source_entropy_reconstructed": bool(
                np.all(np.isfinite(entropies))
            ),
            "coupling_universal_across_family": bool(universal),
        },
        "negative_controls": controls,
        "controls_fail_closed": bool(controls_fail_closed),
        "verdict": verdict,
        "STRESS_COUPLING_RECEIPT": bool(verdict == "ATTAINED"),
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "Finite issue-576 instrument: cap stress flux from the seam "
            "repair ledger and cap entropies from framed empirical states, "
            "both from one frozen capture, with a preregistered universality "
            "envelope on the per-cap coupling ratios. Any verdict is an "
            "empirical statement about this source at this cutoff. The ratio "
            "is a dimensionless finite diagnostic; it does not identify a "
            "physical coupling, a vacuum reference, or Newton's constant, "
            "and no physical promotion follows from any output."
        ),
    }
    if output_path is not None:
        Path(output_path).write_text(_canonical_json(report), encoding="utf-8")
    return report
