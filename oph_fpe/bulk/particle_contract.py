from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from oph_fpe.claims import (
    BRANCH_INSTANTIATION_SANITY,
    BULK_WORLDLINE_PRECURSOR_RECEIPT,
    CLASSICAL_CARRIER_MODE_RECEIPT,
    COLORED_DECONFINEMENT_RECEIPT,
    PRODUCTION_PARTICLE_MATTER_RECEIPT,
    QUANTUM_PARTICLE_RECEIPT,
    with_claim_metadata,
)


EVIDENCE_FILE = "particle_promotion_evidence.json"
SCHEMA = "oph_particle_promotion_evidence_v1"
DEFAULT_TOLERANCE = 1.0e-6
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")

_IGNORED_PROMOTION_KEYS = (
    "particle_matter_receipt",
    "physical_particle_emergence",
    "production_particle_matter_receipt",
    PRODUCTION_PARTICLE_MATTER_RECEIPT,
    "promotion_allowed",
    "passed",
)


def particle_promotion_contract_report(
    run_dir: Path,
    *,
    evidence_file: str = EVIDENCE_FILE,
) -> dict[str, Any]:
    """Recompute P0, classical-carrier, and quantum-particle gates.

    Producer-supplied promotion booleans are deliberately ignored.  The only
    promotable input is the typed primitive-evidence artifact, whose three
    lane payloads must also match hash-pinned sidecars inside ``run_dir``.
    """

    root = Path(run_dir)
    evidence_path = root / evidence_file
    evidence, read_error = _read_mapping(evidence_path)
    tolerance = _tolerance(evidence.get("tolerance"))
    ignored = {key: evidence.get(key) for key in _IGNORED_PROMOTION_KEYS if key in evidence}

    schema_gates = {
        "schema_exact": evidence.get("schema") == SCHEMA,
        "candidate_id_nonempty": _nonempty_string(evidence.get("candidate_id")),
        "candidate_kind_typed": evidence.get("candidate_kind") in {"neutral", "colored"},
        "structural_speed_positive": _positive_number(evidence.get("structural_speed")),
        "tolerance_valid": tolerance is not None,
    }
    provenance = _provenance_report(root, evidence)
    proto = _proto_report(_mapping(evidence.get("proto")), tolerance)
    classical = _classical_report(
        _mapping(evidence.get("classical_carrier")),
        structural_speed=_number(evidence.get("structural_speed")),
        tolerance=tolerance,
    )
    quantum = _quantum_report(
        _mapping(evidence.get("quantum_particle")),
        candidate_kind=evidence.get("candidate_kind"),
        tolerance=tolerance,
    )
    runtime_binding = {
        "mode": "particle_runtime_producer_binding_v1",
        "passed": False,
        "producer_implemented": False,
        "blockers": ["runtime_particle_evidence_producer_not_implemented"],
        "claim_boundary": (
            "The current artifact is a contract-shaped candidate only. No runtime "
            "producer binds these values to simulator states, trajectories, action/Hessian "
            "outputs, spectral calculations, or the production fusion-transport receipt."
        ),
    }

    lane_hashes = provenance["lane_artifacts"]
    proto_candidate = bool(
        all(schema_gates.values())
        and provenance["passed"]
        and lane_hashes["proto"]["passed"]
        and proto["passed"]
    )
    classical_candidate = bool(
        all(schema_gates.values())
        and provenance["passed"]
        and lane_hashes["classical_carrier"]["passed"]
        and classical["passed"]
    )
    quantum_candidate = bool(
        all(schema_gates.values())
        and provenance["passed"]
        and lane_hashes["quantum_particle"]["passed"]
        and quantum["passed"]
    )
    proto_receipt = bool(proto_candidate and runtime_binding["passed"])
    classical_receipt = bool(classical_candidate and runtime_binding["passed"])
    quantum_receipt = bool(quantum_candidate and runtime_binding["passed"])
    colored = evidence.get("candidate_kind") == "colored"
    deconfinement_receipt = bool(
        runtime_binding["passed"]
        and (not colored or quantum["checks"].get("colored_deconfined_asymptotic_sector", False))
    )
    production = bool(
        proto_receipt
        and classical_receipt
        and quantum_receipt
        and deconfinement_receipt
    )

    blockers = []
    if read_error:
        blockers.append(read_error)
    blockers.extend(f"schema:{key}" for key, passed in schema_gates.items() if not passed)
    blockers.extend(f"provenance:{item}" for item in provenance["blockers"])
    blockers.extend(f"P0:{item}" for item in proto["blockers"])
    blockers.extend(f"classical:{item}" for item in classical["blockers"])
    blockers.extend(f"quantum:{item}" for item in quantum["blockers"])
    blockers.extend(f"runtime_binding:{item}" for item in runtime_binding["blockers"])
    if colored and not deconfinement_receipt:
        blockers.append("quantum:colored_deconfinement_missing")

    report = {
        "mode": "oph_particle_promotion_contract_v1",
        "schema": SCHEMA,
        "run_path": str(root),
        "evidence_file": evidence_file,
        "evidence_sha256": _sha256_file(evidence_path) if evidence_path.is_file() else None,
        "candidate_id": evidence.get("candidate_id"),
        "candidate_kind": evidence.get("candidate_kind"),
        "structural_speed": evidence.get("structural_speed"),
        "tolerance": tolerance,
        "schema_gates": schema_gates,
        "provenance": provenance,
        "runtime_producer_binding": runtime_binding,
        "candidate_lane_contracts": {
            "P0_proto_worldline": proto_candidate,
            "classical_carrier_mode": classical_candidate,
            "quantum_particle": quantum_candidate,
        },
        "lanes": {
            "P0_proto_worldline": proto,
            "classical_carrier_mode": classical,
            "quantum_particle": quantum,
        },
        BULK_WORLDLINE_PRECURSOR_RECEIPT: proto_receipt,
        "bulk_worldline_precursor_receipt": proto_receipt,
        CLASSICAL_CARRIER_MODE_RECEIPT: classical_receipt,
        "classical_carrier_mode_receipt": classical_receipt,
        QUANTUM_PARTICLE_RECEIPT: quantum_receipt,
        "quantum_particle_receipt": quantum_receipt,
        COLORED_DECONFINEMENT_RECEIPT: deconfinement_receipt,
        "colored_deconfinement_receipt": deconfinement_receipt,
        PRODUCTION_PARTICLE_MATTER_RECEIPT: production,
        "production_particle_matter_receipt": production,
        "particle_matter_receipt": production,
        "ignored_caller_promotion_fields": ignored,
        "blockers": _unique(blockers),
        "claim_boundary": (
            "P0 records a localized, transported, controlled proto-worldline only. A production "
            "particle requires P0 plus an action-level classical carrier-mode receipt and an "
            "independent positive-energy quantum Hilbert/spectral/asymptotic receipt. Colored "
            "candidates additionally require a deconfined asymptotic sector. Producer top-level "
            "booleans, defect labels, or truthy non-booleans never promote P1."
            " Until a runtime producer binds and independently replays the lane primitives, "
            "even a complete self-contained evidence artifact remains diagnostic."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=BRANCH_INSTANTIATION_SANITY,
        receipt=PRODUCTION_PARTICLE_MATTER_RECEIPT,
        physical_claim=production,
        observable_id="particle_promotion_contract",
        fit_objective="primitive_particle_receipt_recomputation",
    )


def write_particle_promotion_contract_report(
    run_dir: Path,
    out: Path | None = None,
) -> dict[str, Any]:
    report = particle_promotion_contract_report(run_dir)
    destination = Path(out) if out is not None else Path(run_dir) / "particle_promotion_contract_report.json"
    if destination.is_dir():
        destination = destination / "particle_promotion_contract_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _proto_report(data: Mapping[str, Any], tolerance: float | None) -> dict[str, Any]:
    tol = tolerance if tolerance is not None else DEFAULT_TOLERANCE
    checks = {
        "localized_shared_bulk_support": bool(
            _literal_true(data.get("localized_shared_bulk_support"))
            and _positive_number(data.get("bulk_localization_margin"))
            and _in_closed_interval(data.get("maximum_support_fraction"), 0.0, 1.0, upper_open=True)
        ),
        "stable_topological_sector_charge": bool(
            _literal_true(data.get("stable_topological_sector_charge"))
            and _le_nonnegative(data.get("maximum_sector_charge_variation"), tol)
        ),
        "contractible_path_transport": bool(
            _literal_true(data.get("contractible_path_transport"))
            and _le_nonnegative(data.get("maximum_contractible_holonomy_residual"), tol)
        ),
        "fusion_charge_conservation": bool(
            _literal_true(data.get("fusion_charge_conservation"))
            and _le_nonnegative(data.get("maximum_fusion_charge_residual"), tol)
        ),
        "scattering_reproducibility": bool(
            _literal_true(data.get("scattering_reproducibility"))
            and _ge_number(data.get("minimum_scattering_replay_agreement"), 1.0 - tol)
        ),
        "speed_causality_controls": bool(
            _literal_true(data.get("speed_causality_controls"))
            and _le_nonnegative(data.get("maximum_causality_violation"), tol)
        ),
        "observer_resampling_stability": bool(
            _literal_true(data.get("observer_resampling_stability"))
            and _le_nonnegative(data.get("maximum_observer_resampling_drift"), tol)
        ),
        "refinement_stability": bool(
            _literal_true(data.get("refinement_stability"))
            and _le_nonnegative(data.get("maximum_refinement_drift"), tol)
        ),
        "worldline_population": _positive_int(data.get("worldline_count")),
        "independent_seed_control": _int_at_least(data.get("independent_seed_count"), 2),
        "multi_refinement_control": _int_at_least(data.get("refinement_level_count"), 2),
    }
    return _lane(checks)


def _classical_report(
    data: Mapping[str, Any],
    *,
    structural_speed: float | None,
    tolerance: float | None,
) -> dict[str, Any]:
    tol = tolerance if tolerance is not None else DEFAULT_TOLERANCE
    action_rank = _integer(data.get("quadratic_action_rank"))
    projector_rank = _integer(data.get("physical_projector_rank"))
    dispersion_speed = _number(data.get("dispersion_speed"))
    checks = {
        "background_and_phase_stated": bool(
            _nonempty_string(data.get("background")) and _nonempty_string(data.get("phase"))
        ),
        "quadratic_action_explicit": bool(
            _literal_true(data.get("quadratic_action_explicit"))
            and action_rank is not None
            and action_rank >= 1
        ),
        "positive_physical_kinetic_coefficient": _positive_number(
            data.get("physical_kinetic_coefficient")
        ),
        "physical_constraint_reduction": data.get("constraint_reduction")
        in {"gauge_fixing", "constraint_reduction", "brst_cohomology"},
        "physical_projector_nonzero": bool(
            projector_rank is not None
            and projector_rank >= 1
            and action_rank is not None
            and projector_rank <= action_rank
        ),
        "positive_reduced_hamiltonian": _ge_number(data.get("reduced_hamiltonian_minimum"), 0.0),
        "physical_hessian_wave_operator": _le_nonnegative(
            data.get("hessian_wave_operator_residual"), tol
        ),
        "structural_speed_dispersion": bool(
            structural_speed is not None
            and structural_speed > 0.0
            and dispersion_speed is not None
            and abs(dispersion_speed - structural_speed) <= tol * max(1.0, structural_speed)
        ),
        "mass_operator_absent": _le_nonnegative(data.get("forbidden_mass_operator_norm"), tol),
    }
    return _lane(checks)


def _quantum_report(
    data: Mapping[str, Any],
    *,
    candidate_kind: Any,
    tolerance: float | None,
) -> dict[str, Any]:
    tol = tolerance if tolerance is not None else DEFAULT_TOLERANCE
    colored = candidate_kind == "colored"
    checks = {
        "positive_energy_vacuum_quantization": bool(
            data.get("vacuum_quantization")
            in {"canonical", "fock", "brst", "operator_algebraic"}
            and _ge_number(data.get("vacuum_energy_lower_bound"), 0.0)
        ),
        "physical_hilbert_reduction": bool(
            data.get("physical_hilbert_construction")
            in {"constraint_reduction", "brst_cohomology"}
            and _integer(data.get("negative_norm_state_count")) == 0
        ),
        "positive_kallen_lehmann_pole": bool(
            _positive_number(data.get("pole_residue"))
            and _le_nonnegative(data.get("spectral_negative_weight"), tol)
        ),
        "positive_energy_mass_shell": bool(
            _positive_number(data.get("positive_energy_shell_minimum"))
            and _le_nonnegative(data.get("mass_shell_residual"), tol)
        ),
        "stable_asymptotic_state": bool(
            _positive_number(data.get("asymptotic_state_norm"))
            and _le_nonnegative(data.get("decay_width_upper_bound"), tol)
        ),
        "lsz_phase_hypotheses": _positive_number(data.get("lsz_residue")),
        "colored_deconfined_asymptotic_sector": bool(
            not colored or _literal_true(data.get("deconfined_asymptotic_sector"))
        ),
    }
    return _lane(checks)


def _provenance_report(root: Path, evidence: Mapping[str, Any]) -> dict[str, Any]:
    provenance = _mapping(evidence.get("provenance"))
    checks = {
        "generator_named": _nonempty_string(provenance.get("generator")),
        "source_commit_typed": bool(
            isinstance(provenance.get("source_commit"), str)
            and _COMMIT_RE.fullmatch(str(provenance.get("source_commit")))
        ),
        "source_tree_clean": provenance.get("source_dirty") is False,
        "generated_utc_named": _nonempty_string(provenance.get("generated_utc")),
    }
    lane_refs = _mapping(provenance.get("lane_artifacts"))
    lane_artifacts: dict[str, dict[str, Any]] = {}
    for lane, evidence_key in (
        ("proto", "proto"),
        ("classical_carrier", "classical_carrier"),
        ("quantum_particle", "quantum_particle"),
    ):
        lane_artifacts[lane] = _verify_lane_artifact(
            root,
            _mapping(lane_refs.get(lane)),
            _mapping(evidence.get(evidence_key)),
        )
    passed = bool(all(checks.values()) and all(row["passed"] for row in lane_artifacts.values()))
    blockers = [key for key, value in checks.items() if not value]
    blockers.extend(
        f"{lane}:{item}"
        for lane, row in lane_artifacts.items()
        for item in row["blockers"]
    )
    return {
        "passed": passed,
        "checks": checks,
        "lane_artifacts": lane_artifacts,
        "blockers": _unique(blockers),
        "source_commit": provenance.get("source_commit"),
        "source_dirty": provenance.get("source_dirty"),
        "generator": provenance.get("generator"),
        "generated_utc": provenance.get("generated_utc"),
    }


def _verify_lane_artifact(
    root: Path,
    reference: Mapping[str, Any],
    expected_payload: Mapping[str, Any],
) -> dict[str, Any]:
    rel = reference.get("path")
    expected_hash = reference.get("sha256")
    blockers = []
    path: Path | None = None
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
        blockers.append("relative_path_missing")
    else:
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            blockers.append("path_escapes_run_root")
        else:
            path = candidate
    if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
        blockers.append("sha256_invalid")
    actual_hash = _sha256_file(path) if path is not None and path.is_file() else None
    if actual_hash is None:
        blockers.append("artifact_missing")
    elif actual_hash != expected_hash:
        blockers.append("sha256_mismatch")
    artifact_payload, read_error = _read_mapping(path) if path is not None else ({}, "artifact_missing")
    if read_error:
        blockers.append("artifact_not_json_object")
    if artifact_payload != dict(expected_payload):
        blockers.append("artifact_payload_mismatch")
    return {
        "passed": not blockers,
        "path": rel,
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "blockers": _unique(blockers),
    }


def _lane(checks: Mapping[str, bool]) -> dict[str, Any]:
    blockers = [key for key, value in checks.items() if not value]
    return {"passed": not blockers, "checks": dict(checks), "blockers": blockers}


def _read_mapping(path: Path | None) -> tuple[dict[str, Any], str | None]:
    if path is None or not path.is_file():
        return {}, "particle_promotion_evidence_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, "particle_promotion_evidence_invalid_json"
    if not isinstance(payload, dict):
        return {}, "particle_promotion_evidence_not_object"
    return payload, None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _literal_true(value: Any) -> bool:
    return type(value) is bool and value is True


def _number(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    return int(value) if type(value) is int else None


def _positive_number(value: Any) -> bool:
    number = _number(value)
    return number is not None and number > 0.0


def _positive_int(value: Any) -> bool:
    number = _integer(value)
    return number is not None and number > 0


def _int_at_least(value: Any, minimum: int) -> bool:
    number = _integer(value)
    return number is not None and number >= minimum


def _ge_number(value: Any, minimum: float) -> bool:
    number = _number(value)
    return number is not None and number >= minimum


def _le_nonnegative(value: Any, maximum: float) -> bool:
    number = _number(value)
    return number is not None and 0.0 <= number <= maximum


def _in_closed_interval(value: Any, lower: float, upper: float, *, upper_open: bool = False) -> bool:
    number = _number(value)
    if number is None or number < lower:
        return False
    return number < upper if upper_open else number <= upper


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _tolerance(value: Any) -> float | None:
    if value is None:
        return DEFAULT_TOLERANCE
    number = _number(value)
    if number is None or not 0.0 < number <= 1.0e-3:
        return None
    return number


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))
