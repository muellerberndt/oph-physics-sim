"""Fail-closed finite evidence contract for the central-interface collar clause.

The Lean T0/T1/T2 results rule out several tempting producer shortcuts.  A
Gibbs family, relative-entropy/CMI facts, a flux conditional expectation, or
membership in a modular centralizer can be useful diagnostics, but none of
them proves that every retained cross-cut density belongs to the flux sector.

This module therefore accepts a bounded real-coordinate presentation of the
left, right, and flux subspaces and requires an explicit, hash-pinned
factorization through the flux basis for every *computed* cross-cut density.
The coordinate presentation is a finite certificate interface, not a claim
that state-side data construct the collar layer.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.evidence.artifact_paths import companion_input_packet_path
from oph_fpe.evidence.hashes import canonical_json_bytes
from oph_fpe.evidence.validation import utf8_byte_length


SCHEMA_VERSION = "oph-collar-clause-v1"
ALGORITHM_ID = "bounded-retained-family-flux-factorization-v1"
RECEIPT_ID = "COLLAR_CLAUSE_SOURCE_RECEIPT"
PACKET_CONSISTENCY_RECEIPT_ID = "COLLAR_CLAUSE_PACKET_CONSISTENCY_RECEIPT"
INDEPENDENT_RUN_REPLAY_RECEIPT_ID = "INDEPENDENT_COLLAR_RUN_REPLAY_RECEIPT"
MAX_AMBIENT_DIMENSION = 256
MAX_TOLERANCE = 1.0e-6
MAX_PACKET_BYTES = 2_000_000
MAX_IDENTIFIER_BYTES = 256
MAX_SOURCE_ANCESTRY_NODES = 1024
MAX_RETAINED_COORDINATES = 1024
MAX_ABS_COORDINATE = 1.0e12
MAX_COLLAR_LINEAR_WORK = 50_000_000
MIN_BASIS_SINGULAR_VALUE = 1.0e-8
MAX_BASIS_SINGULAR_VALUE = 1.0e8
MAX_BASIS_CONDITION_NUMBER = 1.0e8
COORDINATE_SEMANTICS = "bounded_real_coordinate_vector_not_verified_density_operator"

# Read-only theory registry pinned during the r1556/bec81e2d sync.  These are
# raw-file SHA-256 digests, not hashes of caller-created JSON declarations.
PINNED_COLLAR_THEORY_ARTIFACTS = {
    "rer-r1556-collar-clause-lean": "sha256:813066e7b9ff34005188624131f6cdfc45f112582bb336fd8a88b359f21814a4",
    "rer-r1556-collar-layer-lean": "sha256:34405f7e5e3dc094ec25c9afc597d5c6ff3300765f1589258d54087cd19463e2",
    "rer-r1556-collar-modular-t2-lean": "sha256:515f85fa72c7d16374bd5fc5b885ea082d1b8e608466dbe359c444162315beab",
    "rer-r1556-collar-states-bridge-lean": "sha256:4440785282b1bcf1768ea801649a7f78964dc08019e8d3090408880cd915cbbc",
    "rer-r1556-collar-states-lean": "sha256:b9d426c22d5c84198ded6450e57d5d3b36964a1810d9ea95f5b78a76ea819c33",
    "rer-r1556-collar-states-t1-lean": "sha256:277f3ee8eace4c18d08fc2b588b367476dc81e26c474e9d5e6572f209257254c",
}

_SHA256_PREFIX = "sha256:"
_ALLOWED_SOURCE_KINDS = frozenset(
    {
        "lean_theorem",
        "lean_definition",
        "theory_source",
        "source_derivation",
        "bounded_certificate",
        "simulation_derivation",
    }
)
_FAMILY_BOOLEAN_LAWS = (
    "manifest_complete",
    "gauge_invariant",
    "collar_supported",
    "refinement_closed",
    "refinement_channel_manifest_complete",
)
_DIAGNOSTIC_FIELDS = (
    "density_matrices_valid",
    "gibbs_family_realized",
    "relative_entropy_nonnegative",
    "cmi_zero_or_bounded",
    "conditional_expectation_exists",
    "conditional_expectation_kills_nonflux",
    "modular_centralizer_contains_retained",
    "modular_diagonal_constraint",
)


def canonical_evidence_hash(value: Any) -> str:
    """Return the canonical ``sha256:`` identifier used by this contract."""

    return _SHA256_PREFIX + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def collar_clause_negative_controls() -> dict[str, dict[str, Any]]:
    """Return the mandatory T0/T1/T2 anti-promotion policy receipts.

    These are policy controls derived from the named Lean no-go theorems.  The
    evaluator below never consults state-side diagnostics when deciding the
    positive source receipt, so each attempted shortcut is rejected.
    """

    return {
        "T0_STATE_SIDE": {
            "theorem": "CollarStates.stateSide_axioms_do_not_force",
            "attempted_basis": (
                "density/Gibbs/relative-entropy/CMI or identity-channel closure alone"
            ),
            "explicit_cross_cut_flux_factorization": False,
            "would_promote": False,
            "rejected": True,
        },
        "T1_EFLUX": {
            "theorem": (
                "CollarStatesT1.Eflux_does_not_force;"
                "CollarStatesT1.EfluxChannel_deselects_XXC"
            ),
            "attempted_basis": (
                "existence of a flux conditional expectation that deselects "
                "a non-flux coupling"
            ),
            "explicit_cross_cut_flux_factorization": False,
            "would_promote": False,
            "rejected": True,
        },
        "T2_MODULAR": {
            "theorem": (
                "CollarModularT2.naive_modular_recast_does_not_exclude;"
                "CollarModularT2.centralizer_diagonal_strictly_contains_flux"
            ),
            "attempted_basis": (
                "modular-centralizer or corrected diagonal membership alone"
            ),
            "explicit_cross_cut_flux_factorization": False,
            "would_promote": False,
            "rejected": True,
        },
    }


def verify_collar_clause_packet(
    packet: Mapping[str, Any],
    *,
    tolerance: float = 1.0e-10,
    max_ambient_dimension: int = MAX_AMBIENT_DIMENSION,
) -> dict[str, Any]:
    """Verify a bounded retained-family collar-clause evidence packet.

    Structural malformation raises ``ValueError`` and emits no receipt.  A
    well-formed packet with incomplete or false evidence returns a negative
    receipt with explicit blockers.
    """

    if not isinstance(packet, Mapping):
        raise ValueError("packet must be a mapping")
    try:
        packet_bytes = canonical_json_bytes(packet)
    except (TypeError, UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"packet must be bounded canonical evidence data: {exc}") from exc
    if len(packet_bytes) > MAX_PACKET_BYTES:
        raise ValueError("packet exceeds the bounded byte limit")
    allowed_packet_fields = {
        "packet_id",
        "source_ancestry",
        "source_dag_hash",
        "collar_layer",
        "collar_layer_hash",
        "retained_densities",
        "family_laws",
        "retained_family_hash",
        "state_side_diagnostics",
    }
    required_packet_fields = allowed_packet_fields - {"state_side_diagnostics"}
    if not set(packet).issubset(
        allowed_packet_fields
    ) or not required_packet_fields.issubset(packet):
        raise ValueError("packet has missing or unknown top-level fields")
    try:
        normalized_tolerance = float(tolerance)
    except (OverflowError, TypeError, ValueError):
        normalized_tolerance = math.nan
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(normalized_tolerance)
        or normalized_tolerance <= 0.0
        or normalized_tolerance > MAX_TOLERANCE
    ):
        raise ValueError(f"tolerance must be finite and in (0, {MAX_TOLERANCE}]")
    tolerance = normalized_tolerance
    if (
        isinstance(max_ambient_dimension, bool)
        or not isinstance(max_ambient_dimension, int)
        or max_ambient_dimension < 1
        or max_ambient_dimension > MAX_AMBIENT_DIMENSION
    ):
        raise ValueError(
            f"max_ambient_dimension must be an integer in [1, {MAX_AMBIENT_DIMENSION}]"
        )

    packet_id = packet.get("packet_id")
    _bounded_identifier(packet_id, "packet_id")

    source_ancestry = _source_ancestry(packet.get("source_ancestry"))
    supplied_source_hash = _required_hash(
        packet.get("source_dag_hash"), "source_dag_hash"
    )
    computed_source_hash = canonical_evidence_hash(source_ancestry)
    source_hash_matches = supplied_source_hash == computed_source_hash
    source_kinds = {node["kind"] for node in source_ancestry}
    pinned_theory_matches = {
        node["node_id"]
        for node in source_ancestry
        if PINNED_COLLAR_THEORY_ARTIFACTS.get(node["node_id"]) == node["sha256"]
    }
    pinned_theory_registry_complete = pinned_theory_matches == set(
        PINNED_COLLAR_THEORY_ARTIFACTS
    )
    unregistered_lean_nodes = [
        node["node_id"]
        for node in source_ancestry
        if node["kind"] in {"lean_theorem", "lean_definition"}
        and PINNED_COLLAR_THEORY_ARTIFACTS.get(node["node_id"]) != node["sha256"]
    ]
    source_dag_clean = (
        bool(source_ancestry)
        and source_kinds.issubset(_ALLOWED_SOURCE_KINDS)
        and not unregistered_lean_nodes
        and pinned_theory_registry_complete
    )
    source_ids = {node["node_id"] for node in source_ancestry}

    layer = _required_mapping(packet.get("collar_layer"), "collar_layer")
    if set(layer) != {"ambient_dimension", "left_basis", "right_basis", "flux_basis"}:
        raise ValueError(
            "collar_layer must contain exactly ambient_dimension, left_basis, "
            "right_basis, and flux_basis"
        )
    ambient_dimension = layer["ambient_dimension"]
    if (
        isinstance(ambient_dimension, bool)
        or not isinstance(ambient_dimension, int)
        or ambient_dimension < 1
        or ambient_dimension > max_ambient_dimension
    ):
        raise ValueError(
            f"ambient_dimension must be an integer in [1, {max_ambient_dimension}]"
        )
    left_basis = _basis(layer["left_basis"], ambient_dimension, "left_basis")
    right_basis = _basis(layer["right_basis"], ambient_dimension, "right_basis")
    flux_basis = _basis(layer["flux_basis"], ambient_dimension, "flux_basis")
    if not left_basis.shape[0] or not right_basis.shape[0] or not flux_basis.shape[0]:
        raise ValueError("left, right, and flux bases must each be non-empty")
    declared_sector_rank_sum = int(
        left_basis.shape[0] + right_basis.shape[0] + flux_basis.shape[0]
    )
    if declared_sector_rank_sum > ambient_dimension:
        raise ValueError(
            "declared left/right/flux sector ranks exceed the ambient dimension"
        )

    densities = packet.get("retained_densities")
    if not isinstance(densities, list):
        raise ValueError("retained_densities must be a list")
    if len(densities) > min(4 * ambient_dimension, MAX_RETAINED_COORDINATES):
        raise ValueError("retained_densities exceeds the bounded packet limit")
    combined_basis = np.vstack((left_basis, right_basis, flux_basis))
    decomposition_work = sum(
        _estimated_svd_work(basis)
        for basis in (left_basis, right_basis, flux_basis, combined_basis)
    )
    retained_vector_work = (
        max(1, len(densities))
        * ambient_dimension
        * max(1, 4 * declared_sector_rank_sum)
    )
    estimated_linear_work = decomposition_work + retained_vector_work
    if estimated_linear_work > MAX_COLLAR_LINEAR_WORK:
        raise ValueError("collar linear-algebra operation budget exceeded")

    basis_analyses = {
        name: _basis_analysis(basis, tolerance=float(tolerance))
        for name, basis in (
            ("left", left_basis),
            ("right", right_basis),
            ("flux", flux_basis),
            ("combined", combined_basis),
        )
    }
    for name, basis in (
        ("left", left_basis),
        ("right", right_basis),
        ("flux", flux_basis),
    ):
        if basis_analyses[name]["rank"] != basis.shape[0]:
            raise ValueError(f"{name}_basis rows must be linearly independent")
    combined_sector_rank = int(basis_analyses["combined"]["rank"])
    sector_direct_sum_independent = bool(
        combined_sector_rank == declared_sector_rank_sum
    )
    flux_is_proper_ambient_sector = bool(flux_basis.shape[0] < ambient_dimension)
    basis_conditioning = {
        name: analysis["conditioning"]
        for name, analysis in basis_analyses.items()
    }
    basis_numerical_conditioning_passed = bool(
        all(metrics["passed"] for metrics in basis_conditioning.values())
    )

    supplied_layer_hash = _required_hash(
        packet.get("collar_layer_hash"), "collar_layer_hash"
    )
    computed_layer_hash = canonical_evidence_hash(layer)
    layer_hash_matches = supplied_layer_hash == computed_layer_hash
    flux_basis_hash = canonical_evidence_hash(layer["flux_basis"])

    left_span = basis_analyses["left"]["orthonormal_span"]
    right_span = basis_analyses["right"]["orthonormal_span"]
    flux_span = basis_analyses["flux"]["orthonormal_span"]
    family_laws = _required_mapping(packet.get("family_laws"), "family_laws")
    expected_law_fields = set(_FAMILY_BOOLEAN_LAWS) | {"source_derivation_ids"}
    if set(family_laws) != expected_law_fields:
        raise ValueError(
            "family_laws has missing or unknown fields; expected "
            + ", ".join(sorted(expected_law_fields))
        )
    family_source_ids = _string_list(
        family_laws["source_derivation_ids"], "family_laws.source_derivation_ids"
    )
    for name in _FAMILY_BOOLEAN_LAWS:
        if not isinstance(family_laws[name], bool):
            raise ValueError(f"family_laws.{name} must be boolean")
    family_source_derived = bool(
        family_source_ids
        and source_dag_clean
        and source_hash_matches
        and all(source_id in source_ids for source_id in family_source_ids)
    )
    family_boolean_laws_pass = all(
        family_laws[name] is True for name in _FAMILY_BOOLEAN_LAWS
    )
    retained_family_complete = bool(
        densities and family_boolean_laws_pass and family_source_derived
    )
    family_payload = {"densities": densities, "family_laws": dict(family_laws)}
    supplied_family_hash = _required_hash(
        packet.get("retained_family_hash"), "retained_family_hash"
    )
    computed_family_hash = canonical_evidence_hash(family_payload)
    family_hash_matches = supplied_family_hash == computed_family_hash

    diagnostics = _diagnostics(packet.get("state_side_diagnostics", {}))
    density_results: list[dict[str, Any]] = []
    density_ids: set[str] = set()
    for index, density in enumerate(densities):
        row = _required_mapping(density, f"retained_densities[{index}]")
        allowed_fields = {
            "density_id",
            "coordinates",
            "declared_support",
            "flux_factorization",
        }
        if not set(row).issubset(allowed_fields) or not {
            "density_id",
            "coordinates",
            "declared_support",
        }.issubset(row):
            raise ValueError(
                f"retained_densities[{index}] has missing or unknown fields"
            )
        density_id = row["density_id"]
        _bounded_identifier(
            density_id,
            f"retained_densities[{index}].density_id",
        )
        if density_id in density_ids:
            raise ValueError(f"duplicate density_id: {density_id}")
        density_ids.add(density_id)
        vector = _vector(
            row["coordinates"], ambient_dimension, f"{density_id}.coordinates"
        )
        declared_support = row["declared_support"]
        if declared_support not in {"LEFT", "RIGHT", "CROSS_CUT"}:
            raise ValueError(
                f"{density_id}.declared_support must be LEFT, RIGHT, or CROSS_CUT"
            )

        left_residual = _span_residual(vector, left_span)
        right_residual = _span_residual(vector, right_span)
        flux_residual = _span_residual(vector, flux_span)
        in_left = left_residual <= tolerance
        in_right = right_residual <= tolerance
        cross_cut = not (in_left or in_right)
        if in_left and in_right:
            computed_support = "BOTH_ONE_SIDED"
        elif in_left:
            computed_support = "LEFT"
        elif in_right:
            computed_support = "RIGHT"
        else:
            computed_support = "CROSS_CUT"
        support_matches = (
            (declared_support == "LEFT" and in_left)
            or (declared_support == "RIGHT" and in_right)
            or (declared_support == "CROSS_CUT" and cross_cut)
        )

        factor_report = _factorization_report(
            row.get("flux_factorization"),
            vector=vector,
            flux_basis=flux_basis,
            flux_basis_hash=flux_basis_hash,
            source_ids=source_ids,
            source_dag_clean=source_dag_clean and source_hash_matches,
            tolerance=float(tolerance),
            field_name=f"{density_id}.flux_factorization",
        )
        density_results.append(
            {
                "density_id": density_id,
                "coordinate_semantics": COORDINATE_SEMANTICS,
                "declared_support": declared_support,
                "computed_support": computed_support,
                "support_classification_matches": bool(support_matches),
                "cross_cut": bool(cross_cut),
                "left_span_residual": float(left_residual),
                "right_span_residual": float(right_residual),
                "flux_span_residual": float(flux_residual),
                "numerically_in_flux_span": bool(flux_residual <= tolerance),
                **factor_report,
            }
        )

    cross_cut_rows = [row for row in density_results if row["cross_cut"]]
    nonvacuous_cross_cut = bool(cross_cut_rows)
    all_support_matches = bool(density_results) and all(
        row["support_classification_matches"] for row in density_results
    )
    all_cross_cut_factor = nonvacuous_cross_cut and all(
        row["factor_through_flux"] and row["numerically_in_flux_span"]
        for row in cross_cut_rows
    )

    controls = collar_clause_negative_controls()
    controls_pass = all(
        control["rejected"] and not control["would_promote"]
        for control in controls.values()
    )
    blockers: list[str] = []
    if not source_hash_matches:
        blockers.append("source_dag_hash_mismatch")
    if not source_dag_clean:
        blockers.append("source_dag_has_non_source_or_empty_ancestry")
    if not pinned_theory_registry_complete:
        blockers.append("pinned_collar_theorem_registry_incomplete")
    if unregistered_lean_nodes:
        blockers.append("unregistered_or_mismatched_lean_theorem_artifact")
    if not layer_hash_matches:
        blockers.append("collar_layer_hash_mismatch")
    if not sector_direct_sum_independent:
        blockers.append("left_right_flux_sectors_not_direct_sum_independent")
    if not flux_is_proper_ambient_sector:
        blockers.append("flux_sector_equals_full_ambient_space")
    if not basis_numerical_conditioning_passed:
        blockers.append("basis_numerical_conditioning_out_of_bounds")
    if not family_hash_matches:
        blockers.append("retained_family_hash_mismatch")
    if not retained_family_complete:
        blockers.append("retained_family_laws_or_source_evidence_incomplete")
    if not all_support_matches:
        blockers.append("retained_support_classification_mismatch")
    if not nonvacuous_cross_cut:
        blockers.append("no_retained_cross_cut_density")
    if not all_cross_cut_factor:
        blockers.append("cross_cut_density_lacks_explicit_flux_factorization")
    if not controls_pass:
        blockers.append("mandatory_T0_T1_T2_negative_controls_failed")

    packet_consistency_passed = not blockers
    independent_run_replay = False
    blockers.append("independent_collar_run_artifact_resolution_unavailable")
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM_ID,
        "receipt": RECEIPT_ID,
        "passed": False,
        "source_eligible": False,
        PACKET_CONSISTENCY_RECEIPT_ID: bool(packet_consistency_passed),
        INDEPENDENT_RUN_REPLAY_RECEIPT_ID: independent_run_replay,
        "packet_consistency_passed": bool(packet_consistency_passed),
        "packet_id": packet_id,
        "packet_hash": canonical_evidence_hash(packet),
        "source_dag_hash": supplied_source_hash,
        "computed_source_dag_hash": computed_source_hash,
        "source_dag_hash_matches": bool(source_hash_matches),
        "source_dag_clean": bool(source_dag_clean),
        "pinned_theory_registry_release": "r1556@bec81e2d",
        "pinned_theory_registry_complete": bool(pinned_theory_registry_complete),
        "pinned_theory_artifact_ids": sorted(pinned_theory_matches),
        "unregistered_lean_node_ids": sorted(unregistered_lean_nodes),
        "collar_layer_hash": supplied_layer_hash,
        "computed_collar_layer_hash": computed_layer_hash,
        "collar_layer_hash_matches": bool(layer_hash_matches),
        "flux_basis_hash": flux_basis_hash,
        "retained_family_hash": supplied_family_hash,
        "computed_retained_family_hash": computed_family_hash,
        "retained_family_hash_matches": bool(family_hash_matches),
        "ambient_dimension": int(ambient_dimension),
        "left_basis_rank": int(basis_analyses["left"]["rank"]),
        "right_basis_rank": int(basis_analyses["right"]["rank"]),
        "flux_basis_rank": int(basis_analyses["flux"]["rank"]),
        "combined_sector_rank": combined_sector_rank,
        "declared_sector_rank_sum": declared_sector_rank_sum,
        "sector_direct_sum_independent": sector_direct_sum_independent,
        "flux_is_proper_ambient_sector": flux_is_proper_ambient_sector,
        "basis_conditioning": basis_conditioning,
        "basis_numerical_conditioning_passed": (
            basis_numerical_conditioning_passed
        ),
        "retained_density_count": len(density_results),
        "estimated_linear_algebra_work": estimated_linear_work,
        "retained_object_semantics": COORDINATE_SEMANTICS,
        "retained_family_complete": bool(retained_family_complete),
        "family_laws_source_derived": bool(family_source_derived),
        "nonvacuous_cross_cut": bool(nonvacuous_cross_cut),
        "all_support_classifications_match": bool(all_support_matches),
        "all_cross_cut_factor_through_flux": bool(all_cross_cut_factor),
        "tolerance": float(tolerance),
        "density_results": density_results,
        "state_side_diagnostics": diagnostics,
        "diagnostics_used_for_promotion": False,
        "mandatory_negative_controls": controls,
        "mandatory_negative_controls_passed": bool(controls_pass),
        "blockers": blockers,
        "claim_boundary": (
            "Bounded finite real-coordinate retained-family packet check over a direct-sum-independent "
            "left/right/flux sector presentation: every computed cross-cut coordinate has an explicit, source-derived, "
            "refinement-natural factorization through the declared flux basis. "
            "Coordinates are not called density operators because Hermiticity, positivity, and unit trace are not encoded or verified. "
            "The r1556 Lean file digests are resolved against a simulator-pinned registry; this binds the "
            "abstract theorem layer but does not authenticate caller-supplied simulation-derivation nodes; "
            "therefore the packet-consistency receipt can pass while the source and independent-run receipts "
            "remain false. "
            "It does not derive the collar clause from Gibbs states, density "
            "matrices, relative entropy, CMI, a conditional expectation, or a "
            "modular centralizer, and it does not establish a continuum source "
            "theorem."
        ),
    }


def write_collar_clause_certificate(
    path: Path,
    packet: Mapping[str, Any],
    *,
    tolerance: float = 1.0e-10,
    max_ambient_dimension: int = 256,
) -> dict[str, Any]:
    """Verify and write one standalone bounded collar-clause receipt."""

    report = verify_collar_clause_packet(
        packet,
        tolerance=tolerance,
        max_ambient_dimension=max_ambient_dimension,
    )
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        destination = destination / "collar_clause_certificate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    replay_path = companion_input_packet_path(
        destination,
        canonical_certificate_filename="collar_clause_certificate.json",
        canonical_input_filename="collar_clause_input_packet.json",
    )
    replay_path.write_text(
        json.dumps(packet, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _required_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _required_hash(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith(_SHA256_PREFIX)
    ):
        raise ValueError(f"{field_name} must be a sha256:<64 lowercase hex> identifier")
    try:
        int(value[len(_SHA256_PREFIX) :], 16)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a sha256:<64 lowercase hex> identifier"
        ) from exc
    if value != value.lower():
        raise ValueError(f"{field_name} must use lowercase hexadecimal")
    if value[len(_SHA256_PREFIX) :] == "0" * 64:
        raise ValueError(f"{field_name} must not use the all-zero placeholder digest")
    return value


def _source_ancestry(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("source_ancestry must be a list")
    if len(value) > MAX_SOURCE_ANCESTRY_NODES:
        raise ValueError("source_ancestry exceeds the bounded packet limit")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        node = _required_mapping(raw, f"source_ancestry[{index}]")
        if set(node) != {"node_id", "kind", "sha256"}:
            raise ValueError(
                f"source_ancestry[{index}] must contain exactly node_id, kind, sha256"
            )
        node_id = node["node_id"]
        kind = node["kind"]
        try:
            _bounded_identifier(node_id, f"source_ancestry[{index}].node_id")
        except ValueError as exc:
            raise ValueError(
                f"source_ancestry[{index}].node_id is invalid or duplicated"
            ) from exc
        if node_id in seen:
            raise ValueError(f"source_ancestry[{index}].node_id is duplicated")
        _bounded_identifier(kind, f"source_ancestry[{index}].kind")
        seen.add(node_id)
        result.append(
            {
                "node_id": node_id,
                "kind": kind,
                "sha256": _required_hash(node["sha256"], "sha256"),
            }
        )
    return result


def _basis(value: Any, dimension: int, field_name: str) -> np.ndarray:
    if not isinstance(value, list) or any(not isinstance(row, list) for row in value):
        raise ValueError(f"{field_name} must be a JSON array of numeric rows")
    if any(
        len(row) != dimension
        or any(
            not _finite_bounded_json_number(item)
            for item in row
        )
        for row in value
    ):
        raise ValueError(
            f"{field_name} must have finite numeric rows of length {dimension}"
        )
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError(f"{field_name} must have shape (n, {dimension})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite numbers")
    return array


def _vector(value: Any, dimension: int, field_name: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != dimension
        or any(not _finite_bounded_json_number(item) for item in value)
    ):
        raise ValueError(
            f"{field_name} must be a finite numeric vector of length {dimension}"
        )
    array = np.asarray(value, dtype=float)
    if array.shape != (dimension,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must be a finite vector of length {dimension}")
    return array


def _estimated_svd_work(basis: np.ndarray) -> int:
    rows, columns = basis.shape
    return int(max(1, rows * columns * min(rows, columns)))


def _basis_analysis(basis: np.ndarray, *, tolerance: float) -> dict[str, Any]:
    """Compute rank, row-span basis, and conditioning from one bounded SVD."""

    try:
        _, singular_values, right_vectors = np.linalg.svd(
            basis,
            full_matrices=False,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("collar basis decomposition did not converge") from exc
    rank = int(np.count_nonzero(singular_values > tolerance))
    orthonormal_span = right_vectors[:rank].T
    maximum = float(singular_values[0]) if singular_values.size else 0.0
    full_row_rank_possible = basis.shape[0] <= basis.shape[1]
    minimum = (
        float(singular_values[-1])
        if singular_values.size and full_row_rank_possible
        else 0.0
    )
    ratio = maximum / minimum if minimum > 0.0 else math.inf
    condition_number = float(ratio) if np.isfinite(ratio) else None
    passed = bool(
        condition_number is not None
        and MIN_BASIS_SINGULAR_VALUE <= minimum <= MAX_BASIS_SINGULAR_VALUE
        and MIN_BASIS_SINGULAR_VALUE <= maximum <= MAX_BASIS_SINGULAR_VALUE
        and condition_number <= MAX_BASIS_CONDITION_NUMBER
    )
    return {
        "rank": rank,
        "orthonormal_span": orthonormal_span,
        "conditioning": {
            "minimum_singular_value": minimum,
            "maximum_singular_value": maximum,
            "condition_number": condition_number,
            "passed": passed,
        },
    }


def _span_residual(vector: np.ndarray, orthonormal_columns: np.ndarray) -> float:
    projection = orthonormal_columns @ (orthonormal_columns.T @ vector)
    return float(np.linalg.norm(projection - vector))


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_SOURCE_ANCESTRY_NODES:
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    for item in value:
        _bounded_identifier(item, f"{field_name} item")
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return list(value)


def _factorization_report(
    value: Any,
    *,
    vector: np.ndarray,
    flux_basis: np.ndarray,
    flux_basis_hash: str,
    source_ids: set[str],
    source_dag_clean: bool,
    tolerance: float,
    field_name: str,
) -> dict[str, Any]:
    empty = {
        "explicit_factorization_present": False,
        "factorization_basis_matches": False,
        "factorization_hash_matches": False,
        "factorization_source_derived": False,
        "factorization_refinement_natural": False,
        "factorization_residual": None,
        "factor_through_flux": False,
    }
    if value is None:
        return empty
    witness = _required_mapping(value, field_name)
    expected_fields = {
        "flux_basis_hash",
        "flux_coefficients",
        "source_derivation_ids",
        "refinement_natural",
        "witness_hash",
    }
    if set(witness) != expected_fields:
        raise ValueError(f"{field_name} has missing or unknown fields")
    witness_basis_hash = _required_hash(witness["flux_basis_hash"], "flux_basis_hash")
    witness_hash = _required_hash(witness["witness_hash"], "witness_hash")
    if (
        not isinstance(witness["flux_coefficients"], list)
        or len(witness["flux_coefficients"]) != flux_basis.shape[0]
        or any(
            not _finite_bounded_json_number(item)
            for item in witness["flux_coefficients"]
        )
    ):
        raise ValueError(
            f"{field_name}.flux_coefficients must be a finite numeric vector of length "
            f"{flux_basis.shape[0]}"
        )
    coefficients = np.asarray(witness["flux_coefficients"], dtype=float)
    if coefficients.shape != (flux_basis.shape[0],) or not np.all(
        np.isfinite(coefficients)
    ):
        raise ValueError(
            f"{field_name}.flux_coefficients must be a finite vector of length "
            f"{flux_basis.shape[0]}"
        )
    derivation_ids = _string_list(
        witness["source_derivation_ids"], f"{field_name}.source_derivation_ids"
    )
    if not isinstance(witness["refinement_natural"], bool):
        raise ValueError(f"{field_name}.refinement_natural must be boolean")
    witness_core = {
        "flux_basis_hash": witness_basis_hash,
        "flux_coefficients": list(witness["flux_coefficients"]),
        "source_derivation_ids": derivation_ids,
        "refinement_natural": witness["refinement_natural"],
    }
    hash_matches = witness_hash == canonical_evidence_hash(witness_core)
    source_derived = (
        bool(derivation_ids)
        and source_dag_clean
        and all(source_id in source_ids for source_id in derivation_ids)
    )
    residual = float(np.linalg.norm(coefficients @ flux_basis - vector))
    factor = bool(
        witness_basis_hash == flux_basis_hash
        and hash_matches
        and source_derived
        and witness["refinement_natural"] is True
        and residual <= tolerance
    )
    return {
        "explicit_factorization_present": True,
        "factorization_basis_matches": bool(witness_basis_hash == flux_basis_hash),
        "factorization_hash_matches": bool(hash_matches),
        "factorization_source_derived": bool(source_derived),
        "factorization_refinement_natural": witness["refinement_natural"],
        "factorization_residual": residual,
        "factor_through_flux": factor,
    }


def _diagnostics(value: Any) -> dict[str, bool]:
    raw = _required_mapping(value, "state_side_diagnostics")
    unknown = set(raw) - set(_DIAGNOSTIC_FIELDS)
    if unknown:
        raise ValueError(
            "state_side_diagnostics has unknown fields: " + ", ".join(sorted(unknown))
        )
    result: dict[str, bool] = {}
    for name in _DIAGNOSTIC_FIELDS:
        field_value = raw.get(name, False)
        if not isinstance(field_value, bool):
            raise ValueError(f"state_side_diagnostics.{name} must be boolean")
        result[name] = field_value
    return result


def _bounded_identifier(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or (byte_length := utf8_byte_length(value)) is None
        or byte_length > MAX_IDENTIFIER_BYTES
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ValueError(
            f"{field_name} must be a nonempty bounded string without control characters"
        )
    return value


def _finite_bounded_json_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(parsed) and abs(parsed) <= MAX_ABS_COORDINATE
