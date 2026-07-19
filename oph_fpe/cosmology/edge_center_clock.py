"""Shared semantics for the selected OPH edge-center repair clock.

The selected theorem target is the orientation half of the full-collar
generator density::

    rho_full = P / 24
    theta = rho_full / 2 = P / 48
    n_s = 1 - theta
    kappa_rep = theta / (P - phi)

This module deliberately separates that conditional theorem target from the
evidence needed to bind a finite simulator clock to it.  A finite-step survival
exponent such as ``-log(lambda_2) / dt`` is a diagnostic estimator; it is not
the full-collar derivative and cannot satisfy these gates by itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from oph_fpe.constants.oph_pixel import (
    OPHPixelConstants,
    P_STAR,
    pixel_value_provenance,
)
from oph_fpe.evidence.hashes import canonical_json_bytes


FULL_COLLAR_DIVISOR = 24.0
ORIENTATION_HALF_COUNT = 2
EDGE_CENTER_DIVISOR = FULL_COLLAR_DIVISOR * ORIENTATION_HALF_COUNT
E_DIAGNOSTIC_KAPPA = math.e
DEFAULT_DEFECT_TOLERANCE = 1.0e-12

FULL_COLLAR_DERIVATIVE_RECEIPT = "FULL_COLLAR_DERIVATIVE_P_OVER_24_RECEIPT"
ORIENTATION_HALF_IDENTITY_RECEIPT = "ORIENTATION_HALF_IDENTITY_RECEIPT"
SEMIGROUP_DEFECT_RECEIPT = "SEMIGROUP_DEFECT_RECEIPT"
REFINEMENT_DEFECT_RECEIPT = "REFINEMENT_DEFECT_RECEIPT"
PHYSICAL_CLOCK_BINDING_RECEIPT = "PHYSICAL_CLOCK_BINDING_RECEIPT"
SOURCE_DAG_CLEAN_RECEIPT = "SOURCE_DAG_CLEAN_RECEIPT"
GENERATIVE_PIXEL_PROFILE_RECEIPT = "GENERATIVE_PIXEL_PROFILE_RECEIPT"
EDGE_CENTER_CLOCK_RECEIPT = "EDGE_CENTER_CLOCK_RECEIPT"
EDGE_CENTER_CLOCK_SCHEMA = "oph-edge-center-clock-v1"

_FORBIDDEN_SOURCE_KINDS = frozenset(
    {
        "measurement",
        "observational_target",
        "fit",
        "likelihood",
        "posterior",
        "residual",
        "data_calibrated_proxy",
        "metadata_unknown",
    }
)
_ALLOWED_SOURCE_KINDS = frozenset(
    {
        "source_theorem",
        "lean_theorem",
        "finite_simulation",
        "refinement_certificate",
        "clock_binding",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "clock_binding_source",
        "full_collar_derivative",
        "orientation_halves",
        "orientation_half_identity_defect",
        "semigroup_defect",
        "refinement_defect",
    }
)

EDGE_CENTER_EVIDENCE_RECEIPTS = (
    FULL_COLLAR_DERIVATIVE_RECEIPT,
    ORIENTATION_HALF_IDENTITY_RECEIPT,
    SEMIGROUP_DEFECT_RECEIPT,
    REFINEMENT_DEFECT_RECEIPT,
    PHYSICAL_CLOCK_BINDING_RECEIPT,
    SOURCE_DAG_CLEAN_RECEIPT,
    GENERATIVE_PIXEL_PROFILE_RECEIPT,
)


@dataclass(frozen=True)
class EdgeCenterClockTarget:
    P: float
    phi: float
    delta_P: float
    full_collar_derivative: float
    orientation_halves: int
    theta: float
    n_s: float
    kappa_rep: float

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "selected_branch": "edge_center_orientation_half",
            "formula": "rho_full=P/24; theta=rho_full/2=P/48; n_s=1-theta",
            "P": self.P,
            "phi": self.phi,
            "delta_P": self.delta_P,
            "full_collar_derivative_target": self.full_collar_derivative,
            "orientation_halves": self.orientation_halves,
            "theta": self.theta,
            "eta_R": self.theta,
            "n_s": self.n_s,
            "kappa_rep": self.kappa_rep,
            "kappa_rep_formula": "(P/48)/(P-phi)",
            "selected_theorem_target": True,
            "finite_lattice_derived": False,
            "diagnostic_controls": {
                "e": {
                    "kappa_rep": E_DIAGNOSTIC_KAPPA,
                    "eta_R": E_DIAGNOSTIC_KAPPA * self.delta_P,
                    "n_s": 1.0 - E_DIAGNOSTIC_KAPPA * self.delta_P,
                    "role": "named_nonpromoting_diagnostic_control",
                    "selected": False,
                    "required": False,
                    "canonical": False,
                    "promoting": False,
                }
            },
        }


def edge_center_clock_target(P: float = P_STAR) -> EdgeCenterClockTarget:
    pixel = OPHPixelConstants(P=float(P))
    delta_p = float(pixel.P - pixel.phi)
    if not math.isfinite(delta_p) or delta_p <= 0.0:
        raise ValueError("the selected edge-center branch requires finite P > phi")
    full_collar = float(pixel.P / FULL_COLLAR_DIVISOR)
    theta = float(full_collar / ORIENTATION_HALF_COUNT)
    return EdgeCenterClockTarget(
        P=float(pixel.P),
        phi=float(pixel.phi),
        delta_P=delta_p,
        full_collar_derivative=full_collar,
        orientation_halves=ORIENTATION_HALF_COUNT,
        theta=theta,
        n_s=float(1.0 - theta),
        kappa_rep=float(theta / delta_p),
    )


def validate_edge_center_clock_evidence(
    evidence: Mapping[str, Any] | None = None,
    *,
    P: float = P_STAR,
    tolerance: float = DEFAULT_DEFECT_TOLERANCE,
) -> dict[str, Any]:
    """Validate the evidence bundle that binds a finite clock to ``P/48``.

    Receipt declarations and their underlying values are both required.  A
    missing mapping therefore returns every gate as ``False``.  Hash-bound
    clock/source records are required so copied booleans cannot promote a
    diagnostic transition matrix.
    """

    target = edge_center_clock_target(P)
    source = dict(evidence) if isinstance(evidence, Mapping) else {}
    tol = _finite_nonnegative(tolerance)
    if tol is None:
        raise ValueError("tolerance must be finite and nonnegative")

    binding_payload = source.get("clock_binding_payload")
    binding = dict(binding_payload) if isinstance(binding_payload, Mapping) else {}
    source_dag_payload = source.get("source_dag")
    source_dag = dict(source_dag_payload) if isinstance(source_dag_payload, Mapping) else {}
    clock_binding_sha256 = _normalized_sha256(source.get("clock_binding_sha256"))
    source_dag_sha256 = _normalized_sha256(source.get("source_dag_sha256"))
    computed_clock_binding_sha256 = canonical_edge_clock_hash(binding) if binding else None
    computed_source_dag_sha256 = canonical_edge_clock_hash(source_dag) if source_dag else None
    binding_structure_valid = bool(binding and set(binding) == _BINDING_FIELDS)
    clock_binding_hash_matches = bool(
        binding_structure_valid
        and _is_sha256(clock_binding_sha256)
        and clock_binding_sha256 == computed_clock_binding_sha256
    )
    source_dag_hash_matches = bool(
        _is_sha256(source_dag_sha256)
        and source_dag_sha256 == computed_source_dag_sha256
    )
    source_dag_audit = _audit_source_dag(source_dag)
    pixel_provenance = pixel_value_provenance(
        target.P,
        source.get("pixel_profile"),
    )

    full_collar_value = _finite_or_none(binding.get("full_collar_derivative"))
    orientation_halves = _int_or_none(binding.get("orientation_halves"))
    orientation_defect = _finite_nonnegative(binding.get("orientation_half_identity_defect"))
    semigroup_defect = _finite_nonnegative(binding.get("semigroup_defect"))
    refinement_defect = _finite_nonnegative(binding.get("refinement_defect"))
    clock_binding_source = binding.get("clock_binding_source")

    checks = {
        "full_collar_derivative_equals_P_over_24": bool(
            source.get(FULL_COLLAR_DERIVATIVE_RECEIPT) is True
            and clock_binding_hash_matches
            and full_collar_value is not None
            and math.isclose(
                full_collar_value,
                target.full_collar_derivative,
                rel_tol=tol,
                abs_tol=tol,
            )
        ),
        "orientation_half_identity": bool(
            source.get(ORIENTATION_HALF_IDENTITY_RECEIPT) is True
            and clock_binding_hash_matches
            and orientation_halves == ORIENTATION_HALF_COUNT
            and orientation_defect is not None
            and orientation_defect <= tol
        ),
        "semigroup_defect_bounded": bool(
            source.get(SEMIGROUP_DEFECT_RECEIPT) is True
            and clock_binding_hash_matches
            and semigroup_defect is not None
            and semigroup_defect <= tol
        ),
        "refinement_defect_bounded": bool(
            source.get(REFINEMENT_DEFECT_RECEIPT) is True
            and clock_binding_hash_matches
            and refinement_defect is not None
            and refinement_defect <= tol
        ),
        "physical_clock_binding_hash_bound": bool(
            source.get(PHYSICAL_CLOCK_BINDING_RECEIPT) is True
            and clock_binding_hash_matches
            and isinstance(clock_binding_source, str)
            and bool(clock_binding_source.strip())
        ),
        "source_dag_clean_hash_bound": bool(
            source.get(SOURCE_DAG_CLEAN_RECEIPT) is True
            and source_dag_hash_matches
            and source_dag_audit["clean"]
        ),
        "generative_pixel_profile_bound": bool(
            source.get(GENERATIVE_PIXEL_PROFILE_RECEIPT) is True
            and pixel_provenance[GENERATIVE_PIXEL_PROFILE_RECEIPT]
        ),
    }
    receipts = {
        FULL_COLLAR_DERIVATIVE_RECEIPT: checks["full_collar_derivative_equals_P_over_24"],
        ORIENTATION_HALF_IDENTITY_RECEIPT: checks["orientation_half_identity"],
        SEMIGROUP_DEFECT_RECEIPT: checks["semigroup_defect_bounded"],
        REFINEMENT_DEFECT_RECEIPT: checks["refinement_defect_bounded"],
        PHYSICAL_CLOCK_BINDING_RECEIPT: checks["physical_clock_binding_hash_bound"],
        SOURCE_DAG_CLEAN_RECEIPT: checks["source_dag_clean_hash_bound"],
        GENERATIVE_PIXEL_PROFILE_RECEIPT: checks["generative_pixel_profile_bound"],
    }
    complete = all(receipts.values())
    return {
        "schema": EDGE_CENTER_CLOCK_SCHEMA,
        "selected_target": target.as_jsonable(),
        "evidence_present": bool(source),
        "tolerance": tol,
        "observed": {
            "full_collar_derivative": full_collar_value,
            "orientation_halves": orientation_halves,
            "orientation_half_identity_defect": orientation_defect,
            "semigroup_defect": semigroup_defect,
            "refinement_defect": refinement_defect,
            "clock_binding_source": clock_binding_source,
            "clock_binding_sha256": clock_binding_sha256,
            "computed_clock_binding_sha256": computed_clock_binding_sha256,
            "clock_binding_hash_matches": clock_binding_hash_matches,
            "source_dag_sha256": source_dag_sha256,
            "computed_source_dag_sha256": computed_source_dag_sha256,
            "source_dag_hash_matches": source_dag_hash_matches,
            "source_dag_audit": source_dag_audit,
            "pixel_provenance": pixel_provenance,
        },
        "checks": checks,
        "receipts": receipts,
        **receipts,
        EDGE_CENTER_CLOCK_RECEIPT: complete,
        "edge_center_clock_evidence_complete": complete,
        "missing_receipts": [name for name, passed in receipts.items() if not passed],
        "finite_step_survival_exponent_is_distinct": True,
        "claim_boundary": (
            "The selected P/48 theorem target is conditional on a source-derived full-collar "
            "derivative and its orientation-half identity. Finite-step survival exponents, fitted "
            "decay slopes, and numerical proximity to the target remain diagnostic until all "
            "semigroup, refinement, clock-binding, source-DAG, and generative P-profile receipts pass."
        ),
    }


def canonical_edge_clock_hash(value: Any) -> str:
    """Return the canonical content hash used by standalone clock packets."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_edge_center_clock_certificate(
    path: Path,
    evidence: Mapping[str, Any],
    *,
    P: float = P_STAR,
    tolerance: float = DEFAULT_DEFECT_TOLERANCE,
) -> dict[str, Any]:
    """Validate and write one standalone fail-closed clock certificate."""

    report = validate_edge_center_clock_evidence(evidence, P=P, tolerance=tolerance)
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        destination = destination / "edge_center_clock_certificate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _audit_source_dag(source_dag: Mapping[str, Any]) -> dict[str, Any]:
    if set(source_dag) - {"nodes", "edges"}:
        blockers: list[str] = ["source_dag_unknown_field"]
    else:
        blockers = []
    nodes = source_dag.get("nodes")
    edges = source_dag.get("edges", [])
    if not isinstance(nodes, list) or not nodes:
        blockers.append("source_dag_nodes_missing")
        nodes = []
    if not isinstance(edges, list):
        blockers.append("source_dag_edges_invalid")
        edges = []

    node_ids: list[str] = []
    kinds: list[str] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            blockers.append("source_dag_node_invalid")
            continue
        node_id = node.get("id")
        kind = node.get("kind")
        digest = node.get("sha256")
        if set(node) != {"id", "kind", "sha256"}:
            blockers.append("source_dag_node_fields_invalid")
        if not isinstance(node_id, str) or not node_id.strip():
            blockers.append("source_dag_node_id_invalid")
            continue
        if not isinstance(kind, str) or not kind.strip():
            blockers.append("source_dag_node_kind_invalid")
            continue
        node_ids.append(node_id)
        kinds.append(kind)
        if kind in _FORBIDDEN_SOURCE_KINDS:
            blockers.append(f"forbidden_source_kind:{kind}")
        elif kind not in _ALLOWED_SOURCE_KINDS:
            blockers.append(f"unrecognized_source_kind:{kind}")
        if _normalized_sha256(digest) is None:
            blockers.append("source_dag_node_hash_invalid")
    if len(node_ids) != len(set(node_ids)):
        blockers.append("source_dag_duplicate_node_id")

    adjacency = {node_id: [] for node_id in node_ids}
    for edge in edges:
        if not isinstance(edge, Mapping):
            blockers.append("source_dag_edge_invalid")
            continue
        parent = edge.get("from")
        child = edge.get("to")
        if set(edge) != {"from", "to"}:
            blockers.append("source_dag_edge_fields_invalid")
        if parent not in adjacency or child not in adjacency:
            blockers.append("source_dag_edge_has_unknown_endpoint")
            continue
        adjacency[parent].append(child)
    if _has_cycle(adjacency):
        blockers.append("source_dag_cycle")
    unique_blockers = sorted(set(blockers))
    return {
        "clean": not unique_blockers,
        "node_count": len(node_ids),
        "edge_count": sum(len(children) for children in adjacency.values()),
        "source_kinds": sorted(set(kinds)),
        "blockers": unique_blockers,
    }


def _has_cycle(adjacency: Mapping[str, list[str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(child) for child in adjacency.get(node, [])):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in adjacency)


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_nonnegative(value: Any) -> float | None:
    result = _finite_or_none(value)
    return result if result is not None and result >= 0.0 else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result == value else None


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    token = value.removeprefix("sha256:")
    return bool(
        len(token) == 64
        and token != "0" * 64
        and all(character in "0123456789abcdefABCDEF" for character in token)
    )


def _normalized_sha256(value: Any) -> str | None:
    if not _is_sha256(value) or value != value.lower():
        return None
    return value
