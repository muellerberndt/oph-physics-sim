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
from oph_fpe.evidence.artifact_paths import companion_input_packet_path
from oph_fpe.evidence.hashes import canonical_json_bytes
from oph_fpe.evidence.validation import utf8_byte_length


FULL_COLLAR_DIVISOR = 24.0
ORIENTATION_HALF_COUNT = 2
EDGE_CENTER_DIVISOR = FULL_COLLAR_DIVISOR * ORIENTATION_HALF_COUNT
E_DIAGNOSTIC_KAPPA = math.e
DEFAULT_DEFECT_TOLERANCE = 1.0e-12
MAX_DEFECT_TOLERANCE = 1.0e-6
DEFECT_TOLERANCE_PROFILE = "oph-edge-center-clock-defect-profile-v1"
MAX_SOURCE_DAG_NODES = 256
MAX_SOURCE_DAG_EDGES = 1024
MAX_EVIDENCE_PACKET_BYTES = 2_000_000
MAX_EVIDENCE_JSON_NODES = 100_000
MAX_EVIDENCE_JSON_DEPTH = 32
MAX_EVIDENCE_CONTAINER_ITEMS = 4096
MAX_EVIDENCE_STRING_BYTES = 32_768

FULL_COLLAR_DERIVATIVE_RECEIPT = "FULL_COLLAR_DERIVATIVE_P_OVER_24_RECEIPT"
ORIENTATION_HALF_IDENTITY_RECEIPT = "ORIENTATION_HALF_IDENTITY_RECEIPT"
SEMIGROUP_DEFECT_RECEIPT = "SEMIGROUP_DEFECT_RECEIPT"
REFINEMENT_DEFECT_RECEIPT = "REFINEMENT_DEFECT_RECEIPT"
PHYSICAL_CLOCK_BINDING_RECEIPT = "PHYSICAL_CLOCK_BINDING_RECEIPT"
CLOCK_BINDING_PACKET_CONSISTENCY_RECEIPT = (
    "CLOCK_BINDING_PACKET_CONSISTENCY_RECEIPT"
)
SOURCE_DAG_CLEAN_RECEIPT = "SOURCE_DAG_CLEAN_RECEIPT"
GENERATIVE_PIXEL_PROFILE_RECEIPT = "GENERATIVE_PIXEL_PROFILE_RECEIPT"
INDEPENDENT_FINITE_RUN_REPLAY_RECEIPT = "INDEPENDENT_FINITE_RUN_REPLAY_RECEIPT"
EDGE_CENTER_CLOCK_PACKET_CONSISTENCY_RECEIPT = (
    "EDGE_CENTER_CLOCK_PACKET_CONSISTENCY_RECEIPT"
)
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
_ROOT_SOURCE_KINDS = frozenset({"source_theorem", "lean_theorem"})
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
    INDEPENDENT_FINITE_RUN_REPLAY_RECEIPT,
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
    if evidence is not None and not isinstance(evidence, Mapping):
        raise ValueError("edge-center evidence must be a mapping")
    source = dict(evidence) if isinstance(evidence, Mapping) else {}
    _bounded_canonical_json_bytes(source)
    tol = _finite_nonnegative(tolerance)
    if tol is None or tol > MAX_DEFECT_TOLERANCE:
        raise ValueError(
            "tolerance must be a finite nonnegative number no greater than "
            f"{MAX_DEFECT_TOLERANCE:g}"
        )

    binding_payload = source.get("clock_binding_payload")
    binding = dict(binding_payload) if isinstance(binding_payload, Mapping) else {}
    source_dag_payload = source.get("source_dag")
    source_dag = dict(source_dag_payload) if isinstance(source_dag_payload, Mapping) else {}
    clock_binding_sha256 = _normalized_sha256(source.get("clock_binding_sha256"))
    source_dag_sha256 = _normalized_sha256(source.get("source_dag_sha256"))
    computed_clock_binding_sha256 = canonical_edge_clock_hash(binding) if binding else None
    computed_source_dag_sha256 = canonical_edge_clock_hash(source_dag) if source_dag else None
    clock_binding_source = binding.get("clock_binding_source")
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
    source_dag_audit = _audit_source_dag(
        source_dag,
        binding_source=clock_binding_source,
        binding_sha256=clock_binding_sha256,
    )
    pixel_provenance = pixel_value_provenance(
        target.P,
        source.get("pixel_profile"),
    )

    full_collar_value = _finite_or_none(binding.get("full_collar_derivative"))
    orientation_halves = _int_or_none(binding.get("orientation_halves"))
    orientation_defect = _bounded_defect_or_none(
        binding.get("orientation_half_identity_defect")
    )
    semigroup_defect = _bounded_defect_or_none(binding.get("semigroup_defect"))
    refinement_defect = _bounded_defect_or_none(binding.get("refinement_defect"))

    checks = {
        "full_collar_derivative_equals_P_over_24": bool(
            clock_binding_hash_matches
            and full_collar_value is not None
            and math.isclose(
                full_collar_value,
                target.full_collar_derivative,
                rel_tol=tol,
                abs_tol=tol,
            )
        ),
        "orientation_half_identity": bool(
            clock_binding_hash_matches
            and orientation_halves == ORIENTATION_HALF_COUNT
            and orientation_defect is not None
            and orientation_defect <= tol
        ),
        "semigroup_defect_bounded": bool(
            clock_binding_hash_matches
            and semigroup_defect is not None
            and semigroup_defect <= tol
        ),
        "refinement_defect_bounded": bool(
            clock_binding_hash_matches
            and refinement_defect is not None
            and refinement_defect <= tol
        ),
        "clock_binding_packet_consistency": bool(
            clock_binding_hash_matches
            and source_dag_hash_matches
            and source_dag_audit["binding_node_present"]
            and source_dag_audit["binding_node_kind_valid"]
            and source_dag_audit["binding_node_hash_matches"]
            and source_dag_audit["binding_node_is_leaf"]
            and source_dag_audit["root_source_reaches_binding"]
        ),
        "source_dag_clean_hash_bound": bool(
            source_dag_hash_matches
            and source_dag_audit["clean"]
        ),
        "generative_pixel_profile_bound": bool(
            pixel_provenance[GENERATIVE_PIXEL_PROFILE_RECEIPT]
        ),
        "independent_finite_run_replay": False,
    }
    receipts = {
        FULL_COLLAR_DERIVATIVE_RECEIPT: checks["full_collar_derivative_equals_P_over_24"],
        ORIENTATION_HALF_IDENTITY_RECEIPT: checks["orientation_half_identity"],
        SEMIGROUP_DEFECT_RECEIPT: checks["semigroup_defect_bounded"],
        REFINEMENT_DEFECT_RECEIPT: checks["refinement_defect_bounded"],
        PHYSICAL_CLOCK_BINDING_RECEIPT: False,
        SOURCE_DAG_CLEAN_RECEIPT: checks["source_dag_clean_hash_bound"],
        GENERATIVE_PIXEL_PROFILE_RECEIPT: checks["generative_pixel_profile_bound"],
        INDEPENDENT_FINITE_RUN_REPLAY_RECEIPT: checks[
            "independent_finite_run_replay"
        ],
    }
    packet_consistency = all(
        checks[name]
        for name in (
            "full_collar_derivative_equals_P_over_24",
            "orientation_half_identity",
            "semigroup_defect_bounded",
            "refinement_defect_bounded",
            "clock_binding_packet_consistency",
            "source_dag_clean_hash_bound",
            "generative_pixel_profile_bound",
        )
    )
    complete = all(receipts.values())
    return {
        "schema": EDGE_CENTER_CLOCK_SCHEMA,
        "selected_target": target.as_jsonable(),
        "evidence_present": bool(source),
        "tolerance": tol,
        "tolerance_profile": {
            "profile": DEFECT_TOLERANCE_PROFILE,
            "default": DEFAULT_DEFECT_TOLERANCE,
            "maximum": MAX_DEFECT_TOLERANCE,
        },
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
        CLOCK_BINDING_PACKET_CONSISTENCY_RECEIPT: checks[
            "clock_binding_packet_consistency"
        ],
        EDGE_CENTER_CLOCK_PACKET_CONSISTENCY_RECEIPT: packet_consistency,
        "legacy_receipt_declarations_promoted": False,
        EDGE_CENTER_CLOCK_RECEIPT: complete,
        "edge_center_clock_evidence_complete": complete,
        "missing_receipts": [name for name, passed in receipts.items() if not passed],
        "finite_step_survival_exponent_is_distinct": True,
        "claim_boundary": (
            "The selected P/48 theorem target is conditional on a source-derived full-collar "
            "derivative and its orientation-half identity. Finite-step survival exponents, fitted "
            "decay slopes, and numerical proximity to the target remain diagnostic until all "
            "semigroup, refinement, clock-binding, source-DAG, and generative P-profile receipts pass."
            " Hash equality establishes packet-internal content binding only. The physical clock "
            "receipt remains false until raw finite-run artifacts independently reproduce the "
            "derivative and defect bounds; caller receipt booleans are ignored."
        ),
    }


def canonical_edge_clock_hash(value: Any) -> str:
    """Return the canonical content hash used by standalone clock packets."""

    return "sha256:" + hashlib.sha256(_bounded_canonical_json_bytes(value)).hexdigest()


def independently_replayed_edge_clock_receipt(
    _persisted_report: Mapping[str, Any] | None,
) -> bool:
    """Return the physical clock status available to persisted-report readers.

    Packet hash consistency is not raw-run replay.  Until a resolver can load
    and recompute the finite transition artifacts named by a report, no
    persisted aggregate boolean is eligible for downstream promotion.
    """

    return False


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
    replay_path = companion_input_packet_path(
        destination,
        canonical_certificate_filename="edge_center_clock_certificate.json",
        canonical_input_filename="edge_center_clock_input_packet.json",
    )
    replay_path.write_text(
        json.dumps(
            {"evidence": evidence, "P": P, "tolerance": tolerance},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _audit_source_dag(
    source_dag: Mapping[str, Any],
    *,
    binding_source: Any,
    binding_sha256: str | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    if set(source_dag) != {"nodes", "edges"}:
        blockers.append("source_dag_fields_invalid")
    nodes = source_dag.get("nodes")
    edges = source_dag.get("edges")
    if not isinstance(nodes, list) or not nodes:
        blockers.append("source_dag_nodes_missing")
        nodes = []
    elif len(nodes) > MAX_SOURCE_DAG_NODES:
        blockers.append("source_dag_node_limit_exceeded")
        nodes = []
    if not isinstance(edges, list):
        blockers.append("source_dag_edges_invalid")
        edges = []
    elif len(edges) > MAX_SOURCE_DAG_EDGES:
        blockers.append("source_dag_edge_limit_exceeded")
        edges = []

    kinds: list[str] = []
    nodes_by_id: dict[str, Mapping[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, Mapping):
            blockers.append("source_dag_node_invalid")
            continue
        node_id = node.get("id")
        kind = node.get("kind")
        digest = node.get("sha256")
        if set(node) != {"id", "kind", "sha256"}:
            blockers.append("source_dag_node_fields_invalid")
        if not _valid_node_id(node_id):
            blockers.append("source_dag_node_id_invalid")
            continue
        if not isinstance(kind, str) or kind not in _ALLOWED_SOURCE_KINDS:
            blockers.append("source_dag_node_kind_invalid")
        if node_id in nodes_by_id:
            blockers.append("source_dag_duplicate_node_id")
        else:
            nodes_by_id[node_id] = node
        if isinstance(kind, str):
            kinds.append(kind)
        if kind in _FORBIDDEN_SOURCE_KINDS:
            blockers.append(f"forbidden_source_kind:{kind}")
        elif kind not in _ALLOWED_SOURCE_KINDS:
            blockers.append(f"unrecognized_source_kind:{kind}")
        if _normalized_sha256(digest) is None:
            blockers.append("source_dag_node_hash_invalid")
    adjacency = {node_id: [] for node_id in nodes_by_id}
    incoming = {node_id: [] for node_id in nodes_by_id}
    edge_pairs: set[tuple[str, str]] = set()
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
        pair = (parent, child)
        if pair in edge_pairs:
            blockers.append("source_dag_duplicate_edge")
            continue
        edge_pairs.add(pair)
        adjacency[parent].append(child)
        incoming[child].append(parent)
    if _has_cycle(adjacency):
        blockers.append("source_dag_cycle")

    binding_node_present = bool(
        _valid_node_id(binding_source) and binding_source in nodes_by_id
    )
    binding_node = nodes_by_id.get(binding_source) if binding_node_present else None
    binding_node_kind_valid = bool(
        binding_node is not None and binding_node.get("kind") == "clock_binding"
    )
    binding_node_hash_matches = bool(
        binding_node is not None
        and _is_sha256(binding_sha256)
        and _normalized_sha256(binding_node.get("sha256")) == binding_sha256
    )
    binding_node_is_leaf = bool(
        binding_node_present and not adjacency.get(str(binding_source), [])
    )
    root_ids = {
        node_id
        for node_id, node in nodes_by_id.items()
        if node.get("kind") in _ROOT_SOURCE_KINDS and not incoming.get(node_id)
    }
    root_source_reaches_binding = bool(
        binding_node_present
        and any(
            _path_exists(adjacency, root_id, str(binding_source))
            for root_id in root_ids
        )
    )
    all_nodes_ancestral_to_binding = bool(
        binding_node_present
        and nodes_by_id
        and all(
            _path_exists(adjacency, node_id, str(binding_source))
            for node_id in nodes_by_id
        )
    )
    if not binding_node_present:
        blockers.append("clock_binding_source_node_missing")
    if binding_node_present and not binding_node_kind_valid:
        blockers.append("clock_binding_source_kind_invalid")
    if binding_node_present and not binding_node_hash_matches:
        blockers.append("clock_binding_source_hash_mismatch")
    if binding_node_present and not binding_node_is_leaf:
        blockers.append("clock_binding_source_not_leaf")
    if binding_node_present and not root_source_reaches_binding:
        blockers.append("source_theorem_does_not_reach_clock_binding")
    if binding_node_present and not all_nodes_ancestral_to_binding:
        blockers.append("source_dag_contains_nonancestral_node")
    unique_blockers = sorted(set(blockers))
    return {
        "clean": not unique_blockers,
        "node_count": len(source_dag.get("nodes", []))
        if isinstance(source_dag.get("nodes"), list)
        else 0,
        "edge_count": len(source_dag.get("edges", []))
        if isinstance(source_dag.get("edges"), list)
        else 0,
        "source_kinds": sorted(set(kinds)),
        "binding_node_present": binding_node_present,
        "binding_node_kind_valid": binding_node_kind_valid,
        "binding_node_hash_matches": binding_node_hash_matches,
        "binding_node_is_leaf": binding_node_is_leaf,
        "root_source_reaches_binding": root_source_reaches_binding,
        "all_nodes_ancestral_to_binding": all_nodes_ancestral_to_binding,
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


def _path_exists(
    adjacency: Mapping[str, list[str]],
    start: str,
    target: str,
) -> bool:
    pending = [start]
    visited: set[str] = set()
    while pending:
        node = pending.pop()
        if node == target:
            return True
        if node in visited:
            continue
        visited.add(node)
        pending.extend(adjacency.get(node, []))
    return False


def _valid_node_id(value: Any) -> bool:
    if not isinstance(value, str) or value != value.strip():
        return False
    if not value or len(value) > 128:
        return False
    return value[0].isalnum() and all(
        character.isalnum() or character in "._:-" for character in value
    )


def _bounded_canonical_json_bytes(value: Any) -> bytes:
    pending: list[tuple[Any, int]] = [(value, 0)]
    node_count = 0
    while pending:
        item, depth = pending.pop()
        node_count += 1
        if node_count > MAX_EVIDENCE_JSON_NODES:
            raise ValueError("edge-center evidence exceeds the JSON node budget")
        if depth > MAX_EVIDENCE_JSON_DEPTH:
            raise ValueError("edge-center evidence exceeds the JSON depth budget")
        if item is None or isinstance(item, (bool, int)):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("edge-center evidence contains a nonfinite number")
            continue
        if isinstance(item, str):
            byte_length = utf8_byte_length(item)
            if byte_length is None or byte_length > MAX_EVIDENCE_STRING_BYTES:
                raise ValueError("edge-center evidence contains an oversized string")
            continue
        if isinstance(item, Mapping):
            if (
                len(item) > MAX_EVIDENCE_CONTAINER_ITEMS
                or not all(isinstance(key, str) for key in item)
                or any(
                    (key_bytes := utf8_byte_length(key)) is None
                    or key_bytes > MAX_EVIDENCE_STRING_BYTES
                    for key in item
                )
            ):
                raise ValueError("edge-center evidence contains an invalid mapping")
            pending.extend((child, depth + 1) for child in item.values())
            continue
        if isinstance(item, (list, tuple)):
            if len(item) > MAX_EVIDENCE_CONTAINER_ITEMS:
                raise ValueError("edge-center evidence contains an oversized array")
            pending.extend((child, depth + 1) for child in item)
            continue
        raise ValueError("edge-center evidence contains unsupported JSON data")
    try:
        encoded = canonical_json_bytes(value)
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("edge-center evidence is not canonical JSON data") from exc
    if len(encoded) > MAX_EVIDENCE_PACKET_BYTES:
        raise ValueError("edge-center evidence exceeds the packet byte budget")
    return encoded


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_nonnegative(value: Any) -> float | None:
    result = _finite_or_none(value)
    return result if result is not None and result >= 0.0 else None


def _bounded_defect_or_none(value: Any) -> float | None:
    result = _finite_nonnegative(value)
    return (
        result
        if result is not None and result <= MAX_DEFECT_TOLERANCE
        else None
    )


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
