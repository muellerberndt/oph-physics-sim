"""CR-0 capability inventory for the common-reserve campaign.

This module inventories committed simulator objects.  It does not manufacture
missing source fields, wrap controls as native producers, or run an estimator.
Every source is raw-byte pinned, the audited Python tree is committed by a
deterministic digest, and a separate implementation rechecks the matrix.

The classifications have deliberately narrow meanings:

``AVAILABLE_SIMULATOR_NATIVE``
    The current simulator emits the required finite object from its own state
    or transition data.  This does not establish the corresponding physical
    theorem binding.
``AVAILABLE_CONDITIONAL``
    The finite object exists, while a named physical identity or scale binding
    remains a premise.
``AVAILABLE_CONTROL_ONLY``
    The implementation is useful only as a declared comparator or plumbing
    control.
``MISSING``
    At least one required computational object has no producer in the audited
    catalog.  A placeholder cannot change this classification.
``AMBIGUOUS``
    More than one incompatible source object exists and no selector identifies
    the required one.  CR-0 does not use this exit for the current snapshot.
"""

from __future__ import annotations

import argparse
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator


SCHEMA = "oph.common-reserve.capability-matrix.v1"
ISSUE = 660
STAGE = "CR-0"

AVAILABLE_SIMULATOR_NATIVE = "AVAILABLE_SIMULATOR_NATIVE"
AVAILABLE_CONDITIONAL = "AVAILABLE_CONDITIONAL"
AVAILABLE_CONTROL_ONLY = "AVAILABLE_CONTROL_ONLY"
MISSING = "MISSING"
AMBIGUOUS = "AMBIGUOUS"
CLASSIFICATIONS = (
    AVAILABLE_SIMULATOR_NATIVE,
    AVAILABLE_CONDITIONAL,
    AVAILABLE_CONTROL_ONLY,
    MISSING,
    AMBIGUOUS,
)

CAPABILITY_IDS = (
    "physical_quotient",
    "primitive_repair_law",
    "boundary_sector",
    "scalar_register",
    "protected_z6_reserve",
    "scalar_reserve_coregistration",
    "source_ensemble_action",
    "refinement_tower_physical_scale_ratios",
    "support_geometry",
    "full_half_collars",
    "raw_twelve_port_response",
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas/capability.schema.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data/common_reserve/producer_capability_matrix.json"
DEFAULT_REPORT = REPOSITORY_ROOT / "data/common_reserve/producer_capability_matrix.md"

SOURCE_PATHS = {
    "carrier_manifest": "tests/fixtures/echosahedral_federation_reference.json",
    "charged_response_producer": "oph_fpe/core/charged_response.py",
    "charged_response_artifact": "data/common_reserve/charged_response_artifact.json",
    "canonical_repair_producer": "oph_fpe/dynamics/canonical_seam_repair.py",
    "bounded_repair_receipt": (
        "data/repair_closure/bounded_atomic_self_readback_closure_receipt.json"
    ),
    "record_counting_projection": (
        "data/repair_closure/record_counting_source_projection.json"
    ),
    "normal_form_checker": "oph_fpe/quotient/observable_normal_form.py",
    "finite_source_manifest": (
        "data/earned_runs/oph_universe_64k_3p1d_reearned/"
        "finite_consensus_source_manifest.json"
    ),
    "finite_consensus_replay_report": (
        "data/earned_runs/oph_universe_64k_3p1d_reearned/"
        "finite_consensus_replay_report.json"
    ),
    "defect_sector_producer": "oph_fpe/local_domain/defect_sector_spectra.py",
    "defect_sector_receipt": "data/local_domain/defect_sector_receipt.json",
    "stage1_receipt": "data/local_domain/stage1_receipt.json",
    "stage1_arrays": "data/local_domain/stage1_arrays.npz.gz",
    "stage3_receipt": "data/local_domain/stage3_receipt.json",
    "reference_ensemble_producer": "oph_fpe/ensembles/reference_vacuum.py",
    "collar_clause_checker": "oph_fpe/cosmology/collar_clause.py",
}

# These are the scientific producers whose import closures are relevant to
# rows classified as native or conditional.  The inventory module itself is
# excluded because it necessarily carries the firewall vocabulary.
ANCESTRY_ROOT_MODULES = (
    "oph_fpe.core.charged_response",
    "oph_fpe.dynamics.canonical_seam_repair",
    "oph_fpe.local_domain.defect_sector_spectra",
)
FORBIDDEN_MODULE_PREFIXES = (
    "oph_fpe.constants.oph_pixel",
    "oph_fpe.cosmology.edge_center_clock",
    "oph_fpe.cosmology.scalar_repair_semigroup",
    "oph_fpe.cosmology.source_screen_spectrum",
    "oph_fpe.cosmology.oph_screen_power",
    "common_reserve_closure.closure",
)
FORBIDDEN_LITERAL_PATTERNS = {
    "P_STAR": r"\bP_STAR\b",
    "CODATA": r"\bCODATA\b",
    "Planck": r"\bPlanck\b",
    "ACT": r"\bACT\b",
    "DESI": r"\bDESI\b",
    "familiar_P_decimal": r"1\.630968",
    "P_over_24": r"P\s*/\s*24",
    "P_over_48": r"P\s*/\s*48",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _raw_pin(relative: str) -> dict[str, Any]:
    path = (REPOSITORY_ROOT / relative).resolve(strict=True)
    path.relative_to(REPOSITORY_ROOT.resolve())
    raw = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _load_json(source_id: str) -> dict[str, Any]:
    value = json.loads((REPOSITORY_ROOT / SOURCE_PATHS[source_id]).read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{source_id} must contain a JSON object")
    return value


def _python_tree_snapshot() -> dict[str, Any]:
    rows = []
    for path in sorted((REPOSITORY_ROOT / "oph_fpe").rglob("*.py")):
        raw = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "bytes": len(raw),
                "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            }
        )
    return {
        "root": "oph_fpe",
        "file_count": len(rows),
        "tree_sha256": _value_sha256(rows),
    }


def _module_name(path: Path) -> str:
    relative = path.relative_to(REPOSITORY_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _local_import_graph() -> dict[str, set[str]]:
    module_paths = {
        _module_name(path): path
        for path in (REPOSITORY_ROOT / "oph_fpe").rglob("*.py")
    }
    graph: dict[str, set[str]] = {module: set() for module in module_paths}
    for module, path in module_paths.items():
        tree = ast.parse(path.read_text("utf-8"), filename=str(path))
        package = module.split(".")[:-1]
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[: len(package) - node.level + 1]
                    if node.module:
                        base.extend(node.module.split("."))
                    names.append(".".join(base))
                elif node.module:
                    names.append(node.module)
            for name in names:
                if name in module_paths:
                    graph[module].add(name)
                else:
                    candidates = [item for item in module_paths if item.startswith(name + ".")]
                    graph[module].update(candidates)
    return graph


def _ancestry_firewall() -> dict[str, Any]:
    graph = _local_import_graph()
    visited: set[str] = set()
    queue = deque(ANCESTRY_ROOT_MODULES)
    while queue:
        module = queue.popleft()
        if module in visited:
            continue
        visited.add(module)
        queue.extend(sorted(graph.get(module, ())))
    forbidden = sorted(
        module
        for module in visited
        if any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in FORBIDDEN_MODULE_PREFIXES
        )
    )
    edge_rows = [
        [source, target]
        for source in sorted(visited)
        for target in sorted(graph.get(source, ()))
        if target in visited
    ]
    literal_hits = []
    for module in sorted(visited):
        path = REPOSITORY_ROOT / (module.replace(".", "/") + ".py")
        if not path.is_file():
            path = REPOSITORY_ROOT / module.replace(".", "/") / "__init__.py"
        source = path.read_text("utf-8")
        for pattern_id, pattern in FORBIDDEN_LITERAL_PATTERNS.items():
            if re.search(pattern, source):
                literal_hits.append(f"{module}:{pattern_id}")
    return {
        "root_modules": list(ANCESTRY_ROOT_MODULES),
        "visited_module_count": len(visited),
        "visited_modules_sha256": _value_sha256(sorted(visited)),
        "import_edges_sha256": _value_sha256(edge_rows),
        "forbidden_module_prefixes": list(FORBIDDEN_MODULE_PREFIXES),
        "forbidden_import_paths": forbidden,
        "forbidden_literal_pattern_ids": list(FORBIDDEN_LITERAL_PATTERNS),
        "forbidden_literal_hits": literal_hits,
        "passed": not forbidden and not literal_hits,
        "scope": (
            "static Python import closure of the declared native and conditional "
            "producer roots; dynamic imports and semantic constant encoding are not "
            "proved absent by this bounded check"
        ),
    }


def _integer_adjacency_probe(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild exact recurrent traces from the pinned carrier only."""

    if manifest.get("schema") != "oph.echosahedral_selector_manifest.v1":
        raise ValueError("carrier schema mismatch")
    carrier = manifest.get("carrier")
    if not isinstance(carrier, Mapping):
        raise ValueError("carrier block missing")
    ports = carrier.get("ports")
    edges = carrier.get("edges")
    if not isinstance(ports, list) or len(ports) != 12 or len(set(ports)) != 12:
        raise ValueError("carrier must expose twelve distinct ports")
    if not isinstance(edges, list) or len(edges) != 30:
        raise ValueError("carrier must expose thirty edges")
    index = {port: position for position, port in enumerate(ports)}
    adjacency = [[0 for _ in ports] for _ in ports]
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("edge row malformed")
        left, right = index[edge[0]], index[edge[1]]
        if left == right or adjacency[left][right]:
            raise ValueError("edge row is a loop or duplicate")
        adjacency[left][right] = adjacency[right][left] = 1
    if {sum(row) for row in adjacency} != {5}:
        raise ValueError("carrier is not five-regular")

    def multiply(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]):
        return [
            [
                sum(left[i][k] * right[k][j] for k in range(12))
                for j in range(12)
            ]
            for i in range(12)
        ]

    identity = [[int(i == j) for j in range(12)] for i in range(12)]
    powers = [identity, adjacency]
    powers.append(multiply(powers[-1], adjacency))
    powers.append(multiply(powers[-1], adjacency))

    # Distances are independently read from adjacency rather than from a
    # declared antipode table.
    distance_rows: list[list[int]] = []
    farthest: list[int] = []
    for source in range(12):
        distances = [-1] * 12
        distances[source] = 0
        frontier = [source]
        while frontier:
            nxt = []
            for left in frontier:
                for right, adjacent in enumerate(adjacency[left]):
                    if adjacent and distances[right] < 0:
                        distances[right] = distances[left] + 1
                        nxt.append(right)
            frontier = nxt
        candidates = [i for i, value in enumerate(distances) if value == max(distances)]
        if max(distances) != 3 or len(candidates) != 1:
            raise ValueError("carrier lacks one unique distance-three response per port")
        distance_rows.append(distances)
        farthest.append(candidates[0])

    trace_payload = {
        "port_order": ports,
        "adjacency_powers_k0_through_k3": powers,
        "distance_rows": distance_rows,
        "farthest_port_map": farthest,
    }
    return trace_payload


def _charged_response_probe(
    artifact: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the CR-0 row to the attained charged-response producer artifact."""

    if artifact.get("schema") != "oph.charged_response_semantic_artifact.v3":
        raise ValueError("charged-response artifact schema mismatch")
    payload = dict(artifact)
    artifact_sha256 = payload.pop("artifact_sha256", None)
    if artifact_sha256 != _value_sha256(payload):
        raise ValueError("charged-response artifact payload digest mismatch")
    carrier_binding = artifact.get("carrier_binding")
    source_response = artifact.get("source_response")
    provenance = artifact.get("provenance")
    derived = artifact.get("derived")
    if not all(
        isinstance(value, Mapping)
        for value in (carrier_binding, source_response, provenance, derived)
    ):
        raise ValueError("charged-response artifact is missing a typed block")
    protocol = source_response.get("impulse_readback_protocol")
    runtime = provenance.get("runtime_binding")
    current_lift = derived.get("current_lift_status")
    if not all(isinstance(value, Mapping) for value in (protocol, runtime, current_lift)):
        raise ValueError("charged-response artifact is missing response evidence")
    recurrence_trace = _integer_adjacency_probe(manifest)
    expected_filter = ["1", "-1/2", "-2/5", "1/10"]
    if not (
        carrier_binding.get("carrier_manifest_sha256")
        == _value_sha256(manifest).removeprefix("sha256:")
        and
        protocol.get("homogeneous_filter_coefficients") == expected_filter
        and protocol.get("unique_solution_rank") == 4
        and protocol.get("unique_farthest_port_per_source") is True
        and protocol.get("nearer_shells_cancelled") is True
        and protocol.get("target_labels_used") is False
        and protocol.get("downstream_labels_used") is False
        and source_response.get("operator") == "negative_graph_antipode_involution"
        and source_response.get("source")
        == "target_blind_maximal_distance_impulse_readback"
        and source_response.get("impulse_readback_response_executed") is True
        and source_response.get("physical_perturb_readback_source_bound") is True
        and runtime.get("charged_response_operator_receipt") is True
        and runtime.get("impulse_readback_producer_receipt") is True
        and runtime.get("reversible_response_source")
        == "finite_unitary_carrier_channel"
        and current_lift.get("source_selected") is False
    ):
        raise ValueError("charged-response artifact has not attained its finite response gates")
    port_order = carrier_binding.get("port_order")
    antipode = source_response.get("antipode_port_map")
    if not (
        isinstance(port_order, list)
        and len(port_order) == 12
        and len(set(port_order)) == 12
        and isinstance(antipode, list)
        and sorted(antipode) == list(range(12))
    ):
        raise ValueError("charged-response port basis or antipode map is invalid")
    return {
        "semantic_id": "twelve_port_source_bound_charged_response.v1",
        "source_artifact_schema": artifact["schema"],
        "source_artifact_sha256": artifact_sha256,
        "carrier_manifest_canonical_sha256": (
            "sha256:" + str(carrier_binding.get("carrier_manifest_sha256"))
        ),
        "port_order": port_order,
        "response_operator": source_response["operator"],
        "response_source": source_response["source"],
        "antipode_port_map": antipode,
        "homogeneous_filter_coefficients": expected_filter,
        "recurrence_trace_payload": recurrence_trace,
        "recurrence_trace_payload_sha256": _value_sha256(recurrence_trace),
        "runtime_response_source": runtime["reversible_response_source"],
        "finite_simulator_response_identified": True,
        "current_lift_source_selected": False,
        "physical_A_T_identification": False,
    }


def _capability_rows() -> list[dict[str, Any]]:
    repair = _load_json("bounded_repair_receipt")
    projection = _load_json("record_counting_projection")
    source_manifest = _load_json("finite_source_manifest")
    source_replay = _load_json("finite_consensus_replay_report")
    sector = _load_json("defect_sector_receipt")
    stage1 = _load_json("stage1_receipt")
    stage3 = _load_json("stage3_receipt")

    repair_checks = bool(
        repair.get("FINITE_DIRECTED_SEAM_TORSOR_RECEIPT") is True
        and repair.get("PHYSICAL_REPAIR_LAW_RECEIPT") is False
        and repair.get("GLOBAL_A1_A3_POLICY_UNIQUENESS_RECEIPT") is False
        and (repair.get("distinguished_total_twelve_sector") or {}).get(
            "source_packet_imported_or_hash_verified_here"
        )
        is True
        and ((repair.get("distinguished_total_twelve_sector") or {}).get("source_projection") or {}).get(
            "verified"
        )
        is True
        and projection.get("schema") == "oph.record_counting_source_projection.v1"
        and projection.get("source_issue") == 628
        and projection.get("bounded_exit") == "exact_named_realization"
    )
    sector_checks = bool(
        sector.get("schema") == "oph.local-domain-defect-sector-spectra.v1"
        and sector.get("DEFECT_SECTOR_SPECTRA_RECEIPT") is True
        and (sector.get("sector_family") or {}).get("group")
        == "Z6 character orbit of the declared reversing convention"
        and (sector.get("spectral_interface_identity") or {}).get(
            "rer_exact_flux_12_42_vertex_identity_bridge"
        )
        is False
        and sector.get("physical_promotion_allowed") is False
    )
    stage1_checks = bool(
        stage1.get("STAGE1_EVENT_COMPLEX_RECEIPT") is True
        and (stage1.get("atlas") or {}).get("covered_event_count") == 2304
        and stage1.get("physical_promotion_allowed") is False
    )
    stage3_scalar_partial = bool(
        stage3.get("STAGE3_TYPED_DOMAIN_RECEIPT") is True
        and (((stage3.get("section_typing") or {}).get("species") or {}).get("scalar") or {}).get(
            "fiber_dimension"
        )
        == 2
        and (stage3.get("kinetic_kernel_certificate") or {}).get(
            "twisted_kernel_dimension"
        )
        == 0
    )
    source_quotient_boundary = bool(
        source_manifest.get("schema") == "finite_consensus_replay_source_v1"
        and isinstance(source_manifest.get("source_quotient_hash"), str)
        and source_manifest.get("source_quotient_hash", "").startswith("sha256:")
        and source_replay.get("source_quotient_hash")
        == source_manifest.get("source_quotient_hash")
        and source_replay.get("FINITE_CONSENSUS_THEOREM_RECEIPT") is False
        and source_replay.get("receipt") is False
        and (source_replay.get("exact_endpoint_branch_check") or {}).get(
            "structurally_confluent"
        )
        is False
        and (source_replay.get("exact_endpoint_branch_check") or {}).get(
            "unique_terminal_quotient_hash_count"
        )
        == 2
    )

    rows = [
        {
            "capability_id": "physical_quotient",
            "label": "Physical quotient",
            "classification": MISSING,
            "required_evidence": [
                "normal-form or quotient certificate",
                "stable semantic identifier",
                "physical quotient binding",
            ],
            "verified_evidence": [
                "a generic finite normal-form checker exists",
                "one finite source state carries a gauge-quotient hash",
                "the pinned source replay fails its consensus theorem receipt and has two exact endpoint quotient classes",
            ],
            "missing_evidence": [
                "an attained applied normal-form or quotient certificate",
                "a stable semantic identifier for the required physical quotient",
                "external enumeration completeness",
                "identity with the paper's physical quotient",
            ],
            "source_pin_ids": [
                "normal_form_checker",
                "finite_source_manifest",
                "finite_consensus_replay_report",
            ],
            "producer_module_roots": [],
            "machine_check_ids": [
                "normal_form_checker_symbols",
                "source_quotient_hash",
                "source_replay_exact_nonconfluence",
            ],
            "machine_checks_passed": source_quotient_boundary,
            "theory_binding_status": "NO_ATTAINED_PHYSICAL_QUOTIENT_CERTIFICATE",
            "downstream_effect": "blocks every lane that requires the physical quotient",
            "adapter_promotion_allowed": False,
        },
        {
            "capability_id": "primitive_repair_law",
            "label": "Primitive repair law",
            "classification": AVAILABLE_SIMULATOR_NATIVE,
            "required_evidence": [
                "code and configuration root",
                "transition replay artifact",
            ],
            "verified_evidence": [
                "canonical internal-seam repair producer is pinned",
                "source projection and settled trajectory rows are pinned",
            ],
            "missing_evidence": [
                "A1-A3 uniqueness",
                "physical repair-law identification",
            ],
            "source_pin_ids": [
                "canonical_repair_producer",
                "bounded_repair_receipt",
                "record_counting_projection",
            ],
            "producer_module_roots": ["oph_fpe.dynamics.canonical_seam_repair"],
            "machine_check_ids": ["bounded_repair_scope", "repair_projection_binding"],
            "machine_checks_passed": repair_checks,
            "theory_binding_status": "FINITE_INTERNAL_REPAIR_NATIVE__PHYSICAL_LAW_OPEN",
            "downstream_effect": (
                "supports finite internal-repair audits; cannot select physical collar dynamics"
            ),
            "adapter_promotion_allowed": False,
        },
        {
            "capability_id": "boundary_sector",
            "label": "Boundary sector",
            "classification": AVAILABLE_CONDITIONAL,
            "required_evidence": [
                "exact sector representation",
                "quotient-visible readout",
            ],
            "verified_evidence": [
                "finite local-domain Z6 character orbit is exact and source-pinned",
            ],
            "missing_evidence": [
                "identity with the separate twelve-port global flux sector",
                "physical sector selection",
            ],
            "source_pin_ids": ["defect_sector_producer", "defect_sector_receipt"],
            "producer_module_roots": ["oph_fpe.local_domain.defect_sector_spectra"],
            "machine_check_ids": ["local_z6_sector_receipt", "global_sector_bridge_false"],
            "machine_checks_passed": sector_checks,
            "theory_binding_status": "FINITE_Z6_SECTOR_NATIVE__GLOBAL_IDENTITY_OPEN",
            "downstream_effect": "local sector diagnostics only until the identity bridge exists",
            "adapter_promotion_allowed": False,
        },
        {
            "capability_id": "scalar_register",
            "label": "Scalar register",
            "classification": MISSING,
            "required_evidence": [
                "source-defined scalar register semantic ID",
                "projector",
                "spectral isolation witness",
            ],
            "verified_evidence": [
                "a two-dimensional typed scalar section exists on the local domain",
            ],
            "missing_evidence": [
                "selected scalar register ID",
                "scalar projector",
                "scalar spectral-isolation interval",
            ],
            "source_pin_ids": ["stage3_receipt"],
            "producer_module_roots": [],
            "machine_check_ids": ["typed_scalar_section_only"],
            "machine_checks_passed": stage3_scalar_partial,
            "theory_binding_status": "NO_NATIVE_COMMON_RESERVE_SCALAR_REGISTER",
            "downstream_effect": "blocks CR-3 and the native cocycle lane",
            "adapter_promotion_allowed": False,
        },
        {
            "capability_id": "protected_z6_reserve",
            "label": "Protected Z6 reserve",
            "classification": MISSING,
            "required_evidence": [
                "primitive reserve state field or exact derivation from center-sector data",
                "separate count, presence, absence, and first-hit semantics",
            ],
            "verified_evidence": [
                "an exact finite Z6 sector family exists",
                "a protected total-twelve load diagnostic exists in another domain",
            ],
            "missing_evidence": [
                "primitive protected-reserve field",
                "reserve opportunity records",
                "count/presence/absence/first-hit separation",
            ],
            "source_pin_ids": ["defect_sector_receipt", "bounded_repair_receipt"],
            "producer_module_roots": [],
            "machine_check_ids": ["z6_is_sector_not_reserve", "total_twelve_is_not_z6_reserve"],
            "machine_checks_passed": sector_checks and repair_checks,
            "theory_binding_status": "NO_NATIVE_PROTECTED_RESERVE_OBJECT",
            "downstream_effect": "blocks CR-2 and every reserve-consuming reduction",
            "adapter_promotion_allowed": False,
        },
        {
            "capability_id": "scalar_reserve_coregistration",
            "label": "Scalar/reserve co-registration",
            "classification": MISSING,
            "required_evidence": [
                "same-source scalar/reserve derivation rule",
                "event-level co-registration witness",
            ],
            "verified_evidence": [],
            "missing_evidence": [
                "native scalar register",
                "native protected reserve",
                "co-registration derivation and witness",
            ],
            "source_pin_ids": ["defect_sector_receipt", "stage3_receipt"],
            "producer_module_roots": [],
            "machine_check_ids": ["parents_do_not_supply_coregistration"],
            "machine_checks_passed": sector_checks and stage3_scalar_partial,
            "theory_binding_status": "NO_NATIVE_COREGISTRATION_RULE",
            "downstream_effect": "blocks every cross-lane common-parameter claim",
            "adapter_promotion_allowed": False,
        },
        {
            "capability_id": "source_ensemble_action",
            "label": "Source ensemble/action",
            "classification": AVAILABLE_CONTROL_ONLY,
            "required_evidence": [
                "exact measure or action",
                "sampling law",
                "OPH-native source selection",
            ],
            "verified_evidence": [
                "the conventional E1 free-scalar control has an explicit action and sampling law",
            ],
            "missing_evidence": ["OPH-native ensemble selector"],
            "source_pin_ids": ["reference_ensemble_producer"],
            "producer_module_roots": [],
            "machine_check_ids": ["reference_ensemble_explicitly_e1_control"],
            "machine_checks_passed": True,
            "theory_binding_status": "CONVENTIONAL_E1_CONTROL__SOURCE_SELECTION_OPEN",
            "downstream_effect": "may be run as a separately reported control; blocks native screen statistics",
            "adapter_promotion_allowed": False,
        },
        {
            "capability_id": "refinement_tower_physical_scale_ratios",
            "label": "Refinement tower with physical scale ratios",
            "classification": MISSING,
            "required_evidence": [
                "fine/coarse maps",
                "common lineage",
                "commuting-square evidence",
                "physical scale ratios",
            ],
            "verified_evidence": [
                "exact twelve-port persistence maps exist on a geodesic tower",
                "finite local-domain observations exist at two carrier counts",
            ],
            "missing_evidence": [
                "physical scale-ratio binding",
                "one common refinement lineage joining the common-reserve objects",
                "quotient-state and observable-algebra refinement maps",
                "one commuting bi-refinement square for the selected scalar covariance",
            ],
            "source_pin_ids": ["charged_response_producer", "defect_sector_receipt"],
            "producer_module_roots": [],
            "machine_check_ids": ["charged_response_refinement_symbols", "finite_two_scale_sector_rows"],
            "machine_checks_passed": sector_checks,
            "theory_binding_status": "NO_COMMON_PHYSICAL_BIREFINEMENT_TOWER",
            "downstream_effect": "keeps issue 657 at SOURCE_PRODUCER_MISSING",
            "adapter_promotion_allowed": False,
        },
        {
            "capability_id": "support_geometry",
            "label": "Support geometry",
            "classification": MISSING,
            "required_evidence": [
                "support map and cellulation",
                "coordinates or charts",
                "mass matrix M_r",
                "J_X inputs for the declared screen field",
            ],
            "verified_evidence": [
                "one finite local-domain atlas is source-pinned",
                "unit counting measure is typed on that domain",
            ],
            "missing_evidence": [
                "common-reserve support map",
                "screen mass matrix",
                "source-native J_X input",
            ],
            "source_pin_ids": ["stage1_receipt", "stage1_arrays", "stage3_receipt"],
            "producer_module_roots": [],
            "machine_check_ids": ["local_atlas_partial", "counting_measure_partial"],
            "machine_checks_passed": stage1_checks and stage3_scalar_partial,
            "theory_binding_status": "LOCAL_ATLAS_EXISTS__SCREEN_GEOMETRY_OBJECT_MISSING",
            "downstream_effect": "blocks CR-5 geometric screen production",
            "adapter_promotion_allowed": False,
        },
        {
            "capability_id": "full_half_collars",
            "label": "Full/half collars",
            "classification": MISSING,
            "required_evidence": [
                "source-facing full collar map",
                "left and right half-collar maps",
                "orientation reversal",
            ],
            "verified_evidence": [
                "a fail-closed finite collar factorization checker exists",
            ],
            "missing_evidence": [
                "source-facing full map",
                "source-facing left map",
                "source-facing right map",
                "orientation-reversal witness joining the maps",
            ],
            "source_pin_ids": ["collar_clause_checker", "stage1_receipt"],
            "producer_module_roots": [],
            "machine_check_ids": ["collar_checker_is_interface_not_source_producer"],
            "machine_checks_passed": stage1_checks,
            "theory_binding_status": "COLLAR_INTERFACE_EXISTS__NATIVE_MAPS_MISSING",
            "downstream_effect": "blocks the full/half operator and orientation-factor tests",
            "adapter_promotion_allowed": False,
        },
        {
            "capability_id": "raw_twelve_port_response",
            "label": "Raw twelve-port response",
            "classification": AVAILABLE_SIMULATOR_NATIVE,
            "required_evidence": [
                "immutable impulse or recurrent-response traces",
                "port basis",
            ],
            "verified_evidence": [
                "CR-0 embeds exact integer adjacency-power traces reconstructed from the pinned carrier",
                "the twelve-port order is carried in the trace payload",
            ],
            "missing_evidence": ["bridge from this response to the paper's A_T(p) observable"],
            "source_pin_ids": [
                "carrier_manifest",
                "charged_response_producer",
                "charged_response_artifact",
            ],
            "producer_module_roots": ["oph_fpe.core.charged_response"],
            "machine_check_ids": [
                "attained_charged_response_artifact",
                "independent_exact_recurrence_and_antipode_probe",
            ],
            "machine_checks_passed": True,
            "theory_binding_status": "NATIVE_RESPONSE_TRACE__A_T_BRIDGE_OPEN",
            "downstream_effect": "archive for CR-7; common-parameter use remains blocked",
            "adapter_promotion_allowed": False,
        },
    ]
    if tuple(row["capability_id"] for row in rows) != CAPABILITY_IDS:
        raise AssertionError("capability row order drifted")
    if any(not row["machine_checks_passed"] for row in rows):
        failed = [row["capability_id"] for row in rows if not row["machine_checks_passed"]]
        raise ValueError(f"capability evidence checks failed: {failed}")
    return rows


def build_capability_matrix() -> dict[str, Any]:
    sources = {source_id: _raw_pin(path) for source_id, path in SOURCE_PATHS.items()}
    rows = _capability_rows()
    response_probe = _charged_response_probe(
        _load_json("charged_response_artifact"), _load_json("carrier_manifest")
    )
    firewall = _ancestry_firewall()
    if not firewall["passed"]:
        raise ValueError("a native/conditional producer has forbidden target ancestry")
    counts = {classification: 0 for classification in CLASSIFICATIONS}
    for row in rows:
        counts[row["classification"]] += 1
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "stage": STAGE,
        "status": "CAPABILITY_PROBE_COMPLETE__SCIENTIFIC_PROMOTION_DISABLED",
        "scientific_promotion_allowed": False,
        "allowed_classifications": list(CLASSIFICATIONS),
        "capability_order": list(CAPABILITY_IDS),
        "audit_scope": {
            "repository": "muellerberndt/oph-physics-sim",
            "python_tree": _python_tree_snapshot(),
            "scope_boundary": (
                "bounded audit of the committed producer catalog and named artifacts; "
                "MISSING means no accepted producer in this snapshot, not a theorem that "
                "no future implementation can exist"
            ),
        },
        "sources": sources,
        "target_ancestry": firewall,
        "raw_twelve_port_response_probe": response_probe,
        "capabilities": rows,
        "classification_counts": counts,
        "lane_stop_rules": {
            "reserve_lane_blocked": True,
            "cocycle_lane_blocked": True,
            "screen_lane_blocked": True,
            "raw_response_archived_for_later_bridge": True,
            "independent_internal_repair_audits_may_continue": True,
            "large_simulation_authorized": False,
            "cr1_or_later_implemented_here": False,
        },
        "claim_boundary": (
            "CR-0 capability inventory only. It identifies two simulator-native finite "
            "objects, one conditional object, one control-only ensemble, and seven "
            "missing common-reserve prerequisites. It does not create a reserve, scalar "
            "cocycle, screen field, common parameter, cosmological prediction, or physical "
            "promotion receipt."
        ),
    }
    result["payload_sha256"] = _value_sha256(result)
    schema = json.loads(SCHEMA_PATH.read_text("utf-8"))
    Draft202012Validator(schema).validate(result)
    return result


def render_report(matrix: Mapping[str, Any]) -> str:
    lines = [
        "# Common-reserve CR-0 capability inventory",
        "",
        "Scientific promotion is disabled. The classifications describe the current simulator snapshot.",
        "",
        "| Capability | Classification | Blocking boundary |",
        "|---|---|---|",
    ]
    for row in matrix["capabilities"]:
        missing = "; ".join(row["missing_evidence"]) or "None in CR-0"
        lines.append(f"| {row['label']} | `{row['classification']}` | {missing} |")
    lines.extend(
        [
            "",
            "## Stop rules",
            "",
            "The reserve, cocycle, and screen lanes are blocked by their missing native objects. The exact recurrent twelve-port response is archived for a later source-to-observable bridge. Finite internal-repair audits may proceed independently.",
            "",
            "This report is a rendering of `producer_capability_matrix.json`; the JSON and its independent verifier are authoritative for CR-0.",
            "",
        ]
    )
    return "\n".join(lines)


def write_capability_matrix(
    output: Path = DEFAULT_OUTPUT,
    report: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    matrix = build_capability_matrix()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", "utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(matrix), "utf-8")
    return matrix


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    matrix = write_capability_matrix(args.out, args.report)
    print(json.dumps({"status": matrix["status"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
