"""Independent fail-closed verifier for the issue-660 CR-0 matrix.

This module does not import the capability producer.  It validates the strict
schema, raw source pins, payload digest, Python-tree commitment, import
ancestry, exact twelve-port trace, and every status assignment from the
underlying artifacts.  Its whitelist prevents an added adapter or placeholder
from promoting a missing capability.
"""

from __future__ import annotations

import argparse
import ast
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = REPOSITORY_ROOT / "data/common_reserve/producer_capability_matrix.json"
SCHEMA_PATH = REPOSITORY_ROOT / "oph_fpe/common_reserve/schemas/capability.schema.json"

EXPECTED_SOURCES = {
    "carrier_manifest": "tests/fixtures/echosahedral_federation_reference.json",
    "charged_response_producer": "oph_fpe/core/charged_response.py",
    "charged_response_artifact": "data/common_reserve/charged_response_artifact.json",
    "canonical_repair_producer": "oph_fpe/dynamics/canonical_seam_repair.py",
    "bounded_repair_receipt": "data/repair_closure/bounded_atomic_self_readback_closure_receipt.json",
    "record_counting_projection": "data/repair_closure/record_counting_source_projection.json",
    "normal_form_checker": "oph_fpe/quotient/observable_normal_form.py",
    "finite_source_manifest": "data/earned_runs/oph_universe_64k_3p1d_reearned/finite_consensus_source_manifest.json",
    "finite_consensus_replay_report": "data/earned_runs/oph_universe_64k_3p1d_reearned/finite_consensus_replay_report.json",
    "defect_sector_producer": "oph_fpe/local_domain/defect_sector_spectra.py",
    "defect_sector_receipt": "data/local_domain/defect_sector_receipt.json",
    "stage1_receipt": "data/local_domain/stage1_receipt.json",
    "stage1_arrays": "data/local_domain/stage1_arrays.npz.gz",
    "stage3_receipt": "data/local_domain/stage3_receipt.json",
    "reference_ensemble_producer": "oph_fpe/ensembles/reference_vacuum.py",
    "collar_clause_checker": "oph_fpe/cosmology/collar_clause.py",
}
EXPECTED_ROWS = {
    "physical_quotient": (
        "MISSING",
        ("normal_form_checker", "finite_source_manifest", "finite_consensus_replay_report"),
        ("normal_form_checker_symbols", "source_quotient_hash", "source_replay_exact_nonconfluence"),
    ),
    "primitive_repair_law": (
        "AVAILABLE_SIMULATOR_NATIVE",
        ("canonical_repair_producer", "bounded_repair_receipt", "record_counting_projection"),
        ("bounded_repair_scope", "repair_projection_binding"),
    ),
    "boundary_sector": (
        "AVAILABLE_CONDITIONAL",
        ("defect_sector_producer", "defect_sector_receipt"),
        ("local_z6_sector_receipt", "global_sector_bridge_false"),
    ),
    "scalar_register": (
        "MISSING",
        ("stage3_receipt",),
        ("typed_scalar_section_only",),
    ),
    "protected_z6_reserve": (
        "MISSING",
        ("defect_sector_receipt", "bounded_repair_receipt"),
        ("z6_is_sector_not_reserve", "total_twelve_is_not_z6_reserve"),
    ),
    "scalar_reserve_coregistration": (
        "MISSING",
        ("defect_sector_receipt", "stage3_receipt"),
        ("parents_do_not_supply_coregistration",),
    ),
    "source_ensemble_action": (
        "AVAILABLE_CONTROL_ONLY",
        ("reference_ensemble_producer",),
        ("reference_ensemble_explicitly_e1_control",),
    ),
    "refinement_tower_physical_scale_ratios": (
        "MISSING",
        ("charged_response_producer", "defect_sector_receipt"),
        ("charged_response_refinement_symbols", "finite_two_scale_sector_rows"),
    ),
    "support_geometry": (
        "MISSING",
        ("stage1_receipt", "stage1_arrays", "stage3_receipt"),
        ("local_atlas_partial", "counting_measure_partial"),
    ),
    "full_half_collars": (
        "MISSING",
        ("collar_clause_checker", "stage1_receipt"),
        ("collar_checker_is_interface_not_source_producer",),
    ),
    "raw_twelve_port_response": (
        "AVAILABLE_SIMULATOR_NATIVE",
        ("carrier_manifest", "charged_response_producer", "charged_response_artifact"),
        ("attained_charged_response_artifact", "independent_exact_recurrence_and_antipode_probe"),
    ),
}
EXPECTED_ROOTS = (
    "oph_fpe.core.charged_response",
    "oph_fpe.dynamics.canonical_seam_repair",
    "oph_fpe.local_domain.defect_sector_spectra",
)
FORBIDDEN_PREFIXES = (
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


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _safe_path(relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("path is not a nonempty string")
    candidate = (REPOSITORY_ROOT / relative).resolve(strict=True)
    candidate.relative_to(REPOSITORY_ROOT.resolve())
    if not candidate.is_file():
        raise ValueError("path is not a regular file")
    return candidate


def _load_json(relative: str) -> dict[str, Any]:
    value = json.loads(_safe_path(relative).read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{relative} is not a JSON object")
    return value


def _check_pins(matrix: Mapping[str, Any], reasons: list[str]) -> None:
    sources = matrix.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(EXPECTED_SOURCES):
        reasons.append("source_catalog_mismatch")
        return
    for source_id, expected_path in EXPECTED_SOURCES.items():
        pin = sources.get(source_id)
        if not isinstance(pin, Mapping) or set(pin) != {"path", "bytes", "sha256"}:
            reasons.append(f"source_pin_shape:{source_id}")
            continue
        if pin.get("path") != expected_path:
            reasons.append(f"source_path_mismatch:{source_id}")
            continue
        try:
            raw = _safe_path(expected_path).read_bytes()
        except (OSError, ValueError):
            reasons.append(f"source_unreadable:{source_id}")
            continue
        if pin.get("bytes") != len(raw):
            reasons.append(f"source_bytes_mismatch:{source_id}")
        actual = "sha256:" + hashlib.sha256(raw).hexdigest()
        if pin.get("sha256") != actual:
            reasons.append(f"source_sha256_mismatch:{source_id}")


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPOSITORY_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _graph_and_snapshot() -> tuple[dict[str, set[str]], dict[str, Any]]:
    module_paths = {
        _module_name(path): path
        for path in (REPOSITORY_ROOT / "oph_fpe").rglob("*.py")
    }
    rows = []
    graph: dict[str, set[str]] = {module: set() for module in module_paths}
    for module, path in sorted(module_paths.items()):
        raw = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "bytes": len(raw),
                "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            }
        )
        tree = ast.parse(raw.decode("utf-8"), filename=str(path))
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
                    graph[module].update(
                        candidate
                        for candidate in module_paths
                        if candidate.startswith(name + ".")
                    )
    snapshot = {"root": "oph_fpe", "file_count": len(rows), "tree_sha256": _sha(rows)}
    return graph, snapshot


def _check_tree_and_ancestry(matrix: Mapping[str, Any], reasons: list[str]) -> None:
    graph, snapshot = _graph_and_snapshot()
    audit_scope = matrix.get("audit_scope")
    if not isinstance(audit_scope, Mapping) or audit_scope.get("python_tree") != snapshot:
        reasons.append("python_tree_snapshot_mismatch")
    visited: set[str] = set()
    queue = deque(EXPECTED_ROOTS)
    while queue:
        module = queue.popleft()
        if module in visited:
            continue
        visited.add(module)
        queue.extend(sorted(graph.get(module, ())))
    edges = [
        [source, target]
        for source in sorted(visited)
        for target in sorted(graph.get(source, ()))
        if target in visited
    ]
    forbidden = sorted(
        module
        for module in visited
        if any(module == prefix or module.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES)
    )
    literal_hits = []
    for module in sorted(visited):
        path = REPOSITORY_ROOT / (module.replace(".", "/") + ".py")
        if not path.is_file():
            path = REPOSITORY_ROOT / module.replace(".", "/") / "__init__.py"
        source = path.read_text("utf-8")
        for pattern_id, pattern in FORBIDDEN_LITERAL_PATTERNS.items():
            if re.search(pattern, source):
                literal_hits.append(f"{module}:{pattern_id}")
    ancestry = matrix.get("target_ancestry")
    if not isinstance(ancestry, Mapping):
        reasons.append("target_ancestry_missing")
        return
    expected_fields = {
        "root_modules": list(EXPECTED_ROOTS),
        "visited_module_count": len(visited),
        "visited_modules_sha256": _sha(sorted(visited)),
        "import_edges_sha256": _sha(edges),
        "forbidden_module_prefixes": list(FORBIDDEN_PREFIXES),
        "forbidden_import_paths": forbidden,
        "forbidden_literal_pattern_ids": list(FORBIDDEN_LITERAL_PATTERNS),
        "forbidden_literal_hits": literal_hits,
        "passed": not forbidden and not literal_hits,
    }
    for field, expected in expected_fields.items():
        if ancestry.get(field) != expected:
            reasons.append(f"target_ancestry_mismatch:{field}")


def _matrix_multiply(left, right):
    return [
        [sum(left[i][k] * right[k][j] for k in range(12)) for j in range(12)]
        for i in range(12)
    ]


def _response_probe() -> dict[str, Any]:
    manifest = _load_json(EXPECTED_SOURCES["carrier_manifest"])
    artifact = _load_json(EXPECTED_SOURCES["charged_response_artifact"])
    if manifest.get("schema") != "oph.echosahedral_selector_manifest.v1":
        raise ValueError("carrier schema mismatch")
    carrier = manifest.get("carrier")
    if not isinstance(carrier, Mapping):
        raise ValueError("carrier missing")
    ports = carrier.get("ports")
    edges = carrier.get("edges")
    if not isinstance(ports, list) or len(ports) != 12 or len(set(ports)) != 12:
        raise ValueError("port basis invalid")
    if not isinstance(edges, list) or len(edges) != 30:
        raise ValueError("edge set invalid")
    index = {port: position for position, port in enumerate(ports)}
    adjacency = [[0] * 12 for _ in range(12)]
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("edge invalid")
        left, right = index[edge[0]], index[edge[1]]
        if left == right or adjacency[left][right]:
            raise ValueError("edge loop or duplicate")
        adjacency[left][right] = adjacency[right][left] = 1
    if {sum(row) for row in adjacency} != {5}:
        raise ValueError("degree mismatch")
    identity = [[int(i == j) for j in range(12)] for i in range(12)]
    powers = [identity, adjacency]
    powers.append(_matrix_multiply(powers[-1], adjacency))
    powers.append(_matrix_multiply(powers[-1], adjacency))
    distances_all = []
    farthest = []
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
        candidates = [i for i, distance in enumerate(distances) if distance == max(distances)]
        if max(distances) != 3 or len(candidates) != 1:
            raise ValueError("farthest response is not unique")
        distances_all.append(distances)
        farthest.append(candidates[0])
    recurrence_payload = {
        "port_order": ports,
        "adjacency_powers_k0_through_k3": powers,
        "distance_rows": distances_all,
        "farthest_port_map": farthest,
    }
    if artifact.get("schema") != "oph.charged_response_semantic_artifact.v3":
        raise ValueError("charged-response artifact schema mismatch")
    artifact_payload = dict(artifact)
    artifact_sha256 = artifact_payload.pop("artifact_sha256", None)
    if artifact_sha256 != _sha(artifact_payload):
        raise ValueError("charged-response artifact payload digest mismatch")
    binding = artifact.get("carrier_binding")
    response = artifact.get("source_response")
    provenance = artifact.get("provenance")
    derived = artifact.get("derived")
    if not all(
        isinstance(value, Mapping)
        for value in (binding, response, provenance, derived)
    ):
        raise ValueError("charged-response artifact block missing")
    protocol = response.get("impulse_readback_protocol")
    runtime = provenance.get("runtime_binding")
    current_lift = derived.get("current_lift_status")
    if not all(isinstance(value, Mapping) for value in (protocol, runtime, current_lift)):
        raise ValueError("charged-response evidence block missing")
    expected_filter = ["1", "-1/2", "-2/5", "1/10"]
    if not (
        binding.get("carrier_manifest_sha256") == _sha(manifest).removeprefix("sha256:")
        and binding.get("port_order") == ports
        and response.get("antipode_port_map") == farthest
        and response.get("operator") == "negative_graph_antipode_involution"
        and response.get("source") == "target_blind_maximal_distance_impulse_readback"
        and response.get("unique_nonidentity_central_involution") is True
        and response.get("commutes_with_propagation_generator") is True
        and response.get("self_adjoint_unitary_involution") is True
        and response.get("impulse_readback_response_executed") is True
        and response.get("physical_perturb_readback_source_bound") is True
        and protocol.get("homogeneous_filter_coefficients") == expected_filter
        and protocol.get("unique_solution_rank") == 4
        and protocol.get("unique_farthest_port_per_source") is True
        and protocol.get("nearer_shells_cancelled") is True
        and protocol.get("target_labels_used") is False
        and protocol.get("downstream_labels_used") is False
        and runtime.get("charged_response_operator_receipt") is True
        and runtime.get("impulse_readback_producer_receipt") is True
        and runtime.get("reversible_response_source") == "finite_unitary_carrier_channel"
        and current_lift.get("source_selected") is False
    ):
        raise ValueError("charged-response artifact finite gates failed")
    return {
        "semantic_id": "twelve_port_source_bound_charged_response.v1",
        "source_artifact_schema": artifact["schema"],
        "source_artifact_sha256": artifact_sha256,
        "carrier_manifest_canonical_sha256": "sha256:" + binding["carrier_manifest_sha256"],
        "port_order": ports,
        "response_operator": response["operator"],
        "response_source": response["source"],
        "antipode_port_map": farthest,
        "homogeneous_filter_coefficients": expected_filter,
        "recurrence_trace_payload": recurrence_payload,
        "recurrence_trace_payload_sha256": _sha(recurrence_payload),
        "runtime_response_source": runtime["reversible_response_source"],
        "finite_simulator_response_identified": True,
        "current_lift_source_selected": False,
        "physical_A_T_identification": False,
    }


def _defined_functions(relative: str) -> set[str]:
    tree = ast.parse(_safe_path(relative).read_text("utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _semantic_checks(reasons: list[str]) -> None:
    repair = _load_json(EXPECTED_SOURCES["bounded_repair_receipt"])
    projection = _load_json(EXPECTED_SOURCES["record_counting_projection"])
    sector = _load_json(EXPECTED_SOURCES["defect_sector_receipt"])
    stage1 = _load_json(EXPECTED_SOURCES["stage1_receipt"])
    stage3 = _load_json(EXPECTED_SOURCES["stage3_receipt"])
    source_manifest = _load_json(EXPECTED_SOURCES["finite_source_manifest"])
    source_replay = _load_json(EXPECTED_SOURCES["finite_consensus_replay_report"])

    if not (
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
    ):
        reasons.append("repair_evidence_failed")
    trajectories = (repair.get("distinguished_total_twelve_sector") or {}).get("trajectory_rows")
    if not isinstance(trajectories, list) or not trajectories or not all(
        isinstance(row, Mapping)
        and row.get("settled_to_all_ones") is True
        and row.get("final_energy") == 12
        and row.get("initial_energy", 0) > row.get("final_energy", 0)
        for row in trajectories
    ):
        reasons.append("repair_trajectory_replay_evidence_failed")
    if not (
        source_manifest.get("schema") == "finite_consensus_replay_source_v1"
        and isinstance(source_manifest.get("source_quotient_hash"), str)
        and source_manifest["source_quotient_hash"].startswith("sha256:")
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
    ):
        reasons.append("source_quotient_boundary_evidence_failed")
    if not (
        sector.get("DEFECT_SECTOR_SPECTRA_RECEIPT") is True
        and (sector.get("sector_family") or {}).get("group")
        == "Z6 character orbit of the declared reversing convention"
        and (sector.get("spectral_interface_identity") or {}).get(
            "rer_exact_flux_12_42_vertex_identity_bridge"
        )
        is False
        and sector.get("physical_promotion_allowed") is False
    ):
        reasons.append("finite_z6_sector_evidence_failed")
    if not (
        stage1.get("STAGE1_EVENT_COMPLEX_RECEIPT") is True
        and (stage1.get("atlas") or {}).get("covered_event_count") == 2304
        and stage1.get("physical_promotion_allowed") is False
    ):
        reasons.append("local_atlas_evidence_failed")
    scalar = (((stage3.get("section_typing") or {}).get("species") or {}).get("scalar") or {})
    if not (
        stage3.get("STAGE3_TYPED_DOMAIN_RECEIPT") is True
        and scalar.get("fiber_dimension") == 2
        and (stage3.get("kinetic_kernel_certificate") or {}).get("twisted_kernel_dimension") == 0
    ):
        reasons.append("typed_scalar_partial_evidence_failed")

    symbols = {
        "normal": _defined_functions(EXPECTED_SOURCES["normal_form_checker"]),
        "charged": _defined_functions(EXPECTED_SOURCES["charged_response_producer"]),
        "collar": _defined_functions(EXPECTED_SOURCES["collar_clause_checker"]),
    }
    if not {"verify_observation_determined_normal_form", "recognize_conditional_resampling_kernel"} <= symbols["normal"]:
        reasons.append("normal_form_symbols_missing")
    if not {"refinement_persistence", "produce_charged_response_artifact"} <= symbols["charged"]:
        reasons.append("charged_response_symbols_missing")
    if "verify_collar_clause_packet" not in symbols["collar"]:
        reasons.append("collar_checker_symbol_missing")
    ensemble_source = _safe_path(EXPECTED_SOURCES["reference_ensemble_producer"]).read_text("utf-8")
    if 'claim_tier="E1"' not in ensemble_source or "not an OPH-native vacuum" not in ensemble_source:
        reasons.append("reference_ensemble_not_explicit_control")


def _check_rows(matrix: Mapping[str, Any], reasons: list[str]) -> None:
    rows = matrix.get("capabilities")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_ROWS):
        reasons.append("capability_row_count_mismatch")
        return
    actual_ids = [row.get("capability_id") for row in rows if isinstance(row, Mapping)]
    if actual_ids != list(EXPECTED_ROWS):
        reasons.append("capability_row_order_mismatch")
    for row in rows:
        if not isinstance(row, Mapping):
            reasons.append("capability_row_malformed")
            continue
        capability_id = row.get("capability_id")
        expected = EXPECTED_ROWS.get(capability_id)
        if expected is None:
            reasons.append(f"unknown_capability:{capability_id}")
            continue
        classification, source_ids, check_ids = expected
        if row.get("classification") != classification:
            reasons.append(f"classification_mismatch:{capability_id}")
        if tuple(row.get("source_pin_ids") or ()) != source_ids:
            reasons.append(f"source_bindings_mismatch:{capability_id}")
        if tuple(row.get("machine_check_ids") or ()) != check_ids:
            reasons.append(f"machine_checks_mismatch:{capability_id}")
        if row.get("machine_checks_passed") is not True:
            reasons.append(f"machine_check_not_passed:{capability_id}")
        if row.get("adapter_promotion_allowed") is not False:
            reasons.append(f"adapter_promotion_enabled:{capability_id}")
        missing = row.get("missing_evidence")
        if classification == "MISSING" and (not isinstance(missing, list) or not missing):
            reasons.append(f"missing_row_lacks_blocker:{capability_id}")


def verify_capability_matrix(matrix: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        schema = json.loads(SCHEMA_PATH.read_text("utf-8"))
        errors = sorted(Draft202012Validator(schema).iter_errors(matrix), key=lambda e: list(e.path))
        reasons.extend(f"schema:{'/'.join(map(str, error.path))}:{error.message}" for error in errors)

        payload = copy.deepcopy(dict(matrix))
        digest = payload.pop("payload_sha256", None)
        if digest != _sha(payload):
            reasons.append("payload_sha256_mismatch")
        _check_pins(matrix, reasons)
        _check_tree_and_ancestry(matrix, reasons)
        _semantic_checks(reasons)
        _check_rows(matrix, reasons)
        try:
            if matrix.get("raw_twelve_port_response_probe") != _response_probe():
                reasons.append("raw_twelve_port_response_probe_mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append(f"raw_twelve_port_response_probe_failed:{exc}")

        expected_counts = {
            "AVAILABLE_SIMULATOR_NATIVE": 2,
            "AVAILABLE_CONDITIONAL": 1,
            "AVAILABLE_CONTROL_ONLY": 1,
            "MISSING": 7,
            "AMBIGUOUS": 0,
        }
        if matrix.get("classification_counts") != expected_counts:
            reasons.append("classification_counts_mismatch")
        stop = matrix.get("lane_stop_rules")
        if not isinstance(stop, Mapping) or not (
            stop.get("reserve_lane_blocked") is True
            and stop.get("cocycle_lane_blocked") is True
            and stop.get("screen_lane_blocked") is True
            and stop.get("large_simulation_authorized") is False
            and stop.get("cr1_or_later_implemented_here") is False
        ):
            reasons.append("lane_stop_rules_mismatch")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        reasons.append(f"verification_exception:{type(exc).__name__}:{exc}")
    return {
        "schema": "oph.common-reserve.capability-independent-verification.v1",
        "issue": 660,
        "stage": "CR-0",
        "receipt": not reasons,
        "reasons": sorted(set(reasons)),
        "scientific_promotion_allowed": False,
    }


def verify_file(path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("matrix root is not an object")
        return verify_capability_matrix(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "schema": "oph.common-reserve.capability-independent-verification.v1",
            "issue": 660,
            "stage": "CR-0",
            "receipt": False,
            "reasons": [f"matrix_parse_failed:{type(exc).__name__}:{exc}"],
            "scientific_promotion_allowed": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    args = parser.parse_args(argv)
    result = verify_file(args.matrix)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["receipt"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
