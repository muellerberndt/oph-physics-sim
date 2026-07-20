"""Hypothesis-bound theorem shortcuts for the physical A5-to-SM ladder.

The shortcut rule is intentionally narrow: a verified theorem artifact may
compute a *derived implication* after every theorem hypothesis has an
independent boolean simulator receipt.  It can never create a source law,
physical observation, or QFT tier.  The only Lean artifact in the current
survival-proof package contains negative reconstruction no-gos; those results
are retained as fail-closed policy evidence and never promote a positive node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


THEOREM_AUDIT_SCHEMA_VERSION = "oph.a5-sm-theorem-evidence/1.0.0"
THEOREM_APPLICATION_SCHEMA_VERSION = "oph.a5-sm-theorem-application/1.0.0"
THEOREM_AUDIT_ARTIFACT_TYPE = "OPH_A5_SM_THEOREM_EVIDENCE_AUDIT"
THEOREM_APPLICATION_ARTIFACT_TYPE = "OPH_A5_SM_THEOREM_APPLICATION"


@dataclass(frozen=True)
class TheoremTransform:
    transform_id: str
    theorem_id: str
    hypothesis_receipts: tuple[str, ...]
    conclusion_receipts: tuple[str, ...]
    claim_boundary: str


TRANSFORMS: tuple[TheoremTransform, ...] = (
    TheoremTransform(
        "O_PORT",
        "conditional_icosahedral_port_selector",
        (
            "ECHOSAHEDRAL_LOCAL_PATCH_ARCHITECTURE_RECEIPT",
            "CALIBRATED_CURVATURE_KL_RISK_RECEIPT",
            "GROUND_STATE_COMPLETE_SETTLEMENT_RECEIPT",
            "ATOMIC_DEFECT_PROJECTION_RECEIPT",
            "PAIRWISE_FISHER_POSITIONAL_RISK_RECEIPT",
            "REGULAR_EDGEWISE_COFINAL_REFINEMENT_RECEIPT",
            "A5_SM_SOURCE_LAWS_TARGET_FREE_RECEIPT",
        ),
        (
            "A5_PORT_SELECTOR_THEOREM_APPLICATION_RECEIPT",
            "PHYSICAL_TWELVE_ATOMIC_A5_PORTS_RECEIPT",
        ),
        "Derives the twelve-port A5 orbit; it does not derive the local-to-global physical intertwiner.",
    ),
    TheoremTransform(
        "O_CURRENT",
        "inner_a5_compact_current_recognition",
        (
            "A5_PORT_SELECTOR_THEOREM_APPLICATION_RECEIPT",
            "PHYSICAL_TWELVE_ATOMIC_A5_PORTS_RECEIPT",
            "LOCAL_TO_GLOBAL_A5_INTERTWINER_RECEIPT",
            "MINIMAL_LOSSLESS_RECIPROCAL_RESPONSE_RECEIPT",
            "LOCAL_RANK12_CURRENT_FIBER_PER_PATCH_RECEIPT",
            "PHYSICAL_COMPACT_CONNECTED_CURRENT_GROUP_RECEIPT",
            "RAW_SYMMETRIC_RESPONSE_TOMOGRAPHY_RECEIPT",
            "PORT_CURRENT_FULL_RANK_IMAGE_EQUALITY_RECEIPT",
            "PORT_CURRENT_A5_EQUIVARIANCE_RECEIPT",
            "PORT_CURRENT_IRREP_CONDITIONING_RECEIPT",
            "PORT_CURRENT_GROUP_COMMUTATOR_BRACKET_RECEIPT",
            "REVERSIBLE_CURRENT_REPAIR_COVARIANCE_RECEIPT",
            "OFF_BLOCK_CENTER_VISIBILITY_RECEIPT",
            "FOUR_IRREP_SCALE_CALIBRATION_RECEIPT",
            "PORT_CURRENT_REFINEMENT_INTERTWINER_RECEIPT",
            "PORT_CURRENT_OVERLAP_TRANSPORT_INTERTWINER_RECEIPT",
            "A5_CURRENT_CLASSIFIER_ROUTE_RECEIPT",
        ),
        (
            "A5_COMPACT_CURRENT_RECOGNITION_THEOREM_APPLICATION_RECEIPT",
            "PHYSICAL_SM_LIE_CURRENT_ALGEBRA_RECEIPT",
            "PHYSICAL_LOCAL_SM_LIE_CURRENT_FIBER_RECEIPT",
        ),
        "Classifies every physical local current fiber after full-rank reversible-current realization. It is a classifier, not a producer, and does not identify dissipative repair with gauge flow.",
    ),
    TheoremTransform(
        "O_GLOBAL",
        "carrier_volume_z6_global_form",
        (
            "A5_COMPACT_CURRENT_RECOGNITION_THEOREM_APPLICATION_RECEIPT",
            "PHYSICAL_SM_LIE_CURRENT_ALGEBRA_RECEIPT",
            "MINIMAL_SUFFICIENT_PUBLIC_CARRIER_RECEIPT",
            "ORIENTED_PRIMITIVE_VOLUME_CLOCK_RECEIPT",
            "BW_TO_CENTRAL_VOLUME_CLOCK_INTERTWINER_RECEIPT",
            "BW_KMS_DIRECT_2PI_RECEIPT",
            "COMPLETE_PUBLIC_TENSOR_CATEGORY_RECEIPT",
            "PHYSICAL_PORT_LOOP_COCYCLE_COFINAL_DESCENT_RECEIPT",
            "UV_DEFECT_POLARIZATION_RECEIPT",
            "A5_EQUIVARIANT_CURRENT_CARRIER_INTERTWINER_RECEIPT",
        ),
        (
            "A5_Z6_GLOBAL_FORM_THEOREM_APPLICATION_RECEIPT",
            "PHYSICAL_Z6_GLOBAL_FORM_AND_LATTICE_RECEIPT",
        ),
        "Derives the 3+2 carrier weights and Z6 kernel only after the independently selected 2pi clock and physical deck/line laws.",
    ),
    TheoremTransform(
        "O_SMCORE_Q0",
        "finite_three_family_one_scalar_sm_core",
        (
            "A5_Z6_GLOBAL_FORM_THEOREM_APPLICATION_RECEIPT",
            "PHYSICAL_Z6_GLOBAL_FORM_AND_LATTICE_RECEIPT",
            "PHYSICAL_SPIN_CLIFFORD_STATISTICS_RECEIPT",
            "COMPLETE_ELEMENTARY_POLE_SPECTRAL_LEDGER_RECEIPT",
            "COMPLETE_PRIMITIVE_SCALAR_POLE_LEDGER_RECEIPT",
            "PHYSICAL_FIRST_POSITIVE_A5_BAND_FAMILY_ATTACHMENT_RECEIPT",
            "PROJECTED_AMPUTATED_SOURCE_THREE_POINT_RECEIPT",
            "ELEMENTARY_IMAGE_EQUALITY_RECEIPT",
        ),
        (
            "A5_FINITE_SM_Q0_FORCING_THEOREM_APPLICATION_RECEIPT",
            "PHYSICAL_FINITE_SM_Q0_CORE_RECEIPT",
        ),
        "Derives only the finite Q0 core. It cannot create Q1-Q4 regulator, measure, BV, or continuum receipts.",
    ),
)


EXPECTED_DAG_NODES = (
    "S_CURV",
    "S_SETTLE",
    "S_ATOM",
    "S_POS",
    "S_ESD",
    "O_PORT",
    "S_RESP",
    "S_TOMO",
    "O_CURRENT",
    "S_MINREP",
    "S_VOLUME",
    "S_CATEGORY",
    "S_DECK",
    "S_POLAR",
    "O_GLOBAL",
    "S_CLIFF",
    "S_POLES",
    "S_SCALAR",
    "S_FAMILY",
    "S_3PT",
    "O_SMCORE",
    "Q0",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
)

EXPECTED_DAG_EDGES = (
    ("S_CURV", "O_PORT"),
    ("S_SETTLE", "O_PORT"),
    ("S_ATOM", "O_PORT"),
    ("S_POS", "O_PORT"),
    ("S_ESD", "O_PORT"),
    ("O_PORT", "O_CURRENT"),
    ("S_RESP", "O_CURRENT"),
    ("S_TOMO", "O_CURRENT"),
    ("O_CURRENT", "O_GLOBAL"),
    ("S_MINREP", "O_GLOBAL"),
    ("S_VOLUME", "O_GLOBAL"),
    ("S_CATEGORY", "O_GLOBAL"),
    ("S_DECK", "O_GLOBAL"),
    ("S_POLAR", "O_GLOBAL"),
    ("O_GLOBAL", "O_SMCORE"),
    ("S_CLIFF", "O_SMCORE"),
    ("S_POLES", "O_SMCORE"),
    ("S_SCALAR", "O_SMCORE"),
    ("S_FAMILY", "O_SMCORE"),
    ("S_3PT", "O_SMCORE"),
    ("O_SMCORE", "Q0"),
    ("Q0", "Q1"),
    ("Q1", "Q2"),
    ("Q2", "Q3"),
    ("Q3", "Q4"),
)


def audit_survival_proof_theorems(
    proof_root: str | Path,
    *,
    lean_project: str | Path | None = None,
    run_checkers: bool = True,
) -> dict[str, Any]:
    """Audit the moving survival-proof package without trusting its master verdict."""

    root = Path(proof_root).resolve()
    blockers: list[str] = []
    required_files = {
        "dependency_dag": root / "model" / "forcing_dependency_dag.json",
        "lean_no_go": root / "formal" / "ForcingNoGo.lean",
        "survival_theorem": root / "PHYSICAL_A5_TO_SM_SURVIVAL_THEOREM.md",
        "forcing_theorem": root / "PHYSICAL_A5_TO_SM_FORCING_CONSTRUCTION.md",
        "pro_status": root / "PRO_INTEGRATION_2026-07-20.md",
        "legacy_receipt": root / "receipts" / "physical_a5_to_sm_survival.json",
        "issue_565_566_receipt": root / "model" / "constructive_565_566.receipt.json",
        "issue_567_receipt": root / "receipts" / "constructive-567.json",
        "issue_590_receipt": root / "constructive-590" / "receipt.json",
    }
    file_hashes: dict[str, str] = {}
    for name, path in required_files.items():
        if not path.is_file():
            blockers.append(f"missing_required_file:{name}")
            continue
        file_hashes[name] = _sha256_file(path)

    dag_valid = False
    dag_payload: dict[str, Any] = {}
    dag_path = required_files["dependency_dag"]
    if dag_path.is_file():
        try:
            dag_payload = json.loads(dag_path.read_text(encoding="utf-8"))
            node_ids = tuple(str(row.get("id")) for row in dag_payload.get("nodes", []))
            edges = tuple(tuple(str(value) for value in row) for row in dag_payload.get("edges", []))
            dag_valid = bool(
                node_ids == EXPECTED_DAG_NODES
                and edges == EXPECTED_DAG_EDGES
                and _dag_is_acyclic(node_ids, edges)
            )
        except (OSError, json.JSONDecodeError, TypeError):
            dag_valid = False
    if not dag_valid:
        blockers.append("authoritative_forcing_dependency_dag_mismatch")

    lean_source = (
        required_files["lean_no_go"].read_text(encoding="utf-8")
        if required_files["lean_no_go"].is_file()
        else ""
    )
    forbidden_lean_tokens = sorted(
        set(
            match.group(1)
            for match in re.finditer(
                r"(?m)^\s*(sorry|admit|axiom)\b",
                lean_source,
            )
        )
    )
    lean_compile = {
        "attempted": False,
        "passed": False,
        "command": None,
        "exit_code": None,
        "stdout_sha256": None,
        "stderr_sha256": None,
    }
    if run_checkers and lean_project is not None and required_files["lean_no_go"].is_file():
        lake = shutil.which("lake")
        if lake is None:
            blockers.append("lean_lake_executable_missing")
        else:
            command = [lake, "env", "lean", str(required_files["lean_no_go"])]
            lean_compile = _run_command(command, cwd=Path(lean_project))
            if not lean_compile["passed"]:
                blockers.append("lean_no_go_compile_failed")
    elif lean_project is not None:
        blockers.append("lean_no_go_compile_not_attempted")
    lean_no_go_verified = bool(
        not forbidden_lean_tokens
        and lean_compile["passed"] is True
    )

    checker_specs = {
        "legacy_no_go": (
            root / "scripts" / "verify_survival_proof.py",
            ["--check", str(required_files["legacy_receipt"])],
        ),
        "issue_565_566": (
            root / "scripts" / "verify_constructive_565_566.py",
            ["--check", str(required_files["issue_565_566_receipt"])],
        ),
        "issue_567": (
            root / "scripts" / "verify_constructive_567.py",
            ["--check", str(required_files["issue_567_receipt"])],
        ),
        "issue_590": (
            root / "constructive-590" / "verify.py",
            ["--check"],
        ),
    }
    checker_results: dict[str, dict[str, Any]] = {}
    for name, (script, arguments) in checker_specs.items():
        if not run_checkers:
            checker_results[name] = {
                "attempted": False,
                "passed": False,
                "command": None,
                "exit_code": None,
                "stdout_sha256": None,
                "stderr_sha256": None,
            }
            continue
        if not script.is_file():
            blockers.append(f"checker_script_missing:{name}")
            checker_results[name] = {
                "attempted": False,
                "passed": False,
                "command": None,
                "exit_code": None,
                "stdout_sha256": None,
                "stderr_sha256": None,
            }
            continue
        result = _run_command([sys.executable, str(script), *arguments], cwd=root.parent)
        checker_results[name] = result
        if result["passed"] is not True:
            blockers.append(f"leaf_checker_failed:{name}")

    theorem_rows = {
        "conditional_icosahedral_port_selector": _theorem_row(
            "conditional_icosahedral_port_selector",
            "analytic_conditional_theorem_plus_exact_reference",
            TRANSFORMS[0].hypothesis_receipts,
            TRANSFORMS[0].conclusion_receipts,
            checker_results.get("issue_565_566", {}).get("passed") is True,
            ("survival_theorem", "forcing_theorem", "issue_565_566_receipt"),
            file_hashes,
        ),
        "inner_a5_compact_current_recognition": _theorem_row(
            "inner_a5_compact_current_recognition",
            "analytic_conditional_theorem_plus_exact_reference",
            TRANSFORMS[1].hypothesis_receipts,
            TRANSFORMS[1].conclusion_receipts,
            checker_results.get("issue_565_566", {}).get("passed") is True,
            ("survival_theorem", "forcing_theorem", "issue_565_566_receipt"),
            file_hashes,
        ),
        "carrier_volume_z6_global_form": _theorem_row(
            "carrier_volume_z6_global_form",
            "analytic_conditional_theorem_plus_exact_finite_lattice_certificate",
            TRANSFORMS[2].hypothesis_receipts,
            TRANSFORMS[2].conclusion_receipts,
            checker_results.get("issue_567", {}).get("passed") is True,
            ("survival_theorem", "forcing_theorem", "issue_567_receipt"),
            file_hashes,
        ),
        "finite_three_family_one_scalar_sm_core": _theorem_row(
            "finite_three_family_one_scalar_sm_core",
            "conditional_finite_q0_theorem_plus_exact_reference",
            TRANSFORMS[3].hypothesis_receipts,
            TRANSFORMS[3].conclusion_receipts,
            checker_results.get("issue_590", {}).get("passed") is True,
            ("survival_theorem", "forcing_theorem", "pro_status", "issue_590_receipt"),
            file_hashes,
        ),
    }
    if not dag_valid:
        for row in theorem_rows.values():
            row["verified"] = False
            row["blockers"].append("authoritative_dependency_dag_not_verified")

    return {
        "schema_version": THEOREM_AUDIT_SCHEMA_VERSION,
        "artifact_type": THEOREM_AUDIT_ARTIFACT_TYPE,
        "proof_root": str(root),
        "source_file_sha256": file_hashes,
        "dependency_dag": {
            "verified": dag_valid,
            "schema": dag_payload.get("schema"),
            "node_ids": list(EXPECTED_DAG_NODES),
            "edges": [list(edge) for edge in EXPECTED_DAG_EDGES],
        },
        "lean_no_go": {
            "source_sha256": file_hashes.get("lean_no_go"),
            "forbidden_tokens": forbidden_lean_tokens,
            "compile": lean_compile,
            "LEAN_SOURCE_REDUCT_NO_GO_THEOREMS_RECEIPT": lean_no_go_verified,
            "positive_a5_to_sm_theorem_formalized_in_lean": False,
            "allowed_effect": "fail_closed_policy_only",
        },
        "leaf_checker_results": checker_results,
        "theorems": theorem_rows,
        "master_closure_receipt_consumed": False,
        "master_closure_reason": (
            "The moving master closure has had stale/schema and tuple-vs-list "
            "verification failures and previously carried a rejected line-origin rule. "
            "Only current leaf receipts and the Pro status boundary are consumed."
        ),
        "Q0_FINITE_CONDITIONAL_THEOREM_AVAILABLE": theorem_rows[
            "finite_three_family_one_scalar_sm_core"
        ]["verified"],
        "Q1_CLASSICAL_REGULATOR_THEOREM_AVAILABLE": False,
        "Q2_CHIRAL_MEASURE_THEOREM_AVAILABLE": False,
        "Q3_ALL_ORDERS_THEOREM_AVAILABLE": False,
        "Q4_CONTINUUM_THEOREM_AVAILABLE": False,
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "The Lean artifact proves only finite abstract no-go statements. Positive "
            "rows are imported conditional analytic/exact-reference theorem transforms, "
            "not Lean proofs and not evidence that the simulator emits their hypotheses."
        ),
    }


def apply_a5_sm_theorem_shortcuts(
    run_dir: str | Path,
    theorem_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply verified transforms sequentially to exact run receipts."""

    root = Path(run_dir)
    if not root.is_dir():
        raise ValueError(f"run directory does not exist: {root}")
    audit_errors = validate_theorem_audit(theorem_audit)
    observations, source_inventory = _index_boolean_receipts(root)
    derived: dict[str, bool] = {}
    transform_rows: dict[str, dict[str, Any]] = {}
    theorems = theorem_audit.get("theorems", {})
    for transform in TRANSFORMS:
        theorem = theorems.get(transform.theorem_id, {}) if isinstance(theorems, Mapping) else {}
        hypothesis_rows: dict[str, dict[str, Any]] = {}
        for receipt in transform.hypothesis_receipts:
            if receipt in derived:
                hypothesis_rows[receipt] = {
                    "passed": derived[receipt],
                    "source": "prior_theorem_transform",
                    "blockers": [] if derived[receipt] else ["prior_transform_false"],
                }
            else:
                hypothesis_rows[receipt] = _evaluate_boolean_receipt(
                    receipt,
                    observations,
                )
        theorem_verified = bool(
            not audit_errors
            and isinstance(theorem, Mapping)
            and theorem.get("verified") is True
            and theorem.get("positive_lean_proof") is False
            and theorem.get("hypothesis_receipts") == list(transform.hypothesis_receipts)
            and theorem.get("conclusion_receipts") == list(transform.conclusion_receipts)
        )
        passed = bool(
            theorem_verified
            and all(row["passed"] is True for row in hypothesis_rows.values())
        )
        for conclusion in transform.conclusion_receipts:
            derived[conclusion] = passed
        blockers = []
        if not theorem_verified:
            blockers.append("conditional_theorem_evidence_not_verified")
        blockers.extend(
            f"hypothesis_not_discharged:{receipt}"
            for receipt, row in hypothesis_rows.items()
            if row["passed"] is not True
        )
        transform_rows[transform.transform_id] = {
            "transform_id": transform.transform_id,
            "theorem_id": transform.theorem_id,
            "node_kind": "DERIVED_LOGICAL_IMPLICATION",
            "theorem_verified": theorem_verified,
            "positive_lean_proof": False,
            "hypotheses": hypothesis_rows,
            "conclusion_receipts": {
                receipt: passed for receipt in transform.conclusion_receipts
            },
            "passed": passed,
            "blockers": blockers,
            "claim_boundary": transform.claim_boundary,
        }
    return {
        "schema_version": THEOREM_APPLICATION_SCHEMA_VERSION,
        "artifact_type": THEOREM_APPLICATION_ARTIFACT_TYPE,
        "run_directory": str(root.resolve()),
        "theorem_audit_sha256": _stable_hash(theorem_audit),
        "theorem_audit_validation_errors": audit_errors,
        "source_inventory": source_inventory,
        "transforms": transform_rows,
        **derived,
        "LEAN_NO_GO_USED_FOR_POSITIVE_PROMOTION": False,
        "THEOREM_SHORTCUT_SOURCE_LAW_PROMOTION_COUNT": 0,
        "Q1_CLASSICAL_REGULATOR_RECEIPT": False,
        "Q2_CHIRAL_MEASURE_RECEIPT": False,
        "Q3_ALL_ORDERS_RECEIPT": False,
        "Q4_CONTINUUM_RECEIPT": False,
        "claim_boundary": (
            "A true conclusion means only that a verified conditional theorem was "
            "applied to independently true, hash-indexed run receipts. No theorem "
            "shortcut manufactured a source law or empirical observation."
        ),
    }


def validate_theorem_audit(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != THEOREM_AUDIT_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if report.get("artifact_type") != THEOREM_AUDIT_ARTIFACT_TYPE:
        errors.append("artifact_type_mismatch")
    dependency = report.get("dependency_dag")
    if not isinstance(dependency, Mapping) or dependency.get("verified") is not True:
        errors.append("dependency_dag_not_verified")
    theorems = report.get("theorems")
    if not isinstance(theorems, Mapping):
        return [*errors, "theorems_missing"]
    for transform in TRANSFORMS:
        row = theorems.get(transform.theorem_id)
        if not isinstance(row, Mapping):
            errors.append(f"theorem_missing:{transform.theorem_id}")
            continue
        if row.get("positive_lean_proof") is not False:
            errors.append(f"positive_lean_claim_forbidden:{transform.theorem_id}")
        if row.get("hypothesis_receipts") != list(transform.hypothesis_receipts):
            errors.append(f"hypothesis_contract_mismatch:{transform.theorem_id}")
        if row.get("conclusion_receipts") != list(transform.conclusion_receipts):
            errors.append(f"conclusion_contract_mismatch:{transform.theorem_id}")
        if row.get("verified") is True and not re.fullmatch(
            r"[0-9a-f]{64}", str(row.get("statement_bundle_sha256", ""))
        ):
            errors.append(f"verified_theorem_without_statement_hash:{transform.theorem_id}")
    if report.get("master_closure_receipt_consumed") is not False:
        errors.append("moving_master_closure_must_not_be_consumed")
    return errors


def write_theorem_audit(report: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_theorem_application(report: Mapping[str, Any], path: str | Path) -> Path:
    return write_theorem_audit(report, path)


def _theorem_row(
    theorem_id: str,
    evidence_class: str,
    hypotheses: tuple[str, ...],
    conclusions: tuple[str, ...],
    checker_passed: bool,
    source_names: tuple[str, ...],
    file_hashes: Mapping[str, str],
) -> dict[str, Any]:
    hashes = {name: file_hashes[name] for name in source_names if name in file_hashes}
    statement_hash = _stable_hash(
        {
            "theorem_id": theorem_id,
            "sources": hashes,
            "hypotheses": hypotheses,
            "conclusions": conclusions,
        }
    )
    return {
        "theorem_id": theorem_id,
        "evidence_class": evidence_class,
        "positive_lean_proof": False,
        "verified": bool(checker_passed and len(hashes) == len(source_names)),
        "hypothesis_receipts": list(hypotheses),
        "conclusion_receipts": list(conclusions),
        "source_file_sha256": hashes,
        "statement_bundle_sha256": statement_hash,
        "allowed_effect": "DERIVED_LOGICAL_IMPLICATION_ONLY",
        "blockers": [] if checker_passed else ["leaf_exact_reference_checker_failed"],
    }


def _run_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=180,
    )
    return {
        "attempted": True,
        "passed": completed.returncode == 0,
        "command": command,
        "exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
    }


def _dag_is_acyclic(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> bool:
    incoming = {node: 0 for node in nodes}
    outgoing = {node: [] for node in nodes}
    for source, target in edges:
        if source not in incoming or target not in incoming:
            return False
        incoming[target] += 1
        outgoing[source].append(target)
    queue = [node for node, count in incoming.items() if count == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for target in outgoing[node]:
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    return visited == len(nodes)


def _index_boolean_receipts(
    root: Path,
) -> tuple[dict[str, list[tuple[bool | object, str, str]]], dict[str, Any]]:
    observations: dict[str, list[tuple[bool | object, str, str]]] = {}
    paths: list[str] = []
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*.json")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping) and payload.get("artifact_type") in {
            THEOREM_APPLICATION_ARTIFACT_TYPE,
        }:
            continue
        paths.append(relative)
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        for key, value, pointer in _walk_values(payload):
            observations.setdefault(key, []).append((value, relative, pointer))
    return observations, {"report_paths": paths, "report_sha256": hashes}


def _walk_values(payload: Any, pointer: str = "") -> Iterable[tuple[str, Any, str]]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            name = str(key)
            child = f"{pointer}/{name.replace('~', '~0').replace('/', '~1')}"
            yield name, value, child
            yield from _walk_values(value, child)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from _walk_values(value, f"{pointer}/{index}")


def _evaluate_boolean_receipt(
    receipt: str,
    observations: Mapping[str, Sequence[tuple[Any, str, str]]],
) -> dict[str, Any]:
    rows = list(observations.get(receipt, ()))
    passed = bool(rows and all(type(value) is bool and value is True for value, _, _ in rows))
    blockers: list[str] = []
    if not rows:
        blockers.append("missing")
    if any(type(value) is not bool for value, _, _ in rows):
        blockers.append("non_boolean")
    if any(type(value) is bool and value is False for value, _, _ in rows):
        blockers.append("false_or_contradictory")
    return {
        "passed": passed,
        "source": "run_receipt",
        "observations": [
            {"value": value, "report_path": path, "json_pointer": pointer}
            for value, path, pointer in rows
        ],
        "blockers": blockers,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-survival-proof")
    audit.add_argument("--proof-root", type=Path, required=True)
    audit.add_argument("--lean-project", type=Path)
    audit.add_argument("--out", type=Path, required=True)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--run-dir", type=Path, required=True)
    apply.add_argument("--theorem-audit", type=Path, required=True)
    apply.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "audit-survival-proof":
        report = audit_survival_proof_theorems(
            args.proof_root,
            lean_project=args.lean_project,
        )
        write_theorem_audit(report, args.out)
        return 0 if not validate_theorem_audit(report) else 2
    audit = json.loads(args.theorem_audit.read_text(encoding="utf-8"))
    report = apply_a5_sm_theorem_shortcuts(args.run_dir, audit)
    write_theorem_application(report, args.out)
    return 0 if report["PHYSICAL_FINITE_SM_Q0_CORE_RECEIPT"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
