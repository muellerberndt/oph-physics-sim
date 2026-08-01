from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

from oph_fpe.cosmology.angular_refinement_repair_observability import (
    DEFAULT_OUTPUT,
    build_receipt,
    verify_receipt,
    write_receipt,
)
from oph_fpe.cosmology.verify_angular_refinement_repair_observability_independent import (
    verify as verify_independent,
)


def test_packet_has_two_refinement_repair_completions() -> None:
    report = build_receipt()
    refinement = report["refinement_and_repair"]
    witness = report["detail_witness"]

    assert witness["nonzero_entry_count"] == 6
    assert witness["coefficient_sum"] == 0
    assert witness["squared_norm"] == 6
    assert refinement["both_coarse_grain_to_constant_one"] is True
    assert refinement["completion_b_minimum_value"] == "1/2"
    assert refinement["both_produce_identical_inherited_repair_traces"] is True
    assert refinement["repair_trace_type"] == "uniform-schedule expectation semigroup"
    assert refinement["individual_seam_microhistories_identical"] is False
    observability = refinement["observability_witness"]
    assert observability["power_range"] == [0, 41]
    assert observability["maximum_inherited_absolute_value"] == 0
    assert observability["observability_matrix_shape"] == [504, 42]
    assert observability["observability_rank_over_Q"] == 29
    assert observability["repair_invisible_detail_dimension"] == 13
    assert observability["all_analytic_semigroup_times"] is True


def test_counterensemble_moves_covariance_not_mean() -> None:
    report = build_receipt()
    orbit = report["A5_invariant_counterensemble"]
    statistic = report["normalized_statistic"]
    decision = report["selection_decision"]

    assert orbit["proper_rotation_count"] == 60
    assert orbit["unique_detail_orbit_size"] == 20
    assert orbit["uniform_orbit_mean_is_zero"] is True
    assert orbit["ensemble_mean_field"] == "constant one"
    assert orbit["ensemble_covariance_nonzero"] is True
    assert statistic["statistic_type"] == (
        "centered two-point covariance power at angular degree six"
    )
    assert statistic["completion_a"] == "0"
    assert statistic["completion_b"] == "15/57344"
    assert statistic["completion_b_strictly_positive"] is True
    assert statistic["support_measure_physical"] is False
    assert decision["mean_field_agrees_between_completions"] == "constant one"
    assert decision["mean_field_globally_source_selected"] is False
    assert decision["covariance_selected_by_shared_antecedents"] is False
    assert decision["repair_schedule_source_selected"] is False
    assert decision["unit_counting_measure_physical"] is False
    assert decision["physical_sky_readout_selected"] is False
    assert decision["physical_angular_prediction"] is False
    assert decision["issue_closure_authorized"] is False


def test_committed_receipt_and_both_verifiers_pass() -> None:
    report = build_receipt()
    assert DEFAULT_OUTPUT.read_bytes() == (
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    assert verify_receipt(report)["receipt"] is True
    independent = verify_independent(report)
    assert independent["receipt"] is True
    assert independent["independent_implementation"] is True
    assert independent["producer_imported"] is False


def test_independent_verifier_does_not_import_producer() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "oph_fpe/cosmology/verify_angular_refinement_repair_observability_independent.py"
    )
    tree = ast.parse(path.read_text("utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(
        name.endswith("angular_refinement_repair_observability")
        for name in imported
    )


def test_mutations_fail_closed() -> None:
    report = build_receipt()

    angular = copy.deepcopy(report)
    angular["normalized_statistic"]["completion_b"] = "0"
    assert verify_receipt(angular)["receipt"] is False
    result = verify_independent(angular)
    assert result["receipt"] is False
    assert "payload_hash_mismatch" in result["reasons"]
    assert "reported_degree_six_statistic_mismatch" in result["reasons"]

    promoted = copy.deepcopy(report)
    promoted["selection_decision"]["physical_angular_prediction"] = True
    assert verify_receipt(promoted)["receipt"] is False
    result = verify_independent(promoted)
    assert result["receipt"] is False
    assert "forbidden_selection_or_physical_promotion" in result["reasons"]

    witness = copy.deepcopy(report)
    witness["detail_witness"]["semantic_rows"][0]["coefficient"] = 0
    assert verify_receipt(witness)["receipt"] is False
    result = verify_independent(witness)
    assert result["receipt"] is False
    assert "reported_detail_witness_mismatch" in result["reasons"]


def test_writer_and_independent_entrypoint(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    report = write_receipt(path)
    assert json.loads(path.read_text("ascii")) == report
    assert path.read_bytes().endswith(b"\n")
    assert b"\r\n" not in path.read_bytes()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            (
                "oph_fpe.cosmology."
                "verify_angular_refinement_repair_observability_independent"
            ),
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["receipt"] is True
