from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from oph_fpe.bulk.collar_cmi_decay_307 import (
    CMI_BOUND_RECEIPT,
    DOUBLE_SCALING_RECEIPT,
    FINITE_RANGE_GIBBS_RECEIPT,
    RECEIPT_NAME,
    REGIONAL_CMI_RECEIPT,
    STRONG_MIXING_RECEIPT,
    issue307_collar_cmi_decay_report,
    write_issue307_collar_cmi_decay_report,
)
from oph_fpe.claims import ISSUE_307_COLLAR_CMI_DECAY_RECEIPT


PATCH_COUNTS = (1_000_000, 100_000_000, 10_000_000_000)


def test_receipt_name_is_registered() -> None:
    assert RECEIPT_NAME == ISSUE_307_COLLAR_CMI_DECAY_RECEIPT


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fixture(*, alpha: float = 0.25) -> dict:
    screen_measure = 4.0 * math.pi
    delta_prefactor = 0.8
    c_mix = 2.0
    xi_cells = 1.0
    gibbs_stages = []
    mixing_stages = []
    regional_rows = []
    for patch_count in PATCH_COUNTS:
        stage_id = f"N{patch_count}"
        gibbs_stages.append(
            {
                "stage_id": stage_id,
                "patch_count": patch_count,
                "beta": 1.0,
                "interaction_range_uv": 2.0,
                "max_term_support_diameter_uv": 1.0,
                "max_term_norm": 0.75,
                "max_degree": 4,
                "hamiltonian_term_count": patch_count - 1,
                "term_support_count": patch_count - 1,
                "hamiltonian_reconstruction_residual": 0.0,
                "gibbs_state_trace_residual": 0.0,
                "hamiltonian_terms_sha256": _sha(f"terms-{stage_id}"),
                "gibbs_state_sha256": _sha(f"state-{stage_id}"),
            }
        )
        distance_rows = []
        for distance in (1.0, 2.0, 4.0):
            declared_bound = c_mix * math.exp(-distance / xi_cells)
            distance_rows.append(
                {
                    "distance_uv": distance,
                    "conditional_matrix_influence_upper": 0.5 * declared_bound,
                }
            )
        mixing_stages.append(
            {
                "stage_id": stage_id,
                "c_mix": c_mix,
                "xi_uv_cells": xi_cells,
                "boundary_condition_count": 4,
                "cap_count": 1,
                "conditional_matrix_evidence_sha256": _sha(f"mixing-{stage_id}"),
                "distance_rows": distance_rows,
            }
        )

        ell_uv = math.sqrt(screen_measure / patch_count)
        delta = delta_prefactor * patch_count ** (-alpha)
        boundary_size = int(math.ceil(0.5 * math.sqrt(patch_count)))
        log_bound = (
            math.log(c_mix)
            + math.log(boundary_size)
            - delta / (xi_cells * ell_uv)
        )
        cmi = min(0.01, 0.25 * math.exp(log_bound))
        regional_rows.append(
            {
                "stage_id": stage_id,
                "cap_id": "cap-0",
                "cap_family_id": "fixed-round-cap-family-v1",
                "patch_count": patch_count,
                "delta": delta,
                "boundary_size_uv": boundary_size,
                "boundary_measure_kind": "cut_edge_count",
                "cmi_estimator": "exact_density_matrix_entropy",
                "regions": ["A_delta", "B_delta", "D_delta"],
                "entropy_terms_nats": {"AB": cmi, "BD": 0.0, "B": 0.0, "ABD": 0.0},
                "regional_cmi_nats": cmi,
                "regional_cmi_numerical_error_nats": 0.0,
                "regional_state_evidence_sha256": _sha(f"regional-{stage_id}"),
                "boundary_evidence_sha256": _sha(f"boundary-{stage_id}"),
            }
        )

    return {
        "schema_version": 1,
        "gibbs_evidence": {
            "state_semantics": "noncommutative_density_matrix",
            "interaction_family_id": "finite-range-chain-v1",
            "graph_metric_id": "declared-screen-graph-v1",
            "range_unit": "uv_cell",
            "beta": 1.0,
            "interaction_range_uv": 2.0,
            "max_term_norm_bound": 1.0,
            "max_degree_bound": 6,
            "stages": gibbs_stages,
        },
        "mixing_evidence": {
            "kind": "strong_conditional_matrix_mixing",
            "constants_scope": "uniform_over_declared_stage_cap_boundary_family",
            "influence_norm": "trace_norm",
            "c_mix": c_mix,
            "xi_uv_cells": xi_cells,
            "stages": mixing_stages,
        },
        "regional_cmi_evidence": {
            "state_semantics": "noncommutative_density_matrix",
            "log_unit": "nat",
            "rows": regional_rows,
        },
        "scaling_schedule": {
            "kind": "power_law_patch_count",
            "screen_measure": screen_measure,
            "uv_dimension": 2,
            "boundary_dimension": 1,
            "delta_prefactor": delta_prefactor,
            "alpha": alpha,
            "boundary_prefactor_bound": 0.51,
            "min_stage_count": 3,
            "min_final_log_decay_margin": 10.0,
            "delta_metric": "screen_geodesic",
        },
    }


def test_issue307_verifier_passes_and_ignores_caller_flags() -> None:
    payload = _fixture()
    payload.update(
        {
            RECEIPT_NAME: False,
            FINITE_RANGE_GIBBS_RECEIPT: False,
            "EINSTEIN_BRANCH_ENTRY_RECEIPT": True,
            "MODULAR_SOURCE_CHARGE_RECEIPT": True,
            "physical_claim": True,
        }
    )

    report = issue307_collar_cmi_decay_report(payload)

    assert report[RECEIPT_NAME] is True
    assert report[FINITE_RANGE_GIBBS_RECEIPT] is True
    assert report[STRONG_MIXING_RECEIPT] is True
    assert report[REGIONAL_CMI_RECEIPT] is True
    assert report[CMI_BOUND_RECEIPT] is True
    assert report[DOUBLE_SCALING_RECEIPT] is True
    assert report["claim_level"] == "branch_instantiation_sanity"
    assert report["physical_claim"] is False
    assert report["EINSTEIN_BRANCH_ENTRY_RECEIPT"] is False
    assert report["CMI_TO_MODULAR_SOURCE_MATCHING_RECEIPT"] is False
    assert report["MODULAR_SOURCE_CHARGE_RECEIPT"] is False
    assert report["theorem_grade_parent_collar_ladder"] is False
    assert set(report["ignored_caller_pass_fields"]) >= {
        RECEIPT_NAME,
        FINITE_RANGE_GIBBS_RECEIPT,
        "EINSTEIN_BRANCH_ENTRY_RECEIPT",
        "MODULAR_SOURCE_CHARGE_RECEIPT",
        "physical_claim",
    }
    computed_rows = report["clauses"]["regional_collar_cmi"]["details"]["rows"]
    assert all(row["bound_passed"] for row in computed_rows)
    assert all(row["recomputed_log_bound_slack"] < 0.0 for row in computed_rows)
    assert report["clauses"]["sharp_double_scaling_rate"]["details"][
        "delta_over_ell_power_margin"
    ] == 0.25


def test_issue307_rejects_nonlocal_hamiltonian_term() -> None:
    payload = _fixture()
    payload["gibbs_evidence"]["stages"][0]["max_term_support_diameter_uv"] = 3.0

    report = issue307_collar_cmi_decay_report(payload)

    assert report[FINITE_RANGE_GIBBS_RECEIPT] is False
    assert report[RECEIPT_NAME] is False
    assert any("nonlocal_term_support" in item for item in report["promotion_blockers"])


def test_issue307_rejects_gibbs_state_mismatch() -> None:
    payload = _fixture()
    payload["gibbs_evidence"]["stages"][1]["gibbs_state_trace_residual"] = 1.0e-3

    report = issue307_collar_cmi_decay_report(payload)

    assert report[FINITE_RANGE_GIBBS_RECEIPT] is False
    assert any("gibbs_state_mismatch" in item for item in report["promotion_blockers"])


def test_issue307_rejects_missing_or_ordinary_mixing() -> None:
    missing = _fixture()
    missing.pop("mixing_evidence")
    missing_report = issue307_collar_cmi_decay_report(missing)
    assert missing_report[STRONG_MIXING_RECEIPT] is False
    assert missing_report[CMI_BOUND_RECEIPT] is False

    ordinary = _fixture()
    ordinary["mixing_evidence"]["kind"] = "ordinary_two_point_exponential_clustering"
    ordinary_report = issue307_collar_cmi_decay_report(ordinary)
    assert ordinary_report[STRONG_MIXING_RECEIPT] is False
    assert "ordinary_clustering_is_not_strong_conditional_matrix_mixing" in ordinary_report[
        "promotion_blockers"
    ]


def test_issue307_rejects_mixed_uniform_constants() -> None:
    payload = _fixture()
    payload["mixing_evidence"]["stages"][1]["xi_uv_cells"] = 1.2

    report = issue307_collar_cmi_decay_report(payload)

    assert report[STRONG_MIXING_RECEIPT] is False
    assert any("xi_uv_cells_not_uniform" in item for item in report["promotion_blockers"])


def test_issue307_recomputes_and_rejects_cmi_bound_violation() -> None:
    payload = _fixture()
    row = payload["regional_cmi_evidence"]["rows"][1]
    patch_count = row["patch_count"]
    ell_uv = math.sqrt(payload["scaling_schedule"]["screen_measure"] / patch_count)
    log_bound = (
        math.log(payload["mixing_evidence"]["c_mix"])
        + math.log(row["boundary_size_uv"])
        - row["delta"] / (payload["mixing_evidence"]["xi_uv_cells"] * ell_uv)
    )
    violating_cmi = 2.0 * math.exp(log_bound)
    row["regional_cmi_nats"] = violating_cmi
    row["entropy_terms_nats"]["AB"] = violating_cmi
    # A caller-provided value cannot hide the recomputed failure.
    row["boundary_prefactored_bound_nats"] = violating_cmi * 10.0
    payload[CMI_BOUND_RECEIPT] = True

    report = issue307_collar_cmi_decay_report(payload)

    assert report[REGIONAL_CMI_RECEIPT] is True
    assert report[CMI_BOUND_RECEIPT] is False
    assert report[RECEIPT_NAME] is False
    assert any("boundary_prefactored_cmi_bound_violated" in item for item in report["promotion_blockers"])


def test_issue307_rejects_too_slow_double_scaling_schedule() -> None:
    payload = _fixture(alpha=0.5)

    report = issue307_collar_cmi_decay_report(payload)

    assert report[FINITE_RANGE_GIBBS_RECEIPT] is True
    assert report[STRONG_MIXING_RECEIPT] is True
    assert report[REGIONAL_CMI_RECEIPT] is True
    assert report[CMI_BOUND_RECEIPT] is True
    assert report[DOUBLE_SCALING_RECEIPT] is False
    assert report[RECEIPT_NAME] is False
    assert "delta_over_ell_uv_does_not_diverge_with_positive_power_margin" in report[
        "promotion_blockers"
    ]


def test_issue307_rejects_mixed_cap_family() -> None:
    payload = _fixture()
    payload["regional_cmi_evidence"]["rows"][-1]["cap_family_id"] = "different-cap-family"

    report = issue307_collar_cmi_decay_report(payload)

    assert report[REGIONAL_CMI_RECEIPT] is False
    assert "mixed_cap_families_in_regional_cmi_ladder" in report["promotion_blockers"]


def test_issue307_writer_has_path_friendly_api(tmp_path: Path) -> None:
    source = tmp_path / "primitive.json"
    output = tmp_path / "receipt.json"
    source.write_text(json.dumps(_fixture()), encoding="utf-8")

    report = write_issue307_collar_cmi_decay_report(source, output)
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report[RECEIPT_NAME] is True
    assert written[RECEIPT_NAME] is True
    assert written["algorithm"] == "oph-issue307-collar-cmi-decay-v1"


def test_issue307_nested_primitive_payload_is_supported() -> None:
    primitive = _fixture()
    wrapper = {
        "issue_307_primitive_fields": deepcopy(primitive),
        RECEIPT_NAME: False,
    }

    report = issue307_collar_cmi_decay_report(wrapper)

    assert report[RECEIPT_NAME] is True
    assert report["ignored_caller_pass_fields"][RECEIPT_NAME] is False
