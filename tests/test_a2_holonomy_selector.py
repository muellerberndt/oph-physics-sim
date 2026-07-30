from __future__ import annotations

import copy
import json
from pathlib import Path

from oph_fpe.core.charged_response import produce_charged_response_artifact
from oph_fpe.core.spin_statistics_response import produce_spin_statistics_artifact
from oph_fpe.gauge.a2_holonomy_selector import (
    REQUIRED_RAW_SOURCE_OBJECTS,
    STATUS_OPEN,
    adversarial_controls,
    audit_released_source_artifacts,
    build_report,
    compact_inner_action_classification,
    port_action_fixed_dimension,
)


FIXTURE = Path(__file__).parent / "fixtures" / "echosahedral_federation_reference.json"


def _manifest() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_compact_inner_action_classifier_has_one_survivor() -> None:
    row = compact_inner_action_classification()
    assert row["passed"] is True
    assert row["a5_fixed_dimension"] == 1
    assert row["unique_lie_type"] == "u(1)+su(2)+su(3)"
    assert row["survivors"] == [
        {
            "center_dimension": 1,
            "semisimple_dimension": 11,
            "simple_factor_dimensions": [3, 8],
            "survives_compact_dimension_arithmetic": True,
            "survives_inner_fixed_space_test": True,
        }
    ]


def test_centerless_branch_is_excluded_by_fixed_space_arithmetic() -> None:
    row = compact_inner_action_classification()
    centerless = next(
        branch for branch in row["branches"] if branch["center_dimension"] == 0
    )
    assert centerless["simple_factor_dimensions"] == [3, 3, 3, 3]
    assert centerless["centerless_inner_a5_fixed_dimensions"] == [
        0,
        3,
        6,
        9,
        12,
    ]
    assert centerless["survives_inner_fixed_space_test"] is False


def test_fixed_dimension_is_recomputed_from_the_carrier_action() -> None:
    row = port_action_fixed_dimension(_manifest())
    assert row["port_dimension"] == 12
    assert row["proper_rotation_count"] == 60
    assert row["port_orbits"] == [list(range(12))]
    assert row["fixed_space_dimension"] == 1
    assert row["transitive"] is True


def test_released_artifacts_fail_closed_at_the_exact_missing_bridge() -> None:
    manifest = _manifest()
    charged = produce_charged_response_artifact(manifest)
    spin = produce_spin_statistics_artifact(manifest)
    audit = audit_released_source_artifacts(charged, spin)
    assert audit["same_carrier_source_projection"] is True
    assert audit["binary_icosahedral_deck_measurement_available"] is True
    assert audit["inverse_port_response_constraints_correctly_typed"] is True
    assert audit["a2_holonomy_source_bridge_receipt"] is False
    assert audit["registered_source_verifier"] is False
    assert audit["status"] == STATUS_OPEN
    assert audit["missing_objects"] == list(REQUIRED_RAW_SOURCE_OBJECTS)


def test_summary_boolean_or_model_label_cannot_forge_source_holonomy() -> None:
    manifest = _manifest()
    charged = produce_charged_response_artifact(manifest)
    spin = produce_spin_statistics_artifact(manifest)
    mutant = copy.deepcopy(charged)
    mutant["derived"]["current_lift_status"]["source_selected"] = True
    mutant["derived"]["construction"] = "charged_double_triplet"
    mutant["A2_HOLONOMY_SOURCE_BRIDGE_RECEIPT"] = True
    audit = audit_released_source_artifacts(mutant, spin)
    assert audit["a2_holonomy_source_bridge_receipt"] is False


def test_empty_named_source_objects_cannot_forge_source_holonomy() -> None:
    manifest = _manifest()
    charged = produce_charged_response_artifact(manifest)
    spin = produce_spin_statistics_artifact(manifest)
    charged["a2_holonomy_source_packet"] = {
        name: {} for name in REQUIRED_RAW_SOURCE_OBJECTS
    }
    audit = audit_released_source_artifacts(charged, spin)
    assert audit["all_raw_objects_present"] is True
    assert audit["registered_source_verifier"] is False
    assert audit["a2_holonomy_source_bridge_receipt"] is False


def test_all_shortcut_controls_are_rejected() -> None:
    controls = adversarial_controls()
    assert set(controls) == {
        "abelian_port_records",
        "binary_deck_on_independent_ancilla",
        "ambient_normalizer_without_response_generated_path",
        "semantic_model_label",
    }
    assert all(row["rejected"] is True for row in controls.values())


def test_report_separates_theorem_from_source_receipt() -> None:
    report = build_report(_manifest())
    assert report["theorem_classifier"]["passed"] is True
    assert report["source_bridge_audit"]["status"] == STATUS_OPEN
    assert report["receipts"] == {
        "A2_HOLONOMY_CLASSIFIER_RECEIPT": True,
        "A2_HOLONOMY_SOURCE_BRIDGE_RECEIPT": False,
        "PHYSICAL_SM_LIE_CURRENT_ALGEBRA_RECEIPT": False,
    }
    assert report["status"] == STATUS_OPEN
    assert report["report_sha256"].startswith("sha256:")
