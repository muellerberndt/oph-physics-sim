from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from oph_fpe.gauge.a5_sm_certificate import (
    A5_TWELVE_PORT_STRUCTURAL_RECEIPT,
    CONDITIONAL_EXTERIOR_ONE_GENERATION_WITNESS_RECEIPT,
    NO_INVARIANT_PORT_PARTITION_8_3_1_RECEIPT,
    PHYSICAL_STANDARD_MODEL_PROMOTION_RECEIPT,
    SM_ADJOINT_CHARACTER_MATCH_RECEIPT,
    a5_sm_structural_certificate,
    verify_a5_sm_structural_certificate,
    write_a5_sm_structural_certificate,
)


def test_a5_coset_action_and_character_decomposition_are_recomputed_exactly():
    report = a5_sm_structural_certificate()
    native = report["native_computation"]
    group = native["a5_group"]
    action = native["h5_coset_action"]
    port = native["character_theory"]["port_module"]

    assert report[A5_TWELVE_PORT_STRUCTURAL_RECEIPT] is True
    assert group["order"] == 60
    assert group["element_order_counts"] == {"1": 1, "2": 15, "3": 20, "5": 24}
    assert [row["size"] for row in group["conjugacy_classes"]] == [1, 15, 20, 12, 12]
    assert action["subgroup_order"] == 5
    assert action["coset_count"] == 12
    assert action["action_transitive"] is True
    assert action["base_stabilizer_equals_h5"] is True
    assert action["kernel_order"] == 1
    assert action["faithful"] is True
    assert action["permutation_character"] == [12, 0, 0, 2, 2]
    assert port["decomposition"] == {"1": 1, "3": 1, "3prime": 1, "4": 0, "5": 1}
    assert port["multiplicity_free"] is True


def test_icosahedral_adjacency_spectrum_and_polynomials_are_exact():
    report = a5_sm_structural_certificate()
    adjacency = report["native_computation"]["icosahedral_adjacency"]

    assert adjacency["vertex_count"] == 12
    assert adjacency["edge_count"] == 30
    assert adjacency["degree_sequence"] == [5] * 12
    assert adjacency["triangle_count"] == 20
    assert adjacency["distance_layer_profiles"] == [[1, 5, 5, 1]]
    assert adjacency["a5_invariant"] is True
    assert adjacency["characteristic_polynomial"]["factorization"] == "(x-5)(x+1)^5(x^2-5)^3"
    assert adjacency["characteristic_polynomial"]["factorization_verified"] is True
    assert adjacency["minimal_polynomial"]["annihilator_is_zero"] is True
    assert adjacency["spectrum"] == {"5": 1, "sqrt(5)": 3, "-sqrt(5)": 3, "-1": 5}
    assert adjacency["canonical_rank_sequence"] == [1, 3, 3, 5]


def test_sm_adjoint_character_match_does_not_bypass_set_partition_antibridge():
    report = a5_sm_structural_certificate()
    native = report["native_computation"]
    match = native["character_theory"]["sm_adjoint_match"]
    antibridge = native["invariant_partition_antibridge"]

    assert report[SM_ADJOINT_CHARACTER_MATCH_RECEIPT] is True
    assert match["both_opposite_triplet_assignments_match"] is True
    assert [row["su3_adjoint_dimension"] for row in match["assignments"]] == [8, 8]
    assert [row["sm_adjoint_character"] for row in match["assignments"]] == [
        ["12", "0", "0", "2", "2"],
        ["12", "0", "0", "2", "2"],
    ]
    assert antibridge["invariant_subset_size_counts"] == {"0": 1, "12": 1}
    assert antibridge["invariant_8_3_1_set_partition_exists"] is False
    assert antibridge["linear_module_8_3_1_decomposition_exists"] is True
    assert report[NO_INVARIANT_PORT_PARTITION_8_3_1_RECEIPT] is True
    assert report[PHYSICAL_STANDARD_MODEL_PROMOTION_RECEIPT] is False
    assert report["physical_claim"] is False
    assert report["physical_promotion"]["promotion_allowed"] is False
    assert "PORT_CURRENT_INNER_RECEIPT" in report["blockers"]
    assert "PORT_REFINEMENT_INTERTWINER_RECEIPT" in report["blockers"]
    assert "MAR_MATTER_REALIZATION_RECEIPT" in report["blockers"]


def test_conditional_exterior_generation_fields_and_higgs_lines_are_exact():
    report = a5_sm_structural_certificate()
    exterior = report["native_computation"]["exterior_one_generation_witness"]
    package = exterior["matter_package"]
    fields = package["fields"]
    higgs = exterior["one_higgs_invariant_lines"]

    assert report[CONDITIONAL_EXTERIOR_ONE_GENERATION_WITNESS_RECEIPT] is True
    assert exterior["carrier"]["trace_balance"] == "0"
    assert exterior["carrier"]["trace_balanced"] is True
    assert package["definition"] == "M1=Lambda^2(C+W)+Lambda^4(C+W)"
    assert package["lambda2_dimension"] == 10
    assert package["lambda4_dimension"] == 5
    assert package["complex_dimension"] == 15
    assert package["field_dimension_sum"] == 15
    assert {
        name: (row["su3"], row["su2"], row["hypercharge"], row["dimension"])
        for name, row in fields.items()
    } == {
        "Q": ("3", "2", "1/6", 6),
        "u_c": ("3bar", "1", "-2/3", 3),
        "e_c": ("1", "1", "1", 1),
        "d_c": ("3bar", "1", "1/3", 3),
        "L": ("1", "2", "-1/2", 2),
    }
    assert package["field_signature_matches_canonical_generation"] is True
    assert package["dual_overlap"] == []
    assert package["disjoint_from_dual"] is True
    assert higgs["conditional_scalar_identification"] == "H=W"
    assert higgs["invariant_line_count"] == 3
    assert higgs["all_channels_exact_singlets"] is True
    assert all(row["hypercharge_sum"] == "0" for row in higgs["channels"].values())
    assert all(row["invariant_line_multiplicity"] == 1 for row in higgs["channels"].values())


def test_exterior_anomalies_and_witten_parity_pass_but_physical_gates_stay_false():
    report = a5_sm_structural_certificate()
    exterior = report["native_computation"]["exterior_one_generation_witness"]
    anomalies = exterior["anomalies"]
    weak = exterior["weak_doublet_count"]
    boundary = exterior["selection_boundary"]
    gates = report["physical_promotion"]["gate_status"]

    assert set(anomalies["coefficients"].values()) == {"0"}
    assert anomalies["all_coefficients_zero"] is True
    assert anomalies["su2_witten_doublet_count"] == 4
    assert anomalies["su2_witten_parity_mod_2"] == 0
    assert anomalies["su2_witten_parity_even"] is True
    assert weak["colored_Q_copies"] == 3
    assert weak["lepton_L_copies"] == 1
    assert weak["multiplicity_per_generation"] == 4
    assert weak["conditional_beta_EW"] == 4
    assert weak["physical_load_identification"] is False
    assert boundary["full_even_exterior_dimension_including_lambda0"] == 16
    assert boundary["vacuum_singlet_excluded_by_computation"] is False
    assert boundary["other_anomaly_free_light_sectors_excluded"] is False
    assert gates["CONDITIONAL_EXTERIOR_ONE_GENERATION_WITNESS"] is True
    for gate in (
        "EXTERIOR_PACKAGE_SELECTION_RECEIPT",
        "HIGGS_SCALAR_SELECTION_RECEIPT",
        "A5_FAMILY_ATTACHMENT_RECEIPT",
        "A5_FAMILY_DESCENT_RECEIPT",
        "PORT_WEAK_INTERTWINER_RECEIPT",
        "PORT_LOAD_TRACE_RECEIPT",
        "CONTINUUM_LIMIT_RECEIPT",
        "SPIN_QFT_REALIZATION_RECEIPT",
        "NO_EXTRA_LIGHT_SECTOR_RECEIPT",
    ):
        assert gates[gate] is False
        assert gate in report["blockers"]
    assert report[PHYSICAL_STANDARD_MODEL_PROMOTION_RECEIPT] is False
    assert report["physical_promotion"]["promotion_allowed"] is False


def test_imported_theorem_evidence_is_sanitized_and_never_promotes():
    report = a5_sm_structural_certificate(
        [
            {
                "source": "locked-research/theorem.lean",
                "source_commit": "example",
                "content_sha256": "a" * 64,
                "theorems": ["example.theorem"],
                "verification_status": "declared_verified",
                "simulation_receipt_eligible": True,
                "physical_standard_model_derivation": True,
            }
        ]
    )
    imported = report["evidence_layers"]["imported_theorem_evidence"]
    row = imported["records"][0]

    assert imported["provided_count"] == 1
    assert imported["all_well_formed"] is True
    assert imported["simulation_receipts_promoted_by_import"] is False
    assert imported["used_for_native_computation"] is False
    assert row["simulation_receipt_eligible"] is False
    assert row["used_for_native_receipts"] is False
    assert report[A5_TWELVE_PORT_STRUCTURAL_RECEIPT] is True
    assert report[PHYSICAL_STANDARD_MODEL_PROMOTION_RECEIPT] is False
    assert verify_a5_sm_structural_certificate(report)["verified"] is True


def test_schema_and_independent_verifier_reject_forged_native_fields():
    report = a5_sm_structural_certificate()
    schema = json.loads(
        Path("schemas/gauge/a5_sm_certificate.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)

    forged = copy.deepcopy(report)
    forged["native_computation"]["h5_coset_action"]["permutation_character"] = [12, 0, 0, 1, 3]
    forged["certificate_payload_sha256"] = _payload_hash(forged)
    verification = verify_a5_sm_structural_certificate(forged)

    assert verification["verified"] is False
    assert verification["checks"]["payload_hash_matches"] is True
    assert verification["checks"]["native_computation_matches_recomputation"] is False
    assert "native_computation_matches_recomputation" in verification["blockers"]


def test_verifier_rejects_physical_promotion_even_with_a_rehashed_payload():
    report = a5_sm_structural_certificate()
    forged = copy.deepcopy(report)
    forged[PHYSICAL_STANDARD_MODEL_PROMOTION_RECEIPT] = True
    forged["physical_claim"] = True
    forged["physical_promotion"]["promotion_allowed"] = True
    forged["certificate_payload_sha256"] = _payload_hash(forged)

    verification = verify_a5_sm_structural_certificate(forged)

    assert verification["verified"] is False
    assert verification["checks"]["payload_hash_matches"] is True
    assert verification["checks"]["physical_promotion_receipt_is_false"] is False
    assert verification["checks"]["physical_claim_is_false"] is False
    assert verification["checks"]["physical_promotion_matches_fail_closed_contract"] is False


def test_writer_emits_a_schema_valid_self_verifying_certificate(tmp_path: Path):
    report = write_a5_sm_structural_certificate(tmp_path)
    path = tmp_path / "a5_sm_structural_certificate.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded == report
    assert verify_a5_sm_structural_certificate(loaded)["verified"] is True


def _payload_hash(report: dict) -> str:
    payload = copy.deepcopy(report)
    payload.pop("certificate_payload_sha256", None)
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
