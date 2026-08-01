from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from oph_fpe.cli import main as cli_main
from oph_fpe.core.charged_response import ChargedResponseError, canonical_sha256
from oph_fpe.quantum.icosahedral_chsh_candidate import (
    CandidateError,
    build_receipt,
    validate_receipt,
)
from oph_fpe.quantum.verify_icosahedral_chsh_candidate_independent import (
    VerificationError,
    verify,
)


MANIFEST_PATH = (
    Path(__file__).parent / "fixtures" / "echosahedral_federation_reference.json"
)
RECEIPT_PATH = Path("data/quantum/icosahedral_chsh_candidate_receipt.json")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def rehash(receipt: dict) -> None:
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = canonical_sha256(body)


def test_exact_replay_matches_frozen_receipt(manifest: dict, receipt: dict) -> None:
    assert build_receipt(manifest) == receipt
    validate_receipt(receipt, manifest)


def test_independent_verifier_recomputes_candidate(
    manifest: dict, receipt: dict
) -> None:
    result = verify(receipt, manifest)
    assert result["status"] == "PASS"
    assert result["setting_rows"] == 120
    assert result["invariant_multiplicity"] == "1"
    assert result["absolute_chsh"] == "1 + 3/5*sqrt(5)"
    assert result["covariance_max_residual"] < 1.0e-12
    assert result["singlet_max_residual"] < 1.0e-12
    assert result["ambient_maximizer_count"] == 960


def test_candidate_does_not_promote_quantum_or_physical_record_claims(
    receipt: dict,
) -> None:
    source = receipt["source_derived_content"]
    assert source["carrier_lineage_scope"] == "declared_echosahedral_carrier_lineage"
    assert source["universal_oph_carrier_derivation"] is False
    parents = receipt["source_parents"]
    assert parents["finite_spin_packet_gate_passed"] is True
    assert parents["laboratory_exchange_measurement"] is False
    assert parents["continuum_spin_statistics_theorem"] is False
    gate = receipt["completed_record_gate"]
    assert gate["source_positive_nonclassical_record_witness"] is False
    assert gate["physical_promotion_allowed"] is False
    assert gate["first_missing_producer"] == "two_wing_completed_record_instrument"
    assert len(gate["missing_outputs_in_registered_source"]) == 6
    assert "not a nonclassical record witness" in receipt["claim_boundary"]
    selection = receipt["chsh_candidate"]["setting_selection"]
    assert selection["source_selected"] is False
    assert selection["unique_carrier_selected_family"] is False
    symmetry = receipt["chsh_candidate"]["setting_family_symmetry"]
    assert symmetry["proper_rotation_orbit_sizes"] == [60, 60]
    assert symmetry["ambient_maximizer_count"] == 960
    assert symmetry["ambient_all_distinct_maximizer_count"] == 480
    assert symmetry["declared_family_count"] == 120
    assert symmetry["declared_family_is_unique_maximizer"] is False


def test_false_source_promotion_is_rejected(manifest: dict, receipt: dict) -> None:
    mutant = copy.deepcopy(receipt)
    mutant["completed_record_gate"]["source_positive_nonclassical_record_witness"] = (
        True
    )
    rehash(mutant)
    with pytest.raises(VerificationError, match="false source promotion"):
        verify(mutant, manifest)


def test_false_spin_statistics_promotion_is_rejected(
    manifest: dict, receipt: dict
) -> None:
    mutant = copy.deepcopy(receipt)
    mutant["source_parents"]["continuum_spin_statistics_theorem"] = True
    rehash(mutant)
    with pytest.raises(VerificationError, match="false spin-statistics promotion"):
        verify(mutant, manifest)


def test_tampered_chsh_value_is_rejected(manifest: dict, receipt: dict) -> None:
    mutant = copy.deepcopy(receipt)
    mutant["chsh_candidate"]["absolute_chsh"] = "2*sqrt(2)"
    rehash(mutant)
    with pytest.raises(VerificationError, match="serialized absolute CHSH"):
        verify(mutant, manifest)


def test_tampered_setting_count_is_rejected(manifest: dict, receipt: dict) -> None:
    mutant = copy.deepcopy(receipt)
    mutant["chsh_candidate"]["setting_quadruple_count"] = 60
    rehash(mutant)
    with pytest.raises(VerificationError, match="serialized setting count"):
        verify(mutant, manifest)


def test_tampered_joint_probability_is_rejected(manifest: dict, receipt: dict) -> None:
    mutant = copy.deepcopy(receipt)
    mutant["chsh_candidate"]["joint_law"]["rows"][0]["probability"] = "1/4"
    rehash(mutant)
    with pytest.raises(VerificationError, match="joint-law values"):
        verify(mutant, manifest)


def test_false_setting_selection_is_rejected(manifest: dict, receipt: dict) -> None:
    mutant = copy.deepcopy(receipt)
    mutant["chsh_candidate"]["setting_selection"]["source_selected"] = True
    rehash(mutant)
    with pytest.raises(VerificationError, match="setting-selection boundary"):
        verify(mutant, manifest)


def test_tampered_setting_census_is_rejected(manifest: dict, receipt: dict) -> None:
    mutant = copy.deepcopy(receipt)
    mutant["chsh_candidate"]["setting_family_symmetry"]["ambient_maximizer_count"] = 120
    rehash(mutant)
    with pytest.raises(VerificationError, match="symmetry census"):
        verify(mutant, manifest)


def test_overstated_claim_boundary_is_rejected(manifest: dict, receipt: dict) -> None:
    mutant = copy.deepcopy(receipt)
    mutant["claim_boundary"] = "This is a physical Bell witness."
    rehash(mutant)
    with pytest.raises(VerificationError, match="claim-boundary prose"):
        verify(mutant, manifest)


def test_tampered_spin_parent_is_rejected(manifest: dict, receipt: dict) -> None:
    mutant = copy.deepcopy(receipt)
    mutant["source_parents"]["spin_statistics_artifact_sha256"] = "sha256:" + "0" * 64
    rehash(mutant)
    with pytest.raises(VerificationError, match="spin artifact pin"):
        verify(mutant, manifest)


def test_rewired_carrier_fails_before_candidate_promotion(manifest: dict) -> None:
    mutant = copy.deepcopy(manifest)
    mutant["carrier"]["edges"][0] = ["p00", "p01"]
    with pytest.raises((ChargedResponseError, CandidateError)):
        build_receipt(mutant)


def test_cli_validates_and_independently_verifies_frozen_candidate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    common = ["--manifest", str(MANIFEST_PATH)]
    assert (
        cli_main(
            [
                "icosahedral-chsh-candidate",
                *common,
                "--out",
                str(RECEIPT_PATH),
                "--validate-only",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "ICOSAHEDRAL_CHSH_CANDIDATE_VALID"

    assert (
        cli_main(
            [
                "verify-icosahedral-chsh-candidate",
                *common,
                "--receipt",
                str(RECEIPT_PATH),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PASS"
    assert output["source_positive_nonclassical_record_witness"] is False
