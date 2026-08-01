from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from oph_fpe.dynamics import vertex12_atomic_port_transfer as producer
from oph_fpe.dynamics import (
    verify_vertex12_atomic_port_transfer_independent as independent,
)


def _sha(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _rehash(report: dict) -> dict:
    result = copy.deepcopy(report)
    result.pop("receipt_sha256", None)
    result["receipt_sha256"] = _sha(result)
    return result


@pytest.fixture(scope="module")
def receipt() -> dict:
    return producer.produce_receipt()


def test_packet_attains_internal_transfer_and_complete_source_readback(
    receipt: dict,
) -> None:
    assert producer.verify_receipt(receipt)["receipt"] is True
    assert receipt["status"] == producer.STATUS
    operator = receipt["atomic_transfer_operator"]
    assert operator["carrier_count"] == 8
    assert operator["port_count"] == 12
    assert operator["seam_count"] == 48
    assert operator["complete_matching_on_every_port"] is True
    assert operator["all_twelve_transfer_involutions_exact"] is True
    assert operator["all_twelve_rational_repair_projectors_exact"] is True
    assert operator["source_native_internal_port_transfer_receipt"] is True
    assert operator["source_native_spatial_translation_receipt"] is False
    assert len(operator["port_rows"]) == 12
    for row in operator["port_rows"]:
        permutation = row["carrier_partner_permutation"]
        assert sorted(permutation) == list(range(8))
        assert all(permutation[permutation[index]] == index for index in range(8))
        assert all(permutation[index] != index for index in range(8))
        assert row["matching_size"] == 4
        assert row["repair_projector_idempotent"] is True

    history = receipt["source_history_replay"]
    assert history["event_count"] == 48
    assert history["every_seam_replayed_once"] is True
    assert history["every_event_is_atomic_two_endpoint_mean"] is True
    assert history["terminal_write_coordinate_count"] == 96
    assert history["terminal_write_state_matches_readback_snapshot"] is True
    readback = receipt["post_repair_source_readback"]
    assert readback["covered_carrier_count"] == 8
    assert readback["covered_port_coordinate_count"] == 96
    assert (
        history["terminal_write_state_rows_sha256"]
        == readback["terminal_state_rows_sha256"]
    )
    assert readback["every_carrier_full_port_state_committed"] is True
    assert readback["every_carrier_full_port_state_read_back"] is True
    assert readback["record_and_readback_state_digests_identical"] is True
    assert readback["physical_sector_readout"] is False


def test_packet_keeps_internal_transfer_separate_from_spatial_physics(
    receipt: dict,
) -> None:
    boundary = receipt["quotient_and_spatial_boundary"]
    assert boundary["all_six_antipodal_transfer_pairs_fail_inverse_relation"] is True
    quotient = boundary["quotient_enumeration"]
    assert quotient["set_partition_count_checked"] == 4140
    assert quotient["common_congruence_count"] == 2
    assert quotient["nontrivial_proper_common_congruence_count"] == 0
    assert quotient["antipodal_inverse_compatible_quotient_count"] == 1
    assert quotient["noncollapsed_antipodal_inverse_compatible_quotient_count"] == 0
    assert quotient["only_inverse_compatible_quotient_collapses_all_carriers"] is True
    assert boundary["internal_seam_transfer_is_spatial_translation"] is False
    assert boundary["directed_antipode_inverse_transport_receipt"] is False
    assert boundary["noncollapsed_quotient_site_map_receipt"] is False
    assert boundary["same_operator_physical_readout_receipt"] is False
    assert boundary["physical_prediction_unsealed"] is False
    missing = receipt["smallest_missing_typed_source_object"]
    assert missing["schema"] == "oph.vertex12-directed-transport-ledger.v1"
    assert (
        missing["current_source_requires_topology_or_quotient_producer_change"] is True
    )
    assert receipt["comparison_data_read"] is False


def test_independent_verifier_recomputes_the_packet_without_importing_producer(
    receipt: dict,
) -> None:
    result = independent.verify_report(receipt)
    assert result["receipt"] is True
    assert result["packet_analysis_independently_reimplemented"] is True
    assert result["source_engine_independently_reimplemented"] is False
    tree = ast.parse(Path(independent.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "oph_fpe.dynamics.vertex12_atomic_port_transfer" not in imported


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_top_level_field",
        "status",
        "matching_permutation",
        "spatial_promotion",
        "quotient_count",
        "readback_digest",
        "source_config",
        "missing_object",
        "implementation_pin",
    ),
)
def test_rehashed_claim_mutations_fail_closed(receipt: dict, mutation: str) -> None:
    changed = copy.deepcopy(receipt)
    if mutation == "extra_top_level_field":
        changed["undeclared"] = True
    elif mutation == "status":
        changed["status"] = "ATTAINED"
    elif mutation == "matching_permutation":
        changed["atomic_transfer_operator"]["port_rows"][0][
            "carrier_partner_permutation"
        ][0] = 0
    elif mutation == "spatial_promotion":
        changed["quotient_and_spatial_boundary"][
            "internal_seam_transfer_is_spatial_translation"
        ] = True
    elif mutation == "quotient_count":
        changed["quotient_and_spatial_boundary"]["quotient_enumeration"][
            "set_partition_count_checked"
        ] = 4139
    elif mutation == "readback_digest":
        changed["post_repair_source_readback"]["record_readback_pairs"][0][
            "state_sha256"
        ] = "sha256:" + "0" * 64
    elif mutation == "source_config":
        changed["source_config"]["seed"] += 1
    elif mutation == "missing_object":
        changed["smallest_missing_typed_source_object"]["required_fields"].pop()
    elif mutation == "implementation_pin":
        changed["implementation_pins"]["producer"]["sha256"] = "sha256:" + "0" * 64
    else:  # pragma: no cover
        raise AssertionError(mutation)
    changed = _rehash(changed)
    assert producer.verify_receipt(changed)["receipt"] is False
    assert independent.verify_report(changed)["receipt"] is False


def test_committed_receipt_is_byte_semantically_current(receipt: dict) -> None:
    committed = json.loads(producer.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert committed == receipt
