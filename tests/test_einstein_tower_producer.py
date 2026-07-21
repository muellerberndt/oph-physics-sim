"""Fail-closed tests for the issue-572 physical tower producer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oph_fpe.bulk.einstein_tower_producer import (
    DEFAULT_FOREIGN_CONFIG,
    DEFAULT_MAIN_CONFIG,
    produce_common_source_tower_bundle,
    verify_physical_source_binding,
)
from oph_fpe.common_source_tower import (
    ARRAY_CHANNEL_REALIZATION_DIAGNOSTIC_RECEIPT,
    COMMON_DOMAIN_SOURCE_TOWER_RECEIPT,
    DECLARED_TARGET_PATH_FIREWALL_DIAGNOSTIC_RECEIPT,
    ECHOSAHEDRAL_TO_ABSTRACT_PATCH_NET_REALIZATION_RECEIPT,
    SOURCE_TOWER_CROSS_SOURCE_SPLICE_REJECTION_RECEIPT,
    SOURCE_TOWER_NO_TARGET_PATH_RECEIPT,
    SOURCE_TOWER_PROVENANCE_GRAPH_RECEIPT,
    SOURCE_TOWER_REFINEMENT_COMMUTATION_RECEIPT,
)

_SMALL_MAIN = {"carrier_count": 32, "cycles": 6, "seed": 20260751}
_SMALL_FOREIGN = {"carrier_count": 32, "cycles": 6, "seed": 20260861}


@pytest.fixture(scope="module")
def produced(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("tower")
    return produce_common_source_tower_bundle(
        root, config=_SMALL_MAIN, foreign_config=_SMALL_FOREIGN
    )


def test_all_achievable_receipts_pass(produced: dict) -> None:
    report = produced["verifier_report"]
    assert report[COMMON_DOMAIN_SOURCE_TOWER_RECEIPT] is True
    assert report[SOURCE_TOWER_PROVENANCE_GRAPH_RECEIPT] is True
    assert report[SOURCE_TOWER_REFINEMENT_COMMUTATION_RECEIPT] is True
    assert report[SOURCE_TOWER_CROSS_SOURCE_SPLICE_REJECTION_RECEIPT] is True
    assert report[ARRAY_CHANNEL_REALIZATION_DIAGNOSTIC_RECEIPT] is True
    assert report[DECLARED_TARGET_PATH_FIREWALL_DIAGNOSTIC_RECEIPT] is True


def test_pinned_physical_receipts_stay_false(produced: dict) -> None:
    report = produced["verifier_report"]
    assert report[SOURCE_TOWER_NO_TARGET_PATH_RECEIPT] is False
    assert report[ECHOSAHEDRAL_TO_ABSTRACT_PATCH_NET_REALIZATION_RECEIPT] is False
    assert report["receipt"] is False
    blockers = set(report["blockers"])
    assert "generator_code_dependency_firewall_not_bound" in blockers
    assert "typed_echosahedral_federation_bundle_not_bound" in blockers


def test_refinement_covers_all_eight_roles(produced: dict) -> None:
    rows = produced["verifier_report"]["refinement_commutation"]["rows"]
    assert len(rows) == 8
    assert all(row["passed"] is True for row in rows)


def test_gauge_group_order_is_sixty(produced: dict) -> None:
    realization = produced["verifier_report"]["echosahedral_abstract_realization"]
    control = realization["gauge_relabeling_control"]
    assert control["generated_group_order"] == 60


def test_physical_binding_replays_exactly(produced: dict) -> None:
    binding = verify_physical_source_binding(produced["manifest_path"])
    assert binding["passed"] is True
    assert binding["capture_replay_hash_equal"] is True
    assert binding["authoritative_archive_replay_equal"] is True
    assert binding["physical_promotion_allowed"] is False


def test_binding_fails_closed_on_archive_tamper(
    tmp_path: Path,
) -> None:
    result = produce_common_source_tower_bundle(
        tmp_path, config=_SMALL_MAIN, foreign_config=_SMALL_FOREIGN
    )
    archive = tmp_path / "main_source.npz"
    data = dict(np.load(archive))
    data["entropy"] = data["entropy"] + 1e-12
    np.savez(archive, **data)
    binding = verify_physical_source_binding(result["manifest_path"])
    assert binding["passed"] is False
    assert "authoritative_archive_array_drift" in binding["blockers"]


def test_bundle_commitment_is_deterministic(
    produced: dict, tmp_path: Path
) -> None:
    again = produce_common_source_tower_bundle(
        tmp_path, config=_SMALL_MAIN, foreign_config=_SMALL_FOREIGN
    )
    first = produced["producer_receipt"]
    second = again["producer_receipt"]
    assert first["bundle_commitment"] == second["bundle_commitment"]
    assert first["main_capture_sha256"] == second["main_capture_sha256"]


def test_identical_seeds_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        produce_common_source_tower_bundle(
            tmp_path, config=_SMALL_MAIN, foreign_config=_SMALL_MAIN
        )


def test_default_configs_use_distinct_seeds() -> None:
    assert DEFAULT_MAIN_CONFIG["seed"] != DEFAULT_FOREIGN_CONFIG["seed"]
