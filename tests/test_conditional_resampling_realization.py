"""The A3 conditional-resampling realization producer.

The producer builds the exact fiber-resampling kernel from a run's
committed record classes with the realized joint frequency table as the
pinned common reference, replays it through the independent recognizer,
and drives an integer-count empirical trajectory.  These tests exercise
the producer on synthetic realized data, its fail-closed behavior on a
constant protected record, and the run-directory writer's labeled skip.
"""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest

from oph_fpe.dynamics.conditional_resampling import (
    RealizationInputs,
    produce_conditional_resampling_realization,
    realization_inputs_from_freezeout,
    write_conditional_resampling_realization,
)


def _synthetic_inputs(patches: int = 600, seed: int = 5) -> RealizationInputs:
    rng = np.random.default_rng(seed)
    record = rng.integers(0, 4, size=patches)
    companion = np.where(
        record % 2 == 0,
        rng.integers(0, 3, size=patches),
        rng.integers(1, 5, size=patches),
    )
    return RealizationInputs(
        record_classes=tuple(int(v) for v in record.tolist()),
        companion_classes=tuple(int(v) for v in companion.tolist()),
        record_label="record_signature",
        companion_label="s3_class_density",
        provenance={"source": "synthetic", "patch_count": patches},
    )


def test_realization_receipt_passes_on_nonconstant_record():
    payload = produce_conditional_resampling_realization(
        _synthetic_inputs(), seed=2, empirical_sweeps=4
    )
    assert payload["CONDITIONAL_RESAMPLING_REALIZATION_RECEIPT"] is True
    assert payload["recognizer"]["exact_table_recognition_receipt"] is True
    assert payload["exact_kernel_package"]["idempotent"] is True
    assert payload["exact_kernel_package"]["reference_stationary"] is True
    assert payload["exact_kernel_package"]["chi_squared_contracts"] is True
    assert payload["protected_record"]["unchanged_by_resampling"] is True
    assert payload["protected_record"]["class_count"] >= 2
    # The exact one-step chi-squared contraction is strict here.
    before = Fraction(payload["exact_kernel_package"]["chi_squared_before"])
    after = Fraction(payload["exact_kernel_package"]["chi_squared_after_one_step"])
    assert after < before
    # The displaced empirical start collapses onto the reference in one
    # sweep on this instance.
    rows = payload["empirical_realization"]["sweeps"]
    assert rows[0]["sweep"] == "displaced_start"
    assert payload["empirical_realization"]["one_step_collapse_measured"] is True


def test_singleton_fibers_fail_closed():
    # One companion class per record class: the kernel would be the
    # identity and the realization must refuse it.
    patches = 120
    record = tuple(int(v) for v in np.arange(patches) % 4)
    inputs = RealizationInputs(
        record_classes=record,
        companion_classes=record,
        record_label="record_signature",
        companion_label="degenerate_copy",
        provenance={"source": "synthetic"},
    )
    with pytest.raises(ValueError, match="singleton"):
        produce_conditional_resampling_realization(inputs)


def test_companion_candidate_chain_skips_constant_fields(tmp_path):
    rng = np.random.default_rng(4)
    np.savez(
        tmp_path / "freezeout_fields.npz",
        record_signature=rng.normal(size=300),
        s3_class_density=np.zeros(300),
        repair_load=np.zeros(300),
        cumulative_repair_load=rng.integers(0, 5, size=300).astype(np.float64),
    )
    inputs = realization_inputs_from_freezeout(tmp_path / "freezeout_fields.npz")
    assert inputs.companion_label == "cumulative_repair_load"


def test_constant_protected_record_fails_closed():
    patches = 100
    inputs = RealizationInputs(
        record_classes=tuple([3] * patches),
        companion_classes=tuple(int(v) for v in np.arange(patches) % 4),
        record_label="record_signature",
        companion_label="s3_class_density",
        provenance={"source": "synthetic"},
    )
    with pytest.raises(ValueError, match="nonconstant record"):
        produce_conditional_resampling_realization(inputs)


def test_freezeout_loader_and_run_writer(tmp_path):
    rng = np.random.default_rng(9)
    record = rng.integers(0, 6, size=400).astype(np.float64)
    density = rng.normal(size=400)
    np.savez(
        tmp_path / "freezeout_fields.npz",
        record_signature=record,
        s3_class_density=density,
    )
    inputs = realization_inputs_from_freezeout(tmp_path / "freezeout_fields.npz")
    assert len(inputs.record_classes) == 400

    payload = write_conditional_resampling_realization(tmp_path, seed=3)
    assert payload["CONDITIONAL_RESAMPLING_REALIZATION_RECEIPT"] is True
    written = json.loads(
        (tmp_path / "conditional_resampling_realization_receipt.json").read_text()
    )
    assert written["schema"] == payload["schema"]


def test_run_writer_labels_missing_freezeout_as_skip(tmp_path):
    payload = write_conditional_resampling_realization(tmp_path)
    assert payload["CONDITIONAL_RESAMPLING_REALIZATION_RECEIPT"] is False
    assert payload["skipped"] is True
    assert "freezeout_fields.npz" in payload["reason"]
