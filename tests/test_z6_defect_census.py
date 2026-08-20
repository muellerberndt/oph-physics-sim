"""Receipts and mutation guards for the lane C6 defect census (exploratory,
non-evidential; design record ``oph_fpe/defects/DESIGN.md``)."""

from __future__ import annotations

import json
import random

import pytest

from oph_fpe.defects.z6_a5_action import (
    act_on_config,
    act_on_sector,
    action_receipt,
    declared_generator,
    rotation_group,
    rotation_group_receipt,
    sector_orbit,
)
from oph_fpe.defects.z6_carrier_defects import (
    MOD,
    CarrierDefectError,
    base_carrier_spec,
    chord_holonomies,
    conservation_receipt,
    face_curvature,
    gauge_covariance_receipt,
    gauge_move,
    mismatch_energy,
    sector_representative,
    structural_receipt,
    tree_reduce,
)
from oph_fpe.defects.z6_defect_census import (
    census_json,
    is_stable,
    neutral_escapable,
    repair,
    run_census,
    sample_stream,
    sector_label,
)
from oph_fpe.defects.z6_matter_grammar_verifier import (
    committed_row_weights,
    verify_census,
)

SPEC = base_carrier_spec()
TEST_STREAMS = (("uniform_iid", 11, 24), ("sparse_pair", 12, 12))


def _random_configs(seed: int, count: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [rng.randrange(MOD) for _ in range(SPEC.seams)] for _ in range(count)
    ]


def _random_gauges(seed: int, count: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [rng.randrange(MOD) for _ in range(SPEC.ports)] for _ in range(count)
    ]


def test_carrier_and_conservation_receipts() -> None:
    pin = structural_receipt(SPEC)
    assert pin["ports"] == 12
    assert pin["seams"] == 30
    assert pin["faces"] == 20
    assert pin["chord_count"] == 19
    receipt = conservation_receipt(SPEC)
    assert receipt["checked_pairs"] == 19 * 12
    assert receipt["tree_trivial_reads_chords"] is True
    assert receipt["curvature_classifies_sectors"]["rank_C_mod2"] == 19
    assert receipt["curvature_classifies_sectors"]["rank_C_mod3"] == 19
    assert receipt["evidential"] is False
    assert receipt["instrument_armed"] is False


def test_gauge_moves_conserve_sectors_exactly_on_samples() -> None:
    configs = _random_configs(101, 8)
    gauges = _random_gauges(102, 8)
    for config, gauge in zip(configs, gauges, strict=True):
        moved = gauge_move(SPEC, config, gauge)
        assert chord_holonomies(SPEC, moved) == chord_holonomies(SPEC, config)
        # Curvature is a sector invariant.
        assert face_curvature(SPEC, moved) == face_curvature(SPEC, config)


def test_tree_reduction_classifies() -> None:
    for config in _random_configs(103, 8):
        reduced, gauge = tree_reduce(SPEC, config)
        assert all(reduced[e] == 0 for e in SPEC.tree_seams)
        sector = chord_holonomies(SPEC, config)
        assert tuple(reduced[c] for c in SPEC.chords) == sector
        # The reduction is a gauge move: apply the inverse gauge to recover.
        assert gauge_move(SPEC, reduced, gauge) == list(config)
        # Round trip through the tree-trivial representative.
        assert chord_holonomies(
            SPEC, sector_representative(SPEC, sector)
        ) == sector


def test_conservation_receipt_fails_on_broken_cycle_mutant() -> None:
    mutant = [list(cycle) for cycle in SPEC.cycles]
    mutant[0][SPEC.tree_seams[0]] += 1
    with pytest.raises(CarrierDefectError):
        conservation_receipt(SPEC, cycles=mutant)


def test_repair_gauge_covariant_and_raw_label_mutant_fails() -> None:
    configs = _random_configs(104, 6)
    gauges = _random_gauges(105, 6)
    assert gauge_covariance_receipt(SPEC, repair, configs, gauges) is True

    def raw_label_mutant(spec, config):
        state = [x % MOD for x in config]
        trace = []
        for e in range(spec.seams):
            if state[e] != 0:
                trace.append((e, (-state[e]) % MOD))
                state[e] = 0
        return state, trace

    assert gauge_covariance_receipt(
        SPEC, raw_label_mutant, configs, gauges
    ) is False


def test_repair_descends_energy_terminates_idempotent() -> None:
    for config in _random_configs(106, 6):
        fixed, trace = repair(SPEC, config)
        assert mismatch_energy(SPEC, fixed) <= mismatch_energy(SPEC, config)
        assert mismatch_energy(SPEC, fixed) + len(trace) <= mismatch_energy(
            SPEC, config
        )  # every applied move decreased E by at least 1
        again, second_trace = repair(SPEC, fixed)
        assert again == fixed
        assert second_trace == []
        assert is_stable(SPEC, face_curvature(SPEC, fixed)) is True


def test_rotation_group_is_a5_and_action_receipts() -> None:
    rotations = rotation_group(SPEC)
    receipt = rotation_group_receipt(SPEC, rotations)
    assert receipt["order"] == 60
    assert receipt["element_order_histogram"] == {
        "1": 1, "2": 15, "3": 20, "5": 24,
    }
    generator = declared_generator(rotations)
    samples = _random_configs(107, 5)
    assert action_receipt(SPEC, rotations, generator, samples) is True
    # Mutation guard: the unsigned action drops orientation signs.
    assert action_receipt(
        SPEC, rotations, generator, samples, signed=False
    ) is False


def test_sector_action_equivariance_and_stability_invariance() -> None:
    rotations = rotation_group(SPEC)
    generator = declared_generator(rotations)
    for config in _random_configs(108, 6):
        sector = chord_holonomies(SPEC, config)
        moved = act_on_config(SPEC, generator, config)
        assert act_on_sector(SPEC, generator, sector) == chord_holonomies(
            SPEC, moved
        )
        fixed, _ = repair(SPEC, config)
        final = chord_holonomies(SPEC, fixed)
        image = act_on_sector(SPEC, generator, final)
        for probe in (final, image):
            curvature = face_curvature(
                SPEC, sector_representative(SPEC, probe)
            )
            assert is_stable(SPEC, curvature) is True
        # Stability invariance on a generically unstable sector too.
        raw = chord_holonomies(SPEC, config)
        raw_curv = face_curvature(SPEC, sector_representative(SPEC, raw))
        image_curv = face_curvature(
            SPEC,
            sector_representative(
                SPEC, act_on_sector(SPEC, generator, raw)
            ),
        )
        assert is_stable(SPEC, raw_curv) == is_stable(SPEC, image_curv)
        assert neutral_escapable(SPEC, raw_curv) == neutral_escapable(
            SPEC, image_curv
        )
        # Orbit data is constant on orbits.
        assert sector_orbit(SPEC, rotations, final) == sector_orbit(
            SPEC, rotations, image
        )


def test_census_deterministic_under_fixed_seeds() -> None:
    first = run_census(SPEC, streams=TEST_STREAMS)
    second = run_census(SPEC, streams=TEST_STREAMS)
    assert census_json(first) == census_json(second)
    assert first["exploratory"] is True
    assert first["evidential"] is False
    assert first["frozen"] is False
    assert first["instrument_armed"] is False


def test_census_initial_class_equivariance_under_generator() -> None:
    rotations = rotation_group(SPEC)
    generator = declared_generator(rotations)
    configs = sample_stream(SPEC, "uniform_iid", 13, 16)
    moved_multiset = sorted(
        chord_holonomies(SPEC, act_on_config(SPEC, generator, c))
        for c in configs
    )
    expected_multiset = sorted(
        act_on_sector(SPEC, generator, chord_holonomies(SPEC, c))
        for c in configs
    )
    assert moved_multiset == expected_multiset


def test_label_readout_is_the_declared_diagonal() -> None:
    rng = random.Random(109)
    for _ in range(20):
        sector = tuple(rng.randrange(MOD) for _ in range(len(SPEC.chords)))
        q, t, d = sector_label(sector)
        assert q == sum(sector) % MOD
        assert t == q % 3
        assert d == q % 2


def test_verifier_reports_on_real_census() -> None:
    census = run_census(SPEC, streams=TEST_STREAMS)
    report = verify_census(census)
    assert report["evidential"] is False
    assert report["descent"]["lattice_baseline"] == "1/6"
    assert report["descent"]["diagonal_readout_detected"] is True
    if census["defect_class_count"]:
        assert report["descent"]["fraction_by_class"] == "1"
    assert report["control"]["realized"] is False
    assert report["control"]["class_count"] == 0
    assert len(report["row_occupancy"]["rows"]) == 10
    assert report["labels_realized"]["distinct_descending"] <= 6
    # Committed row weights carry all six descending labels.
    weights = set(committed_row_weights())
    assert len(weights) == 6
    assert all((2 * t + 3 * d + q) % 6 == 0 for q, t, d in weights)


def test_verifier_detects_planted_label_violation() -> None:
    census = run_census(SPEC, streams=TEST_STREAMS)
    planted = json.loads(census_json(census))
    planted["classes"] = planted["classes"] + [{
        "sector": [0] * len(SPEC.chords),
        "label_qtd": [1, 0, 0],
        "multiplicity": 1,
    }]
    report = verify_census(planted)
    assert report["control"]["realized"] is True
    assert report["control"]["class_count"] == 1
    assert report["descent"]["diagonal_readout_detected"] is False
    assert report["descent"]["fraction_by_class"] != "1"
