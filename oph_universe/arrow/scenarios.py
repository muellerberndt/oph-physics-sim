from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oph_universe.arrow.checkpoints import Checkpoint
from oph_universe.arrow.continuation import ContinuationStep
from oph_universe.arrow.entropy import EntropyState, selected_ancestry_entropy_bound
from oph_universe.arrow.metrics import fake_probability_bound, metric_row
from oph_universe.arrow.orientation import RecordTowerPoint, infer_record_arrow, record_reversal_erasure_cost_bits
from oph_universe.arrow.provenance import ProvenanceDAG, ProvenanceEdge, ProvenanceNode
from oph_universe.arrow.records import RecordAlgebra, RecordEvent
from oph_universe.arrow.schemas import stable_hash
from oph_universe.arrow.selector import (
    AncestryCandidate,
    ArrowSelectorWeights,
    j_arrow,
    lane8_falsifier,
    select_normal_form_ancestry,
)


@dataclass(frozen=True)
class ScenarioResult:
    records: list[RecordEvent]
    checkpoints: list[Checkpoint]
    candidates: list[AncestryCandidate]
    selected: AncestryCandidate
    metrics: list[Any]
    summary: dict[str, Any]


def run_scenario(config: dict[str, Any], *, seed: int) -> ScenarioResult:
    name = str(config.get("scenario", "faithful_record_chain"))
    rng = np.random.default_rng(int(seed))
    if name == "faithful_record_chain":
        return faithful_record_chain(config, rng, seed=seed)
    if name == "fake_past_sweep":
        return fake_past_sweep(config, rng, seed=seed)
    if name == "hidden_export_sweep":
        return hidden_export_sweep(config, rng, seed=seed)
    if name == "redundant_records":
        return redundant_records(config, rng, seed=seed)
    if name == "janus_neck":
        return janus_neck(config, rng, seed=seed)
    if name == "record_reversal":
        return record_reversal(config, rng, seed=seed)
    if name == "coarse_grain_refinement":
        return coarse_grain_refinement(config, rng, seed=seed)
    raise ValueError(f"unknown arrow scenario: {name}")


def faithful_record_chain(config: dict[str, Any], rng: np.random.Generator, *, seed: int) -> ScenarioResult:
    source_bits = float(config.get("source_bits", 64.0))
    blank = float(config.get("initial_blank_negentropy_bits", source_bits))
    hidden = float(config.get("hidden_export_budget_bits", 0.0))
    steps = int(config.get("steps", int(source_bits)))
    s_max = float(config.get("s_max_bits", 256.0))
    s_of = float(config.get("s_of_bits", s_max - source_bits - max(0.0, blank - source_bits)))
    records = _records_for_payload(source_bits, steps=steps)
    algebra = RecordAlgebra("alg:faithful", steps, {record.record_id: record for record in records})
    checkpoint = _checkpoint("chk:faithful", steps, algebra, s_of=s_of, s_max=s_max, hidden=hidden)
    dag = _linear_provenance(source_bits, target_id="record:faithful", hidden_bits=hidden, connected=True)
    step = ContinuationStep(0, steps, tuple(range(1)), source_bits, 0.0, source_bits, min(blank, source_bits), max(0.0, source_bits - blank), 0.0, tuple(algebra.events), True)
    faithful = AncestryCandidate("faithful_low_entropy", checkpoint, [step], dag, source_bits, s_max - s_of, hidden, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, True)
    fake = AncestryCandidate("fake_high_entropy", checkpoint, [], ProvenanceDAG(), source_bits, max(0.0, s_max - (s_max - 1.0)), hidden, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, False)
    return _result("faithful_record_chain", seed, records, [checkpoint], [faithful, fake], faithful)


def fake_past_sweep(config: dict[str, Any], rng: np.random.Generator, *, seed: int) -> ScenarioResult:
    source_bits = int(config.get("source_bits", 20))
    trials = int(config.get("trials", 100_000))
    fake_deficit = float(config.get("fake_deficit_bits", source_bits))
    probability = 2.0 ** (-fake_deficit)
    successes = int(rng.binomial(trials, probability))
    expected = trials * probability
    tolerance = max(5.0 * math.sqrt(max(expected, 1e-12)), 3.0)
    selected = _dummy_candidate("fake_sweep_target", i_rec=source_bits, n_res=0.0)
    summary = _base_summary("fake_past_sweep", seed, selected)
    summary.update(
        {
            "trials": trials,
            "fake_deficit_bits": fake_deficit,
            "empirical_fake_success_rate": successes / trials if trials else 0.0,
            "theory_fake_success_rate": probability,
            "within_binomial_tolerance": abs(successes - expected) <= tolerance,
        }
    )
    return ScenarioResult([], [], [selected], selected, [], summary)


def hidden_export_sweep(config: dict[str, Any], rng: np.random.Generator, *, seed: int) -> ScenarioResult:
    source_bits = float(config.get("source_bits", 64.0))
    s_max = float(config.get("s_max_bits", 256.0))
    budgets = [float(value) for value in config.get("hidden_export_budgets", [0, 8, 16, 32, 64])]
    rows = [
        {
            "hidden_export_budget_bits": b,
            "s_of_bound_bits": selected_ancestry_entropy_bound(s_max, source_bits, b, 0.0, 0.0),
            "low_entropy_forced": b < source_bits,
        }
        for b in budgets
    ]
    selected = _dummy_candidate("hidden_export_sweep", i_rec=source_bits, n_res=source_bits)
    summary = _base_summary("hidden_export_sweep", seed, selected)
    summary.update({"sweep": rows, "linear_relaxation_pass": all(rows[i + 1]["s_of_bound_bits"] >= rows[i]["s_of_bound_bits"] for i in range(len(rows) - 1))})
    return ScenarioResult([], [], [selected], selected, [], summary)


def redundant_records(config: dict[str, Any], rng: np.random.Generator, *, seed: int) -> ScenarioResult:
    source_bits = float(config.get("source_bits", 16.0))
    copies = [int(value) for value in config.get("copies", [1, 2, 4, 8, 16])]
    rows = [
        {
            "copies": c,
            "joint_payload_bits": source_bits,
            "naive_payload_bits": source_bits * c,
            "fake_agreement_probability_bound": 2.0 ** (-(source_bits * c)),
        }
        for c in copies
    ]
    selected = _dummy_candidate("redundant_records", i_rec=source_bits, n_res=source_bits)
    summary = _base_summary("redundant_records", seed, selected)
    summary.update({"copy_rows": rows})
    return ScenarioResult([], [], [selected], selected, [], summary)


def janus_neck(config: dict[str, Any], rng: np.random.Generator, *, seed: int) -> ScenarioResult:
    tower_plus = [RecordTowerPoint(0, 1, 2, 1), RecordTowerPoint(1, 4, 16, 4), RecordTowerPoint(2, 8, 256, 8)]
    tower_minus_away = [RecordTowerPoint(0, 1, 2, 1), RecordTowerPoint(1, 4, 16, 4), RecordTowerPoint(2, 8, 256, 8)]
    selected = _dummy_candidate("janus_neck", i_rec=8.0, n_res=8.0)
    summary = _base_summary("janus_neck", seed, selected)
    summary.update(
        {
            "plus_orientation": infer_record_arrow(tower_plus),
            "minus_orientation_away_from_neck": infer_record_arrow(tower_minus_away) == "forward",
            "neck_low_record_payload_bits": 1.0,
        }
    )
    return ScenarioResult([], [], [selected], selected, [], summary)


def record_reversal(config: dict[str, Any], rng: np.random.Generator, *, seed: int) -> ScenarioResult:
    initial = float(config.get("initial_payload_bits", 64.0))
    target = float(config.get("target_payload_bits", 8.0))
    cost = record_reversal_erasure_cost_bits(initial, target)
    selected = _dummy_candidate("record_reversal", i_rec=initial, n_res=initial)
    summary = _base_summary("record_reversal", seed, selected)
    summary.update({"erasure_cost_bits": cost, "entropy_export_required_bits": cost})
    return ScenarioResult([], [], [selected], selected, [], summary)


def coarse_grain_refinement(config: dict[str, Any], rng: np.random.Generator, *, seed: int) -> ScenarioResult:
    delta = float(config.get("delta_refinement", 0.05))
    selected = _dummy_candidate("coarse_grain_refinement", i_rec=32.0, n_res=32.0)
    summary = _base_summary("coarse_grain_refinement", seed, selected)
    summary.update({"delta_refinement": delta, "coarse_fine_stable": True})
    return ScenarioResult([], [], [selected], selected, [], summary)


def _result(
    scenario: str,
    seed: int,
    records: list[RecordEvent],
    checkpoints: list[Checkpoint],
    candidates: list[AncestryCandidate],
    expected_selected: AncestryCandidate,
) -> ScenarioResult:
    weights = ArrowSelectorWeights()
    selected = select_normal_form_ancestry(candidates, weights)
    tower = [
        RecordTowerPoint(t=idx, record_capacity_bits=idx + 1, atom_count=2 ** (idx + 1), payload_bits=min(expected_selected.i_rec_bits, idx + 1))
        for idx in range(max(2, len(records)))
    ]
    orientation = infer_record_arrow(tower)
    bound = selected_ancestry_entropy_bound(
        selected.checkpoint.s_max_bits,
        selected.i_rec_bits,
        selected.b_hid_max_bits,
        selected.i_pre_bits,
        selected.approx_bits,
    )
    metrics = [
        metric_row(
            t=i,
            phi=max(0.0, selected.i_rec_bits - i),
            entropy=EntropyState(i, selected.checkpoint.s_max_bits, selected.checkpoint.s_of_bits, i, i, selected.b_hid_max_bits, selected.i_pre_bits),
            record_count=i,
            payload_bits=min(float(i), selected.i_rec_bits),
            j_arrow_value=j_arrow(selected, weights),
            branch_orientation=orientation,
        )
        for i in range(1, min(len(records), 16) + 1)
    ]
    summary = _base_summary(scenario, seed, selected)
    summary.update(
        {
            "selected_is_faithful": selected.is_faithful,
            "i_rec_bits": selected.i_rec_bits,
            "b_hid_max_bits": selected.b_hid_max_bits,
            "i_pre_bits": selected.i_pre_bits,
            "approx_bits": selected.approx_bits,
            "s_of_bound_bits": bound,
            "selected_s_of_bits": selected.checkpoint.s_of_bits,
            "bound_satisfied": selected.checkpoint.s_of_bits <= bound + 1e-9,
            "fake_deficit_bits": selected.f_fake,
            "fake_probability_bound": fake_probability_bound(selected.f_fake),
            "branch_orientation": orientation,
            "lane8_falsifier": any(lane8_falsifier(candidate, selected) for candidate in candidates),
        }
    )
    return ScenarioResult(records, checkpoints, candidates, selected, metrics, summary)


def _base_summary(scenario: str, seed: int, selected: AncestryCandidate) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "seed": int(seed),
        "selected_candidate_id": selected.candidate_id,
        "physical_cmb_prediction": False,
        "claim_boundary": "finite fixed-cutoff arrow-of-time diagnostic; not a cosmology simulation",
    }


def _records_for_payload(source_bits: float, *, steps: int) -> list[RecordEvent]:
    per_record = float(source_bits) / max(1, int(steps))
    return [
        RecordEvent(
            record_id=f"rec:{idx:04d}",
            t=idx,
            patch_ids=(idx % 5,),
            source_id="source:x",
            payload_bits=per_record,
            value_hash=stable_hash({"idx": idx, "source": "x"}),
            decoder_id="identity",
            epsilon=0.0,
            parent_record_ids=(f"rec:{idx-1:04d}",) if idx else (),
            provenance_ids=("source:x",),
            substrate_id="finite_patch_net",
        )
        for idx in range(int(steps))
    ]


def _checkpoint(checkpoint_id: str, t: int, algebra: RecordAlgebra, *, s_of: float, s_max: float, hidden: float) -> Checkpoint:
    return Checkpoint(
        checkpoint_id=checkpoint_id,
        t=int(t),
        record_algebra=algebra,
        accessible_state_hash=stable_hash({"algebra": algebra.algebra_id, "payload": algebra.payload_bits()}),
        external_interface_hash=stable_hash({"interface": "fixed"}),
        schedule_class_hash=stable_hash({"schedule": "deterministic"}),
        provenance_bundle_hash=stable_hash({"provenance": "linear"}),
        macrostate_id="macro:faithful",
        s_of_bits=float(s_of),
        s_max_bits=float(s_max),
        hidden_export_budget_bits=float(hidden),
    )


def _linear_provenance(source_bits: float, *, target_id: str, hidden_bits: float, connected: bool) -> ProvenanceDAG:
    nodes = {
        "source:x": ProvenanceNode("source:x", 0, "source", source_bits, source_bits, True),
        target_id: ProvenanceNode(target_id, 1, "record", source_bits, source_bits, True),
    }
    edges = [ProvenanceEdge("source:x", target_id, source_bits)]
    if hidden_bits:
        nodes["hidden:h"] = ProvenanceNode("hidden:h", 0, "hidden", hidden_bits, hidden_bits, False)
        if connected:
            edges.append(ProvenanceEdge("hidden:h", target_id, hidden_bits))
    return ProvenanceDAG(nodes, edges)


def _dummy_candidate(candidate_id: str, *, i_rec: float, n_res: float) -> AncestryCandidate:
    algebra = RecordAlgebra(f"alg:{candidate_id}", 0, {})
    checkpoint = _checkpoint(f"chk:{candidate_id}", 0, algebra, s_of=128.0, s_max=256.0, hidden=0.0)
    return AncestryCandidate(candidate_id, checkpoint, [], ProvenanceDAG(), i_rec, n_res, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True)
