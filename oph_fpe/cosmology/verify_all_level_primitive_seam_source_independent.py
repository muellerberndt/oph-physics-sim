"""Independent verifier for the issue-659 primitive seam source packet."""

from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower
from oph_fpe.cosmology import verify_refined_equal_seam_source_gate_independent


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / "data/refinement/all_level_primitive_seam_source_receipt.json"
SCHEMA = "oph.registered-ladder-primitive-seam-source.v1"
STATUS = (
    "TARGET_CLEAN_REGISTERED_LADDER_PRIMITIVE_SEAM_ALPHABET_AND_UNIT_COUNTING_"
    "ATTAINED__EXPECTED_A2_RECONCILIATION_FIRST_ORDER_REFINEMENT_ONLY__INFINITE_"
    "TOWER_CANONICAL_DERIVATION_ATOMIC_RECORD_AND_FULL_SEMIGROUP_OPEN"
)

Edge = tuple[int, int]
Vector = tuple[Fraction, ...]


class VerificationError(RuntimeError):
    """Raised when a serialized claim does not independently replay."""


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    _fail(set(value) == expected, f"{label} key set drift")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _edges(mesh: Any) -> tuple[Edge, ...]:
    result = tuple(
        sorted(
            (min(int(left), int(right)), max(int(left), int(right)))
            for left, right in mesh.edges
        )
    )
    _fail(len(result) == len(set(result)), "registered alphabet has duplicates")
    return result


def _parents(fine: Any, vertex: int) -> frozenset[int]:
    support = fine.vertex_parent_support[vertex]
    result = frozenset(int(parent) for parent, _weight in support)
    _fail(
        len(result) in {1, 2} and len(result) == len(support),
        "malformed immediate parent support",
    )
    return result


def _independent_parent_indices(
    coarse: Any,
    fine: Any,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    coarse_edges = _edges(coarse)
    index = {edge: position for position, edge in enumerate(coarse_edges)}
    parent_indices: list[int] = []
    kinds: list[str] = []
    for left, right in _edges(fine):
        left_parents = _parents(fine, left)
        right_parents = _parents(fine, right)
        if (
            len(left_parents) == 1
            and len(right_parents) == 2
            and left_parents < right_parents
        ):
            parent = tuple(sorted(right_parents))
            kind = "boundary_half"
        elif (
            len(right_parents) == 1
            and len(left_parents) == 2
            and right_parents < left_parents
        ):
            parent = tuple(sorted(left_parents))
            kind = "boundary_half"
        else:
            _fail(
                len(left_parents) == 2
                and len(right_parents) == 2
                and len(left_parents & right_parents) == 1
                and len(left_parents | right_parents) == 3,
                "fine seam has no exact parent classification",
            )
            parent = tuple(sorted(left_parents ^ right_parents))
            kind = "parent_face_interior"
        _fail(parent in index, "classified parent seam is absent")
        parent_indices.append(index[parent])
        kinds.append(kind)
    return tuple(parent_indices), tuple(kinds)


def _event_id(level: int, geometry_hash: str, edge: Edge) -> str:
    return _sha(
        {
            "schema": "oph.primitive-seam-reconciliation-attempt.v1",
            "level": level,
            "geometry_hash": geometry_hash,
            "unoriented_endpoints": list(edge),
            "event_type": "balanced_endpoint_reconciliation_attempt",
            "counting_multiplicity": 1,
        }
    )


def _balanced_kernel(
    left: int,
    right: int,
) -> tuple[tuple[Fraction, tuple[int, int]], ...]:
    total = left + right
    low = total // 2
    high = total - low
    if low == high:
        return ((Fraction(1), (low, high)),)
    return (
        (Fraction(1, 2), (low, high)),
        (Fraction(1, 2), (high, low)),
    )


def _verify_a2_semantics(serialized: Mapping[str, Any]) -> None:
    _exact_keys(
        serialized,
        {
            "declared_semantic_rule",
            "integer_pair_probes",
            "probe_box",
            "endpoint_total_preserved_pathwise",
            "changes_only_selected_endpoints",
            "nearest_balanced_shell_reached_pathwise",
            "endpoint_reversal_natural",
            "expected_endpoint_agreement_exact",
            "expected_rational_action",
            "odd_total_pathwise_exact_agreement",
            "canonical_a2_pathwise_agreement_discharged",
            "odd_total_pathwise_disagreement_witness",
            "rule_selected_by_canonical_a2_alone",
            "issue_628_atomic_record_write_identification",
            "issue_628_relation",
            "declared_rule_covers_every_serialized_event",
            "complete_alphabet_coverage_exact",
            "expected_balancing_is_diagnostic_not_canonical_a2_agreement",
            "a2_does_not_select_the_rule_or_event_alphabet",
        },
        "expectation-level balancing diagnostic",
    )
    witness = serialized.get("odd_total_pathwise_disagreement_witness")
    _fail(isinstance(witness, Mapping), "odd-total witness missing")
    _exact_keys(witness, {"input", "outcomes"}, "odd-total witness")
    outcomes = witness.get("outcomes")
    _fail(isinstance(outcomes, list) and len(outcomes) == 2, "odd-total outcomes")
    for outcome in outcomes:
        _fail(isinstance(outcome, Mapping), "odd-total outcome malformed")
        _exact_keys(outcome, {"probability", "value"}, "odd-total outcome")
    probes = 0
    for left in range(-8, 9):
        for right in range(-8, 9):
            kernel = _balanced_kernel(left, right)
            _fail(
                sum((weight for weight, _outcome in kernel), Fraction()) == 1,
                "kernel normalization failed",
            )
            _fail(
                all(sum(outcome) == left + right for _weight, outcome in kernel),
                "kernel conservation failed",
            )
            _fail(
                all(
                    abs(outcome[0] - outcome[1]) <= 1
                    for _weight, outcome in kernel
                ),
                "nearest-shell property failed",
            )
            target = Fraction(left + right, 2)
            expected = tuple(
                sum(
                    (weight * outcome[index] for weight, outcome in kernel),
                    Fraction(),
                )
                for index in (0, 1)
            )
            _fail(expected == (target, target), "expected agreement failed")
            probes += 1
    _fail(serialized.get("integer_pair_probes") == probes, "A2 probe count drift")
    for key in (
        "endpoint_total_preserved_pathwise",
        "changes_only_selected_endpoints",
        "nearest_balanced_shell_reached_pathwise",
        "endpoint_reversal_natural",
        "expected_endpoint_agreement_exact",
        "declared_rule_covers_every_serialized_event",
        "complete_alphabet_coverage_exact",
        "expected_balancing_is_diagnostic_not_canonical_a2_agreement",
        "a2_does_not_select_the_rule_or_event_alphabet",
    ):
        _fail(serialized.get(key) is True, f"A2 attained field drift: {key}")
    for key in (
        "odd_total_pathwise_exact_agreement",
        "canonical_a2_pathwise_agreement_discharged",
        "rule_selected_by_canonical_a2_alone",
        "issue_628_atomic_record_write_identification",
    ):
        _fail(serialized.get(key) is False, f"A2 boundary field drift: {key}")


def _j_row(fine: Any, vertex: int) -> dict[int, Fraction]:
    support = fine.vertex_parent_support[vertex]
    return {
        int(parent): Fraction(1, len(support))
        for parent, _weight in support
    }


def _add_scaled(
    target: dict[int, Fraction],
    source: Mapping[int, Fraction],
    scale: Fraction,
) -> None:
    for key, value in source.items():
        target[key] += scale * value
        if target[key] == 0:
            del target[key]


def _verify_child_aggregate(
    coarse: Any,
    fine: Any,
    parent_indices: Sequence[int],
) -> None:
    coarse_edges = _edges(coarse)
    child_groups: list[list[Edge]] = [[] for _ in coarse_edges]
    for edge, parent in zip(_edges(fine), parent_indices, strict=True):
        child_groups[parent].append(edge)
    for index, (coarse_left, coarse_right) in enumerate(coarse_edges):
        actual: dict[int, dict[int, Fraction]] = {}
        for fine_left, fine_right in child_groups[index]:
            for endpoint, other in (
                (fine_left, fine_right),
                (fine_right, fine_left),
            ):
                if endpoint >= coarse.vertex_count:
                    continue
                row: defaultdict[int, Fraction] = defaultdict(Fraction)
                _add_scaled(row, _j_row(fine, other), Fraction(1, 2))
                _add_scaled(row, _j_row(fine, endpoint), Fraction(-1, 2))
                actual[endpoint] = dict(row)
        expected = {
            coarse_left: {
                coarse_left: Fraction(-1, 4),
                coarse_right: Fraction(1, 4),
            },
            coarse_right: {
                coarse_left: Fraction(1, 4),
                coarse_right: Fraction(-1, 4),
            },
        }
        _fail(actual == expected, "per-parent child aggregate identity failed")


def _interpolate(values: Sequence[Fraction], fine: Any) -> Vector:
    return tuple(
        sum((values[int(parent)] for parent, _weight in support), Fraction())
        / len(support)
        for support in fine.vertex_parent_support
    )


def _mean_delta(values: Sequence[Fraction], edges: Sequence[Edge]) -> Vector:
    output = [Fraction()] * len(values)
    scale = Fraction(1, 2 * len(edges))
    for left, right in edges:
        delta = (values[right] - values[left]) * scale
        output[left] += delta
        output[right] -= delta
    return tuple(output)


def _first_nonzero(values: Sequence[Fraction]) -> dict[str, Any] | None:
    for index, value in enumerate(values):
        if value:
            return {"index": index, "residual": str(value)}
    return None


def _verify_refinement_row(
    serialized: Mapping[str, Any],
    coarse: Any,
    fine: Any,
) -> None:
    _exact_keys(
        serialized,
        {
            "coarse_level",
            "fine_level",
            "coarse_geometry_hash",
            "fine_geometry_hash",
            "coarse_seam_count",
            "fine_seam_count",
            "fine_to_coarse_count_ratio",
            "fine_parent_seam_indices",
            "child_kind_counts",
            "children_per_parent_seam",
            "boundary_half_children_per_parent",
            "parent_face_interior_children_per_parent",
            "raw_unit_count_pushforward_multiplier",
            "normalized_unit_count_pushforward_exact",
            "normalized_pushforward_identity",
            "child_aggregate_inherited_readback_identity",
            "child_aggregate_identity_exact_for_every_parent",
            "average_child_inherited_readback_identity",
            "level_mean_generator_identity",
            "level_mean_generator_identity_exact",
            "strong_intertwiner",
            "strong_intertwiner_witness_on_coarse_basis_zero",
            "second_order_inherited_identity",
            "second_order_witness_on_coarse_basis_zero",
        },
        "refinement row",
    )
    parent_indices, kinds = _independent_parent_indices(coarse, fine)
    for label in (
        "strong_intertwiner_witness_on_coarse_basis_zero",
        "second_order_witness_on_coarse_basis_zero",
    ):
        witness = serialized.get(label)
        _fail(isinstance(witness, Mapping), f"{label} missing")
        _exact_keys(witness, {"index", "residual"}, label)
    _fail(
        serialized.get("fine_parent_seam_indices") == list(parent_indices),
        "serialized seam lineage drift",
    )
    multiplicities = Counter(parent_indices)
    _fail(
        set(multiplicities) == set(range(coarse.edge_count))
        and set(multiplicities.values()) == {4},
        "lineage is not complete four-to-one",
    )
    _fail(
        serialized.get("child_kind_counts") == dict(sorted(Counter(kinds).items())),
        "child-kind census drift",
    )
    _verify_child_aggregate(coarse, fine, parent_indices)

    coarse_edges = _edges(coarse)
    fine_edges = _edges(fine)
    probe = [Fraction()] * coarse.vertex_count
    probe[0] = Fraction(1)
    interpolated = _interpolate(probe, fine)
    fine_first = _mean_delta(interpolated, fine_edges)
    coarse_first = _mean_delta(probe, coarse_edges)
    _fail(
        fine_first[: coarse.vertex_count]
        == tuple(Fraction(1, 8) * value for value in coarse_first),
        "first-order inherited identity failed",
    )
    strong = tuple(
        left - Fraction(1, 8) * right
        for left, right in zip(
            fine_first,
            _interpolate(coarse_first, fine),
            strict=True,
        )
    )
    _fail(
        serialized.get("strong_intertwiner_witness_on_coarse_basis_zero")
        == _first_nonzero(strong),
        "strong-intertwiner witness drift",
    )
    fine_second = _mean_delta(fine_first, fine_edges)
    coarse_second = _mean_delta(coarse_first, coarse_edges)
    second = tuple(
        fine_second[index] - Fraction(1, 64) * coarse_second[index]
        for index in range(coarse.vertex_count)
    )
    _fail(
        serialized.get("second_order_witness_on_coarse_basis_zero")
        == _first_nonzero(second),
        "second-order witness drift",
    )
    for key in (
        "normalized_unit_count_pushforward_exact",
        "child_aggregate_identity_exact_for_every_parent",
        "level_mean_generator_identity_exact",
    ):
        _fail(serialized.get(key) is True, f"refinement attained field drift: {key}")
    for key in ("strong_intertwiner", "second_order_inherited_identity"):
        _fail(serialized.get(key) is False, f"refinement boundary drift: {key}")


def verify_receipt(receipt_path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _exact_keys(
        receipt,
        {
            "schema",
            "issue",
            "status",
            "source_scope",
            "parent_bindings",
            "primitive_event_declaration",
            "level_alphabets",
            "unit_counting_certificate",
            "a2_reconciliation",
            "refinement_certificate",
            "selection_decision",
            "source_pins",
            "reopen_or_advance_condition",
            "claim_boundary",
            "payload_sha256",
        },
        "top-level receipt",
    )
    _fail(receipt.get("schema") == SCHEMA, "schema mismatch")
    _fail(receipt.get("status") == STATUS, "status mismatch")
    payload = dict(receipt)
    stored_hash = payload.pop("payload_sha256", None)
    _fail(stored_hash == _sha(payload), "payload hash mismatch")

    source_pins = receipt.get("source_pins", [])
    _fail(isinstance(source_pins, list) and len(source_pins) == 7, "source pin set")
    for pin in source_pins:
        _exact_keys(pin, {"path", "bytes", "sha256"}, "source pin")
        path = ROOT / str(pin["path"])
        raw = path.read_bytes()
        _fail(len(raw) == int(pin["bytes"]), f"source size drift: {path}")
        _fail(
            "sha256:" + hashlib.sha256(raw).hexdigest() == pin["sha256"],
            f"source hash drift: {path}",
        )

    parent = receipt["parent_bindings"]
    _exact_keys(
        parent,
        {
            "source_gate_receipt",
            "source_gate_payload_sha256",
            "bounded_repair_receipt",
            "bounded_repair_certificate_payload_sha256",
            "record_counting_projection",
            "record_counting_upstream_file_sha256",
            "record_counting_upstream_manifest_sha256",
        },
        "parent bindings",
    )
    source_gate_path = ROOT / str(parent["source_gate_receipt"])
    parent_verification = (
        verify_refined_equal_seam_source_gate_independent.verify_receipt(
            source_gate_path
        )
    )
    _fail(parent_verification.get("status") == "PASS", "source gate failed replay")
    source_gate = json.loads(source_gate_path.read_text(encoding="utf-8"))
    _fail(
        source_gate.get("payload_sha256")
        == parent["source_gate_payload_sha256"],
        "source-gate parent binding drift",
    )
    bounded = json.loads(
        (ROOT / str(parent["bounded_repair_receipt"])).read_text(encoding="utf-8")
    )
    _fail(
        bounded.get("certificate_payload_sha256")
        == parent["bounded_repair_certificate_payload_sha256"],
        "bounded-repair parent binding drift",
    )
    projection = json.loads(
        (ROOT / str(parent["record_counting_projection"])).read_text(
            encoding="utf-8"
        )
    )
    _fail(
        projection.get("upstream_file_sha256")
        == parent["record_counting_upstream_file_sha256"],
        "record-counting source file binding drift",
    )
    _fail(
        projection.get("upstream_manifest_sha256")
        == parent["record_counting_upstream_manifest_sha256"],
        "record-counting source manifest binding drift",
    )

    level_rows = receipt.get("level_alphabets", [])
    _fail(
        [row.get("level") for row in level_rows] == list(range(6)),
        "canonical level coverage mismatch",
    )
    tower = build_geodesic_icosahedral_tower(5)
    orbit_rows = source_gate["edge_orbit_rows"]
    checked_events = 0
    for row, mesh, orbit_row in zip(level_rows, tower.levels, orbit_rows, strict=True):
        _exact_keys(
            row,
            {
                "level",
                "frequency",
                "geometry_hash",
                "vertex_count",
                "seam_count",
                "complete_registered_unoriented_seams",
                "primitive_event_type",
                "one_event_per_registered_seam",
                "event_id_sequence_sha256",
                "a5_edge_orbit_count_from_residual_gated_parent",
                "declared_multiplicity_by_a5_orbit",
                "all_orbit_multiplicities_equal_to_one",
                "normalized_event_probability",
                "target_or_comparison_fields_read",
            },
            "level alphabet",
        )
        _fail(
            row.get("target_or_comparison_fields_read") == [],
            "level target boundary drift",
        )
        edges = _edges(mesh)
        _fail(
            row.get("complete_registered_unoriented_seams")
            == [list(edge) for edge in edges],
            "complete seam alphabet drift",
        )
        _fail(row.get("seam_count") == len(edges), "seam count drift")
        _fail(row.get("geometry_hash") == mesh.geometry_hash, "geometry hash drift")
        event_ids = [_event_id(mesh.level, mesh.geometry_hash, edge) for edge in edges]
        _fail(
            row.get("event_id_sequence_sha256") == _sha(event_ids),
            "event identity sequence drift",
        )
        orbit_count = int(orbit_row["edge_orbit_count"])
        _fail(
            row.get("a5_edge_orbit_count_from_residual_gated_parent")
            == orbit_count,
            "A5 orbit count drift",
        )
        _fail(
            row.get("declared_multiplicity_by_a5_orbit") == [1] * orbit_count,
            "unit counting across orbit classes failed",
        )
        _fail(
            row.get("one_event_per_registered_seam") is True
            and row.get("all_orbit_multiplicities_equal_to_one") is True,
            "primitive event declaration drift",
        )
        checked_events += len(edges)

    _verify_a2_semantics(receipt["a2_reconciliation"])
    refinement = receipt["refinement_certificate"]
    _exact_keys(
        refinement,
        {
            "rows",
            "complete_event_lineage_exact",
            "normalized_unit_counting_refinement_natural",
            "raw_unit_counting_refinement_natural_without_rescaling",
            "raw_count_scale_factor_per_adjacent_level",
            "expected_reconciliation_first_order_inherited_readback_exact",
            "strong_event_action_refinement_natural",
            "repair_semigroup_refinement_natural",
            "continuum_limit_certified",
        },
        "refinement certificate",
    )
    rows = refinement["rows"]
    _fail(len(rows) == 5, "refinement row count drift")
    for row, coarse, fine in zip(
        rows, tower.levels[:-1], tower.levels[1:], strict=True
    ):
        _verify_refinement_row(row, coarse, fine)
    for key in (
        "complete_event_lineage_exact",
        "normalized_unit_counting_refinement_natural",
        "expected_reconciliation_first_order_inherited_readback_exact",
    ):
        _fail(refinement.get(key) is True, f"refinement summary drift: {key}")
    for key in (
        "raw_unit_counting_refinement_natural_without_rescaling",
        "strong_event_action_refinement_natural",
        "repair_semigroup_refinement_natural",
        "continuum_limit_certified",
    ):
        _fail(refinement.get(key) is False, f"refinement boundary drift: {key}")

    counting = receipt["unit_counting_certificate"]
    _exact_keys(
        counting,
        {
            "complete_alphabet_enumerated",
            "duplicates",
            "omitted_registered_seams",
            "hidden_event_species_declared",
            "event_multiplicity_rule",
            "exact_unit_counting_across_a5_orbit_classes_in_declared_source",
            "normalized_reference_is_uniform_on_each_complete_level",
            "unit_counting_forced_by_a5_alone",
            "unit_counting_derived_from_canonical_a1_a3",
            "unit_counting_is_additional_source_branch_declaration",
        },
        "unit counting certificate",
    )
    for key in (
        "complete_alphabet_enumerated",
        "exact_unit_counting_across_a5_orbit_classes_in_declared_source",
        "normalized_reference_is_uniform_on_each_complete_level",
        "unit_counting_is_additional_source_branch_declaration",
    ):
        _fail(counting.get(key) is True, f"counting field drift: {key}")
    for key in (
        "unit_counting_forced_by_a5_alone",
        "unit_counting_derived_from_canonical_a1_a3",
    ):
        _fail(counting.get(key) is False, f"counting boundary drift: {key}")

    decision = receipt["selection_decision"]
    _exact_keys(
        decision,
        {
            "target_clean_registered_ladder_event_emitter_instantiated",
            "registered_ladder_complete_primitive_attempt_alphabet_source_emitted",
            "registered_ladder_exact_unit_counting_source_emitted_on_declared_branch",
            "infinite_tower_complete_primitive_attempt_alphabet_source_emitted",
            "infinite_tower_exact_unit_counting_source_emitted_on_declared_branch",
            "canonical_a1_a3_force_the_emitter",
            "issue_628_atomic_record_bridge_discharged",
            "pathwise_exact_a2_agreement_for_odd_integer_totals",
            "first_order_refinement_readback_discharged",
            "full_refinement_commuting_diagram_discharged",
            "physical_repair_law_selected",
            "physical_covariance_selected",
            "continuum_equal_seam_operator_selected",
            "physical_prediction",
            "promotion_allowed",
        },
        "selection decision",
    )
    for key in (
        "target_clean_registered_ladder_event_emitter_instantiated",
        "registered_ladder_complete_primitive_attempt_alphabet_source_emitted",
        "registered_ladder_exact_unit_counting_source_emitted_on_declared_branch",
        "first_order_refinement_readback_discharged",
    ):
        _fail(decision.get(key) is True, f"attained decision drift: {key}")
    for key in (
        "infinite_tower_complete_primitive_attempt_alphabet_source_emitted",
        "infinite_tower_exact_unit_counting_source_emitted_on_declared_branch",
        "canonical_a1_a3_force_the_emitter",
        "issue_628_atomic_record_bridge_discharged",
        "pathwise_exact_a2_agreement_for_odd_integer_totals",
        "full_refinement_commuting_diagram_discharged",
        "physical_repair_law_selected",
        "physical_covariance_selected",
        "continuum_equal_seam_operator_selected",
        "physical_prediction",
        "promotion_allowed",
    ):
        _fail(decision.get(key) is False, f"forbidden decision promotion: {key}")
    scope = receipt["source_scope"]
    _exact_keys(
        scope,
        {
            "registered_geometry_levels",
            "source_emitter_kind",
            "external_comparison_data_used",
            "particle_data_used",
            "sky_data_used",
            "target_values_used",
            "canonical_a1_a3_derivation_of_emitter",
        },
        "source scope",
    )
    _fail(
        not any(
            scope.get(key) is True
            for key in (
                "external_comparison_data_used",
                "particle_data_used",
                "sky_data_used",
                "target_values_used",
                "canonical_a1_a3_derivation_of_emitter",
            )
        ),
        "target or canonical-derivation boundary drift",
    )
    event_declaration = receipt["primitive_event_declaration"]
    _exact_keys(
        event_declaration,
        {
            "event_type",
            "event_identity_fields",
            "one_serialized_row_is_one_primitive_attempt_symbol",
            "same_event_type_at_every_registered_level",
            "same_declared_counting_multiplicity_at_every_registered_level",
            "completed_attempt_is_issue_628_atomic_record_event",
            "primitive_here_means_irreducible_in_declared_attempt_alphabet",
            "primitive_here_does_not_mean_proved_atomic_record_write",
        },
        "primitive event declaration",
    )
    return {
        "status": "PASS",
        "receipt": True,
        "checked_levels": len(level_rows),
        "checked_primitive_events": checked_events,
        "checked_refinement_rows": len(rows),
        "producer_imported": False,
        "registered_tower_builder_shared": True,
        "source_engine_independently_reimplemented": False,
        "event_identity_lineage_and_refinement_algebra_independently_reimplemented": True,
        "claim_boundary": (
            "The verifier shares the registered tower builder and independently "
            "reimplements the event identities, four-to-one lineage, unit-count "
            "pushforward, balanced kernel, and exact first- and second-order "
            "refinement algebra. It does not independently reimplement the geometry "
            "source, derive the declared emitter from A1--A3, or identify it with "
            "atomic records or physical dynamics."
        ),
    }


if __name__ == "__main__":
    result = verify_receipt()
    print(json.dumps(result, indent=2, sort_keys=True))
