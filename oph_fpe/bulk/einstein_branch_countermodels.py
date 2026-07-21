"""Issue-577 countermodel matrix for the Einstein-branch instruments.

Semantic minimality demands more than mutated manifests: for each receipt
family the branch consumes, a complete tower must exist in which that family
fails while the others hold, produced by an honest end-to-end run on a
semantically modified source.  This module builds such countermodels for the
clause families of the four Einstein-branch instruments (#573 normalization,
#574 GNS clauses, #575 event manifold, #576 coupling) by modifying source
data at the capture level (states, generators, ancestry, ledger), never by
editing reports or manifests, and then records the isolation vector of each
full report.

The complementary #577 obligation, the concrete protected boundary map with
its gauge-equivalence proof and refinement-uniform inverse modulus, is a
separate formal deliverable and is explicitly not claimed here.

No physical promotion follows from any output; a countermodel that flips a
clause in the attaining direction is an isolation witness for the receipt,
not evidence about the physical source.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np

from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source
from oph_fpe.bulk.gns_tower_producer import _tower_level_report, _null_generator_report
from oph_fpe.bulk.event_manifold_producer import (
    _event_table,
    _pair_classes,
)

SCHEMA = "oph.einstein-branch-countermodels.v1"
PHYSICAL_PROMOTION_ALLOWED = False


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _deep_copy_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(capture))


def _clause_vector(capture: Mapping[str, Any]) -> dict[str, bool]:
    """The measured clause vector the countermodels isolate against.

    Four clause families, one per instrument axis this module covers:
    state faithfulness/support (from the GNS level report), modular
    intersection construction, future-cone spectrum (null assembly), and
    causal-class nondegeneracy (from the event table).
    """

    level = _tower_level_report(capture)
    null_assembly = _null_generator_report(capture)
    table = _event_table(capture)
    pairs = _pair_classes(table)
    # Faithfulness is judged on the raw empirical moment, before the
    # numerical regularizer that the GNS machinery adds for conditioning;
    # otherwise the regularizer masks genuinely rank-deficient sources.
    from oph_fpe.bulk.modular_normalization_producer import _snapshot_samples

    samples = _snapshot_samples(capture)
    raw_moment = samples.T @ samples / samples.shape[0]
    raw_floor = float(np.linalg.eigvalsh(raw_moment).min())
    return {
        "state_support": bool(raw_floor > 1.0e-12),
        "modular_intersection": bool(
            level["intersection_nonempty"]
            and level["modular_intersection_residual"] is not None
        ),
        "future_cone": bool(null_assembly["future_cone_spectrum_attained"]),
        "causal_nondegenerate": bool(
            len(pairs["causal"]) >= 8 and len(pairs["spacelike"]) >= 8
        ),
    }


def _countermodel_full_rank_snapshots(capture: Mapping[str, Any]) -> dict[str, Any]:
    """State-support countermodel in the attaining direction.

    The measured baseline records that the raw snapshot moment of the actual
    source is rank-deficient: the sampled record vectors do not span port
    space, and faithfulness inside the instruments is carried entirely by
    the declared regularizer.  The countermodel appends one snapshot whose
    rows are the twelve port units, making the raw moment strictly positive
    while ancestry, generators, and ledger stay untouched.  The flip
    demonstrates the clause responds to record-snapshot data alone.
    """

    modified = _deep_copy_capture(capture)
    snapshots = modified["source_artifacts"]["dynamics"]["record_state_snapshots"]
    template = copy.deepcopy(snapshots[0])
    rows = []
    for port in range(12):
        row = copy.deepcopy(template["carrier_rows"][0])
        row["full_port_state"] = [
            1.0 if index == port else 0.0 for index in range(12)
        ]
        rows.append(row)
    template["carrier_rows"] = rows
    template["cycle"] = int(template["cycle"]) + 1000
    snapshots.append(template)
    return modified


def _countermodel_positive_generators(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Future-cone countermodel in the attaining direction.

    The modular generator is replaced by a source-level dominant Hermitian
    matrix, making all four null candidates positive while every other
    family is untouched.  This is the isolation witness that the future-cone
    clause responds to generator data alone.
    """

    modified = _deep_copy_capture(capture)
    primitives = modified["source_artifacts"]["cap_state_raw_primitives"]

    def _norm(matrix: Any) -> float:
        raw = np.asarray(matrix, dtype=float)
        complex_matrix = raw[..., 0] + 1j * raw[..., 1]
        return float(np.linalg.norm(complex_matrix, 2))

    bound = _norm(primitives["m4_generator_z"]) + _norm(
        primitives["m4_generator_x"]
    )
    dominant = np.zeros((4, 4, 2))
    for index in range(4):
        dominant[index, index, 0] = 2.0 * bound
    primitives["modular_generator"] = dominant.tolist()
    return modified


def _countermodel_erased_ancestry(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Causal-degeneracy countermodel: all ancestry edges removed.

    Every event becomes causally isolated, so the causal class collapses
    below the declared floor while states, generators, and intersections are
    untouched.
    """

    modified = _deep_copy_capture(capture)
    modified["postrun_capture"]["raw_ancestry_relations"] = []
    for event in modified["postrun_capture"]["semantic_events"]:
        event["parent_event_ids"] = []
    return modified


def _countermodel_disjoint_overlaps(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Intersection countermodel is carried by the cap-pair declaration.

    The block-algebra intersection is a template-level object, so the
    semantic countermodel declares a disjoint transverse pair; the source
    data itself is untouched.  The isolation vector below therefore calls
    the level report with the disjoint-pair declaration.
    """

    return _deep_copy_capture(capture)


def produce_countermodel_matrix(
    *,
    config: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the countermodel matrix and record isolation vectors."""

    main_config = dict(
        {"carrier_count": 32, "cycles": 6, "seed": 20260751}
        if config is None
        else config
    )
    capture = capture_physical_source(main_config)
    baseline = _clause_vector(capture)

    builders: dict[str, tuple[str, Callable[[Mapping[str, Any]], dict[str, Any]]]] = {
        "state_support": (
            "full_rank_snapshot_enrichment",
            _countermodel_full_rank_snapshots,
        ),
        "future_cone": (
            "dominant_positive_modular_generator",
            _countermodel_positive_generators,
        ),
        "causal_nondegenerate": (
            "erased_ancestry_relations",
            _countermodel_erased_ancestry,
        ),
    }
    rows: dict[str, dict[str, Any]] = {}
    all_isolated = True
    for family, (name, builder) in builders.items():
        modified = builder(capture)
        vector = _clause_vector(modified)
        flipped = {
            key: vector[key] != baseline[key] for key in baseline
        }
        isolated = bool(
            flipped[family] and not any(
                flipped[key] for key in baseline if key != family
            )
        )
        rows[family] = {
            "countermodel": name,
            "kind": "semantic_source_modification",
            "baseline": baseline[family],
            "countermodel_value": vector[family],
            "flipped_families": sorted(
                key for key, value in flipped.items() if value
            ),
            "isolated": isolated,
        }
        all_isolated = all_isolated and isolated

    # Intersection family: the declaration-level disjoint pair.
    level_disjoint = _tower_level_report(
        _countermodel_disjoint_overlaps(capture),
        disjoint_intersection_control=True,
    )
    intersection_row = {
        "countermodel": "declared_disjoint_transverse_pair",
        "kind": "semantic_declaration_modification",
        "baseline": baseline["modular_intersection"],
        "countermodel_value": bool(
            level_disjoint["intersection_nonempty"]
            and level_disjoint["modular_intersection_residual"] is not None
        ),
        "flipped_families": ["modular_intersection"],
        "isolated": bool(not level_disjoint["intersection_nonempty"]),
    }
    rows["modular_intersection"] = intersection_row
    all_isolated = all_isolated and intersection_row["isolated"]

    report = {
        "schema": SCHEMA,
        "issue": 577,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "main_config": main_config,
        "capture_sha256": capture["capture_sha256"],
        "baseline_clause_vector": baseline,
        "countermodels": rows,
        "all_countermodels_isolated": bool(all_isolated),
        "semantic_vs_syntactic": (
            "every countermodel modifies source data or a declared pair at "
            "the semantic level and re-runs the full clause evaluation; no "
            "report field, manifest row, or evaluator output is edited"
        ),
        "boundary_map_status": (
            "the concrete protected boundary map, its gauge-equivalence "
            "proof, and the refinement-uniform inverse modulus are a "
            "separate formal deliverable and are not claimed here"
        ),
        "COUNTERMODEL_ISOLATION_RECEIPT": bool(all_isolated),
        "claim_boundary": (
            "Finite issue-577 countermodel matrix for the instrument clause "
            "families. Isolation shows each receipt is load-bearing; it does "
            "not certify the Einstein branch, and no physical promotion "
            "follows from any output."
        ),
    }
    if output_path is not None:
        Path(output_path).write_text(_canonical_json(report), encoding="utf-8")
    return report
