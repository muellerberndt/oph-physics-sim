"""Stage 4: inhabitation, provenance DAG, and replay verifier.

Stage 4 closes the staged construction: one target-clean source, the
declared sixteen-thousand-carrier capture, inhabits the complete typed
domain when the three stage receipts are attained, bound to one
canonical discrete source projection, resolvable from one provenance
DAG, and reproducible exactly on the serialized finite interface by a
fresh run of the same producers.

The fast bundle verifier does not trust a stored top-level verdict.  It
checks that each recorded clause agrees with the supporting serialized
summary fields available in the receipt, checks nested source/domain
bindings, and checks the control summary.  Some supporting summaries
are themselves stored booleans, so this is an internal-consistency
check rather than a source-level recomputation.  A coordinated rewrite of
receipt and manifest bytes is outside this static verifier's trust
model; the fresh chained producer replay is the source-level check.  The
provenance DAG hashes the configuration,
the consumed source projection, the full-capture diagnostic, the
producer and capture modules, and every receipt, and
the configuration gate refuses the declared forbidden input keys.  The
negative-control matrix maps the six
declared control families, collapsed prescribed chart rank, split-source
feature form, failed cocycle, nonlocal operator, mixed source, and target
injection, to the
detected controls of the stage receipts.  Preservation rows map gauge
relabelling, chart changes, lift-rank checks, and one subcomplex
restriction to the finite certificates that carry them.  No output of
this module carries physical promotion.
"""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import oph_fpe.bulk.event_manifold_producer as event_manifold_producer
import oph_fpe.bulk.physical_h3_kms_source_capture as source_capture
import oph_fpe.local_domain.stage1_event_complex as stage1_module
import oph_fpe.local_domain.stage2_spin_layer as stage2_module
import oph_fpe.local_domain.stage3_typed_domain as stage3_module
from oph_fpe.local_domain.receipt_io import load_manifest_pinned_receipt
from oph_fpe.local_domain.stage1_event_complex import (
    FORBIDDEN_CONFIG_KEYS,
    MAIN_CONFIG,
)

SCHEMA = "oph.local-domain-stage4.v1"
PHYSICAL_PROMOTION_ALLOWED = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local_domain"

STAGE_RECEIPTS = {
    "stage1": ("stage1_receipt.json", "receipt_sha256"),
    "stage2": ("stage2_receipt.json", "stage2_receipt_sha256"),
    "stage3": ("stage3_receipt.json", "stage3_receipt_sha256"),
}
STAGE_SCHEMAS = {
    "stage1": "oph.local-domain-stage1.v1",
    "stage2": "oph.local-domain-stage2.v1",
    "stage3": "oph.local-domain-stage3.v1",
}
MANIFEST_SCHEMA = "oph.local-domain-stage1.manifest.v1"

CONTROL_MATRIX = {
    "collapsed_prescribed_spatial_axis": (
        "stage1",
        "collapsed_prescribed_spatial_axis",
    ),
    "split_source_feature_form": (
        "stage1",
        "split_source_feature_form",
    ),
    "failed_cocycle": ("stage1", "failed_cocycle"),
    "nonlocal_operator": ("stage3", "nonlocal_operator"),
    "mixed_source": ("stage2", "mixed_source"),
    "target_injection": ("stage2", "target_injection"),
}

PRESERVATION_SQUARES = {
    "gauge_relabelling": ("stage3", "gauge_relabelling_covariant"),
    "chart_changes": ("stage1", "transitions_supported"),
    "lift_changes": ("stage2", "lift_ambiguity_rank_cross_checked"),
    "subcomplex_restriction": (
        "stage3",
        "subcomplex_restriction_naturality_exact",
    ),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _module_sha(module) -> str:
    return _sha256_bytes(Path(module.__file__).read_bytes())


def _recompute_verdict(receipt: Mapping[str, Any]) -> str:
    clauses_pass = all(receipt.get("clause_verdicts", {}).values()) and bool(
        receipt.get("clause_verdicts")
    )
    controls = receipt.get("negative_controls", {})
    controls_pass = bool(controls) and all(
        row.get("control_failure_detected") for row in controls.values()
    )
    blockers_empty = receipt.get("blockers") == []
    return (
        "ATTAINED"
        if clauses_pass and controls_pass and blockers_empty
        else "NOT_ATTAINED"
    )


def _expected_stage_clauses(
    stage: str, receipt: Mapping[str, Any]
) -> dict[str, bool]:
    """Rebuild each stage clause from its recorded supporting payload.

    This is a fast internal-consistency verifier for the serialized
    receipt.  The producer semantic replay is the stronger check against
    a fresh source evaluation.
    """

    if stage == "stage1":
        causal = receipt["causal_certificates"]
        chart_rank = receipt["prescribed_chart_rank_certificate"]
        construction = receipt["prescribed_chart_construction"]
        gates = receipt["declared_acceptance_gates"]
        rank_floor = gates[
            "prescribed_chart_full_rank_ratio_floor"
        ]
        target_inertia = tuple(
            gates["held_out_feature_form_target_inertia"]
        )
        full_singulars = chart_rank["full_chart_singulars"]
        rebuilt_full_rank_ratio = (
            full_singulars[-1] / full_singulars[0]
            if len(full_singulars) == 4 and full_singulars[0] > 0
            else 0.0
        )
        fit = receipt["held_out_quadratic_fit"]
        atlas = receipt["atlas"]
        supported = {
            f"{edge[0]}-{edge[1]}" for edge in atlas["supported_edges"]
        }
        transition_residuals = atlas["transition_residuals"]
        transition_scope = receipt["cover_transition_scope"]
        orientation = receipt["orientation_certificate"]
        time_orientation = receipt["time_orientation_certificate"]
        return {
            "event_classes_and_order_constructed": bool(
                receipt["event_count"] > 0
            ),
            "source_relation_identifiers_well_formed": bool(
                causal["event_keys_unique"]
                and causal["ancestry_endpoints_resolved"]
            ),
            "causal_order_acyclic": bool(
                causal["acyclic"] and causal["antisymmetric"]
            ),
            "strict_time_function": bool(causal["strict_time_function"]),
            "prescribed_four_coordinate_chart_nondegenerate": bool(
                chart_rank[
                    "prescribed_four_coordinate_chart_nondegenerate"
                ]
                and chart_rank["coordinate_count"] == 4
                and len(chart_rank["full_chart_singulars"]) == 4
                and chart_rank["full_chart_rank_ratio"] >= rank_floor
                and abs(
                    chart_rank["full_chart_rank_ratio"]
                    - rebuilt_full_rank_ratio
                )
                <= 1.0e-15
                and chart_rank["full_chart_rank_ratio_floor"]
                == rank_floor
                and rank_floor
                == stage1_module.FULL_CHART_RANK_RATIO_FLOOR
                and construction["ancestry_depth_coordinate_count"] == 1
                and construction[
                    "spectral_embedding_coordinate_count"
                ]
                == 3
                and construction["total_coordinate_count"] == 4
            ),
            "held_out_feature_form_inertia_1_3": bool(
                fit["fitted"]
                and tuple(fit["inertia"]) == target_inertia
                and target_inertia == stage1_module.TARGET_INERTIA
            ),
            "atlas_covers_all_events": bool(
                atlas["covered_event_count"] == receipt["event_count"]
                and all(
                    count >= stage1_module.MIN_CHART_EVENTS
                    for count in atlas["chart_event_counts"]
                )
                and atlas["unreached_observers"] == 0
            ),
            "atlas_nerve_connected": bool(atlas["nerve_connected"]),
            "transitions_supported": bool(
                supported
                and transition_scope[
                    "independent_observer_chart_reconstruction"
                ]
                is False
                and all(
                    key in transition_residuals
                    and transition_residuals[key]
                    <= stage1_module.TRANSITION_RESIDUAL_GATE
                    for key in supported
                )
            ),
            "cocycles_within_gate": bool(
                atlas["triple_defects"]
                and all(
                    defect <= stage1_module.COCYCLE_DEFECT_GATE
                    for defect in atlas["triple_defects"].values()
                )
            ),
            "orientation_assignment_exists": bool(
                orientation["orientable"]
            ),
            "time_orientation_consistent": bool(
                time_orientation["time_orientable"]
                and all(
                    coefficient >= stage1_module.TIME_COEFF_FLOOR
                    for coefficient in time_orientation[
                        "time_coefficients"
                    ].values()
                )
            ),
        }
    if stage == "stage2":
        layer = receipt["seam_layer"]
        domain = layer["domain_complex"]
        frame = receipt["atlas_frame_layer"]
        return {
            "seam_convention_verified": bool(
                layer["convention_gate"][
                    "uniform_reversing_single_port"
                ]
            ),
            "same_source_binding": bool(
                receipt["stage1_binding"]["same_source"]
            ),
            "visible_domain_nonempty": bool(
                domain["node_count"] > 0 and domain["edge_count"] > 0
            ),
            "parallel_signs_consistent": bool(
                domain["parallel_seam_sign_conflicts"] == 0
            ),
            "holonomy_census_exact": bool(
                layer["holonomy_census"]["minus_holonomy_count"]
                == domain["triangle_count"]
            ),
            "orientability_cross_check": bool(
                layer["orientability"]["triangle_consistency_holds"]
            ),
            "lift_ambiguity_rank_cross_checked": bool(
                layer["lift_ambiguity"]["rank_cross_checked"]
            ),
            "atlas_frame_lift_recorded": bool(
                frame is not None and frame["nerve_rank_cross_checked"]
            ),
            "scale_ladder_recorded": bool(
                receipt["scale_ladder"]["verdict_stable_across_scales"]
            ),
        }
    if stage == "stage3":
        sections = receipt["section_typing"]
        adjoint = receipt["adjoint_certificate"]
        kernel = receipt["kinetic_kernel_certificate"]
        covariance = receipt["gauge_covariance_certificate"]
        gluing = receipt["restriction_gluing_certificate"]
        naturality = receipt[
            "subcomplex_restriction_naturality_certificate"
        ]
        boundary = receipt["boundary_typing"]
        binding = receipt["stage_binding"]
        return {
            "section_spaces_typed": bool(
                sections["conjugation_involution_exact"]
                and sections["chirality_grading_exact"]
                and sections["gauge_sections_unit"]
            ),
            "declared_counting_measure_typed": bool(
                receipt["measure_typing"]["positive"]
            ),
            "operators_local": bool(
                receipt["locality_certificate"]["local"]
            ),
            "covariant_derivative_typed": bool(
                receipt["covariant_derivative_typing"][
                    "matches_domain_edge_count"
                ]
            ),
            "adjoint_identity_exact": bool(
                adjoint["adjoint_identity_exact"]
            ),
            "kinetic_identities_exact": bool(
                adjoint["kinetic_identity_exact"]
                and adjoint["kinetic_nonnegative"]
            ),
            "kinetic_kernel_matches_stage2": bool(
                kernel["witnesses_annihilated"]
                and kernel["frustrated_component_witnesses_verified"]
                and kernel["rank_theorem"]["applied"]
                and kernel["matches_stage2"]
            ),
            "gauge_relabelling_covariant": bool(
                covariance["derivative_covariant"]
                and covariance["kinetic_norm_invariant"]
            ),
            "restriction_gluing_exact": bool(
                gluing["cover_complete"]
                and gluing["overlap_sections_agree"]
                and gluing["gluing_reconstructs_section"]
            ),
            "subcomplex_restriction_naturality_exact": bool(
                naturality["restriction_commutes_with_derivative"]
            ),
            "boundary_typed": bool(
                boundary["dirichlet_restriction_exact"]
            ),
            "same_source_binding": bool(
                binding["same_source"] and binding["same_domain_freeze"]
            ),
        }
    raise ValueError(f"unknown stage {stage}")


def _stage_content_consistency(
    stage: str, receipt: Mapping[str, Any]
) -> list[str]:
    """Return serialized content-consistency blockers for one stage."""

    blockers: list[str] = []
    try:
        expected = _expected_stage_clauses(stage, receipt)
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        return [f"{stage}_supporting_payload_invalid:{type(exc).__name__}"]
    recorded = receipt.get("clause_verdicts", {})
    if not isinstance(recorded, Mapping):
        return [f"{stage}_clause_payload_invalid:TypeError"]
    for name, value in expected.items():
        if recorded.get(name) is not value:
            blockers.append(f"{stage}_clause_content_mismatch:{name}")
    if set(recorded) != set(expected):
        blockers.append(f"{stage}_clause_schema_mismatch")
    controls = receipt.get("negative_controls", {})
    if not isinstance(controls, Mapping) or not all(
        isinstance(row, Mapping) for row in controls.values()
    ):
        blockers.append(f"{stage}_control_payload_invalid:TypeError")
        controls_recomputed = False
    else:
        controls_recomputed = bool(controls) and all(
            row.get("control_failure_detected")
            for row in controls.values()
        )
    if receipt.get("controls_fail_closed") is not controls_recomputed:
        blockers.append(f"{stage}_control_summary_mismatch")
    if stage == "stage1":
        try:
            chart_rank = receipt["prescribed_chart_rank_certificate"]
            chart_rank_recomputed = bool(
                chart_rank["coordinate_count"] == 4
                and len(chart_rank["full_chart_singulars"]) == 4
                and chart_rank["full_chart_rank_ratio"]
                >= stage1_module.FULL_CHART_RANK_RATIO_FLOOR
                and chart_rank["full_chart_rank_ratio_floor"]
                == stage1_module.FULL_CHART_RANK_RATIO_FLOOR
            )
            if (
                chart_rank[
                    "prescribed_four_coordinate_chart_nondegenerate"
                ]
                is not chart_rank_recomputed
            ):
                blockers.append("stage1_chart_rank_summary_mismatch")
        except (KeyError, TypeError, ValueError):
            blockers.append("stage1_chart_rank_payload_invalid")
    return blockers


def _cross_stage_consistency(
    receipts: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Check nested source/domain links without trusting their summaries."""

    blockers: list[str] = []
    try:
        projection_hashes = {
            receipt.get("source_projection_sha256")
            for receipt in receipts.values()
        }
        if None in projection_hashes or len(projection_hashes) != 1:
            blockers.append(
                "source_projection_hashes_disagree_across_stages"
            )

        stage2_binding = receipts["stage2"]["stage1_binding"]
        stage3_binding = receipts["stage3"]["stage_binding"]
        if not isinstance(stage2_binding, Mapping) or not isinstance(
            stage3_binding, Mapping
        ):
            raise TypeError("stage bindings must be objects")
        if not stage2_binding.get("same_source"):
            blockers.append("stage2_not_bound_to_stage1_source")
        if not stage3_binding.get("same_domain_freeze"):
            blockers.append("stage3_not_bound_to_stage2_domain")

        configs = [receipt.get("main_config", {}) for receipt in receipts.values()]
        if not all(isinstance(config, Mapping) for config in configs):
            raise TypeError("stage configurations must be objects")
        if any(config != configs[0] for config in configs):
            blockers.append("configs_disagree_across_stages")
        if configs and configs[0] != MAIN_CONFIG:
            blockers.append("config_not_the_declared_main_config")
        for config in configs:
            forbidden = sorted(FORBIDDEN_CONFIG_KEYS.intersection(config))
            blockers.extend(
                f"forbidden_config_key:{key}" for key in forbidden
            )

        stage1_projection = receipts["stage1"][
            "source_projection_sha256"
        ]
        stage2_projection = receipts["stage2"][
            "source_projection_sha256"
        ]
        stage3_projection = receipts["stage3"][
            "source_projection_sha256"
        ]
        if (
            stage2_binding.get("stage1_source_projection_sha256")
            != stage1_projection
            or stage2_binding.get("stage2_source_projection_sha256")
            != stage2_projection
        ):
            blockers.append(
                "stage2_nested_source_projection_binding_invalid"
            )
        if (
            stage3_binding.get("stage2_source_projection_sha256")
            != stage2_projection
            or stage3_binding.get("stage3_source_projection_sha256")
            != stage3_projection
        ):
            blockers.append(
                "stage3_nested_source_projection_binding_invalid"
            )

        stage2_domain = receipts["stage2"]["seam_layer"][
            "domain_complex"
        ]["complex_freeze_sha256"]
        stage3_domain = receipts["stage3"].get("domain_freeze_sha256")
        if (
            stage3_domain != stage2_domain
            or stage3_binding.get("stage2_domain_freeze_sha256")
            != stage2_domain
            or stage3_binding.get("stage3_domain_freeze_sha256")
            != stage3_domain
        ):
            blockers.append("stage3_nested_domain_binding_invalid")
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        blockers.append(
            f"cross_stage_binding_payload_invalid:{type(exc).__name__}"
        )
    return blockers


def _stage1_array_consistency(
    arrays_path: Path,
    stage1_receipt: Mapping[str, Any],
) -> list[str]:
    """Recompute the Stage-1 array bundle's receipt-bound content."""

    import numpy as np

    blockers: list[str] = []
    try:
        binding = stage1_receipt["array_bundle_binding"]
        specs = binding["array_specs"]
        declared_names = binding["array_names"]
        if (
            not isinstance(binding, Mapping)
            or binding.get("schema")
            != "oph.local-domain-stage1-arrays.v1"
            or not isinstance(specs, Mapping)
            or not isinstance(declared_names, list)
        ):
            raise TypeError("array binding must use the declared schema")
        raw_npz = gzip.decompress(arrays_path.read_bytes())
        with np.load(io.BytesIO(raw_npz), allow_pickle=False) as bundle:
            actual_names = sorted(bundle.files)
            if actual_names != sorted(declared_names):
                blockers.append("stage1_arrays_declared_names_mismatch")
            if actual_names != sorted(specs):
                blockers.append("stage1_arrays_spec_names_mismatch")
            arrays = {name: bundle[name] for name in actual_names}
    except (
        AttributeError,
        EOFError,
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        return [f"stage1_arrays_payload_invalid:{type(exc).__name__}"]

    for name, array in arrays.items():
        spec = specs.get(name)
        if not isinstance(spec, Mapping):
            blockers.append(f"stage1_array_spec_invalid:{name}")
            continue
        if spec.get("dtype") != str(array.dtype):
            blockers.append(f"stage1_array_dtype_mismatch:{name}")
        if spec.get("shape") != list(array.shape):
            blockers.append(f"stage1_array_shape_mismatch:{name}")
        try:
            value_sha256 = _sha256_value(array.tolist())
        except (TypeError, ValueError):
            blockers.append(f"stage1_array_value_invalid:{name}")
            continue
        if spec.get("value_sha256") != value_sha256:
            blockers.append(f"stage1_array_value_hash_mismatch:{name}")

    try:
        seed_count = int(stage1_receipt["atlas"]["seed_count"])
        event_count = int(stage1_receipt["event_count"])
        expected_names = {
            "chart",
            "causal_pairs",
            "direct_ancestry_edges",
            "spacelike_pairs",
            *{f"chart_coords_{index}" for index in range(seed_count)},
            *{f"chart_events_{index}" for index in range(seed_count)},
        }
        if set(arrays) != expected_names:
            blockers.append("stage1_arrays_required_names_mismatch")
        if list(arrays["chart"].shape) != [event_count, 4]:
            blockers.append("stage1_chart_shape_disagrees_with_receipt")
        if list(arrays["direct_ancestry_edges"].shape) != [
            int(stage1_receipt["ancestry_edge_count"]),
            2,
        ]:
            blockers.append("stage1_direct_ancestry_shape_mismatch")
        if stage1_receipt.get("direct_ancestry_array_name") != (
            "direct_ancestry_edges"
        ):
            blockers.append("stage1_direct_ancestry_name_mismatch")
        if _sha256_value(arrays["chart"].tolist()) != stage1_receipt.get(
            "chart_freeze_sha256"
        ):
            blockers.append("stage1_chart_freeze_mismatch")
        chart_coords = [
            arrays[f"chart_coords_{index}"].tolist()
            for index in range(seed_count)
        ]
        if _sha256_value(chart_coords) != stage1_receipt["atlas"].get(
            "atlas_freeze_sha256"
        ):
            blockers.append("stage1_atlas_freeze_mismatch")
        chart_event_counts = stage1_receipt["atlas"][
            "chart_event_counts"
        ]
        for index in range(seed_count):
            if list(arrays[f"chart_coords_{index}"].shape) != [
                event_count,
                4,
            ]:
                blockers.append(
                    f"stage1_chart_coords_shape_mismatch:{index}"
                )
            if list(arrays[f"chart_events_{index}"].shape) != [
                chart_event_counts[index]
            ]:
                blockers.append(
                    f"stage1_chart_events_shape_mismatch:{index}"
                )
        expected_causal = min(
            int(stage1_receipt["causal_pair_total"]),
            int(stage1_receipt["pair_cap"]),
        )
        if list(arrays["causal_pairs"].shape) != [
            expected_causal,
            2,
        ]:
            blockers.append("stage1_causal_pairs_shape_mismatch")
        expected_spacelike = min(
            int(stage1_receipt["spacelike_pair_total"]),
            int(stage1_receipt["pair_cap"]),
        )
        if list(arrays["spacelike_pairs"].shape) != [
            expected_spacelike,
            2,
        ]:
            blockers.append("stage1_spacelike_pairs_shape_mismatch")
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        blockers.append(
            f"stage1_arrays_receipt_binding_invalid:{type(exc).__name__}"
        )
    return blockers


def verify_local_domain_bundle(
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve every artifact and recompute every stage verdict.

    Fail-closed: any missing file, hash mismatch, serialized-content
    inconsistency, cross-stage binding failure, or verdict disagreement
    lands in the blocker list.  The source-level check is the separate
    fresh chained producer replay."""

    base = Path(data_dir) if data_dir is not None else DATA_DIR
    blockers: list[str] = []
    receipts: dict[str, Mapping[str, Any]] = {}

    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "blockers": ["manifest_missing"]}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"passed": False, "blockers": ["manifest_json_invalid"]}
    if not isinstance(manifest, Mapping):
        return {"passed": False, "blockers": ["manifest_json_invalid"]}
    if manifest.get("schema") != MANIFEST_SCHEMA:
        blockers.append("manifest_schema_invalid")

    for stage, (filename, manifest_key) in STAGE_RECEIPTS.items():
        path = base / filename
        if not path.exists():
            blockers.append(f"{stage}_receipt_missing")
            continue
        raw = path.read_bytes()
        if manifest.get(manifest_key) != _sha256_bytes(raw):
            blockers.append(f"{stage}_receipt_hash_mismatch")
            continue
        try:
            receipt = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            blockers.append(f"{stage}_receipt_json_invalid")
            continue
        if not isinstance(receipt, Mapping):
            blockers.append(f"{stage}_receipt_json_invalid")
            continue
        if receipt.get("schema") != STAGE_SCHEMAS[stage]:
            blockers.append(f"{stage}_schema_invalid")
        try:
            recomputed = _recompute_verdict(receipt)
        except (AttributeError, TypeError, ValueError):
            recomputed = "NOT_ATTAINED"
            blockers.append(f"{stage}_verdict_payload_invalid")
        if receipt.get("verdict") != recomputed:
            blockers.append(f"{stage}_verdict_disagrees_with_recomputation")
        if receipt.get("physical_promotion_allowed") is not False:
            blockers.append(f"{stage}_promotion_flag_invalid")
        blockers.extend(_stage_content_consistency(stage, receipt))
        receipts[stage] = receipt

    arrays_path = base / "stage1_arrays.npz.gz"
    if not arrays_path.exists():
        blockers.append("stage1_arrays_missing")
    elif manifest.get("arrays_sha256") != _sha256_bytes(
        arrays_path.read_bytes()
    ):
        blockers.append("stage1_arrays_hash_mismatch")
    elif "stage1" in receipts:
        blockers.extend(
            _stage1_array_consistency(arrays_path, receipts["stage1"])
        )

    if len(receipts) == len(STAGE_RECEIPTS):
        blockers.extend(_cross_stage_consistency(receipts))

    return {
        "passed": bool(not blockers),
        "blockers": sorted(set(blockers)),
        "stage_verdicts": {
            stage: receipt.get("verdict") for stage, receipt in receipts.items()
        },
        "capture_sha256": (
            next(iter(receipts.values())).get("capture_sha256")
            if receipts
            else None
        ),
        "source_projection_sha256": (
            next(iter(receipts.values())).get("source_projection_sha256")
            if receipts
            and len(
                {
                    r.get("source_projection_sha256")
                    for r in receipts.values()
                }
            )
            == 1
            else None
        ),
        "domain_freeze_sha256": (
            receipts.get("stage2", {})
            .get("seam_layer", {})
            .get("domain_complex", {})
            .get("complex_freeze_sha256")
        ),
        "full_capture_diagnostic_hashes": sorted(
            {
                value
                for receipt in receipts.values()
                if (value := receipt.get("capture_sha256")) is not None
            }
        ),
    }


def provenance_dag(receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Hash-bound provenance DAG from configuration to receipts."""

    config = receipts["stage1"]["main_config"]
    artifacts = {
        "configuration": {
            "kind": "declared_configuration",
            "sha256": _sha256_bytes(_canonical_json(config).encode("utf-8")),
        },
        "capture_module": {
            "kind": "evaluator",
            "path": "oph_fpe/bulk/physical_h3_kms_source_capture.py",
            "sha256": _module_sha(source_capture),
        },
        "reconstruction_module": {
            "kind": "evaluator",
            "path": "oph_fpe/bulk/event_manifold_producer.py",
            "sha256": _module_sha(event_manifold_producer),
        },
        "stage1_module": {
            "kind": "evaluator",
            "path": "oph_fpe/local_domain/stage1_event_complex.py",
            "sha256": _module_sha(stage1_module),
        },
        "stage2_module": {
            "kind": "evaluator",
            "path": "oph_fpe/local_domain/stage2_spin_layer.py",
            "sha256": _module_sha(stage2_module),
        },
        "stage3_module": {
            "kind": "evaluator",
            "path": "oph_fpe/local_domain/stage3_typed_domain.py",
            "sha256": _module_sha(stage3_module),
        },
        "source_projection": {
            "kind": "consumed_discrete_source_projection",
            "sha256": receipts["stage1"]["source_projection_sha256"],
        },
        "full_capture_diagnostic": {
            "kind": "environment_sensitive_diagnostic",
            "sha256": receipts["stage1"]["capture_sha256"],
        },
        "stage1_receipt": {
            "kind": "readout",
            "sha256": _sha256_bytes(
                _canonical_json(receipts["stage1"]).encode("utf-8")
            ),
        },
        "stage1_arrays": {
            "kind": "receipt_bound_array_bundle",
            "sha256": _sha256_bytes(
                (DATA_DIR / "stage1_arrays.npz.gz").read_bytes()
            ),
        },
        "stage2_receipt": {
            "kind": "readout",
            "sha256": _sha256_bytes(
                _canonical_json(receipts["stage2"]).encode("utf-8")
            ),
        },
        "stage3_receipt": {
            "kind": "readout",
            "sha256": _sha256_bytes(
                _canonical_json(receipts["stage3"]).encode("utf-8")
            ),
        },
    }
    processes = [
        {
            "process": "capture_physical_source",
            "inputs": ["configuration"],
            "evaluator": "capture_module",
            "output": "full_capture_diagnostic",
        },
        {
            "process": "project_local_domain_source",
            "inputs": ["full_capture_diagnostic"],
            "evaluator": "stage1_module",
            "output": "source_projection",
        },
        {
            "process": "produce_stage1_receipt",
            "inputs": ["source_projection"],
            "evaluator": "stage1_module",
            "supporting_evaluators": ["reconstruction_module"],
            "output": "stage1_receipt",
        },
        {
            "process": "produce_stage1_array_bundle",
            "inputs": ["source_projection", "stage1_receipt"],
            "evaluator": "stage1_module",
            "supporting_evaluators": ["reconstruction_module"],
            "output": "stage1_arrays",
        },
        {
            "process": "produce_stage2_receipt",
            "inputs": ["source_projection", "stage1_receipt"],
            "evaluator": "stage2_module",
            "output": "stage2_receipt",
        },
        {
            "process": "produce_stage3_receipt",
            "inputs": ["source_projection", "stage2_receipt"],
            "evaluator": "stage3_module",
            "output": "stage3_receipt",
        },
    ]
    artifact_names = set(artifacts)
    output_names = [process["output"] for process in processes]
    outputs_unique = len(output_names) == len(set(output_names))
    references_resolve = all(
        process["output"] in artifact_names
        and process["evaluator"] in artifact_names
        and all(name in artifact_names for name in process["inputs"])
        and all(
            name in artifact_names
            for name in process.get("supporting_evaluators", [])
        )
        for process in processes
    )
    adjacency: dict[str, set[str]] = {
        name: set() for name in artifact_names
    }
    indegree = {name: 0 for name in artifact_names}
    if references_resolve:
        for process in processes:
            output = process["output"]
            dependencies = [
                *process["inputs"],
                process["evaluator"],
                *process.get("supporting_evaluators", []),
            ]
            for dependency in dependencies:
                if output not in adjacency[dependency]:
                    adjacency[dependency].add(output)
                    indegree[output] += 1
    queue = sorted(name for name, degree in indegree.items() if degree == 0)
    visited: list[str] = []
    while queue:
        node = queue.pop(0)
        visited.append(node)
        for child in sorted(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
                queue.sort()
    acyclic = bool(
        references_resolve
        and outputs_unique
        and len(visited) == len(artifact_names)
    )
    roots = sorted(
        name
        for name in artifact_names
        if not any(name in children for children in adjacency.values())
    )

    def reachable(start: str) -> set[str]:
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for child in adjacency[node]:
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return seen

    source_reach = reachable("configuration")
    source_chain_outputs = {
        "full_capture_diagnostic",
        "source_projection",
        "stage1_receipt",
        "stage1_arrays",
        "stage2_receipt",
        "stage3_receipt",
    }
    single_source = bool(
        acyclic
        and sum(
            1
            for row in artifacts.values()
            if row["kind"] == "declared_configuration"
        )
        == 1
        and source_chain_outputs.issubset(source_reach)
    )
    return {
        "artifacts": artifacts,
        "processes": processes,
        "roots": roots,
        "references_resolve": references_resolve,
        "outputs_unique": outputs_unique,
        "single_source": single_source,
        "acyclic": bool(acyclic),
        "dag_sha256": _sha256_bytes(
            _canonical_json({"artifacts": artifacts, "processes": processes}).encode(
                "utf-8"
            )
        ),
    }


def _round_trip_canonical(value: Any) -> str:
    """Canonical form after a JSON round trip.

    Serialization is compared after parse and re-serialization on both
    sides, so a producer emitting non-string keys cannot make a frozen
    file and a fresh dict of equal content compare unequal."""

    return _canonical_json(json.loads(_canonical_json(value)))


_ENVIRONMENT_DIAGNOSTIC_PATHS = frozenset(
    {
        ("capture_sha256",),
        ("stage1_binding", "stage1_capture_sha256"),
        ("stage1_binding", "stage2_capture_sha256"),
    }
)


def _semantic_receipt(value: Any, _path: tuple[str, ...] = ()) -> Any:
    """Remove only schema-declared full-capture diagnostic paths.

    Those hashes include floating diagnostics not consumed by the local
    domain and can vary across otherwise equivalent numerical runtimes.
    Every scientific field and the canonical source-projection digest
    remains in the comparison.
    """

    if isinstance(value, Mapping):
        normalized = {}
        for key, child in value.items():
            child_path = (*_path, str(key))
            if child_path in _ENVIRONMENT_DIAGNOSTIC_PATHS:
                continue
            normalized[key] = _semantic_receipt(child, child_path)
        return normalized
    if isinstance(value, list):
        return [
            _semantic_receipt(child, (*_path, str(index)))
            for index, child in enumerate(value)
        ]
    return value


def producer_semantic_replay(
    receipts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Re-run the same three producers and compare their finite semantics."""

    with tempfile.TemporaryDirectory() as scratch:
        stage1_module.produce_stage1_receipt(output_dir=scratch)
        replay_dir = Path(scratch)
        stage1_bytes = (replay_dir / "stage1_receipt.json").read_bytes()
        replayed_stage1 = json.loads(stage1_bytes.decode("utf-8"))
        replayed_stage2 = stage2_module.produce_stage2_receipt(
            stage1_receipt=replayed_stage1
        )
        replayed_stage3 = stage3_module.produce_stage3_receipt(
            stage2_receipt=replayed_stage2
        )
    comparisons = {
        "stage1": bool(
            _round_trip_canonical(
                _semantic_receipt(json.loads(stage1_bytes.decode("utf-8")))
            )
            == _round_trip_canonical(_semantic_receipt(receipts["stage1"]))
        ),
        "stage2": bool(
            _round_trip_canonical(_semantic_receipt(replayed_stage2))
            == _round_trip_canonical(_semantic_receipt(receipts["stage2"]))
        ),
        "stage3": bool(
            _round_trip_canonical(_semantic_receipt(replayed_stage3))
            == _round_trip_canonical(_semantic_receipt(receipts["stage3"]))
        ),
    }
    return {
        "receipt_semantic_exact": comparisons,
        "all_semantic_exact": bool(all(comparisons.values())),
        "producer_independence": False,
        "comparison_scope": (
            "fresh runs of the same producers; exact after excluding only "
            "the environment-sensitive complete-capture diagnostic hashes"
        ),
    }


def produce_stage4_receipt(
    *,
    output_dir: str | Path | None = None,
    run_replay: bool = True,
) -> dict[str, Any]:
    """Produce the stage-4 inhabitation receipt for the frozen bundle."""

    resolution = verify_local_domain_bundle()
    if not resolution["passed"]:
        return {
            "schema": SCHEMA,
            "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
            "bundle_resolution": resolution,
            "verdict": "NOT_ATTAINED",
            "blockers": [
                f"bundle_resolution:{blocker}"
                for blocker in resolution["blockers"]
            ],
        }
    receipts = {
        stage: load_manifest_pinned_receipt(DATA_DIR, filename, manifest_key)
        for stage, (filename, manifest_key) in STAGE_RECEIPTS.items()
    }
    if len(receipts) != len(STAGE_RECEIPTS):
        return {
            "schema": SCHEMA,
            "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
            "verdict": "NOT_ATTAINED",
            "blockers": ["stage_receipts_incomplete"],
        }
    if any(receipt is None for receipt in receipts.values()):
        return {
            "schema": SCHEMA,
            "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
            "verdict": "NOT_ATTAINED",
            "blockers": ["stage_receipts_missing_or_unpinned"],
        }

    dag = provenance_dag(receipts)

    control_rows = {}
    for name, (stage, key) in CONTROL_MATRIX.items():
        detected = bool(
            receipts[stage]["negative_controls"]
            .get(key, {})
            .get("control_failure_detected")
        )
        control_rows[name] = {"stage": stage, "control": key, "detected": detected}
    preservation_rows = {}
    for name, (stage, clause) in PRESERVATION_SQUARES.items():
        preservation_rows[name] = {
            "stage": stage,
            "clause": clause,
            "holds": bool(receipts[stage]["clause_verdicts"].get(clause)),
        }

    replay = (
        producer_semantic_replay(receipts)
        if run_replay
        else {"receipt_semantic_exact": {}, "all_semantic_exact": False}
    )

    controls: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        for name in ("manifest.json", "stage1_arrays.npz.gz"):
            (scratch_dir / name).write_bytes((DATA_DIR / name).read_bytes())
        for stage, (filename, _) in STAGE_RECEIPTS.items():
            (scratch_dir / filename).write_bytes(
                (DATA_DIR / filename).read_bytes()
            )

        tampered = json.loads(
            (scratch_dir / "stage2_receipt.json").read_text(encoding="utf-8")
        )
        tampered["seam_layer"]["lift_ambiguity"]["lift_ambiguity_rank"] += 1
        (scratch_dir / "stage2_receipt.json").write_text(
            _canonical_json(tampered), encoding="utf-8"
        )
        tampered_result = verify_local_domain_bundle(scratch_dir)
        controls["tampered_receipt"] = {
            "control_failure_detected": bool(not tampered_result["passed"])
        }

        (scratch_dir / "stage2_receipt.json").write_bytes(
            (DATA_DIR / "stage2_receipt.json").read_bytes()
        )
        overridden = json.loads(
            (scratch_dir / "stage1_receipt.json").read_text(encoding="utf-8")
        )
        overridden["clause_verdicts"][
            "prescribed_four_coordinate_chart_nondegenerate"
        ] = False
        overridden["verdict"] = "ATTAINED"
        raw = _canonical_json(overridden).encode("utf-8")
        (scratch_dir / "stage1_receipt.json").write_bytes(raw)
        manifest = json.loads(
            (scratch_dir / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["receipt_sha256"] = _sha256_bytes(raw)
        (scratch_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=1) + "\n",
            encoding="utf-8",
        )
        override_result = verify_local_domain_bundle(scratch_dir)
        controls["truth_flag_override"] = {
            "control_failure_detected": bool(
                "stage1_verdict_disagrees_with_recomputation"
                in override_result["blockers"]
            )
        }

        (scratch_dir / "stage3_receipt.json").unlink()
        missing_result = verify_local_domain_bundle(scratch_dir)
        controls["missing_artifact"] = {
            "control_failure_detected": bool(
                "stage3_receipt_missing" in missing_result["blockers"]
            )
        }

    controls_fail_closed = bool(controls) and all(
        row["control_failure_detected"] for row in controls.values()
    )

    continuum_conditions = receipts["stage1"].get("continuum_conditions", {})
    clause_verdicts = {
        "bundle_resolves_fail_closed": resolution["passed"],
        "single_source_dag": bool(
            dag["single_source"] and dag["acyclic"]
        ),
        "continuum_conditions_unsatisfied_at_current_cutoff": bool(
            continuum_conditions.get("status")
            == "NOT_ATTAINED_AT_CURRENT_FINITE_CUTOFF"
            and continuum_conditions.get("conditions")
            and not any(
                condition.get("holds")
                for condition in continuum_conditions["conditions"].values()
            )
        ),
        "forbidden_config_keys_absent": bool(
            not any(
                blocker.startswith("forbidden_config_key")
                for blocker in resolution["blockers"]
            )
        ),
        "stage_verdicts_recomputed_attained": bool(
            resolution["passed"]
            and all(
                verdict == "ATTAINED"
                for verdict in resolution["stage_verdicts"].values()
            )
        ),
        "negative_control_matrix_complete": all(
            row["detected"] for row in control_rows.values()
        ),
        "finite_preservation_rows_hold": all(
            row["holds"] for row in preservation_rows.values()
        ),
        "producer_semantic_replay_exact": replay["all_semantic_exact"],
    }
    blockers = sorted(
        f"clause_failed:{name}"
        for name, verdict in clause_verdicts.items()
        if not verdict
    )
    if not controls_fail_closed:
        blockers.append("negative_control_did_not_fail")
    verdict = "ATTAINED" if not blockers else "NOT_ATTAINED"

    receipt = {
        "schema": SCHEMA,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "capture_sha256": resolution.get("capture_sha256"),
        "source_projection_sha256": resolution.get(
            "source_projection_sha256"
        ),
        "bundle_resolution": resolution,
        "provenance_dag": dag,
        "negative_control_matrix": control_rows,
        "preservation_rows": preservation_rows,
        "producer_semantic_replay": replay,
        "clause_verdicts": clause_verdicts,
        "negative_controls": controls,
        "controls_fail_closed": controls_fail_closed,
        "verdict": verdict,
        "LOCAL_DOMAIN_INHABITATION_RECEIPT": bool(verdict == "ATTAINED"),
        "blockers": blockers,
        "claim_boundary": (
            "Finite stage-4 object: the declared source inhabits "
            "the typed local domain in the finite sense that the three "
            "stage receipts are attained on one canonical discrete source "
            "projection, resolvable "
            "from one acyclic provenance DAG under the declared "
            "configuration-key gate, with the six "
            "negative-control families detected, the finite "
            "preservation rows carried by their stated certificates, and "
            "fresh runs of the "
            "same producers reproduce the serialized finite semantics "
            "exactly. This is a producer replay, not an independent "
            "implementation. The four-coordinate feature chart is "
            "prescribed as ancestry depth plus three spectral coordinates; "
            "its rank check does not select dimension. The inertia "
            "certificate is the held-out fitted form of the finite "
            "event-manifold candidate, a "
            "fitted-form result rather than a manifold signature theorem, "
            "and the stage-1 continuum conditions are not attained at the "
            "current finite cutoff: the cone margins are negative, one "
            "neighborhood fit has inertia (0, 4), the neighborhoods are "
            "closed and finite, and no cofinal refinement family is "
            "certified. The sign layer is the declared reversing convention; "
            "continuum characteristic-class identification and physical "
            "fiber selection require separate hypotheses and certificates. The "
            "domain is a finite causal and local-operator object; no "
            "continuum Lorentzian spacetime and no physical promotion "
            "follows from any output."
        ),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        receipt_bytes = _canonical_json(receipt).encode("utf-8")
        (out / "stage4_receipt.json").write_bytes(receipt_bytes)
        manifest_path = out / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"schema": "oph.local-domain-stage1.manifest.v1"}
        )
        manifest["stage4_receipt"] = "stage4_receipt.json"
        manifest["stage4_receipt_sha256"] = _sha256_bytes(receipt_bytes)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=1) + "\n",
            encoding="utf-8",
        )
    return receipt
