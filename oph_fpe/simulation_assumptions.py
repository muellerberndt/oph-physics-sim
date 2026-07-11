from __future__ import annotations

import math
import re
from typing import Any, Mapping


ASSUMPTION_DEFINITIONS: dict[str, dict[str, Any]] = {
    "screen_s2": {
        "label": "finite observer screen is S2",
        "applies_to": ["screen", "visualizationViews.fluctuatingQuantumVacuum"],
    },
    "bw_2pi_geometric_branch": {
        "label": "BW 2*pi branch supplies the geometric observer clock",
        "applies_to": ["observerModularTime", "consensusBulk"],
    },
    "observer_modular_time_interpretation": {
        "label": "finite modular-depth readout is rendered as observer-experienced time",
        "applies_to": ["observerModularTime", "subjectiveObserverCameras[*].timeFrames"],
    },
    "h3_observer_chart": {
        "label": "observer-facing spatial chart is H3",
        "applies_to": ["consensusBulk", "coordinateSystems.h3_hyperboloid_spatial_components_v1"],
    },
    "screen_observer_to_h3_camera_embedding": {
        "label": "an S2 observer axis determines an inward-looking H3 camera frame",
        "applies_to": ["subjectiveObserverCameras", "assumedDs4Spacetime.observerReferenceFrames"],
    },
    "record_population_on_h3": {
        "label": "persistent consensus records populate the observer-facing H3 chart",
        "applies_to": ["consensusBulk.objects", "emergentCurvedSpacetime.continuousBulkField"],
    },
    "refinement_naturality_visualization": {
        "label": "the visual chart/population handoff is treated as refinement-natural",
        "applies_to": ["consensusBulk", "visualizationRenderData.sceneGraph.bulk"],
    },
    "ds4_open_slicing_background": {
        "label": "the H3 chart is placed in an open-slicing de Sitter background",
        "applies_to": ["assumedDs4Spacetime"],
    },
    "positive_cosmological_constant": {
        "label": "the renderer uses positive Lambda=3H^2",
        "applies_to": ["assumedDs4Spacetime.geometry.cosmologicalConstant"],
    },
    "observer_tetrad_visualization": {
        "label": "observer cameras carry comoving reference frames/tetrads",
        "applies_to": ["assumedDs4Spacetime.observerReferenceFrames"],
    },
    "topological_defects_render_as_matter": {
        "label": "topological defect worldlines receive matter-like visual styling",
        "applies_to": ["assumedDs4Spacetime.defectMatterRendering"],
    },
    "cmb_screen_to_temperature_transfer_visualization": {
        "label": "screen readout is handed to a CMB-temperature transfer renderer",
        "applies_to": ["cmbComparison.assumedVisualization", "visualizationRenderData.plotSeries"],
    },
    "cmb_tt_reference_shape_visualization": {
        "label": "a pinned measured/reference TT shape completes the explanatory CMB view",
        "applies_to": ["cmbComparison.assumedVisualization.referenceRows"],
    },
}

VISUAL_UNIVERSE_ASSUMPTIONS: tuple[str, ...] = tuple(ASSUMPTION_DEFINITIONS)

_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


def simulation_assumption_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the explicit, visualization-only known-universe assumption ledger.

    This lane is intentionally disjoint from theorem and physical receipts.  It
    authorizes complete explanatory rendering while preserving every open proof
    obligation as open.
    """

    raw_section = config.get("simulation_assumptions", {})
    section = dict(raw_section) if isinstance(raw_section, Mapping) else {}
    enabled = section.get("enabled") is True
    scope = str(section.get("scope", "visualization_only"))
    scope_valid = scope == "visualization_only"
    raw_declared = section.get("assumed", {})
    declared = dict(raw_declared) if isinstance(raw_declared, Mapping) else {}
    assumptions = {
        name: {
            "label": definition["label"],
            "applies_to": list(definition["applies_to"]),
            "status": (
                "assumed_for_visualization"
                if enabled and scope_valid and declared.get(name) is True
                else "blocked"
            ),
            "assumed": bool(enabled and scope_valid and declared.get(name) is True),
            "provenance": (
                "explicit_simulation_assumption"
                if enabled and scope_valid and declared.get(name) is True
                else None
            ),
            "proof_receipt": False,
            "physical_measurement_receipt": False,
        }
        for name, definition in ASSUMPTION_DEFINITIONS.items()
    }
    missing = [name for name, row in assumptions.items() if row["assumed"] is not True]

    raw_ds4 = section.get("ds4", {})
    ds4 = dict(raw_ds4) if isinstance(raw_ds4, Mapping) else {}
    ds4_parameters, ds4_valid = _ds4_parameters(ds4)

    raw_camera = section.get("observer_camera", {})
    camera = dict(raw_camera) if isinstance(raw_camera, Mapping) else {}
    camera_parameters, camera_valid = _observer_camera_parameters(camera)

    raw_cmb = section.get("cmb_visualization", {})
    cmb = dict(raw_cmb) if isinstance(raw_cmb, Mapping) else {}
    cmb_parameters, cmb_valid = _cmb_visualization_parameters(cmb)

    assumed_bridges = sorted(
        {str(value) for value in section.get("assumed_bridges", []) if isinstance(value, str) and value}
    )
    parameter_sets_valid = bool(ds4_valid and camera_valid and cmb_valid)
    receipt = bool(enabled and scope_valid and not missing and parameter_sets_valid)
    manifest = {
        "schema": "oph_simulation_assumption_manifest_v1",
        "profile": str(section.get("profile", "none")),
        "scope": scope,
        "scope_valid": scope_valid,
        "policy_id": str(section.get("policy_id", "oph-visual-bridges-v1")),
        "enabled": enabled,
        "assumed_bridges": assumed_bridges,
        "assumptions": assumptions,
        "missing_assumptions": missing,
        "parameter_sets_valid": parameter_sets_valid,
        "SIMULATION_ASSUMPTIONS_COMPLETE_RECEIPT": receipt,
        "simulation_assumptions_complete_receipt": receipt,
        "SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT": receipt,
        "simulation_assumed_visual_universe_receipt": receipt,
        "ds4_visualization_parameters": ds4_parameters,
        "observer_camera_visualization_parameters": camera_parameters,
        "cmb_visualization_parameters": cmb_parameters,
        "computed_theorem_receipts_unchanged": True,
        "proof_receipt": False,
        "physical_measurement_receipt": False,
        "claim_boundary": (
            "Renderer-completion ledger only. True rows are assumptions authorized for the visual OPH "
            "universe simulation; they do not prove OPH, discharge Lean obligations, establish neutral "
            "bulk, identify defects as measured particles, or turn Einstein/CMB receipts true."
        ),
    }
    return revalidate_simulation_assumption_manifest(manifest)


def revalidate_simulation_assumption_manifest(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recompute trust-sensitive fields of a serialized assumption manifest.

    Consumers must call this instead of trusting a top-level receipt copied from
    JSON.  Literal booleans, scope, parameter relations, and the no-promotion
    guard are checked again.
    """

    manifest = dict(value) if isinstance(value, Mapping) else {}
    rows = manifest.get("assumptions") if isinstance(manifest.get("assumptions"), Mapping) else {}
    row_structure_valid = True
    normalized_rows: dict[str, dict[str, Any]] = {}
    for name, definition in ASSUMPTION_DEFINITIONS.items():
        raw = rows.get(name)
        if not isinstance(raw, Mapping):
            raw = {}
            row_structure_valid = False
        assumed = raw.get("assumed") is True
        if raw.get("assumed") not in (True, False):
            row_structure_valid = False
        if raw.get("proof_receipt", False) is not False:
            row_structure_valid = False
        if raw.get("physical_measurement_receipt", False) is not False:
            row_structure_valid = False
        normalized_rows[name] = {
            "label": str(raw.get("label") or definition["label"]),
            "applies_to": list(raw.get("applies_to") or definition["applies_to"]),
            "status": "assumed_for_visualization" if assumed else "blocked",
            "assumed": assumed,
            "provenance": "explicit_simulation_assumption" if assumed else None,
            "proof_receipt": False,
            "physical_measurement_receipt": False,
        }
    if set(rows) != set(ASSUMPTION_DEFINITIONS):
        row_structure_valid = False

    ds4 = manifest.get("ds4_visualization_parameters")
    camera = manifest.get("observer_camera_visualization_parameters")
    cmb = manifest.get("cmb_visualization_parameters")
    ds4_valid = _serialized_ds4_parameters_valid(ds4)
    camera_valid = _serialized_camera_parameters_valid(camera)
    cmb_valid = _serialized_cmb_parameters_valid(cmb)
    scope_valid = manifest.get("scope") == "visualization_only" and manifest.get("scope_valid") is True
    integrity_valid = bool(
        manifest.get("schema") == "oph_simulation_assumption_manifest_v1"
        and manifest.get("enabled") is True
        and scope_valid
        and manifest.get("computed_theorem_receipts_unchanged") is True
        and row_structure_valid
        and ds4_valid
        and camera_valid
        and cmb_valid
    )
    missing = [name for name, row in normalized_rows.items() if row["assumed"] is not True]
    complete = bool(integrity_valid and not missing)
    manifest.update(
        {
            "schema": manifest.get("schema", "oph_simulation_assumption_manifest_v1"),
            "scope_valid": scope_valid,
            "assumptions": normalized_rows,
            "missing_assumptions": missing,
            "parameter_sets_valid": bool(ds4_valid and camera_valid and cmb_valid),
            "manifest_integrity_valid": integrity_valid,
            "SIMULATION_ASSUMPTIONS_COMPLETE_RECEIPT": complete,
            "simulation_assumptions_complete_receipt": complete,
            "SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT": complete,
            "simulation_assumed_visual_universe_receipt": complete,
            "computed_theorem_receipts_unchanged": (
                manifest.get("computed_theorem_receipts_unchanged") is True
            ),
            "proof_receipt": False,
            "physical_measurement_receipt": False,
        }
    )
    return manifest


def manifest_assumptions_pass(manifest: Mapping[str, Any], *names: str) -> bool:
    validated = revalidate_simulation_assumption_manifest(manifest)
    rows = validated.get("assumptions") or {}
    return bool(
        validated.get("manifest_integrity_valid") is True
        and names
        and all(isinstance(rows.get(name), Mapping) and rows[name].get("assumed") is True for name in names)
    )


def _ds4_parameters(ds4: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    radius_declared = "curvature_radius" in ds4
    hubble_declared = "hubble_parameter" in ds4
    declared_radius = _optional_positive_float(ds4.get("curvature_radius"))
    declared_hubble = _optional_positive_float(ds4.get("hubble_parameter"))
    inputs_valid = bool(
        (radius_declared or hubble_declared)
        and (not radius_declared or declared_radius is not None)
        and (not hubble_declared or declared_hubble is not None)
    )
    if declared_radius is not None:
        curvature_radius = declared_radius
        hubble_parameter = 1.0 / curvature_radius
    elif declared_hubble is not None:
        hubble_parameter = declared_hubble
        curvature_radius = 1.0 / hubble_parameter
    else:
        curvature_radius = 1.0
        hubble_parameter = 1.0
    relation_valid = bool(
        inputs_valid
        and (
            declared_radius is None
            or declared_hubble is None
            or math.isclose(declared_radius * declared_hubble, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-12)
        )
    )
    proper_time_min = _optional_positive_float(ds4.get("proper_time_min_over_h"))
    proper_time_max = _optional_positive_float(ds4.get("proper_time_max_over_h"))
    time_range_valid = bool(
        proper_time_min is not None
        and proper_time_max is not None
        and proper_time_max > proper_time_min
    )
    time_sample_raw = ds4.get("time_sample_count")
    time_sample_valid = bool(type(time_sample_raw) is int and 2 <= time_sample_raw <= 4096)
    units_valid = bool(isinstance(ds4.get("units"), str) and str(ds4.get("units")).strip())
    parameters = {
        "coordinate_chart": "open_h3_slicing",
        "metric": "ds^2=-d_tau^2+H^-2*sinh(H*tau)^2*dH3^2",
        "curvature_radius": curvature_radius,
        "hubble_parameter": hubble_parameter,
        "declared_curvature_radius": ds4.get("curvature_radius") if radius_declared else None,
        "declared_hubble_parameter": ds4.get("hubble_parameter") if hubble_declared else None,
        "parameter_inputs_valid": inputs_valid,
        "de_sitter_radius_relation_valid": relation_valid,
        "proper_time_min_over_h": proper_time_min if proper_time_min is not None else 0.05,
        "proper_time_max_over_h": proper_time_max if proper_time_max is not None else 3.0,
        "proper_time_range_valid": time_range_valid,
        "normalization_policy": "H=1/curvature_radius; radius wins when both are declared",
        "time_sample_count": _bounded_int(time_sample_raw, 96, minimum=2, maximum=4096),
        "time_sample_count_valid": time_sample_valid,
        "units": str(ds4.get("units", "simulation_units")),
        "units_valid": units_valid,
        "provenance": "explicit_simulation_assumption",
    }
    return parameters, bool(
        inputs_valid and relation_valid and time_range_valid and time_sample_valid and units_valid
    )


def _observer_camera_parameters(camera: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    radial = _optional_positive_float(camera.get("h3_radial_coordinate"))
    fov = _optional_positive_float(camera.get("fov_degrees"))
    look_at = _finite_vec3(camera.get("look_at"))
    orientation = camera.get("orientation")
    valid = bool(
        camera.get("coordinate_system") == "h3_hyperboloid_spatial_components_v1"
        and
        radial is not None
        and fov is not None
        and 1.0 <= fov <= 170.0
        and look_at is not None
        and orientation == "inward_radial"
    )
    return {
        "coordinate_system": "h3_hyperboloid_spatial_components_v1",
        "h3_radial_coordinate": radial if radial is not None else 1.18,
        "look_at": look_at if look_at is not None else [0.0, 0.0, 0.0],
        "orientation": str(orientation or "inward_radial"),
        "fov_degrees": fov if fov is not None else 72.0,
        "parameter_inputs_valid": valid,
        "provenance": "explicit_simulation_assumption",
    }, valid


def _cmb_visualization_parameters(cmb: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    label = cmb.get("reference_label")
    path = cmb.get("reference_path")
    source_url = cmb.get("reference_source_url")
    expected_hash = str(cmb.get("reference_sha256") or "").lower()
    transfer_model = cmb.get("transfer_model")
    sky_seed = _optional_int(cmb.get("sky_realization_seed"))
    valid = bool(
        isinstance(label, str)
        and bool(label.strip())
        and isinstance(path, str)
        and bool(path.strip())
        and isinstance(source_url, str)
        and bool(source_url.strip())
        and _SHA256_RE.fullmatch(expected_hash)
        and transfer_model == "pinned_tt_reference_best_fit_visualization"
        and sky_seed is not None
    )
    normalized_hash = expected_hash if expected_hash.startswith("sha256:") else f"sha256:{expected_hash}"
    return {
        "reference_label": str(label or "none"),
        "reference_path": str(path or ""),
        "reference_source_url": str(source_url or ""),
        "reference_sha256": normalized_hash,
        "transfer_model": str(transfer_model or "none"),
        "sky_realization_seed": int(sky_seed or 0),
        "parameter_inputs_valid": valid,
        "provenance": "explicit_simulation_assumption",
        "claim_boundary": (
            "Pinned observed/reference TT rows and their published best-fit column are visual references, "
            "not an OPH prediction or fitted simulator output."
        ),
    }, valid


def _serialized_ds4_parameters_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    radius = _optional_positive_float(value.get("curvature_radius"))
    hubble = _optional_positive_float(value.get("hubble_parameter"))
    tau_min = _optional_positive_float(value.get("proper_time_min_over_h"))
    tau_max = _optional_positive_float(value.get("proper_time_max_over_h"))
    return bool(
        value.get("parameter_inputs_valid") is True
        and value.get("de_sitter_radius_relation_valid") is True
        and value.get("proper_time_range_valid") is True
        and value.get("time_sample_count_valid") is True
        and value.get("units_valid") is True
        and radius is not None
        and hubble is not None
        and math.isclose(radius * hubble, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-12)
        and tau_min is not None
        and tau_max is not None
        and tau_max > tau_min
    )


def _serialized_camera_parameters_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    radial = _optional_positive_float(value.get("h3_radial_coordinate"))
    fov = _optional_positive_float(value.get("fov_degrees"))
    return bool(
        value.get("parameter_inputs_valid") is True
        and value.get("coordinate_system") == "h3_hyperboloid_spatial_components_v1"
        and radial is not None
        and fov is not None
        and 1.0 <= fov <= 170.0
        and _finite_vec3(value.get("look_at")) is not None
        and value.get("orientation") == "inward_radial"
    )


def _serialized_cmb_parameters_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return bool(
        value.get("parameter_inputs_valid") is True
        and isinstance(value.get("reference_label"), str)
        and bool(value.get("reference_label"))
        and isinstance(value.get("reference_path"), str)
        and bool(value.get("reference_path"))
        and isinstance(value.get("reference_source_url"), str)
        and bool(value.get("reference_source_url"))
        and _SHA256_RE.fullmatch(str(value.get("reference_sha256") or "").lower())
        and value.get("transfer_model") == "pinned_tt_reference_best_fit_visualization"
        and type(value.get("sky_realization_seed")) is int
    )


def _optional_positive_float(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _optional_int(value: Any) -> int | None:
    return int(value) if type(value) is int else None


def _finite_vec3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        parsed = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return parsed if all(math.isfinite(item) for item in parsed) else None


def _bounded_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    parsed = int(value) if type(value) is int else int(default)
    return max(int(minimum), min(int(maximum), parsed))
