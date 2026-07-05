from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal, Mapping, Sequence


ANOMALY_RELEASE_STATE_RECEIPT = "ANOMALY_RELEASE_STATE_RECEIPT"
ANOMALY_LOAD_OBSERVABLE_RECEIPT = "ANOMALY_LOAD_OBSERVABLE_RECEIPT"
SOURCE_MAXENT_RELEASE_LAW_RECEIPT = "SOURCE_MAXENT_RELEASE_LAW_RECEIPT"
LOAD_QUOTIENT_INVARIANCE_RECEIPT = "LOAD_QUOTIENT_INVARIANCE_RECEIPT"
LOAD_NO_CIRCULARITY_RECEIPT = "LOAD_NO_CIRCULARITY_RECEIPT"
LOAD_NO_DATA_USE_RECEIPT = "LOAD_NO_DATA_USE_RECEIPT"
LOAD_REFINEMENT_COMPATIBILITY_RECEIPT = "LOAD_REFINEMENT_COMPATIBILITY_RECEIPT"
ANOMALY_ABUNDANCE_SOURCE_RECEIPT = "ANOMALY_ABUNDANCE_SOURCE_RECEIPT"
RHO_A_TRANSPORT_RECEIPT = "RHO_A_TRANSPORT_RECEIPT"
RHO_A_SOURCE_RECEIPT = "RHO_A_SOURCE_RECEIPT"

SOURCE_ONLY_ANOMALY_ABUNDANCE = "SOURCE_ONLY_ANOMALY_ABUNDANCE"
PHYSICAL_PARENT_WITH_CONDITIONAL_ABUNDANCE = "PHYSICAL_PARENT_WITH_CONDITIONAL_ABUNDANCE"
CONDITIONAL_SOURCE_STATE = "CONDITIONAL_SOURCE_STATE"

FORBIDDEN_DATA_TOKENS = (
    "cmb",
    "bao",
    "supernova",
    "weak_lensing",
    "weak-lensing",
    "rsd",
    "sparc",
    "cluster",
    "planck",
    "likelihood",
    "posterior",
    "nuisance",
    "diagnostic_residual",
    "boltzmann_residual",
)

FORBIDDEN_CIRCULARITY_TOKENS = (
    "external_omega_a",
    "omega_a",
    "omega_a0",
    "rho_a0",
    "planck",
    "sparc",
    "cluster",
    "boltzmann",
    "likelihood",
    "friedmann_residual",
)


@dataclass(frozen=True)
class AnomalyReleaseStateArtifact:
    artifact_id: str
    regulator_id: str
    parent_generation_id: str
    release_surface_hash: str
    normal_tetrad_hash: str
    scale_factor_hash: str
    physical_volume_hash: str
    comoving_volume_hash: str
    homogeneous_projector_hash: str
    release_boundary_sector_hash: str
    P_star_hash: str
    N_CRC_hash: str
    generation_manifest_hash: str
    no_data_ledger_hash: str
    source_dag_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnomalyLoadObservableArtifact:
    artifact_id: str
    regulator_id: str
    parent_generation_id: str
    release_state_artifact_id: str
    stress_readout_hash: str
    bw_source_readout_hash: str | None
    tomography_artifact_hash: str | None
    variational_moment_residual_hash: str | None
    cell_volume_hash: str
    normal_hash: str
    homogeneous_projector_hash: str
    load_values_hash: str
    load_units: Literal["J", "GeV", "natural_energy"]
    load_bound_hash: str
    quotient_invariance_residual: float
    circularity_guard_hash: str
    normalization_source: str = "source_amplitude_free_parent"
    computed_from_loaded_parent: bool = False
    circularity_inputs: tuple[str, ...] = ()
    hidden_relabel_residual: float = 0.0


@dataclass(frozen=True)
class SourceMaxEntReleaseLawArtifact:
    artifact_id: str
    regulator_id: str
    quotient_space_hash: str
    base_weight_hash: str
    action_hash: str
    constraint_observable_hashes: list[str]
    constraint_target_hashes: list[str]
    constraint_units_hash: str
    zero_mode_policy_hash: str
    sector_policy_hash: str
    lagrange_multiplier_hash: str
    release_law_hash: str
    no_data_ledger_hash: str
    refinement_map_hash: str
    source_dag_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnomalyAbundanceSelectorArtifact:
    artifact_id: str
    regulator_id: str
    parent_generation_id: str
    release_state_artifact_id: str
    release_law_artifact_id: str
    load_observable_artifact_id: str
    quotient_ensemble_hash: str
    P_star_hash: str
    N_CRC_hash: str
    selected_load_energy_hash: str
    rho_A_bar_hash: str
    refinement_residual_hash: str
    no_data_use_receipt_hash: str
    source_claim_label: Literal[
        "PHYSICAL_PARENT_WITH_CONDITIONAL_ABUNDANCE",
        "SOURCE_ONLY_ANOMALY_ABUNDANCE",
    ]
    source_dag_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadRefinementCompatibilityArtifact:
    artifact_id: str
    fine_regulator_id: str
    coarse_regulator_id: str
    coarse_map_hash: str
    fine_release_law_hash: str
    coarse_release_law_hash: str
    law_tv_defect: float
    load_naturality_defect: float
    load_sup_bound: float
    selected_load_difference_bound: float
    passes_exact: bool


def verify_anomaly_release_state(artifact: Any) -> dict[str, Any]:
    data = _mapping(artifact)
    blockers = _required(data, ("artifact_id", "regulator_id", "parent_generation_id"))
    blockers.extend(
        _required_hashes(
            data,
            (
                "release_surface_hash",
                "normal_tetrad_hash",
                "scale_factor_hash",
                "physical_volume_hash",
                "comoving_volume_hash",
                "homogeneous_projector_hash",
                "release_boundary_sector_hash",
                "P_star_hash",
                "N_CRC_hash",
                "generation_manifest_hash",
                "no_data_ledger_hash",
            ),
        )
    )
    bad_inputs = _forbidden_inputs(data)
    if bad_inputs:
        blockers.append("release_state_reads_observational_data")
    return _receipt(ANOMALY_RELEASE_STATE_RECEIPT, blockers, forbidden_inputs=bad_inputs)


def verify_anomaly_load_observable(artifact: Any, *, tolerance: float = 1.0e-9) -> dict[str, Any]:
    data = _mapping(artifact)
    blockers = _required(data, ("artifact_id", "regulator_id", "parent_generation_id", "release_state_artifact_id"))
    blockers.extend(
        _required_hashes(
            data,
            (
                "stress_readout_hash",
                "cell_volume_hash",
                "normal_hash",
                "homogeneous_projector_hash",
                "load_values_hash",
                "load_bound_hash",
                "circularity_guard_hash",
            ),
        )
    )
    if data.get("load_units") not in {"J", "GeV", "natural_energy"}:
        blockers.append("load_units_invalid")
    invariance = verify_load_quotient_invariance(data, tolerance=tolerance)
    circularity = verify_load_no_circularity(data)
    blockers.extend(invariance["blockers"])
    blockers.extend(circularity["blockers"])
    passed = not blockers
    return {
        ANOMALY_LOAD_OBSERVABLE_RECEIPT: passed,
        "passed": passed,
        "blockers": _dedupe(blockers),
        LOAD_QUOTIENT_INVARIANCE_RECEIPT: invariance[LOAD_QUOTIENT_INVARIANCE_RECEIPT],
        LOAD_NO_CIRCULARITY_RECEIPT: circularity[LOAD_NO_CIRCULARITY_RECEIPT],
    }


def verify_load_quotient_invariance(artifact: Any, *, tolerance: float = 1.0e-9) -> dict[str, Any]:
    data = _mapping(artifact)
    blockers: list[str] = []
    residual = _float(data.get("quotient_invariance_residual"))
    hidden_residual = _float(data.get("hidden_relabel_residual", 0.0))
    if residual is None or residual > float(tolerance):
        blockers.append("load_quotient_invariance_residual_too_large")
    if hidden_residual is None or hidden_residual > float(tolerance):
        blockers.append("hidden_carrier_relabel_changes_load")
    return _receipt(LOAD_QUOTIENT_INVARIANCE_RECEIPT, blockers)


def verify_load_no_circularity(artifact: Any) -> dict[str, Any]:
    data = _mapping(artifact)
    blockers: list[str] = []
    inputs = _as_strings(
        [
            data.get("normalization_source"),
            data.get("parent_normalization_source"),
            data.get("external_normalization_source"),
            *_sequence(data.get("circularity_inputs")),
        ]
    )
    bad_inputs = _matching_tokens(inputs, FORBIDDEN_CIRCULARITY_TOKENS)
    if bool(data.get("computed_from_loaded_parent", False)):
        blockers.append("load_computed_from_amplitude_loaded_parent")
    if bad_inputs:
        blockers.append("load_normalization_reads_external_abundance")
    if not _hashish(data.get("circularity_guard_hash")):
        blockers.append("circularity_guard_hash_missing")
    return _receipt(LOAD_NO_CIRCULARITY_RECEIPT, blockers, forbidden_inputs=bad_inputs)


def verify_source_maxent_release_law(artifact: Any) -> dict[str, Any]:
    data = _mapping(artifact)
    blockers = _required(data, ("artifact_id", "regulator_id"))
    blockers.extend(
        _required_hashes(
            data,
            (
                "quotient_space_hash",
                "base_weight_hash",
                "action_hash",
                "constraint_units_hash",
                "zero_mode_policy_hash",
                "sector_policy_hash",
                "lagrange_multiplier_hash",
                "release_law_hash",
                "no_data_ledger_hash",
                "refinement_map_hash",
            ),
        )
    )
    if not _nonempty_sequence(data.get("constraint_observable_hashes")):
        blockers.append("constraint_observable_hashes_missing")
    if not _nonempty_sequence(data.get("constraint_target_hashes")):
        blockers.append("constraint_target_hashes_missing")
    bad_inputs = _forbidden_inputs(data)
    if bad_inputs:
        blockers.append("release_law_reads_observational_data")
    return _receipt(SOURCE_MAXENT_RELEASE_LAW_RECEIPT, blockers, forbidden_inputs=bad_inputs)


def verify_abundance_selector_no_data_use(artifact: Any) -> dict[str, Any]:
    data = _mapping(artifact)
    blockers: list[str] = []
    if not _hashish(data.get("no_data_use_receipt_hash")):
        blockers.append("no_data_use_receipt_hash_missing")
    bad_inputs = _forbidden_inputs(data)
    if bad_inputs:
        blockers.append("selector_source_dag_reads_observational_data")
    return _receipt(LOAD_NO_DATA_USE_RECEIPT, blockers, forbidden_inputs=bad_inputs)


def verify_load_refinement_compatibility(
    artifact: Any,
    *,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    data = _mapping(artifact)
    blockers = _required(data, ("artifact_id", "fine_regulator_id", "coarse_regulator_id"))
    blockers.extend(
        _required_hashes(
            data,
            ("coarse_map_hash", "fine_release_law_hash", "coarse_release_law_hash"),
        )
    )
    law_tv = _float(data.get("law_tv_defect"))
    load_defect = _float(data.get("load_naturality_defect"))
    load_sup = _float(data.get("load_sup_bound"))
    selected = _float(data.get("selected_load_difference_bound"))
    if law_tv is None or law_tv < 0.0:
        blockers.append("law_tv_defect_invalid")
    if load_defect is None or load_defect < 0.0:
        blockers.append("load_naturality_defect_invalid")
    if load_sup is None or load_sup < 0.0:
        blockers.append("load_sup_bound_invalid")
    if selected is None or selected < 0.0:
        blockers.append("selected_load_difference_invalid")
    computed_bound = None
    if law_tv is not None and load_defect is not None and load_sup is not None:
        computed_bound = float(load_defect) + 2.0 * float(load_sup) * float(law_tv)
    if bool(data.get("passes_exact", False)):
        if (law_tv or 0.0) > float(tolerance) or (load_defect or 0.0) > float(tolerance):
            blockers.append("exact_refinement_defect_nonzero")
    elif selected is not None and computed_bound is not None and selected > computed_bound + float(tolerance):
        blockers.append("load_refinement_bound_exceeded")
    passed = not blockers
    return {
        LOAD_REFINEMENT_COMPATIBILITY_RECEIPT: passed,
        "passed": passed,
        "blockers": _dedupe(blockers),
        "computed_bound": computed_bound,
        "selected_load_difference_bound": selected,
    }


def verify_anomaly_abundance_source_receipt(bundle: Any) -> dict[str, Any]:
    data = _mapping(bundle)
    release_state = data.get("release_state")
    load_observable = data.get("load_observable")
    release_law = data.get("release_law")
    selector = data.get("selector")
    refinement = data.get("refinement")
    primitive_reports = [
        verify_anomaly_release_state(release_state),
        verify_anomaly_load_observable(load_observable),
        verify_source_maxent_release_law(release_law),
        verify_load_quotient_invariance(load_observable),
        verify_load_no_circularity(load_observable),
        verify_abundance_selector_no_data_use(selector),
        verify_load_refinement_compatibility(refinement),
    ]
    selector_report = _verify_selector_artifact(selector)
    primitive_reports.append(selector_report)
    blockers = [blocker for report in primitive_reports for blocker in report.get("blockers", [])]
    anomaly_receipt = not blockers
    transport_receipt = bool(
        data.get(RHO_A_TRANSPORT_RECEIPT, False)
        or data.get("rho_A_transport_receipt", False)
    )
    rho_source_receipt = bool(anomaly_receipt and transport_receipt)
    selector_label = _value(selector, "source_claim_label")
    if selector_label == SOURCE_ONLY_ANOMALY_ABUNDANCE and not rho_source_receipt:
        blockers.append("source_label_without_rho_A_source_receipt")
        anomaly_receipt = False
        rho_source_receipt = False
    claim_label = (
        SOURCE_ONLY_ANOMALY_ABUNDANCE
        if rho_source_receipt
        else PHYSICAL_PARENT_WITH_CONDITIONAL_ABUNDANCE
        if transport_receipt
        else CONDITIONAL_SOURCE_STATE
    )
    return {
        ANOMALY_ABUNDANCE_SOURCE_RECEIPT: anomaly_receipt,
        RHO_A_TRANSPORT_RECEIPT: transport_receipt,
        RHO_A_SOURCE_RECEIPT: rho_source_receipt,
        "passed": anomaly_receipt,
        "claim_label": claim_label,
        "blockers": _dedupe(blockers),
        "primitive_reports": primitive_reports,
    }


def compute_selector(release_law: Sequence[float], load_values: Sequence[float]) -> float:
    weights = [float(value) for value in release_law]
    loads = [float(value) for value in load_values]
    if not weights or len(weights) != len(loads):
        raise ValueError("release_law and load_values must have the same nonzero length")
    if not all(math.isfinite(value) and value >= 0.0 for value in weights):
        raise ValueError("release_law weights must be finite and nonnegative")
    if not all(math.isfinite(value) for value in loads):
        raise ValueError("load_values must be finite")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("release_law must have positive total weight")
    return sum(weight * load for weight, load in zip(weights, loads, strict=True)) / total


def compute_anomaly_load(load_artifact: Any, release_state: Any | None = None, parent: Any | None = None) -> float:
    data = _mapping(load_artifact)
    parent_data = _mapping(parent or {})
    if bool(parent_data.get("amplitude_loaded", False)):
        raise ValueError("anomaly load must be computed from an amplitude-free source parent")
    values = data.get("load_values")
    if values is None and release_state is not None:
        values = _mapping(release_state).get("local_load_values")
    loads = [float(value) for value in _sequence(values)]
    if not loads or not all(math.isfinite(value) for value in loads):
        raise ValueError("load_values must be a nonempty finite sequence")
    return float(sum(loads))


def _verify_selector_artifact(artifact: Any) -> dict[str, Any]:
    data = _mapping(artifact)
    blockers = _required(
        data,
        (
            "artifact_id",
            "regulator_id",
            "parent_generation_id",
            "release_state_artifact_id",
            "release_law_artifact_id",
            "load_observable_artifact_id",
        ),
    )
    blockers.extend(
        _required_hashes(
            data,
            (
                "quotient_ensemble_hash",
                "P_star_hash",
                "N_CRC_hash",
                "selected_load_energy_hash",
                "rho_A_bar_hash",
                "refinement_residual_hash",
                "no_data_use_receipt_hash",
            ),
        )
    )
    if data.get("source_claim_label") not in {
        PHYSICAL_PARENT_WITH_CONDITIONAL_ABUNDANCE,
        SOURCE_ONLY_ANOMALY_ABUNDANCE,
    }:
        blockers.append("source_claim_label_invalid")
    return _receipt("ANOMALY_ABUNDANCE_SELECTOR_ARTIFACT_RECEIPT", blockers)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {key: getattr(value, key) for key in dir(value) if not key.startswith("_")}


def _value(value: Any, key: str) -> Any:
    if is_dataclass(value):
        return getattr(value, key, None)
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _required(data: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    return [f"{key}_missing" for key in keys if not _nonempty(data.get(key))]


def _required_hashes(data: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    return [f"{key}_missing" for key in keys if not _hashish(data.get(key))]


def _receipt(name: str, blockers: Sequence[str], **extra: Any) -> dict[str, Any]:
    passed = not blockers
    return {name: passed, "passed": passed, "blockers": _dedupe(blockers), **extra}


def _hashish(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return bool(len(stripped) >= 8 and stripped not in {"0" * len(stripped), "x" * len(stripped)})


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonempty_sequence(value: Any) -> bool:
    return bool(_sequence(value))


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _as_strings(values: Sequence[Any]) -> list[str]:
    return [str(value).strip().lower() for value in values if value is not None and str(value).strip()]


def _matching_tokens(values: Sequence[str], tokens: Sequence[str]) -> list[str]:
    matches = []
    lowered = [str(value).lower() for value in values]
    for token in tokens:
        token_l = token.lower()
        if any(token_l in value for value in lowered):
            matches.append(token)
    return _dedupe(matches)


def _forbidden_inputs(data: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("source_dag_inputs", "forbidden_inputs", "dependency_inputs", "observational_inputs"):
        values.extend(_sequence(data.get(key)))
    return _matching_tokens(_as_strings(values), FORBIDDEN_DATA_TOKENS)


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
