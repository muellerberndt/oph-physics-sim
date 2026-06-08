from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from oph_fpe.claims import (
    CONTINUATION,
    OPH_STATIC_GALAXY_BRIDGE_RECEIPT,
    PROXY,
    STATIC_GALAXY_RAR_BTFR_RECEIPT,
    with_claim_metadata,
)
from oph_fpe.cosmology.galaxy_proxy import DEFAULT_A0_OPH, btfr_summary, rar_curve


KPC_IN_M = 3.0856775814913673e19
KM2_IN_M2 = 1.0e6
G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30


@dataclass(frozen=True)
class StaticGalaxyDataset:
    rows: list[dict[str, float | str]]
    source_paths: list[str]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def galaxy_count(self) -> int:
        names = {str(row.get("galaxy", "")) for row in self.rows if str(row.get("galaxy", ""))}
        return len(names) if names else int(self.row_count > 0)


def load_static_galaxy_dataset(path: Path) -> StaticGalaxyDataset:
    """Load SPARC/RAR-like rows from CSV files or official SPARC MRT tables."""

    input_path = Path(path)
    files = [input_path] if input_path.is_file() else sorted([*input_path.glob("*.csv"), *input_path.glob("*.mrt")])
    rows: list[dict[str, float | str]] = []
    for file_path in files:
        if file_path.suffix.lower() == ".mrt":
            rows.extend(_load_sparc_mrt_rows(file_path))
        else:
            with file_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for raw in reader:
                    parsed: dict[str, float | str] = {}
                    for key, value in raw.items():
                        if key is None:
                            continue
                        normalized_key = _normalize_key(key)
                        text = str(value or "").strip()
                        if not text:
                            continue
                        if normalized_key in {"galaxy", "name", "id"}:
                            parsed["galaxy"] = text
                        else:
                            try:
                                parsed[normalized_key] = float(text)
                            except ValueError:
                                parsed[normalized_key] = text
                    if parsed:
                        rows.append(parsed)
    return StaticGalaxyDataset(rows=rows, source_paths=[str(path) for path in files])


def static_galaxy_measurement_report(
    dataset: StaticGalaxyDataset,
    *,
    a0_initial: float = DEFAULT_A0_OPH,
    lambda_initial: float = 1.0,
    min_points: int = 12,
    min_galaxies: int = 1,
    physical_claim: bool = True,
) -> dict[str, Any]:
    rar = _rar_arrays(dataset)
    btfr = _btfr_arrays(dataset)
    if rar is None:
        report = {
            "mode": "oph_static_galaxy_measurement_fit",
            "bridge": "static_galaxy",
            "bridge_name": "OPH Continuation Effective Theory / static galaxy",
            "claim_tier": "Tier0_diagnostic_proxy_no_measurement_fit",
            "allowed_free_parameters": ["a0", "lambda_collar"],
            "calibrated_on": [],
            "tested_on": [],
            "bulk_required": False,
            "full_bulk_required": False,
            "physical_cmb_claim": False,
            "physical_matter_power_claim": False,
            "dataset_row_count": dataset.row_count,
            "dataset_galaxy_count": dataset.galaxy_count,
            "galaxy_count": dataset.galaxy_count,
            "measurement_galaxy_count": dataset.galaxy_count,
            "source_paths": dataset.source_paths,
            "STATIC_GALAXY_RAR_BTFR_RECEIPT": False,
            "OPH_STATIC_GALAXY_BRIDGE_RECEIPT": False,
            "bridge_receipt": False,
            "bridge_receipt_name": OPH_STATIC_GALAXY_BRIDGE_RECEIPT,
            "receipt": False,
            "reason": "no usable RAR acceleration rows or SPARC-style velocity rows",
            "physical_claim": False,
            "bridge_prediction_boundary": (
                "No external acceleration/velocity rows were supplied, so the OPH-CET static bridge "
                "has not made a measurement-facing comparison."
            ),
            "claim_boundary": (
                "No external acceleration/velocity rows were supplied; no measurement-facing galaxy fit. "
                "This lane does not require the populated 3D-bulk gate and is not a CMB prediction."
            ),
        }
        return with_claim_metadata(
            report,
            claim_level=PROXY,
            receipt=STATIC_GALAXY_RAR_BTFR_RECEIPT,
            physical_claim=False,
            observable_id="external_static_galaxy_dataset",
            fit_objective="rar_btfr_external_fit",
        )

    fit = _fit_shared_rar(rar["g_baryon"], rar["g_observed"], a0_initial=a0_initial, lambda_initial=lambda_initial)
    predicted = rar_curve(rar["g_baryon"], a0_oph=fit["a0"], lambda_collar=fit["lambda_collar"])
    log_residual = np.log10(np.maximum(predicted, 1e-300)) - np.log10(np.maximum(rar["g_observed"], 1e-300))
    scatter = float(np.sqrt(np.mean(log_residual * log_residual))) if log_residual.size else None
    btfr_report = (
        btfr_summary(btfr["baryonic_mass"], btfr["flat_velocity"])
        if btfr is not None
        else {"usable": False, "reason": "btfr_columns_not_provided"}
    )
    btfr_prediction = btfr_prediction_from_rar_fit(
        a0=float(fit["a0"]),
        lambda_collar=float(fit["lambda_collar"]),
        observed_btfr=btfr_report,
    )
    holdout_validation = static_galaxy_holdout_report(
        dataset,
        a0_initial=a0_initial,
        lambda_initial=lambda_initial,
    )
    btfr_galaxy_count = int(btfr_report.get("galaxy_count") or 0) if btfr_report.get("usable") else 0
    measurement_galaxy_count = max(int(dataset.galaxy_count), int(rar["galaxy_count"]), btfr_galaxy_count)
    rar_galaxy_support_count = int(rar["galaxy_count"])
    if (
        rar_galaxy_support_count <= 1
        and str(rar.get("source") or "") == "direct_acceleration_columns"
        and int(measurement_galaxy_count) > 1
    ):
        # Official SPARC RAR rows are aggregate acceleration points without
        # per-row galaxy labels.  Keep the raw RAR label count visible, but
        # let the measurement-support gate use the named companion SPARC
        # BTFR/mass-model rows when they are loaded in the same dataset.
        rar_galaxy_support_count = int(measurement_galaxy_count)
    receipt = bool(
        int(rar["point_count"]) >= int(min_points)
        and int(rar_galaxy_support_count) >= int(min_galaxies)
        and scatter is not None
        and scatter < 0.25
    )
    report = {
        "mode": "oph_static_galaxy_measurement_fit",
        "bridge": "static_galaxy",
        "bridge_name": "OPH Continuation Effective Theory / static galaxy",
        "claim_tier": "Tier1_phenomenological_continuation",
        "allowed_free_parameters": ["a0", "lambda_collar"],
        "calibrated_on": [str(rar.get("source") or "RAR_acceleration_rows")],
        "tested_on": ["SPARC_BTFR"] if bool(btfr_report.get("usable")) else [],
        "bulk_required": False,
        "full_bulk_required": False,
        "physical_cmb_claim": False,
        "physical_matter_power_claim": False,
        "dataset_row_count": dataset.row_count,
        "dataset_galaxy_count": int(dataset.galaxy_count),
        "galaxy_count": int(measurement_galaxy_count),
        "measurement_galaxy_count": int(measurement_galaxy_count),
        "rar_galaxy_count": int(rar["galaxy_count"]),
        "rar_galaxy_support_count": int(rar_galaxy_support_count),
        "rar_point_count": int(rar["point_count"]),
        "rar_source": str(rar.get("source") or "unknown"),
        "btfr_galaxy_count": int(btfr_galaxy_count),
        "source_paths": dataset.source_paths,
        "STATIC_GALAXY_RAR_BTFR_RECEIPT": receipt,
        "OPH_STATIC_GALAXY_BRIDGE_RECEIPT": receipt,
        "bridge_receipt": receipt,
        "bridge_receipt_name": OPH_STATIC_GALAXY_BRIDGE_RECEIPT,
        "receipt": receipt,
        "shared_a0": float(fit["a0"]),
        "shared_lambda_collar": float(fit["lambda_collar"]),
        "rar_scatter_dex": scatter,
        "chi2_proxy": float(fit["loss"]),
        "btfr": btfr_report,
        "btfr_prediction_from_rar_fit": btfr_prediction,
        "holdout_validation": holdout_validation,
        "physical_claim": bool(physical_claim and receipt),
        "bridge_prediction_boundary": (
            "Tier-1 OPH-CET static-galaxy bridge: the OPH continuation law is calibrated on RAR "
            "acceleration rows with free a0 and lambda_collar, then checked against the BTFR table "
            "when available. This is measurement-facing phenomenology, not a recovered-chain CMB "
            "prediction and not evidence for populated 3D bulk."
        ),
        "claim_boundary": (
            "External static-galaxy RAR/BTFR fit for the OPH continuation law. This is a relaxed-galaxy "
            "measurement lane only; it is not a 3D bulk proof, not CMB, and not a dynamic cluster/cosmology claim."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=CONTINUATION,
        receipt=STATIC_GALAXY_RAR_BTFR_RECEIPT,
        physical_claim=bool(physical_claim and receipt),
        observable_id="external_static_galaxy_dataset",
        fit_objective="shared_a0_lambda_rar_fit",
    )


def static_galaxy_holdout_report(
    dataset: StaticGalaxyDataset,
    *,
    a0_initial: float = DEFAULT_A0_OPH,
    lambda_initial: float = 1.0,
    train_fraction: float = 0.7,
    seed: int = 1729,
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.7,
    min_train_galaxies: int = 8,
    min_test_galaxies: int = 4,
) -> dict[str, Any]:
    rows = _mass_model_acceleration_rows(
        dataset,
        upsilon_disk=float(upsilon_disk),
        upsilon_bulge=float(upsilon_bulge),
    )
    if rows is None:
        return {
            "usable": False,
            "reason": "no usable SPARC mass-model rotation rows",
            "claim_boundary": (
                "No galaxy-level holdout was computed. The RAR fit may still be usable if direct "
                "acceleration rows are present."
            ),
        }
    galaxies = np.asarray(sorted(set(str(name) for name in rows["galaxy"] if str(name))), dtype=object)
    if galaxies.size < min_train_galaxies + min_test_galaxies:
        return {
            "usable": False,
            "reason": "not enough galaxies for train/test split",
            "galaxy_count": int(galaxies.size),
            "min_train_galaxies": int(min_train_galaxies),
            "min_test_galaxies": int(min_test_galaxies),
            "claim_boundary": "Galaxy-level holdout requires enough distinct named galaxies.",
        }
    rng = np.random.default_rng(int(seed))
    shuffled = galaxies.copy()
    rng.shuffle(shuffled)
    train_count = int(round(float(train_fraction) * float(shuffled.size)))
    train_count = max(int(min_train_galaxies), min(train_count, int(shuffled.size) - int(min_test_galaxies)))
    train_galaxies = set(str(name) for name in shuffled[:train_count])
    train_mask = np.asarray([str(name) in train_galaxies for name in rows["galaxy"]], dtype=bool)
    test_mask = ~train_mask
    fit = _fit_shared_rar(
        rows["g_baryon"][train_mask],
        rows["g_observed"][train_mask],
        a0_initial=float(a0_initial),
        lambda_initial=float(lambda_initial),
    )
    train_metrics = _mass_model_metrics(rows, train_mask, fit["a0"], fit["lambda_collar"])
    test_metrics = _mass_model_metrics(rows, test_mask, fit["a0"], fit["lambda_collar"])
    receipt = bool(
        test_metrics.get("usable")
        and test_metrics.get("log_acceleration_rmse_dex") is not None
        and float(test_metrics["log_acceleration_rmse_dex"]) < 0.35
        and float(test_metrics.get("velocity_rmse_improvement_fraction") or 0.0) > 0.05
    )
    return {
        "usable": True,
        "mode": "galaxy_level_train_test_mass_model_holdout",
        "receipt": receipt,
        "claim_tier": "Tier1_phenomenological_continuation_holdout",
        "split_seed": int(seed),
        "train_fraction": float(train_fraction),
        "fixed_upsilon_disk": float(upsilon_disk),
        "fixed_upsilon_bulge": float(upsilon_bulge),
        "allowed_free_parameters": ["a0", "lambda_collar"],
        "fit_on": "SPARC_MassModels_Lelli2016c_train_galaxies",
        "tested_on": "SPARC_MassModels_Lelli2016c_heldout_galaxies",
        "galaxy_count": int(galaxies.size),
        "train_galaxy_count": int(len(train_galaxies)),
        "test_galaxy_count": int(galaxies.size - len(train_galaxies)),
        "train_point_count": int(np.sum(train_mask)),
        "test_point_count": int(np.sum(test_mask)),
        "shared_a0": float(fit["a0"]),
        "shared_lambda_collar": float(fit["lambda_collar"]),
        "train": train_metrics,
        "test": test_metrics,
        "bulk_required": False,
        "physical_cmb_claim": False,
        "claim_boundary": (
            "Galaxy-level Tier-1 holdout on SPARC mass-model rotation rows with fixed stellar "
            "mass-to-light assumptions and only shared a0/lambda_collar fitted on train galaxies. "
            "This is a static measurement bridge, not a full galaxy likelihood, not CMB, and not "
            "a populated 3D-bulk proof."
        ),
    }


def write_static_galaxy_measurement_report(
    dataset_path: Path,
    out: Path,
    *,
    a0_initial: float = DEFAULT_A0_OPH,
    lambda_initial: float = 1.0,
    min_points: int = 12,
    min_galaxies: int = 1,
    physical_claim: bool = True,
) -> dict[str, Any]:
    dataset = load_static_galaxy_dataset(dataset_path)
    report = static_galaxy_measurement_report(
        dataset,
        a0_initial=float(a0_initial),
        lambda_initial=float(lambda_initial),
        min_points=int(min_points),
        min_galaxies=int(min_galaxies),
        physical_claim=bool(physical_claim),
    )
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "static_galaxy_measurement_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    return report


def _mass_model_acceleration_rows(
    dataset: StaticGalaxyDataset,
    *,
    upsilon_disk: float,
    upsilon_bulge: float,
) -> dict[str, np.ndarray] | None:
    galaxies: list[str] = []
    radius_kpc: list[float] = []
    g_baryon: list[float] = []
    g_observed: list[float] = []
    v_observed: list[float] = []
    e_v_observed: list[float] = []
    for row in dataset.rows:
        if str(row.get("source_table", "")) != "SPARC_MassModels_Lelli2016c" and not {
            "radius_kpc",
            "v_obs",
            "v_gas",
            "v_disk",
        }.issubset(row):
            continue
        converted = _sparc_velocity_accelerations(
            {
                **row,
                "upsilon_disk": float(upsilon_disk),
                "upsilon_bulge": float(upsilon_bulge),
            }
        )
        if converted is None:
            continue
        radius = _float_value(row, "radius_kpc")
        v_obs = _float_value(row, "v_obs")
        if radius is None or v_obs is None or radius <= 0.0 or v_obs <= 0.0:
            continue
        galaxies.append(str(row.get("galaxy") or f"row_{len(galaxies)}"))
        radius_kpc.append(float(radius))
        g_baryon.append(float(converted[0]))
        g_observed.append(float(converted[1]))
        v_observed.append(float(v_obs))
        e_v_observed.append(float(max(_float_value(row, "e_v_obs") or 1.0, 1.0)))
    if not galaxies:
        return None
    return {
        "galaxy": np.asarray(galaxies, dtype=object),
        "radius_kpc": np.asarray(radius_kpc, dtype=float),
        "g_baryon": np.asarray(g_baryon, dtype=float),
        "g_observed": np.asarray(g_observed, dtype=float),
        "v_observed": np.asarray(v_observed, dtype=float),
        "e_v_observed": np.asarray(e_v_observed, dtype=float),
    }


def _mass_model_metrics(
    rows: dict[str, np.ndarray],
    mask: np.ndarray,
    a0: float,
    lambda_collar: float,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    if int(np.sum(selected)) == 0:
        return {"usable": False, "reason": "empty split"}
    gb = rows["g_baryon"][selected]
    go = rows["g_observed"][selected]
    radius_m = rows["radius_kpc"][selected] * KPC_IN_M
    v_obs = rows["v_observed"][selected]
    e_v_obs = rows["e_v_observed"][selected]
    pred_g = rar_curve(gb, a0_oph=float(a0), lambda_collar=float(lambda_collar))
    baryon_v = np.sqrt(np.maximum(gb * radius_m, 0.0)) / 1000.0
    pred_v = np.sqrt(np.maximum(pred_g * radius_m, 0.0)) / 1000.0
    log_resid = np.log10(np.maximum(pred_g, 1e-300)) - np.log10(np.maximum(go, 1e-300))
    baryon_log_resid = np.log10(np.maximum(gb, 1e-300)) - np.log10(np.maximum(go, 1e-300))
    velocity_resid = pred_v - v_obs
    baryon_velocity_resid = baryon_v - v_obs
    velocity_rmse = float(np.sqrt(np.mean(velocity_resid * velocity_resid)))
    baryon_velocity_rmse = float(np.sqrt(np.mean(baryon_velocity_resid * baryon_velocity_resid)))
    weighted_chi2 = float(np.mean((velocity_resid / np.maximum(e_v_obs, 1.0)) ** 2))
    galaxies = {str(name) for name in rows["galaxy"][selected] if str(name)}
    improvement = (
        1.0 - velocity_rmse / max(baryon_velocity_rmse, 1e-300)
        if baryon_velocity_rmse > 0.0
        else None
    )
    return {
        "usable": True,
        "galaxy_count": int(len(galaxies)),
        "point_count": int(np.sum(selected)),
        "log_acceleration_rmse_dex": float(np.sqrt(np.mean(log_resid * log_resid))),
        "baryon_only_log_acceleration_rmse_dex": float(np.sqrt(np.mean(baryon_log_resid * baryon_log_resid))),
        "velocity_rmse_km_s": velocity_rmse,
        "baryon_only_velocity_rmse_km_s": baryon_velocity_rmse,
        "velocity_rmse_improvement_fraction": improvement,
        "mean_abs_velocity_residual_km_s": float(np.mean(np.abs(velocity_resid))),
        "velocity_chi2_proxy_per_point": weighted_chi2,
    }


def _fit_shared_rar(g_baryon: np.ndarray, g_observed: np.ndarray, *, a0_initial: float, lambda_initial: float) -> dict[str, float]:
    gb = np.asarray(g_baryon, dtype=float)
    go = np.asarray(g_observed, dtype=float)
    mask = (gb > 0.0) & (go > 0.0) & np.isfinite(gb) & np.isfinite(go)
    gb = gb[mask]
    go = go[mask]
    log_go = np.log(np.maximum(go, 1e-300))

    def loss(theta: np.ndarray) -> float:
        log_a0, log_lambda = theta
        predicted = rar_curve(gb, a0_oph=float(np.exp(log_a0)), lambda_collar=float(np.exp(log_lambda)))
        residual = np.log(np.maximum(predicted, 1e-300)) - log_go
        return float(np.mean(residual * residual))

    result = minimize(
        loss,
        np.asarray([np.log(max(float(a0_initial), 1e-30)), np.log(max(float(lambda_initial), 1e-12))], dtype=float),
        method="Nelder-Mead",
        options={"maxiter": 512},
    )
    log_a0, log_lambda = result.x
    return {
        "a0": float(np.exp(log_a0)),
        "lambda_collar": float(np.exp(log_lambda)),
        "loss": float(result.fun),
    }


def btfr_prediction_from_rar_fit(
    *,
    a0: float,
    lambda_collar: float,
    observed_btfr: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Asymptotic BTFR implied by the fitted OPH RAR continuation law.

    In the low-acceleration limit of nu_OPH, g_obs ~= sqrt(a0*g_b)/lambda.
    For a flat circular orbit this gives v^4 = G M_b a0 / lambda^2.
    The intercept below assumes M_b is in solar masses and v is in km/s.
    """

    a0_value = max(float(a0), 1e-300)
    lam = max(float(lambda_collar), 1e-300)
    intercept = float(np.log10((lam * lam * 1000.0**4) / (G_SI * M_SUN_KG * a0_value)))
    result: dict[str, Any] = {
        "usable": True,
        "asymptotic_relation": "log10(M_b/Msun) = 4 log10(v_flat/km_s) + intercept",
        "predicted_slope_logM_vs_logV": 4.0,
        "predicted_intercept_logM_vs_logV": intercept,
        "a0": float(a0_value),
        "lambda_collar": float(lam),
        "claim_boundary": (
            "Asymptotic low-acceleration BTFR implied by the fitted static OPH RAR bridge; "
            "not a full SPARC likelihood or galaxy-by-galaxy mass model."
        ),
    }
    if observed_btfr and observed_btfr.get("usable"):
        observed_slope = observed_btfr.get("slope_logM_vs_logV")
        observed_intercept = observed_btfr.get("intercept_logM_vs_logV")
        if observed_slope is not None:
            slope_delta = float(observed_slope) - 4.0
            result["observed_slope_logM_vs_logV"] = float(observed_slope)
            result["slope_delta_observed_minus_predicted"] = slope_delta
            result["abs_slope_delta"] = abs(slope_delta)
        if observed_intercept is not None:
            intercept_delta = float(observed_intercept) - intercept
            result["observed_intercept_logM_vs_logV"] = float(observed_intercept)
            result["intercept_delta_observed_minus_predicted"] = intercept_delta
            result["abs_intercept_delta_dex"] = abs(intercept_delta)
        result["observed_rms_dex"] = observed_btfr.get("rms_dex")
        result["observed_galaxy_count"] = observed_btfr.get("galaxy_count")
    return result


def _rar_arrays(dataset: StaticGalaxyDataset) -> dict[str, Any] | None:
    direct_gb: list[float] = []
    direct_go: list[float] = []
    direct_galaxies: list[str] = []
    velocity_gb: list[float] = []
    velocity_go: list[float] = []
    velocity_galaxies: list[str] = []
    for row in dataset.rows:
        galaxy = str(row.get("galaxy", ""))
        gb = _float_value(row, "g_baryon")
        go = _float_value(row, "g_observed")
        if gb is not None and go is not None and gb > 0.0 and go > 0.0:
            direct_gb.append(gb)
            direct_go.append(go)
            direct_galaxies.append(galaxy)
            continue
        converted = _sparc_velocity_accelerations(row)
        if converted is not None:
            velocity_gb.append(converted[0])
            velocity_go.append(converted[1])
            velocity_galaxies.append(galaxy)
    if direct_gb:
        galaxies = {name for name in direct_galaxies if name}
        return {
            "g_baryon": np.asarray(direct_gb, dtype=float),
            "g_observed": np.asarray(direct_go, dtype=float),
            "point_count": len(direct_gb),
            "galaxy_count": len(galaxies) if galaxies else int(bool(direct_gb)),
            "source": "direct_acceleration_columns",
        }
    if velocity_gb:
        galaxies = {name for name in velocity_galaxies if name}
        return {
            "g_baryon": np.asarray(velocity_gb, dtype=float),
            "g_observed": np.asarray(velocity_go, dtype=float),
            "point_count": len(velocity_gb),
            "galaxy_count": len(galaxies) if galaxies else int(bool(velocity_gb)),
            "source": "sparc_velocity_columns",
        }
    return None


def _btfr_arrays(dataset: StaticGalaxyDataset) -> dict[str, np.ndarray] | None:
    masses: list[float] = []
    velocities: list[float] = []
    seen: set[str] = set()
    for row in dataset.rows:
        galaxy = str(row.get("galaxy", len(seen)))
        if galaxy in seen:
            continue
        mass = _float_value(row, "baryonic_mass")
        velocity = _float_value(row, "flat_velocity")
        if mass is not None and velocity is not None and mass > 0.0 and velocity > 0.0:
            masses.append(mass)
            velocities.append(velocity)
            seen.add(galaxy)
    if not masses:
        return None
    return {"baryonic_mass": np.asarray(masses, dtype=float), "flat_velocity": np.asarray(velocities, dtype=float)}


def _load_sparc_mrt_rows(path: Path) -> list[dict[str, float | str]]:
    """Parse the official SPARC MRT tables used by the measurement lane.

    The SPARC site publishes plain MRT files with byte-by-byte headers.  This
    parser intentionally handles only the tables needed here: RAR.mrt,
    BTFR_Lelli2019.mrt, and MassModels_Lelli2016c.mrt.  Unknown MRT files are
    ignored rather than guessed from arbitrary fixed-width metadata.
    """

    name = path.name.lower()
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = [line.rstrip("\n") for line in handle]
    if name == "rar.mrt":
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                log_gbar = float(parts[0])
                e_log_gbar = float(parts[1])
                log_gobs = float(parts[2])
                e_log_gobs = float(parts[3])
            except ValueError:
                continue
            rows.append(
                {
                    "g_baryon": float(10.0 ** log_gbar),
                    "g_observed": float(10.0 ** log_gobs),
                    "e_log_g_baryon": e_log_gbar,
                    "e_log_g_observed": e_log_gobs,
                    "source_table": "SPARC_RAR_Lelli2017",
                }
            )
        return rows
    if name == "btfr_lelli2019.mrt":
        for line in lines:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                log_mass = float(parts[1])
                flat_velocity = float(parts[5])
            except ValueError:
                continue
            if flat_velocity <= 0.0:
                continue
            rows.append(
                {
                    "galaxy": str(parts[0]),
                    "baryonic_mass": float(10.0 ** log_mass),
                    "flat_velocity": flat_velocity,
                    "source_table": "SPARC_BTFR_Lelli2019",
                }
            )
        return rows
    if name == "massmodels_lelli2016c.mrt":
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                radius = float(parts[2])
                v_obs = float(parts[3])
                e_v_obs = float(parts[4])
                v_gas = float(parts[5])
                v_disk = float(parts[6])
                v_bulge = float(parts[7])
            except ValueError:
                continue
            if radius <= 0.0 or v_obs <= 0.0:
                continue
            rows.append(
                {
                    "galaxy": str(parts[0]),
                    "radius_kpc": radius,
                    "v_obs": v_obs,
                    "e_v_obs": e_v_obs,
                    "v_gas": v_gas,
                    "v_disk": v_disk,
                    "v_bulge": v_bulge,
                    "source_table": "SPARC_MassModels_Lelli2016c",
                }
            )
        return rows
    return rows


def _sparc_velocity_accelerations(row: dict[str, float | str]) -> tuple[float, float] | None:
    radius_kpc = _float_value(row, "radius_kpc") or _float_value(row, "r_kpc") or _float_value(row, "r")
    v_obs = _float_value(row, "v_obs") or _float_value(row, "vobs")
    v_gas = _float_value(row, "v_gas") or _float_value(row, "vgas") or 0.0
    v_disk = _float_value(row, "v_disk") or _float_value(row, "vdisk") or 0.0
    v_bulge = _float_value(row, "v_bulge") or _float_value(row, "vbul") or _float_value(row, "v_bul") or 0.0
    upsilon_disk = _float_value(row, "upsilon_disk") or 0.5
    upsilon_bulge = _float_value(row, "upsilon_bulge") or 0.7
    if radius_kpc is None or v_obs is None or radius_kpc <= 0.0 or v_obs <= 0.0:
        return None
    radius_m = float(radius_kpc) * KPC_IN_M
    vbar2 = abs(v_gas) * v_gas + upsilon_disk * abs(v_disk) * v_disk + upsilon_bulge * abs(v_bulge) * v_bulge
    if vbar2 <= 0.0:
        return None
    g_baryon = float(vbar2) * KM2_IN_M2 / radius_m
    g_observed = float(v_obs * v_obs) * KM2_IN_M2 / radius_m
    return g_baryon, g_observed


def _normalize_key(key: str) -> str:
    text = str(key).strip().lower()
    for char in " -./()[]":
        text = text.replace(char, "_")
    while "__" in text:
        text = text.replace("__", "_")
    aliases = {
        "galaxy_id": "galaxy",
        "name": "galaxy",
        "galaxy_name": "galaxy",
        "rad": "radius_kpc",
        "radius": "radius_kpc",
        "vobs": "v_obs",
        "evobs": "e_v_obs",
        "vgas": "v_gas",
        "vdisk": "v_disk",
        "vbul": "v_bulge",
        "vbulge": "v_bulge",
        "mbar": "baryonic_mass",
        "m_baryonic": "baryonic_mass",
        "vflat": "flat_velocity",
        "v_flat": "flat_velocity",
        "gb": "g_baryon",
        "gbar": "g_baryon",
        "gobs": "g_observed",
        "gtot": "g_observed",
    }
    return aliases.get(text.strip("_"), text.strip("_"))


def _float_value(row: dict[str, float | str], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result
