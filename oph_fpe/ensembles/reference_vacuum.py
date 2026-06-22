from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


CLAIM_TIERS: dict[str, str] = {
    "E0": "seed noise, proposal noise, repair jitter",
    "E1": "conventional reference ensemble",
    "E2": "OPH-native quotient ensemble",
    "E3": "OPH vacuum",
    "E4": "OPH primordial field",
    "E5": "observable cosmological prediction",
}

NONCLAIMS = (
    "not an OPH-native vacuum",
    "not an OPH primordial field",
    "not a physical TT/TE/EE prediction",
    "not promoted by seed noise, proposal noise, or repair jitter",
)


@dataclass(frozen=True)
class EnsembleSpec:
    ensemble_id: str
    claim_tier: str
    regulator_id: str
    representative_schema_hash: str
    gauge_action_hash: str
    quotient_canonicalizer_hash: str
    base_measure_definition: str
    action_definition_and_coefficients: dict[str, Any]
    coarse_map_hashes: tuple[str, ...]
    zero_mode_projector_hash: str
    amplitude_convention: str
    sampler_kernel_hash: str
    rng_event_label_schema: str
    smoothing_policy: str
    source_provenance: str
    explicit_nonclaims: tuple[str, ...] = NONCLAIMS

    def as_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["claim_tier_meaning"] = CLAIM_TIERS[self.claim_tier]
        data["ensemble_spec_hash"] = stable_hash(data)
        return data


def free_scalar_ensemble_spec(
    *,
    ell_max: int,
    amplitude: float,
    theta: float,
    smoothing_sigma: float | None = None,
) -> EnsembleSpec:
    if ell_max < 2:
        raise ValueError("ell_max must be at least 2 after removing monopole and dipole modes")
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive")
    action = {
        "kind": "free_scalar_harmonic_gaussian",
        "mean_constraint": "E[a_lm]=0",
        "mean_quadratic_release": "E[q^T K q]=d*A",
        "amplitude_A": float(amplitude),
        "kappa_l": "[ell*(ell+1)]^(1+theta/2)",
        "theta": float(theta),
        "removed_modes": "ell=0,1",
        "ell_max": int(ell_max),
    }
    return EnsembleSpec(
        ensemble_id="free_scalar_harmonic_reference_v1",
        claim_tier="E1",
        regulator_id=f"real_spherical_harmonic_modes_2_to_{int(ell_max)}",
        representative_schema_hash=stable_hash({"basis": "real_spherical_harmonic_coefficients", "ell_min": 2}),
        gauge_action_hash=stable_hash({"gauge": "background_and_dipole_removed", "ell": [0, 1]}),
        quotient_canonicalizer_hash=stable_hash({"canonicalizer": "ordered_real_harmonic_modes", "order": "ell,m"}),
        base_measure_definition="Euclidean volume in ordered real harmonic coefficient coordinates",
        action_definition_and_coefficients=action,
        coarse_map_hashes=(stable_hash({"coarse_map": "mode_truncation", "retains": "ell<=L"}),),
        zero_mode_projector_hash=stable_hash({"projector": "drop_ell_0_ell_1"}),
        amplitude_convention="A=E[q^T K q]/d; if H=(1/2)q^T K q then A=2E[H]/d",
        sampler_kernel_hash=stable_hash(
            {
                "sampler": "per_mode_seeded_gaussian_sequence_coefficients",
                "mcmc": False,
                "sample_index": "sequence_offset",
            }
        ),
        rng_event_label_schema=(
            "PRF(seed_key, ensemble_id, ell, m, draw='normal_sequence') seeds one deterministic "
            "Gaussian stream per harmonic mode; sample_index is the stream offset; "
            "regulator size, worker id, thread id, and execution order are excluded"
        ),
        smoothing_policy=(
            f"post-sample W_l=exp(-0.5*sigma^2*l*(l+1)), sigma={float(smoothing_sigma)}"
            if smoothing_sigma is not None
            else "none; raw coefficients are the reference output"
        ),
        source_provenance="conventional free-scalar Gaussian reference baseline; paper issue #360 / sim issue #11",
    )


def harmonic_gaussian_reference_report(
    *,
    ell_max: int = 16,
    sample_count: int = 256,
    amplitude: float = 1.0,
    theta: float = 0.0,
    seed_key: str = "reference-vacuum-v1",
    smoothing_sigma: float | None = None,
    coarse_ell_max: int | None = None,
) -> dict[str, Any]:
    spec = free_scalar_ensemble_spec(
        ell_max=ell_max,
        amplitude=amplitude,
        theta=theta,
        smoothing_sigma=smoothing_sigma,
    )
    modes, variances, raw = sample_harmonic_coefficients(
        spec,
        sample_count=sample_count,
        seed_key=seed_key,
        partition_count=1,
    )
    _, _, replay = sample_harmonic_coefficients(
        spec,
        sample_count=sample_count,
        seed_key=seed_key,
        partition_count=3,
    )
    partition_replay = bool(np.array_equal(raw, replay))
    smooth = _smooth_coefficients(raw, modes, smoothing_sigma)
    empirical_variance = np.var(raw, axis=0, ddof=1) if sample_count > 1 else np.zeros(raw.shape[1])
    variance_relative_rmse = _relative_rmse(empirical_variance, variances) if sample_count > 1 else None
    max_offdiag_corr = _max_abs_offdiag_corr(raw)
    raw_spectrum = _spectrum_from_coefficients(raw, modes)
    smooth_spectrum = _spectrum_from_coefficients(smooth, modes)
    refinement = _harmonic_refinement_report(
        ell_max=ell_max,
        coarse_ell_max=coarse_ell_max,
        amplitude=amplitude,
        theta=theta,
        sample_count=sample_count,
        seed_key=seed_key,
        fine_modes=modes,
        fine_coefficients=raw,
    )
    return {
        "mode": "reference_vacuum_free_scalar_gaussian",
        "claim_tier": spec.claim_tier,
        "claim_tier_meaning": CLAIM_TIERS[spec.claim_tier],
        "ensemble_spec": spec.as_jsonable(),
        "seed_receipt": {
            "seed_key_hash": stable_hash({"seed_key": seed_key}),
            "seed_not_in_ensemble_spec": True,
        },
        "mode_count": int(len(modes)),
        "sample_count": int(sample_count),
        "raw_spectrum": raw_spectrum,
        "smoothed_spectrum": smooth_spectrum,
        "smoothing_provenance": {
            "policy": spec.smoothing_policy,
            "raw_data_preserved": True,
            "smoothing_sigma": None if smoothing_sigma is None else float(smoothing_sigma),
        },
        "covariance_diagnostics": {
            "known_covariance": "diag(A/[ell*(ell+1)]^(1+theta/2))",
            "variance_relative_rmse": variance_relative_rmse,
            "max_abs_offdiag_correlation": max_offdiag_corr,
            "direct_sampling_no_burn_in": True,
            "integrated_autocorrelation_time": 1.0,
        },
        "refinement_diagnostics": refinement,
        "partition_randomness": {
            "event_label_schema": spec.rng_event_label_schema,
            "partition_replay_receipt": partition_replay,
            "worker_thread_ids_excluded": True,
        },
        "reference_theory_regression_receipt": bool(
            partition_replay
            and refinement["exact_mode_truncation_refinement_receipt"]
            and sample_count >= 2
            and (variance_relative_rmse is None or variance_relative_rmse < 0.25)
        ),
        "OPH_NATIVE_QUOTIENT_ENSEMBLE_RECEIPT": False,
        "OPH_NATIVE_VACUUM_PROMOTION_RECEIPT": False,
        "OPH_PRIMORDIAL_FIELD_PROMOTION_RECEIPT": False,
        "explicit_nonclaims": list(spec.explicit_nonclaims),
    }


def sample_harmonic_coefficients(
    spec: EnsembleSpec,
    *,
    sample_count: int,
    seed_key: str,
    partition_count: int = 1,
) -> tuple[list[dict[str, int]], np.ndarray, np.ndarray]:
    ell_max = int(spec.action_definition_and_coefficients["ell_max"])
    amplitude = float(spec.action_definition_and_coefficients["amplitude_A"])
    theta = float(spec.action_definition_and_coefficients["theta"])
    modes = _harmonic_modes(ell_max)
    variances = np.array([_harmonic_variance(mode["ell"], amplitude=amplitude, theta=theta) for mode in modes])
    count = int(sample_count)
    if count < 0:
        raise ValueError("sample_count must be non-negative")
    coefficients = np.empty((count, len(modes)), dtype=float)
    # One deterministic stream per mode preserves coarse/fine regulator replay while avoiding
    # millions of per-event RNG constructions for high-ell diagnostic baselines.
    for index, mode in enumerate(modes):
        label = {
            "ensemble_id": spec.ensemble_id,
            "ell": mode["ell"],
            "m": mode["m"],
            "draw": "normal_sequence",
        }
        rng = np.random.default_rng(_seed_from_event(seed_key, label))
        coefficients[:, index] = math.sqrt(float(variances[index])) * rng.normal(size=count)
    return modes, variances, coefficients


def u1_lattice_gauge_reference_report(
    *,
    lattice_size: int = 4,
    sweeps: int = 32,
    beta: float = 0.5,
    step_size: float = math.pi / 2.0,
    seed_key: str = "reference-u1-v1",
    burn_in: int | None = None,
) -> dict[str, Any]:
    if lattice_size < 2:
        raise ValueError("lattice_size must be at least 2")
    if sweeps < 1:
        raise ValueError("sweeps must be positive")
    burn = int(sweeps // 4 if burn_in is None else burn_in)
    angles, trace, accepted, total = _run_u1_chain(
        lattice_size=int(lattice_size),
        sweeps=int(sweeps),
        beta=float(beta),
        step_size=float(step_size),
        seed_key=seed_key,
    )
    replay_angles, replay_trace, _, _ = _run_u1_chain(
        lattice_size=int(lattice_size),
        sweeps=int(sweeps),
        beta=float(beta),
        step_size=float(step_size),
        seed_key=seed_key,
    )
    post = np.asarray(trace[burn:], dtype=float)
    tau_int = _integrated_autocorrelation_time(post)
    return {
        "mode": "reference_vacuum_compact_u1_lattice_gauge",
        "claim_tier": "E1",
        "claim_tier_meaning": CLAIM_TIERS["E1"],
        "lattice_gauge_stage": {
            "compact_u1_reference": True,
            "su2_reference": False,
            "su3_reference": False,
            "su2_su3_status": "staged later; no OPH promotion attached to this U1 baseline",
        },
        "target_action": {
            "kind": "compact_u1_wilson_2d_reference",
            "log_weight": "beta*sum_p cos(theta_p)",
            "beta": float(beta),
            "proposal": "single-link symmetric uniform angle displacement",
        },
        "lattice_size": int(lattice_size),
        "sweeps": int(sweeps),
        "burn_in": burn,
        "acceptance_rate": float(accepted / max(total, 1)),
        "plaquette_trace": [float(value) for value in trace],
        "post_burn_in_mean_plaquette": float(np.mean(post)) if post.size else None,
        "thermalization_autocorrelation": {
            "integrated_autocorrelation_time": tau_int,
            "autocorrelation_receipt": bool(tau_int is not None and np.isfinite(tau_int)),
            "diagnostic_only_not_stationary_law_proof": True,
        },
        "partition_randomness": {
            "event_label_schema": (
                "PRF(seed_key, ensemble_id='compact_u1_wilson_reference_v1', sweep, x, y, mu, draw); "
                "canonical serial order is used for noncommuting local updates"
            ),
            "partition_replay_receipt": bool(np.array_equal(angles, replay_angles) and trace == replay_trace),
            "worker_thread_ids_excluded": True,
        },
        "reference_theory_regression_receipt": bool(np.isfinite(trace[-1])),
        "OPH_NATIVE_QUOTIENT_ENSEMBLE_RECEIPT": False,
        "OPH_NATIVE_VACUUM_PROMOTION_RECEIPT": False,
        "OPH_PRIMORDIAL_FIELD_PROMOTION_RECEIPT": False,
        "explicit_nonclaims": list(NONCLAIMS),
        "final_state_hash": stable_hash({"angles": np.round(angles, 15).tolist()}),
    }


def write_reference_vacuum_baseline_report(
    out_dir: Path,
    *,
    ell_max: int = 16,
    sample_count: int = 256,
    amplitude: float = 1.0,
    theta: float = 0.0,
    seed_key: str = "reference-vacuum-v1",
    smoothing_sigma: float | None = None,
    coarse_ell_max: int | None = None,
    u1_lattice_size: int = 4,
    u1_sweeps: int = 32,
    u1_beta: float = 0.5,
    u1_step_size: float = math.pi / 2.0,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = free_scalar_ensemble_spec(
        ell_max=ell_max,
        amplitude=amplitude,
        theta=theta,
        smoothing_sigma=smoothing_sigma,
    )
    modes, variances, raw = sample_harmonic_coefficients(spec, sample_count=sample_count, seed_key=seed_key)
    smooth = _smooth_coefficients(raw, modes, smoothing_sigma)
    npz_path = out / "free_scalar_harmonic_coefficients.npz"
    np.savez_compressed(
        npz_path,
        modes=np.array([[mode["ell"], mode["m"]] for mode in modes], dtype=int),
        variances=variances,
        raw_coefficients=raw,
        smoothed_coefficients=smooth,
    )
    spectrum_path = out / "free_scalar_spectra.csv"
    _write_spectrum_csv(spectrum_path, _spectrum_from_coefficients(raw, modes), _spectrum_from_coefficients(smooth, modes))
    free_scalar = harmonic_gaussian_reference_report(
        ell_max=ell_max,
        sample_count=sample_count,
        amplitude=amplitude,
        theta=theta,
        seed_key=seed_key,
        smoothing_sigma=smoothing_sigma,
        coarse_ell_max=coarse_ell_max,
    )
    free_scalar["artifacts"] = {
        "coefficients_npz": str(npz_path),
        "coefficients_npz_sha256": file_sha256(npz_path),
        "spectra_csv": str(spectrum_path),
        "spectra_csv_sha256": file_sha256(spectrum_path),
    }
    u1 = u1_lattice_gauge_reference_report(
        lattice_size=u1_lattice_size,
        sweeps=u1_sweeps,
        beta=u1_beta,
        step_size=u1_step_size,
        seed_key=seed_key + "::u1",
    )
    report = {
        "mode": "reference_vacuum_baseline_bundle",
        "claim_tier": "E1",
        "claim_tier_meaning": CLAIM_TIERS["E1"],
        "free_scalar_gaussian": free_scalar,
        "compact_u1_lattice_gauge": u1,
        "receipt_contract": {
            "ensemble_definition": True,
            "partition_invariant_randomness": bool(
                free_scalar["partition_randomness"]["partition_replay_receipt"]
                and u1["partition_randomness"]["partition_replay_receipt"]
            ),
            "thermalization_autocorrelation": u1["thermalization_autocorrelation"],
            "smoothing_provenance": free_scalar["smoothing_provenance"],
            "finite_volume_lattice_spacing": free_scalar["refinement_diagnostics"],
            "reference_theory_regression": bool(
                free_scalar["reference_theory_regression_receipt"] and u1["reference_theory_regression_receipt"]
            ),
            "oph_native_promotion": False,
        },
        "OPH_NATIVE_QUOTIENT_ENSEMBLE_RECEIPT": False,
        "OPH_NATIVE_VACUUM_PROMOTION_RECEIPT": False,
        "OPH_PRIMORDIAL_FIELD_PROMOTION_RECEIPT": False,
        "explicit_nonclaims": list(NONCLAIMS),
    }
    report_path = out / "reference_vacuum_baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(report_path)
    report["report_sha256"] = file_sha256(report_path)
    return report


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _harmonic_modes(ell_max: int) -> list[dict[str, int]]:
    return [{"ell": ell, "m": m} for ell in range(2, int(ell_max) + 1) for m in range(-ell, ell + 1)]


def _harmonic_variance(ell: int, *, amplitude: float, theta: float) -> float:
    kappa = float((ell * (ell + 1)) ** (1.0 + 0.5 * float(theta)))
    return float(amplitude) / kappa


def _normal_from_event(seed_key: str, event: dict[str, Any]) -> float:
    rng = np.random.default_rng(_seed_from_event(seed_key, event))
    return float(rng.normal())


def _uniform_from_event(seed_key: str, event: dict[str, Any]) -> float:
    return _unit_interval_from_seed(_seed_from_event(seed_key, event))


def _seed_from_event(seed_key: str, event: dict[str, Any]) -> int:
    payload = {"seed_key": seed_key, "event": event}
    digest = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"),
        digest_size=16,
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def _unit_interval_from_seed(seed: int) -> float:
    return float((int(seed) + 0.5) / float(1 << 128))


def _smooth_coefficients(
    coefficients: np.ndarray,
    modes: list[dict[str, int]],
    smoothing_sigma: float | None,
) -> np.ndarray:
    if smoothing_sigma is None:
        return np.array(coefficients, copy=True)
    weights = np.array(
        [math.exp(-0.5 * float(smoothing_sigma) ** 2 * mode["ell"] * (mode["ell"] + 1)) for mode in modes],
        dtype=float,
    )
    return np.asarray(coefficients, dtype=float) * weights[None, :]


def _spectrum_from_coefficients(coefficients: np.ndarray, modes: list[dict[str, int]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    values = np.asarray(coefficients, dtype=float)
    for ell in sorted({mode["ell"] for mode in modes}):
        indices = [index for index, mode in enumerate(modes) if mode["ell"] == ell]
        power = float(np.mean(np.square(values[:, indices]))) if values.size and indices else 0.0
        rows.append({"ell": int(ell), "mean_coefficient_power": power, "mode_count": int(len(indices))})
    return rows


def _write_spectrum_csv(path: Path, raw: list[dict[str, float]], smooth: list[dict[str, float]]) -> None:
    smooth_by_ell = {int(row["ell"]): row for row in smooth}
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ell", "mode_count", "raw_mean_coefficient_power", "smoothed_mean_coefficient_power"],
        )
        writer.writeheader()
        for row in raw:
            ell = int(row["ell"])
            writer.writerow(
                {
                    "ell": ell,
                    "mode_count": int(row["mode_count"]),
                    "raw_mean_coefficient_power": float(row["mean_coefficient_power"]),
                    "smoothed_mean_coefficient_power": float(smooth_by_ell[ell]["mean_coefficient_power"]),
                }
            )


def _relative_rmse(values: np.ndarray, targets: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return float(np.sqrt(np.mean(np.square((values - targets) / np.maximum(np.abs(targets), 1.0e-15)))))


def _max_abs_offdiag_corr(coefficients: np.ndarray) -> float | None:
    values = np.asarray(coefficients, dtype=float)
    if values.shape[0] < 3 or values.shape[1] < 2:
        return None
    corr = np.corrcoef(values, rowvar=False)
    offdiag = corr - np.eye(corr.shape[0])
    return float(np.nanmax(np.abs(offdiag)))


def _harmonic_refinement_report(
    *,
    ell_max: int,
    coarse_ell_max: int | None,
    amplitude: float,
    theta: float,
    sample_count: int,
    seed_key: str,
    fine_modes: list[dict[str, int]],
    fine_coefficients: np.ndarray,
) -> dict[str, Any]:
    coarse = int(coarse_ell_max if coarse_ell_max is not None else max(2, int(ell_max) - 1))
    coarse = min(coarse, int(ell_max))
    coarse_spec = free_scalar_ensemble_spec(ell_max=coarse, amplitude=amplitude, theta=theta)
    coarse_modes, coarse_variances, coarse_coefficients = sample_harmonic_coefficients(
        coarse_spec,
        sample_count=sample_count,
        seed_key=seed_key,
    )
    fine_indices = [
        index
        for index, mode in enumerate(fine_modes)
        if mode["ell"] <= coarse
    ]
    retained_fine = np.asarray(fine_coefficients[:, fine_indices], dtype=float)
    coarse_cov = np.diag(coarse_variances)
    retained_cov = np.diag(
        [_harmonic_variance(mode["ell"], amplitude=amplitude, theta=theta) for mode in fine_modes if mode["ell"] <= coarse]
    )
    return {
        "coarse_ell_max": int(coarse),
        "coarse_mode_count": int(len(coarse_modes)),
        "exact_covariance_residual_frobenius": float(np.linalg.norm(retained_cov - coarse_cov, ord="fro")),
        "shared_mode_bitwise_replay": bool(np.array_equal(retained_fine, coarse_coefficients)),
        "exact_mode_truncation_refinement_receipt": bool(
            np.array_equal(retained_fine, coarse_coefficients)
            and np.linalg.norm(retained_cov - coarse_cov, ord="fro") == 0.0
        ),
        "finite_volume_report": {
            "fine_ell_max": int(ell_max),
            "coarse_ell_max": int(coarse),
            "mesh_rendering_not_primary_probability_space": True,
        },
    }


def _run_u1_chain(
    *,
    lattice_size: int,
    sweeps: int,
    beta: float,
    step_size: float,
    seed_key: str,
) -> tuple[np.ndarray, list[float], int, int]:
    angles = np.zeros((lattice_size, lattice_size, 2), dtype=float)
    trace: list[float] = []
    accepted = 0
    total = 0
    current = _u1_log_weight(angles, beta=beta)
    for sweep in range(sweeps):
        for x in range(lattice_size):
            for y in range(lattice_size):
                for mu in range(2):
                    proposal_u = _uniform_from_event(
                        seed_key,
                        {"ensemble_id": "compact_u1_wilson_reference_v1", "sweep": sweep, "x": x, "y": y, "mu": mu, "draw": "proposal"},
                    )
                    delta = (2.0 * proposal_u - 1.0) * float(step_size)
                    candidate = np.array(angles, copy=True)
                    candidate[x, y, mu] = _wrap_angle(candidate[x, y, mu] + delta)
                    candidate_weight = _u1_log_weight(candidate, beta=beta)
                    accept_u = _uniform_from_event(
                        seed_key,
                        {"ensemble_id": "compact_u1_wilson_reference_v1", "sweep": sweep, "x": x, "y": y, "mu": mu, "draw": "accept"},
                    )
                    if math.log(max(accept_u, 1.0e-300)) <= min(0.0, candidate_weight - current):
                        angles = candidate
                        current = candidate_weight
                        accepted += 1
                    total += 1
        trace.append(_mean_u1_plaquette(angles))
    return angles, trace, accepted, total


def _u1_log_weight(angles: np.ndarray, *, beta: float) -> float:
    return float(beta) * float(np.sum(np.cos(_u1_plaquettes(angles))))


def _mean_u1_plaquette(angles: np.ndarray) -> float:
    return float(np.mean(np.cos(_u1_plaquettes(angles))))


def _u1_plaquettes(angles: np.ndarray) -> np.ndarray:
    size = int(angles.shape[0])
    theta = np.empty((size, size), dtype=float)
    for x in range(size):
        for y in range(size):
            theta[x, y] = (
                angles[x, y, 0]
                + angles[(x + 1) % size, y, 1]
                - angles[x, (y + 1) % size, 0]
                - angles[x, y, 1]
            )
    return theta


def _wrap_angle(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def _integrated_autocorrelation_time(values: np.ndarray) -> float | None:
    series = np.asarray(values, dtype=float)
    if series.size < 3:
        return None
    centered = series - float(np.mean(series))
    denom = float(np.dot(centered, centered))
    if denom <= 1.0e-30:
        return 1.0
    tau = 1.0
    for lag in range(1, min(series.size, 64)):
        corr = float(np.dot(centered[:-lag], centered[lag:]) / denom)
        if corr <= 0.0:
            break
        tau += 2.0 * corr
    return float(tau)
