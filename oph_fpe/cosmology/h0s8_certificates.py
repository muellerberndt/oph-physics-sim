from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from oph_universe.arrow.entropy import fake_deficit_bits, selected_ancestry_entropy_bound
from oph_universe.arrow.information import fano_payload_lower_bound


@dataclass(frozen=True)
class PayloadCertificate:
    h_x_bits: float
    alphabet_size: int
    empirical_error: float
    calibration_samples: int
    delta: float
    epsilon_delta: float
    i_rec_cert_bits: float


@dataclass(frozen=True)
class FakeTrialCertificate:
    fake_deficit_bits: float
    trial_count_log2: float
    redundant_copy_count: int
    redundant_p_star: float
    redundant_suppression_bits: float
    gamma_margin_bits: float
    probability_bound_log2: float
    probability_bound: float
    gate_pass: bool


@dataclass(frozen=True)
class H0S8CertificateInputs:
    h_x_bits: float = 64.0
    alphabet_size: int = 2**16
    empirical_error: float = 0.0
    calibration_samples: int = 10_000
    delta: float = 1e-6
    b_hid_max_bits: float = 8.0
    i_pre_bits: float = 0.0
    approx_bits: float = 0.0
    s_max_bits: float = 256.0
    selected_s_of_bits: float = 180.0
    n_res_bits: float = 56.0
    hidden_log_dim_bits: float = 8.0
    hidden_initial_entropy_bits: float = 0.0
    continuation_slack_bits: float = 0.0
    template_description_bits: float = 20.0
    branch_time: float = 1024.0
    tau_min: float = 1.0
    redundant_copy_count: int = 1
    redundant_p_star: float = 0.5
    faithful_selector_cost: float = 0.0
    best_nonfaithful_selector_cost: float = 10.0
    coarse_refinement_delta_bits: float = 0.0
    coarse_minimizer_gap_bits: float = 1.0
    gamma_target_bits: float = 0.0
    values_source: str = "diagnostic_defaults_not_finite_collar_measurements"


def certified_epsilon(empirical_error: float, calibration_samples: int, delta: float) -> float:
    samples = max(1, int(calibration_samples))
    confidence = min(max(float(delta), 1e-300), 1.0)
    eps = float(empirical_error) + math.sqrt(math.log(1.0 / confidence) / (2.0 * samples))
    return min(max(eps, 0.0), 1.0)


def payload_certificate(
    *,
    h_x_bits: float,
    alphabet_size: int,
    empirical_error: float,
    calibration_samples: int,
    delta: float,
) -> PayloadCertificate:
    eps_delta = certified_epsilon(empirical_error, calibration_samples, delta)
    return PayloadCertificate(
        h_x_bits=float(h_x_bits),
        alphabet_size=max(1, int(alphabet_size)),
        empirical_error=float(empirical_error),
        calibration_samples=max(1, int(calibration_samples)),
        delta=float(delta),
        epsilon_delta=eps_delta,
        i_rec_cert_bits=fano_payload_lower_bound(float(h_x_bits), max(1, int(alphabet_size)), eps_delta),
    )


def hidden_export_capacity_bound(
    *,
    hidden_log_dim_bits: float,
    hidden_initial_entropy_bits: float,
    continuation_slack_bits: float,
) -> float:
    return max(
        0.0,
        float(hidden_log_dim_bits) - float(hidden_initial_entropy_bits) + float(continuation_slack_bits),
    )


def observer_template_trial_count_log2(
    *,
    template_description_bits: float,
    branch_time: float,
    tau_min: float,
) -> float:
    tau = max(float(tau_min), 1e-300)
    ticks = max(1, math.ceil(max(0.0, float(branch_time)) / tau))
    return float(template_description_bits) + math.log2(ticks)


def fake_trial_certificate(
    *,
    fake_deficit_bits_value: float,
    trial_count_log2: float,
    redundant_copy_count: int = 1,
    redundant_p_star: float = 1.0,
    gamma_target_bits: float = 0.0,
) -> FakeTrialCertificate:
    copies = max(1, int(redundant_copy_count))
    p_star = min(max(float(redundant_p_star), 1e-300), 1.0)
    redundant_bits = (copies - 1) * (-math.log2(p_star))
    probability_log2 = float(trial_count_log2) - float(fake_deficit_bits_value) - redundant_bits
    probability = 1.0 if probability_log2 >= 0.0 else min(1.0, 2.0**probability_log2)
    gamma_margin = float(fake_deficit_bits_value) + redundant_bits - float(trial_count_log2)
    return FakeTrialCertificate(
        fake_deficit_bits=float(fake_deficit_bits_value),
        trial_count_log2=float(trial_count_log2),
        redundant_copy_count=copies,
        redundant_p_star=p_star,
        redundant_suppression_bits=redundant_bits,
        gamma_margin_bits=gamma_margin,
        probability_bound_log2=probability_log2,
        probability_bound=probability,
        gate_pass=gamma_margin > float(gamma_target_bits),
    )


def selector_dominance_margin(*, faithful_cost: float, best_nonfaithful_cost: float) -> float:
    return float(best_nonfaithful_cost) - float(faithful_cost)


def refinement_stability_gate(*, coarse_minimizer_gap_bits: float, coarse_refinement_delta_bits: float) -> bool:
    return float(coarse_minimizer_gap_bits) > 2.0 * float(coarse_refinement_delta_bits)


def h0s8_lane8_certificate_report(
    inputs: H0S8CertificateInputs | None = None,
) -> dict[str, Any]:
    """Return executable Lane-8/H0-S8 certificate gates.

    The defaults are diagnostic placeholders. A finite-collar cosmology run must
    replace them with measured certificate values before this becomes a physical
    H0/S8 or cosmological-arrow claim.
    """

    cfg = inputs or H0S8CertificateInputs()
    payload = payload_certificate(
        h_x_bits=cfg.h_x_bits,
        alphabet_size=cfg.alphabet_size,
        empirical_error=cfg.empirical_error,
        calibration_samples=cfg.calibration_samples,
        delta=cfg.delta,
    )
    hidden_capacity = hidden_export_capacity_bound(
        hidden_log_dim_bits=cfg.hidden_log_dim_bits,
        hidden_initial_entropy_bits=cfg.hidden_initial_entropy_bits,
        continuation_slack_bits=cfg.continuation_slack_bits,
    )
    b0 = min(float(cfg.b_hid_max_bits), hidden_capacity)
    low_entropy_bound = selected_ancestry_entropy_bound(
        cfg.s_max_bits,
        payload.i_rec_cert_bits,
        b0,
        cfg.i_pre_bits,
        cfg.approx_bits,
    )
    low_entropy_gap = (
        payload.i_rec_cert_bits - b0 - float(cfg.i_pre_bits) - float(cfg.approx_bits)
    )
    fake_deficit = fake_deficit_bits(
        payload.i_rec_cert_bits,
        cfg.n_res_bits,
        b0,
        cfg.i_pre_bits,
        cfg.approx_bits,
    )
    trial_log2 = observer_template_trial_count_log2(
        template_description_bits=cfg.template_description_bits,
        branch_time=cfg.branch_time,
        tau_min=cfg.tau_min,
    )
    fake_trials = fake_trial_certificate(
        fake_deficit_bits_value=fake_deficit,
        trial_count_log2=trial_log2,
        redundant_copy_count=cfg.redundant_copy_count,
        redundant_p_star=cfg.redundant_p_star,
        gamma_target_bits=cfg.gamma_target_bits,
    )
    selector_margin = selector_dominance_margin(
        faithful_cost=cfg.faithful_selector_cost,
        best_nonfaithful_cost=cfg.best_nonfaithful_selector_cost,
    )
    refinement_gate = refinement_stability_gate(
        coarse_minimizer_gap_bits=cfg.coarse_minimizer_gap_bits,
        coarse_refinement_delta_bits=cfg.coarse_refinement_delta_bits,
    )
    values_are_run_derived = cfg.values_source == "finite_collar_run"
    certificate_ready = (
        values_are_run_derived
        and payload.i_rec_cert_bits > 0.0
        and low_entropy_gap > float(cfg.gamma_target_bits)
        and fake_trials.gate_pass
        and selector_margin > 0.0
        and refinement_gate
    )
    report = {
        "mode": "oph_h0_s8_lane8_certificate_stack_v0",
        "values_source": cfg.values_source,
        "values_are_run_derived": values_are_run_derived,
        "payload_certificate": asdict(payload),
        "hidden_export_capacity": {
            "hidden_log_dim_bits": float(cfg.hidden_log_dim_bits),
            "hidden_initial_entropy_bits": float(cfg.hidden_initial_entropy_bits),
            "continuation_slack_bits": float(cfg.continuation_slack_bits),
            "bound_bits": hidden_capacity,
            "charged_b0_bits": b0,
        },
        "low_entropy_certificate": {
            "s_max_bits": float(cfg.s_max_bits),
            "selected_s_of_bits": float(cfg.selected_s_of_bits),
            "s_of_bound_bits": low_entropy_bound,
            "i0_bits": payload.i_rec_cert_bits,
            "b0_bits": b0,
            "p0_bits": float(cfg.i_pre_bits),
            "a0_bits": float(cfg.approx_bits),
            "gamma_target_bits": float(cfg.gamma_target_bits),
            "low_entropy_gap_bits": low_entropy_gap,
            "bound_satisfied_by_selected": float(cfg.selected_s_of_bits) <= low_entropy_bound + 1e-9,
            "gate_pass": low_entropy_gap > float(cfg.gamma_target_bits),
        },
        "fake_history_certificate": asdict(fake_trials),
        "selector_dominance": {
            "faithful_cost": float(cfg.faithful_selector_cost),
            "best_nonfaithful_cost": float(cfg.best_nonfaithful_selector_cost),
            "margin_bits": selector_margin,
            "gate_pass": selector_margin > 0.0,
        },
        "refinement_stability": {
            "coarse_refinement_delta_bits": float(cfg.coarse_refinement_delta_bits),
            "coarse_minimizer_gap_bits": float(cfg.coarse_minimizer_gap_bits),
            "gate_pass": refinement_gate,
        },
        "theorem_gates": {
            "audited_record_payload_lower_bound": payload.i_rec_cert_bits > 0.0,
            "finite_hidden_export_capacity": hidden_capacity >= 0.0,
            "fake_trial_suppression": fake_trials.gate_pass,
            "selector_dominance_margin_positive": selector_margin > 0.0,
            "refinement_stability": refinement_gate,
            "cosmological_certificate_values_filled": values_are_run_derived,
            "low_entropy_ancestry_certificate_ready": certificate_ready,
        },
        "physical_prediction_ready": False,
        "claim_boundary": (
            "Executable Lane-8 certificate stack for H0/S8 cosmology notes. Defaults are diagnostic "
            "placeholders unless values_source='finite_collar_run'. The certificate is not a physical "
            "H0/S8 prediction until finite OPH runs supply I0, B0, P0, A0, fake-trial, selector-margin, "
            "and refinement-stability values."
        ),
    }
    return report


def write_h0s8_lane8_certificate_report(out_dir: Path, inputs: H0S8CertificateInputs | None = None) -> dict[str, Any]:
    report = h0s8_lane8_certificate_report(inputs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "h0s8_lane8_certificate_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "h0s8_lane8_certificate_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    low = report["low_entropy_certificate"]
    fake = report["fake_history_certificate"]
    gates = report["theorem_gates"]
    return "\n".join(
        [
            "# OPH H0/S8 Lane-8 Certificate Stack",
            "",
            f"- values source: `{report['values_source']}`",
            f"- certified payload I0: `{low['i0_bits']:.6f}` bits",
            f"- charged hidden export B0: `{low['b0_bits']:.6f}` bits",
            f"- pre-existing provenance P0: `{low['p0_bits']:.6f}` bits",
            f"- approximation A0: `{low['a0_bits']:.6f}` bits",
            f"- low-entropy gap: `{low['low_entropy_gap_bits']:.6f}` bits",
            f"- fake-history gamma margin: `{fake['gamma_margin_bits']:.6f}` bits",
            f"- fake-history probability bound: `{fake['probability_bound']:.6e}`",
            "",
            "## Gates",
            "",
            *[f"- {name}: `{str(value).lower()}`" for name, value in gates.items()],
            "",
            "## Claim Boundary",
            "",
            str(report.get("claim_boundary", "")),
            "",
        ]
    )
