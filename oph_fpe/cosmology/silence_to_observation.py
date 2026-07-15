from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oph_fpe.constants.oph_pixel import OPHPixelConstants, P_STAR
from oph_fpe.cosmology.pn_resonance import (
    ALPHA_U_P_STAR,
    PNResonanceInputs,
    pn_resonance_report,
)
from oph_fpe.cosmology.repair_scale_closure import (
    effective_repair_round_depth,
    local_repair_contraction_from_p,
)
from oph_fpe.microphysics.shape_constants import loop_detuning_phase


@dataclass(frozen=True)
class SilenceToObservationInputs:
    """Inputs for the finite P/N silence-to-observation witness.

    The finite simulator cannot instantiate the literal global capacity
    ``N_CRC`` as cells. This report therefore treats ``P`` and ``N_CRC`` as
    theorem-side closure coordinates and compares them post hoc with a finite
    regulator run that starts from record silence and emits observer readouts.
    The relaxation dynamics does not consume ``P``.
    """

    P_star: float | None = None
    alpha_U: float = ALPHA_U_P_STAR
    N_star: float | None = None
    N_source: str = "ew-bridge"
    repair_rounds: int = 24
    initial_committed_fraction_max: float = 0.0
    initial_record_entropy_max: float = 0.0
    observation_committed_fraction_min: float = 0.95
    min_observer_count: int = 1
    min_h3_object_count: int = 1
    source: str = "finite_scale_compressed_pn_silence_to_observation"


def silence_to_observation_report(
    run_dir: Path,
    inputs: SilenceToObservationInputs | None = None,
) -> dict[str, Any]:
    run = Path(run_dir)
    if not run.exists():
        raise FileNotFoundError(run)
    opts = inputs or SilenceToObservationInputs()

    screen = _read_json(run / "screen_microphysics.json")
    trace_rows = _read_trace(run / "mismatch_trace.csv")
    theorem = _read_json(run / "theorem_core_receipts.json")
    observer = _read_json(run / "observer_modular_experience_report.json")
    object_chart = _read_json(run / "observer_chart_object_h3_report.json")
    readout = _read_json(run / "observer_consensus_bulk" / "observer_consensus_bulk_readout_report.json")
    defects = _read_json(run / "defect_h3_worldlines_report.json")

    p_value = _p_value(opts, screen)
    pixel = OPHPixelConstants(P=p_value, source="run_screen_microphysics")
    pn = pn_resonance_report(
        PNResonanceInputs(
            P_star=p_value,
            alpha_U=float(opts.alpha_U),
            N_star=opts.N_star,
            N_source=opts.N_source,
            repair_rounds=int(opts.repair_rounds),
            source=opts.source,
            regulator_patch_counts=_regulator_counts(screen),
        ),
        run_dirs=[run],
    )

    patch_count = _patch_count(screen, run)
    regulator_entropy_capacity = _regulator_entropy_capacity(screen, patch_count, p_value)
    repair_contraction = local_repair_contraction_from_p(p_value)
    finite_round_depth = effective_repair_round_depth(regulator_entropy_capacity, repair_contraction)

    first = trace_rows[0] if trace_rows else {}
    last = trace_rows[-1] if trace_rows else {}
    initial_record_silence = bool(
        _float(first.get("committed_fraction"), 1.0) <= float(opts.initial_committed_fraction_max)
        and _float(first.get("record_entropy"), 1.0) <= float(opts.initial_record_entropy_max)
        and _float(first.get("committed_records"), 1.0) <= 0.0
    )
    observation_emergence = bool(
        _float(last.get("committed_fraction"), 0.0) >= float(opts.observation_committed_fraction_min)
        and _float(last.get("committed_records"), 0.0) > 0.0
        and bool(observer.get("observer_modular_time_receipt", False))
        and int(observer.get("observer_count") or 0) >= int(opts.min_observer_count)
    )
    h3_objects = int(
        object_chart.get("object_count")
        or object_chart.get("localized_not_boundary_object_count")
        or readout.get("h3_object_readout", {}).get("object_count")
        or 0
    )
    h3_object_emergence = bool(
        h3_objects >= int(opts.min_h3_object_count)
        and (
            object_chart.get("observer_chart_bulk_population_receipt", False)
            or object_chart.get("OBJECT_BULK_POPULATION_RECEIPT", False)
            or readout.get("observer_h3_object_population_receipt", False)
        )
    )
    finite_consensus = bool(
        theorem.get("FINITE_CONSENSUS_THEOREM_RECEIPT", False)
        or theorem.get("finite_consensus_theorem_receipt", False)
    )
    observer_h3_experience = bool(
        observer.get("observer_facing_3p1d_h3_experience_receipt", False)
        or observer.get("OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT", False)
    )
    pn_numeric = bool(pn.get("PN_RESONANCE_NUMERIC_REPLAY", False))
    p_detuned = bool(pixel.P > pixel.phi and pixel.alpha_from_P > 0.0)
    finite_depth = bool(finite_round_depth > 0.0 and regulator_entropy_capacity > 0.0)
    scale_compressed_receipt = bool(
        pn_numeric
        and p_detuned
        and finite_depth
        and initial_record_silence
        and finite_consensus
        and observation_emergence
        and observer_h3_experience
        and h3_object_emergence
    )

    controls = _analytic_detuning_controls(
        p_value=p_value,
        alpha_u=float(opts.alpha_U),
        selected_n=float(pn["paper_bridge_relation"]["selected_N_star"]),
    )

    return {
        "mode": "oph_pn_silence_to_observation_witness_v1",
        "source": opts.source,
        "run_dir": str(run),
        "scale_compressed_pn_silence_to_observation_receipt": scale_compressed_receipt,
        "SCALE_COMPRESSED_PN_SILENCE_TO_OBSERVATION_RECEIPT": scale_compressed_receipt,
        "literal_global_N_capacity_simulated_receipt": False,
        "LITERAL_GLOBAL_N_CAPACITY_SIMULATED_RECEIPT": False,
        "dynamic_p_detuning_control_receipt": False,
        "DYNAMIC_P_DETUNING_CONTROL_RECEIPT": False,
        "p_role": "post_hoc_analytic_branch_association",
        "relaxation_dynamics_consumed_p": False,
        "closure_coordinates": {
            "P": pixel.P,
            "phi_silent_equilibrium": pixel.phi,
            "sqrt_pi": pixel.sqrt_pi,
            "P_detuning_delta": pixel.P - pixel.phi,
            "alpha_from_P": pixel.alpha_from_P,
            "alpha_inverse_from_P": pixel.alpha_inverse_from_P,
            "loop_detuning_phase": loop_detuning_phase(pixel.alpha_from_P),
            "N_star_source": pn["inputs"]["N_source"],
            "N_star": pn["inputs"]["N_star"],
            "PN_RESONANCE_NUMERIC_REPLAY": pn_numeric,
            "PN_RESONANCE_RECEIPT": bool(pn.get("PN_RESONANCE_RECEIPT", False)),
        },
        "finite_regulator_depth": {
            "patch_count": patch_count,
            "regulator_entropy_capacity_N_eff": regulator_entropy_capacity,
            "local_repair_contraction_abs_gprime": repair_contraction,
            "effective_repair_round_depth": finite_round_depth,
            "declared_cosmic_repair_round_depth": int(opts.repair_rounds),
            "depth_fraction_of_declared_cosmic_rounds": finite_round_depth / float(opts.repair_rounds),
            "claim_boundary": (
                "N_eff is a postprocessed P-weighted entropy-capacity coordinate for the finite "
                "regulator. It is not an input to the relaxation law or the literal cosmic capacity N_CRC."
            ),
        },
        "silence_initial_state": {
            "cycle": first.get("cycle"),
            "committed_records": first.get("committed_records"),
            "committed_fraction": first.get("committed_fraction"),
            "record_entropy": first.get("record_entropy"),
            "mismatch_phi": first.get("phi"),
            "initial_record_silence_receipt": initial_record_silence,
            "claim_boundary": (
                "Silence means no committed observer records/readback entropy at the first recorded "
                "repair cycle. It does not mean the finite carrier has no screen geometry or no hot "
                "pre-record interface state."
            ),
        },
        "observation_emergence": {
            "final_cycle": last.get("cycle"),
            "final_committed_records": last.get("committed_records"),
            "final_committed_fraction": last.get("committed_fraction"),
            "final_record_entropy": last.get("record_entropy"),
            "final_phi": last.get("phi"),
            "finite_consensus_theorem_receipt": finite_consensus,
            "observer_modular_time_receipt": bool(observer.get("observer_modular_time_receipt", False)),
            "observer_facing_3p1d_h3_experience_receipt": observer_h3_experience,
            "observer_count": int(observer.get("observer_count") or 0),
            "h3_object_count": h3_objects,
            "h3_object_emergence_receipt": h3_object_emergence,
            "persistent_h3_worldline_count": int(defects.get("persistent_h3_worldline_count") or 0),
            "observation_emergence_receipt": observation_emergence,
        },
        "detuning_controls": controls,
        "component_reports": {
            "pn_resonance_report": pn,
        },
        "readiness_gates": {
            "paper_pn_numeric_replay": pn_numeric,
            "P_detuning_nonzero": p_detuned,
            "finite_regulator_depth_positive": finite_depth,
            "initial_record_silence": initial_record_silence,
            "finite_consensus_theorem": finite_consensus,
            "observation_records_emerged": observation_emergence,
            "observer_modular_time": bool(observer.get("observer_modular_time_receipt", False)),
            "observer_facing_3p1d_h3_experience": observer_h3_experience,
            "h3_object_emergence": h3_object_emergence,
            "analytic_no_detuning_control_blocks_pn_bridge": bool(
                controls["no_detuning_phi_equilibrium"]["blocks_pn_bridge"]
            ),
            "analytic_wrong_detuning_controls_block_selected_bridge": all(
                row["blocks_selected_bridge"] for row in controls["wrong_detuning_multipliers"]
            ),
            "dynamic_no_wrong_detuning_reruns_present": False,
            "literal_N_CRC_cells_instantiated": False,
            "scale_compressed_pn_silence_to_observation": scale_compressed_receipt,
        },
        "claim_boundary": (
            "Finite scale-compressed association between an observed record-silence/readout transition "
            "and the paper-side P/N closure coordinates. The relaxation dynamics did not consume P. "
            "The analytic detuning controls test bridge arithmetic rather than dynamic causation; no "
            "P=phi or wrong-detuning reruns are present. The report neither instantiates the astronomical "
            "N_CRC cell count nor solves the global F(N) fixed point from simulator data."
        ),
    }


def write_silence_to_observation_report(
    run_dir: Path,
    out_dir: Path | None = None,
    inputs: SilenceToObservationInputs | None = None,
) -> dict[str, Any]:
    report = silence_to_observation_report(run_dir, inputs)
    out = Path(out_dir) if out_dir is not None else Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "silence_to_observation_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (out / "silence_to_observation_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _analytic_detuning_controls(*, p_value: float, alpha_u: float, selected_n: float) -> dict[str, Any]:
    pixel = OPHPixelConstants(P=p_value)
    no_detuning_p = pixel.phi
    controls = []
    for multiplier in (0.5, 1.5, 2.0):
        candidate_p = pixel.phi + multiplier * (pixel.P - pixel.phi)
        candidate_n = _capacity_from_p_alpha(candidate_p, alpha_u)
        log_residual = math.log(candidate_n / selected_n)
        controls.append(
            {
                "detuning_multiplier": multiplier,
                "candidate_P": candidate_p,
                "candidate_alpha_from_P": (candidate_p - pixel.phi) / pixel.sqrt_pi,
                "candidate_N_EW_bridge": candidate_n,
                "log_residual_vs_selected_N": log_residual,
                "blocks_selected_bridge": bool(abs(log_residual) > 1.0e-6),
            }
        )
    return {
        "no_detuning_phi_equilibrium": {
            "P": no_detuning_p,
            "P_detuning_delta": 0.0,
            "alpha_from_P": 0.0,
            "blocks_pn_bridge": True,
            "reason": "alpha_from_P=0 makes N=pi*exp(6*pi/(P*alpha_U(P))) undefined/infinite on this bridge",
        },
        "wrong_detuning_multipliers": controls,
        "claim_boundary": (
            "Analytic P-detuning controls only. They prove that the selected P/N bridge is not invariant "
            "under no/wrong detuning, but they are not dynamic finite reruns."
        ),
    }


def _capacity_from_p_alpha(p_value: float, alpha_u: float) -> float:
    return float(math.pi * math.exp(6.0 * math.pi / (float(p_value) * float(alpha_u))))


def _p_value(opts: SilenceToObservationInputs, screen: dict[str, Any]) -> float:
    if opts.P_star is not None:
        return float(opts.P_star)
    pixel_scale = screen.get("pixel_scale") if isinstance(screen.get("pixel_scale"), dict) else {}
    value = pixel_scale.get("P") or screen.get("cell_area_planck")
    return float(value) if value is not None else P_STAR


def _patch_count(screen: dict[str, Any], run: Path) -> int:
    if screen.get("patch_count") is not None:
        return int(screen["patch_count"])
    manifest = _read_json(run / "manifest.json")
    if manifest.get("patch_count") is not None:
        return int(manifest["patch_count"])
    return 0


def _regulator_counts(screen: dict[str, Any]) -> tuple[int, ...]:
    count = int(screen.get("patch_count") or 0)
    base = [4096, 65536, 262144, 1048576]
    if count > 0 and count not in base:
        base.append(count)
    return tuple(sorted(set(base)))


def _regulator_entropy_capacity(screen: dict[str, Any], patch_count: int, p_value: float) -> float:
    units = screen.get("screen_units") if isinstance(screen.get("screen_units"), dict) else {}
    value = units.get("regulator_entropy_weight_sum")
    if value is not None:
        return float(value)
    return float(int(patch_count) * float(p_value) / 4.0)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: _coerce(value) for key, value in row.items()})
    return rows


def _coerce(value: Any) -> Any:
    if value is None or value == "":
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _markdown_report(report: dict[str, Any]) -> str:
    closure = report["closure_coordinates"]
    depth = report["finite_regulator_depth"]
    silence = report["silence_initial_state"]
    emergence = report["observation_emergence"]
    gates = report["readiness_gates"]
    return "\n".join(
        [
            "# OPH P/N Silence To Observation",
            "",
            str(report["claim_boundary"]),
            "",
            "## Status",
            "",
            "- scale-compressed P/N silence-to-observation receipt: "
            f"`{str(report['scale_compressed_pn_silence_to_observation_receipt']).lower()}`",
            "- literal global N capacity simulated: "
            f"`{str(report['literal_global_N_capacity_simulated_receipt']).lower()}`",
            "- dynamic P-detuning controls present: "
            f"`{str(report['dynamic_p_detuning_control_receipt']).lower()}`",
            "",
            "## Closure Coordinates",
            "",
            f"- P: `{closure['P']:.15g}`",
            f"- phi silent equilibrium: `{closure['phi_silent_equilibrium']:.15g}`",
            f"- P detuning delta: `{closure['P_detuning_delta']:.15g}`",
            f"- alpha from P: `{closure['alpha_from_P']:.15g}`",
            f"- N_star: `{closure['N_star']:.12e}`",
            "",
            "## Finite Regulator Depth",
            "",
            f"- patch count: `{depth['patch_count']}`",
            f"- N_eff entropy capacity: `{depth['regulator_entropy_capacity_N_eff']:.12g}`",
            f"- effective repair-round depth: `{depth['effective_repair_round_depth']:.12g}`",
            f"- fraction of 24-round cosmic depth: `{depth['depth_fraction_of_declared_cosmic_rounds']:.12g}`",
            "",
            "## Silence To Observation",
            "",
            f"- initial committed records: `{silence['committed_records']}`",
            f"- initial record entropy: `{silence['record_entropy']}`",
            f"- final committed records: `{emergence['final_committed_records']}`",
            f"- observer count: `{emergence['observer_count']}`",
            f"- H3 object count: `{emergence['h3_object_count']}`",
            f"- persistent H3 worldlines: `{emergence['persistent_h3_worldline_count']}`",
            "",
            "## Gates",
            "",
            *[f"- {key}: `{str(value).lower()}`" for key, value in gates.items()],
            "",
        ]
    )
