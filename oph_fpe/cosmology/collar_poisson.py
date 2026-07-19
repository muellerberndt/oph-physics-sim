from __future__ import annotations

import math
import hashlib
import json
from numbers import Real
from pathlib import Path
from typing import Any, Sequence

from oph_fpe.claims import CONTINUATION, with_claim_metadata
from oph_fpe.evidence.artifact_paths import companion_input_packet_path


MAX_BERNOULLI_CELLS = 4096
MAX_EXACT_TV_WORK = 2_000_000
MAX_COLLAR_POISSON_PACKET_BYTES = 1_000_000
IMPORTED_THEORY_WITNESS_CERTIFICATE_ID = "issue-320-collar-poisson-witness-v1"
IMPORTED_THEORY_WITNESS_SHA256 = (
    "sha256:01d77edd9ea7867e27013370a6d580d62c85a54085dc7a09494b5a5f124415e2"
)


def collar_poisson_counting_certificate(
    *,
    activation_probabilities: Sequence[float],
    limiting_mean: float,
    cut_sqrt_measure: float | None = None,
) -> dict[str, Any]:
    """Recompute the finite Le Cam collar-opportunity certificate.

    The supplied probabilities define the *declared independent Bernoulli
    model*.  The function recomputes its intensity, mean-drift term, Le Cam
    bound and, within the operation budget, the exact Poisson-binomial total
    variation distance.  It deliberately does not assert that a physical
    settled-galaxy run realizes the independent collar model.
    """

    blockers: list[str] = []
    probabilities = _probabilities_or_none(activation_probabilities)
    independent_model_valid = bool(
        probabilities is not None
        and probabilities
        and len(probabilities) <= MAX_BERNOULLI_CELLS
    )
    if probabilities is None:
        blockers.append("activation_probabilities_invalid")
        probabilities = []
    if not probabilities:
        blockers.append("activation_probability_family_empty")
    if len(probabilities) > MAX_BERNOULLI_CELLS:
        blockers.append("bernoulli_cell_budget_exceeded")

    mu_limit = _finite_nonnegative_or_none(limiting_mean)
    if mu_limit is None:
        blockers.append("limiting_mean_invalid")

    sqrt_measure = _finite_positive_or_none(cut_sqrt_measure)
    if cut_sqrt_measure is not None and sqrt_measure is None:
        blockers.append("cut_sqrt_measure_invalid")

    mu_r = float(sum(probabilities))
    p_max = max(probabilities, default=0.0)
    sum_p_squared = float(sum(probability * probability for probability in probabilities))
    delta_r = abs(mu_r - mu_limit) if mu_limit is not None else None
    theorem_bound = sum_p_squared + delta_r if delta_r is not None else None
    coarse_bound = mu_r * p_max + delta_r if delta_r is not None else None

    exact_tv: float | None = None
    exact_tv_computed = False
    exact_tv_within_bound = False
    if (
        not blockers
        and mu_limit is not None
        and len(probabilities) * len(probabilities) <= MAX_EXACT_TV_WORK
    ):
        poisson_binomial = _poisson_binomial_pmf(probabilities)
        exact_tv = _total_variation_to_poisson(poisson_binomial, mu_limit)
        exact_tv_computed = True
        exact_tv_within_bound = exact_tv <= float(theorem_bound) + 2.0e-12
        if not exact_tv_within_bound:
            blockers.append("exact_total_variation_exceeds_theorem_bound")

    lambda_collar = mu_limit / sqrt_measure if mu_limit is not None and sqrt_measure else None
    if lambda_collar is not None and not math.isfinite(lambda_collar):
        lambda_collar = None
        blockers.append("lambda_collar_not_finite")
    arithmetic_receipt = bool(
        not blockers
        and theorem_bound is not None
        and coarse_bound is not None
        and sum_p_squared <= mu_r * p_max + 1.0e-15
        and (not exact_tv_computed or exact_tv_within_bound)
    )
    report = {
        "schema_version": "collar-poisson-counting-v1",
        "COLLAR_POISSON_COUNTING_RECOMPUTATION_RECEIPT": arithmetic_receipt,
        "DECLARED_INDEPENDENT_BERNOULLI_MODEL": independent_model_valid,
        "PHYSICAL_COLLAR_MODEL_REALIZATION_RECEIPT": False,
        "PHYSICAL_GALAXY_POISSON_RECEIPT": False,
        "RUN_ARTIFACT_BINDING_RECEIPT": False,
        "IMPORTED_THEORY_WITNESS_SIMULATION_RECEIPT_ELIGIBLE": False,
        "imported_theory_witness_certificate_id": IMPORTED_THEORY_WITNESS_CERTIFICATE_ID,
        "imported_theory_witness_sha256": IMPORTED_THEORY_WITNESS_SHA256,
        "cell_count": len(probabilities),
        "mu_r": mu_r,
        "limiting_mean": mu_limit,
        "delta_r": delta_r,
        "p_max": p_max,
        "sum_p_squared": sum_p_squared,
        "le_cam_mean_continuity_bound": theorem_bound,
        "mu_r_p_max_plus_delta_bound": coarse_bound,
        "exact_total_variation_computed": exact_tv_computed,
        "exact_total_variation": exact_tv,
        "exact_total_variation_within_bound": exact_tv_within_bound if exact_tv_computed else None,
        "cut_sqrt_measure": sqrt_measure,
        "lambda_collar_from_declared_limit": lambda_collar,
        "blockers": blockers,
        "claim_boundary": (
            "exact finite counting check for a declared independent Bernoulli collar model; physical "
            "realization, refinement-stable source intensity and flux-recovery closure remain separate"
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=CONTINUATION,
        receipt="COLLAR_POISSON_COUNTING_RECOMPUTATION_RECEIPT",
    )


def write_collar_poisson_certificate(
    path: Path,
    packet: dict[str, Any],
) -> dict[str, Any]:
    """Write a replayable arithmetic certificate and its nonpromoting input."""

    expected_fields = {
        "activation_probabilities",
        "limiting_mean",
        "cut_sqrt_measure",
    }
    if not isinstance(packet, dict) or set(packet) - expected_fields or not {
        "activation_probabilities",
        "limiting_mean",
    }.issubset(packet):
        raise ValueError("collar-Poisson packet has missing or unknown fields")
    probabilities = packet["activation_probabilities"]
    if not isinstance(probabilities, list) or len(probabilities) > MAX_BERNOULLI_CELLS:
        raise ValueError(
            "collar-Poisson writer requires a bounded JSON activation-probability array"
        )
    encoded = _bounded_input_packet_bytes(packet)
    report = collar_poisson_counting_certificate(
        activation_probabilities=probabilities,
        limiting_mean=packet["limiting_mean"],
        cut_sqrt_measure=packet.get("cut_sqrt_measure"),
    )
    report["input_packet_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        destination = destination / "collar_poisson_certificate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    replay_path = companion_input_packet_path(
        destination,
        canonical_certificate_filename="collar_poisson_certificate.json",
        canonical_input_filename="collar_poisson_input_packet.json",
    )
    replay_path.write_text(
        json.dumps(packet, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _probabilities_or_none(value: Sequence[float]) -> list[float] | None:
    if isinstance(value, (str, bytes)):
        return None
    probabilities: list[float] = []
    try:
        iterator = iter(value)
    except TypeError:
        return None
    for index, item in enumerate(iterator):
        if index > MAX_BERNOULLI_CELLS:
            break
        if isinstance(item, bool) or not isinstance(item, Real):
            return None
        try:
            probability = float(item)
        except (OverflowError, TypeError, ValueError):
            return None
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            return None
        probabilities.append(probability)
    return probabilities


def _finite_nonnegative_or_none(value: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def _finite_positive_or_none(value: float | None) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _bounded_input_packet_bytes(packet: dict[str, Any]) -> bytes:
    probabilities = packet["activation_probabilities"]
    for value in probabilities:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("activation probabilities must be finite JSON numbers")
        try:
            parsed = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                "activation probabilities must be finite JSON numbers"
            ) from exc
        if not math.isfinite(parsed):
            raise ValueError("activation probabilities must be finite JSON numbers")
    for field in ("limiting_mean", "cut_sqrt_measure"):
        value = packet.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field} must be a finite JSON number or null")
        try:
            parsed = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a finite JSON number or null") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"{field} must be a finite JSON number or null")
    try:
        encoded = json.dumps(
            packet,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("collar-Poisson packet is not canonical JSON data") from exc
    if len(encoded) > MAX_COLLAR_POISSON_PACKET_BYTES:
        raise ValueError("collar-Poisson packet exceeds the byte budget")
    return encoded


def _poisson_binomial_pmf(probabilities: Sequence[float]) -> list[float]:
    pmf = [1.0]
    for probability in probabilities:
        updated = [0.0] * (len(pmf) + 1)
        for count, mass in enumerate(pmf):
            updated[count] += mass * (1.0 - probability)
            updated[count + 1] += mass * probability
        pmf = updated
    return pmf


def _total_variation_to_poisson(poisson_binomial: Sequence[float], mean: float) -> float:
    poisson_masses = [_poisson_mass(count, mean) for count in range(len(poisson_binomial))]
    represented_poisson_mass = math.fsum(poisson_masses)
    poisson_tail = max(0.0, 1.0 - represented_poisson_mass)
    return 0.5 * (
        math.fsum(abs(left - right) for left, right in zip(poisson_binomial, poisson_masses))
        + poisson_tail
    )


def _poisson_mass(count: int, mean: float) -> float:
    if mean == 0.0:
        return 1.0 if count == 0 else 0.0
    return math.exp(-mean + count * math.log(mean) - math.lgamma(count + 1.0))
