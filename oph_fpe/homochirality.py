from __future__ import annotations

import math
from typing import Any


def analytic_frank_excess(e0: float, kappa: float, t: float) -> float:
    """Closed Frank-normal-form trajectory for ``mu = h = 0``.

    The sign of ``e0`` is retained.  The exactly racemic branch stays racemic.
    This is a reduced-model result, not a claim about a particular chemistry.
    """

    seed = _bounded_excess(e0)
    gain = _nonnegative("kappa", kappa)
    time = _nonnegative("t", t)
    if seed == 0.0:
        return 0.0
    denominator = 1.0 + (seed ** -2 - 1.0) * math.exp(-2.0 * gain * time)
    return math.copysign(denominator**-0.5, seed)


def chiral_fixed_points(kappa: float, mu: float) -> list[float]:
    """Return fixed points of ``de/dt=e[(kappa-2mu)-kappa e^2]``."""

    gain = _nonnegative("kappa", kappa)
    erasure = _nonnegative("mu", mu)
    if gain <= 2.0 * erasure or gain == 0.0:
        return [0.0]
    branch = math.sqrt(1.0 - 2.0 * erasure / gain)
    return [-branch, 0.0, branch]


def homochirality_demo_report(
    *,
    e0: float = 0.01,
    kappa: float = 1.0,
    mu: float = 0.15,
    bias: float = 0.0,
    t_max: float = 12.0,
    steps: int = 241,
) -> dict[str, Any]:
    """Emit a bounded visual fixture for the OPH homochirality note.

    The trajectory integrates the phenomenological normal form

    ``de/dt = (1-e^2)(kappa*e + bias) - 2*mu*e``.

    Its purpose is to make the source/readback/record/repair chain visible.  It
    deliberately does not promote the chosen rates to prebiotic chemistry.
    """

    seed = _bounded_excess(e0)
    gain = _nonnegative("kappa", kappa)
    erasure = _nonnegative("mu", mu)
    signed_bias = float(bias)
    duration = _nonnegative("t_max", t_max)
    count = int(steps)
    if count < 2:
        raise ValueError("steps must be at least two")
    if not math.isfinite(signed_bias):
        raise ValueError("bias must be finite")

    dt = duration / (count - 1)
    times = [index * dt for index in range(count)]
    values = [seed]
    for _ in range(1, count):
        values.append(_rk4_step(values[-1], dt, gain, erasure, signed_bias))

    phase_e = [-1.0 + 2.0 * index / 160.0 for index in range(161)]
    phase_de = [_drift(value, gain, erasure, signed_bias) for value in phase_e]
    fixed = chiral_fixed_points(gain, erasure) if signed_bias == 0.0 else []
    threshold_margin = gain - 2.0 * erasure
    bounded = all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values)

    return {
        "schema": "oph_homochirality_demo_v1",
        "status": "model_demonstrator",
        "epistemicStatus": "DERIVED_WITHIN_DECLARED_NORMAL_FORM",
        "physicalClaim": False,
        "promotionAllowed": False,
        "parameters": {
            "initialExcess": seed,
            "kappa": gain,
            "mu": erasure,
            "bias": signed_bias,
            "tMax": duration,
            "steps": count,
            "units": "dimensionless e; kappa, mu, and bias in inverse demo-time units",
        },
        "branchCriterion": {
            "formula": "kappa > 2*mu",
            "margin": threshold_margin,
            "macroscopicBranchesInModel": threshold_margin > 0.0,
            "fixedPoints": fixed,
        },
        "trajectory": {"time": times, "enantiomericExcess": values},
        "phasePortrait": {"enantiomericExcess": phase_e, "drift": phase_de},
        "ophChain": [
            {"stage": "bounded patch", "value": "prebiotic compartment with declared feed and exchange ports"},
            {"stage": "readback", "value": "measured product enantiomeric excess and copy yield"},
            {"stage": "record", "value": "persistent chiral template, catalyst, crystal, or surface state"},
            {"stage": "repair", "value": "copying, inhibition, recycling, or error correction"},
            {"stage": "public receipt", "value": "rates, chromatograms, controls, and checkpoint survival"},
        ],
        "assumptions": [
            "well-mixed bounded compartment",
            "constant phenomenological kappa, mu, and bias",
            "deterministic Frank-type continuum normal form",
            "no resource depletion or spatial domain competition in this panel",
        ],
        "derivedWithinModel": [
            "exact symmetry does not choose an absolute hand",
            "the racemic fixed point changes stability at kappa=2*mu",
            "for zero bias and kappa>2*mu the model has two stable chiral branches",
        ],
        "blockedPhysicalClaims": [
            "historical terrestrial L-amino-acid sign",
            "geochemically plausible source rates",
            "planetary spatial fixation",
            "amino-acid and sugar-network coupling",
        ],
        "receipts": {
            "NORMAL_FORM_NUMERICAL_BOUNDEDNESS_CHECK": bounded,
            "BRANCH_THRESHOLD_ALGEBRA_CHECK": True,
            "PREBIOTIC_SOURCE_RATE_RECEIPT": False,
            "SPATIAL_FIXATION_RECEIPT": False,
            "AMINO_SUGAR_COUPLING_RECEIPT": False,
        },
        "claimBoundary": (
            "This is an end-to-end visual model of record-branch selection. The threshold and trajectory "
            "are consequences of the declared normal form; the rates, molecular mechanism, global sign, "
            "and historical prebiotic pathway are not derived by the simulator."
        ),
    }


def _drift(e: float, kappa: float, mu: float, bias: float) -> float:
    return (1.0 - e * e) * (kappa * e + bias) - 2.0 * mu * e


def _rk4_step(e: float, dt: float, kappa: float, mu: float, bias: float) -> float:
    if dt == 0.0:
        return e
    k1 = _drift(e, kappa, mu, bias)
    k2 = _drift(e + 0.5 * dt * k1, kappa, mu, bias)
    k3 = _drift(e + 0.5 * dt * k2, kappa, mu, bias)
    k4 = _drift(e + dt * k3, kappa, mu, bias)
    updated = e + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    # The vector field points inward at e=+/-1 for mu >= 0.  Clipping only
    # removes finite-step overshoot so the visual fixture respects its state
    # space; a physical stochastic model should evolve nonnegative L,R counts.
    return max(-1.0, min(1.0, updated))


def _bounded_excess(value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not -1.0 <= number <= 1.0:
        raise ValueError("enantiomeric excess must be finite and lie in [-1, 1]")
    return number


def _nonnegative(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number
