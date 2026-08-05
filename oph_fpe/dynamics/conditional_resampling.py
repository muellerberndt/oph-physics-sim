"""Exact conditional-resampling realization on observation fibers.

This is the producer side of the A3 transition kernel whose recognizer
lives in ``oph_fpe.quotient.observable_normal_form``: the kernel

    P(x, y) = 1[b(y) = b(x)] * pi(y) / pi(F_b(x))

that holds the protected observation datum ``b`` fixed and resamples
everything it leaves unconstrained from the pinned reference law ``pi``
restricted to the fiber.  The realization here is earned from run data:
the protected datum is the run's committed record class (nonconstant
across the screen), the companion coordinate is a committed field class,
and the pinned common reference is the run's realized joint frequency
table converted to exact rationals.  The kernel table is built from the
target formula over the rationals, replayed through the independent
recognizer, and its stationarity, idempotence, and chi-squared
contraction are verified exactly.  An empirical resampling trajectory is
then driven with integer counts and its fiber-conditional chi-squared
divergence to the reference is reported per sweep.

Everything is a finite exact statement on declared data; nothing here
promotes a physical claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from oph_fpe.quotient.observable_normal_form import (
    recognize_conditional_resampling_kernel,
)

CONDITIONAL_RESAMPLING_REALIZATION_SCHEMA = (
    "oph.sim.conditional_resampling_realization.v1"
)

_MAX_RECORD_CLASSES = 32
_MAX_COMPANION_CLASSES = 16


@dataclass(frozen=True)
class RealizationInputs:
    """Realized integer class labels feeding the exact construction."""

    record_classes: tuple[int, ...]
    companion_classes: tuple[int, ...]
    record_label: str
    companion_label: str
    provenance: dict[str, Any]


def _class_bins(values: np.ndarray, max_classes: int) -> np.ndarray:
    """Map a realized field to consecutive integer classes.

    Distinct realized values map to distinct classes when few; otherwise
    quantile edges over the realized values define the classes.  The
    binning is declared data, not an estimate.
    """

    flat = np.asarray(values).ravel()
    finite = flat[np.isfinite(flat.astype(float))] if flat.dtype.kind == "f" else flat
    distinct = np.unique(finite)
    if distinct.size <= max_classes:
        lookup = {value: index for index, value in enumerate(distinct.tolist())}
        return np.asarray([lookup.get(value, 0) for value in flat.tolist()], dtype=np.int64)
    edges = np.quantile(finite.astype(float), np.linspace(0.0, 1.0, max_classes + 1)[1:-1])
    return np.digitize(flat.astype(float), edges).astype(np.int64)


_COMPANION_CANDIDATES = (
    "cumulative_repair_load",
    "stable_count",
    "repair_load",
    "s3_class_density",
)


def realization_inputs_from_freezeout(
    freezeout_path: Path,
    *,
    record_field: str = "record_signature",
    companion_field: str | None = None,
    max_record_classes: int = _MAX_RECORD_CLASSES,
    max_companion_classes: int = _MAX_COMPANION_CLASSES,
) -> RealizationInputs:
    """Build realization inputs from a run's committed freezeout fields.

    When no companion field is named, the first declared candidate whose
    realized values are nonconstant is selected; fields frozen to a
    single value cannot carry the resampled coordinate.
    """

    path = Path(freezeout_path)
    bundle = np.load(path, allow_pickle=True)
    if record_field not in bundle:
        raise ValueError(
            f"freezeout bundle lacks {record_field!r}: "
            f"available {sorted(bundle.keys())}"
        )
    if companion_field is None:
        for candidate in _COMPANION_CANDIDATES:
            if candidate in bundle and np.unique(bundle[candidate]).size >= 2:
                companion_field = candidate
                break
        else:
            raise ValueError(
                "no declared companion candidate is nonconstant: "
                f"{_COMPANION_CANDIDATES}"
            )
    elif companion_field not in bundle:
        raise ValueError(
            f"freezeout bundle lacks {companion_field!r}: "
            f"available {sorted(bundle.keys())}"
        )
    record = _class_bins(bundle[record_field], max_record_classes)
    companion = _class_bins(bundle[companion_field], max_companion_classes)
    if record.shape != companion.shape:
        raise ValueError("record and companion fields must be aligned per patch")
    return RealizationInputs(
        record_classes=tuple(int(v) for v in record.tolist()),
        companion_classes=tuple(int(v) for v in companion.tolist()),
        record_label=record_field,
        companion_label=companion_field,
        provenance={
            "source": str(path),
            "record_field": record_field,
            "companion_field": companion_field,
            "patch_count": int(record.size),
        },
    )


def _joint_reference(
    inputs: RealizationInputs,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], Fraction], dict[tuple[int, int], int]]:
    """The pinned common reference: the realized joint frequency table
    over (record class, companion class), exact over the rationals.

    Only realized cells enter the state space, so every weight is
    strictly positive by construction and no regularization is applied.
    """

    counts: dict[tuple[int, int], int] = {}
    for record, companion in zip(inputs.record_classes, inputs.companion_classes):
        counts[(record, companion)] = counts.get((record, companion), 0) + 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError("realization inputs are empty")
    states = sorted(counts)
    weights = {state: Fraction(counts[state], total) for state in states}
    return states, weights, counts


def _fiber_kernel(
    states: Sequence[tuple[int, int]],
    weights: dict[tuple[int, int], Fraction],
) -> dict[tuple[int, int], dict[tuple[int, int], Fraction]]:
    fiber_mass: dict[int, Fraction] = {}
    for state in states:
        fiber_mass[state[0]] = fiber_mass.get(state[0], Fraction(0)) + weights[state]
    return {
        x: {
            y: (weights[y] / fiber_mass[x[0]] if y[0] == x[0] else Fraction(0))
            for y in states
        }
        for x in states
    }


def _chi_squared(
    p: dict[tuple[int, int], Fraction],
    reference: dict[tuple[int, int], Fraction],
) -> Fraction:
    return sum(
        ((p.get(state, Fraction(0)) - weight) ** 2) / weight
        for state, weight in reference.items()
    )


def produce_conditional_resampling_realization(
    inputs: RealizationInputs,
    *,
    seed: int = 0,
    empirical_sweeps: int = 8,
) -> dict[str, Any]:
    """Produce the exact realization receipt.

    The kernel is constructed from the target formula, replayed through
    the independent recognizer, and checked for exact idempotence,
    stationarity of the reference, and chi-squared contraction from a
    perturbed start.  The empirical trajectory resamples every patch's
    companion class within its record fiber and reports the exact
    fiber-conditional chi-squared per sweep from integer counts.  The
    protected record datum is asserted unchanged throughout.
    """

    states, weights, counts = _joint_reference(inputs)
    record_values = sorted({state[0] for state in states})
    if len(record_values) < 2:
        raise ValueError(
            "the protected record datum is constant; the realization "
            "requires a nonconstant record"
        )
    fiber_sizes = {
        record: sum(1 for state in states if state[0] == record)
        for record in record_values
    }
    nontrivial_fibers = sum(1 for size in fiber_sizes.values() if size >= 2)
    if nontrivial_fibers == 0:
        raise ValueError(
            "every observation fiber is a singleton; the resampling "
            "kernel degenerates to the identity"
        )
    kernel = _fiber_kernel(states, weights)

    audit = recognize_conditional_resampling_kernel(
        states, kernel, weights=weights, observation_map=lambda state: state[0]
    )

    # Exact kernel-package replay over the rationals.
    idempotent = all(
        sum((kernel[x][z] * kernel[z][y] for z in states), start=Fraction(0))
        == kernel[x][y]
        for x in states
        for y in states
    )
    stationary = all(
        sum((weights[x] * kernel[x][y] for x in states), start=Fraction(0))
        == weights[y]
        for y in states
    )
    # Chi-squared contraction from a perturbed start supported on the
    # same states (mass moved between the two heaviest states).
    perturbed = dict(weights)
    heavy = sorted(states, key=lambda s: weights[s], reverse=True)[:2]
    shift = min(weights[heavy[1]], Fraction(1, 8))
    perturbed[heavy[0]] = perturbed[heavy[0]] + shift
    perturbed[heavy[1]] = perturbed[heavy[1]] - shift
    pushed = {
        y: sum((perturbed[x] * kernel[x][y] for x in states), start=Fraction(0))
        for y in states
    }
    chi_before = _chi_squared(perturbed, weights)
    chi_after = _chi_squared(pushed, weights)

    # Empirical realization: resample each patch's companion class from
    # the reference conditioned on its protected record class.
    rng = np.random.default_rng(seed)
    fiber_states: dict[int, list[tuple[int, int]]] = {}
    for state in states:
        fiber_states.setdefault(state[0], []).append(state)
    fiber_laws = {
        record: np.asarray(
            [float(kernel[fiber[0]][y]) for y in fiber], dtype=float
        )
        for record, fiber in fiber_states.items()
    }
    patches = np.asarray(inputs.record_classes, dtype=np.int64)
    # Declared displaced start: every patch's companion class is moved to
    # its fiber's least-likely realized class, so the first resampling
    # sweep exhibits the one-step collapse onto the fiber reference law
    # (the kernel is idempotent) as a measured event.
    least_likely = {
        record: min(fiber, key=lambda s: weights[s])[1]
        for record, fiber in fiber_states.items()
    }
    companion = np.asarray(
        [least_likely[int(r)] for r in patches.tolist()], dtype=np.int64
    )
    protected_before = patches.copy()

    def _empirical_chi() -> Fraction:
        empirical_counts: dict[tuple[int, int], int] = {}
        for r_value, c_value in zip(patches.tolist(), companion.tolist()):
            key = (int(r_value), int(c_value))
            empirical_counts[key] = empirical_counts.get(key, 0) + 1
        total = sum(empirical_counts.values())
        empirical_law = {
            state: Fraction(count, total)
            for state, count in empirical_counts.items()
        }
        return _chi_squared(empirical_law, weights)

    start_chi = _empirical_chi()
    sweep_rows: list[dict[str, Any]] = [
        {
            "sweep": "displaced_start",
            "chi_squared_to_reference": str(start_chi),
            "chi_squared_float": float(start_chi),
        }
    ]
    for sweep in range(int(empirical_sweeps)):
        for record in record_values:
            mask = patches == record
            fiber = fiber_states[record]
            law = fiber_laws[record]
            draws = rng.choice(len(fiber), size=int(mask.sum()), p=law / law.sum())
            companion[mask] = np.asarray(
                [fiber[i][1] for i in draws], dtype=np.int64
            )
        chi_emp = _empirical_chi()
        sweep_rows.append(
            {
                "sweep": sweep,
                "chi_squared_to_reference": str(chi_emp),
                "chi_squared_float": float(chi_emp),
            }
        )
    protected_record_unchanged = bool(np.array_equal(protected_before, patches))

    passed = bool(
        audit.exact_table_recognition_receipt
        and idempotent
        and stationary
        and chi_after <= chi_before
        and protected_record_unchanged
    )
    return {
        "schema": CONDITIONAL_RESAMPLING_REALIZATION_SCHEMA,
        "CONDITIONAL_RESAMPLING_REALIZATION_RECEIPT": passed,
        "protected_record": {
            "label": inputs.record_label,
            "class_count": len(record_values),
            "nonconstant": True,
            "unchanged_by_resampling": protected_record_unchanged,
        },
        "companion": {"label": inputs.companion_label},
        "pinned_reference": {
            "kind": "realized_joint_frequency_table",
            "state_count": len(states),
            "fiber_count": len(record_values),
            "nontrivial_fiber_count": nontrivial_fibers,
            "total_mass_count": sum(counts.values()),
        },
        "recognizer": {
            "exact_table_recognition_receipt": bool(
                audit.exact_table_recognition_receipt
            ),
            "r1_fiber_supported": bool(audit.r1_fiber_supported),
            "r2_fiber_rows_constant": bool(audit.r2_fiber_rows_constant),
            "r3_weighted_detailed_balance": bool(audit.r3_weighted_detailed_balance),
            "explicit_formula_match": bool(audit.explicit_formula_match),
            "verifier_version": audit.verifier_version,
            "theorem_reference": audit.theorem_reference,
        },
        "exact_kernel_package": {
            "idempotent": bool(idempotent),
            "reference_stationary": bool(stationary),
            "chi_squared_before": str(chi_before),
            "chi_squared_after_one_step": str(chi_after),
            "chi_squared_contracts": bool(chi_after <= chi_before),
        },
        "empirical_realization": {
            "seed": int(seed),
            "start_mode": "fiberwise_least_likely_companion_class",
            "sweeps": sweep_rows,
            "patch_count": int(patches.size),
            "one_step_collapse_measured": bool(
                len(sweep_rows) >= 2
                and sweep_rows[1]["chi_squared_float"]
                < sweep_rows[0]["chi_squared_float"]
            ),
        },
        "provenance": inputs.provenance,
        "claim_boundary": (
            "Exact finite realization of the conditional-resampling kernel "
            "package on this run's committed record classes with the "
            "realized joint frequency table as the pinned common reference. "
            "The empirical sweep rows are integer-count measurements, not "
            "estimates. No physical claim is promoted."
        ),
    }


def write_conditional_resampling_realization(
    run_dir: Path,
    *,
    seed: int = 0,
    empirical_sweeps: int = 8,
) -> dict[str, Any]:
    """Produce the realization receipt for a run directory holding
    ``freezeout_fields.npz``; report a labeled skip otherwise."""

    run_dir = Path(run_dir)
    freezeout = run_dir / "freezeout_fields.npz"
    output = run_dir / "conditional_resampling_realization_receipt.json"
    if not freezeout.exists():
        payload: dict[str, Any] = {
            "schema": CONDITIONAL_RESAMPLING_REALIZATION_SCHEMA,
            "CONDITIONAL_RESAMPLING_REALIZATION_RECEIPT": False,
            "skipped": True,
            "reason": "freezeout_fields.npz absent (cosmology freezeout gate)",
        }
    else:
        try:
            inputs = realization_inputs_from_freezeout(freezeout)
            payload = produce_conditional_resampling_realization(
                inputs, seed=seed, empirical_sweeps=empirical_sweeps
            )
        except (ValueError, KeyError) as exc:
            payload = {
                "schema": CONDITIONAL_RESAMPLING_REALIZATION_SCHEMA,
                "CONDITIONAL_RESAMPLING_REALIZATION_RECEIPT": False,
                "skipped": True,
                "reason": f"{type(exc).__name__}: {exc}",
            }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
