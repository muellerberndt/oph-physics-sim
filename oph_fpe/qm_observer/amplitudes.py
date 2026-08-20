"""Amplitude path of the observer-frame probe.

Exploratory, non-evidential. This module computes exact squared-amplitude
Born weights and Lueders updates over Q(sqrt(3), i) through the committed
PR-04 machinery of ``oph_fpe.quantum.phase_operation``, imported
read-only. It never imports the counting modules and never references a
count; independence from the counting path is receipted by the
import-graph check in ``receipt.py``. Conditioning follows the Lueders
rule of the committed corpus (Lean/EventAlgebra/Lueders.lean):
``rho -> P rho P / Tr(rho P)``.
"""

from __future__ import annotations

from fractions import Fraction

from oph_fpe.quantum import phase_operation as po

_CACHE: dict[str, object] = {}


def _built() -> dict[str, object]:
    if not _CACHE:
        projectors = po.context_projectors()
        operation = po.declared_phase_operation()
        effects = dict(po.named_effects())
        effect_by_key = {
            "P_rec": po.RECORD_PROJECTOR,
            "E_A": projectors[3],
            "E_B": projectors[4],
            "Y_plus": operation,
        }
        class_projectors = {
            "rec0": po.RECORD_PROJECTOR,
            "rec1": po.msub(po.IDENTITY, po.RECORD_PROJECTOR),
            "A0": projectors[3],
            "A1": po.msub(po.IDENTITY, projectors[3]),
            "B0": projectors[4],
            "B1": po.msub(po.IDENTITY, projectors[4]),
            "Y0": operation,
            "Y1": po.msub(po.IDENTITY, operation),
        }
        _CACHE["effects"] = effects
        _CACHE["effect_by_key"] = effect_by_key
        _CACHE["class_projectors"] = class_projectors
    return _CACHE


def effect(context_name: str) -> po.Mat:
    """The exact effect matrix of one committed context."""

    effects = _built()["effects"]
    po.require(
        context_name in effects,
        "CONTEXT_UNKNOWN",
        f"context {context_name} is not a committed context",
    )
    return effects[context_name]


def outcome_effect(context_name: str, outcome: int) -> po.Mat:
    """The outcome effect: the context effect for outcome 0, its
    complement for outcome 1."""

    po.require(outcome in (0, 1), "OUTCOME_RANGE", "outcome must be 0 or 1")
    matrix = effect(context_name)
    if outcome == 0:
        return matrix
    return po.msub(po.IDENTITY, matrix)


def outcome_effect_by_key(effect_key: str, outcome: int) -> po.Mat:
    """The outcome effect addressed by effect key."""

    po.require(outcome in (0, 1), "OUTCOME_RANGE", "outcome must be 0 or 1")
    effect_by_key = _built()["effect_by_key"]
    po.require(
        effect_key in effect_by_key,
        "EFFECT_UNKNOWN",
        f"effect key {effect_key} is not declared",
    )
    matrix = effect_by_key[effect_key]
    if outcome == 0:
        return matrix
    return po.msub(po.IDENTITY, matrix)


def class_projector(label: str) -> po.Mat:
    """The exact projector of one record class label."""

    class_projectors = _built()["class_projectors"]
    po.require(
        label in class_projectors,
        "CLASS_UNKNOWN",
        f"class {label} carries no projector",
    )
    return class_projectors[label]


def base_state() -> po.Mat:
    """The committed record-diagonal run state diag(111/179, 68/179)."""

    return po.record_diagonal_state(*po.RUN_COUNTS)


def born(state: po.Mat, context_name: str, outcome: int = 0) -> Fraction:
    """The exact Born weight Tr(state * effect) of one context outcome."""

    return po.born_weight(state, outcome_effect(context_name, outcome))


def effect_conditional_weight(label: str, effect_key: str) -> Fraction:
    """The exact conditional weight Tr(p F) of an effect on a class
    projector."""

    return po.born_weight(
        class_projector(label), outcome_effect_by_key(effect_key, 0)
    )


def lueders(state: po.Mat, context_name: str, outcome: int) -> po.Mat:
    """The Lueders update ``P rho P / Tr(rho P)`` for one context outcome.
    A weight-zero outcome fails closed."""

    projector = outcome_effect(context_name, outcome)
    weight = po.born_weight(state, projector)
    po.require(
        weight > 0,
        "LUEDERS_NULL",
        f"context {context_name}, outcome {outcome}: conditioning on a "
        "weight-zero outcome",
    )
    compressed = po.mmul(po.mmul(projector, state), projector)
    updated = po.mscale(po.C3.of(Fraction(1) / weight), compressed)
    po.require(
        po.mtrace(updated) == po.C3.of(1),
        "LUEDERS_TRACE",
        "updated state trace is not one",
    )
    return updated
