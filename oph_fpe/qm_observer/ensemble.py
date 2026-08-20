"""Record-ensemble counting path of the observer-frame probe.

Exploratory, non-evidential. This module enumerates finite deterministic
record ensembles and tallies integer outcome counts. Integers and
Fractions only: no amplitude, no matrix, no trace, no probability
postulate, no sampling, no floating point. The declared branch tables of
``tables.py`` are its only data source. Independence from the amplitude
path is receipted by the import-graph check in ``receipt.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm

from oph_fpe.qm_observer import tables


class QMObserverError(ValueError):
    """Typed fail-closed probe error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise QMObserverError(code, message)


@dataclass(frozen=True)
class MicroConfiguration:
    """One deterministic micro-configuration: a record class label and the
    tuple of (context, outcome) records it carries."""

    label: str
    history: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class Ensemble:
    """A finite deterministic ensemble of micro-configurations."""

    configs: tuple[MicroConfiguration, ...]

    @property
    def mass(self) -> int:
        return len(self.configs)

    @property
    def last_context(self) -> str | None:
        if not self.configs or not self.configs[0].history:
            return None
        return self.configs[0].history[-1][0]


def base_ensemble() -> Ensemble:
    """The committed run population: 111 micro-configurations in class
    rec0 and 68 in class rec1 (the run literals of the reference receipt
    inputs)."""

    configs: list[MicroConfiguration] = []
    for label, multiplicity in tables.BASE_POPULATION:
        require(
            label in tables.BRANCH_TABLES,
            "BASE_CLASS",
            f"base class {label} carries no branch table",
        )
        require(
            isinstance(multiplicity, int) and multiplicity > 0,
            "BASE_MASS",
            f"base class {label} carries no positive integer mass",
        )
        for _ in range(multiplicity):
            configs.append(MicroConfiguration(label=label))
    return Ensemble(configs=tuple(configs))


def branch_fraction(label: str, effect_key: str) -> Fraction:
    """The declared outcome-0 fraction of one class under one effect."""

    require(
        label in tables.BRANCH_TABLES,
        "CLASS_UNKNOWN",
        f"class {label} carries no branch table",
    )
    row = tables.BRANCH_TABLES[label]
    require(
        effect_key in row,
        "EFFECT_UNKNOWN",
        f"class {label} carries no entry for effect {effect_key}",
    )
    numerator, denominator = row[effect_key]
    require(
        isinstance(numerator, int) and isinstance(denominator, int),
        "TABLE_SHAPE",
        f"class {label}, effect {effect_key}: entry is not an integer pair",
    )
    value = Fraction(numerator, denominator)
    require(
        0 <= value <= 1,
        "TABLE_RANGE",
        f"class {label}, effect {effect_key}: fraction outside [0, 1]",
    )
    return value


def apply_context(ens: Ensemble, context_name: str) -> Ensemble:
    """Uniform refinement and deterministic outcome assignment (DESIGN.md
    section 3.2): every micro-configuration splits into L sub-configurations
    with L the least common multiple of the class denominators, and the
    first L * num / den of them record outcome 0."""

    require(
        ens.mass > 0, "EMPTY_ENSEMBLE", "context applied to an empty ensemble"
    )
    require(
        context_name in tables.CONTEXT_EFFECT,
        "CONTEXT_UNKNOWN",
        f"context {context_name} is not a committed context",
    )
    effect_key = tables.CONTEXT_EFFECT[context_name]
    class_fractions = {
        config.label: branch_fraction(config.label, effect_key)
        for config in ens.configs
    }
    refinement = lcm(
        *sorted({value.denominator for value in class_fractions.values()})
    )
    refined: list[MicroConfiguration] = []
    for config in ens.configs:
        fraction = class_fractions[config.label]
        require(
            refinement * fraction.numerator % fraction.denominator == 0,
            "REFINEMENT",
            f"class {config.label}: refinement does not clear the denominator",
        )
        threshold = refinement * fraction.numerator // fraction.denominator
        for j in range(refinement):
            outcome = 0 if j < threshold else 1
            refined.append(
                MicroConfiguration(
                    label=tables.CONDITIONED_CLASS[(effect_key, outcome)],
                    history=config.history + ((context_name, outcome),),
                )
            )
    return Ensemble(configs=tuple(refined))


def outcome_counts(ens: Ensemble) -> tuple[int, int]:
    """Exact integer tally of the most recent recorded outcome, obtained by
    explicit enumeration of the micro-configurations."""

    require(ens.mass > 0, "EMPTY_ENSEMBLE", "tally over an empty ensemble")
    context = ens.last_context
    require(
        context is not None, "NO_RECORD", "tally over an unmeasured ensemble"
    )
    zero = 0
    one = 0
    for config in ens.configs:
        require(
            bool(config.history) and config.history[-1][0] == context,
            "RECORD_MIXED",
            "micro-configurations disagree on the most recent context",
        )
        if config.history[-1][1] == 0:
            zero += 1
        else:
            one += 1
    return zero, one


def outcome_ratio(ens: Ensemble) -> Fraction:
    """The exact outcome-0 count ratio of the most recent measurement."""

    zero, _ = outcome_counts(ens)
    return Fraction(zero, ens.mass)


def class_counts(ens: Ensemble) -> dict[str, int]:
    """Integer tally of the record classes present in the ensemble."""

    tally: dict[str, int] = {}
    for config in ens.configs:
        tally[config.label] = tally.get(config.label, 0) + 1
    return dict(sorted(tally.items()))


def condition(ens: Ensemble, context_name: str, outcome: int) -> Ensemble:
    """Collapse as conditioning: the sub-ensemble whose record shows the
    stated outcome of the most recent context. Conditioning on an outcome
    realized by zero micro-configurations fails closed."""

    require(outcome in (0, 1), "OUTCOME_RANGE", "outcome must be 0 or 1")
    require(
        ens.last_context == context_name,
        "CONDITION_CONTEXT",
        f"most recent recorded context is not {context_name}",
    )
    selected = tuple(
        config for config in ens.configs if config.history[-1][1] == outcome
    )
    require(
        len(selected) > 0,
        "CONDITION_NULL",
        f"outcome {outcome} of context {context_name} is realized by zero "
        "micro-configurations",
    )
    return Ensemble(configs=selected)
