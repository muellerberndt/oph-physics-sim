"""Receipt and export layer of the observer-frame probe.

Exploratory, non-evidential. This module runs the counting path and the
amplitude path on every committed scenario, demands the exact identities
of ``DESIGN.md`` sections 3 to 5, verifies the declared branch tables
against the trace formula, receipts the import-graph independence of the
two paths, and builds the byte-reproducible receipt JSON
(``oph.sim.qm_observer_probe.v1``) plus the visualizer export JSON
(``oph.sim.qm_observer_viz.v1``). Every counted quantity is an integer or
a Fraction; floats appear only inside display blocks labeled derived.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from fractions import Fraction
from math import lcm
from pathlib import Path
from typing import Any

from oph_fpe.core.charged_response import canonical_sha256
from oph_fpe.qm_observer import amplitudes, ensemble, tables
from oph_fpe.qm_observer.ensemble import QMObserverError, require
from oph_fpe.quantum import phase_operation as po

SCHEMA_RECEIPT = "oph.sim.qm_observer_probe.v1"
SCHEMA_VIZ = "oph.sim.qm_observer_viz.v1"

PACKAGE_DIR = Path(__file__).resolve().parent

PROBE_CONTEXTS = ("web_diagonal", "web_conjugated_3", "phase")
INTERFERENCE_FIRST = "web_conjugated_3"
INTERFERENCE_SECOND = "web_diagonal"

LABELS = {
    "exploratory": True,
    "evidential": False,
    "statement": (
        "Exploratory, non-evidential probe: observer-frame record counting "
        "reproduces the exact committed quantum weights without a "
        "probability postulate, and conditioning on the observer's record "
        "reproduces projection under the declared record-persistence and "
        "conditioned-class conventions. No instrument, no freeze, no "
        "evidential run."
    ),
}

BOUNDARY_STATEMENT = (
    "The phase effect is a declared phase-sensitive effect under the "
    "PR-04 recorded decision (2026-08-18, disposition axiomatize, lane "
    "issue 730), not derived from source dynamics. This probe establishes "
    "that observer-frame record counting reproduces the exact quantum "
    "weights without a probability postulate, and that conditioning "
    "reproduces projection under the declared record-persistence and "
    "conditioned-class conventions. It does not derive the Hilbert-space "
    "structure, does not source-produce a phase-sensitive effect, and does "
    "not create evidence: no instrument, no freeze, no evidential run. "
    "INS-01 remains the controlling OL-A1 verdict; OL-C5 status is "
    "untouched; the PR-52 physical attachments and the source-production "
    "obligation on register row PR-04 stay open; PR-64 owns "
    "completely-positive trace-nonincreasing outcome maps and the "
    "trace-preserving summed channel; PR-65 owns source provenance, common "
    "preparation, public outcomes, and custody."
)

SEMANTICS_STATEMENT = (
    "Observer-frame statistics by exhaustive deterministic record "
    "counting: a finite ensemble of micro-configurations, each producing "
    "exactly one recorded outcome under a declared integer branch rule; "
    "the statistic of an outcome is its exact integer count, a ratio of "
    "counts is a Fraction of two integers, and collapse is conditioning "
    "on the observer's record. No probability postulate, no sampling, no "
    "randomness, no floating-point arithmetic on any counted quantity."
)

SOURCES = {
    "sim_phase_operation": "oph_fpe/quantum/phase_operation.py",
    "reference_receipt": (
        "reverse-engineering-reality/code/phase_operation_producer/"
        "PHASE_OPERATION_RECEIPT.v1.json"
    ),
    "reference_payload_sha256": po.REFERENCE_PAYLOAD_SHA256,
    "lueders_rule": "reverse-engineering-reality/Lean/EventAlgebra/Lueders.lean",
    "determination": (
        "reverse-engineering-reality/Lean/EventAlgebra/"
        "PhaseInstrumentDetermination.lean"
    ),
    "conjugation_orbit": "reverse-engineering-reality/Lean/QFT/ConjugationGauge.lean",
    "claim_boundary": (
        "reverse-engineering-reality/paper/observers_are_all_you_need.tex, "
        "Born passages; Lean/EventAlgebra/FiniteBornFrame.lean"
    ),
    "design": "oph_fpe/qm_observer/DESIGN.md",
}

CONVENTIONS = (
    "base ensemble population (111 in rec0, 68 in rec1) reads the "
    "committed run literals as an exhaustive record population",
    "branch tables are declared literal data on the counting path, "
    "transcribed from the exact conditional weights Tr(p F)",
    "uniform refinement by the least common multiple of class denominators",
    "record-persistence rule: a conditioned class answers its own context "
    "with its conditioning outcome on every micro-configuration",
    "probe set {web_diagonal, web_conjugated_3, phase} for collapse chains",
    "class labels rec0, rec1, A0, A1, B0, B1, Y0, Y1",
    "interference comparison at the common mass 2864",
    "committed integer window checked on the base phase pair only; "
    "conditioned sub-ensembles are outside the committed core",
    "export schema oph.sim.qm_observer_viz.v1 and receipt schema "
    "oph.sim.qm_observer_probe.v1",
)

FRACTION_ENCODING = (
    "every exact rational is a two-integer array [numerator, denominator]"
)

FORBIDDEN_IMPORT_TOKENS = {
    "tables.py": ("amplitudes", "ensemble", "oph_fpe.quantum", "phase_operation"),
    "ensemble.py": ("amplitudes", "oph_fpe.quantum", "phase_operation"),
    "amplitudes.py": ("ensemble", "tables", "oph_fpe.qm_observer"),
}

# Dynamic-import call forms rejected outright in the inspected modules:
# none of the three carries a legitimate dynamic import, so any occurrence
# is an escape from the static import contract.
DYNAMIC_IMPORT_CALLS = (
    "__import__",
    "import_module",
    "importlib.import_module",
)


def fraction_pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


# ---------------------------------------------------------------------------
# Independence receipt
# ---------------------------------------------------------------------------


def imports_from_source(source: str) -> list[str]:
    """Absolute import names found by AST parsing, with from-imports
    expanded to dotted candidates."""

    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = "oph_fpe.qm_observer" + (
                    "." + module if module else ""
                )
            if module:
                names.add(module)
            for alias in node.names:
                names.add((module + "." if module else "") + alias.name)
    return sorted(names)


def _call_name(node: ast.Call) -> str | None:
    """The dotted name of a call target, when the target is a name or an
    attribute chain over names."""

    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        value: ast.expr = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return None


def dynamic_imports_from_source(source: str) -> list[str]:
    """Dynamic-import call nodes found by AST parsing: ``__import__`` and
    ``importlib.import_module`` in any binding, each rendered with its
    literal string argument when the call carries one."""

    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name is None:
            continue
        if name not in DYNAMIC_IMPORT_CALLS and not name.endswith(
            ".import_module"
        ):
            continue
        argument = "<unresolved>"
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            argument = first.value
        found.add(f"{name}({argument})")
    return sorted(found)


def check_source_independence(filename: str, source: str) -> list[str]:
    """Fail-closed check of one module source against its forbidden import
    tokens and against the dynamic-import call forms. A mutant that computes
    counts from amplitudes by a static cross-import, by ``__import__``, or by
    ``importlib.import_module`` is rejected here; the check covers those
    forms and no other route between the two paths."""

    require(
        filename in FORBIDDEN_IMPORT_TOKENS,
        "INDEPENDENCE_SCOPE",
        f"{filename} carries no independence contract",
    )
    imports = imports_from_source(source)
    forbidden = FORBIDDEN_IMPORT_TOKENS[filename]
    violations = sorted(
        {name for name in imports for token in forbidden if token in name}
    )
    require(
        not violations,
        "IMPORT_INDEPENDENCE",
        f"{filename} imports {violations}",
    )
    dynamic = dynamic_imports_from_source(source)
    require(
        not dynamic,
        "IMPORT_INDEPENDENCE",
        f"{filename} carries dynamic-import calls {dynamic}",
    )
    return imports


def check_independence() -> dict[str, Any]:
    """The import-graph independence receipt over the package sources, with
    the sha256 of every inspected source pinning the code that produced the
    counts."""

    modules: dict[str, Any] = {}
    for filename in sorted(FORBIDDEN_IMPORT_TOKENS):
        raw = (PACKAGE_DIR / filename).read_bytes()
        source = raw.decode("utf-8")
        modules[filename] = {
            "imports": check_source_independence(filename, source),
            "forbidden_tokens": list(FORBIDDEN_IMPORT_TOKENS[filename]),
            "dynamic_import_calls": dynamic_imports_from_source(source),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
        }
    return {
        "modules": modules,
        "verdict": (
            "the counting path and the amplitude path share no static "
            "import edge and no __import__ or importlib.import_module call; "
            "each inspected source is pinned by its sha256"
        ),
    }


# ---------------------------------------------------------------------------
# Branch-table verification against the trace formula
# ---------------------------------------------------------------------------


def verify_tables() -> dict[str, Any]:
    """Every declared branch-table entry equals the exact trace conditional
    Tr(p F), and every conditioned-class projector equals its outcome
    effect."""

    checked = 0
    for label in sorted(tables.BRANCH_TABLES):
        for effect_key, (numerator, denominator) in sorted(
            tables.BRANCH_TABLES[label].items()
        ):
            declared = Fraction(numerator, denominator)
            traced = amplitudes.effect_conditional_weight(label, effect_key)
            require(
                declared == traced,
                "TABLE_TRACE",
                f"class {label}, effect {effect_key}: declared {declared} "
                f"differs from the trace value {traced}",
            )
            checked += 1
    for (effect_key, outcome), label in sorted(tables.CONDITIONED_CLASS.items()):
        require(
            amplitudes.class_projector(label)
            == amplitudes.outcome_effect_by_key(effect_key, outcome),
            "TABLE_CLASS",
            f"conditioned class {label}: projector differs from the outcome "
            f"effect of ({effect_key}, {outcome})",
        )
    return {
        "entries_checked": checked,
        "verdict": (
            "declared branch tables equal the exact trace conditionals"
        ),
    }


# ---------------------------------------------------------------------------
# Count-ratio identities on the committed contexts
# ---------------------------------------------------------------------------

_COMMITTED: dict[str, Any] = {}


def committed_receipt() -> dict[str, Any]:
    if not _COMMITTED:
        _COMMITTED.update(po.build_receipt())
    return _COMMITTED


def committed_row(context_name: str) -> dict[str, Any]:
    for row in committed_receipt()["contexts"]:
        if row["context"] == context_name:
            return row
    raise QMObserverError(
        "COMMITTED_ROW", f"context {context_name} carries no committed row"
    )


def base_context_identity(
    context_name: str, ens: ensemble.Ensemble | None = None
) -> dict[str, Any]:
    """The receipted identity of DESIGN.md section 3.4: enumerated count
    ratio = exact amplitude weight = committed exact rational weight, with
    counts and mass equal to the committed row."""

    if ens is None:
        ens = ensemble.base_ensemble()
    applied = ensemble.apply_context(ens, context_name)
    zero, one = ensemble.outcome_counts(applied)
    mass = applied.mass
    ratio = Fraction(zero, mass)
    amplitude = amplitudes.born(amplitudes.base_state(), context_name, 0)
    row = committed_row(context_name)
    committed_weight = Fraction(row["born_weight"])
    require(
        ratio == amplitude,
        "COUNT_RATIO",
        f"context {context_name}: count ratio {ratio} differs from the "
        f"amplitude weight {amplitude}",
    )
    require(
        ratio == committed_weight,
        "COUNT_RATIO",
        f"context {context_name}: count ratio {ratio} differs from the "
        f"committed weight {committed_weight}",
    )
    require(
        [zero, one] == list(row["counts"]) and mass == row["mass"],
        "COUNT_SCALE",
        f"context {context_name}: counts ({zero}, {one}) at mass {mass} "
        "differ from the committed row",
    )
    return {
        "context": context_name,
        "effect_key": tables.CONTEXT_EFFECT[context_name],
        "counts": [zero, one],
        "mass": mass,
        "count_ratio": fraction_pair(ratio),
        "amplitude_weight": fraction_pair(amplitude),
        "committed_weight": fraction_pair(committed_weight),
        "identity": "count ratio = amplitude weight = committed weight",
    }


# ---------------------------------------------------------------------------
# Collapse as conditioning
# ---------------------------------------------------------------------------


def collapse_chain(context_name: str) -> dict[str, Any]:
    """One collapse chain: apply the declared context effect, condition on
    each realized outcome, receipt record persistence and the projection
    statistics of every probe context against the Lueders-updated state."""

    applied = ensemble.apply_context(ensemble.base_ensemble(), context_name)
    zero, one = ensemble.outcome_counts(applied)
    chain: dict[str, Any] = {
        "context": context_name,
        "measurement": {"counts": [zero, one], "mass": applied.mass},
        "conditionings": [],
    }
    for outcome in (0, 1):
        sub = ensemble.condition(applied, context_name, outcome)
        label = sub.configs[0].label
        repeat = ensemble.apply_context(sub, context_name)
        repeat_counts = ensemble.outcome_counts(repeat)
        require(
            repeat_counts[outcome] == repeat.mass
            and repeat_counts[1 - outcome] == 0,
            "PERSISTENCE",
            f"context {context_name}, outcome {outcome}: repeated context "
            "is not certain on the conditioned sub-ensemble",
        )
        post = amplitudes.lueders(
            amplitudes.base_state(), context_name, outcome
        )
        require(
            amplitudes.born(post, context_name, outcome) == 1,
            "PERSISTENCE_AMPLITUDE",
            f"context {context_name}, outcome {outcome}: Lueders "
            "repeatability fails",
        )
        probes = []
        for probe_name in PROBE_CONTEXTS:
            probed = ensemble.apply_context(sub, probe_name)
            probe_zero, probe_one = ensemble.outcome_counts(probed)
            probe_ratio = Fraction(probe_zero, probed.mass)
            lueders_weight = amplitudes.born(post, probe_name, 0)
            require(
                probe_ratio == lueders_weight,
                "COLLAPSE_PROJECTION",
                f"context {context_name}, outcome {outcome}, probe "
                f"{probe_name}: count ratio {probe_ratio} differs from the "
                f"Lueders weight {lueders_weight}",
            )
            probes.append(
                {
                    "probe": probe_name,
                    "counts": [probe_zero, probe_one],
                    "mass": probed.mass,
                    "count_ratio": fraction_pair(probe_ratio),
                    "lueders_weight": fraction_pair(lueders_weight),
                }
            )
        chain["conditionings"].append(
            {
                "outcome": outcome,
                "conditioned_class": label,
                "sub_mass": sub.mass,
                "repeat": {
                    "counts": list(repeat_counts),
                    "fraction": fraction_pair(
                        Fraction(repeat_counts[outcome], repeat.mass)
                    ),
                },
                "probes": probes,
            }
        )
    return chain


# ---------------------------------------------------------------------------
# Interference receipt
# ---------------------------------------------------------------------------


def require_interference(gap: Fraction) -> None:
    require(
        gap != 0,
        "INTERFERENCE_NULL",
        "recorded statistics equal the classical mixture prediction",
    )


def interference_receipt() -> dict[str, Any]:
    """The exact record-disturbance gap of DESIGN.md section 5 between
    the direct second-context statistics and the statistics mediated by the
    first context."""

    base = ensemble.base_ensemble()
    direct = ensemble.apply_context(base, INTERFERENCE_SECOND)
    direct_zero, direct_one = ensemble.outcome_counts(direct)
    first = ensemble.apply_context(base, INTERFERENCE_FIRST)
    first_zero, first_one = ensemble.outcome_counts(first)
    mediated = ensemble.apply_context(first, INTERFERENCE_SECOND)
    mediated_zero, mediated_one = ensemble.outcome_counts(mediated)
    branches = []
    sewn = [0, 0]
    for outcome in (0, 1):
        sub = ensemble.condition(first, INTERFERENCE_FIRST, outcome)
        probed = ensemble.apply_context(sub, INTERFERENCE_SECOND)
        branch_zero, branch_one = ensemble.outcome_counts(probed)
        sewn[0] += branch_zero
        sewn[1] += branch_one
        branches.append(
            {
                "first_outcome": outcome,
                "conditioned_class": sub.configs[0].label,
                "sub_mass": sub.mass,
                "counts": [branch_zero, branch_one],
                "mass": probed.mass,
            }
        )
    require(
        sewn == [mediated_zero, mediated_one],
        "SEWING",
        "conditioned branches do not sew to the mediated tally",
    )
    common = lcm(direct.mass, mediated.mass)
    gap = Fraction(direct_zero, direct.mass) - Fraction(
        mediated_zero, mediated.mass
    )
    require_interference(gap)
    return {
        "first_context": INTERFERENCE_FIRST,
        "second_context": INTERFERENCE_SECOND,
        "direct": {"counts": [direct_zero, direct_one], "mass": direct.mass},
        "first_measurement": {
            "counts": [first_zero, first_one],
            "mass": first.mass,
        },
        "mediated": {
            "counts": [mediated_zero, mediated_one],
            "mass": mediated.mass,
        },
        "branches": branches,
        "common_mass": common,
        "direct_at_common_mass": [
            direct_zero * (common // direct.mass),
            direct_one * (common // direct.mass),
        ],
        "mediated_at_common_mass": [
            mediated_zero * (common // mediated.mass),
            mediated_one * (common // mediated.mass),
        ],
        "gap": fraction_pair(gap),
        "classical_family": (
            "every model in which each micro-configuration carries "
            "simultaneous definite outcome values for both contexts "
            "and context application reveals the value without "
            "rewriting the record satisfies direct = mediated; the "
            "exact gap excludes every member of that family. No "
            "Bell-type claim and no locality claim is carried: the "
            "receipt is a record-disturbance identity of the committed "
            "weights (DESIGN.md section 5)"
        ),
    }


# ---------------------------------------------------------------------------
# Visualizer export
# ---------------------------------------------------------------------------


def _root_node(node_id: str, ens: ensemble.Ensemble) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "context": None,
        "class_counts": ensemble.class_counts(ens),
        "mass": ens.mass,
        "counts": None,
        "weights": None,
        "children": [],
    }


def _context_node(
    node_id: str, context_name: str, applied: ensemble.Ensemble
) -> dict[str, Any]:
    zero, one = ensemble.outcome_counts(applied)
    return {
        "node_id": node_id,
        "context": context_name,
        "class_counts": ensemble.class_counts(applied),
        "mass": applied.mass,
        "counts": [zero, one],
        "weights": [
            fraction_pair(Fraction(zero, applied.mass)),
            fraction_pair(Fraction(one, applied.mass)),
        ],
        "children": [],
    }


def _base_scenario(context_name: str) -> dict[str, Any]:
    base = ensemble.base_ensemble()
    applied = ensemble.apply_context(base, context_name)
    zero, one = ensemble.outcome_counts(applied)
    root = _root_node("base", base)
    root["children"].append(
        _context_node(f"base>{context_name}", context_name, applied)
    )
    return {
        "scenario_id": f"base_context/{context_name}",
        "kind": "base_context",
        "context_sequence": [context_name],
        "tree": root,
        "collapse_events": [],
        "display": {
            "derived_for_display": True,
            "float_weights": {
                "outcome_0": zero / applied.mass,
                "outcome_1": one / applied.mass,
            },
        },
    }


def _collapse_scenario(context_name: str) -> dict[str, Any]:
    base = ensemble.base_ensemble()
    applied = ensemble.apply_context(base, context_name)
    zero, one = ensemble.outcome_counts(applied)
    root = _root_node("base", base)
    recorded = _context_node(
        f"base>{context_name}", context_name, applied
    )
    root["children"].append(recorded)
    collapse_events = []
    float_weights: dict[str, float] = {}
    for outcome in (0, 1):
        sub = ensemble.condition(applied, context_name, outcome)
        conditioned = _root_node(f"base>{context_name}={outcome}", sub)
        collapse_events.append(
            {
                "after_context": context_name,
                "conditioned_outcome": outcome,
                "before": {"mass": applied.mass, "counts": [zero, one]},
                "after": {
                    "class": sub.configs[0].label,
                    "mass": sub.mass,
                },
            }
        )
        repeat = ensemble.apply_context(sub, context_name)
        conditioned["children"].append(
            _context_node(
                f"base>{context_name}={outcome}>{context_name}",
                context_name,
                repeat,
            )
        )
        for probe_name in PROBE_CONTEXTS:
            probed = ensemble.apply_context(sub, probe_name)
            probe_zero, _ = ensemble.outcome_counts(probed)
            conditioned["children"].append(
                _context_node(
                    f"base>{context_name}={outcome}>{probe_name}",
                    probe_name,
                    probed,
                )
            )
            float_weights[f"outcome_{outcome}/{probe_name}"] = (
                probe_zero / probed.mass
            )
        recorded["children"].append(conditioned)
    return {
        "scenario_id": f"collapse_chain/{context_name}",
        "kind": "collapse_chain",
        "context_sequence": [context_name, context_name, *PROBE_CONTEXTS],
        "tree": root,
        "collapse_events": collapse_events,
        "display": {
            "derived_for_display": True,
            "float_weights": float_weights,
        },
    }


def _interference_scenario() -> dict[str, Any]:
    report = interference_receipt()
    base = ensemble.base_ensemble()
    direct = ensemble.apply_context(base, INTERFERENCE_SECOND)
    first = ensemble.apply_context(base, INTERFERENCE_FIRST)
    root = _root_node("base", base)
    root["children"].append(
        _context_node(
            f"base>{INTERFERENCE_SECOND}", INTERFERENCE_SECOND, direct
        )
    )
    first_node = _context_node(
        f"base>{INTERFERENCE_FIRST}", INTERFERENCE_FIRST, first
    )
    root["children"].append(first_node)
    mediated = ensemble.apply_context(first, INTERFERENCE_SECOND)
    first_node["children"].append(
        _context_node(
            f"base>{INTERFERENCE_FIRST}>{INTERFERENCE_SECOND}",
            INTERFERENCE_SECOND,
            mediated,
        )
    )
    collapse_events = []
    for branch in report["branches"]:
        outcome = branch["first_outcome"]
        sub = ensemble.condition(first, INTERFERENCE_FIRST, outcome)
        conditioned = _root_node(
            f"base>{INTERFERENCE_FIRST}={outcome}", sub
        )
        probed = ensemble.apply_context(sub, INTERFERENCE_SECOND)
        conditioned["children"].append(
            _context_node(
                f"base>{INTERFERENCE_FIRST}={outcome}>"
                f"{INTERFERENCE_SECOND}",
                INTERFERENCE_SECOND,
                probed,
            )
        )
        first_node["children"].append(conditioned)
        collapse_events.append(
            {
                "after_context": INTERFERENCE_FIRST,
                "conditioned_outcome": outcome,
                "before": {
                    "mass": report["first_measurement"]["mass"],
                    "counts": report["first_measurement"]["counts"],
                },
                "after": {
                    "class": branch["conditioned_class"],
                    "mass": branch["sub_mass"],
                },
            }
        )
    gap = Fraction(*report["gap"])
    return {
        "scenario_id": (
            f"interference/{INTERFERENCE_FIRST}-{INTERFERENCE_SECOND}"
        ),
        "kind": "interference",
        "context_sequence": [INTERFERENCE_FIRST, INTERFERENCE_SECOND],
        "tree": root,
        "collapse_events": collapse_events,
        "comparison": {
            "common_mass": report["common_mass"],
            "direct_at_common_mass": report["direct_at_common_mass"],
            "mediated_at_common_mass": report["mediated_at_common_mass"],
            "gap": report["gap"],
            "classical_family": report["classical_family"],
        },
        "display": {
            "derived_for_display": True,
            "float_weights": {
                "direct_outcome_0": (
                    report["direct"]["counts"][0] / report["direct"]["mass"]
                ),
                "mediated_outcome_0": (
                    report["mediated"]["counts"][0]
                    / report["mediated"]["mass"]
                ),
                "gap": gap.numerator / gap.denominator,
            },
        },
    }


def build_export() -> dict[str, Any]:
    """The visualizer export (schema oph.sim.qm_observer_viz.v1)."""

    scenarios = [
        _base_scenario(name) for name, _ in tables.CONTEXTS
    ]
    scenarios += [_collapse_scenario(name) for name, _ in tables.CONTEXTS]
    scenarios.append(_interference_scenario())
    return {
        "schema": SCHEMA_VIZ,
        "labels": LABELS,
        "boundary": BOUNDARY_STATEMENT,
        "fraction_encoding": FRACTION_ENCODING,
        "scenarios": scenarios,
    }


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def reference_cross_check() -> dict[str, Any]:
    """Fail-closed replay of the committed sim receipt against the
    committed reference receipt file, when the checkout carries it."""

    available = po.DEFAULT_REFERENCE_RECEIPT.is_file()
    result: dict[str, Any] = {
        "available": available,
        "reference_payload_sha256": po.REFERENCE_PAYLOAD_SHA256,
    }
    if available:
        reference = po.load_json(po.DEFAULT_REFERENCE_RECEIPT)
        report = po.replay_against_reference(committed_receipt(), reference)
        result["contexts_replayed"] = report["contexts_replayed"]
        result["verdict"] = "committed rows replay the reference receipt"
    return result


def build_receipt() -> dict[str, Any]:
    """The probe receipt (schema oph.sim.qm_observer_probe.v1), with a
    canonical sha over its own body and over the export body."""

    identities = [
        base_context_identity(name) for name, _ in tables.CONTEXTS
    ]
    phase_rows = [row for row in identities if row["context"] == "phase"]
    require(
        len(phase_rows) == 1,
        "PHASE_ROW",
        "the count-ratio identities carry no single phase row",
    )
    phase_counts = phase_rows[0]["counts"]
    window_lhs, window_rhs = po.check_phase_window(phase_counts)
    receipt: dict[str, Any] = {
        "schema": SCHEMA_RECEIPT,
        "labels": LABELS,
        "boundary": BOUNDARY_STATEMENT,
        "semantics": SEMANTICS_STATEMENT,
        "sources": SOURCES,
        "conventions": list(CONVENTIONS),
        "fraction_encoding": FRACTION_ENCODING,
        "count_ratio_identities": identities,
        "phase_receipt_window": {
            "inequality": "32041*(a - b)^2 <= 30192*(a + b)^2",
            "counts": phase_counts,
            "lhs": window_lhs,
            "rhs": window_rhs,
            "satisfied": window_lhs <= window_rhs,
            "scope": (
                "base ensemble phase pair only; conditioned sub-ensembles "
                "are outside the committed core. A conditioned phase pair "
                "carries a zero-mass outcome and fails WINDOW_POSITIVITY, "
                "so the window is never evaluated on one. On any "
                "record-diagonal base state the phase pair is balanced and "
                "lhs is identically 0, so the window clause holds "
                "vacuously and discriminates nothing here."
            ),
        },
        "collapse_chains": [
            collapse_chain(name) for name, _ in tables.CONTEXTS
        ],
        "interference": interference_receipt(),
        "branch_table_verification": verify_tables(),
        "independence": check_independence(),
        "reference_cross_check": reference_cross_check(),
        "export": {
            "schema": SCHEMA_VIZ,
            "body_sha256": canonical_sha256(build_export()),
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory receiving the receipt and the visualizer export",
    )
    args = parser.parse_args()
    receipt = build_receipt()
    export = build_export()
    require(
        receipt["export"]["body_sha256"] == canonical_sha256(export),
        "EXPORT_DRIFT",
        "export body differs from the hashed body",
    )
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = args.output_dir / "QM_OBSERVER_RECEIPT.v1.json"
        export_path = args.output_dir / "QM_OBSERVER_VIZ.v1.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        export_path.write_text(
            json.dumps(export, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {receipt_path}")
        print(f"wrote {export_path}")
        print(f"receipt_sha256 {receipt['receipt_sha256']}")
        print(f"export_body_sha256 {receipt['export']['body_sha256']}")
        return
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
