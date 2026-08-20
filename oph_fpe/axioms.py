"""Canonical A1--A3 axiom manifest for simulator runs.

The theory repository's machine registry (``claims/axiom_registry.yaml``)
is the normative source of the three-axiom basis.  This module carries a
verbatim pin of the canonical statements (``data/theory/
axiom_registry_pin.json``, pinned by commit and content hash) and maps
each axiom to the simulator structures that realize a finite fragment of
it in a given run configuration.  The manifest states realization status
explicitly: the simulator instantiates architectural and diagnostic
fragments; carrying the axiom text promotes no physical claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PIN_PATH = Path(__file__).resolve().parent.parent / "data" / "theory" / "axiom_registry_pin.json"

AXIOM_MANIFEST_SCHEMA = "oph.sim.axiom_manifest.v1"


def load_axiom_registry_pin(pin_path: Path | None = None) -> dict[str, Any]:
    """Load the pinned verbatim copy of the canonical axiom registry."""

    path = Path(pin_path) if pin_path is not None else _PIN_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "oph.sim.axiom_registry_pin.v1":
        raise ValueError(f"unexpected axiom pin schema: {payload.get('schema')!r}")
    ids = [axiom.get("id") for axiom in payload.get("axioms", [])]
    if ids != ["A1", "A2", "A3"]:
        raise ValueError(f"axiom pin must carry exactly A1, A2, A3; found {ids}")
    return payload


def _a1_realization(config: dict[str, Any]) -> dict[str, Any]:
    graph = dict(config.get("graph", {}) or {})
    screen = dict(config.get("screen", {}) or {})
    ports = screen.get("ports_per_patch")
    true_tower_selected = graph.get("family") == "icosahedral_tower"
    return {
        "axiom": "A1",
        "realized_fragment": (
            "Declared production icosahedral-tower carrier geometry with a "
            "twelve-port per-patch interface, typed seams, records, repair "
            "moves, and checkpoints."
            if true_tower_selected
            else "Declared twelve-port observer interface evaluated on a "
            "legacy support-chart/KNN control graph."
        ),
        "simulator_structures": {
            "patch_graph_family": graph.get("family"),
            "patch_count": graph.get("patch_count"),
            "neighbors_per_patch": graph.get("neighbors"),
            "screen_chart": screen.get("chart"),
            "carrier": screen.get("carrier"),
            "ports_per_patch": ports,
            "seam_tables": "oph_fpe/core/echosahedral_federation.py (30-seam/12-port canonical tables)",
            "records_and_checkpoints": "oph_fpe/core/patch_state.py (record, candidate_record, commit_count)",
            "repair_moves": "oph_fpe/dynamics/repair.py, oph_fpe/repair/transaction.py",
        },
        "theory_correspondence": {
            "lean_theorems": [
                "Lean/Screen/CarrierUniqueness.lean",
                "Lean/Screen/DiscreteRefinement.lean",
            ],
            "code_certificate": (
                "code/a5_closure/receipts/"
                "echosahedral_federation_reference.receipt.json"
            ),
            "simulator_receipts": [
                "TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT",
                "A5_EQUIVARIANT_REFINEMENT_RECEIPT",
                "OPERATIONAL_SELF_READING_OBSERVER_RECEIPT",
                "PHYSICAL_FEDERATION_SUPPORT_BRIDGE_RECEIPT",
            ],
        },
        "twelve_port_carrier_declared": ports == 12,
        "production_icosahedral_tower_selected": true_tower_selected,
        "full_axiom_instantiated": False,
        "unrealized_axiom_clauses": [
            "source-reconstructed faithful port-response map D",
            "commutator closure and public-response completeness from ordered histories",
        ],
        "realization_status": (
            "architectural fragment on the selected production tower" if true_tower_selected
            else "declared carrier interface on a legacy/control graph family"
        ),
    }


def _a2_realization(config: dict[str, Any]) -> dict[str, Any]:
    observers = dict(config.get("observers", {}) or {})
    return {
        "axiom": "A2",
        "realized_fragment": (
            "Observer agreement on jointly interpreted data, checked as pair "
            "re-gauging defects, cocycle triples, and integer chart verdicts "
            "over materialized observer views."
        ),
        "simulator_structures": {
            "observer_sample_count": observers.get("sample_count"),
            "observer_neighborhood_size": observers.get("neighborhood_size"),
            "agreement_certificate": "observer_agreement_report.json (observer-agreement-report CLI)",
            "gauge_overlap_state": "s3_gauge_state.npz (edge re-gauging input)",
            "parity_null_control": "shuffled-incidence and parity null controls in the agreement lane",
        },
        "theory_correspondence": {
            "lean_theorems": ["Lean/Screen/A2HolonomyBridge.lean"],
            "code_certificate": (
                "code/a5_closure/receipts/port_current_inner_reference.receipt.json"
            ),
            "simulator_receipts": [
                "observer_agreement_report.json",
                "data/repair_closure/vertex12_a2_endpoint_commutator_receipt.json",
            ],
        },
        "full_axiom_instantiated": False,
        "unrealized_axiom_clauses": [
            "surjective proper-recharting holonomy from accepted source histories",
            "same-response projective implementers for every proper carrier automorphism",
        ],
        "realization_status": (
            "diagnostic fragment; naturality is audited on the realized "
            "overlap data, not proved for all recharting maps; the algebraic "
            "current fixture is an exact named realization, not source instantiation"
        ),
    }


def _a3_realization(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "axiom": "A3",
        "realized_fragment": (
            "Exact conditional-resampling kernel on the committed observation "
            "fibers: everything the protected record datum leaves unconstrained "
            "is resampled from the pinned reference law restricted to that "
            "finite fiber."
        ),
        "simulator_structures": {
            "kernel_producer": "oph_fpe/dynamics/conditional_resampling.py",
            "exact_recognizer": (
                "oph_fpe/quotient/observable_normal_form.py "
                "recognize_conditional_resampling_kernel (R1-R3 over the rationals)"
            ),
            "run_receipt": "conditional_resampling_realization_receipt.json",
            "theorem_reference": (
                "extra/observable_normal_forms.tex:thm:fiber-conditional-expectation; "
                "Lean/ObservableNormalForms/ObservableNormalForms/"
                "ConditionalResampling.lean:"
                "kernel_eq_conditionalResamplingKernel_iff_recognition"
            ),
        },
        "theory_correspondence": {
            "lean_theorems": [
                "Lean/ObservableNormalForms/ObservableNormalForms/ConditionalResampling.lean",
                "Lean/Screen/EqualStateWeights.lean",
            ],
            "code_certificates": [
                "code/a5_closure/manifests/equal_state_weights_reference.json",
                "code/a5_closure/manifests/a3_scheduler_kernel_reference.json",
            ],
            "simulator_receipts": ["conditional_resampling_realization_receipt.json"],
        },
        "full_axiom_instantiated": False,
        "unrealized_axiom_clauses": [
            "complete A1-generated observable grammar for the full run state",
            "factorization of every A2-visible constraint through that grammar",
            "unique information projection on the complete feasible family",
        ],
        "realization_status": (
            "exact finite realization on the run's committed record classes "
            "with a pinned common reference; this is a conditional-resampling "
            "kernel counterpart, not a full A3 information-projection instantiation"
        ),
    }


def axiom_manifest(config: dict[str, Any] | None = None, *, pin_path: Path | None = None) -> dict[str, Any]:
    """Build the per-run axiom manifest: verbatim canonical statements plus
    the map from each axiom to the simulator structures realizing it."""

    pin = load_axiom_registry_pin(pin_path)
    cfg = dict(config or {})
    return {
        "schema": AXIOM_MANIFEST_SCHEMA,
        "canonical_source": pin["source"],
        "axioms": pin["axioms"],
        "simulator_realizations": [
            _a1_realization(cfg),
            _a2_realization(cfg),
            _a3_realization(cfg),
        ],
        "claim_boundary": pin["claim_boundary"],
    }
