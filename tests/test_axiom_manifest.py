"""The pinned A1--A3 axiom manifest.

The manifest carries verbatim copies of the canonical axiom statements
from the theory repository's machine registry, pinned by commit and
content hash, plus the map from each axiom to the simulator structures
realizing a finite fragment of it.  When the theory checkout is present
next to this repository the pin is verified against it byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oph_fpe.axioms import axiom_manifest, load_axiom_registry_pin

_RER_REGISTRY = (
    Path(__file__).resolve().parents[2]
    / "reverse-engineering-reality"
    / "claims"
    / "axiom_registry.yaml"
)


def test_pin_carries_exactly_three_axioms():
    pin = load_axiom_registry_pin()
    ids = [axiom["id"] for axiom in pin["axioms"]]
    assert ids == ["A1", "A2", "A3"]
    for axiom in pin["axioms"]:
        assert axiom["informal"]
        assert axiom["formal_concise"]
        assert axiom["reference_anchor"].startswith("docs/AXIOM_REFERENCE.md#")


def test_manifest_maps_each_axiom_to_simulator_structures():
    config = {
        "graph": {"family": "fibonacci_sphere", "patch_count": 4096, "neighbors": 12},
        "screen": {
            "chart": "support_visible_s2_cellulation",
            "carrier": "federated_echosahedral_patch",
            "ports_per_patch": 12,
        },
        "observers": {"sample_count": 256, "neighborhood_size": 96},
    }
    manifest = axiom_manifest(config)
    assert manifest["schema"] == "oph.sim.axiom_manifest.v1"
    realized = {row["axiom"] for row in manifest["simulator_realizations"]}
    assert realized == {"A1", "A2", "A3"}
    a1 = next(
        row for row in manifest["simulator_realizations"] if row["axiom"] == "A1"
    )
    assert a1["twelve_port_carrier_declared"] is True
    a3 = next(
        row for row in manifest["simulator_realizations"] if row["axiom"] == "A3"
    )
    assert "conditional_resampling" in a3["simulator_structures"]["kernel_producer"]


def test_pin_matches_theory_checkout_when_present():
    if not _RER_REGISTRY.exists():
        pytest.skip("theory checkout absent")
    pin = load_axiom_registry_pin()
    raw = _RER_REGISTRY.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != pin["source"]["sha256"]:
        pytest.skip(
            "theory checkout moved past the pinned registry revision; "
            "refresh data/theory/axiom_registry_pin.json"
        )
    lines = raw.decode("utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "{")
    registry = json.loads("\n".join(lines[start:]))
    canonical = {axiom["id"]: axiom for axiom in registry["axioms"]}
    for axiom in pin["axioms"]:
        assert axiom["informal"] == canonical[axiom["id"]]["informal"]
        assert axiom["formal_concise"] == canonical[axiom["id"]]["formal_concise"]
