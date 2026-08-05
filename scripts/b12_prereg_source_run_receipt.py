"""B12 preregistered-run comparison receipt.

Verifies the fresh bounded source run against every gate declared in
``docs/B12_PREREGISTERED_SOURCE_RUN_2026-08-05.md`` (precondition P0 and
gates G1 through G4), recomputing the recomputable gates from the raw
run artifacts with exact rational arithmetic, and writes
``docs/B12_PREREGISTERED_SOURCE_RUN_2026-08-05_receipt.json``.

The script lives outside the pinned ``oph_fpe/`` tree.  It imports only
the declared binning and candidate-chain constants from the pinned
producer module (their definitions are part of the preregistration); the
joint reference table, the fiber kernel, the stationarity equation, and
the chi-squared comparisons are rebuilt inline.
"""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from oph_fpe.dynamics.conditional_resampling import (  # noqa: E402
    _COMPANION_CANDIDATES,
    _MAX_COMPANION_CLASSES,
    _MAX_RECORD_CLASSES,
    _class_bins,
)

# Declared constants (Section 2 of the preregistration document).
RUN_ID = "b12_prereg_16k_20260806"
DECLARED_SEED = 20260806
DECLARED_PATCH_CAP = 16384
DECLARED_OBSERVER_CAP = 1024
DECLARED_MIN_RECORD_CLASSES = 8
PREREG_DOC = REPO / "docs" / "B12_PREREGISTERED_SOURCE_RUN_2026-08-05.md"
PREREG_DOC_SHA256 = "cde3ddfabf2e33908f02f6c7a28e7385c9c9c90d489425e29aebe975cab8872d"
DERIVED_CONFIG = REPO / "configs" / "local" / "b12_prereg_16k_20260806.yml"
DERIVED_CONFIG_SHA256 = "ba1ec33e87f7920742b2f8491fc923a8d8f37bc60b7e4621a06d59f67f827302"
RUN_DIR = REPO / "runs" / RUN_ID
RECEIPT_OUT = REPO / "docs" / "B12_PREREGISTERED_SOURCE_RUN_2026-08-05_receipt.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    gates: dict[str, dict] = {}
    receipt_json = _load_json(RUN_DIR / "conditional_resampling_realization_receipt.json")
    manifest = _load_json(RUN_DIR / "manifest.json")
    stored_config = _load_yaml(RUN_DIR / "config.yml")

    # ---- P0: run conformance -------------------------------------------
    doc_hash = _sha256(PREREG_DOC)
    cfg_hash = _sha256(DERIVED_CONFIG)
    manifest_seed = int(manifest.get("rng_streams", {}).get("run_seed", -1))
    manifest_patches = int(manifest.get("patch_count", -1))
    stored_seed = int(stored_config.get("seed", -1))
    stored_observers = int(stored_config.get("observers", {}).get("sample_count", -1))
    stored_patches = int(stored_config.get("graph", {}).get("patch_count", -1))
    receipt_seed = int(receipt_json.get("empirical_realization", {}).get("seed", -1))
    receipt_patches = int(receipt_json.get("provenance", {}).get("patch_count", -1))
    p0_checks = {
        "prereg_doc_sha256_matches": doc_hash == PREREG_DOC_SHA256,
        "derived_config_sha256_matches": cfg_hash == DERIVED_CONFIG_SHA256,
        "manifest_run_seed": manifest_seed,
        "manifest_run_seed_matches": manifest_seed == DECLARED_SEED,
        "manifest_patch_count": manifest_patches,
        "manifest_patch_count_matches": manifest_patches == DECLARED_PATCH_CAP,
        "stored_config_seed_matches": stored_seed == DECLARED_SEED,
        "stored_config_observer_cap": stored_observers,
        "stored_config_observer_cap_matches": stored_observers == DECLARED_OBSERVER_CAP,
        "stored_config_patch_cap_matches": stored_patches == DECLARED_PATCH_CAP,
        "receipt_empirical_seed": receipt_seed,
        "receipt_empirical_seed_matches": receipt_seed == DECLARED_SEED,
        "receipt_provenance_patch_count_matches": receipt_patches == DECLARED_PATCH_CAP,
    }
    gates["P0_run_conformance"] = {
        "pass": all(v for k, v in p0_checks.items() if k.endswith("matches")),
        **p0_checks,
    }

    # ---- G1: recognizer receipt ----------------------------------------
    recognizer = receipt_json.get("recognizer", {})
    g1_flags = {
        "CONDITIONAL_RESAMPLING_REALIZATION_RECEIPT": bool(
            receipt_json.get("CONDITIONAL_RESAMPLING_REALIZATION_RECEIPT", False)
        ),
        "not_a_labeled_skip": not bool(receipt_json.get("skipped", False)),
        "exact_table_recognition_receipt": bool(
            recognizer.get("exact_table_recognition_receipt", False)
        ),
        "r1_fiber_supported": bool(recognizer.get("r1_fiber_supported", False)),
        "r2_fiber_rows_constant": bool(recognizer.get("r2_fiber_rows_constant", False)),
        "r3_weighted_detailed_balance": bool(
            recognizer.get("r3_weighted_detailed_balance", False)
        ),
        "explicit_formula_match": bool(recognizer.get("explicit_formula_match", False)),
    }
    gates["G1_recognizer_receipt"] = {"pass": all(g1_flags.values()), **g1_flags}

    # ---- Independent rebuild from the raw freezeout arrays -------------
    bundle = np.load(RUN_DIR / "freezeout_fields.npz", allow_pickle=True)
    record = _class_bins(bundle["record_signature"], _MAX_RECORD_CLASSES)
    companion_field = next(
        c for c in _COMPANION_CANDIDATES
        if c in bundle and np.unique(bundle[c]).size >= 2
    )
    companion = _class_bins(bundle[companion_field], _MAX_COMPANION_CLASSES)
    counts: dict[tuple[int, int], int] = {}
    for r_val, c_val in zip(record.tolist(), companion.tolist()):
        key = (int(r_val), int(c_val))
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    states = sorted(counts)
    # The single pinned common reference, rebuilt once; both the G3
    # stationarity check and the G4 chi-squared recomputation read it.
    pi = {s: Fraction(counts[s], total) for s in states}
    recomputed_record_classes = len({s[0] for s in states})

    # ---- G2: nonconstant protected record, at least 8 classes ----------
    protected = receipt_json.get("protected_record", {})
    receipt_class_count = int(protected.get("class_count", -1))
    g2_checks = {
        "receipt_nonconstant": bool(protected.get("nonconstant", False)),
        "receipt_unchanged_by_resampling": bool(
            protected.get("unchanged_by_resampling", False)
        ),
        "receipt_class_count": receipt_class_count,
        "receipt_class_count_at_least_8": receipt_class_count >= DECLARED_MIN_RECORD_CLASSES,
        "recomputed_class_count": recomputed_record_classes,
        "recomputed_class_count_at_least_8": (
            recomputed_record_classes >= DECLARED_MIN_RECORD_CLASSES
        ),
        "recomputed_equals_receipt": recomputed_record_classes == receipt_class_count,
        "recomputed_companion_field": companion_field,
        "receipt_companion_field_matches": (
            receipt_json.get("companion", {}).get("label") == companion_field
        ),
    }
    gates["G2_nonconstant_protected_record"] = {
        "pass": all(
            g2_checks[k]
            for k in (
                "receipt_nonconstant",
                "receipt_unchanged_by_resampling",
                "receipt_class_count_at_least_8",
                "recomputed_class_count_at_least_8",
                "recomputed_equals_receipt",
                "receipt_companion_field_matches",
            )
        ),
        **g2_checks,
    }

    # ---- G3: kernel stationary law equals the reference exactly --------
    fiber_mass: dict[int, Fraction] = {}
    for s in states:
        fiber_mass[s[0]] = fiber_mass.get(s[0], Fraction(0)) + pi[s]
    kernel = {
        x: {
            y: (pi[y] / fiber_mass[x[0]] if y[0] == x[0] else Fraction(0))
            for y in states
        }
        for x in states
    }
    stationary_exact = all(
        sum((pi[x] * kernel[x][y] for x in states), start=Fraction(0)) == pi[y]
        for y in states
    )
    package = receipt_json.get("exact_kernel_package", {})
    g3_checks = {
        "receipt_reference_stationary": bool(package.get("reference_stationary", False)),
        "receipt_idempotent": bool(package.get("idempotent", False)),
        "recomputed_stationarity_exact_over_Q": bool(stationary_exact),
        "reference_state_count": len(states),
        "reference_state_count_matches_receipt": (
            len(states) == int(receipt_json.get("pinned_reference", {}).get("state_count", -1))
        ),
        "reference_total_mass_count": total,
        "reference_total_mass_matches_receipt": (
            total == int(receipt_json.get("pinned_reference", {}).get("total_mass_count", -1))
        ),
    }
    gates["G3_stationary_law_equals_reference"] = {
        "pass": all(v for k, v in g3_checks.items() if isinstance(v, bool)),
        **g3_checks,
    }

    # ---- G4: chi-squared contraction -----------------------------------
    chi_before = Fraction(package.get("chi_squared_before", "1"))
    chi_after = Fraction(package.get("chi_squared_after_one_step", "0"))
    sweeps = receipt_json.get("empirical_realization", {}).get("sweeps", [])
    displaced = Fraction(sweeps[0]["chi_squared_to_reference"]) if sweeps else None
    first_sweep = Fraction(sweeps[1]["chi_squared_to_reference"]) if len(sweeps) > 1 else None
    g4_checks = {
        "receipt_chi_squared_contracts": bool(package.get("chi_squared_contracts", False)),
        "chi_squared_before": str(chi_before),
        "chi_squared_after_one_step": str(chi_after),
        "exact_after_leq_before": chi_after <= chi_before,
        "receipt_one_step_collapse_measured": bool(
            receipt_json.get("empirical_realization", {}).get(
                "one_step_collapse_measured", False
            )
        ),
        "displaced_start_chi_squared": str(displaced) if displaced is not None else None,
        "first_sweep_chi_squared": str(first_sweep) if first_sweep is not None else None,
        "exact_first_sweep_below_displaced_start": (
            displaced is not None and first_sweep is not None and first_sweep < displaced
        ),
    }
    gates["G4_chi_squared_contraction"] = {
        "pass": all(v for k, v in g4_checks.items() if isinstance(v, bool)),
        **g4_checks,
    }

    overall = all(g["pass"] for g in gates.values())
    payload = {
        "schema": "oph.sim.b12_preregistered_source_run_receipt.v1",
        "B12_PREREGISTERED_SOURCE_RUN_RECEIPT": overall,
        "preregistration": {
            "document": str(PREREG_DOC.relative_to(REPO)),
            "document_sha256": doc_hash,
            "declared_document_sha256": PREREG_DOC_SHA256,
            "derived_config": str(DERIVED_CONFIG.relative_to(REPO)),
            "derived_config_sha256": cfg_hash,
            "declared_derived_config_sha256": DERIVED_CONFIG_SHA256,
            "declared_seed": DECLARED_SEED,
            "declared_patch_cap": DECLARED_PATCH_CAP,
            "declared_observer_cap": DECLARED_OBSERVER_CAP,
            "run_id": RUN_ID,
        },
        "run_dir": str(RUN_DIR.relative_to(REPO)),
        "gates": gates,
        "claim_boundary": (
            "Comparison of one preregistered bounded source run against its "
            "declared gates, exact over the rationals where declared. A "
            "negative verdict is a valid preregistered negative result. No "
            "physical claim is promoted."
        ),
    }
    RECEIPT_OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["pass"] for k, v in gates.items()} | {"overall": overall}, indent=2))
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
