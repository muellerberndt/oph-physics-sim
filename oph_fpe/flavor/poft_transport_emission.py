"""Fail-closed assay for POFT T0/T1 emission by saved S3 carrier states.

The natural observable of the current carrier is fixed before comparison: the
edge-average of the three-label permutation representation.  This is a
necessary direct-transport witness, not an arbitrary fitted map from simulator
features to the POFT matrices.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from oph_fpe.defects.array_s3_holonomy import S3_CLASS, S3_ELEMENTS


POFT_SINGULAR_RATIO_TOLERANCE = 0.05
HAAR_SINGULAR_RATIO_TOLERANCE = 0.02


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _permutation_representation() -> np.ndarray:
    matrices = np.zeros((len(S3_ELEMENTS), 3, 3), dtype=float)
    for group_index, element in enumerate(S3_ELEMENTS):
        for source_label, target_label in enumerate(element):
            matrices[group_index, target_label, source_label] = 1.0
    return matrices


def _poft_targets() -> dict[str, np.ndarray]:
    return {
        "T0": np.array(
            [
                [1, 1 / 10 + 1j / 5, 1j / 10],
                [1 / 10 - 1j / 5, 7 / 10, 1 / 5],
                [-1j / 10, 1 / 5, 2 / 5],
            ],
            dtype=complex,
        ),
        "T1": np.array(
            [
                [51 / 50, 11 / 100 + 19j / 100, 1 / 100 + 9j / 100],
                [11 / 100 - 19j / 100, 69 / 100, 9 / 50 + 1j / 100],
                [1 / 100 - 9j / 100, 9 / 50 - 1j / 100, 41 / 100],
            ],
            dtype=complex,
        ),
    }


def _matrix_payload(matrix: np.ndarray) -> list[list[dict[str, float]]]:
    values = np.asarray(matrix, dtype=complex)
    return [
        [{"real": float(value.real), "imag": float(value.imag)} for value in row]
        for row in values
    ]


def _singular_ratios(matrix: np.ndarray) -> np.ndarray:
    values = np.linalg.svd(matrix, compute_uv=False)
    return values / values[0]


def _block_bootstrap_ratios(
    left: np.ndarray,
    gauge: np.ndarray,
    representation: np.ndarray,
    *,
    seed: int,
    replicates: int = 128,
) -> dict[str, list[float]]:
    """Bootstrap source-node blocks to retain local edge correlation."""

    unique, inverse = np.unique(left, return_inverse=True)
    block_sums = np.zeros((unique.size, 3, 3), dtype=float)
    block_counts = np.zeros(unique.size, dtype=np.int64)
    np.add.at(block_sums, inverse, representation[gauge])
    np.add.at(block_counts, inverse, 1)
    rng = np.random.default_rng(seed)
    samples = np.zeros((replicates, 3), dtype=float)
    for index in range(replicates):
        chosen = rng.integers(0, unique.size, size=unique.size)
        matrix = block_sums[chosen].sum(axis=0) / block_counts[chosen].sum()
        samples[index] = _singular_ratios(matrix)
    return {
        "p05": np.quantile(samples, 0.05, axis=0).tolist(),
        "p50": np.quantile(samples, 0.50, axis=0).tolist(),
        "p95": np.quantile(samples, 0.95, axis=0).tolist(),
    }


def _state_report(path: Path, *, label: str, seed: int) -> dict[str, Any]:
    with np.load(path) as source:
        required = {"left", "right", "gauge"}
        missing = sorted(required - set(source.files))
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        left = np.asarray(source["left"], dtype=np.int64)
        right = np.asarray(source["right"], dtype=np.int64)
        gauge = np.asarray(source["gauge"], dtype=np.int64)
        exported_fields = sorted(source.files)
    if left.shape != right.shape or left.shape != gauge.shape or gauge.ndim != 1:
        raise ValueError(f"incompatible edge arrays in {path}")
    if gauge.size == 0 or np.any((gauge < 0) | (gauge >= len(S3_ELEMENTS))):
        raise ValueError(f"invalid S3 labels in {path}")

    representation = _permutation_representation()
    operator = representation[gauge].mean(axis=0)
    ratios = _singular_ratios(operator)
    targets = _poft_targets()
    target_ratios = {name: _singular_ratios(matrix) for name, matrix in targets.items()}
    distances = {
        name: float(np.max(np.abs(ratios - target_ratio)))
        for name, target_ratio in target_ratios.items()
    }
    class_counts = np.bincount(S3_CLASS[gauge], minlength=3)
    element_counts = np.bincount(gauge, minlength=len(S3_ELEMENTS))
    complex_fields = [
        name
        for name in exported_fields
        if any(token in name.lower() for token in ("phase", "complex", "amplitude", "transport_matrix"))
    ]
    return {
        "label": label,
        "source_path": str(path),
        "source_sha256": _sha256(path),
        "edge_count": int(gauge.size),
        "node_count_seen": int(np.unique(np.concatenate((left, right))).size),
        "exported_fields": exported_fields,
        "complex_transport_fields": complex_fields,
        "complex_oriented_amplitude_exported": bool(complex_fields),
        "s3_element_counts": element_counts.tolist(),
        "s3_class_counts": {
            "identity": int(class_counts[0]),
            "transposition": int(class_counts[1]),
            "threecycle": int(class_counts[2]),
        },
        "natural_direct_transport_operator": _matrix_payload(operator),
        "singular_value_ratios": ratios.tolist(),
        "source_node_block_bootstrap": _block_bootstrap_ratios(
            left, gauge, representation, seed=seed
        ),
        "max_abs_singular_ratio_distance": distances,
        "poft_T0_necessary_spectral_match": distances["T0"] <= POFT_SINGULAR_RATIO_TOLERANCE,
        "poft_T1_necessary_spectral_match": distances["T1"] <= POFT_SINGULAR_RATIO_TOLERANCE,
        "haar_rank_one_distance": float(max(abs(ratios[1]), abs(ratios[2]))),
        "haar_rank_one_compatible": bool(
            max(abs(ratios[1]), abs(ratios[2])) <= HAAR_SINGULAR_RATIO_TOLERANCE
        ),
    }


def build_poft_transport_emission_report(
    states: Iterable[tuple[str, Path]],
    *,
    refinement_map: Path | None = None,
    bootstrap_seed: int = 20260712,
) -> dict[str, Any]:
    state_rows = [
        _state_report(Path(path), label=label, seed=bootstrap_seed + index)
        for index, (label, path) in enumerate(states)
    ]
    if not state_rows:
        raise ValueError("at least one S3 state is required")

    targets = _poft_targets()
    target_rows = {
        name: {
            "matrix": _matrix_payload(matrix),
            "singular_value_ratios": _singular_ratios(matrix).tolist(),
        }
        for name, matrix in targets.items()
    }
    refinement_receipt = False
    refinement_details: dict[str, Any] = {"path": None, "sha256": None}
    if refinement_map is not None:
        refinement_map = Path(refinement_map)
        payload = json.loads(refinement_map.read_text(encoding="utf-8"))
        refinement_receipt = bool(payload.get("coarse_to_fine_edge_intertwiner_receipt"))
        refinement_details = {
            "path": str(refinement_map),
            "sha256": _sha256(refinement_map),
            "coarse_to_fine_edge_intertwiner_receipt": refinement_receipt,
        }

    any_t0_shape = any(row["poft_T0_necessary_spectral_match"] for row in state_rows)
    any_t1_shape = any(row["poft_T1_necessary_spectral_match"] for row in state_rows)
    complex_carrier = all(row["complex_oriented_amplitude_exported"] for row in state_rows)
    t0_receipt = bool(any_t0_shape and complex_carrier)
    t1_receipt = bool(any_t1_shape and complex_carrier and refinement_receipt)
    emission_receipt = bool(t0_receipt and t1_receipt)
    all_haar = all(row["haar_rank_one_compatible"] for row in state_rows)

    return {
        "artifact": "oph_poft_transport_emission_targeted_assay_v1",
        "claim_class": "direct_s3_edge_transport_falsification_assay",
        "observable_frozen_before_state_comparison": (
            "mean of the three-label permutation matrices carried by saved oriented S3 edges"
        ),
        "comparison_uses_quark_masses": False,
        "thresholds": {
            "poft_singular_ratio_max_abs": POFT_SINGULAR_RATIO_TOLERANCE,
            "haar_rank_one_max_nontrivial_ratio": HAAR_SINGULAR_RATIO_TOLERANCE,
        },
        "poft_targets": target_rows,
        "states": state_rows,
        "refinement_map": refinement_details,
        "receipts": {
            "direct_T0_necessary_spectral_shape_seen": any_t0_shape,
            "direct_T1_necessary_spectral_shape_seen": any_t1_shape,
            "complex_oriented_family_amplitude_exported": complex_carrier,
            "coarse_to_fine_edge_intertwiner_exported": refinement_receipt,
            "poft_T0_direct_emission_receipt": t0_receipt,
            "poft_T1_refined_emission_receipt": t1_receipt,
            "poft_T0_T1_physical_emission_receipt": emission_receipt,
        },
        "verdict": (
            "CURRENT_S3_EDGE_CARRIER_HAAR_EQUILIBRATED_NOT_POFT"
            if all_haar and not emission_receipt
            else "POFT_NOT_EMITTED_BY_TESTED_DIRECT_TRANSPORT"
            if not emission_receipt
            else "POFT_DIRECT_EMISSION_CANDIDATE_REQUIRES_INDEPENDENT_REPLICATION"
        ),
        "claim_boundary": (
            "The assay tests the current carrier's natural direct S3 permutation transport. "
            "It cannot exclude a future source-derived complex lift, but prescribing the POFT "
            "phase/path map merely to recover T0/T1 would be circular."
        ),
    }
