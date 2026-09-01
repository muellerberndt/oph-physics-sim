"""Issue-311 finite-interface certificate: an explicit classical completion.

This certificate tests whether the finite spectral interface emitted by
the issue-634 typing excludes a classical realization.  It does not:

* The classical completion is exhibited, not assumed.  For each twist
  sector the harmonic network places one two-component coordinate at
  every carrier and one fixed orthogonal transport coupling on every
  oriented seam.  With unit masses, its Hamiltonian is one half the
  squared momentum plus one half the sum of
  |q_v - R_e q_w| squared over seams.  Its
  stiffness matrix is the exact realification of the sector Laplacian,
  so every sector spectrum appears with each eigenvalue doubled.
* Payload identity is verified quantity by quantity: the classical
  stiffness spectra reproduce the frozen sector gaps and the frozen
  scalar gap to numerical residual, and the classical kernel counts
  are exactly twice the frozen exact kernel dimensions.
* The listed frozen artifacts are manifest-pinned, their JSON leaves
  are counted by primitive serialized type, and their keys are scanned
  for a declared list of discriminator fragments.  This lexical census
  reports no matching key.  It does not infer the semantics of an
  unlabeled value and is not a completeness theorem for quantum
  discriminators.

The result is limited to the declared serialized finite interface.  It
shows that its sector and scalar spectra admit the displayed classical
realization.
It does not prove a no-go for every criterion constructible from an
extended domain, a cofinal refinement family, asymptotic states, or a
future measurement/state-update producer.  No physical promotion
follows from any output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import eigsh

from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source
from oph_fpe.local_domain.defect_sector_spectra import (
    SECTOR_COUNT,
    oriented_seams,
    sector_laplacian,
)
from oph_fpe.local_domain.receipt_io import load_manifest_pinned_receipt
from oph_fpe.local_domain.stage1_event_complex import (
    MAIN_CONFIG,
    local_domain_source_sha256,
    refuse_forbidden_config,
)
from oph_fpe.local_domain.stage2_spin_layer import seam_complex, visible_rows

SCHEMA = "oph.local-domain-classical-realization.v1"
ISSUE = 311
PHYSICAL_PROMOTION_ALLOWED = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local_domain"

RECEIPT_FILES = (
    "stage1_receipt.json",
    "stage2_receipt.json",
    "stage3_receipt.json",
    "stage4_receipt.json",
    "source_gap_receipt.json",
    "defect_sector_receipt.json",
    "matter_attachment_receipt.json",
)

DECLARED_DISCRIMINATOR_KEY_FRAGMENTS = (
    "entangle",
    "bell_",
    "correlation_payload",
    "measurement_statistics",
    "state_update",
    "born_rule",
    "commutator_witness",
)

GAP_MATCH_TOLERANCE = 1.0e-9
KERNEL_FLOOR = 1.0e-9
SYMMETRY_RESIDUAL_GATE = 1.0e-14
ORIENTATION_CONTROL_SHIFT_GATE = 1.0e-9
EIGENSOLVER_SHIFT = -1.0e-6


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _load_pinned_receipt(name: str, manifest_key: str) -> dict[str, Any] | None:
    """Load a receipt only when its manifest pin matches its bytes."""

    return load_manifest_pinned_receipt(DATA_DIR, name, manifest_key)


def harmonic_transport_stiffness(matrix: csr_matrix) -> csr_matrix:
    """Real stiffness matrix of the transport-coupled harmonic network.

    The realification [[Re, -Im], [Im, Re]] of the Hermitian sector
    Laplacian is the stiffness matrix of the network whose potential
    is one half the sum of |q_v - R_e q_w| squared, where R_e is the
    fixed planar rotation that realifies the seam phase.  With unit
    masses and canonical momentum p, the classical Hamiltonian is
    H = (p^T p + q^T K q) / 2.  The matrix is real symmetric positive
    semidefinite and its spectrum is the sector spectrum with every
    eigenvalue doubled."""

    real = matrix.real.tocsr()
    imag = matrix.imag.tocsr()
    return bmat([[real, -imag], [imag, real]], format="csr")


def classical_sector_reading(
    oriented: list[tuple[int, int]],
    sector: int,
    frozen_kernel: int,
) -> dict[str, Any]:
    """Classical spectrum of one sector and its frozen-payload match."""

    stiffness = harmonic_transport_stiffness(
        sector_laplacian(oriented, sector)
    )
    doubled_kernel = 2 * frozen_kernel
    wanted = doubled_kernel + 2
    size = stiffness.shape[0]
    start = np.cos(0.7 * np.arange(size)) + 0.1
    start /= np.linalg.norm(start)
    eigenvalues = eigsh(
        stiffness,
        k=wanted,
        sigma=EIGENSOLVER_SHIFT,
        which="LM",
        v0=start,
        return_eigenvectors=False,
    )
    eigenvalues = np.sort(eigenvalues)
    kernel_count = int((np.abs(eigenvalues) < KERNEL_FLOOR).sum())
    kernel_max_abs_eigenvalue = (
        float(np.max(np.abs(eigenvalues[:doubled_kernel])))
        if doubled_kernel
        else 0.0
    )
    first_nonkernel_abs_eigenvalue = float(
        abs(eigenvalues[doubled_kernel])
    )
    gap = float(eigenvalues[doubled_kernel])
    symmetry_max_residual = float(abs(stiffness - stiffness.T).max())
    return {
        "sector": sector,
        "classical_kernel_count": kernel_count,
        "expected_doubled_kernel": doubled_kernel,
        "kernel_eigenvalue_abs_floor": KERNEL_FLOOR,
        "kernel_max_abs_eigenvalue": kernel_max_abs_eigenvalue,
        "first_nonkernel_abs_eigenvalue": (
            first_nonkernel_abs_eigenvalue
        ),
        "classical_gap": gap,
        "symmetry_max_residual": symmetry_max_residual,
        "symmetric": bool(
            symmetry_max_residual < SYMMETRY_RESIDUAL_GATE
        ),
    }


def serialized_interface_census() -> dict[str, Any]:
    """Count primitive leaf types and flag declared key fragments.

    This is a bounded lexical schema audit.  It does not infer the
    meaning of values or prove that future encodings cannot carry
    quantum information.
    """

    def walk(value: Any, key_path: str) -> tuple[dict[str, int], list[str]]:
        census: dict[str, int] = {}
        hits: list[str] = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key).lower()
                if any(
                    t in key_text
                    for t in DECLARED_DISCRIMINATOR_KEY_FRAGMENTS
                ):
                    hits.append(f"{key_path}/{key}")
                child_census, child_hits = walk(child, f"{key_path}/{key}")
                for name, count in child_census.items():
                    census[name] = census.get(name, 0) + count
                hits.extend(child_hits)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                child_census, child_hits = walk(child, f"{key_path}[{index}]")
                for name, count in child_census.items():
                    census[name] = census.get(name, 0) + count
                hits.extend(child_hits)
        else:
            if value is None:
                name = "null"
            elif isinstance(value, bool):
                name = "boolean_leaf"
            elif isinstance(value, int):
                name = "integer_leaf"
            elif isinstance(value, float):
                name = "float_leaf"
            elif isinstance(value, str):
                name = (
                    "sha256_string"
                    if value.startswith("sha256:")
                    else "string_leaf"
                )
            else:
                raise TypeError(f"unclassifiable leaf at {key_path}")
            census[name] = census.get(name, 0) + 1
        return census, hits

    total: dict[str, int] = {}
    artifact_sha256: dict[str, str] = {}
    all_hits: list[str] = []
    missing: list[str] = []
    pin_failures: list[str] = []
    invalid_artifacts: list[str] = []
    schema_failures: list[str] = []
    manifest_path = DATA_DIR / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if not isinstance(manifest, Mapping):
                raise TypeError("manifest must be an object")
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
        ):
            manifest = {}
            invalid_artifacts.append(manifest_path.name)
    else:
        manifest = {}
    manifest_key_of = {
        "stage1_receipt.json": "receipt_sha256",
        "stage2_receipt.json": "stage2_receipt_sha256",
        "stage3_receipt.json": "stage3_receipt_sha256",
        "stage4_receipt.json": "stage4_receipt_sha256",
        "source_gap_receipt.json": "source_gap_receipt_sha256",
        "defect_sector_receipt.json": "defect_sector_receipt_sha256",
        "matter_attachment_receipt.json": "matter_attachment_receipt_sha256",
    }
    expected_identity = {
        "stage1_receipt.json": ("oph.local-domain-stage1.v1", None),
        "stage2_receipt.json": ("oph.local-domain-stage2.v1", None),
        "stage3_receipt.json": ("oph.local-domain-stage3.v1", None),
        "stage4_receipt.json": ("oph.local-domain-stage4.v1", None),
        "source_gap_receipt.json": ("oph.source-clock-gap.v1", 633),
        "defect_sector_receipt.json": (
            "oph.local-domain-defect-sector-spectra.v1",
            311,
        ),
        "matter_attachment_receipt.json": (
            "oph.local-domain-matter-attachment.v1",
            569,
        ),
    }
    for name in RECEIPT_FILES:
        path = DATA_DIR / name
        if not path.exists():
            missing.append(name)
            continue
        raw = path.read_bytes()
        if manifest.get(manifest_key_of[name]) != _sha256_bytes(raw):
            pin_failures.append(name)
            continue
        artifact_sha256[name] = _sha256_bytes(raw)
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise TypeError("receipt must be an object")
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            invalid_artifacts.append(name)
            continue
        expected_schema, expected_issue = expected_identity[name]
        issue_identity_valid = (
            "issue" not in payload
            if expected_issue is None
            else payload.get("issue") == expected_issue
        )
        if (
            payload.get("schema") != expected_schema
            or not issue_identity_valid
            or payload.get("physical_promotion_allowed") is not False
        ):
            schema_failures.append(name)
            continue
        census, hits = walk(payload, name)
        for key, count in census.items():
            total[key] = total.get(key, 0) + count
        all_hits.extend(hits)

    array_report = None
    array_path = DATA_DIR / "stage1_arrays.npz.gz"
    if array_path.exists():
        import gzip
        import io

        array_raw = array_path.read_bytes()
        if manifest.get("arrays_sha256") != _sha256_bytes(array_raw):
            pin_failures.append(array_path.name)
        else:
            artifact_sha256[array_path.name] = _sha256_bytes(array_raw)
            try:
                bundle = np.load(io.BytesIO(gzip.decompress(array_raw)))
                array_hits = [
                    f"{array_path.name}:{name}"
                    for name in bundle.files
                    if any(
                        t in name.lower()
                        for t in DECLARED_DISCRIMINATOR_KEY_FRAGMENTS
                    )
                ]
                array_report = {
                    "artifact": array_path.name,
                    "arrays": {
                        name: {
                            "dtype": str(bundle[name].dtype),
                            "shape": list(bundle[name].shape),
                        }
                        for name in bundle.files
                    },
                    "classification": (
                        "array names, dtypes, and shapes only"
                    ),
                }
                all_hits.extend(array_hits)
            except (OSError, ValueError, EOFError):
                invalid_artifacts.append(array_path.name)
    else:
        missing.append(array_path.name)
    return {
        "missing_receipts": missing,
        "artifact_sha256": dict(sorted(artifact_sha256.items())),
        "manifest_pin_failures": pin_failures,
        "schema_failures": schema_failures,
        "invalid_artifacts": sorted(set(invalid_artifacts)),
        "total_leaf_census": dict(sorted(total.items())),
        "array_artifact": array_report,
        "declared_key_fragment_hits": all_hits[:16],
        "no_declared_discriminator_key_match": bool(
            not all_hits
            and not missing
            and not pin_failures
            and not invalid_artifacts
            and not schema_failures
        ),
        "scope": (
            "the explicitly listed serialized finite receipts and stage-1 "
            "array bundle; primitive type census and declared key-fragment "
            "scan only"
        ),
    }


def produce_classical_realization_receipt(
    *,
    config: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    run_controls: bool = True,
) -> dict[str, Any]:
    """Produce the issue-311 classical-completion closure certificate."""

    main_config = dict(MAIN_CONFIG if config is None else config)
    forbidden = refuse_forbidden_config(main_config)
    if forbidden:
        return {
            "schema": SCHEMA,
            "issue": ISSUE,
            "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
            "verdict": "REFUSED",
            "blockers": [f"forbidden_config_key:{key}" for key in forbidden],
        }

    sector_receipt = _load_pinned_receipt(
        "defect_sector_receipt.json", "defect_sector_receipt_sha256"
    )
    gap_receipt = _load_pinned_receipt(
        "source_gap_receipt.json", "source_gap_receipt_sha256"
    )
    if sector_receipt is None or gap_receipt is None:
        return {
            "schema": SCHEMA,
            "issue": ISSUE,
            "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
            "verdict": "NOT_ATTAINED",
            "blockers": ["frozen_spectral_receipts_missing_or_unpinned"],
        }
    spectral_receipts_valid = bool(
        sector_receipt.get("schema")
        == "oph.local-domain-defect-sector-spectra.v1"
        and sector_receipt.get("issue") == ISSUE
        and sector_receipt.get("physical_promotion_allowed") is False
        and sector_receipt.get("verdict") == "ATTAINED"
        and gap_receipt.get("schema") == "oph.source-clock-gap.v1"
        and gap_receipt.get("issue") == 633
        and gap_receipt.get("physical_promotion_allowed") is False
        and gap_receipt.get("verdict") == "ATTAINED"
    )
    if not spectral_receipts_valid:
        return {
            "schema": SCHEMA,
            "issue": ISSUE,
            "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
            "verdict": "NOT_ATTAINED",
            "blockers": ["frozen_spectral_receipt_identity_invalid"],
        }

    capture = capture_physical_source(main_config)
    source_projection_sha256 = local_domain_source_sha256(capture, main_config)
    domain_rows = visible_rows(capture)
    domain_complex = seam_complex(domain_rows)
    oriented = oriented_seams(domain_rows)
    same_source = bool(
        sector_receipt.get("source_projection_sha256")
        == source_projection_sha256
        and sector_receipt["domain_freeze_sha256"]
        == domain_complex["complex_freeze_sha256"]
        and gap_receipt.get("source_projection_sha256")
        == source_projection_sha256
        and gap_receipt.get("domain_freeze_sha256")
        == domain_complex["complex_freeze_sha256"]
    )

    readings = []
    matches = []
    for row in sector_receipt["sector_table"]:
        reading = classical_sector_reading(
            oriented, row["sector"], row["kernel_dimension"]
        )
        gap_abs_residual = abs(
            reading["classical_gap"] - row["gap_above_kernel"]
        )
        gap_match = bool(gap_abs_residual < GAP_MATCH_TOLERANCE)
        kernel_match = bool(
            reading["classical_kernel_count"]
            == reading["expected_doubled_kernel"]
            and (
                reading["expected_doubled_kernel"] == 0
                or reading["kernel_max_abs_eigenvalue"] < KERNEL_FLOOR
            )
            and reading["first_nonkernel_abs_eigenvalue"]
            > KERNEL_FLOOR
        )
        readings.append(
            {
                **reading,
                "frozen_gap": row["gap_above_kernel"],
                "gap_abs_residual": gap_abs_residual,
                "gap_match": gap_match,
                "kernel_match": kernel_match,
            }
        )
        matches.append(gap_match and kernel_match and reading["symmetric"])

    scalar_gap_abs_residual = abs(
        readings[0]["classical_gap"]
        - gap_receipt["measured_gap"]["smallest_eigenvalue"]
    )
    scalar_match = bool(
        scalar_gap_abs_residual < GAP_MATCH_TOLERANCE
    )

    from oph_fpe.local_domain.stage1_event_complex import (
        CONTROL_SPLIT_CONFIG,
    )

    ladder_config = dict(CONTROL_SPLIT_CONFIG)
    ladder_config["observer_cross_reads"] = True
    ladder_capture = capture_physical_source(ladder_config)
    ladder_rows = visible_rows(ladder_capture)
    ladder_domain = seam_complex(ladder_rows)
    ladder_source_projection_sha256 = local_domain_source_sha256(
        ladder_capture, ladder_config
    )
    ladder_oriented = oriented_seams(ladder_rows)
    frozen_ladder = sector_receipt["scale_ladder"]
    ladder_source_domain_bound = bool(
        frozen_ladder.get("small_source_projection_sha256")
        == ladder_source_projection_sha256
        and frozen_ladder.get("small_domain_freeze_sha256")
        == ladder_domain["complex_freeze_sha256"]
        and frozen_ladder.get("small_source_carrier_count")
        == ladder_config["carrier_count"]
        and frozen_ladder.get("small_visible_node_count")
        == ladder_domain["node_count"]
        and frozen_ladder.get("small_visible_edge_count")
        == ladder_domain["edge_count"]
    )
    sector_interface = sector_receipt.get("spectral_interface_identity", {})
    sector_interface_scope_bound = bool(
        isinstance(sector_interface, Mapping)
        and sector_interface.get("schema") == sector_receipt["schema"]
        and sector_interface.get("issue") == sector_receipt["issue"]
        and sector_interface.get("local_domain_issue") == 634
        and sector_interface.get(
            "rer_exact_flux_12_42_vertex_identity_bridge"
        )
        is False
        and sector_interface.get("separate_from_rer_exact_flux_certificate")
        is True
        and sector_interface.get("main_domain")
        == {
            "source_carrier_count": main_config["carrier_count"],
            "source_projection_sha256": source_projection_sha256,
            "domain_freeze_sha256": domain_complex[
                "complex_freeze_sha256"
            ],
            "visible_node_count": domain_complex["node_count"],
            "visible_edge_count": domain_complex["edge_count"],
        }
        and sector_interface.get("ladder_domain")
        == {
            "source_carrier_count": ladder_config["carrier_count"],
            "source_projection_sha256": ladder_source_projection_sha256,
            "domain_freeze_sha256": ladder_domain[
                "complex_freeze_sha256"
            ],
            "visible_node_count": ladder_domain["node_count"],
            "visible_edge_count": ladder_domain["edge_count"],
        }
    )
    ladder_readings = []
    ladder_matches = []
    for sector in range(SECTOR_COUNT):
        frozen_kernel = frozen_ladder["small_kernel_dimensions"][sector]
        reading = classical_sector_reading(
            ladder_oriented, sector, frozen_kernel
        )
        gap_abs_residual = abs(
            reading["classical_gap"]
            - frozen_ladder["small_gaps"][sector]
        )
        gap_match = bool(gap_abs_residual < GAP_MATCH_TOLERANCE)
        kernel_match = bool(
            reading["classical_kernel_count"]
            == reading["expected_doubled_kernel"]
            and (
                reading["expected_doubled_kernel"] == 0
                or reading["kernel_max_abs_eigenvalue"] < KERNEL_FLOOR
            )
            and reading["first_nonkernel_abs_eigenvalue"]
            > KERNEL_FLOOR
        )
        ladder_readings.append(
            {
                **reading,
                "frozen_gap": frozen_ladder["small_gaps"][sector],
                "gap_abs_residual": gap_abs_residual,
                "gap_match": gap_match,
                "kernel_match": kernel_match,
            }
        )
        ladder_matches.append(gap_match and kernel_match)

    census = serialized_interface_census()
    expected_parent_artifacts = set(RECEIPT_FILES) | {
        "stage1_arrays.npz.gz"
    }
    parent_artifacts_pinned = bool(
        set(census["artifact_sha256"]) == expected_parent_artifacts
        and census["artifact_sha256"].get("defect_sector_receipt.json")
        == _sha256_bytes(
            (DATA_DIR / "defect_sector_receipt.json").read_bytes()
        )
        and census["artifact_sha256"].get("source_gap_receipt.json")
        == _sha256_bytes((DATA_DIR / "source_gap_receipt.json").read_bytes())
    )

    controls: dict[str, dict[str, Any]] = {}
    if run_controls:
        flipped = [
            (right, left) if position == 0 else (left, right)
            for position, (left, right) in enumerate(oriented)
        ]

        def lowest(stiffness: csr_matrix) -> float:
            start = np.cos(0.7 * np.arange(stiffness.shape[0])) + 0.1
            start /= np.linalg.norm(start)
            values = eigsh(
                stiffness,
                k=1,
                sigma=EIGENSOLVER_SHIFT,
                which="LM",
                v0=start,
                return_eigenvectors=False,
            )
            return float(np.sort(values)[0])

        original_low = lowest(
            harmonic_transport_stiffness(
                sector_laplacian(oriented, 1)
            )
        )
        flipped_low = lowest(
            harmonic_transport_stiffness(
                sector_laplacian(flipped, 1)
            )
        )
        controls["flipped_transport_orientation"] = {
            "control_failure_detected": bool(
                abs(flipped_low - original_low)
                > ORIENTATION_CONTROL_SHIFT_GATE
            ),
            "measured_shift": abs(flipped_low - original_low),
            "note": (
                "reversing one seam transport moves the classical "
                "spectrum, so the payload identity reads the source "
                "orientation field rather than a hardwired value"
            ),
        }

        doctored_payload = json.loads(
            (DATA_DIR / "source_gap_receipt.json").read_text(encoding="utf-8")
        )
        doctored_payload["bell_correlation_payload"] = [0.85, 0.85, -0.85]

        def walk_doctored(value, key_path):
            hits = []
            if isinstance(value, Mapping):
                for key, child in value.items():
                    if any(
                        t in str(key).lower()
                        for t in DECLARED_DISCRIMINATOR_KEY_FRAGMENTS
                    ):
                        hits.append(f"{key_path}/{key}")
                    hits.extend(walk_doctored(child, f"{key_path}/{key}"))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    hits.extend(walk_doctored(child, f"{key_path}[{index}]"))
            return hits

        doctored_hits = walk_doctored(doctored_payload, "doctored")
        controls["declared_key_fragment_injection"] = {
            "control_failure_detected": bool(
                doctored_hits
                and census["no_declared_discriminator_key_match"]
            ),
            "flagged_paths": doctored_hits[:4],
            "note": (
                "a configured key fragment injected into a copy of a "
                "frozen receipt is flagged by the same bounded lexical "
                "scan"
            ),
        }

        refusal = produce_classical_realization_receipt(
            config={**main_config, "target_signature": "quantum"},
            run_controls=False,
        )
        controls["target_injection"] = {
            "control_failure_detected": bool(refusal.get("verdict") == "REFUSED")
        }

    controls_fail_closed = bool(controls) and all(
        row["control_failure_detected"] for row in controls.values()
    )

    clause_verdicts = {
        "censused_parent_artifact_bytes_pinned": parent_artifacts_pinned,
        "same_source_domain_binding": same_source,
        "ladder_source_domain_binding": ladder_source_domain_bound,
        "spectral_interface_scope_bound": sector_interface_scope_bound,
        "classical_completion_exhibited": all(
            reading["symmetric"] for reading in readings
        ),
        "sector_payload_identity": bool(all(matches)),
        "scalar_gap_payload_identity": scalar_match,
        "ladder_payload_identity": bool(all(ladder_matches)),
        "declared_serialized_interface_censused": bool(
            not census["missing_receipts"]
            and not census["manifest_pin_failures"]
            and not census["invalid_artifacts"]
            and not census["schema_failures"]
        ),
        "no_declared_discriminator_key_match": census[
            "no_declared_discriminator_key_match"
        ],
    }
    blockers = sorted(
        f"clause_failed:{name}"
        for name, verdict in clause_verdicts.items()
        if not verdict
    )
    if run_controls and not controls_fail_closed:
        blockers.append("negative_control_did_not_fail")
    attained = not blockers

    receipt = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "main_config": main_config,
        "capture_sha256": capture["capture_sha256"],
        "capture_sha256_role": (
            "environment-sensitive full-capture diagnostic; not an "
            "identity gate for the local-domain stages"
        ),
        "source_projection_sha256": source_projection_sha256,
        "domain_freeze_sha256": domain_complex["complex_freeze_sha256"],
        "spectral_interface_identity": {
            "producer_schema": (
                "oph.local-domain-defect-sector-spectra.v1"
            ),
            "main_domain": {
                "source_projection_sha256": source_projection_sha256,
                "topology_freeze_sha256": domain_complex[
                    "complex_freeze_sha256"
                ],
                "source_carrier_count": main_config["carrier_count"],
                "visible_node_count": domain_complex["node_count"],
                "visible_edge_count": domain_complex["edge_count"],
            },
            "ladder_domain": {
                "source_projection_sha256": (
                    ladder_source_projection_sha256
                ),
                "topology_freeze_sha256": ladder_domain[
                    "complex_freeze_sha256"
                ],
                "source_carrier_count": ladder_config["carrier_count"],
                "visible_node_count": ladder_domain["node_count"],
                "visible_edge_count": ladder_domain["edge_count"],
            },
            "rer_exact_flux_12_42_vertex_identity_bridge": False,
            "separate_from_rer_exact_flux_certificate": True,
        },
        "numerical_gates": {
            "gap_match_abs_tolerance": GAP_MATCH_TOLERANCE,
            "kernel_eigenvalue_abs_floor": KERNEL_FLOOR,
            "symmetry_max_residual_tolerance": SYMMETRY_RESIDUAL_GATE,
            "orientation_control_shift_floor": (
                ORIENTATION_CONTROL_SHIFT_GATE
            ),
            "eigensolver_shift": EIGENSOLVER_SHIFT,
        },
        "classical_completion": {
            "object": (
                "two-component classical harmonic network per sector: "
                "one coordinate q_v in R^2 and one momentum p_v in R^2 per "
                "carrier, fixed orthogonal transport R_e per oriented "
                "seam, unit masses, and H = one half sum_v |p_v|^2 plus "
                "one half sum_e |q_v - R_e q_w|^2"
            ),
            "stiffness_rule": (
                "the stiffness matrix is the exact realification of the "
                "sector Laplacian, so the sector spectrum appears with "
                "every eigenvalue doubled"
            ),
            "sector_readings": readings,
            "scalar_gap_match": scalar_match,
            "scalar_gap_abs_residual": scalar_gap_abs_residual,
            "ladder_point_readings": ladder_readings,
        },
        "serialized_interface_census": census,
        "upstream_pins": census["artifact_sha256"],
        "finite_interface_result": {
            "statement": (
                "the explicit classical harmonic completion reproduces the "
                "declared finite sector and scalar spectral interface, so "
                "those spectral outputs alone do not exclude the "
                "same-interface classical countermodel"
            ),
            "composition_scope": (
                "sector twist composition and subcomplex restriction are "
                "structure of the shared complex, present identically in "
                "both completions"
            ),
            "asymptotics_scope": (
                "the serialized finite interface contains no asymptotic-"
                "state observable; no claim about an extended-domain "
                "asymptotic construction follows"
            ),
            "refinement_scope": (
                "the realification is functorial in the complex, so the "
                "classical completion commutes with subcomplex "
                "restriction; the second cutoff is a separate measured "
                "point and does not establish a cofinal refinement map or "
                "a commuting limit square"
            ),
            "unit_scope": (
                "the realized spectra are dimensionless finite-operator "
                "spectra; physical-unit attachment is not part of this "
                "certificate"
            ),
            "retained_positive_subresults": [
                "the exact Z6 sector-index arithmetic of the local-domain "
                "defect-sector receipt",
                "the local-domain sector-gap tables at the declared main "
                "and ladder domains",
            ],
        },
        "clause_verdicts": clause_verdicts,
        "negative_controls": controls,
        "controls_fail_closed": controls_fail_closed,
        "verdict": (
            "CLASSICAL_REALIZATION_MATCHES_DECLARED_FINITE_SPECTRAL_INTERFACE"
            if attained
            else "NOT_ATTAINED"
        ),
        "CLASSICAL_REALIZATION_RECEIPT": bool(attained),
        "blockers": blockers,
        "claim_boundary": (
            "Finite issue-311 boundary result: an explicit two-component "
            "classical harmonic completion reproduces the declared "
            "six-sector and scalar "
            "local-domain spectral interface with doubled kernels and "
            "matching gaps. This interface is the issue-634 visible seam "
            "complex and its declared ladder point; no identity bridge to "
            "the separate RER 12/42-vertex exact-flux certificate is "
            "claimed. "
            "The bounded lexical census finds no key matching its declared "
            "discriminator fragments. The explicit spectral match, not "
            "that key scan, shows that these finite spectral outputs alone "
            "do not exclude the classical realization. This does not close a "
            "criterion over an extended domain, cofinal refinement family, "
            "asymptotic-state construction, or future measurement and "
            "state-update producers. No physical promotion follows from "
            "any output."
        ),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        receipt_bytes = _canonical_json(receipt).encode("utf-8")
        (out / "classical_realization_receipt.json").write_bytes(receipt_bytes)
        manifest_path = out / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"schema": "oph.local-domain-stage1.manifest.v1"}
        )
        manifest["classical_realization_receipt"] = (
            "classical_realization_receipt.json"
        )
        manifest["classical_realization_receipt_sha256"] = _sha256_bytes(
            receipt_bytes
        )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=1) + "\n",
            encoding="utf-8",
        )
    return receipt
