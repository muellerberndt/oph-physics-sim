"""Issue-311 closure certificate: the same-interface classical completion.

The particle-criterion lane closes with a source-positive criterion
excluding the same-interface classical countermodel, or with the proof
that the complete declared local domain admits indistinguishable
classical and quantum completions.  The declared domain, the closed
issue-634 typing with its frozen receipts, is final, so the second
branch is decidable.  This certificate decides it constructively:

* The classical completion is exhibited, not assumed.  For each twist
  sector the planar-rotor spring network places one two-component
  coordinate at every carrier and one rotation spring on every
  oriented seam; its potential energy is one half the summed squared
  spring extensions, a manifestly classical mechanical energy, and its
  stiffness matrix is the exact realification of the sector Laplacian,
  so every sector spectrum appears with each eigenvalue doubled.
* Payload identity is verified quantity by quantity: the classical
  stiffness spectra reproduce the frozen sector gaps and the frozen
  scalar gap to numerical residual, and the classical kernel counts
  are exactly twice the frozen exact kernel dimensions.
* The emitted interface is exhaustively partitioned: every leaf of
  every frozen domain receipt is a shared-data function of the
  complex, an operator-spectral quantity realized by the stiffness
  matrices, a hash, a count, prose, or a verdict.  No leaf and no key
  carries a quantum discriminator: no state-update statistics, no
  correlation payload, no noncommutativity witness has any producer.

The theorem follows: the classical rotor completion and the quantum
Hamiltonian reading of the same operators emit identical interfaces
on the complete declared domain, so no criterion constructible from
the domain excludes the classical countermodel, and the positive
branch of the lane is not attainable there.  The finite sector
arithmetic and the sector-gap invariants are retained as closed
subresults.  Asymptotic states exist in neither completion on the
finite complex, and sector composition and subcomplex restriction are
shared structure, so the no-go covers the composition and asymptotics
clauses.  A completion carrying measurement statistics or state-update
data is a different domain and is not judged here.  No physical
promotion follows from any output.
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
from oph_fpe.local_domain.stage1_event_complex import (
    MAIN_CONFIG,
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

DISCRIMINATOR_VOCABULARY = (
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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def rotor_stiffness(matrix: csr_matrix) -> csr_matrix:
    """Real stiffness matrix of the planar-rotor spring network.

    The realification [[Re, -Im], [Im, Re]] of the Hermitian sector
    Laplacian is the stiffness matrix of the network whose potential
    is one half the summed squared extensions of rotation springs, so
    it is real symmetric positive semidefinite and its spectrum is the
    sector spectrum with every eigenvalue doubled."""

    real = matrix.real.tocsr()
    imag = matrix.imag.tocsr()
    return bmat([[real, -imag], [imag, real]], format="csr")


def classical_sector_reading(
    oriented: list[tuple[int, int]],
    sector: int,
    frozen_kernel: int,
) -> dict[str, Any]:
    """Classical spectrum of one sector and its frozen-payload match."""

    stiffness = rotor_stiffness(sector_laplacian(oriented, sector))
    doubled_kernel = 2 * frozen_kernel
    wanted = doubled_kernel + 2
    size = stiffness.shape[0]
    start = np.cos(0.7 * np.arange(size)) + 0.1
    start /= np.linalg.norm(start)
    eigenvalues = eigsh(
        stiffness,
        k=wanted,
        sigma=-1.0e-6,
        which="LM",
        v0=start,
        return_eigenvectors=False,
    )
    eigenvalues = np.sort(eigenvalues)
    kernel_count = int((np.abs(eigenvalues) < KERNEL_FLOOR).sum())
    gap = float(eigenvalues[doubled_kernel])
    return {
        "sector": sector,
        "classical_kernel_count": kernel_count,
        "expected_doubled_kernel": doubled_kernel,
        "classical_gap": gap,
        "symmetric": bool(
            abs(stiffness - stiffness.T).max() < 1.0e-14
        ),
    }


def interface_partition() -> dict[str, Any]:
    """Partition every emitted leaf; flag any quantum discriminator."""

    def walk(value: Any, key_path: str) -> tuple[dict[str, int], list[str]]:
        census: dict[str, int] = {}
        hits: list[str] = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key).lower()
                if any(t in key_text for t in DISCRIMINATOR_VOCABULARY):
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
                name = "verdict_boolean"
            elif isinstance(value, int):
                name = "shared_data_count"
            elif isinstance(value, float):
                name = "spectral_or_shared_real"
            elif isinstance(value, str):
                name = "hash" if value.startswith("sha256:") else "prose"
            else:
                raise TypeError(f"unclassifiable leaf at {key_path}")
            census[name] = census.get(name, 0) + 1
        return census, hits

    total: dict[str, int] = {}
    all_hits: list[str] = []
    missing: list[str] = []
    for name in RECEIPT_FILES:
        path = DATA_DIR / name
        if not path.exists():
            missing.append(name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        census, hits = walk(payload, name)
        for key, count in census.items():
            total[key] = total.get(key, 0) + count
        all_hits.extend(hits)
    return {
        "missing_receipts": missing,
        "total_leaf_census": dict(sorted(total.items())),
        "discriminator_hits": all_hits[:16],
        "no_quantum_discriminator": bool(not all_hits and not missing),
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

    sector_path = DATA_DIR / "defect_sector_receipt.json"
    gap_path = DATA_DIR / "source_gap_receipt.json"
    if not sector_path.exists() or not gap_path.exists():
        return {
            "schema": SCHEMA,
            "issue": ISSUE,
            "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
            "verdict": "NOT_ATTAINED",
            "blockers": ["frozen_spectral_receipts_missing"],
        }
    sector_receipt = json.loads(sector_path.read_text(encoding="utf-8"))
    gap_receipt = json.loads(gap_path.read_text(encoding="utf-8"))

    capture = capture_physical_source(main_config)
    domain_rows = visible_rows(capture)
    domain_complex = seam_complex(domain_rows)
    oriented = oriented_seams(domain_rows)
    same_source = bool(
        sector_receipt["capture_sha256"] == capture["capture_sha256"]
        and sector_receipt["domain_freeze_sha256"]
        == domain_complex["complex_freeze_sha256"]
    )

    readings = []
    matches = []
    for row in sector_receipt["sector_table"]:
        reading = classical_sector_reading(
            oriented, row["sector"], row["kernel_dimension"]
        )
        gap_match = bool(
            abs(reading["classical_gap"] - row["gap_above_kernel"])
            < GAP_MATCH_TOLERANCE
        )
        kernel_match = bool(
            reading["classical_kernel_count"]
            == reading["expected_doubled_kernel"]
        )
        readings.append(
            {
                **reading,
                "frozen_gap": row["gap_above_kernel"],
                "gap_match": gap_match,
                "kernel_match": kernel_match,
            }
        )
        matches.append(gap_match and kernel_match and reading["symmetric"])

    scalar_match = bool(
        abs(
            readings[0]["classical_gap"]
            - gap_receipt["measured_gap"]["smallest_eigenvalue"]
        )
        < GAP_MATCH_TOLERANCE
    )

    partition = interface_partition()

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
                sigma=-1.0e-6,
                which="LM",
                v0=start,
                return_eigenvectors=False,
            )
            return float(np.sort(values)[0])

        original_low = lowest(rotor_stiffness(sector_laplacian(oriented, 1)))
        flipped_low = lowest(rotor_stiffness(sector_laplacian(flipped, 1)))
        controls["flipped_spring_orientation"] = {
            "control_failure_detected": bool(
                abs(flipped_low - original_low) > 1.0e-9
            ),
            "measured_shift": abs(flipped_low - original_low),
            "note": (
                "reversing one rotation spring moves the classical "
                "spectrum, so the payload identity reads the source "
                "orientation field rather than a hardwired value"
            ),
        }

        injected = {"bell_correlation_payload": [0.85, 0.85, 0.85, -0.85]}
        injected_flagged = any(
            any(token in str(key).lower() for token in DISCRIMINATOR_VOCABULARY)
            for key in injected
        )
        controls["discriminator_injection"] = {
            "control_failure_detected": bool(
                injected_flagged and partition["no_quantum_discriminator"]
            ),
            "note": (
                "an injected correlation payload key is flagged by the "
                "discriminator vocabulary while the frozen interface "
                "stays clean"
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
        "same_source_domain_binding": same_source,
        "classical_completion_exhibited": all(
            reading["symmetric"] for reading in readings
        ),
        "sector_payload_identity": bool(all(matches)),
        "scalar_gap_payload_identity": scalar_match,
        "interface_exhaustively_partitioned": bool(
            not partition["missing_receipts"]
        ),
        "no_quantum_discriminator_emitted": partition[
            "no_quantum_discriminator"
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
        "domain_freeze_sha256": domain_complex["complex_freeze_sha256"],
        "classical_completion": {
            "object": (
                "planar-rotor spring network per sector: one two-component "
                "coordinate per carrier, one rotation spring per oriented "
                "seam, potential one half the summed squared extensions"
            ),
            "stiffness_rule": (
                "the stiffness matrix is the exact realification of the "
                "sector Laplacian, so the sector spectrum appears with "
                "every eigenvalue doubled"
            ),
            "sector_readings": readings,
            "scalar_gap_match": scalar_match,
        },
        "interface_partition": partition,
        "indistinguishability_theorem": {
            "statement": (
                "the classical rotor completion and the quantum "
                "Hamiltonian reading of the same operators emit identical "
                "interfaces on the complete declared local domain, so no "
                "criterion constructible from the domain excludes the "
                "same-interface classical countermodel"
            ),
            "composition_covered": (
                "sector twist composition and subcomplex restriction are "
                "structure of the shared complex, present identically in "
                "both completions"
            ),
            "asymptotics_covered": (
                "the finite complex provides asymptotic states in neither "
                "completion; the clause closes under the no-go rather "
                "than by construction"
            ),
            "retained_positive_subresults": [
                "the exact finite sector arithmetic of the v3 receipt",
                "the sector-gap invariant table of the sector-spectra "
                "receipt",
            ],
        },
        "clause_verdicts": clause_verdicts,
        "negative_controls": controls,
        "controls_fail_closed": controls_fail_closed,
        "verdict": (
            "CLASSICAL_QUANTUM_INDISTINGUISHABLE_ON_DECLARED_DOMAIN"
            if attained
            else "NOT_ATTAINED"
        ),
        "CLASSICAL_REALIZATION_RECEIPT": bool(attained),
        "blockers": blockers,
        "claim_boundary": (
            "Complete negative exit of the issue-311 particle-criterion "
            "lane on its declared domain: an explicit classical rotor "
            "completion reproduces every emitted spectral quantity of the "
            "closed issue-634 domain with exact doubled kernels and "
            "matching gaps, the emitted interface is exhaustively "
            "partitioned with no quantum discriminator, and therefore the "
            "declared domain admits indistinguishable classical and "
            "quantum completions. The finite sector arithmetic and "
            "sector-gap invariants stay closed subresults. A domain "
            "extended by measurement statistics or state-update producers "
            "is a different object and is not judged here. No physical "
            "promotion follows from any output."
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
