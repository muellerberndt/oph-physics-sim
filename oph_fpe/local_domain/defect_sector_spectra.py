"""Issue-311 instrument: sector-resolved spectra on the local domain.

The particle-criterion lane owns a refinement-stable defect sector with
a pole or equivalent physical spectral criterion on the source-derived
local domain.  This instrument supplies the first sector-resolved
spectral object on that domain: six flat twist sectors extending the
declared reversing seam convention, each carrying an exact kernel
verdict and a measured dimensionless gap.

Sectors.  The declared convention transports every seam by minus one.
Sector k, for k from zero through five, multiplies that transport by
the k-th sixth root of unity applied along the source-declared seam
orientation, left carrier to right, with the conjugate transport in
the reverse direction.  Every sector transport is a twelfth root of
unity, the orientation field is capture data rather than index
bookkeeping, so the classes are invariant under carrier relabelling,
and the whole family is the Z6 character orbit of the declared
convention.  Sector indices add under composition of twists, the
finite echo of additive fusion in the exact flux packet.

Exact layer.  Each sector Laplacian is the Hermitian twisted operator
on the observer-visible seam complex.  Its kernel is the space of flat
sections, and flatness is decided exactly: transport phases live in
the integers modulo twelve, a spanning-tree propagation assigns each
carrier a phase exponent, and every non-tree seam either confirms or
refutes the assignment by integer arithmetic.  A consistent sector has
kernel dimension one per component; an inconsistent sector has kernel
dimension zero, by the same witness-and-rank argument as the stage-3
kernel certificate, so gap positivity above the kernel is exact.

Measured layer.  The smallest nonzero eigenvalue of each sector
Laplacian is computed by sparse iteration and recorded with its
residual as the sector gap, a dimensionless target-independent
invariant of the source.  Sectors with different gaps are spectrally
distinguished; the six-gap table is the emitted criterion candidate.

Claim boundary.  This instrument does not close issue 311: the sector
gaps are spectral invariants of the finite twisted kinetic operators,
not particle masses; no asymptotic states, cofinal composition limit,
or pole structure is constructed; and the same-interface classical
countermodel is not excluded, since a classical field on the same
complex carries the same spectra.  No Yukawa, mass, or pole datum
enters, and no physical promotion follows from any output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source
from oph_fpe.local_domain.stage1_event_complex import (
    CONTROL_SPLIT_CONFIG,
    MAIN_CONFIG,
    refuse_forbidden_config,
)
from oph_fpe.local_domain.stage2_spin_layer import seam_complex, visible_rows

SCHEMA = "oph.local-domain-defect-sector-spectra.v1"
ISSUE = 311
PHYSICAL_PROMOTION_ALLOWED = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local_domain"

SECTOR_COUNT = 6
PHASE_MODULUS = 12
RESIDUAL_GATE = 1.0e-8
KERNEL_EIGENVALUE_FLOOR = 1.0e-9

FORBIDDEN_INPUT_KEY_FRAGMENTS = ("yukawa", "pole_mass", "mass_gev", "mev")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sector_phase_exponent(sector: int) -> int:
    """Transport phase exponent of the sector, in twelfths of a turn.

    The declared convention is minus one, exponent six; sector k
    multiplies by the k-th sixth root, exponent two k, so the sector
    transport exponent is six plus two k modulo twelve."""

    return (6 + 2 * sector) % PHASE_MODULUS


def oriented_seams(rows: list[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """Source-declared seam orientation field, left carrier to right.

    Each capture row declares its left and right carrier, so the
    orientation is source data rather than index bookkeeping and the
    sector classes are invariant under carrier relabelling.  Parallel
    rows on one carrier pair keep the first row's orientation, matching
    the reduced-sign convention of the seam complex."""

    seen: set[tuple[int, int]] = set()
    oriented: list[tuple[int, int]] = []
    for row in rows:
        left = int(row["left_carrier_id"].rsplit("-", 1)[1])
        right = int(row["right_carrier_id"].rsplit("-", 1)[1])
        key = (min(left, right), max(left, right))
        if key in seen:
            continue
        seen.add(key)
        oriented.append((left, right))
    return oriented


def flat_section_verdict(
    oriented: list[tuple[int, int]], sector: int
) -> dict[str, Any]:
    """Exact kernel verdict for the sector by integer phase propagation.

    A flat section assigns each carrier a phase exponent so that every
    oriented seam advances the phase by the sector exponent from left
    to right, hence retards it from right to left; a spanning-tree
    propagation fixes the assignment per component and every non-tree
    seam either confirms it or refutes it by integer arithmetic."""

    exponent = sector_phase_exponent(sector)
    nodes = sorted({v for pair in oriented for v in pair})
    index = {v: i for i, v in enumerate(nodes)}
    neighbor: dict[int, list[tuple[int, int]]] = {
        i: [] for i in range(len(nodes))
    }
    for left, right in oriented:
        il, ir = index[left], index[right]
        neighbor[il].append((ir, exponent))
        neighbor[ir].append((il, (-exponent) % PHASE_MODULUS))
    phase = [None] * len(nodes)
    components = 0
    flat_components = 0
    for start in range(len(nodes)):
        if phase[start] is not None:
            continue
        components += 1
        consistent = True
        phase[start] = 0
        stack = [start]
        while stack:
            u = stack.pop()
            for v, step in neighbor[u]:
                expected = (phase[u] + step) % PHASE_MODULUS
                if phase[v] is None:
                    phase[v] = expected
                    stack.append(v)
                elif phase[v] != expected:
                    consistent = False
        if consistent:
            flat_components += 1
    return {
        "sector": sector,
        "transport_exponent_twelfths": exponent,
        "component_count": components,
        "flat_component_count": flat_components,
        "kernel_dimension": flat_components,
        "kernel_trivial": bool(flat_components == 0),
    }


def sector_laplacian(
    oriented: list[tuple[int, int]], sector: int
) -> csr_matrix:
    """Hermitian twisted Laplacian on the oriented seams, sparse complex.

    The left-to-right entry carries the sector transport and the
    right-to-left entry carries its conjugate, so the operator is the
    Hermitian form of the same directional transport the flat-section
    verdict propagates."""

    exponent = sector_phase_exponent(sector)
    transport = np.exp(2j * np.pi * exponent / PHASE_MODULUS)
    nodes = sorted({v for pair in oriented for v in pair})
    index = {v: i for i, v in enumerate(nodes)}
    count = len(nodes)
    rows: list[int] = []
    cols: list[int] = []
    values: list[complex] = []
    degree = [0] * count
    for left, right in oriented:
        il, ir = index[left], index[right]
        degree[il] += 1
        degree[ir] += 1
        rows.extend((il, ir))
        cols.extend((ir, il))
        values.extend((-transport, -np.conj(transport)))
    rows.extend(range(count))
    cols.extend(range(count))
    values.extend(float(d) for d in degree)
    return csr_matrix(
        (np.asarray(values, dtype=np.complex128), (rows, cols)),
        shape=(count, count),
    )


def measured_sector_gap(
    matrix: csr_matrix, kernel_dimension: int
) -> dict[str, Any]:
    """Smallest eigenvalue above the exact kernel, with residual."""

    wanted = kernel_dimension + 1
    start = np.ones(matrix.shape[0], dtype=np.complex128)
    start /= np.linalg.norm(start)
    eigenvalues, vectors = eigsh(
        matrix, k=wanted, sigma=-1.0e-6, which="LM", v0=start
    )
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    vectors = vectors[:, order]
    kernel_leak = float(
        max(abs(eigenvalues[i]) for i in range(kernel_dimension))
        if kernel_dimension
        else 0.0
    )
    gap_value = float(eigenvalues[kernel_dimension])
    gap_vector = vectors[:, kernel_dimension]
    residual = float(
        np.linalg.norm(matrix @ gap_vector - gap_value * gap_vector)
        / max(abs(gap_value), 1.0e-300)
    )
    return {
        "gap_above_kernel": gap_value,
        "kernel_eigenvalue_leak": kernel_leak,
        "relative_residual": residual,
        "residual_within_gate": bool(
            residual < RESIDUAL_GATE
            and kernel_leak < KERNEL_EIGENVALUE_FLOOR
        ),
        "status": "measured numerical refinement, not a certificate",
    }


def sector_table(oriented: list[tuple[int, int]]) -> list[dict[str, Any]]:
    """Exact kernel verdicts and measured gaps for all six sectors."""

    table = []
    for sector in range(SECTOR_COUNT):
        verdict = flat_section_verdict(oriented, sector)
        matrix = sector_laplacian(oriented, sector)
        measured = measured_sector_gap(matrix, verdict["kernel_dimension"])
        table.append(
            {
                **verdict,
                "measured": measured,
            }
        )
    return table


def fusion_additivity() -> dict[str, Any]:
    """Exact additivity of sector exponents under twist composition."""

    holds = all(
        (
            sector_phase_exponent(a)
            + sector_phase_exponent(b)
            - sector_phase_exponent(0)
        )
        % PHASE_MODULUS
        == sector_phase_exponent((a + b) % SECTOR_COUNT)
        for a in range(SECTOR_COUNT)
        for b in range(SECTOR_COUNT)
    )
    return {
        "rule": (
            "composing the twists of sectors a and b relative to the "
            "declared convention lands in sector a plus b modulo six"
        ),
        "holds": bool(holds),
    }


def produce_defect_sector_receipt(
    *,
    config: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    run_controls: bool = True,
) -> dict[str, Any]:
    """Produce the issue-311 sector-spectra receipt on the frozen domain."""

    main_config = dict(MAIN_CONFIG if config is None else config)
    forbidden = refuse_forbidden_config(main_config)
    injected = [
        key
        for key in main_config
        if any(f in str(key).lower() for f in FORBIDDEN_INPUT_KEY_FRAGMENTS)
    ]
    if forbidden or injected:
        return {
            "schema": SCHEMA,
            "issue": ISSUE,
            "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
            "verdict": "REFUSED",
            "blockers": sorted(
                [f"forbidden_config_key:{key}" for key in forbidden]
                + [f"forbidden_input_fragment:{key}" for key in injected]
            ),
        }

    capture = capture_physical_source(main_config)
    domain_rows = visible_rows(capture)
    domain_complex = seam_complex(domain_rows)
    oriented = oriented_seams(domain_rows)

    stage2_path = DATA_DIR / "stage2_receipt.json"
    stage2 = (
        json.loads(stage2_path.read_text(encoding="utf-8"))
        if stage2_path.exists()
        else None
    )
    domain_bound = bool(
        stage2 is not None
        and stage2["seam_layer"]["domain_complex"]["complex_freeze_sha256"]
        == domain_complex["complex_freeze_sha256"]
        and stage2["capture_sha256"] == capture["capture_sha256"]
    )

    table = sector_table(oriented)
    fusion = fusion_additivity()

    gaps = [row["measured"]["gap_above_kernel"] for row in table]
    pairwise = {
        f"{i}-{j}": abs(gaps[i] - gaps[j])
        for i in range(SECTOR_COUNT)
        for j in range(i + 1, SECTOR_COUNT)
    }
    conjugation_pairs_degenerate = all(
        abs(gaps[k] - gaps[(SECTOR_COUNT - k) % SECTOR_COUNT]) < 1.0e-9
        for k in range(SECTOR_COUNT)
    )

    ladder_config = dict(CONTROL_SPLIT_CONFIG)
    ladder_config["observer_cross_reads"] = True
    ladder_capture = capture_physical_source(ladder_config)
    ladder_table = sector_table(oriented_seams(visible_rows(ladder_capture)))
    ladder_kernels = [row["kernel_dimension"] for row in ladder_table]
    main_kernels = [row["kernel_dimension"] for row in table]

    controls: dict[str, dict[str, Any]] = {}
    if run_controls:
        wrapped = flat_section_verdict(oriented, 7)
        base = flat_section_verdict(oriented, 1)
        controls["sector_index_wraparound"] = {
            "control_failure_detected": bool(
                wrapped["transport_exponent_twelfths"]
                == base["transport_exponent_twelfths"]
                and wrapped["kernel_dimension"] == base["kernel_dimension"]
            ),
            "note": (
                "sector seven reduces to sector one exactly, so the sector "
                "group is genuinely Z6 and not an unbounded index"
            ),
        }

        matrix0 = sector_laplacian(oriented, 0)
        matrix3 = sector_laplacian(oriented, 3)
        controls["sector_spectra_not_hardwired"] = {
            "control_failure_detected": bool(
                table[0]["kernel_dimension"] == 0
                and table[3]["kernel_dimension"] == 1
                and abs(
                    table[0]["measured"]["gap_above_kernel"]
                    - table[3]["measured"]["gap_above_kernel"]
                )
                > 1.0e-6
                and matrix0.nnz == matrix3.nnz
            ),
            "note": (
                "sector zero is kernel-free and sector three carries the "
                "flat mode, with different measured gaps on operators of "
                "identical sparsity, so the sector label reaches the "
                "spectrum"
            ),
        }

        refusal = produce_defect_sector_receipt(
            config={**main_config, "pole_mass_target": "input"},
            run_controls=False,
        )
        controls["target_injection"] = {
            "control_failure_detected": bool(refusal.get("verdict") == "REFUSED")
        }

    controls_fail_closed = bool(controls) and all(
        row["control_failure_detected"] for row in controls.values()
    )

    clause_verdicts = {
        "same_source_domain_binding": domain_bound,
        "kernel_exact_numeric_agreement": all(
            row["kernel_dimension"] in (0, 1)
            and row["measured"]["gap_above_kernel"] > 1.0e-6
            and (
                row["kernel_dimension"] == 0
                or row["measured"]["kernel_eigenvalue_leak"]
                < KERNEL_EIGENVALUE_FLOOR
            )
            for row in table
        ),
        "gaps_measured_within_gate": all(
            row["measured"]["residual_within_gate"] for row in table
        ),
        "sector_spectra_distinguish": bool(
            len({round(g, 9) for g in gaps}) > 1
        ),
        "scale_ladder_kernel_stable": bool(ladder_kernels == main_kernels),
    }
    blockers = sorted(
        f"clause_failed:{name}"
        for name, verdict in clause_verdicts.items()
        if not verdict
    )
    if run_controls and not controls_fail_closed:
        blockers.append("negative_control_did_not_fail")
    verdict = "ATTAINED" if not blockers else "NOT_ATTAINED"

    receipt = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "main_config": main_config,
        "capture_sha256": capture["capture_sha256"],
        "domain_freeze_sha256": domain_complex["complex_freeze_sha256"],
        "sector_family": {
            "group": "Z6 character orbit of the declared reversing convention",
            "transport_exponents_twelfths": [
                sector_phase_exponent(k) for k in range(SECTOR_COUNT)
            ],
            "fusion_additivity": fusion,
        },
        "sector_table": [
            {
                "sector": row["sector"],
                "transport_exponent_twelfths": row[
                    "transport_exponent_twelfths"
                ],
                "kernel_dimension": row["kernel_dimension"],
                "gap_above_kernel": row["measured"]["gap_above_kernel"],
                "relative_residual": row["measured"]["relative_residual"],
            }
            for row in table
        ],
        "gap_separations": {
            "pairwise_absolute": {
                key: round(value, 12) for key, value in pairwise.items()
            },
            "distinct_gap_count": len({round(g, 9) for g in gaps}),
            "conjugate_sector_pairs_degenerate": bool(
                conjugation_pairs_degenerate
            ),
        },
        "scale_ladder": {
            "small_config_carriers": ladder_config["carrier_count"],
            "small_kernel_dimensions": ladder_kernels,
            "main_kernel_dimensions": main_kernels,
            "small_gaps": [
                row["measured"]["gap_above_kernel"] for row in ladder_table
            ],
        },
        "clause_verdicts": clause_verdicts,
        "negative_controls": controls,
        "controls_fail_closed": controls_fail_closed,
        "verdict": verdict,
        "DEFECT_SECTOR_SPECTRA_RECEIPT": bool(verdict == "ATTAINED"),
        "blockers": blockers,
        "claim_boundary": (
            "Finite issue-311 instrument on the issue-634 local domain: six "
            "Z6 twist sectors extending the declared reversing convention, "
            "each with an exact integer-arithmetic kernel verdict and a "
            "measured dimensionless gap, with exact sector-index fusion "
            "additivity and a recorded scale-ladder point. The gaps are "
            "spectral invariants of finite twisted kinetic operators, not "
            "particle masses; no asymptotic states, cofinal composition "
            "limit, or pole structure is constructed; the same-interface "
            "classical countermodel is not excluded; and issue 311 is not "
            "closed by this receipt. No Yukawa, mass, or pole datum enters, "
            "and no physical promotion follows from any output."
        ),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        receipt_bytes = _canonical_json(receipt).encode("utf-8")
        (out / "defect_sector_receipt.json").write_bytes(receipt_bytes)
        manifest_path = out / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"schema": "oph.local-domain-stage1.manifest.v1"}
        )
        manifest["defect_sector_receipt"] = "defect_sector_receipt.json"
        manifest["defect_sector_receipt_sha256"] = _sha256_bytes(receipt_bytes)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=1) + "\n",
            encoding="utf-8",
        )
    return receipt
