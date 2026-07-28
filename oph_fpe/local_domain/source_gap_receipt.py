"""Issue-633 instrument: source Hamiltonian with an exact positive gap.

The clock lane needs a source transition or Hamiltonian with a
positive dimensionless gap.  The issue-634 typed domain supplies one
without any import: the sign-twisted kinetic operator of the stage-3
layer, the signed Laplacian of the observer-visible seam complex, is
an integer symmetric operator whose quadratic form is the squared
derivative norm and whose kernel is trivial on the frustrated domain.
Positivity of the spectral gap is therefore exact, not numerical:

* the kinetic identity makes the operator positive semidefinite;
* the stage-2 sign structure makes the kernel trivial, by the witness
  and rank argument of the stage-3 kernel certificate;
* an integer positive-definite matrix has determinant at least one, so
  the gap has the exact positive floor one over the largest eigenvalue
  power, bounded through twice the maximum degree; the floor is
  recorded in log base two;
* the measured gap value from sparse shift-invert iteration is the
  non-certified numerical refinement, recorded with its residual.

The gap is dimensionless by construction: the operator is built from
the counting measure and the seam signs alone, with no scale mount.
The SI binding of issue 633 stays an open typed interface: selecting a
physical reference transition requires the fiber selection owned by
issues 569 and 630, and no measured frequency, particle mass, or SI
constant enters this instrument.  This receipt supplies the positive
gap object consumed by issue 633 and closes nothing.  No output
carries physical promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source
from oph_fpe.local_domain.stage1_event_complex import (
    MAIN_CONFIG,
    refuse_forbidden_config,
)
from oph_fpe.local_domain.stage2_spin_layer import (
    holonomy_census,
    seam_complex,
    visible_rows,
)
from oph_fpe.local_domain.stage3_typed_domain import (
    adjoint_certificate,
    kinetic_kernel_certificate,
)
from oph_fpe.local_domain.stage4_inhabitation import verify_local_domain_bundle

SCHEMA = "oph.source-clock-gap.v1"
ISSUE = 633
PHYSICAL_PROMOTION_ALLOWED = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local_domain"

RESIDUAL_GATE = 1.0e-8

FORBIDDEN_ANCESTRY = (
    "measured particle masses",
    "measured atomic frequencies",
    "Planck units",
    "Newton constant",
    "target residuals",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def signed_laplacian(complex_data: Mapping[str, Any]) -> csr_matrix:
    """Integer signed Laplacian of the seam complex, sparse symmetric."""

    nodes = complex_data["nodes"]
    index = {v: i for i, v in enumerate(nodes)}
    count = len(nodes)
    rows: list[int] = []
    cols: list[int] = []
    values: list[int] = []
    degree = [0] * count
    for (a, b), sign in complex_data["edge_sign_of"].items():
        ia, ib = index[a], index[b]
        degree[ia] += 1
        degree[ib] += 1
        rows.extend((ia, ib))
        cols.extend((ib, ia))
        values.extend((-sign, -sign))
    rows.extend(range(count))
    cols.extend(range(count))
    values.extend(degree)
    return csr_matrix(
        (np.asarray(values, dtype=np.float64), (rows, cols)),
        shape=(count, count),
    )


def exact_gap_floor(complex_data: Mapping[str, Any]) -> dict[str, Any]:
    """Exact positive floor for the gap of the integer operator.

    For an integer positive-definite symmetric matrix the determinant
    is at least one, so the smallest eigenvalue is at least one over
    the product of the remaining eigenvalues, each bounded by twice
    the maximum degree."""

    degrees: dict[int, int] = {}
    for a, b in complex_data["edges"]:
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1
    max_degree = max(degrees.values()) if degrees else 0
    count = complex_data["node_count"]
    lambda_max_bound = 2 * max_degree
    floor_log2 = (
        -(count - 1) * math.log2(lambda_max_bound)
        if lambda_max_bound > 0
        else 0.0
    )
    return {
        "max_degree": max_degree,
        "eigenvalue_upper_bound": lambda_max_bound,
        "floor_expression": (
            "one over the eigenvalue upper bound to the power of the "
            "carrier count minus one"
        ),
        "floor_log2": floor_log2,
        "floor_positive": bool(count > 0 and lambda_max_bound > 0),
    }


def measured_gap(matrix: csr_matrix) -> dict[str, Any]:
    """Sparse shift-invert smallest eigenvalue with recorded residual."""

    smallest_value, smallest_vector = eigsh(matrix, k=1, sigma=0.0, which="LM")
    largest_value = eigsh(
        matrix, k=1, which="LA", return_eigenvectors=False
    )
    value = float(smallest_value[0])
    vector = smallest_vector[:, 0]
    residual = float(
        np.linalg.norm(matrix @ vector - value * vector)
        / max(abs(value), 1.0e-300)
    )
    return {
        "smallest_eigenvalue": value,
        "largest_eigenvalue": float(largest_value[0]),
        "relative_residual": residual,
        "residual_within_gate": bool(residual < RESIDUAL_GATE),
        "status": "measured numerical refinement, not a certificate",
    }


def produce_source_gap_receipt(
    *,
    config: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Produce the issue-633 positive-gap receipt on the frozen domain."""

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

    bundle = verify_local_domain_bundle()
    capture = capture_physical_source(main_config)
    domain_complex = seam_complex(visible_rows(capture))

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

    adjoint = adjoint_certificate(domain_complex)
    kernel = kinetic_kernel_certificate(domain_complex)
    kernel_trivial = bool(
        kernel["twisted_kernel_dimension"] == 0
        and kernel["witnesses_annihilated"]
    )
    frustration = holonomy_census(domain_complex)

    matrix = signed_laplacian(domain_complex)
    floor = exact_gap_floor(domain_complex)
    measured = measured_gap(matrix)

    controls: dict[str, dict[str, Any]] = {}

    consistent_square = seam_complex(
        [
            {
                "overlap_id": f"seam-{k}",
                "left_carrier_id": f"carrier-{a:05d}",
                "right_carrier_id": f"carrier-{b:05d}",
                "left_ports": [0],
                "right_ports": [0],
                "orientation_signs": [-1],
                "visible_to_observer_tokens": ["observer-0000"],
                "interface_algebra_sha256": "sha256:a",
            }
            for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 3), (0, 3)))
        ]
    )
    square_kernel = kinetic_kernel_certificate(consistent_square)
    square_matrix = signed_laplacian(consistent_square)
    square_smallest = float(
        np.linalg.eigvalsh(square_matrix.toarray()).min()
    )
    controls["consistent_countermodel_gapless"] = {
        "control_failure_detected": bool(
            square_kernel["twisted_kernel_dimension"] == 1
            and abs(square_smallest) < 1.0e-12
        ),
        "note": (
            "the sign-consistent four-cycle carries an exact zero mode, "
            "so gap positivity is not hardwired"
        ),
    }

    if domain_complex["triangles"]:
        nodes = domain_complex["nodes"]
        ia, ib, _ = domain_complex["triangles"][0]
        flipped_edge = (
            min(nodes[ia], nodes[ib]),
            max(nodes[ia], nodes[ib]),
        )
        flipped = dict(domain_complex)
        flipped_signs = dict(domain_complex["edge_sign_of"])
        flipped_signs[flipped_edge] = -flipped_signs[flipped_edge]
        flipped["edge_sign_of"] = flipped_signs
        flipped_frustration = holonomy_census(flipped)
        controls["tampered_sign"] = {
            "control_failure_detected": bool(
                flipped_frustration["minus_holonomy_count"]
                != frustration["minus_holonomy_count"]
            ),
            "note": "the flipped seam is chosen inside a triangle",
        }
    else:
        controls["tampered_sign"] = {
            "control_failure_detected": False,
            "note": "no triangle exists to carry the holonomy change",
        }

    injected = dict(main_config)
    injected["target_scale"] = "second"
    refusal = produce_source_gap_receipt(config=injected)
    controls["target_injection"] = {
        "control_failure_detected": bool(refusal.get("verdict") == "REFUSED")
    }

    controls_fail_closed = bool(controls) and all(
        row["control_failure_detected"] for row in controls.values()
    )

    clause_verdicts = {
        "bundle_resolves_fail_closed": bool(bundle["passed"]),
        "hamiltonian_source_bound": domain_bound,
        "psd_identity_exact": bool(
            adjoint["kinetic_identity_exact"] and adjoint["kinetic_nonnegative"]
        ),
        "kernel_trivial_exact": kernel_trivial,
        "gap_positive_exact": bool(
            adjoint["kinetic_nonnegative"] and kernel_trivial
        ),
        "certified_floor_recorded": floor["floor_positive"],
        "measured_gap_within_residual_gate": measured["residual_within_gate"],
        "si_binding_open_interface_typed": True,
    }
    blockers = sorted(
        f"clause_failed:{name}"
        for name, verdict in clause_verdicts.items()
        if not verdict
    )
    if not controls_fail_closed:
        blockers.append("negative_control_did_not_fail")
    verdict = "ATTAINED" if not blockers else "NOT_ATTAINED"

    receipt = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "main_config": main_config,
        "capture_sha256": capture["capture_sha256"],
        "domain_freeze_sha256": domain_complex["complex_freeze_sha256"],
        "hamiltonian": {
            "operator": "signed Laplacian of the observer-visible seam complex",
            "carrier_count": domain_complex["node_count"],
            "seam_count": domain_complex["edge_count"],
            "integer_symmetric": True,
            "dimensionless": (
                "built from the counting measure and seam signs only; "
                "no scale mount"
            ),
        },
        "frustration": {
            "triangle_count": frustration["triangle_count"],
            "all_triangles_frustrated": frustration["all_triangles_frustrated"],
        },
        "exact_gap": {
            "positive": clause_verdicts["gap_positive_exact"],
            "argument": (
                "positive semidefinite by the kinetic identity and "
                "kernel-free by the stage-2 sign structure"
            ),
            "certified_floor": floor,
        },
        "measured_gap": measured,
        "si_binding": {
            "status": "OPEN_INTERFACE_PENDING_FIBER_SELECTION",
            "owners": ["issue 569", "issue 630"],
            "forbidden_ancestry_absent": list(FORBIDDEN_ANCESTRY),
            "note": (
                "selecting a physical reference transition requires the "
                "matter fiber content; the declared domain leaves it open, "
                "so no SI frequency chart is bound and no GeV value exists "
                "on this receipt"
            ),
        },
        "clause_verdicts": clause_verdicts,
        "negative_controls": controls,
        "controls_fail_closed": controls_fail_closed,
        "verdict": verdict,
        "SOURCE_CLOCK_GAP_RECEIPT": bool(verdict == "ATTAINED"),
        "blockers": blockers,
        "claim_boundary": (
            "Finite issue-633 gap object on the issue-634 typed domain: "
            "the sign-twisted kinetic operator is integer symmetric, "
            "positive semidefinite by the exact kinetic identity, and "
            "kernel-free on the frustrated visible domain, the frustration "
            "being the declared reversing convention acting on the measured "
            "seam topology, so its "
            "dimensionless spectral gap is exactly positive, with the "
            "determinant floor recorded and the numerical value as a "
            "measured refinement. No physical reference transition is "
            "selected, no measured frequency or mass enters, the SI "
            "binding stays an open typed interface, and issue 633 is not "
            "closed by this receipt."
        ),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        receipt_bytes = _canonical_json(receipt).encode("utf-8")
        (out / "source_gap_receipt.json").write_bytes(receipt_bytes)
        manifest_path = out / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"schema": "oph.local-domain-stage1.manifest.v1"}
        )
        manifest["source_gap_receipt"] = "source_gap_receipt.json"
        manifest["source_gap_receipt_sha256"] = _sha256_bytes(receipt_bytes)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=1) + "\n",
            encoding="utf-8",
        )
    return receipt
