"""Issue-569 instrument: finite assembled matter boundary packet.

The family lane holds a simulator-realized rank-three multiplicity
object and an imported fifteen-state generation packet.  This
instrument assembles a finite boundary packet from that material, the
issue-634 local operator domain, and a separate issue-314 spin packet.
No transport or identity bridge between the twelve-vertex spin support
and the issue-634 visible seam domain is certified.  Charge, tensor-rank,
and operator arithmetic are exact; the rank-three band remains the
declared measured input:

* The declared imported generation table is recomputed with
  exact fractions: five field rows, fifteen Weyl states, all four
  anomaly forms zero, four weak doublets per family.
* The canonical diagonal Z6 kernel fixes every matter state: for each
  row the phase triality over three plus weak parity over two plus
  hypercharge is an exact integer.
* Chirality is nondegenerate by exact multiset arithmetic: the table
  shares no state with its conjugate, so the Z2 grading of the typed
  chiral species has no vectorlike collapse.
* The declared attachment fiber is the measured rank-three band of the
  pole-residue artifact tensored with the recomputed generation, of
  complex rank exactly forty-five, and the sign-twisted seam operators
  of the stage-3 layer act on it with support radius one, exact
  integer adjoint and kinetic identities, and subcomplex naturality on
  one deterministic rank-forty-five probe.
* Under the declared tensor extension, the matter kinetic operator is
  the scalar operator tensored with the identity.  Its spectrum is the
  scalar spectrum with multiplicity forty-five and the exactly positive
  source-gap is inherited conditionally.  The source does not select
  this matter action.
* The spin data is retained as a separate pinned issue-314 artifact:
  one spin structure on its twelve-vertex oriented support, the
  non-split Klein-four section obstruction, and the unique nontrivial
  central involution.  Stage-2 sign transport belongs to the distinct
  local-domain context.  Relating the two is an open interface.

The declared input keys are scanned for a bounded list of Yukawa and
laboratory mass fragments, with no match.  This lexical check does not
type unlabeled values or prove semantic input closure.  The
finite generator-frequency residue selecting the band is an explicit
input and carries no laboratory scale.  The receipt supplies conditional
finite arithmetic on the assembled packet; it does not select a matter
action or supply a Spin/locality bridge.  The continuum Spin/locality gate, the
matter-pole identification, the physical seam-action selection, and
the laboratory current identification of issue 569 stay open.  No
output of this module carries physical promotion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source
from oph_fpe.core.pole_residue_readback import produce_pole_residue_artifact
from oph_fpe.core.spin_statistics_response import produce_spin_statistics_artifact
from oph_fpe.local_domain.receipt_io import (
    load_manifest_pinned_receipt,
    manifest_pinned_artifact_sha256,
    stage2_matches_source_domain,
)
from oph_fpe.local_domain.stage1_event_complex import (
    MAIN_CONFIG,
    local_domain_source_sha256,
    refuse_forbidden_config,
)
from oph_fpe.local_domain.stage2_spin_layer import seam_complex, visible_rows
from oph_fpe.local_domain.stage3_typed_domain import (
    _deterministic_int,
    probe_section,
    seam_adjoint,
    seam_derivative,
)
from oph_fpe.local_domain.stage4_inhabitation import verify_local_domain_bundle

SCHEMA = "oph.local-domain-matter-attachment.v1"
ISSUE = 569
PHYSICAL_PROMOTION_ALLOWED = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local_domain"
RESEARCH_MANIFESTS = (
    Path(__file__).resolve().parents[3]
    / "reverse-engineering-reality"
    / "code"
    / "a5_closure"
    / "manifests"
)
CARRIER_MANIFEST_NAME = "echosahedral_federation_reference.json"
RESPONSE_ARTIFACT_NAME = "charged_response_semantic_artifact.json"

MATTER_FIBER_RANK = 45
GENERATION_STATES = 15
BAND_RANK = 3

FORBIDDEN_INPUT_KEY_FRAGMENTS = ("yukawa", "pole_mass", "mass_gev", "mev")

GENERATION_TABLE = (
    {"label": "Q", "color": 3, "weak": 2, "hypercharge": Fraction(1, 6)},
    {"label": "u_c", "color": -3, "weak": 1, "hypercharge": Fraction(-2, 3)},
    {"label": "d_c", "color": -3, "weak": 1, "hypercharge": Fraction(1, 3)},
    {"label": "L", "color": 1, "weak": 2, "hypercharge": Fraction(-1, 2)},
    {"label": "e_c", "color": 1, "weak": 1, "hypercharge": Fraction(1)},
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def generation_certificate(
    table: tuple[Mapping[str, Any], ...] = GENERATION_TABLE,
) -> dict[str, Any]:
    """Exact fifteen-state generation arithmetic from the charge table."""

    rows = []
    weyl_total = 0
    doublets = 0
    anomaly_u1_cubed = Fraction(0)
    anomaly_su3_u1 = Fraction(0)
    anomaly_su2_u1 = Fraction(0)
    anomaly_grav_u1 = Fraction(0)
    for row in table:
        color_dim = abs(int(row["color"]))
        weak_dim = int(row["weak"])
        hypercharge = Fraction(row["hypercharge"])
        weyl = color_dim * weak_dim
        weyl_total += weyl
        if weak_dim == 2:
            doublets += color_dim
        anomaly_u1_cubed += weyl * hypercharge**3
        anomaly_grav_u1 += weyl * hypercharge
        if color_dim == 3:
            anomaly_su3_u1 += weak_dim * hypercharge
        if weak_dim == 2:
            anomaly_su2_u1 += color_dim * hypercharge
        rows.append(
            {
                "label": row["label"],
                "color_dimension": color_dim,
                "color_conjugate": bool(int(row["color"]) < 0),
                "weak_dimension": weak_dim,
                "hypercharge": str(hypercharge),
                "weyl_states": weyl,
            }
        )
    return {
        "rows": rows,
        "weyl_state_count": weyl_total,
        "weak_doublets_per_family": doublets,
        "witten_parity_even": bool(doublets % 2 == 0),
        "anomaly_forms": {
            "u1_cubed": str(anomaly_u1_cubed),
            "su3_sq_u1": str(anomaly_su3_u1),
            "su2_sq_u1": str(anomaly_su2_u1),
            "grav_u1": str(anomaly_grav_u1),
        },
        "anomalies_vanish": bool(
            anomaly_u1_cubed == 0
            and anomaly_su3_u1 == 0
            and anomaly_su2_u1 == 0
            and anomaly_grav_u1 == 0
        ),
        "state_count_exact": bool(weyl_total == GENERATION_STATES),
    }


def z6_kernel_certificate(
    table: tuple[Mapping[str, Any], ...] = GENERATION_TABLE,
) -> dict[str, Any]:
    """Exact fixing of every state by the diagonal Z6 generator.

    The generator acts on a row with triality t, weak parity w, and
    hypercharge Y by the phase t over three plus w over two plus Y;
    the row is fixed exactly when that phase is an integer."""

    phases = {}
    fixed = True
    for row in table:
        color = int(row["color"])
        triality = Fraction(1, 3) if color == 3 else (
            Fraction(2, 3) if color == -3 else Fraction(0)
        )
        weak_parity = Fraction(1, 2) if int(row["weak"]) == 2 else Fraction(0)
        phase = triality + weak_parity + Fraction(row["hypercharge"])
        phases[row["label"]] = str(phase)
        if phase.denominator != 1:
            fixed = False
    return {
        "generator": (
            "diagonal color-weak-hypercharge Z6: triality over three plus "
            "weak parity over two plus hypercharge"
        ),
        "row_phases": phases,
        "all_states_fixed": bool(fixed),
    }


def chirality_certificate(
    table: tuple[Mapping[str, Any], ...] = GENERATION_TABLE,
) -> dict[str, Any]:
    """Exact nondegeneracy of the chirality grading.

    The conjugate of a row negates the hypercharge and conjugates the
    color; the grading is nondegenerate when the table shares no
    state with its conjugate, so no vectorlike pair collapses the Z2
    grading of the typed chiral species."""

    states = {
        (int(row["color"]), int(row["weak"]), Fraction(row["hypercharge"]))
        for row in table
    }
    conjugates = {
        (-color if abs(color) == 3 else color, weak, -hypercharge)
        for color, weak, hypercharge in states
    }
    overlap = sorted(
        str(state) for state in states.intersection(conjugates)
    )
    return {
        "state_count": len(states),
        "conjugate_overlap": overlap,
        "chirality_nondegenerate": bool(not overlap),
    }


def _load_research_inputs() -> tuple[dict | None, dict | None, list[str]]:
    blockers: list[str] = []
    carrier = None
    response = None
    carrier_path = RESEARCH_MANIFESTS / CARRIER_MANIFEST_NAME
    response_path = RESEARCH_MANIFESTS / RESPONSE_ARTIFACT_NAME
    if not carrier_path.is_file():
        blockers.append("carrier_manifest_not_checked_out")
    else:
        carrier = json.loads(carrier_path.read_text(encoding="utf-8"))
    if not response_path.is_file():
        blockers.append("response_artifact_not_checked_out")
    else:
        response = json.loads(response_path.read_text(encoding="utf-8"))
    return carrier, response, blockers


def _scan_forbidden_keys(value: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if any(f in key_text for f in FORBIDDEN_INPUT_KEY_FRAGMENTS):
                hits.append(f"{path}/{key}")
            hits.extend(_scan_forbidden_keys(child, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(_scan_forbidden_keys(child, f"{path}[{index}]"))
    return hits


def matter_operator_certificate(
    domain_complex: Mapping[str, Any],
) -> dict[str, Any]:
    """Exact identities on one deterministic rank-45 section probe."""

    nodes = domain_complex["nodes"]
    section = probe_section(nodes, MATTER_FIBER_RANK, salt=569)
    edge_probe = {
        edge: tuple(
            _deterministic_int(569, *edge, component)
            for component in range(MATTER_FIBER_RANK)
        )
        for edge in domain_complex["edges"]
    }
    derivative = seam_derivative(domain_complex, section)
    support_radius_one = set(derivative) == set(domain_complex["edges"])
    adjoint = seam_adjoint(domain_complex, edge_probe)

    pairing_left = 0
    for edge, values in derivative.items():
        probe = edge_probe[edge]
        for component, value in enumerate(values):
            pairing_left += value * probe[component]
    pairing_right = 0
    for node, values in section.items():
        adj = adjoint[node]
        for component, value in enumerate(values):
            pairing_right += value * adj[component]

    kinetic = 0
    for values in derivative.values():
        for value in values:
            kinetic += value * value
    laplacian = seam_adjoint(domain_complex, derivative)
    quadratic = 0
    for node, values in section.items():
        lap = laplacian[node]
        for component, value in enumerate(values):
            quadratic += value * lap[component]

    edges = domain_complex["edges"]
    half = max(1, len(edges) // 2)
    sub_edges = edges[:half]
    sub_nodes = sorted({v for edge in sub_edges for v in edge})
    sub_complex = {
        "nodes": sub_nodes,
        "node_count": len(sub_nodes),
        "edges": sub_edges,
        "edge_count": len(sub_edges),
        "edge_sign_of": {
            edge: domain_complex["edge_sign_of"][edge] for edge in sub_edges
        },
    }
    sub_section = {node: section[node] for node in sub_nodes}
    sub_derivative = seam_derivative(sub_complex, sub_section)
    natural = all(
        sub_derivative[edge] == derivative[edge] for edge in sub_edges
    )
    return {
        "status": "declared_tensor_extension",
        "source_selected": False,
        "fiber_rank": MATTER_FIBER_RANK,
        "probe_count": 1,
        "probe_salt": 569,
        "identity_scope": (
            "one deterministic rank-45 section and edge probe"
        ),
        "support_radius_one": bool(support_radius_one),
        "adjoint_identity_exact": bool(pairing_left == pairing_right),
        "kinetic_identity_exact": bool(kinetic == quadratic),
        "kinetic_nonnegative": bool(kinetic >= 0),
        "subcomplex_naturality_exact": bool(natural),
    }


def gap_inheritance_certificate(
    gap_receipt: Mapping[str, Any] | None,
    *,
    source_projection_sha256: str | None = None,
    domain_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    """Conditional tensor-block inheritance of the scalar gap at fiber 45.

    Under the declared tensor extension, the matter kinetic operator is
    the scalar signed Laplacian tensored with the identity on the fiber.
    The source does not select that matter action."""

    if gap_receipt is None:
        return {
            "status": (
                "conditional_algebraic_inheritance_under_declared_"
                "tensor_extension"
            ),
            "matter_action_source_selected": False,
            "gap_receipt_present": False,
            "inherited": False,
        }
    source_bound = bool(
        source_projection_sha256 is not None
        and gap_receipt.get("source_projection_sha256")
        == source_projection_sha256
    )
    domain_bound = bool(
        domain_freeze_sha256 is not None
        and gap_receipt.get("domain_freeze_sha256")
        == domain_freeze_sha256
    )
    schema_valid = bool(
        gap_receipt.get("schema") == "oph.source-clock-gap.v1"
        and gap_receipt.get("issue") == 633
        and gap_receipt.get("physical_promotion_allowed") is False
    )
    receipt_attained = bool(
        schema_valid and gap_receipt.get("verdict") == "ATTAINED"
    )
    exact_gap = gap_receipt.get("exact_gap", {})
    measured_gap = gap_receipt.get("measured_gap", {})
    exact_positive = bool(
        isinstance(exact_gap, Mapping) and exact_gap.get("positive")
    )
    measured_value = (
        measured_gap.get("smallest_eigenvalue")
        if isinstance(measured_gap, Mapping)
        else None
    )
    return {
        "status": (
            "conditional_algebraic_inheritance_under_declared_tensor_extension"
        ),
        "matter_action_source_selected": False,
        "gap_receipt_present": True,
        "same_source_projection": source_bound,
        "same_domain_freeze": domain_bound,
        "gap_receipt_schema_valid": schema_valid,
        "gap_receipt_attained": receipt_attained,
        "structure": (
            "matter operator equals scalar operator tensor identity on "
            "the rank-45 fiber; spectrum equals the scalar spectrum with "
            "multiplicity forty-five"
        ),
        "scalar_gap_exactly_positive": exact_positive,
        "scalar_measured_gap": measured_value,
        "inherited": bool(
            exact_positive and source_bound and domain_bound and receipt_attained
        ),
    }


def _load_frozen(name: str, manifest_key: str) -> dict | None:
    return load_manifest_pinned_receipt(DATA_DIR, name, manifest_key)


def produce_matter_attachment_receipt(
    *,
    config: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    run_controls: bool = True,
) -> dict[str, Any]:
    """Produce the issue-569 finite boundary-packet receipt."""

    main_config = dict(MAIN_CONFIG if config is None else config)
    forbidden = refuse_forbidden_config(main_config)
    injected_fragments = [
        key
        for key in main_config
        if any(f in str(key).lower() for f in FORBIDDEN_INPUT_KEY_FRAGMENTS)
    ]
    if forbidden or injected_fragments:
        return {
            "schema": SCHEMA,
            "issue": ISSUE,
            "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
            "verdict": "REFUSED",
            "blockers": sorted(
                [f"forbidden_config_key:{key}" for key in forbidden]
                + [f"forbidden_input_fragment:{key}" for key in injected_fragments]
            ),
        }

    carrier, response, input_blockers = _load_research_inputs()
    if input_blockers:
        return {
            "schema": SCHEMA,
            "issue": ISSUE,
            "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
            "verdict": "NOT_ATTAINED",
            "blockers": sorted(input_blockers),
        }

    pole_residue = produce_pole_residue_artifact(carrier, response)
    spin_artifact = produce_spin_statistics_artifact(carrier)

    bundle = verify_local_domain_bundle()
    capture = capture_physical_source(main_config)
    source_projection_sha256 = local_domain_source_sha256(capture, main_config)
    domain_complex = seam_complex(visible_rows(capture))
    stage2 = _load_frozen("stage2_receipt.json", "stage2_receipt_sha256")
    gap_receipt = _load_frozen(
        "source_gap_receipt.json", "source_gap_receipt_sha256"
    )
    stage2_receipt_sha256 = manifest_pinned_artifact_sha256(
        DATA_DIR,
        "stage2_receipt.json",
        "stage2_receipt_sha256",
    )
    source_gap_receipt_sha256 = manifest_pinned_artifact_sha256(
        DATA_DIR,
        "source_gap_receipt.json",
        "source_gap_receipt_sha256",
    )
    domain_bound = stage2_matches_source_domain(
        stage2,
        source_projection_sha256,
        domain_complex["complex_freeze_sha256"],
    )

    generation = generation_certificate()
    z6 = z6_kernel_certificate()
    chirality = chirality_certificate()
    operators = matter_operator_certificate(domain_complex)
    gap = gap_inheritance_certificate(
        gap_receipt,
        source_projection_sha256=source_projection_sha256,
        domain_freeze_sha256=domain_complex["complex_freeze_sha256"],
    )

    band = pole_residue["pole_residue_readback"]["family_band_residue"]
    band_rank = int(band["measured_rank"])
    attachment_rank = band_rank * generation["weyl_state_count"]

    spin_gate = spin_artifact["physical_source_gate"]
    spin_rows = {
        key: bool(value)
        for key, value in spin_gate.items()
        if isinstance(value, bool)
    }
    spin_consumed = {
        "artifact_sha256": spin_artifact.get("artifact_sha256"),
        "spin_structure_count": spin_artifact["support_homology"][
            "spin_structure_count"
        ],
        "klein_four_obstruction": spin_artifact["section_obstruction"][
            "no_section_over_any_klein_four_subgroup"
        ],
        "gate_rows": spin_rows,
    }
    try:
        lift_ambiguity_rank = (
            stage2["seam_layer"]["lift_ambiguity"][
                "lift_ambiguity_rank"
            ]
            if domain_bound
            else None
        )
    except (KeyError, TypeError):
        lift_ambiguity_rank = None

    scan_targets = {
        "carrier_manifest": carrier,
        "response_artifact": response,
        "pole_residue_artifact": pole_residue,
        "spin_artifact": spin_artifact,
    }
    forbidden_hits: list[str] = []
    for name, payload in scan_targets.items():
        forbidden_hits.extend(_scan_forbidden_keys(payload, name))

    controls: dict[str, dict[str, Any]] = {}
    if run_controls:
        vectorlike = GENERATION_TABLE + tuple(
            {
                "label": row["label"] + "_bar",
                "color": -int(row["color"]) if abs(int(row["color"])) == 3
                else int(row["color"]),
                "weak": row["weak"],
                "hypercharge": -Fraction(row["hypercharge"]),
            }
            for row in GENERATION_TABLE
        )
        vector_chirality = chirality_certificate(vectorlike)
        controls["vectorlike_table"] = {
            "control_failure_detected": bool(
                not vector_chirality["chirality_nondegenerate"]
            )
        }

        mutated = tuple(
            dict(row, hypercharge=Fraction(row["hypercharge"]) + Fraction(1, 6))
            if row["label"] == "e_c"
            else row
            for row in GENERATION_TABLE
        )
        mutated_generation = generation_certificate(mutated)
        mutated_z6 = z6_kernel_certificate(mutated)
        controls["hypercharge_mutation"] = {
            "control_failure_detected": bool(
                not mutated_generation["anomalies_vanish"]
                and not mutated_z6["all_states_fixed"]
            )
        }

        doctored_band = dict(band)
        doctored_band["measured_rank"] = 4
        controls["band_rank_mutation"] = {
            "control_failure_detected": bool(
                int(doctored_band["measured_rank"])
                * generation["weyl_state_count"]
                != MATTER_FIBER_RANK
            )
        }

        doctored_bound = False
        if domain_bound:
            doctored_stage2 = json.loads(json.dumps(stage2))
            doctored_stage2["seam_layer"]["domain_complex"][
                "complex_freeze_sha256"
            ] = "sha256:" + "0" * 64
            doctored_bound = bool(
                doctored_stage2["seam_layer"]["domain_complex"][
                    "complex_freeze_sha256"
                ]
                == domain_complex["complex_freeze_sha256"]
                and doctored_stage2.get("source_projection_sha256")
                == source_projection_sha256
            )
        controls["tampered_domain_freeze"] = {
            "control_failure_detected": bool(
                domain_bound and not doctored_bound
            )
        }

        refusal = produce_matter_attachment_receipt(
            config={**main_config, "yukawa_target": "input"},
            run_controls=False,
        )
        controls["yukawa_injection"] = {
            "control_failure_detected": bool(refusal.get("verdict") == "REFUSED")
        }

    controls_fail_closed = bool(controls) and all(
        row["control_failure_detected"] for row in controls.values()
    )

    clause_verdicts = {
        "upstream_inputs_resolved": bool(bundle["passed"]),
        "local_parent_receipt_bytes_pinned": bool(
            stage2_receipt_sha256 is not None
            and source_gap_receipt_sha256 is not None
        ),
        "local_stage2_same_source_domain_binding": domain_bound,
        "generation_table_exact": bool(
            generation["state_count_exact"]
            and generation["anomalies_vanish"]
            and generation["weak_doublets_per_family"] == 4
        ),
        "z6_kernel_fixes_all_states": z6["all_states_fixed"],
        "chirality_nondegenerate": chirality["chirality_nondegenerate"],
        "band_generation_rank_45": bool(
            band_rank == BAND_RANK
            and attachment_rank == MATTER_FIBER_RANK
            and bool(band["equals_exact_frame_projector"])
        ),
        "declared_matter_operator_probe_identities_exact": bool(
            operators["support_radius_one"]
            and operators["adjoint_identity_exact"]
            and operators["kinetic_identity_exact"]
            and operators["subcomplex_naturality_exact"]
        ),
        "conditional_gap_inheritance_exact": bool(gap["inherited"]),
        "separate_issue_314_spin_packet_resolved": bool(
            spin_consumed["spin_structure_count"] == 1
            and spin_consumed["klein_four_obstruction"]
            and spin_rows.get("unique_nontrivial_central_involution")
            and spin_rows.get("unique_spin_structure_on_oriented_support")
        ),
        "local_domain_stage2_context_recorded": bool(
            lift_ambiguity_rank is not None
        ),
        "bounded_declared_key_scan_has_no_configured_target_fragments": bool(
            not forbidden_hits
        ),
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
        "capture_sha256_role": (
            "environment-sensitive full-capture diagnostic; not an "
            "identity gate for the local-domain stages"
        ),
        "source_projection_sha256": source_projection_sha256,
        "declared_matter_packet": {
            "status": "declared_imported_matter_packet",
            "source_selected": False,
            "table_sha256": _sha256_value(
                [
                    {
                        **dict(row),
                        "hypercharge": str(Fraction(row["hypercharge"])),
                    }
                    for row in GENERATION_TABLE
                ]
            ),
            "scope": (
                "the certificate recomputes conditional arithmetic from "
                "this packet; it does not derive or select the packet"
            ),
        },
        "upstream_pins": {
            "carrier_manifest_sha256": _sha256_value(carrier),
            "response_artifact_sha256": response.get("artifact_sha256"),
            "pole_residue_artifact_sha256": pole_residue.get("artifact_sha256"),
            "spin_artifact_sha256": spin_artifact.get("artifact_sha256"),
            "stage2_receipt_sha256": stage2_receipt_sha256,
            "source_gap_receipt_sha256": source_gap_receipt_sha256,
            "local_domain_bundle_passed": bundle["passed"],
            "stage2_receipt_present": bool(stage2 is not None),
            "gap_receipt_present": bool(gap_receipt is not None),
        },
        "generation_certificate": generation,
        "z6_kernel_certificate": z6,
        "chirality_certificate": chirality,
        "attachment": {
            "band_rank_measured": band_rank,
            "generation_states_recomputed": generation["weyl_state_count"],
            "complex_rank": attachment_rank,
            "band_residue_gates": {
                key: band[key]
                for key in (
                    "equals_exact_frame_projector",
                    "faithful_kernel_order",
                    "equivariant_under_all_automorphisms",
                    "galois_partner_at_maximal_pole",
                )
            },
        },
        "matter_operator_certificate": operators,
        "gap_inheritance_certificate": gap,
        "spin_layer": {
            "packet_status": "separate_pinned_issue_314_packet",
            "spin_to_local_domain_bridge_certified": False,
            "same_source_domain_certified": False,
            "spin_support_identity": {
                "artifact_sha256": spin_artifact.get("artifact_sha256"),
                "cell_counts": spin_artifact["support_homology"]["cells"],
            },
            "local_domain_identity": {
                "source_projection_sha256": (
                    source_projection_sha256
                ),
                "topology_freeze_sha256": domain_complex[
                    "complex_freeze_sha256"
                ],
                "visible_node_count": domain_complex["node_count"],
                "visible_edge_count": domain_complex["edge_count"],
            },
            "issue_314_artifact": spin_consumed,
            "local_domain_stage2_context": {
                "sign_transport": (
                    "matter sections are sign-twisted sections of the "
                    "stage-3 layer under the declared reversing seam "
                    "convention"
                ),
                "lift_ambiguity_rank": lift_ambiguity_rank,
            },
            "open_interface": "physical Spin/locality bridge",
        },
        "bounded_declared_key_scan": {
            "fragments": list(FORBIDDEN_INPUT_KEY_FRAGMENTS),
            "hits": forbidden_hits[:8],
            "scope": (
                "declared mapping keys only; no semantic classification "
                "of unlabeled values and no transitive input-closure claim"
            ),
        },
        "clause_verdicts": clause_verdicts,
        "negative_controls": controls,
        "controls_fail_closed": controls_fail_closed,
        "verdict": verdict,
        "MATTER_ATTACHMENT_RECEIPT": bool(verdict == "ATTAINED"),
        "blockers": blockers,
        "claim_boundary": (
            "Finite issue-569 assembled boundary packet: conditional on "
            "the declared imported fifteen-state "
            "matter packet, its exact recomputation gives "
            "vanishing anomalies, the exact diagonal Z6 fixing, the exact "
            "nondegenerate chirality grading, the measured rank-three band "
            "tensored to complex rank forty-five, and a declared "
            "sign-twisted tensor extension with exact identities on one "
            "deterministic rank-forty-five probe. The exactly positive "
            "local-domain gap is inherited conditionally under that "
            "declared extension. The source does not select a matter "
            "action. The issue-314 "
            "spin packet is pinned separately on its twelve-vertex support; "
            "no source, domain, or transport bridge attaches it to the "
            "issue-634 operator domain. The physical Spin/locality bridge, "
            "the matter-pole identification, the "
            "physical seam-action selection, and the laboratory current "
            "identification of issue 569 stay open, and the issue-617 "
            "copy-count invisibility for external completions is untouched. "
            "The bounded declared-key scan finds no configured Yukawa or "
            "laboratory-mass fragment; it is not semantic input closure. "
            "The finite generator-frequency residue used to select the "
            "band is recorded explicitly. No physical promotion follows "
            "from any output."
        ),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        receipt_bytes = _canonical_json(receipt).encode("utf-8")
        (out / "matter_attachment_receipt.json").write_bytes(receipt_bytes)
        manifest_path = out / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"schema": "oph.local-domain-stage1.manifest.v1"}
        )
        manifest["matter_attachment_receipt"] = "matter_attachment_receipt.json"
        manifest["matter_attachment_receipt_sha256"] = _sha256_bytes(
            receipt_bytes
        )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=1) + "\n",
            encoding="utf-8",
        )
    return receipt
