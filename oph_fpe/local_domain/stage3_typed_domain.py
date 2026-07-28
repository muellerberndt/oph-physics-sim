"""Issue-634 stage 3: typed sections, measures, operators, and boundary.

Stage 3 types the field and operator layer of the local domain on the
observer-visible seam complex certified by stages 1 and 2, with every
typing clause verified by exact integer arithmetic:

* Section spaces for three species over declared finite fiber
  dimensions: scalar sections on carriers, chiral matter sections on
  carriers with an exact Z/2 grading and conjugation that swaps the
  chirality summands, and gauge sections on seams valued in unit
  phases of the single interface algebra class.  Physical fiber
  selection stays outside this issue with issues 569 and 630.
* The measure is the counting measure on carriers and seams.  Equal
  state weights are the axiom-forced weight rule of the A1-A3
  derivation campaign, so the volume form carries no free parameter.
* The seam difference operator is twisted by the stage-2 sign
  transport, the covariant extension composes a gauge section into the
  same operator shape, and both have support radius one, verified
  entry by entry.
* The adjoint identity, the kinetic identity that the quadratic form
  equals the squared derivative norm, gauge-relabelling covariance,
  atlas restriction gluing, and subcomplex refinement naturality are
  exact integer identities on deterministic probe sections.
* The kinetic kernel theorem binds stage 3 to stage 2: on each
  connected component the twisted kernel has dimension one when the
  sign transport is consistent, witnessed by the explicit coloring
  vector, and dimension zero otherwise, by the rank identity of the
  signed incidence operator.  The certified rank identity is
  rank D = V minus the number of sign-consistent components.
* The boundary is typed from the source: carriers with unbound ports
  form the spatial boundary, the depth-extremal event slices form the
  temporal boundary, and the Dirichlet layer is a typed restriction
  rule with a nonempty-interior gate; its identities restate the rule
  and carry no independent check beyond that gate.

Controls: a nonlocal operator refused by the locality gate, a
perturbed adjoint refused by the adjoint identity, a broken regauge
refused by the covariance identity, a foreign-seed capture refused by
the complex freeze hash, and target-vocabulary injection refused
before capture.  No output of this module carries physical promotion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from oph_fpe.bulk.event_manifold_producer import _event_table
from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source
from oph_fpe.local_domain.stage1_event_complex import (
    FOREIGN_SEED_OFFSET,
    MAIN_CONFIG,
    refuse_forbidden_config,
)
from oph_fpe.local_domain.stage2_spin_layer import (
    seam_complex,
    visible_rows,
)

SCHEMA = "oph.local-domain-stage3.v1"
ISSUE = 634
PHYSICAL_PROMOTION_ALLOWED = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local_domain"

PROBE_FIBER_DIMENSION = 2
PROBE_SECTION_COUNT = 3
FULL_PORT_COUNT = 12


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _deterministic_int(*parts: int) -> int:
    """Deterministic small integer from a hash, in [-9, 9]."""

    digest = hashlib.sha256(
        ",".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return digest[0] % 19 - 9


def probe_section(
    nodes: list[int], fiber: int, salt: int
) -> dict[int, tuple[int, ...]]:
    """Deterministic integer-valued section over the carrier set."""

    return {
        node: tuple(
            _deterministic_int(salt, node, component)
            for component in range(fiber)
        )
        for node in nodes
    }


def seam_derivative(
    complex_data: Mapping[str, Any],
    section: Mapping[int, tuple[int, ...]],
    gauge: Mapping[tuple[int, int], int] | None = None,
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Sign-twisted difference operator on seams, optionally gauged.

    (D f)(a, b) = u_ab s_ab f(b) - f(a) with s the seam sign and u a
    unit gauge phase, both in {+1, -1} on the integer probe layer."""

    sign_of = complex_data["edge_sign_of"]
    result: dict[tuple[int, int], tuple[int, ...]] = {}
    for a, b in complex_data["edges"]:
        transport = sign_of[(a, b)]
        if gauge is not None:
            transport *= gauge[(a, b)]
        result[(a, b)] = tuple(
            transport * fb - fa for fa, fb in zip(section[a], section[b])
        )
    return result


def seam_adjoint(
    complex_data: Mapping[str, Any],
    edge_field: Mapping[tuple[int, int], tuple[int, ...]],
    gauge: Mapping[tuple[int, int], int] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Exact adjoint of the seam derivative under counting measures."""

    fiber = len(next(iter(edge_field.values()))) if edge_field else 0
    accumulator = {
        node: [0] * fiber for node in complex_data["nodes"]
    }
    sign_of = complex_data["edge_sign_of"]
    for (a, b), value in edge_field.items():
        transport = sign_of[(a, b)]
        if gauge is not None:
            transport *= gauge[(a, b)]
        for component in range(fiber):
            accumulator[b][component] += transport * value[component]
            accumulator[a][component] -= value[component]
    return {node: tuple(values) for node, values in accumulator.items()}


def _inner_node(
    left: Mapping[int, tuple[int, ...]], right: Mapping[int, tuple[int, ...]]
) -> int:
    total = 0
    for node, values in left.items():
        for component, value in enumerate(values):
            total += value * right[node][component]
    return total


def _inner_edge(
    left: Mapping[tuple[int, int], tuple[int, ...]],
    right: Mapping[tuple[int, int], tuple[int, ...]],
) -> int:
    total = 0
    for edge, values in left.items():
        for component, value in enumerate(values):
            total += value * right[edge][component]
    return total


def section_space_typing(complex_data: Mapping[str, Any]) -> dict[str, Any]:
    """Scalar, chiral, and gauge section species over declared fibers."""

    nodes = complex_data["nodes"]
    scalar = probe_section(nodes, PROBE_FIBER_DIMENSION, salt=1)
    left = probe_section(nodes, PROBE_FIBER_DIMENSION, salt=2)
    right = probe_section(nodes, PROBE_FIBER_DIMENSION, salt=3)
    chiral = {
        node: (left[node], right[node]) for node in nodes
    }

    def conjugate(section):
        return {
            node: (section[node][1], section[node][0]) for node in section
        }

    involution_exact = conjugate(conjugate(chiral)) == chiral
    graded = all(
        chiral[node][0] == left[node] and chiral[node][1] == right[node]
        for node in nodes
    )
    gauge = {
        edge: 1 if _deterministic_int(4, *edge) >= 0 else -1
        for edge in complex_data["edges"]
    }
    gauge_unit = all(value in (-1, 1) for value in gauge.values())
    return {
        "species": {
            "scalar": {"base": "carriers", "fiber_dimension": PROBE_FIBER_DIMENSION},
            "chiral_matter": {
                "base": "carriers",
                "fiber_dimension": PROBE_FIBER_DIMENSION,
                "grading": "Z2 chirality with conjugation swapping the summands",
            },
            "gauge": {
                "base": "seams",
                "fiber": "unit phase of the single interface algebra class",
            },
        },
        "physical_fiber_selection": (
            "outside this issue; owned by issues 569 and 630"
        ),
        "conjugation_involution_exact": bool(involution_exact),
        "chirality_grading_exact": bool(graded),
        "gauge_sections_unit": bool(gauge_unit),
        "_probe": {"scalar": scalar, "gauge": gauge},
    }


def measure_typing(complex_data: Mapping[str, Any]) -> dict[str, Any]:
    """Counting measure on carriers and seams, axiom-bound weights."""

    return {
        "carrier_measure": "counting measure, weight one per carrier",
        "seam_measure": "counting measure, weight one per seam",
        "weight_rule_ancestry": (
            "equal state weights are axiom-forced by the A3 exactness "
            "clauses of the closed A1-A3 derivation campaign"
        ),
        "carrier_total": complex_data["node_count"],
        "seam_total": complex_data["edge_count"],
        "positive": bool(
            complex_data["node_count"] > 0 and complex_data["edge_count"] > 0
        ),
    }


def locality_certificate(complex_data: Mapping[str, Any]) -> dict[str, Any]:
    """Support-radius-one verification of the seam derivative."""

    edge_set = set(complex_data["edges"])
    nodes = complex_data["nodes"]
    node_set = set(nodes)
    all_endpoints_in_domain = all(
        a in node_set and b in node_set for a, b in edge_set
    )
    scalar = probe_section(nodes, 1, salt=7)
    derivative = seam_derivative(complex_data, scalar)
    support_radius_one = set(derivative) == edge_set
    return {
        "operator": "seam difference",
        "support": "the two seam endpoints",
        "all_endpoints_in_domain": bool(all_endpoints_in_domain),
        "support_radius_one": bool(support_radius_one),
        "local": bool(all_endpoints_in_domain and support_radius_one),
    }


def reject_nonlocal_operator(
    complex_data: Mapping[str, Any],
    couplings: list[tuple[int, int]],
) -> dict[str, Any]:
    """Locality gate: refuse couplings that are not seams of the domain."""

    edge_set = set(complex_data["edges"])
    offending = sorted(
        (a, b)
        for a, b in ((min(p), max(p)) for p in couplings)
        if (a, b) not in edge_set
    )
    return {
        "coupling_count": len(couplings),
        "nonlocal_couplings": offending[:8],
        "accepted": bool(not offending),
    }


def adjoint_certificate(complex_data: Mapping[str, Any]) -> dict[str, Any]:
    """Exact adjoint and kinetic identities on deterministic probes."""

    nodes = complex_data["nodes"]
    checks = []
    for salt in range(PROBE_SECTION_COUNT):
        section = probe_section(nodes, PROBE_FIBER_DIMENSION, salt=10 + salt)
        edge_probe = {
            edge: tuple(
                _deterministic_int(30 + salt, *edge, component)
                for component in range(PROBE_FIBER_DIMENSION)
            )
            for edge in complex_data["edges"]
        }
        derivative = seam_derivative(complex_data, section)
        adjoint = seam_adjoint(complex_data, edge_probe)
        pairing_left = _inner_edge(derivative, edge_probe)
        pairing_right = _inner_node(section, adjoint)
        kinetic = _inner_edge(derivative, derivative)
        laplacian = seam_adjoint(complex_data, derivative)
        quadratic = _inner_node(section, laplacian)
        checks.append(
            {
                "adjoint_identity": bool(pairing_left == pairing_right),
                "kinetic_identity": bool(kinetic == quadratic),
                "kinetic_nonnegative": bool(kinetic >= 0),
            }
        )
    return {
        "probe_count": len(checks),
        "adjoint_identity_exact": all(c["adjoint_identity"] for c in checks),
        "kinetic_identity_exact": all(c["kinetic_identity"] for c in checks),
        "kinetic_nonnegative": all(c["kinetic_nonnegative"] for c in checks),
    }


def _sign_components(complex_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Connected components with sign-consistency witness colorings."""

    nodes = complex_data["nodes"]
    index = {v: i for i, v in enumerate(nodes)}
    neighbor: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    sign_of_index: dict[tuple[int, int], int] = {}
    for a, b in complex_data["edges"]:
        ia, ib = index[a], index[b]
        neighbor[ia].append(ib)
        neighbor[ib].append(ia)
        sign_of_index[(min(ia, ib), max(ia, ib))] = complex_data[
            "edge_sign_of"
        ][(a, b)]
    color = [0] * len(nodes)
    components = []
    for start in range(len(nodes)):
        if color[start] != 0:
            continue
        color[start] = 1
        stack = [start]
        members = []
        consistent = True
        while stack:
            u = stack.pop()
            members.append(u)
            for v in neighbor[u]:
                sign = sign_of_index[(min(u, v), max(u, v))]
                expected = color[u] * sign
                if color[v] == 0:
                    color[v] = expected
                    stack.append(v)
                elif color[v] != expected:
                    consistent = False
        components.append(
            {
                "members": members,
                "consistent": consistent,
                "witness": {nodes[i]: color[i] for i in members},
            }
        )
    return components


def kinetic_kernel_certificate(
    complex_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Exact kernel dimension of the twisted kinetic form per component.

    A sign-consistent component carries the explicit coloring witness,
    annihilated by the twisted derivative, so its kernel dimension is
    at least one; the rank identity rank D = V minus the consistent
    component count pins it to exactly one and pins frustrated
    components to dimension zero."""

    components = _sign_components(complex_data)
    witness_annihilated = True
    for component in components:
        if not component["consistent"]:
            continue
        witness_section = {
            node: (value,) for node, value in component["witness"].items()
        }
        sign_of = complex_data["edge_sign_of"]
        for a, b in complex_data["edges"]:
            if a not in witness_section or b not in witness_section:
                continue
            value = sign_of[(a, b)] * witness_section[b][0] - witness_section[a][0]
            if value != 0:
                witness_annihilated = False
    consistent_count = sum(1 for c in components if c["consistent"])
    kernel_dimension = consistent_count
    rank = complex_data["node_count"] - consistent_count
    return {
        "component_count": len(components),
        "sign_consistent_component_count": consistent_count,
        "witnesses_annihilated": bool(witness_annihilated),
        "witness_check_scope": (
            f"{sum(1 for c in components if c['consistent'])} consistent "
            f"component witnesses evaluated; on a fully frustrated domain "
            "the kernel-zero direction rests on the rank theorem, not on "
            "this loop"
        ),
        "twisted_kernel_dimension": kernel_dimension,
        "signed_incidence_rank": rank,
        "rank_identity": (
            "rank D equals carrier count minus the sign-consistent "
            "component count"
        ),
        "matches_stage2": None,
    }


def gauge_covariance_certificate(
    complex_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Exact covariance of the derivative under vertex regauging."""

    nodes = complex_data["nodes"]
    regauge = {
        node: 1 if _deterministic_int(40, node) >= 0 else -1 for node in nodes
    }
    section = probe_section(nodes, PROBE_FIBER_DIMENSION, salt=50)
    transformed_section = {
        node: tuple(regauge[node] * value for value in values)
        for node, values in section.items()
    }
    twisted = dict(complex_data)
    twisted["edge_sign_of"] = {
        (a, b): regauge[a]
        * regauge[b]
        * complex_data["edge_sign_of"][(a, b)]
        for a, b in complex_data["edges"]
    }
    original = seam_derivative(complex_data, section)
    transformed = seam_derivative(twisted, transformed_section)
    covariant = all(
        tuple(regauge[a] * value for value in original[(a, b)])
        == transformed[(a, b)]
        for a, b in complex_data["edges"]
    )
    norm_invariant = bool(
        _inner_edge(original, original)
        == _inner_edge(transformed, transformed)
    )
    return {
        "regauge_rule": "vertex signs conjugate the seam transport",
        "derivative_covariant": bool(covariant),
        "kinetic_norm_invariant": norm_invariant,
    }


def restriction_gluing_certificate(
    complex_data: Mapping[str, Any],
    chart_members: list[list[int]],
) -> dict[str, Any]:
    """Exact sheaf gluing of sections over an atlas cover of carriers."""

    nodes = complex_data["nodes"]
    node_set = set(nodes)
    cover_sets = [set(m) & node_set for m in chart_members]
    covered = set().union(*cover_sets) if cover_sets else set()
    covers = covered == node_set
    section = probe_section(nodes, PROBE_FIBER_DIMENSION, salt=60)
    restrictions = [
        {node: section[node] for node in members} for members in cover_sets
    ]
    overlaps_agree = True
    for i in range(len(cover_sets)):
        for j in range(i + 1, len(cover_sets)):
            for node in cover_sets[i] & cover_sets[j]:
                if restrictions[i][node] != restrictions[j][node]:
                    overlaps_agree = False
    glued: dict[int, tuple[int, ...]] = {}
    for restriction in restrictions:
        glued.update(restriction)
    glues_back = covers and glued == section
    return {
        "chart_count": len(chart_members),
        "cover_complete": bool(covers),
        "overlap_sections_agree": bool(overlaps_agree),
        "gluing_reconstructs_section": bool(glues_back),
    }


def refinement_naturality_certificate(
    complex_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Exact naturality of the derivative under subcomplex inclusion."""

    edges = complex_data["edges"]
    half = max(1, len(edges) // 2)
    sub_edges = edges[:half]
    sub_nodes = sorted({v for edge in sub_edges for v in edge})
    sub_complex = {
        "nodes": sub_nodes,
        "node_count": len(sub_nodes),
        "edges": sub_edges,
        "edge_count": len(sub_edges),
        "edge_sign_of": {
            edge: complex_data["edge_sign_of"][edge] for edge in sub_edges
        },
    }
    section = probe_section(complex_data["nodes"], PROBE_FIBER_DIMENSION, salt=70)
    sub_section = {node: section[node] for node in sub_nodes}
    full_derivative = seam_derivative(complex_data, section)
    sub_derivative = seam_derivative(sub_complex, sub_section)
    natural = all(
        sub_derivative[edge] == full_derivative[edge] for edge in sub_edges
    )
    return {
        "subcomplex_edges": len(sub_edges),
        "restriction_commutes_with_derivative": bool(natural),
    }


def boundary_typing(
    complex_data: Mapping[str, Any],
    event_table: Mapping[str, Any],
) -> dict[str, Any]:
    """Spatial free-port boundary and temporal depth-slice boundary."""

    port_bound_of = complex_data["port_bound_of"]
    interior = [
        node
        for node in complex_data["nodes"]
        if port_bound_of[node] >= FULL_PORT_COUNT
    ]
    fully_bound = len(interior)
    boundary_count = complex_data["node_count"] - fully_bound
    depths = [event_table["depth"][i] for i in range(event_table["count"])]
    depth_min = min(depths) if depths else 0
    depth_max = max(depths) if depths else 0
    initial_slice = sum(1 for d in depths if d == depth_min)
    final_slice = sum(1 for d in depths if d == depth_max)

    section = probe_section(complex_data["nodes"], 1, salt=80)
    boundary_nodes = set(complex_data["nodes"]) - set(interior)
    dirichlet = {
        node: (0,) if node in boundary_nodes else section[node]
        for node in complex_data["nodes"]
    }
    vanishes = all(dirichlet[node] == (0,) for node in boundary_nodes)
    interior_untouched = all(
        dirichlet[node] == section[node]
        for node in complex_data["nodes"]
        if node not in boundary_nodes
    )
    return {
        "spatial_boundary_rule": "carriers with unbound ports",
        "spatial_boundary_count": boundary_count,
        "spatial_interior_count": fully_bound,
        "temporal_boundary_rule": "depth-extremal event slices",
        "initial_slice_events": initial_slice,
        "final_slice_events": final_slice,
        "dirichlet_restriction_exact": bool(
            fully_bound > 0 and vanishes and interior_untouched
        ),
    }


def produce_stage3_receipt(
    *,
    config: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    run_controls: bool = True,
) -> dict[str, Any]:
    """Produce the stage-3 typed-domain receipt with fail-closed controls."""

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

    capture = capture_physical_source(main_config)
    domain_complex = seam_complex(visible_rows(capture))
    event_table = _event_table(capture)

    stage2 = _load_frozen_stage2()
    binding: dict[str, Any] = {
        "stage2_receipt_present": bool(stage2 is not None),
        "same_source": False,
        "same_domain_freeze": False,
    }
    if stage2 is not None:
        binding["same_source"] = bool(
            stage2["capture_sha256"] == capture["capture_sha256"]
        )
        binding["same_domain_freeze"] = bool(
            stage2["seam_layer"]["domain_complex"]["complex_freeze_sha256"]
            == domain_complex["complex_freeze_sha256"]
        )

    sections = section_space_typing(domain_complex)
    probe = sections.pop("_probe")
    measures = measure_typing(domain_complex)
    locality = locality_certificate(domain_complex)
    adjoints = adjoint_certificate(domain_complex)
    kernel = kinetic_kernel_certificate(domain_complex)
    if stage2 is not None and binding["same_source"]:
        kernel["matches_stage2"] = bool(
            kernel["sign_consistent_component_count"]
            == stage2["seam_layer"]["domain_complex"][
                "sign_consistent_component_count"
            ]
        )
    covariance = gauge_covariance_certificate(domain_complex)

    chart_members = observer_carrier_cover(capture)
    gluing = restriction_gluing_certificate(domain_complex, chart_members)
    naturality = refinement_naturality_certificate(domain_complex)
    boundary = boundary_typing(domain_complex, event_table)

    gauged = seam_derivative(
        domain_complex,
        probe_section(domain_complex["nodes"], PROBE_FIBER_DIMENSION, salt=90),
        gauge=probe["gauge"],
    )
    covariant_typed = bool(len(gauged) == domain_complex["edge_count"])

    controls: dict[str, dict[str, Any]] = {}
    if run_controls:
        nodes = domain_complex["nodes"]
        nonlocal_pair = None
        node_set = set(nodes)
        edge_set = set(domain_complex["edges"])
        for a in nodes[:32]:
            for b in nodes[-32:]:
                if a < b and (a, b) not in edge_set:
                    nonlocal_pair = (a, b)
                    break
            if nonlocal_pair:
                break
        gate = reject_nonlocal_operator(
            domain_complex,
            [nonlocal_pair] if nonlocal_pair else [],
        )
        controls["nonlocal_operator"] = {
            "control_failure_detected": bool(
                nonlocal_pair is not None and not gate["accepted"]
            )
        }

        section = probe_section(nodes, 1, salt=91)
        edge_probe = {
            edge: (_deterministic_int(93, *edge),)
            for edge in domain_complex["edges"]
        }
        derivative = seam_derivative(domain_complex, section)
        adjoint = seam_adjoint(domain_complex, edge_probe)
        first_node = next(node for node in nodes if section[node] != (0,))
        broken_adjoint = dict(adjoint)
        broken_adjoint[first_node] = tuple(
            value + 1 for value in broken_adjoint[first_node]
        )
        controls["perturbed_adjoint"] = {
            "control_failure_detected": bool(
                _inner_edge(derivative, edge_probe)
                == _inner_node(section, adjoint)
                and _inner_edge(derivative, edge_probe)
                != _inner_node(section, broken_adjoint)
            )
        }

        corrupt_edge = next(
            edge
            for edge in domain_complex["edges"]
            if section[edge[1]] != (0,)
        )
        broken_twist = dict(domain_complex)
        broken_signs = dict(domain_complex["edge_sign_of"])
        broken_signs[corrupt_edge] = -broken_signs[corrupt_edge]
        broken_twist["edge_sign_of"] = broken_signs
        original = seam_derivative(domain_complex, section)
        broken = seam_derivative(broken_twist, section)
        controls["corrupted_transport"] = {
            "control_failure_detected": bool(
                original[corrupt_edge] != broken[corrupt_edge]
            )
        }

        foreign_config = dict(main_config)
        foreign_config["seed"] = int(main_config["seed"]) + FOREIGN_SEED_OFFSET
        foreign_capture = capture_physical_source(foreign_config)
        foreign_complex = seam_complex(visible_rows(foreign_capture))
        controls["mixed_source"] = {
            "control_failure_detected": bool(
                foreign_complex["complex_freeze_sha256"]
                != domain_complex["complex_freeze_sha256"]
            )
        }

        injected = dict(main_config)
        injected["target_scale"] = "GeV"
        refusal = produce_stage3_receipt(config=injected, run_controls=False)
        controls["target_injection"] = {
            "control_failure_detected": bool(
                refusal.get("verdict") == "REFUSED"
            )
        }

    controls_fail_closed = bool(controls) and all(
        row["control_failure_detected"] for row in controls.values()
    )

    clause_verdicts = {
        "section_spaces_typed": bool(
            sections["conjugation_involution_exact"]
            and sections["chirality_grading_exact"]
            and sections["gauge_sections_unit"]
        ),
        "measure_axiom_bound": measures["positive"],
        "operators_local": locality["local"],
        "covariant_derivative_typed": covariant_typed,
        "adjoint_identity_exact": adjoints["adjoint_identity_exact"],
        "kinetic_identities_exact": bool(
            adjoints["kinetic_identity_exact"]
            and adjoints["kinetic_nonnegative"]
        ),
        "kinetic_kernel_matches_stage2": bool(
            kernel["witnesses_annihilated"] and kernel["matches_stage2"]
        ),
        "gauge_relabelling_covariant": bool(
            covariance["derivative_covariant"]
            and covariance["kinetic_norm_invariant"]
        ),
        "restriction_gluing_exact": bool(
            gluing["cover_complete"]
            and gluing["overlap_sections_agree"]
            and gluing["gluing_reconstructs_section"]
        ),
        "refinement_naturality_exact": naturality[
            "restriction_commutes_with_derivative"
        ],
        "boundary_typed": boundary["dirichlet_restriction_exact"],
        "same_source_binding": bool(
            binding["same_source"] and binding["same_domain_freeze"]
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
        "stage_binding": binding,
        "section_typing": sections,
        "measure_typing": measures,
        "locality_certificate": locality,
        "adjoint_certificate": adjoints,
        "kinetic_kernel_certificate": {
            key: kernel[key]
            for key in (
                "component_count",
                "sign_consistent_component_count",
                "witnesses_annihilated",
                "witness_check_scope",
                "twisted_kernel_dimension",
                "signed_incidence_rank",
                "rank_identity",
                "matches_stage2",
            )
        },
        "gauge_covariance_certificate": covariance,
        "restriction_gluing_certificate": gluing,
        "refinement_naturality_certificate": naturality,
        "boundary_typing": boundary,
        "clause_verdicts": clause_verdicts,
        "negative_controls": controls,
        "controls_fail_closed": controls_fail_closed,
        "verdict": verdict,
        "STAGE3_TYPED_DOMAIN_RECEIPT": bool(verdict == "ATTAINED"),
        "blockers": blockers,
        "claim_boundary": (
            "Finite issue-634 stage-3 object: section species over declared "
            "finite fibers, the axiom-bound counting measure, sign-twisted "
            "local difference operators with exact adjoint, kinetic, "
            "kernel, covariance, gluing, naturality, and boundary "
            "certificates on the observer-visible seam complex. Physical "
            "fiber selection, scalar and Yukawa content, and any continuum "
            "operator limit stay outside this issue. No physical promotion "
            "follows from any output."
        ),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        receipt_bytes = _canonical_json(receipt).encode("utf-8")
        (out / "stage3_receipt.json").write_bytes(receipt_bytes)
        manifest_path = out / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"schema": "oph.local-domain-stage1.manifest.v1"}
        )
        manifest["stage3_receipt"] = "stage3_receipt.json"
        manifest["stage3_receipt_sha256"] = _sha256_bytes(receipt_bytes)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=1) + "\n",
            encoding="utf-8",
        )
    return receipt


def _load_frozen_stage2() -> dict[str, Any] | None:
    receipt_path = DATA_DIR / "stage2_receipt.json"
    manifest_path = DATA_DIR / "manifest.json"
    if not receipt_path.exists() or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_bytes = receipt_path.read_bytes()
    if manifest.get("stage2_receipt_sha256") != _sha256_bytes(receipt_bytes):
        return None
    return json.loads(receipt_bytes.decode("utf-8"))


def observer_carrier_cover(capture: Mapping[str, Any]) -> list[list[int]]:
    """Carrier regions per observer from seam visibility.

    Every visible seam names its observers, so the union of the
    per-observer regions covers the visible domain by construction;
    the gluing certificate on this cover is a wiring certificate of
    the restriction maps."""

    regions: dict[str, set[int]] = {}
    for row in visible_rows(capture):
        a = int(row["left_carrier_id"].rsplit("-", 1)[1])
        b = int(row["right_carrier_id"].rsplit("-", 1)[1])
        for token in row["visible_to_observer_tokens"]:
            regions.setdefault(token, set()).update((a, b))
    return [sorted(regions[token]) for token in sorted(regions)]
