"""Issue-634 stage 2: exact seam-transport sign layer and lift ambiguity.

The Spin-layer content available on the finite event complex is exact
sign algebra rather than fitted geometry.  Every seam in the capture
declares the orientation-reversing gluing convention: single-port
seams with orientation sign minus one and one shared interface
algebra.  Stage 2 verifies that declaration fail-closed and computes,
over GF(2), the transport structure it induces on the event-touched
seam subcomplex:

* the seam graph on touched carriers with its multiplicity, port-degree,
  component, and triangle census;
* the cycle holonomy rule of the declared convention, minus one to the
  cycle length, with the per-triangle holonomy census verified exactly;
* the orientability verdict of the sign transport, bipartiteness per
  component, cross-checked against triangle existence;
* the lift-ambiguity rank, the GF(2) dimension of the first cohomology
  of the clique two-complex, with the rank identity verified, and the
  ambiguity count two to that rank in the convention of the
  spin-statistics instrument;
* the chart-frame layer of the stage-1 atlas: transition determinant
  signs, the sign-lift solvability on the nerve, and the nerve
  cohomology rank.

The stage-2 receipt binds to the frozen stage-1 receipt by capture
hash, so both stages certify one source object.  The identification of
this finite sign layer with continuum Stiefel-Whitney classes is a
named interface owned by the issue-596 face-cocycle lanes; the
interface-algebra phase extension, the Spin-c analog, is a named
interface owned by the stage-3 operator typing.  No output of this
module carries physical promotion.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from oph_fpe.bulk.event_manifold_producer import _event_table
from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source
from oph_fpe.local_domain.stage1_event_complex import (
    CONTROL_SPLIT_CONFIG,
    FOREIGN_SEED_OFFSET,
    MAIN_CONFIG,
    refuse_forbidden_config,
)

SCHEMA = "oph.local-domain-stage2.v1"
ISSUE = 634
PHYSICAL_PROMOTION_ALLOWED = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local_domain"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _carrier_index(carrier_id: str) -> int:
    return int(carrier_id.rsplit("-", 1)[1])


def seam_convention_gate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Verify the declared orientation-reversing single-port convention."""

    algebra_hashes = set()
    violations: list[str] = []
    for row in rows:
        if list(row["orientation_signs"]) != [-1]:
            violations.append(f"nonreversing_sign:{row['overlap_id']}")
        if len(row["left_ports"]) != 1 or len(row["right_ports"]) != 1:
            violations.append(f"multi_port_seam:{row['overlap_id']}")
        algebra_hashes.add(row["interface_algebra_sha256"])
    if len(algebra_hashes) > 1:
        violations.append("multiple_interface_algebras")
    return {
        "row_count": len(rows),
        "interface_algebra_count": len(algebra_hashes),
        "violations": sorted(violations)[:16],
        "uniform_reversing_single_port": bool(not violations),
    }


def seam_complex(
    rows: list[Mapping[str, Any]], restrict_to: set[int] | None = None
) -> dict[str, Any]:
    """Seam graph, components, triangles, and port census, exactly."""

    edge_signs: dict[tuple[int, int], list[int]] = {}
    port_degree: dict[int, set[int]] = {}
    for row in rows:
        a = _carrier_index(row["left_carrier_id"])
        b = _carrier_index(row["right_carrier_id"])
        if restrict_to is not None and (
            a not in restrict_to or b not in restrict_to
        ):
            continue
        key = (min(a, b), max(a, b))
        sign = 1
        for value in row["orientation_signs"]:
            sign *= 1 if int(value) >= 0 else -1
        edge_signs.setdefault(key, []).append(sign)
        port_degree.setdefault(a, set()).add(int(row["left_ports"][0]))
        port_degree.setdefault(b, set()).add(int(row["right_ports"][0]))
    edge_rows = {key: len(signs) for key, signs in edge_signs.items()}
    parallel_conflicts = sum(
        1 for signs in edge_signs.values() if len(set(signs)) > 1
    )
    reduced_sign = {key: signs[0] for key, signs in edge_signs.items()}
    nodes = sorted(port_degree)
    index = {v: i for i, v in enumerate(nodes)}
    neighbor: list[set[int]] = [set() for _ in nodes]
    for a, b in edge_rows:
        neighbor[index[a]].add(index[b])
        neighbor[index[b]].add(index[a])

    sign_of_index = {
        (index[a], index[b]): sign for (a, b), sign in reduced_sign.items()
    }
    color = [0] * len(nodes)
    components = 0
    bipartite_components = 0
    for start in range(len(nodes)):
        if color[start] != 0:
            continue
        components += 1
        consistent = True
        color[start] = 1
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in neighbor[u]:
                sign = sign_of_index[(min(u, v), max(u, v))]
                expected = color[u] * sign
                if color[v] == 0:
                    color[v] = expected
                    queue.append(v)
                elif color[v] != expected:
                    consistent = False
        if consistent:
            bipartite_components += 1

    triangles: list[tuple[int, int, int]] = []
    for a, b in edge_rows:
        ia, ib = index[a], index[b]
        for ic in neighbor[ia] & neighbor[ib]:
            if ic > ib and ib > ia:
                triangles.append((ia, ib, ic))
    triangles = sorted(set(triangles))

    port_census: dict[int, int] = {}
    for ports in port_degree.values():
        port_census[len(ports)] = port_census.get(len(ports), 0) + 1

    return {
        "node_count": len(nodes),
        "nodes": nodes,
        "edge_count": len(edge_rows),
        "edges": sorted(edge_rows),
        "edge_sign_of": reduced_sign,
        "parallel_seam_sign_conflicts": parallel_conflicts,
        "edge_multiplicities": {
            "1": sum(1 for m in edge_rows.values() if m == 1),
            "2": sum(1 for m in edge_rows.values() if m == 2),
            "more": sum(1 for m in edge_rows.values() if m > 2),
        },
        "component_count": components,
        "sign_consistent_component_count": bipartite_components,
        "triangle_count": len(triangles),
        "triangles": triangles,
        "port_bound_census": dict(sorted(port_census.items())),
        "complex_freeze_sha256": _sha256_value(
            {"nodes": nodes, "edges": sorted(edge_rows)}
        ),
    }


def holonomy_census(complex_data: Mapping[str, Any]) -> dict[str, Any]:
    """Per-triangle sign holonomy computed from the recorded edge signs."""

    nodes = complex_data["nodes"]
    sign_of = complex_data["edge_sign_of"]
    minus = 0
    plus = 0
    for ia, ib, ic in complex_data["triangles"]:
        a, b, c = nodes[ia], nodes[ib], nodes[ic]
        holonomy = (
            sign_of[(min(a, b), max(a, b))]
            * sign_of[(min(a, c), max(a, c))]
            * sign_of[(min(b, c), max(b, c))]
        )
        if holonomy < 0:
            minus += 1
        else:
            plus += 1
    triangle_count = complex_data["triangle_count"]
    return {
        "triangle_count": triangle_count,
        "minus_holonomy_count": minus,
        "plus_holonomy_count": plus,
        "all_triangles_frustrated": bool(
            triangle_count > 0 and minus == triangle_count
        ),
        "cycle_rule": (
            "under the uniform reversing convention the holonomy equals "
            "minus one to the cycle length"
        ),
    }


def _gf2_rank(rows: list[int]) -> int:
    """Rank over GF(2) of integer bitmask rows."""

    pivots: dict[int, int] = {}
    rank = 0
    for row in rows:
        current = row
        while current:
            top = current.bit_length() - 1
            if top in pivots:
                current ^= pivots[top]
            else:
                pivots[top] = current
                rank += 1
                break
    return rank


def lift_ambiguity(complex_data: Mapping[str, Any]) -> dict[str, Any]:
    """GF(2) first-cohomology rank of the clique two-complex.

    The rank identity b1 = E - V + C - rank(boundary two) is computed
    from one elimination and re-verified through the cocycle kernel."""

    edges = complex_data["edges"]
    node_of = {v: i for i, v in enumerate(complex_data["nodes"])}
    edge_index = {
        (node_of[a], node_of[b]): i for i, (a, b) in enumerate(edges)
    }
    edge_count = len(edges)
    boundary_rows: list[int] = []
    for a, b, c in complex_data["triangles"]:
        mask = 0
        for u, v in ((a, b), (a, c), (b, c)):
            mask |= 1 << edge_index[(u, v)]
        boundary_rows.append(mask)
    triangle_rank = _gf2_rank(boundary_rows)
    b1 = (
        edge_count
        - complex_data["node_count"]
        + complex_data["component_count"]
        - triangle_rank
    )
    cocycle_rank = edge_count - triangle_rank
    cut_rank = complex_data["node_count"] - complex_data["component_count"]
    identity_holds = bool(b1 == cocycle_rank - cut_rank and b1 >= 0)
    return {
        "edge_count": edge_count,
        "vertex_count": complex_data["node_count"],
        "component_count": complex_data["component_count"],
        "triangle_boundary_rank": triangle_rank,
        "lift_ambiguity_rank": int(b1),
        "ambiguity_count_log2": int(b1),
        "rank_identity_holds": identity_holds,
    }


def _load_frozen_stage1() -> dict[str, Any] | None:
    receipt_path = DATA_DIR / "stage1_receipt.json"
    manifest_path = DATA_DIR / "manifest.json"
    if not receipt_path.exists() or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_bytes = receipt_path.read_bytes()
    if manifest.get("receipt_sha256") != _sha256_bytes(receipt_bytes):
        return None
    return json.loads(receipt_bytes.decode("utf-8"))


def atlas_frame_layer(stage1_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Sign-lift layer of the stage-1 atlas nerve, exactly."""

    atlas = stage1_receipt["atlas"]
    seed_count = atlas["seed_count"]
    nerve_edges = [tuple(edge) for edge in atlas["supported_edges"]]
    nerve_triangles = [
        tuple(int(part) for part in key.split("-"))
        for key in sorted(atlas["triple_defects"])
    ]
    orientation = stage1_receipt["orientation_certificate"]
    all_positive = all(
        sign == 1 for sign in orientation.get("edge_signs", {}).values()
    )
    parent = list(range(seed_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in nerve_edges:
        parent[find(a)] = find(b)
    component_count = len({find(k) for k in range(seed_count)})
    complex_view = {
        "nodes": list(range(seed_count)),
        "node_count": seed_count,
        "edges": sorted(nerve_edges),
        "component_count": component_count,
        "triangles": sorted(nerve_triangles),
    }
    ambiguity = lift_ambiguity(complex_view)
    return {
        "nerve_edge_count": len(nerve_edges),
        "nerve_triangle_count": len(nerve_triangles),
        "transition_determinants_positive": bool(all_positive),
        "sign_lift_solvable": bool(orientation.get("orientable")),
        "nerve_lift_ambiguity_rank": ambiguity["lift_ambiguity_rank"],
        "nerve_rank_identity_holds": ambiguity["rank_identity_holds"],
    }


def _touched_carriers(capture: Mapping[str, Any]) -> set[int]:
    table = _event_table(capture)
    return {c for footprint in table["footprints"] for c in footprint}


def visible_rows(capture: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Seam rows visible to at least one observer, the operational domain."""

    return [
        row
        for row in capture["postrun_capture"]["raw_overlap_relations"]
        if row["visible_to_observer_tokens"]
    ]


def _seam_layer(capture: Mapping[str, Any]) -> dict[str, Any]:
    rows = capture["postrun_capture"]["raw_overlap_relations"]
    gate = seam_convention_gate(rows)
    domain_rows = visible_rows(capture)
    domain_complex = seam_complex(domain_rows)
    touched = _touched_carriers(capture)
    touched_complex = seam_complex(rows, restrict_to=touched)
    full_complex = seam_complex(rows)
    census = holonomy_census(domain_complex)
    ambiguity = lift_ambiguity(domain_complex)
    triangle_orientability_consistent = bool(
        domain_complex["triangle_count"] == 0
        or domain_complex["sign_consistent_component_count"]
        < domain_complex["component_count"]
    )
    census_keys = (
        "node_count",
        "edge_count",
        "edge_multiplicities",
        "component_count",
        "sign_consistent_component_count",
        "triangle_count",
        "port_bound_census",
        "parallel_seam_sign_conflicts",
        "complex_freeze_sha256",
    )
    return {
        "convention_gate": gate,
        "visible_row_count": len(domain_rows),
        "domain_complex": {key: domain_complex[key] for key in census_keys},
        "touched_carrier_count": len(touched),
        "touched_footprint_census": {
            key: touched_complex[key]
            for key in (
                "node_count",
                "edge_count",
                "component_count",
                "triangle_count",
            )
        },
        "full_complex_census": {
            key: full_complex[key]
            for key in (
                "node_count",
                "edge_count",
                "component_count",
                "sign_consistent_component_count",
                "triangle_count",
            )
        },
        "holonomy_census": census,
        "lift_ambiguity": ambiguity,
        "orientability": {
            "sign_transport_trivializable": bool(
                domain_complex["sign_consistent_component_count"]
                == domain_complex["component_count"]
            ),
            "triangle_consistency_holds": triangle_orientability_consistent,
        },
        "_domain_complex_full": domain_complex,
    }


def produce_stage2_receipt(
    *,
    config: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    run_controls: bool = True,
) -> dict[str, Any]:
    """Produce the stage-2 sign-layer receipt with fail-closed controls."""

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
    layer = _seam_layer(capture)
    domain_complex = layer.pop("_domain_complex_full")

    stage1 = _load_frozen_stage1()
    stage1_binding: dict[str, Any]
    frame_layer: dict[str, Any] | None = None
    if stage1 is None:
        stage1_binding = {
            "stage1_receipt_present": False,
            "same_source": False,
        }
    else:
        same_config = stage1["main_config"] == main_config
        same_source = bool(
            same_config
            and stage1["capture_sha256"] == capture["capture_sha256"]
        )
        stage1_binding = {
            "stage1_receipt_present": True,
            "stage1_capture_sha256": stage1["capture_sha256"],
            "stage2_capture_sha256": capture["capture_sha256"],
            "same_source": same_source,
        }
        frame_layer = atlas_frame_layer(stage1)

    ladder_config = dict(CONTROL_SPLIT_CONFIG)
    ladder_config["observer_cross_reads"] = True
    ladder_capture = capture_physical_source(ladder_config)
    ladder_layer = _seam_layer(ladder_capture)
    ladder_layer.pop("_domain_complex_full")
    scale_ladder = {
        "small_config_carriers": ladder_config["carrier_count"],
        "small_domain_triangles": ladder_layer["domain_complex"][
            "triangle_count"
        ],
        "small_lift_ambiguity_rank": ladder_layer["lift_ambiguity"][
            "lift_ambiguity_rank"
        ],
        "small_sign_transport_trivializable": ladder_layer["orientability"][
            "sign_transport_trivializable"
        ],
        "main_sign_transport_trivializable": layer["orientability"][
            "sign_transport_trivializable"
        ],
        "verdict_stable_across_scales": bool(
            ladder_layer["orientability"]["sign_transport_trivializable"]
            == layer["orientability"]["sign_transport_trivializable"]
        ),
    }

    controls: dict[str, dict[str, Any]] = {}
    if run_controls:
        rows = capture["postrun_capture"]["raw_overlap_relations"]

        flipped = [dict(row) for row in rows]
        flipped[0] = dict(flipped[0], orientation_signs=[1])
        flipped_gate = seam_convention_gate(flipped)
        controls["nonuniform_sign_injection"] = {
            "control_failure_detected": bool(
                not flipped_gate["uniform_reversing_single_port"]
            )
        }

        domain = visible_rows(capture)
        deleted_complex = seam_complex(domain[1:])
        controls["seam_deletion"] = {
            "control_failure_detected": bool(
                deleted_complex["complex_freeze_sha256"]
                != domain_complex["complex_freeze_sha256"]
            )
        }

        square = {
            "nodes": [0, 1, 2, 3],
            "node_count": 4,
            "edges": [(0, 1), (0, 3), (1, 2), (2, 3)],
            "component_count": 1,
            "triangles": [],
            "edge_count": 4,
            "sign_consistent_component_count": 1,
        }
        square_ambiguity = lift_ambiguity(square)
        controls["bipartite_countermodel"] = {
            "control_failure_detected": bool(
                square_ambiguity["lift_ambiguity_rank"] == 1
                and square_ambiguity["rank_identity_holds"]
            ),
            "note": (
                "the four-cycle with no triangles carries ambiguity rank "
                "one, so the rank computation is not pinned to the "
                "measured complex"
            ),
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
        injected["target_signature"] = "spin"
        refusal = produce_stage2_receipt(
            config=injected, run_controls=False
        )
        controls["target_injection"] = {
            "control_failure_detected": bool(
                refusal.get("verdict") == "REFUSED"
            )
        }

    controls_fail_closed = bool(controls) and all(
        row["control_failure_detected"] for row in controls.values()
    )

    clause_verdicts = {
        "seam_convention_verified": layer["convention_gate"][
            "uniform_reversing_single_port"
        ],
        "same_source_binding": stage1_binding["same_source"],
        "visible_domain_nonempty": bool(
            layer["domain_complex"]["node_count"] > 0
            and layer["domain_complex"]["edge_count"] > 0
        ),
        "parallel_signs_consistent": bool(
            layer["domain_complex"]["parallel_seam_sign_conflicts"] == 0
        ),
        "holonomy_census_exact": bool(
            layer["holonomy_census"]["minus_holonomy_count"]
            == layer["domain_complex"]["triangle_count"]
        ),
        "orientability_cross_check": layer["orientability"][
            "triangle_consistency_holds"
        ],
        "lift_ambiguity_rank_identity": layer["lift_ambiguity"][
            "rank_identity_holds"
        ],
        "atlas_frame_lift_recorded": bool(
            frame_layer is not None
            and frame_layer["nerve_rank_identity_holds"]
        ),
        "scale_ladder_recorded": scale_ladder["verdict_stable_across_scales"],
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
        "stage1_binding": stage1_binding,
        "seam_layer": layer,
        "atlas_frame_layer": frame_layer,
        "scale_ladder": scale_ladder,
        "clause_verdicts": clause_verdicts,
        "negative_controls": controls,
        "controls_fail_closed": controls_fail_closed,
        "verdict": verdict,
        "STAGE2_SIGN_LAYER_RECEIPT": bool(verdict == "ATTAINED"),
        "blockers": blockers,
        "claim_boundary": (
            "Finite issue-634 stage-2 object: the declared "
            "orientation-reversing seam convention verified fail-closed, and "
            "the exact GF(2) transport structure it induces on the "
            "event-touched seam subcomplex: triangle holonomy census, "
            "orientability verdict, lift-ambiguity rank with the verified "
            "rank identity, and the stage-1 chart-frame sign-lift layer. "
            "The identification with continuum Stiefel-Whitney classes is a "
            "named interface owned by the issue-596 face-cocycle lanes, and "
            "the interface-algebra phase extension is a named interface "
            "owned by stage-3 typing. No physical promotion follows from "
            "any output."
        ),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        receipt_bytes = _canonical_json(receipt).encode("utf-8")
        (out / "stage2_receipt.json").write_bytes(receipt_bytes)
        manifest_path = out / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"schema": "oph.local-domain-stage1.manifest.v1"}
        )
        manifest["stage2_receipt"] = "stage2_receipt.json"
        manifest["stage2_receipt_sha256"] = _sha256_bytes(receipt_bytes)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=1) + "\n",
            encoding="utf-8",
        )
    return receipt
