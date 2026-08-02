"""Certify the exact proper-carrier action on the repair-derived completion.

This packet is downstream of the conditional port-Gram completion receipt.  It
uses only the pinned twelve-port carrier, its oriented faces, its declared
three-level recharting tower, and the exact Gram selected by the repair
response.  No physical or comparison data are read.

The finite result is exact.  The sixty orientation-preserving incidence
automorphisms preserve the selected Gram, act faithfully on its rank-three
quotient, and act by signed integral automorphisms on the dense antipodal
record module.  Consequently their isometries extend uniquely to the metric
completion.  The three declared tower maps lie in the same action and satisfy
the exact direct-versus-composite cocycle.

The tower maps are permutations of one twelve-port carrier.  They certify
finite recharting naturality, not scale refinement, atlas gluing, a cofinal
limit, global space, or a physical action.  The completion and its readback
remain conditional on the premises named by the parent receipt.
"""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = (
    ROOT / "data/repair_closure/port_gram_equivariant_action_receipt.json"
)
PARENT_RECEIPT = ROOT / "data/repair_closure/port_gram_completion_bridge_receipt.json"
CARRIER_MANIFEST = ROOT / "tests/fixtures/echosahedral_federation_reference.json"
PRODUCER_PATH = Path(__file__).resolve()
VERIFIER_PATH = (
    ROOT / "oph_fpe/dynamics/verify_port_gram_equivariant_action_independent.py"
)
TEST_PATH = ROOT / "tests/test_port_gram_equivariant_action.py"

SCHEMA = "oph.port-gram-equivariant-completion-action.v1"
STATUS = (
    "EXACT_FAITHFUL_PROPER_CARRIER_ACTION_AND_FINITE_RECHARTING_COCYCLE_"
    "ATTAINED__SOURCE_SELECTION_COFINAL_GLUING_AND_PHYSICAL_ACTION_OPEN"
)
PARENT_SCHEMA = "oph.port-gram-hausdorff-completion-bridge.v1"
PARENT_STATUS = (
    "EXACT_REPAIR_RESPONSE_GRAM_QUOTIENT_AND_3D_COMPLETION_ATTAINED__"
    "A1R_SIGNED_RECORD_MODULE_AND_A2R_POSITION_READBACK_PREMISES_OPEN"
)

Q5 = tuple[Fraction, Fraction]
ZERO: Q5 = (Fraction(0), Fraction(0))
NEGATIVE_ONE: Q5 = (Fraction(-1), Fraction(0))

CLAIM_BOUNDARY = (
    "The pinned oriented twelve-port carrier has exactly sixty proper incidence "
    "automorphisms. They preserve the repair-selected Gram, act faithfully on "
    "its rank-three real quotient, preserve the antipodal relations, and induce "
    "signed integral isometries of the dense cumulative-record module. Each "
    "therefore extends uniquely to an isometry of the conditional metric "
    "completion. Two explicit generators satisfy the (2,3,5) presentation and "
    "generate all sixty maps, identifying the action with the proper "
    "icosahedral group A5. The three manifest-declared tower maps are members of "
    "this action, intertwine the Gram and repair operator, and obey the exact "
    "r0-r1-r2 cocycle. Those maps are permutations of one finite carrier, so "
    "this is finite recharting naturality rather than scale refinement, cofinal "
    "gluing, or an atlas theorem. The parent completion remains conditional on "
    "signed-record and position-readback premises. No global or physical space, "
    "physical action, field, time, length scale, prediction, or comparison is "
    "promoted."
)


class EquivariantActionError(RuntimeError):
    """Raised when the conditional equivariant-action packet fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EquivariantActionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise EquivariantActionError(f"non-finite JSON constant is forbidden: {value}")


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EquivariantActionError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise EquivariantActionError(f"{path} is not a JSON object")
    return value


def _validated_parent() -> dict[str, Any]:
    parent = load_json_strict(PARENT_RECEIPT)
    payload = copy.deepcopy(parent)
    digest = payload.pop("receipt_sha256", None)
    attainment = parent.get("attainment")
    if (
        digest != _sha(payload)
        or parent.get("schema") != PARENT_SCHEMA
        or parent.get("status") != PARENT_STATUS
        or not isinstance(attainment, Mapping)
        or attainment.get("exact_lowest_repair_band_selects_port_gram") is not True
        or attainment.get("completion_is_three_dimensional_euclidean_vector_group")
        is not True
        or attainment.get("physical_three_space_promoted") is not False
        or attainment.get("comparison_permitted") is not False
    ):
        raise EquivariantActionError("parent completion contract drifted")
    return parent


def _fraction(text: str) -> Fraction:
    return Fraction(text)


def _parse_q5(text: str) -> Q5:
    suffix = "*sqrt5"
    if not text.endswith(suffix):
        raise EquivariantActionError(f"invalid Q(sqrt5) value: {text}")
    body = text[: -len(suffix)]
    split = body.find("+", 1)
    if split < 1:
        raise EquivariantActionError(f"invalid Q(sqrt5) value: {text}")
    return _fraction(body[:split]), _fraction(body[split + 1 :])


def _qtext(value: Q5) -> str:
    def render(item: Fraction) -> str:
        return str(item.numerator) if item.denominator == 1 else f"{item.numerator}/{item.denominator}"

    return f"{render(value[0])}+{render(value[1])}*sqrt5"


def _carrier_data(manifest: Mapping[str, Any]) -> tuple[list[str], list[list[int]], list[list[int]]]:
    carrier = manifest.get("carrier")
    if not isinstance(carrier, Mapping):
        raise EquivariantActionError("carrier manifest has no carrier object")
    ports = carrier.get("ports")
    edges = carrier.get("edges")
    faces = carrier.get("oriented_faces")
    if not (
        isinstance(ports, list)
        and len(ports) == 12
        and len(set(ports)) == 12
        and isinstance(edges, list)
        and isinstance(faces, list)
    ):
        raise EquivariantActionError("carrier manifest shape drifted")
    index = {str(port): position for position, port in enumerate(ports)}
    adjacency = [[0] * 12 for _ in range(12)]
    for row in edges:
        if not isinstance(row, list) or len(row) != 2:
            raise EquivariantActionError("invalid carrier edge")
        left, right = (index[str(value)] for value in row)
        adjacency[left][right] = adjacency[right][left] = 1
    indexed_faces = [[index[str(value)] for value in row] for row in faces]
    if any(sum(row) != 5 for row in adjacency) or len(indexed_faces) != 20:
        raise EquivariantActionError("carrier incidence drifted")
    return [str(value) for value in ports], adjacency, indexed_faces


def _automorphisms(adjacency: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    count = len(adjacency)
    assignment: list[int | None] = [None] * count
    used = [False] * count
    result: list[tuple[int, ...]] = []

    def consistent(vertex: int, image: int) -> bool:
        return all(
            adjacency[vertex][other]
            == adjacency[image][assignment[other]]  # type: ignore[index]
            for other in range(vertex)
        )

    def search(vertex: int) -> None:
        if vertex == count:
            result.append(tuple(assignment))  # type: ignore[arg-type]
            return
        for image in range(count):
            if used[image] or not consistent(vertex, image):
                continue
            assignment[vertex] = image
            used[image] = True
            search(vertex + 1)
            used[image] = False
            assignment[vertex] = None

    search(0)
    return sorted(result)


def _face_set(faces: Sequence[Sequence[int]]) -> frozenset[tuple[int, int, int]]:
    return frozenset(
        min((a, b, c), (b, c, a), (c, a, b)) for a, b, c in faces
    )


def _proper_automorphisms(
    automorphisms: Sequence[tuple[int, ...]], faces: Sequence[Sequence[int]]
) -> list[tuple[int, ...]]:
    oriented = _face_set(faces)
    return [
        permutation
        for permutation in automorphisms
        if _face_set(
            [
                [permutation[a], permutation[b], permutation[c]]
                for a, b, c in faces
            ]
        )
        == oriented
    ]


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    """Return left after right."""

    return tuple(left[right[index]] for index in range(len(left)))


def _inverse(permutation: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(permutation)
    for source, target in enumerate(permutation):
        result[target] = source
    return tuple(result)


def _order(permutation: tuple[int, ...]) -> int:
    identity = tuple(range(len(permutation)))
    current = identity
    for exponent in range(1, 61):
        current = _compose(permutation, current)
        if current == identity:
            return exponent
    raise EquivariantActionError("proper carrier permutation order exceeds 60")


def _generated(generators: Sequence[tuple[int, ...]]) -> set[tuple[int, ...]]:
    identity = tuple(range(len(generators[0])))
    result = {identity}
    frontier = [identity]
    moves = tuple(generators) + tuple(_inverse(value) for value in generators)
    while frontier:
        current = frontier.pop()
        for move in moves:
            candidate = _compose(move, current)
            if candidate not in result:
                result.add(candidate)
                frontier.append(candidate)
    return result


def _presentation_generators(
    proper: Sequence[tuple[int, ...]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    order_two = [value for value in proper if _order(value) == 2]
    order_three = [value for value in proper if _order(value) == 3]
    for left in order_two:
        for right in order_three:
            if _order(_compose(left, right)) == 5 and len(_generated((left, right))) == 60:
                return left, right
    raise EquivariantActionError("no generating (2,3,5) pair found")


def _gram_in_manifest_order(parent: Mapping[str, Any]) -> list[list[Q5]]:
    exact = parent["exact_repair_selected_gram"]
    source_gram = [[_parse_q5(value) for value in row] for row in exact["full_gram_qsqrt5"]]
    relabel = exact["independent_repair_incidence"]["fixture_to_source_port_map"]
    if sorted(relabel) != list(range(12)):
        raise EquivariantActionError("fixture/source relabeling is not a permutation")
    return [[source_gram[relabel[i]][relabel[j]] for j in range(12)] for i in range(12)]


def _preserves_matrix(permutation: tuple[int, ...], matrix: Sequence[Sequence[Any]]) -> bool:
    return all(
        matrix[permutation[i]][permutation[j]] == matrix[i][j]
        for i in range(len(permutation))
        for j in range(len(permutation))
    )


def _commutes_with_matrix(permutation: tuple[int, ...], matrix: Sequence[Sequence[Any]]) -> bool:
    # P M = M P iff M[p(i),p(j)] = M[i,j] for a symmetric M.
    return _preserves_matrix(permutation, matrix)


def _antipodes_from_gram(gram: Sequence[Sequence[Q5]]) -> tuple[int, ...]:
    result = []
    for row in gram:
        candidates = [index for index, value in enumerate(row) if value == NEGATIVE_ONE]
        if len(candidates) != 1:
            raise EquivariantActionError("Gram does not expose one antipode per port")
        result.append(candidates[0])
    antipodes = tuple(result)
    if any(antipodes[antipodes[i]] != i or antipodes[i] == i for i in range(12)):
        raise EquivariantActionError("invalid antipodal involution")
    return antipodes


def _signed_basis(antipodes: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(i for i in range(12) if i < antipodes[i])


def _signed_action(
    permutation: tuple[int, ...], antipodes: tuple[int, ...], basis: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    basis_index = {port: index for index, port in enumerate(basis)}
    matrix = [[0] * 6 for _ in range(6)]
    for column, port in enumerate(basis):
        image = permutation[port]
        if image in basis_index:
            matrix[basis_index[image]][column] = 1
        else:
            representative = antipodes[image]
            if representative not in basis_index:
                raise EquivariantActionError("rotation does not preserve antipodal pairs")
            matrix[basis_index[representative]][column] = -1
    return tuple(tuple(row) for row in matrix)


def _matrix_multiply(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(6)) for j in range(6))
        for i in range(6)
    )


def _determinant_integer(matrix: Sequence[Sequence[int]]) -> int:
    work = [[Fraction(value) for value in row] for row in matrix]
    sign = 1
    determinant = Fraction(1)
    for column in range(len(work)):
        pivot = next((row for row in range(column, len(work)) if work[row][column]), None)
        if pivot is None:
            return 0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            sign *= -1
        value = work[column][column]
        determinant *= value
        for row in range(column + 1, len(work)):
            factor = work[row][column] / value
            for index in range(column, len(work)):
                work[row][index] -= factor * work[column][index]
    result = sign * determinant
    if result.denominator != 1:
        raise EquivariantActionError("integral action has nonintegral determinant")
    return result.numerator


def _refinement_maps(manifest: Mapping[str, Any], ports: Sequence[str]) -> dict[tuple[str, str], tuple[int, ...]]:
    tower = manifest.get("refinement_tower")
    if not isinstance(tower, Mapping) or tower.get("levels") != ["r0", "r1", "r2"]:
        raise EquivariantActionError("declared refinement tower drifted")
    index = {port: position for position, port in enumerate(ports)}
    result: dict[tuple[str, str], tuple[int, ...]] = {}
    for row in tower.get("maps", []):
        if not isinstance(row, Mapping):
            raise EquivariantActionError("invalid tower map")
        key = str(row.get("source")), str(row.get("target"))
        values = row.get("port_map")
        if not isinstance(values, list):
            raise EquivariantActionError("tower port map absent")
        permutation = tuple(index[str(value)] for value in values)
        if sorted(permutation) != list(range(12)) or key in result:
            raise EquivariantActionError("invalid tower permutation")
        result[key] = permutation
    expected = {("r0", "r1"), ("r1", "r2"), ("r0", "r2")}
    if set(result) != expected:
        raise EquivariantActionError("tower map family drifted")
    return result


def produce_receipt() -> dict[str, Any]:
    parent = _validated_parent()
    manifest = load_json_strict(CARRIER_MANIFEST)
    ports, adjacency, faces = _carrier_data(manifest)
    all_automorphisms = _automorphisms(adjacency)
    proper = _proper_automorphisms(all_automorphisms, faces)
    if len(all_automorphisms) != 120 or len(proper) != 60:
        raise EquivariantActionError("carrier automorphism census drifted")

    gram = _gram_in_manifest_order(parent)
    if not all(_preserves_matrix(permutation, gram) for permutation in proper):
        raise EquivariantActionError("proper action does not preserve selected Gram")
    if not all(_commutes_with_matrix(permutation, adjacency) for permutation in proper):
        raise EquivariantActionError("proper action does not commute with incidence")

    identity = tuple(range(12))
    gram_kernel = [
        permutation
        for permutation in proper
        if all(gram[permutation[i]][j] == gram[i][j] for i in range(12) for j in range(12))
    ]
    if gram_kernel != [identity]:
        raise EquivariantActionError("completion action is not faithful")

    antipodes = _antipodes_from_gram(gram)
    basis = _signed_basis(antipodes)
    signed_actions = {permutation: _signed_action(permutation, antipodes, basis) for permutation in proper}
    if len(set(signed_actions.values())) != 60:
        raise EquivariantActionError("signed-module action is not faithful")
    determinants = sorted({_determinant_integer(matrix) for matrix in signed_actions.values()})
    if determinants != [1]:
        raise EquivariantActionError("unexpected signed-module determinants")
    for left in proper:
        for right in proper:
            composite = _compose(left, right)
            if composite not in signed_actions or _matrix_multiply(
                signed_actions[left], signed_actions[right]
            ) != signed_actions[composite]:
                raise EquivariantActionError("signed action composition failed")

    generator_two, generator_three = _presentation_generators(proper)
    order_histogram: dict[str, int] = {}
    for permutation in proper:
        key = str(_order(permutation))
        order_histogram[key] = order_histogram.get(key, 0) + 1
    if order_histogram != {"1": 1, "2": 15, "3": 20, "5": 24}:
        raise EquivariantActionError("proper action order histogram drifted")

    tower = _refinement_maps(manifest, ports)
    direct = tower[("r0", "r2")]
    composed = _compose(tower[("r1", "r2")], tower[("r0", "r1")])
    if direct != composed:
        raise EquivariantActionError("declared tower cocycle fails")
    rows = []
    for key in sorted(tower):
        permutation = tower[key]
        rows.append(
            {
                "source": key[0],
                "target": key[1],
                "port_permutation": list(permutation),
                "proper_carrier_action_member": permutation in proper,
                "selected_Gram_intertwined_exactly": _preserves_matrix(permutation, gram),
                "repair_incidence_intertwined_exactly": _commutes_with_matrix(
                    permutation, adjacency
                ),
                "antipodal_signed_module_action": [
                    list(row) for row in signed_actions[permutation]
                ],
            }
        )
    if not all(
        row["proper_carrier_action_member"]
        and row["selected_Gram_intertwined_exactly"]
        and row["repair_incidence_intertwined_exactly"]
        for row in rows
    ):
        raise EquivariantActionError("tower naturality failed")

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "issues": [655, 663],
        "target_data_read": False,
        "comparison_data_read": False,
        "parent_pins": {
            "repair_selected_completion": {
                **_raw_pin(PARENT_RECEIPT),
                "schema": parent["schema"],
                "status": parent["status"],
                "receipt_sha256": parent["receipt_sha256"],
                "completion_conditional": True,
                "physical_three_space_promoted": False,
            },
            "oriented_carrier_and_declared_tower": {
                **_raw_pin(CARRIER_MANIFEST),
                "schema": manifest.get("schema"),
                "levels": manifest["refinement_tower"]["levels"],
            },
        },
        "exact_proper_carrier_action": {
            "full_incidence_automorphism_count": len(all_automorphisms),
            "oriented_proper_automorphism_count": len(proper),
            "proper_action_permutations_sha256": _sha([list(value) for value in proper]),
            "element_order_histogram": order_histogram,
            "presentation": "<s,t | s^2=t^3=(s*t)^5=1>",
            "order_two_generator": list(generator_two),
            "order_three_generator": list(generator_three),
            "product_order": _order(_compose(generator_two, generator_three)),
            "generated_subgroup_order": len(_generated((generator_two, generator_three))),
            "abstract_group_identification": "A5 (proper icosahedral rotations)",
            "orientation_field_is_load_bearing": True,
            "removing_orientation_retains_full_order_120_group": True,
        },
        "exact_completion_action": {
            "selected_Gram_in_manifest_order_qsqrt5": [
                [_qtext(value) for value in row] for row in gram
            ],
            "all_proper_maps_preserve_selected_Gram": True,
            "all_proper_maps_commute_with_repair_incidence": True,
            "quotient_action_kernel_size": len(gram_kernel),
            "quotient_action_faithful": True,
            "antipodal_involution": list(antipodes),
            "signed_module_basis_ports": list(basis),
            "all_proper_maps_preserve_antipodal_relations": True,
            "signed_integral_action_count": len(set(signed_actions.values())),
            "signed_integral_action_faithful": True,
            "signed_integral_determinant_values": determinants,
            "signed_action_composition_exact": True,
            "dense_module_isometries_extend_uniquely_to_metric_completion": True,
            "completion_action_faithful": True,
            "extension_is_conditional_on_parent_completion_premises": True,
            "source_native_physical_action_promoted": False,
        },
        "declared_finite_tower_cocycle": {
            "map_rows": rows,
            "direct_r0_r2_equals_r1_r2_after_r0_r1": True,
            "signed_module_cocycle_exact": _matrix_multiply(
                signed_actions[tower[("r1", "r2")]],
                signed_actions[tower[("r0", "r1")]],
            )
            == signed_actions[direct],
            "completion_isometry_cocycle_exact": True,
            "finite_recharting_naturality_attained": True,
            "maps_add_new_carrier_degrees_of_freedom": False,
            "scale_refinement_semigroup_proved": False,
            "cofinal_refinement_family_proved": False,
            "overlap_atlas_gluing_proved": False,
            "global_carrier_gluing_proved": False,
        },
        "attainment": {
            "conditional_faithful_A5_completion_action_certified": True,
            "finite_declared_tower_recharting_cocycle_certified": True,
            "parent_completion_premises_discharged": False,
            "canonical_signed_record_source_selected": False,
            "A2_operational_position_topology_selected": False,
            "scale_refinement_naturality_proved": False,
            "cofinal_overlap_refinement_gluing_proved": False,
            "global_space_promoted": False,
            "physical_action_promoted": False,
            "physical_prediction_promoted": False,
            "comparison_permitted": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "implementation_pins": [
            _raw_pin(PRODUCER_PATH),
            _raw_pin(VERIFIER_PATH),
            _raw_pin(TEST_PATH),
        ],
    }
    receipt["receipt_sha256"] = _sha(receipt)
    return receipt


def verify_receipt(receipt: Mapping[str, Any]) -> dict[str, bool]:
    expected = produce_receipt()
    if _canonical_bytes(receipt) != _canonical_bytes(expected):
        raise EquivariantActionError("receipt does not match exact replay")
    return {
        "receipt": True,
        "faithful_completion_action": True,
        "finite_recharting_cocycle": True,
        "cofinal_gluing": False,
        "physical_action": False,
        "comparison_permitted": False,
    }


def write_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    receipt = produce_receipt()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        verify_receipt(load_json_strict(args.verify))
        print("PORT_GRAM_EQUIVARIANT_ACTION_VALID")
        return
    write_receipt(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
