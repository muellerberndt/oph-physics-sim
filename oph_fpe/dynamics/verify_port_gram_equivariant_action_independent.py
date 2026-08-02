"""Independent verifier for the conditional port-Gram equivariant action.

The verifier does not import the producer.  It independently enumerates the
carrier automorphisms, reconstructs the oriented subgroup and selected Gram,
checks faithfulness on the quotient and signed module, and checks the declared
three-map cocycle.  Physical and cofinal-gluing promotions must remain false.
"""

from __future__ import annotations

import argparse
import ast
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
PRODUCER_PATH = ROOT / "oph_fpe/dynamics/port_gram_equivariant_action.py"
VERIFIER_PATH = Path(__file__).resolve()
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
TOP_LEVEL_KEYS = {
    "schema",
    "status",
    "issues",
    "target_data_read",
    "comparison_data_read",
    "parent_pins",
    "exact_proper_carrier_action",
    "exact_completion_action",
    "declared_finite_tower_cocycle",
    "attainment",
    "claim_boundary",
    "implementation_pins",
    "receipt_sha256",
}

Q5 = tuple[Fraction, Fraction]
NEGATIVE_ONE: Q5 = (Fraction(-1), Fraction(0))


class IndependentEquivariantActionError(RuntimeError):
    """Raised when an independent check fails."""


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentEquivariantActionError(message)


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


def _raw_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IndependentEquivariantActionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise IndependentEquivariantActionError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _load(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentEquivariantActionError(f"cannot load {path}: {error}") from error
    _fail(isinstance(result, dict), f"{path} is not an object")
    return result


def _parse_q5(text: str) -> Q5:
    suffix = "*sqrt5"
    _fail(isinstance(text, str) and text.endswith(suffix), "Q(sqrt5) encoding")
    body = text[: -len(suffix)]
    split = body.find("+", 1)
    _fail(split > 0, "Q(sqrt5) split")
    return Fraction(body[:split]), Fraction(body[split + 1 :])


def _carrier(manifest: Mapping[str, Any]) -> tuple[list[str], list[list[int]], list[list[int]]]:
    carrier = manifest.get("carrier")
    _fail(isinstance(carrier, Mapping), "carrier object")
    ports = carrier.get("ports")
    edges = carrier.get("edges")
    faces = carrier.get("oriented_faces")
    _fail(
        isinstance(ports, list)
        and len(ports) == 12
        and len(set(ports)) == 12
        and isinstance(edges, list)
        and isinstance(faces, list),
        "carrier shape",
    )
    index = {str(port): position for position, port in enumerate(ports)}
    adjacency = [[0] * 12 for _ in range(12)]
    for edge in edges:
        _fail(isinstance(edge, list) and len(edge) == 2, "edge shape")
        left, right = index[str(edge[0])], index[str(edge[1])]
        adjacency[left][right] = adjacency[right][left] = 1
    indexed_faces = [[index[str(value)] for value in row] for row in faces]
    _fail(all(sum(row) == 5 for row in adjacency), "five-regular carrier")
    _fail(len(indexed_faces) == 20, "face count")
    return [str(value) for value in ports], adjacency, indexed_faces


def _enumerate_graph_automorphisms(
    adjacency: Sequence[Sequence[int]],
) -> list[tuple[int, ...]]:
    count = len(adjacency)
    partial: list[int | None] = [None] * count
    available = set(range(count))
    output: list[tuple[int, ...]] = []

    def visit(vertex: int) -> None:
        if vertex == count:
            output.append(tuple(partial))  # type: ignore[arg-type]
            return
        for image in sorted(available):
            if any(
                adjacency[vertex][other]
                != adjacency[image][partial[other]]  # type: ignore[index]
                for other in range(vertex)
            ):
                continue
            partial[vertex] = image
            available.remove(image)
            visit(vertex + 1)
            available.add(image)
            partial[vertex] = None

    visit(0)
    return sorted(output)


def _oriented_faces(faces: Sequence[Sequence[int]]) -> frozenset[tuple[int, int, int]]:
    output = set()
    for row in faces:
        a, b, c = row
        output.add(min((a, b, c), (b, c, a), (c, a, b)))
    return frozenset(output)


def _proper(
    automorphisms: Sequence[tuple[int, ...]], faces: Sequence[Sequence[int]]
) -> list[tuple[int, ...]]:
    reference = _oriented_faces(faces)
    output = []
    for permutation in automorphisms:
        transported = [
            [permutation[a], permutation[b], permutation[c]] for a, b, c in faces
        ]
        if _oriented_faces(transported) == reference:
            output.append(permutation)
    return output


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[index]] for index in range(len(left)))


def _inverse(permutation: tuple[int, ...]) -> tuple[int, ...]:
    output = [0] * len(permutation)
    for source, target in enumerate(permutation):
        output[target] = source
    return tuple(output)


def _permutation_order(permutation: tuple[int, ...]) -> int:
    identity = tuple(range(len(permutation)))
    current = identity
    for exponent in range(1, 61):
        current = _compose(permutation, current)
        if current == identity:
            return exponent
    raise IndependentEquivariantActionError("permutation order bound")


def _closure(generators: Sequence[tuple[int, ...]]) -> set[tuple[int, ...]]:
    identity = tuple(range(len(generators[0])))
    known = {identity}
    frontier = [identity]
    moves = tuple(generators) + tuple(_inverse(value) for value in generators)
    while frontier:
        current = frontier.pop()
        for move in moves:
            candidate = _compose(move, current)
            if candidate not in known:
                known.add(candidate)
                frontier.append(candidate)
    return known


def _manifest_gram(parent: Mapping[str, Any]) -> list[list[Q5]]:
    exact = parent.get("exact_repair_selected_gram")
    _fail(isinstance(exact, Mapping), "parent exact Gram packet")
    serialized = exact.get("full_gram_qsqrt5")
    incidence = exact.get("independent_repair_incidence")
    _fail(isinstance(serialized, list) and isinstance(incidence, Mapping), "parent Gram")
    source = [[_parse_q5(value) for value in row] for row in serialized]
    relabel = incidence.get("fixture_to_source_port_map")
    _fail(isinstance(relabel, list) and sorted(relabel) == list(range(12)), "relabel")
    return [[source[relabel[i]][relabel[j]] for j in range(12)] for i in range(12)]


def _preserves(permutation: tuple[int, ...], matrix: Sequence[Sequence[Any]]) -> bool:
    return all(
        matrix[permutation[i]][permutation[j]] == matrix[i][j]
        for i in range(12)
        for j in range(12)
    )


def _antipodes(gram: Sequence[Sequence[Q5]]) -> tuple[int, ...]:
    output = []
    for row in gram:
        candidates = [index for index, value in enumerate(row) if value == NEGATIVE_ONE]
        _fail(len(candidates) == 1, "unique Gram antipode")
        output.append(candidates[0])
    result = tuple(output)
    _fail(all(result[result[i]] == i and result[i] != i for i in range(12)), "antipodes")
    return result


def _signed_action(
    permutation: tuple[int, ...], antipodes: tuple[int, ...], basis: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    lookup = {port: index for index, port in enumerate(basis)}
    output = [[0] * 6 for _ in range(6)]
    for column, port in enumerate(basis):
        image = permutation[port]
        sign = 1
        if image not in lookup:
            image = antipodes[image]
            sign = -1
        _fail(image in lookup, "antipodal action descent")
        output[lookup[image]][column] = sign
    return tuple(tuple(row) for row in output)


def _matrix_product(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(6)) for j in range(6))
        for i in range(6)
    )


def _signed_permutation_determinant(matrix: Sequence[Sequence[int]]) -> int:
    targets = []
    product = 1
    for column in range(6):
        entries = [row for row in range(6) if matrix[row][column] != 0]
        _fail(len(entries) == 1, "signed permutation column")
        target = entries[0]
        _fail(matrix[target][column] in (-1, 1), "signed permutation entry")
        targets.append(target)
        product *= matrix[target][column]
    _fail(sorted(targets) == list(range(6)), "signed permutation rows")
    inversions = sum(
        targets[left] > targets[right]
        for left in range(6)
        for right in range(left + 1, 6)
    )
    return product * (-1 if inversions % 2 else 1)


def _tower_maps(
    manifest: Mapping[str, Any], ports: Sequence[str]
) -> dict[tuple[str, str], tuple[int, ...]]:
    tower = manifest.get("refinement_tower")
    _fail(isinstance(tower, Mapping), "tower object")
    _fail(tower.get("levels") == ["r0", "r1", "r2"], "tower levels")
    lookup = {port: index for index, port in enumerate(ports)}
    output = {}
    for row in tower.get("maps", []):
        _fail(isinstance(row, Mapping), "tower row")
        values = row.get("port_map")
        _fail(isinstance(values, list), "tower map values")
        key = str(row.get("source")), str(row.get("target"))
        permutation = tuple(lookup[str(value)] for value in values)
        _fail(sorted(permutation) == list(range(12)), "tower permutation")
        output[key] = permutation
    _fail(
        set(output) == {("r0", "r1"), ("r1", "r2"), ("r0", "r2")},
        "tower map keys",
    )
    return output


def _check_implementation_pins(receipt: Mapping[str, Any]) -> None:
    pins = receipt.get("implementation_pins")
    _fail(isinstance(pins, list) and len(pins) == 3, "implementation pins")
    expected_paths = (PRODUCER_PATH, VERIFIER_PATH, TEST_PATH)
    for pin, path in zip(pins, expected_paths, strict=True):
        _fail(isinstance(pin, Mapping), "implementation pin object")
        _fail(pin.get("path") == path.relative_to(ROOT).as_posix(), "pin path")
        _fail(pin.get("bytes") == len(path.read_bytes()), "pin bytes")
        _fail(pin.get("sha256") == _raw_sha(path), "pin hash")


def _check_no_producer_import() -> None:
    tree = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    _fail(
        "oph_fpe.dynamics.port_gram_equivariant_action" not in imported,
        "independent verifier imports producer",
    )


def verify_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, bool]:
    receipt = _load(path)
    _fail(set(receipt) == TOP_LEVEL_KEYS, "top-level schema")
    payload = copy.deepcopy(receipt)
    digest = payload.pop("receipt_sha256", None)
    _fail(digest == _sha(payload), "receipt digest")
    _fail(receipt.get("schema") == SCHEMA, "receipt schema")
    _fail(receipt.get("status") == STATUS, "receipt status")
    _fail(receipt.get("issues") == [655, 663], "issue binding")
    _fail(receipt.get("target_data_read") is False, "target-data firewall")
    _fail(receipt.get("comparison_data_read") is False, "comparison-data firewall")

    parent = _load(PARENT_RECEIPT)
    parent_payload = copy.deepcopy(parent)
    parent_digest = parent_payload.pop("receipt_sha256", None)
    _fail(parent_digest == _sha(parent_payload), "parent digest")
    _fail(parent.get("schema") == PARENT_SCHEMA, "parent schema")
    _fail(parent.get("status") == PARENT_STATUS, "parent status")
    parent_pins = receipt.get("parent_pins")
    _fail(isinstance(parent_pins, Mapping), "parent pins")
    completion_pin = parent_pins.get("repair_selected_completion")
    carrier_pin = parent_pins.get("oriented_carrier_and_declared_tower")
    _fail(isinstance(completion_pin, Mapping), "completion pin")
    _fail(isinstance(carrier_pin, Mapping), "carrier pin")
    _fail(completion_pin.get("sha256") == _raw_sha(PARENT_RECEIPT), "parent raw pin")
    _fail(completion_pin.get("receipt_sha256") == parent_digest, "parent self pin")
    _fail(carrier_pin.get("sha256") == _raw_sha(CARRIER_MANIFEST), "carrier raw pin")

    manifest = _load(CARRIER_MANIFEST)
    ports, adjacency, faces = _carrier(manifest)
    automorphisms = _enumerate_graph_automorphisms(adjacency)
    proper = _proper(automorphisms, faces)
    _fail(len(automorphisms) == 120 and len(proper) == 60, "automorphism counts")

    action = receipt.get("exact_proper_carrier_action")
    _fail(isinstance(action, Mapping), "proper action packet")
    _fail(action.get("full_incidence_automorphism_count") == 120, "full action count")
    _fail(action.get("oriented_proper_automorphism_count") == 60, "proper count")
    _fail(
        action.get("proper_action_permutations_sha256")
        == _sha([list(value) for value in proper]),
        "proper action digest",
    )
    histogram: dict[str, int] = {}
    for value in proper:
        key = str(_permutation_order(value))
        histogram[key] = histogram.get(key, 0) + 1
    _fail(histogram == {"1": 1, "2": 15, "3": 20, "5": 24}, "order census")
    _fail(action.get("element_order_histogram") == histogram, "reported order census")
    generator_two = tuple(action.get("order_two_generator", []))
    generator_three = tuple(action.get("order_three_generator", []))
    _fail(generator_two in proper and generator_three in proper, "presentation generators")
    _fail(_permutation_order(generator_two) == 2, "order-two generator")
    _fail(_permutation_order(generator_three) == 3, "order-three generator")
    _fail(_permutation_order(_compose(generator_two, generator_three)) == 5, "product order")
    _fail(len(_closure((generator_two, generator_three))) == 60, "generated action")
    _fail(action.get("abstract_group_identification") == "A5 (proper icosahedral rotations)", "A5 identification")

    gram = _manifest_gram(parent)
    completion = receipt.get("exact_completion_action")
    _fail(isinstance(completion, Mapping), "completion action packet")
    serialized = [[_parse_q5(value) for value in row] for row in completion.get("selected_Gram_in_manifest_order_qsqrt5", [])]
    _fail(serialized == gram, "reported manifest Gram")
    _fail(all(_preserves(value, gram) for value in proper), "Gram invariance")
    _fail(all(_preserves(value, adjacency) for value in proper), "incidence invariance")
    identity = tuple(range(12))
    kernel = [
        value
        for value in proper
        if all(gram[value[i]][j] == gram[i][j] for i in range(12) for j in range(12))
    ]
    _fail(kernel == [identity], "quotient faithfulness")
    antipodes = _antipodes(gram)
    basis = tuple(index for index in range(12) if index < antipodes[index])
    signed = {value: _signed_action(value, antipodes, basis) for value in proper}
    _fail(len(set(signed.values())) == 60, "signed action faithfulness")
    determinants = sorted(
        {_signed_permutation_determinant(matrix) for matrix in signed.values()}
    )
    _fail(determinants == [1], "signed action determinants")
    for left in proper:
        for right in proper:
            _fail(
                _matrix_product(signed[left], signed[right])
                == signed[_compose(left, right)],
                "signed action homomorphism",
            )
    _fail(completion.get("quotient_action_kernel_size") == 1, "reported kernel")
    _fail(completion.get("quotient_action_faithful") is True, "reported faithfulness")
    _fail(completion.get("antipodal_involution") == list(antipodes), "reported antipodes")
    _fail(completion.get("signed_module_basis_ports") == list(basis), "reported signed basis")
    _fail(completion.get("signed_integral_action_count") == 60, "reported signed count")
    _fail(
        completion.get("signed_integral_determinant_values") == determinants,
        "reported signed determinants",
    )
    _fail(completion.get("signed_action_composition_exact") is True, "reported composition")
    _fail(
        completion.get("dense_module_isometries_extend_uniquely_to_metric_completion")
        is True,
        "completion extension",
    )
    _fail(completion.get("source_native_physical_action_promoted") is False, "physical action boundary")

    tower = _tower_maps(manifest, ports)
    direct = tower[("r0", "r2")]
    _fail(
        direct == _compose(tower[("r1", "r2")], tower[("r0", "r1")]),
        "tower cocycle",
    )
    _fail(all(value in proper for value in tower.values()), "tower action membership")
    _fail(all(_preserves(value, gram) for value in tower.values()), "tower Gram naturality")
    tower_packet = receipt.get("declared_finite_tower_cocycle")
    _fail(isinstance(tower_packet, Mapping), "tower packet")
    rows = tower_packet.get("map_rows")
    _fail(isinstance(rows, list) and len(rows) == 3, "tower rows")
    reported = {
        (str(row["source"]), str(row["target"])): tuple(row["port_permutation"])
        for row in rows
    }
    _fail(reported == tower, "reported tower maps")
    _fail(tower_packet.get("direct_r0_r2_equals_r1_r2_after_r0_r1") is True, "reported cocycle")
    _fail(tower_packet.get("finite_recharting_naturality_attained") is True, "finite naturality")
    for key in (
        "maps_add_new_carrier_degrees_of_freedom",
        "scale_refinement_semigroup_proved",
        "cofinal_refinement_family_proved",
        "overlap_atlas_gluing_proved",
        "global_carrier_gluing_proved",
    ):
        _fail(tower_packet.get(key) is False, f"tower boundary {key}")

    attainment = receipt.get("attainment")
    _fail(isinstance(attainment, Mapping), "attainment packet")
    _fail(attainment.get("conditional_faithful_A5_completion_action_certified") is True, "action attainment")
    _fail(attainment.get("finite_declared_tower_recharting_cocycle_certified") is True, "cocycle attainment")
    for key in (
        "parent_completion_premises_discharged",
        "canonical_signed_record_source_selected",
        "A2_operational_position_topology_selected",
        "scale_refinement_naturality_proved",
        "cofinal_overlap_refinement_gluing_proved",
        "global_space_promoted",
        "physical_action_promoted",
        "physical_prediction_promoted",
        "comparison_permitted",
    ):
        _fail(attainment.get(key) is False, f"attainment boundary {key}")

    claim_boundary = receipt.get("claim_boundary")
    _fail(isinstance(claim_boundary, str), "claim boundary text")
    for phrase in (
        "finite recharting naturality",
        "rather than scale refinement",
        "No global or physical space",
        "No global or physical space, physical action, field, time, length scale, prediction, or comparison is promoted.",
    ):
        _fail(phrase in claim_boundary, f"claim boundary phrase: {phrase}")

    _check_implementation_pins(receipt)
    _check_no_producer_import()
    return {
        "receipt": True,
        "producer_imported": False,
        "automorphisms_independently_enumerated": True,
        "faithful_completion_action": True,
        "finite_recharting_cocycle": True,
        "cofinal_gluing": False,
        "physical_action": False,
        "comparison_permitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    verify_receipt(args.receipt)
    print("PORT_GRAM_EQUIVARIANT_ACTION_INDEPENDENT_VALID")


if __name__ == "__main__":
    main()
