from __future__ import annotations

from itertools import product
from typing import Any, Callable, Iterable, TypeAlias

from oph_fpe.claims import DEMO, with_claim_metadata

Bit: TypeAlias = int
Row: TypeAlias = tuple[Bit, Bit, Bit]
Record: TypeAlias = tuple[Row, Row]
BoundaryFn: TypeAlias = Callable[[Record], tuple[Bit, ...]]

RULE90_LEAN_CONSENSUS_FIXTURE_RECEIPT = "RULE90_LEAN_CONSENSUS_FIXTURE_RECEIPT"


def rule90t(seed: Row) -> Row:
    """Width-3 Rule 90 image with zero boundary: (a,b,c) -> (b,a xor c,b)."""

    a, b, c = _row(seed)
    return (b, a ^ c, b)


def rule90_obs_map(record: Record) -> tuple[Row, Row]:
    """Declared observable overlap data: (Rule90(seed), bottom row)."""

    seed, bottom = _record(record)
    return rule90t(seed), bottom


def rule90_phi(record: Record) -> int:
    """Unit-weight discrete mismatch potential for the one Rule-90 edge."""

    seed, bottom = _record(record)
    return int(rule90t(seed) != bottom)


def rule90_consistent(record: Record) -> bool:
    return rule90_phi(record) == 0


def rule90_gauge_equiv(left: Record, right: Record) -> bool:
    return rule90_obs_map(left) == rule90_obs_map(right)


def rule90_boundary_good(record: Record) -> tuple[Bit, Bit]:
    """Information-set boundary: bottom-row cells {0,1}."""

    _seed, bottom = _record(record)
    return bottom[0], bottom[1]


def rule90_boundary_bad(record: Record) -> tuple[Bit, Bit]:
    """Deficient boundary: bottom-row cells {0,2}; misses the middle bit."""

    _seed, bottom = _record(record)
    return bottom[0], bottom[2]


def rule90_records() -> list[Record]:
    rows = list(_rows())
    return [(seed, bottom) for seed in rows for bottom in rows]


def rule90_consistent_records() -> list[Record]:
    return [record for record in rule90_records() if rule90_consistent(record)]


def rule90_hfib_holds(boundary: BoundaryFn) -> bool:
    """Check Hfib exhaustively on the finite consistent Rule-90 carrier."""

    consistent = rule90_consistent_records()
    for left in consistent:
        for right in consistent:
            if boundary(left) == boundary(right) and not rule90_gauge_equiv(left, right):
                return False
    return True


def rule90_hfib_failure_witness(boundary: BoundaryFn) -> dict[str, Any] | None:
    consistent = rule90_consistent_records()
    for left in consistent:
        for right in consistent:
            if boundary(left) == boundary(right) and not rule90_gauge_equiv(left, right):
                return {
                    "left": _record_json(left),
                    "right": _record_json(right),
                    "boundary": list(boundary(left)),
                    "leftObsMap": _obs_json(rule90_obs_map(left)),
                    "rightObsMap": _obs_json(rule90_obs_map(right)),
                }
    return None


def rule90_gauge_nontrivial_witness() -> dict[str, Any] | None:
    consistent = rule90_consistent_records()
    for left in consistent:
        for right in consistent:
            left_seed, _ = left
            right_seed, _ = right
            if left_seed != right_seed and rule90_gauge_equiv(left, right):
                return {
                    "left": _record_json(left),
                    "right": _record_json(right),
                    "leftSeedDiffersFromRightSeed": True,
                    "obsMap": _obs_json(rule90_obs_map(left)),
                }
    return None


def rule90_no_frustration_free_local_repair_witness() -> dict[str, Any]:
    """Executable form of the Lean no-go proof for local H1/H2/H3 repair.

    This does not enumerate impossible repair functions. It checks the finite
    obstruction used by the Lean proof: bottom row (0,0,1) is outside the
    Rule-90 image for every seed. If H2 forces the seed-site move, H1 leaves
    that bottom row fixed, and H3 demands a Rule-90 preimage, contradiction.
    """

    impossible_bottom: Row = (0, 0, 1)
    preimages = [seed for seed in _rows() if rule90t(seed) == impossible_bottom]
    witness_record: Record = ((0, 0, 0), impossible_bottom)
    edge_broken_for_every_seed = all(rule90t(seed) != impossible_bottom for seed in _rows())
    h2_forces_seed_patch_move = edge_broken_for_every_seed
    h1_pins_bottom_row = True
    h3_requires_rule90_preimage = True
    receipt = bool(
        edge_broken_for_every_seed
        and h2_forces_seed_patch_move
        and h1_pins_bottom_row
        and h3_requires_rule90_preimage
        and not preimages
    )
    return {
        "receipt": receipt,
        "witnessRecord": _record_json(witness_record),
        "outOfImageBottom": list(impossible_bottom),
        "rule90PreimageCount": len(preimages),
        "rule90Preimages": [list(row) for row in preimages],
        "edgeBrokenForEverySeed": edge_broken_for_every_seed,
        "h2ForcesSeedPatchMove": h2_forces_seed_patch_move,
        "h1PinsBottomRow": h1_pins_bottom_row,
        "h3RequiresRule90Preimage": h3_requires_rule90_preimage,
        "reason": (
            "A bottom row with unequal outer cells cannot be a Rule-90 image; "
            "local H1/H2/H3 repair would have to both preserve it and make it consistent."
        ),
    }


def rule90_lean_consensus_fixture_report() -> dict[str, Any]:
    records = rule90_records()
    consistent = rule90_consistent_records()
    phi_consistency_violations = [
        _record_json(record)
        for record in records
        if (rule90_phi(record) == 0) != (rule90t(record[0]) == record[1])
    ]
    good_hfib = rule90_hfib_holds(rule90_boundary_good)
    bad_witness = rule90_hfib_failure_witness(rule90_boundary_bad)
    gauge_witness = rule90_gauge_nontrivial_witness()
    no_go = rule90_no_frustration_free_local_repair_witness()
    blockers: list[str] = []
    if phi_consistency_violations:
        blockers.append("phi_zero_iff_edge_consistency_failed")
    if not good_hfib:
        blockers.append("rule90_good_boundary_hfib_failed")
    if bad_witness is None:
        blockers.append("rule90_bad_boundary_failure_witness_missing")
    if gauge_witness is None:
        blockers.append("rule90_nontrivial_gauge_witness_missing")
    if not no_go["receipt"]:
        blockers.append("rule90_no_frustration_free_local_repair_witness_failed")
    receipt = not blockers
    report = {
        "mode": "rule90_lean_consensus_fixture_v1",
        RULE90_LEAN_CONSENSUS_FIXTURE_RECEIPT: receipt,
        "receipt": receipt,
        "recordCount": len(records),
        "consistentRecordCount": len(consistent),
        "phiZeroIffEdgeConsistencyReceipt": not phi_consistency_violations,
        "phiConsistencyViolationCount": len(phi_consistency_violations),
        "rule90GoodBoundaryHfibReceipt": good_hfib,
        "rule90BadBoundaryFailureReceipt": bad_witness is not None,
        "rule90GaugeNontrivialReceipt": gauge_witness is not None,
        "rule90NoFrustrationFreeLocalRepairReceipt": bool(no_go["receipt"]),
        "badBoundaryWitness": bad_witness,
        "gaugeNontrivialWitness": gauge_witness,
        "noFrustrationFreeLocalRepairWitness": no_go,
        "blockers": blockers,
        "leanSource": {
            "module": "ObserverPatchHolography.Rule90",
            "theorems": [
                "rule90_Hfib_good",
                "rule90_Hfib_bad_fails",
                "rule90_gauge_nontrivial",
                "rule90_no_frustrationFree_repair",
            ],
            "claimBoundary": (
                "Simulator executable fixture mirrored from Lean proofs. It is a finite regression contract, "
                "not a runtime Lean proof and not a physical prediction."
            ),
        },
        "claim_boundary": (
            "Exact finite Rule-90 carrier contract for simulator regression. It mirrors Lean-positive and "
            "Lean-negative consensus facts so the simulator cannot silently treat boundary uniqueness, "
            "gauge quotienting, or local repair as generic assumptions."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=DEMO,
        receipt=RULE90_LEAN_CONSENSUS_FIXTURE_RECEIPT,
        physical_claim=False,
        observable_id="rule90_consensus_boundary_gauge_fixture",
        fit_objective="lean_mirrored_consensus_contract_regression",
    )


def _rows() -> Iterable[Row]:
    for values in product((0, 1), repeat=3):
        yield _row(values)


def _row(value: Iterable[Any]) -> Row:
    row = tuple(int(bit) for bit in value)
    if len(row) != 3 or any(bit not in (0, 1) for bit in row):
        raise ValueError(f"Rule-90 rows must be three bits, got {value!r}")
    return row  # type: ignore[return-value]


def _record(value: Record) -> Record:
    seed, bottom = value
    return _row(seed), _row(bottom)


def _record_json(record: Record) -> dict[str, list[int]]:
    seed, bottom = _record(record)
    return {"seed": list(seed), "bottom": list(bottom)}


def _obs_json(obs: tuple[Row, Row]) -> dict[str, list[int]]:
    return {"srcProjection": list(obs[0]), "tgtProjection": list(obs[1])}
