"""Extract the B13 source-produced convex preparation family from the B12 run.

The conditional-resampling realization receipt of the B12 preregistered run
pins a realized joint frequency table over (record class, companion class).
That table carries a convex preparation family produced by the source run
itself:

* the fibre-conditional laws ``pi(. | r)`` of the 32 record fibres are 32
  realized states on the companion alphabet, each an exact rational vector
  (integer cell counts over the integer fibre mass);
* convex mixtures of fibre laws weighted by any record distribution are
  exactly the companion marginals of record-rebalanced versions of the same
  table, so the record-marginal mixing operation is the declared convex
  structure and it is realized by the run's own record statistics.

The script recomputes the joint table with the same binning producer that
built the receipt
(``oph_fpe.dynamics.conditional_resampling.realization_inputs_from_freezeout``
on ``freezeout_fields.npz``), fails hard when the recomputed class structure
disagrees with the receipt's pinned reference, and emits:

* the 32 exact fibre-conditional laws (integer counts and reduced fractions);
* the realized record marginal (exact);
* three declared mixtures: the realized record marginal and two rebalanced
  record distributions with exact weights, each with its exact mixed
  companion law; every mixed law is cross-checked cell by cell against the
  companion marginal of the correspondingly reweighted table, and the
  realized-marginal mixture is additionally checked against the realized
  companion marginal of the raw table;
* a Lean subfamily block: the first four record classes in label order,
  their laws, exact pairwise distinctness witnesses (a distinguishing
  companion coordinate per pair), the realized record marginal restricted to
  the chosen fibres and renormalized, one declared rebalanced simplex point,
  and the exact mixed laws at both points, for a Lean witness to mirror as
  literals;
* provenance pins (run id, git commit, seed, receipt and input sha256s,
  binning producer, declaration rules).

Every law and weight in the payload is exact rational data; no float enters
any emitted value.  The payload promotes no physical claim.

Usage (from the repository root, venv python):

    .venv/bin/python scripts/extract_b13_convex_family.py

Options allow a different run directory or output path; defaults reproduce
``docs/B13_CONVEX_FAMILY_PAYLOAD.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oph_fpe.dynamics.conditional_resampling import (  # noqa: E402
    realization_inputs_from_freezeout,
)

PAYLOAD_SCHEMA = "oph.sim.b13_convex_family_payload.v1"
MAX_PAYLOAD_BYTES = 200_000

DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "b12_prereg_16k_20260806"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "B13_CONVEX_FAMILY_PAYLOAD.json"
RECEIPT_NAME = "conditional_resampling_realization_receipt.json"
FREEZEOUT_NAME = "freezeout_fields.npz"

LEAN_SUBFAMILY_SIZE = 4
LEAN_REBALANCED_POINT = (
    Fraction(1, 2),
    Fraction(1, 4),
    Fraction(1, 8),
    Fraction(1, 8),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frac(f: Fraction) -> list[int]:
    """A reduced fraction as an exact [numerator, denominator] pair."""

    return [f.numerator, f.denominator]


def _law_fractions(counts: list[int], mass: int) -> list[list[int]]:
    return [_frac(Fraction(c, mass)) for c in counts]


def _mixed_law(
    weights: dict[int, Fraction],
    laws: dict[int, list[Fraction]],
    companion_count: int,
) -> list[Fraction]:
    """The exact convex mixture ``sum_r w_r * pi(. | r)``."""

    mixed = [Fraction(0) for _ in range(companion_count)]
    for r, w in weights.items():
        for j in range(companion_count):
            mixed[j] += w * laws[r][j]
    return mixed


def _reweighted_marginal(
    weights: dict[int, Fraction],
    joint: dict[tuple[int, int], int],
    fibre_mass: dict[int, int],
    companions: list[int],
) -> list[Fraction]:
    """The companion marginal of the record-rebalanced table.

    Each cell (r, c) of the reweighted table carries exact mass
    ``w_r * n_{rc} / n_r``; the marginal sums over r.  This is an
    independent route to the same quantity as ``_mixed_law`` and is used
    as the exactness cross-check.
    """

    marginal = [Fraction(0) for _ in companions]
    for (r, c), count in joint.items():
        w = weights.get(r, Fraction(0))
        if w:
            marginal[companions.index(c)] += w * Fraction(count, fibre_mass[r])
    return marginal


def build_payload(run_dir: Path) -> dict:
    receipt_path = run_dir / RECEIPT_NAME
    freezeout_path = run_dir / FREEZEOUT_NAME
    for path in (receipt_path, freezeout_path):
        if not path.exists():
            raise FileNotFoundError(path)

    receipt = json.loads(receipt_path.read_text())
    inputs = realization_inputs_from_freezeout(freezeout_path)
    record = inputs.record_classes
    companion = inputs.companion_classes

    joint = Counter(zip(record, companion))
    records = sorted(set(record))
    companions = sorted(set(companion))
    fibre_count = len(records)
    state_count = len(joint)
    total_mass = sum(joint.values())

    pinned = receipt["pinned_reference"]
    checks = {
        "fiber_count": (fibre_count, pinned["fiber_count"]),
        "state_count": (state_count, pinned["state_count"]),
        "total_mass_count": (total_mass, pinned["total_mass_count"]),
        "record_class_count": (
            fibre_count,
            receipt["protected_record"]["class_count"],
        ),
    }
    mismatches = {
        name: values for name, values in checks.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(
            f"recomputed class structure disagrees with receipt: {mismatches}"
        )

    # Fibre masses, fibre-conditional laws, and the realized record marginal.
    fibre_mass = {
        r: sum(joint[(r, c)] for c in companions if (r, c) in joint)
        for r in records
    }
    for r in records:
        if fibre_mass[r] == 0:
            raise ValueError(f"record fibre {r} carries zero mass")
    fibre_counts = {
        r: [joint.get((r, c), 0) for c in companions] for r in records
    }
    laws = {
        r: [Fraction(cnt, fibre_mass[r]) for cnt in fibre_counts[r]]
        for r in records
    }
    for r in records:
        if sum(laws[r]) != 1:
            raise ValueError(f"fibre law {r} does not sum to one")
    marginal_weights = {r: Fraction(fibre_mass[r], total_mass) for r in records}
    if sum(marginal_weights.values()) != 1:
        raise ValueError("realized record marginal does not sum to one")

    # Realized companion marginal of the raw table (for the mixture-1 check).
    companion_marginal = [
        Fraction(
            sum(joint.get((r, c), 0) for r in records), total_mass
        )
        for c in companions
    ]

    # The three declared mixtures.
    linear_weights = {
        r: Fraction(r + 1, sum(s + 1 for s in records)) for r in records
    }
    even_records = [r for r in records if r % 2 == 0]
    even_weights = {r: Fraction(1, len(even_records)) for r in even_records}
    mixture_specs = [
        (
            "realized_record_marginal",
            "the run's realized record marginal itself: w_r = n_r / N",
            marginal_weights,
        ),
        (
            "linear_index_rebalance",
            "weights proportional to the 1-based record-class index: "
            "w_r = (r + 1) / sum_{s}(s + 1)",
            linear_weights,
        ),
        (
            "even_fibre_restriction",
            "uniform weights on the even record classes, zero on the odd "
            "ones: w_r = 1 / |{even r}| for even r",
            even_weights,
        ),
    ]

    mixtures = []
    for name, rule, weights in mixture_specs:
        if sum(weights.values()) != 1:
            raise ValueError(f"mixture {name} weights do not sum to one")
        mixed = _mixed_law(weights, laws, len(companions))
        reweighted = _reweighted_marginal(weights, joint, fibre_mass, companions)
        if mixed != reweighted:
            raise ValueError(
                f"mixture {name}: convex mixture of fibre laws disagrees "
                "with the reweighted-table companion marginal"
            )
        if sum(mixed) != 1:
            raise ValueError(f"mixture {name} mixed law does not sum to one")
        entry = {
            "name": name,
            "declared_weight_rule": rule,
            "weights": [
                [r, _frac(weights[r])] for r in sorted(weights) if weights[r]
            ],
            "weight_sum_is_one": True,
            "mixed_companion_law": [_frac(f) for f in mixed],
            "mixed_law_sum_is_one": True,
            "equals_reweighted_table_companion_marginal": True,
        }
        if name == "realized_record_marginal":
            if mixed != companion_marginal:
                raise ValueError(
                    "realized-marginal mixture disagrees with the realized "
                    "companion marginal of the raw table"
                )
            entry["equals_realized_companion_marginal"] = True
        mixtures.append(entry)

    # Lean subfamily: the first LEAN_SUBFAMILY_SIZE record classes in label
    # order, with exact pairwise distinctness verified and witnessed.
    chosen = records[:LEAN_SUBFAMILY_SIZE]
    distinctness = []
    for a_pos, a in enumerate(chosen):
        for b in chosen[a_pos + 1 :]:
            coord = next(
                (
                    j
                    for j in range(len(companions))
                    if laws[a][j] != laws[b][j]
                ),
                None,
            )
            if coord is None:
                raise ValueError(
                    f"fibre laws {a} and {b} coincide; the declared "
                    "subfamily selection rule fails on this run"
                )
            distinctness.append(
                {
                    "pair": [a, b],
                    "distinguishing_companion_index": coord,
                    "distinguishing_companion_label": companions[coord],
                    "values": [
                        _frac(laws[a][coord]),
                        _frac(laws[b][coord]),
                    ],
                }
            )

    restricted_total = sum(fibre_mass[r] for r in chosen)
    restricted_marginal = {
        r: Fraction(fibre_mass[r], restricted_total) for r in chosen
    }
    if len(LEAN_REBALANCED_POINT) != len(chosen):
        raise ValueError("rebalanced point size disagrees with subfamily size")
    rebalanced = dict(zip(chosen, LEAN_REBALANCED_POINT))
    if sum(rebalanced.values()) != 1:
        raise ValueError("declared rebalanced point does not sum to one")
    sub_points = [
        (
            "restricted_realized_marginal",
            "the realized record marginal restricted to the chosen fibres "
            "and renormalized: w_r = n_r / sum_{chosen} n_s",
            restricted_marginal,
        ),
        (
            "declared_rebalanced_point",
            "the declared literal simplex point (1/2, 1/4, 1/8, 1/8) on the "
            "chosen fibres in label order",
            rebalanced,
        ),
    ]
    sub_mixtures = []
    for name, rule, weights in sub_points:
        mixed = _mixed_law(weights, laws, len(companions))
        reweighted = _reweighted_marginal(weights, joint, fibre_mass, companions)
        if mixed != reweighted or sum(mixed) != 1:
            raise ValueError(f"subfamily point {name} fails the exact checks")
        sub_mixtures.append(
            {
                "name": name,
                "declared_weight_rule": rule,
                "weights": [[r, _frac(weights[r])] for r in chosen],
                "mixed_companion_law": [_frac(f) for f in mixed],
                "mixed_law_sum_is_one": True,
                "equals_reweighted_table_companion_marginal": True,
            }
        )

    seed = receipt["empirical_realization"]["seed"]
    git_commit_path = run_dir / "git_commit.txt"
    git_commit = (
        git_commit_path.read_text().strip() if git_commit_path.exists() else None
    )

    payload = {
        "schema": PAYLOAD_SCHEMA,
        "provenance": {
            "run_id": run_dir.name,
            "run_git_commit": git_commit,
            "seed": int(seed),
            "receipt_path": str(receipt_path.relative_to(REPO_ROOT)),
            "receipt_sha256": _sha256(receipt_path),
            "receipt_schema": receipt["schema"],
            "freezeout_path": str(freezeout_path.relative_to(REPO_ROOT)),
            "freezeout_sha256": _sha256(freezeout_path),
            "extraction": {
                "script": "scripts/extract_b13_convex_family.py",
                "binning_producer": (
                    "oph_fpe.dynamics.conditional_resampling."
                    "realization_inputs_from_freezeout"
                ),
                "record_field": inputs.record_label,
                "companion_field": inputs.companion_label,
                "subfamily_selection_rule": (
                    f"first {LEAN_SUBFAMILY_SIZE} record classes in label "
                    "order, with exact pairwise law distinctness required"
                ),
            },
        },
        "class_structure": {
            "record_class_count": fibre_count,
            "companion_class_labels_realized": companions,
            "realized_companion_class_count": len(companions),
            "joint_state_count": state_count,
            "total_patch_count": total_mass,
            "fibre_mass_uniform": len(set(fibre_mass.values())) == 1,
            "fibre_mass": [[r, fibre_mass[r]] for r in records],
            "matches_receipt_pinned_reference": True,
        },
        "convex_family_statement": (
            "The fibre-conditional laws pi(. | r) below are the realized "
            "states of the family; the declared convex mixing operation is "
            "record-marginal rebalancing, and each emitted mixture is "
            "verified exactly to equal the companion marginal of the "
            "correspondingly reweighted joint table.  The realized record "
            "marginal is itself one of the declared mixtures, so the mixing "
            "operation is realized by the run's own record statistics."
        ),
        "fibre_conditional_laws": [
            {
                "record_class": r,
                "fibre_mass": fibre_mass[r],
                "counts": fibre_counts[r],
                "law": _law_fractions(fibre_counts[r], fibre_mass[r]),
                "law_sum_is_one": True,
            }
            for r in records
        ],
        "realized_record_marginal": [
            [r, fibre_mass[r], _frac(marginal_weights[r])] for r in records
        ],
        "realized_companion_marginal": [_frac(f) for f in companion_marginal],
        "mixtures": mixtures,
        "lean_subfamily": {
            "chosen_record_classes": chosen,
            "companion_alphabet": {
                "lean_index_to_realized_label": companions,
                "size": len(companions),
            },
            "laws": [
                {
                    "record_class": r,
                    "counts": fibre_counts[r],
                    "law": _law_fractions(fibre_counts[r], fibre_mass[r]),
                }
                for r in chosen
            ],
            "pairwise_distinctness": distinctness,
            "simplex_points": sub_mixtures,
        },
        "claim_boundary": (
            "Exact realized finite data from one committed run under the "
            "receipt's own binning.  The laws are realized conditional "
            "frequencies, the mixtures are declared exact reweightings of "
            "the run's record statistics, and every identity asserted above "
            "is checked in exact rational arithmetic before emission.  No "
            "physical claim, no probability claim beyond realized "
            "frequencies, and no canonical status for the declared "
            "rebalancings are asserted."
        ),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=DEFAULT_RUN_DIR,
        help="run directory holding the receipt and freezeout fields",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="payload output path",
    )
    args = parser.parse_args()

    payload = build_payload(args.run_dir.resolve())
    text = json.dumps(payload, indent=1)
    encoded = text.encode()
    if len(encoded) > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"payload is {len(encoded)} bytes; limit {MAX_PAYLOAD_BYTES}"
        )
    args.output.write_text(text + "\n")
    print(f"wrote {args.output} ({len(encoded)} bytes)")


if __name__ == "__main__":
    main()
