#!/usr/bin/env python3
"""B7 history-law packet: source-produce (reference, action, empirical law).

Reads the pinned transition-clock objects of runs/b12_prereg_16k_20260806 and
the committed mixing-chain payload docs/B12_MIXING_CHAIN_PAYLOAD.json, and
produces the history-law packet of issue B7 (#683) over the restricted
two-state recurrent class {18, 19} of that run:

* enumerates all eight length-3 paths (three states, two transitions) over
  the recurrent class and counts their realized windows with exact integer
  arithmetic: every maximal in-class segment of an observer transition
  history contributes each of its consecutive state triples as one window;
* the empirical path law tau_emp is the exact rational normalization of the
  integer window counts;
* the reference path law tau_chain is the stationary Markov path law of the
  committed chain: exact products pi(s0) P(s0,s1) P(s1,s2) of the committed
  rationals of the mixing-chain payload;
* the action S(path) is the number of state changes along the path, a
  source-intrinsic repair-move count needing no target data;
* the mean action under each law is exact rational;
* the relative entropies between the two laws carry outward-rounded interval
  enclosures (mpmath.iv at ENCLOSURE_PREC_BITS bits of working precision);
* the mean-action matching multiplier is reduced to exact algebra: with
  x = exp(-lambda), matching the tilted mean of tau_chain to the declared
  alternative mean (the empirical mean) is a quadratic in x with exact
  rational coefficients; the packet carries an exact-rational bisection
  enclosure of the positive root x_star of width at most X_STAR_WIDTH, the
  induced interval for lambda_star = -log x_star, interval enclosures of the
  tilted law at x_star, and a declared rational multiplier point x0 at which
  the tilted law is exactly rational.

The payload is written to docs/B7_HISTORY_PACKET_PAYLOAD.json.  The Lean
mirror is reverse-engineering-reality Lean/InformationProjection/
SourceHistoryPacket.lean; the block of literals it consumes is printed on
completion.

Claim boundary.  Representation-level production of the history-law inputs
from one preregistered bounded source run.  The run produces the reference
law, the action, and the empirical law; the multiplier is matched to the
declared empirical mean, not selected by any source mechanism.  The
multiplier selection principle from source data is the remaining gap of B7.
"""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import mpmath

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "runs" / "b12_prereg_16k_20260806"
B12_PAYLOAD_PATH = REPO_ROOT / "docs" / "B12_MIXING_CHAIN_PAYLOAD.json"
OUT_PATH = REPO_ROOT / "docs" / "B7_HISTORY_PACKET_PAYLOAD.json"

PINNED_INPUTS = (
    "finite_repair_transition_matrix_report.json",
    "observer_views.jsonl",
    "manifest.json",
    "seed_material.json",
    "git_commit.txt",
)

ENCLOSURE_PREC_BITS = 256
X_STAR_WIDTH = Fraction(1, 10**40)
DECLARED_X0_DENOM = 10**12
BRACKET_DENOM = 10**6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(obstruction: str) -> None:
    print(f"NEGATIVE RESULT: {obstruction}", file=sys.stderr)
    raise SystemExit(1)


def require(condition: bool, obstruction: str) -> None:
    if not condition:
        fail(obstruction)


def raw_mpf_as_fraction(raw) -> Fraction:
    """Exact rational value of a raw mpmath mpf tuple (binary float, so exact)."""
    sign, man, exp, _ = raw
    frac = Fraction(int(man), 1)
    if exp >= 0:
        frac *= Fraction(2) ** exp
    else:
        frac /= Fraction(2) ** (-exp)
    return -frac if sign else frac


def iv_log_fraction(value: Fraction):
    """Outward-rounded interval enclosure of log(value) for value > 0."""
    x = mpmath.iv.mpf(value.numerator) / mpmath.iv.mpf(value.denominator)
    result = mpmath.iv.log(x)
    raw_lo, raw_hi = result._mpi_
    return raw_mpf_as_fraction(raw_lo), raw_mpf_as_fraction(raw_hi)


def interval_str(lo: Fraction, hi: Fraction) -> dict:
    return {
        "lo_exact": str(lo),
        "hi_exact": str(hi),
        "lo_float": float(lo),
        "hi_float": float(hi),
        "width_exact": str(hi - lo),
        "width_float": float(hi - lo),
    }


def main() -> None:
    require(RUN_DIR.is_dir(), f"missing preregistered run directory {RUN_DIR}")
    for name in PINNED_INPUTS:
        require((RUN_DIR / name).is_file(), f"missing pinned input {name}")
    require(B12_PAYLOAD_PATH.is_file(), "missing committed mixing-chain payload")

    b12 = json.loads(B12_PAYLOAD_PATH.read_text())
    pinned_digests = b12["provenance"]["pinned_input_sha256"]
    for name in ("observer_views.jsonl", "finite_repair_transition_matrix_report.json"):
        require(
            sha256_file(RUN_DIR / name) == pinned_digests[name],
            f"{name} digest disagrees with the committed mixing-chain payload",
        )

    report = json.loads((RUN_DIR / "finite_repair_transition_matrix_report.json").read_text())
    manifest = json.loads((RUN_DIR / "manifest.json").read_text())
    seed_material = json.loads((RUN_DIR / "seed_material.json").read_text())
    git_commit = (RUN_DIR / "git_commit.txt").read_text().strip()

    fields = [str(f) for f in report["packet_fields"]]
    labels = [tuple((str(f), int(v)) for f, v in json.loads(s)) for s in report["state_labels"]]
    state_index = {label: i for i, label in enumerate(labels)}

    # Committed restricted chain.
    rrc = b12["restricted_recurrent_chain"]
    recurrent = [int(s) for s in rrc["global_state_indices"]]
    require(recurrent == [18, 19], "restricted recurrent class disagrees with the committed payload")
    local_of_global = {g: i for i, g in enumerate(recurrent)}
    committed_pair_counts = [[int(v) for v in row] for row in rrc["integer_counts"]]
    chain = [[Fraction(v) for v in row] for row in rrc["transition_matrix_exact"]]
    stationary = [Fraction(v) for v in rrc["stationary_law_exact"]]
    for i in range(2):
        row_sum = sum(committed_pair_counts[i])
        for j in range(2):
            require(
                chain[i][j] == Fraction(committed_pair_counts[i][j], row_sum),
                "committed exact chain disagrees with committed integer counts",
            )
    require(sum(stationary) == 1, "committed stationary law fails normalization")
    for j in range(2):
        require(
            sum(stationary[i] * chain[i][j] for i in range(2)) == stationary[j],
            "committed stationary law fails exact stationarity",
        )

    # Window pass over the observer transition histories.
    triple_counts = [0] * 8
    pair_recount = [[0, 0], [0, 0]]
    segment_count = 0
    observer_count = 0
    skipped = 0
    with (RUN_DIR / "observer_views.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            observer_count += 1
            view = json.loads(line)
            steps = (view.get("transition_history_descriptor") or {}).get("steps") or []
            if len(steps) < 2:
                skipped += 1
                continue
            encoded = []
            for step in steps:
                key = tuple((f, int(step.get(f, 0))) for f in fields)
                require(key in state_index, f"transition step visits an unknown state: {key}")
                encoded.append(state_index[key])
            # Maximal in-class segments; closure of the class was certified in B12.
            i = 0
            length = len(encoded)
            while i < length:
                if encoded[i] not in local_of_global:
                    i += 1
                    continue
                j = i
                while j < length and encoded[j] in local_of_global:
                    j += 1
                segment = [local_of_global[s] for s in encoded[i:j]]
                if len(segment) >= 2:
                    segment_count += 1
                for a, b in zip(segment, segment[1:]):
                    pair_recount[a][b] += 1
                for a, b, c in zip(segment, segment[1:], segment[2:]):
                    triple_counts[4 * a + 2 * b + c] += 1
                i = j

    require(observer_count == int(report["observer_count"]), "observer count disagrees with the report")
    require(skipped == int(report["skipped_observer_count"]), "skipped count disagrees with the report")
    require(
        pair_recount == committed_pair_counts,
        f"in-class pair recount {pair_recount} disagrees with the committed counts",
    )
    window_total = sum(triple_counts)
    require(window_total > 0, "no length-3 window lies inside the recurrent class")
    require(all(c > 0 for c in triple_counts), f"a length-3 path has zero windows: {triple_counts}")

    # The three literal families.
    def path_states(g: int) -> tuple[int, int, int]:
        return (g >> 2) & 1, (g >> 1) & 1, g & 1

    def action(g: int) -> int:
        s0, s1, s2 = path_states(g)
        return (0 if s0 == s1 else 1) + (0 if s1 == s2 else 1)

    tau_emp = [Fraction(triple_counts[g], window_total) for g in range(8)]
    tau_chain = []
    for g in range(8):
        s0, s1, s2 = path_states(g)
        tau_chain.append(stationary[s0] * chain[s0][s1] * chain[s1][s2])
    require(sum(tau_emp) == 1, "empirical path law fails normalization")
    require(sum(tau_chain) == 1, "chain path law fails normalization")
    require(all(p > 0 for p in tau_chain), "chain path law has a nonpositive entry")

    actions = [action(g) for g in range(8)]
    require(len(set(actions)) >= 2, "path action is constant")
    mean_emp = sum(tau_emp[g] * actions[g] for g in range(8))
    mean_chain = sum(tau_chain[g] * actions[g] for g in range(8))

    # Relative entropies with outward-rounded interval enclosures.
    mpmath.iv.prec = ENCLOSURE_PREC_BITS

    def kl_interval(p: list[Fraction], q: list[Fraction]) -> tuple[Fraction, Fraction]:
        lo = Fraction(0)
        hi = Fraction(0)
        for pg, qg in zip(p, q):
            log_lo, log_hi = iv_log_fraction(pg / qg)
            lo += pg * log_lo
            hi += pg * log_hi
        return lo, hi

    kl_emp_chain = kl_interval(tau_emp, tau_chain)
    kl_chain_emp = kl_interval(tau_chain, tau_emp)
    require(kl_emp_chain[0] > 0, "KL(tau_emp || tau_chain) enclosure fails strict positivity")

    # Mean-action matching quadratic in x = exp(-lambda).
    c = [Fraction(0)] * 3
    for g in range(8):
        c[actions[g]] += tau_chain[g]
    m = mean_emp
    quad_a = c[2] * (2 - m)
    quad_b = c[1] * (1 - m)
    quad_c = -c[0] * m
    require(quad_a > 0 and quad_c < 0, "matching quadratic loses its sign structure")

    def quad(x: Fraction) -> Fraction:
        return quad_a * x * x + quad_b * x + quad_c

    # F(1) = mean_chain - mean_emp exactly.
    require(quad(Fraction(1)) == mean_chain - mean_emp, "quadratic cross check fails at x = 1")

    lo, hi = Fraction(0), Fraction(1)
    if quad(hi) <= 0:
        while quad(hi) <= 0:
            hi *= 2
        lo = hi / 2
    while hi - lo > X_STAR_WIDTH:
        mid = (lo + hi) / 2
        if quad(mid) < 0:
            lo = mid
        else:
            hi = mid
    x_star_lo, x_star_hi = lo, hi
    require(quad(x_star_lo) < 0 < quad(x_star_hi), "bisection enclosure fails the sign test")

    # lambda_star = -log x_star: outward interval from the exact-rational enclosure.
    log_lo_lo, _ = iv_log_fraction(x_star_lo)
    _, log_hi_hi = iv_log_fraction(x_star_hi)
    lam_star_lo, lam_star_hi = -log_hi_hi, -log_lo_lo

    # Tilted law enclosures at x_star: Z is increasing in x > 0 (positive
    # coefficients), so outward bounds come from the endpoint evaluations.
    def z_of(x: Fraction) -> Fraction:
        return c[0] + c[1] * x + c[2] * x * x

    tilt_star_intervals = []
    for g in range(8):
        n = actions[g]
        t_lo = tau_chain[g] * x_star_lo**n / z_of(x_star_hi)
        t_hi = tau_chain[g] * x_star_hi**n / z_of(x_star_lo)
        tilt_star_intervals.append((t_lo, t_hi))

    # Declared rational multiplier point: x0 truncates x_star at 10^-12.
    x0 = Fraction(int(x_star_lo * DECLARED_X0_DENOM), DECLARED_X0_DENOM)
    require(0 < x0, "declared multiplier point fails positivity")
    z0 = z_of(x0)
    tilt_x0 = [tau_chain[g] * x0 ** actions[g] / z0 for g in range(8)]
    require(sum(tilt_x0) == 1, "declared tilted law fails normalization")
    mean_tilt_x0 = sum(tilt_x0[g] * actions[g] for g in range(8))
    mismatch = abs(mean_tilt_x0 - mean_emp)
    eps = Fraction(1, 10**12)
    while eps <= mismatch:
        eps *= 10
    lam0_lo, lam0_hi = iv_log_fraction(x0)
    lam0_interval = (-lam0_hi, -lam0_lo)

    # Bracket literals for the Lean intermediate-value receipt.
    x_lo_lean = Fraction(int(x_star_lo * BRACKET_DENOM), BRACKET_DENOM)
    x_hi_lean = x_lo_lean + Fraction(1, BRACKET_DENOM)
    require(quad(x_lo_lean) < 0 < quad(x_hi_lean), "Lean bracket literals fail the sign test")

    payload = {
        "schema": "oph.b7.history_law_packet.v1",
        "issue": 683,
        "receipt": "source-produced history-law packet (reference, action, empirical law)",
        "claim_boundary": (
            "Representation-level production of the history-law inputs from one "
            "preregistered bounded source run. The run produces the reference "
            "path law (the stationary Markov path law of the committed "
            "mixing chain), the source-intrinsic action (the repair-move count "
            "of the path), and the empirical path law at exactly computed "
            "divergence from the reference. The mean-action matching "
            "multiplier is matched to the DECLARED alternative mean, the "
            "empirical mean action of the same run; no source mechanism "
            "selects the multiplier. The multiplier selection principle from "
            "source data is the remaining gap of B7."
        ),
        "provenance": {
            "run_id": manifest["run_id"],
            "run_dir": str(RUN_DIR.relative_to(REPO_ROOT)),
            "seed": seed_material["seed"],
            "config_hash": seed_material["config_hash"],
            "git_commit": git_commit,
            "extraction_script": "scripts/extract_b7_history_packet.py",
            "committed_chain_payload": "docs/B12_MIXING_CHAIN_PAYLOAD.json",
            "committed_chain_payload_sha256": sha256_file(B12_PAYLOAD_PATH),
            "pinned_input_sha256": {name: sha256_file(RUN_DIR / name) for name in PINNED_INPUTS},
            "pinned_digests_match_committed_payload": True,
        },
        "path_space": {
            "states": "local 0 = global 18 (checkpoint_class 3), local 1 = global 19 (checkpoint_class 2)",
            "path_length_states": 3,
            "path_length_transitions": 2,
            "path_count": 8,
            "encoding": "g = 4*s0 + 2*s1 + s2",
            "window_convention": (
                "each maximal in-class segment of an observer transition history "
                "contributes every consecutive state triple as one window; a "
                "segment of length L contributes L-2 windows"
            ),
            "in_class_segment_count": segment_count,
            "window_total": window_total,
            "pair_recount_matches_committed_counts": True,
        },
        "action": {
            "definition": "S(path) = number of state changes along the path",
            "reading": (
                "the repair-move count of the length-3 history: a source-"
                "intrinsic action computed from the path alone, needing no "
                "target data"
            ),
            "values_by_path": actions,
            "value_range": [min(actions), max(actions)],
            "nonconstant": True,
        },
        "paths": [
            {
                "g": g,
                "states": list(path_states(g)),
                "global_states": [recurrent[s] for s in path_states(g)],
                "action": actions[g],
                "window_count": triple_counts[g],
                "tau_emp_exact": str(tau_emp[g]),
                "tau_emp_float": float(tau_emp[g]),
                "tau_chain_exact": str(tau_chain[g]),
                "tau_chain_float": float(tau_chain[g]),
            }
            for g in range(8)
        ],
        "empirical_law": {
            "source": "integer window counts, normalized",
            "strictly_positive": True,
            "sum_exact_one": True,
            "mean_action_exact": str(mean_emp),
            "mean_action_float": float(mean_emp),
        },
        "reference_law": {
            "source": (
                "stationary Markov path law of the committed chain: "
                "pi(s0) * P(s0,s1) * P(s1,s2) with the committed exact rationals"
            ),
            "initial_law_declared": "committed exact stationary law of the mixing-chain payload",
            "strictly_positive": True,
            "sum_exact_one": True,
            "mean_action_exact": str(mean_chain),
            "mean_action_float": float(mean_chain),
        },
        "divergence": {
            "method": (
                "outward-rounded interval arithmetic (mpmath.iv) at "
                f"{ENCLOSURE_PREC_BITS} bits of working precision on exact "
                "rational log arguments; endpoints are exact binary rationals"
            ),
            "kl_emp_chain_nats": interval_str(*kl_emp_chain),
            "kl_chain_emp_nats": interval_str(*kl_chain_emp),
            "kl_emp_chain_strictly_positive": True,
        },
        "tilted_family": {
            "parametrization": "x = exp(-lambda); tilt_g proportional to tau_chain_g * x^S_g",
            "lambda_zero_receipt": "at lambda = 0 the tilt equals tau_chain exactly (proved in the Lean mirror)",
            "declared_alternative_mean": {
                "value_exact": str(mean_emp),
                "declaration": (
                    "the empirical mean action of the same run; DECLARED as the "
                    "matching target, since a multiplier selection principle "
                    "from source data is absent"
                ),
            },
            "matching_equation": {
                "form": "quad_a * x^2 + quad_b * x + quad_c = 0",
                "derivation": (
                    "mean-action matching of the tilt at level m: "
                    "sum_g tau_chain_g (S_g - m) x^S_g = 0 with "
                    "c_k = sum of tau_chain over paths of action k"
                ),
                "c0_exact": str(c[0]),
                "c1_exact": str(c[1]),
                "c2_exact": str(c[2]),
                "quad_a_exact": str(quad_a),
                "quad_b_exact": str(quad_b),
                "quad_c_exact": str(quad_c),
                "value_at_one_exact": str(mean_chain - mean_emp),
                "value_at_one_identity": "F(1) = mean_chain - mean_emp, verified exactly",
                "positive_root_unique": (
                    "quad_a > 0, quad_b > 0, quad_c < 0, so F is increasing on "
                    "x > 0 with F(0) < 0: exactly one positive root"
                ),
            },
            "x_star_enclosure": {
                "method": "exact-rational bisection; declared width bound 10^-40",
                **interval_str(x_star_lo, x_star_hi),
            },
            "lambda_star_enclosure": {
                "method": (
                    "outward interval negation of mpmath.iv log on the exact "
                    f"x_star enclosure endpoints at {ENCLOSURE_PREC_BITS} bits"
                ),
                **interval_str(lam_star_lo, lam_star_hi),
            },
            "tilted_law_at_x_star_enclosures": [
                {
                    "g": g,
                    "action": actions[g],
                    **interval_str(*tilt_star_intervals[g]),
                }
                for g in range(8)
            ],
            "declared_multiplier_point": {
                "x0_exact": str(x0),
                "x0_declaration": f"x_star truncated at denominator {DECLARED_X0_DENOM}",
                "lambda0_form": "lambda0 = -log x0",
                "lambda0_enclosure": interval_str(*lam0_interval),
                "tilted_law_at_x0_exact": [str(t) for t in tilt_x0],
                "tilted_law_at_x0_float": [float(t) for t in tilt_x0],
                "tilted_law_at_x0_is_exactly_rational": True,
                "mean_action_at_x0_exact": str(mean_tilt_x0),
                "mean_action_mismatch_exact": str(mismatch),
                "mean_action_mismatch_bound": str(eps),
            },
            "lean_bracket_literals": {
                "x_lo": str(x_lo_lean),
                "x_hi": str(x_hi_lean),
                "F_x_lo_exact": str(quad(x_lo_lean)),
                "F_x_hi_exact": str(quad(x_hi_lean)),
                "sign_change_verified_exact": True,
            },
        },
        "lean_mirror": {
            "file": "reverse-engineering-reality Lean/InformationProjection/SourceHistoryPacket.lean",
            "contents": (
                "the eight path literals, both laws, and the action as "
                "literals; strict positivity and normalization of both laws; "
                "nonconstant action (kernel-decided); the chain law equal to "
                "the Markov product of the committed chain, with the "
                "cleared-denominator integer identity kernel-decided; strict "
                "positivity of KL(tau_emp || tau_chain); the tilt of "
                "tau_chain at lambda = 0 equal to tau_chain exactly; the "
                "earned inhabitation of the HistoryLaw interface at the "
                "packet literals (the tilt strictly positive, normalized, "
                "and the unique constrained KL minimizer at every "
                "multiplier, the multiplier a free input); the tilt at the "
                "declared lambda0 = -log x0 equal to the declared rational "
                "tilted law exactly; the mean-action mismatch bound at x0; "
                "and the intermediate-value receipt locating a matching "
                "multiplier inside the declared rational bracket"
            ),
        },
        "remaining_gap": (
            "the multiplier selection principle from source data: the "
            "multiplier is matched here to the declared empirical mean action "
            "of the same run, so the constraint level is an input to the "
            "packet; producing lambda (equivalently the constraint level) "
            "from the substrate is the open core of B7"
        ),
        "verification": {
            "exact_arithmetic": "python fractions.Fraction throughout except the declared interval enclosures",
            "pinned_digests_match_committed_payload": True,
            "committed_chain_recovered_from_committed_counts": True,
            "committed_stationarity_reverified_exact": True,
            "pair_recount_matches_committed_counts": True,
            "all_eight_paths_realized": True,
            "both_laws_sum_to_one_exact": True,
            "both_laws_strictly_positive": True,
            "action_nonconstant": True,
            "quadratic_cross_check_at_one": True,
            "bisection_sign_test": True,
            "lean_bracket_sign_test": True,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")

    print("\n=== Lean literal block ===")
    print("window counts:", triple_counts, "total:", window_total)
    print("actions:", actions)
    print("tau_emp:", [str(t) for t in tau_emp])
    print("tau_chain:", [str(t) for t in tau_chain])
    print("mean_emp:", str(mean_emp), " mean_chain:", str(mean_chain))
    print("c0,c1,c2:", str(c[0]), str(c[1]), str(c[2]))
    print("quad_a,b,c:", str(quad_a), str(quad_b), str(quad_c))
    print("x_star ~", float((x_star_lo + x_star_hi) / 2))
    print("lambda_star in", float(lam_star_lo), float(lam_star_hi))
    print("x0:", str(x0))
    print("z0:", str(z0))
    print("tilt_x0:", [str(t) for t in tilt_x0])
    print("mean_tilt_x0:", str(mean_tilt_x0), " mismatch:", str(mismatch), " eps:", str(eps))
    print("x_lo,x_hi:", str(x_lo_lean), str(x_hi_lean))
    print("F(x_lo):", str(quad(x_lo_lean)), " F(x_hi):", str(quad(x_hi_lean)))


if __name__ == "__main__":
    main()
