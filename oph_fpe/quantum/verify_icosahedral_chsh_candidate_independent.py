"""Independent verifier for the exact issue-652 projective candidate."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from oph_fpe.core.charged_response import (
    Q5,
    canonical_sha256,
    load_carrier,
    match_vertex_frame,
    q5_dot,
)
from oph_fpe.core.spin_statistics_response import (
    measure_deck_realization,
    measure_lift_group,
    produce_spin_statistics_artifact,
)


SCHEMA = "oph.icosahedral_chsh_candidate.v1"
VERDICT = (
    "EXACT_PROJECTIVE_BRANCH_CANDIDATE__"
    "TWO_WING_COMPLETED_RECORD_SOURCE_PRODUCER_MISSING"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "echosahedral_federation_reference.json"
)
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT / "data" / "quantum" / "icosahedral_chsh_candidate_receipt.json"
)


class VerificationError(ValueError):
    """Fail-closed independent verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def signed_lifts(lifts: Mapping[tuple[int, ...], Any]) -> list[Any]:
    group = {}
    for value in lifts.values():
        group[value.key()] = value
        group[value.neg().key()] = value.neg()
    return list(group.values())


def character_invariant_multiplicity(group: list[Any]) -> Q5:
    """Multiplicity of the trivial representation in S tensor S.

    For a unit quaternion q=(w,x,y,z), the defining SU(2) spinor character is
    2w.  The finite-group character inner product gives
    dim (S tensor S)^G = |G|^-1 sum_g character(g)^2.
    """

    total = Q5.of(0)
    for quaternion in group:
        character = Q5.of(2) * quaternion.w
        total = total + character * character
    return total * Q5.of(Fraction(1, len(group)))


def spinor_matrix_float(quaternion: Any) -> np.ndarray:
    w = quaternion.w.to_float()
    x = quaternion.x.to_float()
    y = quaternion.y.to_float()
    z = quaternion.z.to_float()
    return np.asarray(
        [[w - 1j * z, -y - 1j * x], [y - 1j * x, w + 1j * z]],
        dtype=np.complex128,
    )


def pauli_float(vector: Any) -> np.ndarray:
    x, y, z = (entry.to_float() for entry in vector)
    return np.asarray([[z, x - 1j * y], [x + 1j * y, -z]], dtype=np.complex128)


def verify_covariance_numerically(
    deck: Mapping[str, Any], lifts: Mapping[Any, Any], frame: Any
) -> float:
    maximum = 0.0
    for permutation in deck["rotations"]:
        unitary = spinor_matrix_float(lifts[permutation])
        rotation = np.asarray(
            [
                [entry.to_float() for entry in row]
                for row in deck["matrices"][permutation]
            ],
            dtype=float,
        )
        for vector in frame:
            raw = np.asarray([entry.to_float() for entry in vector], dtype=float)
            left = unitary @ pauli_float(vector) @ unitary.conjugate().T
            rotated = rotation @ raw
            right = np.asarray(
                [
                    [rotated[2], rotated[0] - 1j * rotated[1]],
                    [rotated[0] + 1j * rotated[1], -rotated[2]],
                ],
                dtype=np.complex128,
            )
            maximum = max(maximum, float(np.max(np.abs(left - right))))
    return maximum


def verify_singlet_invariance_numerically(group: list[Any]) -> float:
    """Numerically cross-check the explicitly named alternating invariant ray."""

    singlet = np.asarray([0.0, 1.0, -1.0, 0.0], dtype=np.complex128)
    maximum = 0.0
    for quaternion in group:
        unitary = spinor_matrix_float(quaternion)
        residual = np.kron(unitary, unitary) @ singlet - singlet
        maximum = max(maximum, float(np.max(np.abs(residual))))
    return maximum


def setting_rows(carrier: Mapping[str, Any]) -> list[tuple[int, int, int, int]]:
    rows = []
    adjacency = carrier["adjacency"]
    antipode = carrier["antipode"]
    for a0 in range(12):
        for a1 in range(12):
            if a1 == a0 or a1 == antipode[a0] or adjacency[a0][a1]:
                continue
            for b0 in range(12):
                if adjacency[a0][b0] and adjacency[a1][b0]:
                    rows.append((a0, a1, b0, antipode[a1]))
    return rows


def independent_scores(carrier: Mapping[str, Any], frame: Any) -> list[Q5]:
    norm_squared = q5_dot(frame[0], frame[0])
    inverse_norm = norm_squared.inverse()

    def singlet_correlation(left: int, right: int) -> Q5:
        return -(q5_dot(frame[left], frame[right]) * inverse_norm)

    scores = []
    for a0, a1, b0, b1 in setting_rows(carrier):
        scores.append(
            singlet_correlation(a0, b0)
            + singlet_correlation(a0, b1)
            + singlet_correlation(a1, b0)
            - singlet_correlation(a1, b1)
        )
    return scores


def independent_correlations(frame: Any, row: tuple[int, int, int, int]) -> list[Q5]:
    norm_squared = q5_dot(frame[0], frame[0])
    inverse_norm = norm_squared.inverse()

    def correlation(left: int, right: int) -> Q5:
        return -(q5_dot(frame[left], frame[right]) * inverse_norm)

    a0, a1, b0, b1 = row
    return [
        correlation(a0, b0),
        correlation(a0, b1),
        correlation(a1, b0),
        correlation(a1, b1),
    ]


def independent_setting_census(
    carrier: Mapping[str, Any], frame: Any, deck: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute orbit structure and ambient maximizer counts independently."""

    declared_rows = set(setting_rows(carrier))
    remaining = set(declared_rows)
    proper_orbits = []
    for seed in sorted(declared_rows):
        if seed not in remaining:
            continue
        orbit = {
            tuple(permutation[index] for index in seed)
            for permutation in deck["rotations"]
        }
        require(
            orbit <= declared_rows, "setting family is not proper-rotation covariant"
        )
        proper_orbits.append(orbit)
        remaining -= orbit
    require(not remaining, "setting-family orbit exhaustion failed")
    orbit_sizes = sorted(len(orbit) for orbit in proper_orbits)
    require(orbit_sizes == [60, 60], "setting-family proper-orbit census failed")
    require(
        all(
            {tuple(permutation[index] for index in row) for row in proper_orbits[0]}
            == proper_orbits[1]
            for permutation in deck["improper"]
        ),
        "improper coset does not exchange the setting orbits",
    )

    norm_squared = q5_dot(frame[0], frame[0])
    inverse_norm = norm_squared.inverse()

    def correlation(left: int, right: int) -> Q5:
        return -(q5_dot(frame[left], frame[right]) * inverse_norm)

    maximum = Q5.of(0)
    maximizer_count = 0
    all_distinct_maximizer_count = 0
    for a0 in range(12):
        for a1 in range(12):
            for b0 in range(12):
                for b1 in range(12):
                    score = (
                        correlation(a0, b0)
                        + correlation(a0, b1)
                        + correlation(a1, b0)
                        - correlation(a1, b1)
                    )
                    magnitude = score if score.sign() >= 0 else -score
                    comparison = (magnitude - maximum).sign()
                    if comparison > 0:
                        maximum = magnitude
                        maximizer_count = 1
                        all_distinct_maximizer_count = int(len({a0, a1, b0, b1}) == 4)
                    elif comparison == 0:
                        maximizer_count += 1
                        all_distinct_maximizer_count += int(len({a0, a1, b0, b1}) == 4)
    return {
        "proper_rotation_orbit_count": len(proper_orbits),
        "proper_rotation_orbit_sizes": orbit_sizes,
        "improper_coset_exchanges_orbits": True,
        "ambient_ordered_port_quadruple_count": 12**4,
        "ambient_maximum_absolute_chsh": maximum.render(),
        "ambient_maximizer_count": maximizer_count,
        "ambient_all_distinct_maximizer_count": all_distinct_maximizer_count,
        "declared_family_count": len(declared_rows),
        "declared_family_is_unique_maximizer": False,
        "meaning": (
            "the 120 rows form a covariant declared construction, not a unique "
            "setting family selected by the carrier or repair dynamics"
        ),
    }


def verify(receipt: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    require(receipt.get("schema") == SCHEMA, "unexpected receipt schema")
    require(receipt.get("issue") == 652, "unexpected issue")
    require(receipt.get("verdict") == VERDICT, "unexpected verdict")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    require(
        receipt.get("receipt_sha256") == canonical_sha256(body), "receipt hash failed"
    )
    gate = receipt["completed_record_gate"]
    require(
        gate.get("source_positive_nonclassical_record_witness") is False,
        "false source promotion",
    )
    require(gate.get("physical_promotion_allowed") is False, "false physical promotion")

    carrier = load_carrier(manifest)
    frame = match_vertex_frame(carrier)
    spin = produce_spin_statistics_artifact(manifest)
    parents = receipt["source_parents"]
    expected_parents = {
        "carrier_manifest_sha256": carrier["manifest_sha256"],
        "spin_statistics_artifact_sha256": spin["artifact_sha256"],
        "finite_spin_packet_gate_passed": spin["physical_source_gate"]["passed"],
        "laboratory_exchange_measurement": spin["physical_source_gate"][
            "laboratory_exchange_measurement"
        ],
        "continuum_spin_statistics_theorem": spin["physical_source_gate"][
            "continuum_spin_statistics_theorem"
        ],
        "lift_group_order": spin["lift_measurement"]["lift_group_order"],
        "lift_group_centre_order": spin["lift_measurement"]["centre_order"],
        "spin_structure_count": spin["support_homology"]["spin_structure_count"],
    }
    require(
        parents.get("spin_statistics_artifact_sha256") == spin["artifact_sha256"],
        "spin artifact pin failed",
    )
    require(
        parents.get("laboratory_exchange_measurement") is False
        and parents.get("continuum_spin_statistics_theorem") is False,
        "false spin-statistics promotion",
    )
    require(parents == expected_parents, "source-parent projection failed")
    require(spin["physical_source_gate"]["passed"] is True, "spin source gate failed")
    require(
        receipt["source_derived_content"]
        == {
            "carrier_lineage_scope": "declared_echosahedral_carrier_lineage",
            "universal_oph_carrier_derivation": False,
            "oriented_twelve_port_frame": True,
            "proper_deck_rotation_count": 60,
            "binary_icosahedral_spin_lift": True,
            "non_split_central_extension": True,
            "port_incidence_and_antipode": True,
        },
        "source-derived content projection failed",
    )

    deck = measure_deck_realization(carrier, frame)
    lift = measure_lift_group(deck)
    group = signed_lifts(lift["lifts"])
    require(len(group) == 120, "signed lift group order failed")
    multiplicity = character_invariant_multiplicity(group)
    require((multiplicity - Q5.of(1)).is_zero(), "invariant-line multiplicity failed")
    singlet_residual = verify_singlet_invariance_numerically(group)
    require(singlet_residual < 1.0e-12, "explicit alternating invariant ray failed")
    adapter = receipt["exact_projective_adapter"]
    require(
        adapter["status"]
        == "mathematical construction from the source-derived spin lift",
        "projective-adapter status failed",
    )
    require(adapter["spinor_complex_dimension"] == 2, "spinor dimension failed")
    invariant = adapter["diagonal_invariant_line"]
    require(
        invariant
        == {
            "signed_lift_count": 120,
            "singlet_invariant_under_all_lifts": True,
            "common_constraint_rank": 3,
            "tensor_dimension": 4,
            "invariant_dimension": 1,
            "invariant_ray": "span(|01>-|10>)",
        },
        "serialized invariant-line certificate failed",
    )
    require(
        adapter["meaning"]
        == (
            "the binary lift admits its defining two-complex-dimensional spinor "
            "realization and a unique diagonal invariant ray within that declared "
            "tensor representation; this does not make that realization the "
            "physical completed-record algebra"
        ),
        "projective-adapter meaning failed",
    )

    covariance_residual = verify_covariance_numerically(deck, lift["lifts"], frame)
    require(covariance_residual < 1.0e-12, "independent Pauli covariance failed")
    require(
        adapter["pauli_axis_covariance"]
        == {
            "proper_rotation_count": 60,
            "port_axis_count": 12,
            "exact_covariance_checks": 720,
            "all_passed": True,
        },
        "serialized covariance certificate failed",
    )

    rows = setting_rows(carrier)
    scores = independent_scores(carrier, frame)
    expected = -Q5.of(1, Fraction(3, 5))
    require(len(rows) == len(scores) == 120, "setting-family count failed")
    require(
        all((score - expected).is_zero() for score in scores),
        "CHSH score replay failed",
    )
    magnitude = -expected
    margin = magnitude - Q5.of(2)
    require(margin.sign() > 0, "CHSH value does not exceed two")
    candidate = receipt["chsh_candidate"]
    require(
        candidate["setting_quadruple_count"] == len(rows),
        "serialized setting count failed",
    )
    require(
        candidate["signed_chsh"] == expected.render(), "serialized signed CHSH failed"
    )
    require(
        candidate["absolute_chsh"] == magnitude.render(),
        "serialized absolute CHSH failed",
    )
    require(
        math.isclose(
            candidate["absolute_chsh_float"],
            magnitude.to_float(),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "serialized CHSH float failed",
    )
    require(
        candidate["selection_rule"]
        == (
            "ordered nonadjacent, nonantipodal A0/A1 ports; B0 any common "
            "incidence neighbor; B1 the antipode of A1"
        ),
        "serialized setting rule failed",
    )
    require(candidate["all_rows_same_exact_score"] is True, "same-score flag failed")
    representative = rows[0]
    correlations = independent_correlations(frame, representative)
    require(
        candidate["representative_ports"]
        == [carrier["ports"][index] for index in representative],
        "representative ports failed",
    )
    require(
        candidate["representative_correlations"]
        == [correlation.render() for correlation in correlations],
        "representative correlations failed",
    )
    require(
        candidate["classical_margin"] == margin.render(), "serialized margin failed"
    )
    require(
        candidate["classical_margin_positive_exact"] is True,
        "positive-margin flag failed",
    )
    require(
        candidate["setting_selection"]
        == {
            "status": "declared_combinatorial_construction",
            "source_selected": False,
            "unique_carrier_selected_family": False,
        },
        "setting-selection boundary failed",
    )
    require(
        candidate["setting_family_symmetry"]
        == independent_setting_census(carrier, frame, deck),
        "setting-family symmetry census failed",
    )

    probability_rows = candidate["joint_law"]["rows"]
    expected_probability_rows = []
    exact_probabilities: dict[tuple[int, int, int, int], Q5] = {}
    for setting_index, correlation in enumerate(correlations):
        setting_x, setting_y = setting_index // 2, setting_index % 2
        for left_outcome in (-1, 1):
            for right_outcome in (-1, 1):
                probability = (
                    Q5.of(1) + Q5.of(left_outcome * right_outcome) * correlation
                ) * Q5.of(Fraction(1, 4))
                require(probability.sign() >= 0, "independent joint law is negative")
                exact_probabilities[
                    (setting_x, setting_y, left_outcome, right_outcome)
                ] = probability
                expected_probability_rows.append(
                    {
                        "setting_pair": [setting_x, setting_y],
                        "outcomes": [left_outcome, right_outcome],
                        "probability": probability.render(),
                    }
                )
    require(
        probability_rows == expected_probability_rows,
        "serialized joint-law values failed",
    )
    joint_law = candidate["joint_law"]
    require(
        joint_law["formula"] == "p(a,b|x,y)=(1+a*b*E(x,y))/4",
        "joint-law formula failed",
    )
    require(
        joint_law["local_marginals"] == "1/2 on every setting slice",
        "joint-law marginal label failed",
    )
    require(joint_law["algebraic_no_signalling"] is True, "no-signalling flag failed")
    for setting_x in (0, 1):
        for setting_y in (0, 1):
            selected = {
                (left_outcome, right_outcome): exact_probabilities[
                    (setting_x, setting_y, left_outcome, right_outcome)
                ]
                for left_outcome in (-1, 1)
                for right_outcome in (-1, 1)
            }
            total = sum(selected.values(), Q5.of(0))
            require((total - Q5.of(1)).is_zero(), "joint-law normalization failed")
            for wing in (0, 1):
                for outcome in (-1, 1):
                    marginal = sum(
                        (
                            probability
                            for outcomes, probability in selected.items()
                            if outcomes[wing] == outcome
                        ),
                        Q5.of(0),
                    )
                    require(
                        (marginal - Q5.of(Fraction(1, 2))).is_zero(),
                        "joint-law marginal failed",
                    )
            recovered_correlation = sum(
                (
                    Q5.of(left_outcome * right_outcome) * probability
                    for (left_outcome, right_outcome), probability in selected.items()
                ),
                Q5.of(0),
            )
            require(
                (
                    recovered_correlation - correlations[2 * setting_x + setting_y]
                ).is_zero(),
                "joint-law correlation recovery failed",
            )

    deterministic_scores = []
    for a0 in (-1, 1):
        for a1 in (-1, 1):
            for b0 in (-1, 1):
                for b1 in (-1, 1):
                    deterministic_scores.append(a0 * b0 + a0 * b1 + a1 * b0 - a1 * b1)
    require(
        set(deterministic_scores) == {-2, 2}, "classical deterministic bound failed"
    )
    require(
        receipt["classical_reference"]
        == {
            "deterministic_assignment_count": 16,
            "score_counts": {"-2": 8, "2": 8},
            "maximum_absolute_score": 2,
        },
        "classical reference failed",
    )

    expected_gate = {
        "source_positive_nonclassical_record_witness": False,
        "physical_promotion_allowed": False,
        "first_missing_producer": "two_wing_completed_record_instrument",
        "required_outputs": [
            "two source-produced spinor wings on one declared physical domain",
            "preparation or repair selection of the diagonal invariant ray",
            "independent binary local setting choices with measurement-independence custody",
            "local Pauli-axis readout maps for the covariant port-setting family",
            "binary completed outcomes with no postselection",
            "a spacelike or operational-isolation certificate for the two wings",
            "setting-conditioned joint counts and no-signalling checks",
        ],
        "missing_outputs_in_registered_source": [
            "wing tensor-factor producer",
            "source state preparation",
            "setting intervention channel",
            "setting-to-readout attachment",
            "completed two-wing outcome records",
            "physical locality and measurement-independence certificates",
        ],
    }
    require(gate == expected_gate, "completed-record gate failed")

    expected_boundaries = {
        "carrier_scope": (
            "the parent is the declared echosahedral carrier lineage certified "
            "by issue 565; the packet does not derive this carrier from every "
            "possible OPH screen"
        ),
        "issue_230": (
            "the quantum algebra-state and Born representation are not derived "
            "from pre-quantum repair records"
        ),
        "issue_311": (
            "the separate finite spectral interfaces retain their explicit "
            "classical harmonic completions"
        ),
        "central_defect_scope": (
            "the source-derived non-split spin lift supplies projective geometry; "
            "it does not by itself supply quantum operational semantics"
        ),
        "simulation_scope": (
            "the exact candidate is a source-geometry calculation, not a simulated "
            "Bell experiment or a laboratory prediction"
        ),
        "setting_scope": (
            "the 120-row family is a declared covariant construction and is not "
            "uniquely selected by the source geometry or repair dynamics"
        ),
    }
    require(receipt["boundaries"] == expected_boundaries, "claim boundaries failed")
    require(
        receipt["claim_boundary"]
        == (
            "Exact finite projective-branch candidate on the declared echosahedral "
            "carrier lineage: the binary-icosahedral lift, its defining spinor adapter, "
            "and a declared covariant port-setting family give "
            "|CHSH|=1+3/sqrt(5)>2 on the adapter's unique diagonal invariant ray. The "
            "setting family is not uniquely source-selected. No registered source "
            "producer prepares two physical spinor wings or emits setting-controlled "
            "spacelike completed records, so this is not a nonclassical record witness "
            "on the source surface. The representation lane stops at its named-producer "
            "boundary."
        ),
        "claim-boundary prose failed",
    )

    return {
        "status": "PASS",
        "setting_rows": len(rows),
        "invariant_multiplicity": multiplicity.render(),
        "absolute_chsh": magnitude.render(),
        "covariance_max_residual": covariance_residual,
        "singlet_max_residual": singlet_residual,
        "ambient_maximizer_count": candidate["setting_family_symmetry"][
            "ambient_maximizer_count"
        ],
        "source_positive_nonclassical_record_witness": False,
        "first_missing_producer": gate["first_missing_producer"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    result = verify(load_json(args.receipt), load_json(args.manifest))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
