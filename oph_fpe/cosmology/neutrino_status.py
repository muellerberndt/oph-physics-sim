from __future__ import annotations

from typing import Any


CONVENTIONAL_CAMB_NEUTRINO_ASSUMPTION = "sum_mnu_0.06eV_one_massive_two_massless"
CONVENTIONAL_CAMB_NEUTRINO_MASSES_EV = (0.0, 0.0, 0.06)
CONVENTIONAL_CAMB_SUM_MNU_EV = 0.06

HISTORICAL_REJECTED_WEIGHTED_CYCLE_MASSES_EV = (
    0.017454720257976796,
    0.019481987935919015,
    0.05307522145074924,
)

NEUTRINO_LANE_CLOSURE_ARTIFACT = (
    "reverse-engineering-reality/code/particles/runs/neutrino/"
    "neutrino_lane_closure_contract.json"
)
NEUTRINO_LANE_CLOSURE_SHA256 = "2d473f1d7b41e52f2c2821931839c324c692f2ceb8b8a1bca34028402183822f"
NEUTRINO_NUFIT_SCORE_ARTIFACT = (
    "reverse-engineering-reality/code/particles/runs/neutrino/"
    "nufit61_weighted_cycle_retrospective_score.json"
)
NEUTRINO_NUFIT_SCORE_SHA256 = "3b0632d9a946ae83abb74d4ba1171356ace195f93c20f0a3e03e6f8be10251cd"


def neutrino_mass_status(*, include_rejected_benchmark: bool = False) -> dict[str, Any]:
    """Return the fail-closed neutrino-mass status shared by cosmology lanes.

    The weighted-cycle row is deliberately value-redacted unless a caller
    explicitly opts into the rejected historical benchmark.  Even in that
    mode its promotion flag is immutable and false.
    """

    include = bool(include_rejected_benchmark)
    historical_masses = (
        [float(value) for value in HISTORICAL_REJECTED_WEIGHTED_CYCLE_MASSES_EV]
        if include
        else None
    )
    return {
        "schema_version": "oph_neutrino_mass_status_v1",
        "oph_derived_prediction": {
            "available": False,
            "mass_ordering": None,
            "masses_eV": None,
            "sum_mnu_eV": None,
            "public_promotion_allowed": False,
            "status": "no_source_derived_neutrino_mass_prediction",
            "missing_source_object": "source_closed_neutrino_operator_basis_ordering_and_absolute_scale",
        },
        "conventional_camb_baseline": {
            "assumption": CONVENTIONAL_CAMB_NEUTRINO_ASSUMPTION,
            "solver_mass_components_eV": [
                float(value) for value in CONVENTIONAL_CAMB_NEUTRINO_MASSES_EV
            ],
            "sum_mnu_eV": float(CONVENTIONAL_CAMB_SUM_MNU_EV),
            "epistemic_status": "conventional_solver_reference_not_oph_prediction",
            "counts_as_oph_prediction": False,
        },
        "historical_rejected_weighted_cycle_benchmark": {
            "included": include,
            "opt_in_required": True,
            "mass_ordering": "normal" if include else None,
            "masses_eV": historical_masses,
            "sum_mnu_eV": float(sum(HISTORICAL_REJECTED_WEIGHTED_CYCLE_MASSES_EV)) if include else None,
            "public_promotion_allowed": False,
            "prospective_evidence_eligible": False,
            "source_only_prediction_eligible": False,
            "historical_target_exposure": True,
            "status": "rejected_weighted_cycle_retrospective_benchmark",
            "declared_gate": {
                "current_weighted_cycle_candidate_rejected": True,
                "threshold_delta_chi2_2d_3sigma": 11.829158081900795,
                "TBoff_NO_delta_chi2_lower_bound": 18.435275228909504,
                "TByes_NO_delta_chi2_lower_bound": 20.119548723965472,
            },
            "provenance": {
                "lane_closure_artifact": NEUTRINO_LANE_CLOSURE_ARTIFACT,
                "lane_closure_sha256": NEUTRINO_LANE_CLOSURE_SHA256,
                "nufit61_score_artifact": NEUTRINO_NUFIT_SCORE_ARTIFACT,
                "nufit61_score_sha256": NEUTRINO_NUFIT_SCORE_SHA256,
                "nufit_release": "NuFIT 6.1",
                "artifact_generated_utc": "2026-07-12T02:31:32Z",
            },
        },
    }
