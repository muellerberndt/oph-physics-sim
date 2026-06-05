from __future__ import annotations

from typing import Any

from oph_fpe.claims import CONTINUATION, with_claim_metadata


EXPECTED_SM_GATE = {
    "g_phys": "(SU(3)xSU(2)xU(1))/Z6",
    "hypercharge_lattice": "exact",
    "Nc": 3,
    "Ng": 3,
    "higgs_doublets": 1,
    "light_chiral_exotics": 0,
    "extra_low_scale_u1": 0,
    "xy_gauge_bosons": 0,
}


def standard_model_candidate_sieve(candidate: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "sm_quotient": _normalize_group(candidate.get("g_phys", candidate.get("G_phys"))) == EXPECTED_SM_GATE["g_phys"],
        "exact_hypercharge_lattice": str(candidate.get("hypercharge_lattice", "")).lower() == "exact",
        "Nc_equals_3": int(candidate.get("Nc", candidate.get("colors", -1))) == 3,
        "Ng_equals_3": int(candidate.get("Ng", candidate.get("generations", -1))) == 3,
        "one_higgs_package": int(candidate.get("higgs_doublets", candidate.get("higgs_package_count", -1))) == 1,
        "no_light_chiral_exotics": int(candidate.get("light_chiral_exotics", -1)) == 0,
        "no_extra_low_scale_u1": int(candidate.get("extra_low_scale_u1", -1)) == 0,
        "no_xy_gauge_bosons": int(candidate.get("xy_gauge_bosons", candidate.get("X_Y_gauge_bosons", -1))) == 0,
    }
    receipt = all(checks.values())
    report = {
        "mode": "finite_mar_standard_model_candidate_sieve",
        "SM_QUOTIENT_GATE_RECEIPT": bool(receipt),
        "receipt": bool(receipt),
        "checks": checks,
        "candidate": dict(candidate),
        "expected_gate": dict(EXPECTED_SM_GATE),
        "claim_boundary": (
            "finite visible low-energy Standard Model gate harness. This instantiates the paper's "
            "classification/selection boundary but is not a derivation of MAR or compact-gauge "
            "reconstruction from first principles"
        ),
    }
    return with_claim_metadata(report, claim_level=CONTINUATION, receipt="SM_QUOTIENT_GATE_RECEIPT")


def _normalize_group(value: Any) -> str:
    return str(value).replace(" ", "").replace("*", "x")
