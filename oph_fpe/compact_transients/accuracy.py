from __future__ import annotations

from typing import Any


def simulator_accuracy_receipt(error_ledger: dict[str, float | None]) -> dict[str, Any]:
    required = (
        "epsilon_mu",
        "epsilon_K",
        "expected_path_length",
        "epsilon_E",
        "epsilon_prop",
        "epsilon_detector",
        "epsilon_canon",
        "epsilon_clock",
        "epsilon_mc",
    )
    missing = [name for name in required if error_ledger.get(name) is None]
    if missing:
        return {
            "receipt_type": "SIMULATOR_ACCURACY_RECEIPT",
            **{name: error_ledger.get(name) for name in required},
            "tv_bound": None,
            "status": "OPEN_GATE",
            "missing_error_terms": missing,
        }
    tv_bound = (
        float(error_ledger["epsilon_mu"])
        + float(error_ledger["expected_path_length"]) * float(error_ledger["epsilon_K"])
        + float(error_ledger["epsilon_E"])
        + float(error_ledger["epsilon_prop"])
        + float(error_ledger["epsilon_detector"])
        + float(error_ledger["epsilon_canon"])
        + float(error_ledger["epsilon_clock"])
        + float(error_ledger["epsilon_mc"])
    )
    return {
        "receipt_type": "SIMULATOR_ACCURACY_RECEIPT",
        **{name: float(error_ledger[name]) for name in required},
        "tv_bound": tv_bound,
        "status": "PASS",
        "missing_error_terms": [],
    }
