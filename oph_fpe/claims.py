from __future__ import annotations

from typing import Any


RECOVERED_CORE = "recovered_core"
QUANTITATIVE_BRANCH = "quantitative_branch"
CONTINUATION = "continuation"
PROXY = "proxy"
DEMO = "demo"
DEBUG = "debug"
BRANCH_INSTANTIATION_SANITY = "branch_instantiation_sanity"

CLAIM_LEVELS = {
    RECOVERED_CORE,
    QUANTITATIVE_BRANCH,
    CONTINUATION,
    PROXY,
    DEMO,
    DEBUG,
    BRANCH_INSTANTIATION_SANITY,
}


def checked_claim_level(value: str) -> str:
    level = str(value)
    if level not in CLAIM_LEVELS:
        raise ValueError(f"unknown OPH claim level: {value}")
    return level


def with_claim_metadata(
    report: dict[str, Any],
    *,
    claim_level: str,
    receipt: str | None = None,
    physical_claim: bool | None = None,
) -> dict[str, Any]:
    result = dict(report)
    result["claim_level"] = checked_claim_level(claim_level)
    if receipt is not None:
        result["receipt_name"] = str(receipt)
    if physical_claim is not None:
        result["physical_claim"] = bool(physical_claim)
    return result
