from __future__ import annotations

from typing import Any

from oph_fpe.claims import RECOVERED_CORE, with_claim_metadata


def boundary_conditioned_uniqueness_receipt(
    *,
    boundary_map_preserved: bool,
    sector_map_preserved: bool,
    consistent_extension_count: int,
    checked_states: int | None = None,
) -> dict[str, Any]:
    extension_count = int(consistent_extension_count)
    receipt = bool(boundary_map_preserved) and bool(sector_map_preserved) and extension_count == 1
    report = {
        "mode": "boundary_conditioned_fiber_uniqueness",
        "BOUNDARY_CONDITIONED_UNIQUENESS_RECEIPT": receipt,
        "receipt": receipt,
        "boundary_map_preserved": bool(boundary_map_preserved),
        "sector_map_preserved": bool(sector_map_preserved),
        "consistent_extension_count": extension_count,
        "checked_states": None if checked_states is None else int(checked_states),
        "claim_boundary": (
            "finite same-boundary fiber check: uniqueness is certified only inside the declared "
            "preserved boundary/sector map, not for arbitrary overlap nets"
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=RECOVERED_CORE,
        receipt="BOUNDARY_CONDITIONED_UNIQUENESS_RECEIPT",
    )
