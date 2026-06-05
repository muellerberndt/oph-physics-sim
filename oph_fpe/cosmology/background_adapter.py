from __future__ import annotations

from typing import Any

from oph_fpe.claims import DEBUG, with_claim_metadata


def background_adapter_status() -> dict[str, Any]:
    report = {
        "mode": "background_adapter_status",
        "enabled": False,
        "receipt": False,
        "desi_dr2_bao_adapter": "not_implemented",
        "pantheon_plus_adapter": "not_implemented",
        "claim_boundary": (
            "placeholder status for future DESI/Pantheon+ phenomenological background adapters; "
            "no cosmological parameter inference is performed"
        ),
    }
    return with_claim_metadata(report, claim_level=DEBUG, receipt="BACKGROUND_ADAPTER_STATUS", physical_claim=False)
