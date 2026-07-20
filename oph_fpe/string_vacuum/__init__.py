"""Fail-closed string-vacuum evidence contracts."""

from .contract import (
    verify_candidate_evidence,
    verify_catalogue_evidence,
)
from .receipt_targets import observable_target_registry, receipt_target_registry

__all__ = [
    "observable_target_registry",
    "receipt_target_registry",
    "verify_candidate_evidence",
    "verify_catalogue_evidence",
]
