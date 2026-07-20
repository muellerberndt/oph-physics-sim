"""Recursive presentation/target firewall for source-separated producers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import re
from typing import Any

from ._canonical import FrozenMap, canonical_hash, freeze_value, require_sha256


PRESENTATION_ONLY_FIELDS = frozenset(
    {
        "cache_state",
        "carrier_local_coordinates",
        "diagnostic_metadata",
        "executor_id",
        "local_coordinates",
        "local_port_names",
        "memory_address",
        "object_memory_address",
        "port_coordinates",
        "port_names",
        "presentation_state",
        "process_id",
        "process_ids",
        "queue_index",
        "queue_position",
        "queue_state",
        "retry_count",
        "retry_counter",
        "rng_state",
        "scheduler_state",
        "storage_address",
        "trace_uuid",
        "wall_clock",
        "wall_clock_ns",
        "worker_assignment",
        "worker_id",
        "worker_ownership",
        "worker_state",
    }
)

DOWNSTREAM_TARGET_FIELDS = frozenset(
    {
        "acceptance_threshold",
        "candidate_clock_scale",
        "candidate_geometry",
        "candidate_kms_normalization",
        "candidate_model",
        "comparison_data",
        "comparison_residual",
        "desired_branch",
        "expected_normalization",
        "fit_output",
        "fit_result",
        "measured_target",
        "observed_cosmology",
        "pass_threshold",
        "preferred_geometry",
        "preferred_kms_normalization",
        "target_beta",
        "target_coupling",
        "target_geometry",
        "target_scale",
        "target_signature",
        "target_vacuum",
    }
)

_PRESENTATION_PREFIXES = (
    "cache_",
    "executor_",
    "memory_",
    "queue_",
    "retry_",
    "rng_",
    "scheduler_",
    "wall_clock_",
    "worker_",
)
_TARGET_PREFIXES = (
    "candidate_",
    "desired_",
    "expected_",
    "observed_",
    "preferred_",
    "target_",
)


class SourceFirewallViolation(ValueError):
    """A source packet contains presentation or downstream target data."""


@dataclass(frozen=True, slots=True)
class FirewallFinding:
    path: str
    category: str
    field: str

    def to_jsonable(self) -> dict[str, str]:
        return {"path": self.path, "category": self.category, "field": self.field}


@dataclass(frozen=True, slots=True)
class SourceFirewallReport:
    packet_hash: str
    findings: tuple[FirewallFinding, ...]
    allowed_top_level_fields: tuple[str, ...]
    verifier_version: str = "oph-source-firewall-v1"

    def __post_init__(self) -> None:
        require_sha256(self.packet_hash, field_name="packet_hash")
        object.__setattr__(
            self,
            "findings",
            tuple(
                sorted(
                    set(self.findings),
                    key=lambda row: (row.path, row.category, row.field),
                )
            ),
        )
        object.__setattr__(
            self,
            "allowed_top_level_fields",
            tuple(sorted(set(self.allowed_top_level_fields))),
        )

    @property
    def passed(self) -> bool:
        return not self.findings

    @property
    def physical_promotion_receipt(self) -> bool:
        """A firewall audit is necessary evidence, never a physical result."""

        return False

    @property
    def report_hash(self) -> str:
        return canonical_hash(
            self.to_jsonable(), domain="oph.source-firewall-report.v1"
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "oph.source-firewall-report.v1",
            "packet_hash": self.packet_hash,
            "verifier_version": self.verifier_version,
            "passed": self.passed,
            "findings": [finding.to_jsonable() for finding in self.findings],
            "allowed_top_level_fields": list(self.allowed_top_level_fields),
            "physical_promotion_receipt": False,
        }


def normalize_field_name(value: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower()


def classify_forbidden_field(field_name: str) -> str | None:
    normalized = normalize_field_name(field_name)
    if normalized in PRESENTATION_ONLY_FIELDS or normalized.endswith("_uuid"):
        return "HIDDEN_PRESENTATION"
    if normalized.startswith(_PRESENTATION_PREFIXES):
        return "HIDDEN_PRESENTATION"
    if normalized in DOWNSTREAM_TARGET_FIELDS or normalized.startswith(
        _TARGET_PREFIXES
    ):
        return "DOWNSTREAM_TARGET"
    return None


def _object_marker(value: Any) -> str | None:
    module = type(value).__module__
    name = type(value).__name__
    if module == "oph_fpe.ontology.states" and name == "PresentationState":
        return "PRESENTATION_STATE_OBJECT"
    if module == "oph_fpe.ontology.records" and name == "ExecutionLogEntry":
        return "EXECUTION_LOG_OBJECT"
    if module == "oph_fpe.ontology.clocks" and name == "ExecutionClockReading":
        return "EXECUTION_CLOCK_OBJECT"
    return None


def _walk_source(
    value: Any,
    *,
    path: str,
    findings: list[FirewallFinding],
) -> Any:
    marker = _object_marker(value)
    if marker is not None:
        findings.append(
            FirewallFinding(path=path, category="HIDDEN_PRESENTATION", field=marker)
        )

    if isinstance(value, FrozenMap):
        value = dict(value.items_tuple)
    if isinstance(value, Mapping):
        canonical: dict[str, Any] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                findings.append(
                    FirewallFinding(
                        path=path,
                        category="UNSUPPORTED_SOURCE_VALUE",
                        field="non_string_or_empty_key",
                    )
                )
                continue
            key = normalize_field_name(raw_key)
            child_path = f"{path}.{raw_key}"
            category = classify_forbidden_field(raw_key)
            if category is not None:
                findings.append(
                    FirewallFinding(path=child_path, category=category, field=key)
                )
            canonical[raw_key] = _walk_source(child, path=child_path, findings=findings)
        return canonical
    if is_dataclass(value) and not isinstance(value, type):
        canonical = {}
        for field in fields(value):
            child_path = f"{path}.{field.name}"
            category = classify_forbidden_field(field.name)
            if category is not None:
                findings.append(
                    FirewallFinding(
                        path=child_path,
                        category=category,
                        field=normalize_field_name(field.name),
                    )
                )
            canonical[field.name] = _walk_source(
                getattr(value, field.name), path=child_path, findings=findings
            )
        return canonical
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _walk_source(child, path=f"{path}[{index}]", findings=findings)
            for index, child in enumerate(value)
        ]
    if isinstance(value, str):
        category = classify_forbidden_field(value)
        if category is not None:
            findings.append(
                FirewallFinding(
                    path=path,
                    category=f"{category}_SEMANTIC_VALUE",
                    field=normalize_field_name(value),
                )
            )
        return value
    try:
        return freeze_value(value)
    except (TypeError, ValueError):
        findings.append(
            FirewallFinding(
                path=path,
                category="UNSUPPORTED_SOURCE_VALUE",
                field=f"{type(value).__module__}.{type(value).__name__}",
            )
        )
        return {"unsupported_type": f"{type(value).__module__}.{type(value).__name__}"}


def audit_source_packet(
    packet: Any,
    *,
    allowed_top_level_fields: frozenset[str] | None = None,
) -> SourceFirewallReport:
    findings: list[FirewallFinding] = []
    canonical = _walk_source(packet, path="$", findings=findings)
    if allowed_top_level_fields is not None:
        allowed = frozenset(
            normalize_field_name(name) for name in allowed_top_level_fields
        )
        if not isinstance(packet, Mapping):
            findings.append(
                FirewallFinding(
                    path="$",
                    category="SOURCE_SCHEMA",
                    field="top_level_packet_is_not_a_mapping",
                )
            )
        else:
            for key in packet:
                normalized = normalize_field_name(str(key))
                if normalized not in allowed:
                    findings.append(
                        FirewallFinding(
                            path=f"$.{key}",
                            category="SOURCE_SCHEMA_EXTRA_FIELD",
                            field=normalized,
                        )
                    )
    else:
        allowed = frozenset()
    return SourceFirewallReport(
        packet_hash=canonical_hash(canonical, domain="oph.source-packet.v1"),
        findings=tuple(findings),
        allowed_top_level_fields=tuple(allowed),
    )


def require_source_packet_safe(
    packet: Any,
    *,
    allowed_top_level_fields: frozenset[str] | None = None,
) -> SourceFirewallReport:
    report = audit_source_packet(
        packet, allowed_top_level_fields=allowed_top_level_fields
    )
    if not report.passed:
        detail = ", ".join(f"{row.category}:{row.path}" for row in report.findings)
        raise SourceFirewallViolation(detail)
    return report


def require_no_presentation_fields(value: Any, *, context: str) -> None:
    report = audit_source_packet(value)
    presentation = tuple(
        finding
        for finding in report.findings
        if finding.category.startswith("HIDDEN_PRESENTATION")
    )
    if presentation:
        detail = ", ".join(finding.path for finding in presentation)
        raise SourceFirewallViolation(
            f"{context} contains presentation-only fields: {detail}"
        )
