from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import Any


class CurvatureClaimStatus(str, Enum):
    OPEN_THEOREM = "OPEN_THEOREM"
    EXPLICIT_ASSUMPTION = "EXPLICIT_ASSUMPTION"
    EXPLICIT_SECTOR_ASSUMPTION = "EXPLICIT_ASSUMPTION"
    CONDITIONAL_CMH = "CONDITIONAL_CMH"
    DIRECT_THEOREM = "DIRECT_THEOREM"
    APPROXIMATE_BOUND = "APPROXIMATE_BOUND"


class GeometryBranch(str, Enum):
    FLAT_EXACT = "FLAT_EXACT"
    FLAT_ASSUMED = "FLAT_ASSUMED"
    OPEN_CURVED = "OPEN_CURVED"
    CLOSED_CURVED = "CLOSED_CURVED"
    UNRESOLVED = "UNRESOLVED"


class DensitySource(str, Enum):
    INDEPENDENT_OPH_SOURCE = "INDEPENDENT_OPH_SOURCE"
    GEOMETRIC_ESTIMATE = "GEOMETRIC_ESTIMATE"
    FRIEDMANN_ESTIMATE = "FRIEDMANN_ESTIMATE"
    EXTERNAL_INPUT = "EXTERNAL_INPUT"
    RESIDUAL_DEFINITION = "RESIDUAL_DEFINITION"
    EXPLICIT_ASSUMPTION = "EXPLICIT_ASSUMPTION"
    UNKNOWN = "UNKNOWN"


class PhysicalPromotionBlocked(RuntimeError):
    """Raised when a physical projection is requested before theorem gates pass."""


@dataclass(frozen=True)
class SpatialCurvatureStatus:
    status: CurvatureClaimStatus = CurvatureClaimStatus.OPEN_THEOREM
    geometry_branch: GeometryBranch = GeometryBranch.UNRESOLVED
    kappa: int | None = None
    K_release_interval: tuple[float, float] | None = None
    Omega_K_release_interval: tuple[float, float] | None = None
    topology_policy: str = "UNDECLARED"
    claim_basis: str = "NO_FLAT_SELECTOR_RECEIPT"
    clock_hash: str | None = None
    boundary_packet_hash: str | None = None
    theorem_hash: str | None = None
    proof_certificate_hash: str | None = None
    blockers: tuple[str, ...] = ("SPATIAL_CURVATURE_SELECTION_OPEN",)

    def as_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["geometry_branch"] = self.geometry_branch.value
        data["exact_flat_sector_selection"] = bool(
            self.status in {CurvatureClaimStatus.CONDITIONAL_CMH, CurvatureClaimStatus.DIRECT_THEOREM}
            and self.geometry_branch is GeometryBranch.FLAT_EXACT
            and self.kappa == 0
            and not self.blockers
        )
        data["flat_branch_assumption"] = bool(self.geometry_branch is GeometryBranch.FLAT_ASSUMED)
        data["curvature_suppression_bound_receipt"] = bool(
            self.status is CurvatureClaimStatus.APPROXIMATE_BOUND and self.Omega_K_release_interval is not None
        )
        data["selected_curvature_holonomy"] = None
        data["selected_K"] = (
            0.0
            if self.geometry_branch in {GeometryBranch.FLAT_EXACT, GeometryBranch.FLAT_ASSUMED}
            and self.K_release_interval == (0.0, 0.0)
            else None
        )
        data["selected_Omega_K"] = (
            0.0
            if self.geometry_branch in {GeometryBranch.FLAT_EXACT, GeometryBranch.FLAT_ASSUMED}
            and self.Omega_K_release_interval == (0.0, 0.0)
            else None
        )
        return data


def anomaly_curvature_budget(
    *,
    omega_lambda_oph: float,
    omega_b0: float,
    omega_nu0: float,
    omega_r0: float,
    omega_k0: float | None = None,
    omega_k_source: str | DensitySource = DensitySource.UNKNOWN,
    omega_a0: float | None = None,
    omega_a_source: str | DensitySource = DensitySource.UNKNOWN,
) -> dict[str, Any]:
    """Return the FLRW closure degeneracy without assuming Omega_K=0.

    The homogeneous closure relation determines Omega_A0 + Omega_K0 unless
    one of those two terms is supplied by an independent source. Treating the
    residual as Omega_A0 is therefore an explicit flat-branch assumption, not
    a curvature proof.
    """

    residual = 1.0 - float(omega_lambda_oph) - float(omega_b0) - float(omega_nu0) - float(omega_r0)
    omega_k_source_value = _source_value(omega_k_source)
    omega_a_source_value = _source_value(omega_a_source)
    derived_omega_a0 = omega_a0
    derived_omega_a_source = omega_a_source_value
    if derived_omega_a0 is None and omega_k0 is not None:
        derived_omega_a0 = residual - float(omega_k0)
        derived_omega_a_source = DensitySource.RESIDUAL_DEFINITION.value
    independent_sources = {
        DensitySource.INDEPENDENT_OPH_SOURCE.value,
        DensitySource.GEOMETRIC_ESTIMATE.value,
        DensitySource.FRIEDMANN_ESTIMATE.value,
        DensitySource.EXTERNAL_INPUT.value,
    }
    return {
        "Omega_A0_plus_Omega_K0": float(residual),
        "Omega_A0": float(derived_omega_a0) if derived_omega_a0 is not None else None,
        "Omega_K0": float(omega_k0) if omega_k0 is not None else None,
        "Omega_A_source": derived_omega_a_source,
        "Omega_K_source": omega_k_source_value,
        "Omega_A0_residual": None,
        "rho_A_over_rho_b": (
            float(derived_omega_a0 / omega_b0)
            if derived_omega_a0 is not None and float(omega_b0) != 0.0
            else None
        ),
        "anomaly_curvature_degeneracy": omega_k0 is None,
        "friedmann_curvature_independence_receipt": bool(
            omega_k_source_value in independent_sources and derived_omega_a_source in independent_sources
        ),
        "claim_boundary": (
            "FLRW closure gives Omega_A0+Omega_K0 unless curvature or anomaly density is read "
            "independently. A residual Omega_A0 value alone cannot prove flatness."
        ),
    }


def spatial_curvature_status_report(
    *,
    omega_lambda_oph: float = 0.684095,
    omega_b0: float = 0.04931,
    omega_nu0: float = 0.00230,
    omega_r0: float = 9.2e-5,
    omega_k0: float | None = None,
    omega_k_source: str | DensitySource = DensitySource.UNKNOWN,
    omega_a0: float | None = None,
    omega_a_source: str | DensitySource = DensitySource.UNKNOWN,
    geometry_branch: str | GeometryBranch = GeometryBranch.UNRESOLVED,
    topology_policy: str = "LOCAL_TOPOLOGY_UNRESOLVED",
    clock_hash: str | None = None,
    boundary_packet_hash: str | None = None,
    theorem_hash: str | None = None,
    proof_certificate_hash: str | None = None,
    conditional_cmh: bool = False,
    direct_theorem: bool = False,
    flat_extension_exists: bool = False,
    quotient_refinement_naturality: bool = False,
    curvature_functional_certified: bool = False,
    zero_set_unique: bool = False,
    approximate_bound_certified: bool = False,
    K_release_interval: tuple[float, float] | list[float] | None = None,
    Omega_K_release_interval: tuple[float, float] | list[float] | None = None,
    manual_exact_flat_sector_selection: bool | None = None,
) -> dict[str, Any]:
    branch = _branch_value(geometry_branch)
    k_interval = _interval_or_none(K_release_interval)
    omega_k_interval = _interval_or_none(Omega_K_release_interval)
    hashes_present = bool(clock_hash and boundary_packet_hash and theorem_hash and proof_certificate_hash)
    blockers: list[str] = []
    claim_basis = "NO_FLAT_SELECTOR_RECEIPT"
    status = CurvatureClaimStatus.OPEN_THEOREM
    kappa: int | None = None

    if manual_exact_flat_sector_selection:
        blockers.append("MANUAL_EXACT_FLAT_BOOLEAN_REJECTED")

    if branch is GeometryBranch.FLAT_ASSUMED:
        status = CurvatureClaimStatus.EXPLICIT_ASSUMPTION
        kappa = 0
        k_interval = (0.0, 0.0)
        omega_k_interval = (0.0, 0.0)
        claim_basis = "EXPLICIT_FLAT_BRANCH_ASSUMPTION"
        blockers.append("FLATNESS_ASSUMED_NOT_DERIVED")
        omega_k0 = 0.0 if omega_k0 is None else omega_k0
        omega_k_source = DensitySource.EXPLICIT_ASSUMPTION
    elif direct_theorem:
        required = {
            "clock_boundary_theorem_proof_hashes": hashes_present,
            "curvature_functional_certified": curvature_functional_certified,
            "zero_set_unique": zero_set_unique,
        }
        blockers.extend(name for name, passed in required.items() if not passed)
        if not blockers:
            status = CurvatureClaimStatus.DIRECT_THEOREM
            branch = GeometryBranch.FLAT_EXACT
            kappa = 0
            k_interval = (0.0, 0.0)
            omega_k_interval = (0.0, 0.0)
            claim_basis = "DIRECT_OPH_NATIVE_FLATNESS_THEOREM"
    elif conditional_cmh:
        required = {
            "clock_boundary_theorem_proof_hashes": hashes_present,
            "flat_extension_exists": flat_extension_exists,
            "quotient_refinement_naturality": quotient_refinement_naturality,
            "curvature_functional_certified": curvature_functional_certified,
            "zero_set_unique_under_topology_policy": zero_set_unique,
        }
        blockers.extend(name for name, passed in required.items() if not passed)
        if not blockers:
            status = CurvatureClaimStatus.CONDITIONAL_CMH
            branch = GeometryBranch.FLAT_EXACT
            kappa = 0
            k_interval = (0.0, 0.0)
            omega_k_interval = (0.0, 0.0)
            claim_basis = "CONDITIONAL_PHASE_III_COSMOLOGICAL_MINIMAL_HOLONOMY"
    elif approximate_bound_certified and omega_k_interval is not None:
        status = CurvatureClaimStatus.APPROXIMATE_BOUND
        claim_basis = "CURVATURE_SUPPRESSION_BOUND"
        blockers.append("APPROXIMATE_BOUND_DOES_NOT_SELECT_KAPPA")
    else:
        blockers.append("CMH_OR_DIRECT_FLATNESS_THEOREM_MISSING")

    curvature_status = SpatialCurvatureStatus(
        status=status,
        geometry_branch=branch,
        kappa=kappa,
        K_release_interval=k_interval,
        Omega_K_release_interval=omega_k_interval,
        topology_policy=str(topology_policy),
        claim_basis=claim_basis,
        clock_hash=clock_hash,
        boundary_packet_hash=boundary_packet_hash,
        theorem_hash=theorem_hash,
        proof_certificate_hash=proof_certificate_hash,
        blockers=tuple(dict.fromkeys(blockers)),
    ).as_jsonable()
    budget = anomaly_curvature_budget(
        omega_lambda_oph=omega_lambda_oph,
        omega_b0=omega_b0,
        omega_nu0=omega_nu0,
        omega_r0=omega_r0,
        omega_k0=omega_k0,
        omega_k_source=omega_k_source,
        omega_a0=omega_a0,
        omega_a_source=omega_a_source,
    )
    return {
        "mode": "oph_spatial_curvature_status_v0",
        **curvature_status,
        "inputs": {
            "Omega_Lambda_OPH": float(omega_lambda_oph),
            "Omega_b0": float(omega_b0),
            "Omega_nu0": float(omega_nu0),
            "Omega_r0": float(omega_r0),
        },
        **budget,
        "selector_statement": (
            "Zero clock-slice spatial Levi-Civita holonomy identifies the flat FLRW branch. "
            "Selection of that branch is a separate direct theorem, conditional CMH theorem, "
            "or explicit assumption; MAR and S3 screen defects do not by themselves select Omega_K=0."
        ),
        "claim_boundary": (
            "Fail-closed curvature report. The default state is OPEN_THEOREM/UNRESOLVED; exact "
            "flatness is emitted only with direct or conditional-CMH proof hashes, while explicit "
            "flat inputs remain assumption-conditioned."
        ),
    }


def s3_holonomy_spatial_curvature_gate(report: dict[str, Any] | None = None) -> dict[str, Any]:
    report = report or {}
    structure_group = str(report.get("structure_group") or report.get("group") or "S3")
    connection = str(report.get("connection_type") or report.get("geometric_connection") or "screen_permutation")
    passed = bool(structure_group == "SO3_SPATIAL_LEVI_CIVITA" and connection == "spatial_levi_civita")
    return {
        "S3_SCREEN_DEFECT_IS_SPATIAL_CURVATURE_RECEIPT": passed,
        "structure_group": structure_group,
        "geometric_connection": passed,
        "spatial_levi_civita_interpretation": passed,
        "blockers": [] if passed else ["S3_SCREEN_DEFECT_NOT_SPATIAL_LEVI_CIVITA_HOLONOMY"],
    }


def friedmann_curvature_readout(
    *,
    H: float,
    rho_total: float,
    G: float,
    Lambda: float,
    ancestry_labels: list[str] | tuple[str, ...] = (),
    unit_consistency_receipt: bool = False,
    clock_epoch_alignment_receipt: bool = False,
) -> dict[str, Any]:
    forbidden = {"FRIEDMANN_RESIDUAL", "ASSUMED_KAPPA_ZERO", "OMEGA_A_CLOSURE_RESIDUAL"}
    labels = {str(label) for label in ancestry_labels}
    finite_inputs = all(isfinite(float(value)) for value in (H, rho_total, G, Lambda)) and float(H) != 0.0
    K_f = (8.0 * 3.141592653589793 * float(G) / 3.0) * float(rho_total) + float(Lambda) / 3.0 - float(H) ** 2
    omega_k_f = -K_f / (float(H) ** 2) if finite_inputs else None
    blockers = []
    if not finite_inputs:
        blockers.append("NONFINITE_FRIEDMANN_INPUT")
    if labels & forbidden:
        blockers.append("CIRCULAR_FRIEDMANN_ANCESTRY")
    if not unit_consistency_receipt:
        blockers.append("UNIT_CONSISTENCY_RECEIPT_MISSING")
    if not clock_epoch_alignment_receipt:
        blockers.append("CLOCK_EPOCH_ALIGNMENT_RECEIPT_MISSING")
    return {
        "K_F": float(K_f) if finite_inputs else None,
        "Omega_K_F": float(omega_k_f) if omega_k_f is not None else None,
        "friedmann_curvature_readout_receipt": not blockers,
        "blockers": blockers,
    }


def _source_value(source: str | DensitySource) -> str:
    return source.value if isinstance(source, DensitySource) else str(source)


def _branch_value(branch: str | GeometryBranch) -> GeometryBranch:
    if isinstance(branch, GeometryBranch):
        return branch
    return GeometryBranch(str(branch))


def _interval_or_none(values: tuple[float, float] | list[float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("intervals must have two endpoints")
    lo = float(values[0])
    hi = float(values[1])
    if not (isfinite(lo) and isfinite(hi)):
        raise ValueError("interval endpoints must be finite")
    return (lo, hi) if lo <= hi else (hi, lo)
