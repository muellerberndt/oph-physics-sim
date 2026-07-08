from __future__ import annotations

from dataclasses import dataclass

from .receipts import fail, pass_report
from .topological_ledger import TopologicalLedger


@dataclass(frozen=True)
class PhaseCertificate:
    source_hamiltonian_frozen: bool
    active_band_projector: bool
    chern_number: bool
    band_geometry: bool
    manybody_gap: bool
    ground_sector_degeneracy: bool
    flux_insertion_pump: bool
    hall_conductance: bool
    edge_spectrum: bool
    topological_sector_ledger: bool
    refinement_stability: bool
    no_target_leak: bool
    certificate_map_injective: bool

    @property
    def receipts(self) -> dict[str, bool]:
        return {
            "SOURCE_HAMILTONIAN_FROZEN": self.source_hamiltonian_frozen,
            "ACTIVE_BAND_PROJECTOR": self.active_band_projector,
            "CHERN_NUMBER": self.chern_number,
            "BAND_GEOMETRY": self.band_geometry,
            "MANYBODY_GAP": self.manybody_gap,
            "GROUND_SECTOR_DEGENERACY": self.ground_sector_degeneracy,
            "FLUX_INSERTION_PUMP": self.flux_insertion_pump,
            "HALL_CONDUCTANCE": self.hall_conductance,
            "EDGE_SPECTRUM": self.edge_spectrum,
            "TOPOLOGICAL_SECTOR_LEDGER": self.topological_sector_ledger,
            "REFINEMENT_STABILITY": self.refinement_stability,
            "NO_TARGET_LEAK": self.no_target_leak,
        }

    @property
    def passed(self) -> bool:
        return all(self.receipts.values())


def promote_hamiltonian_to_ledger(cert: PhaseCertificate, ledger: TopologicalLedger) -> dict:
    if not cert.source_hamiltonian_frozen:
        return fail("SOURCE_NOT_FROZEN", details={"receipts": cert.receipts})
    if not cert.manybody_gap:
        return fail("NO_GAP_CERTIFICATE", details={"receipts": cert.receipts})
    if not cert.chern_number:
        return fail("CHERN_NUMBER_UNSTABLE", details={"receipts": cert.receipts})
    if not cert.no_target_leak:
        return fail("TARGET_LEAK_DETECTED", details={"receipts": cert.receipts})
    if not cert.certificate_map_injective:
        return fail("PHASE_CERTIFICATE_NONINJECTIVE", details={"receipts": cert.receipts})
    if not cert.passed:
        missing = [name for name, value in cert.receipts.items() if not value]
        return fail("SECTOR_AMBIGUOUS", details={"missing_receipts": missing})
    return pass_report(
        receipts=cert.receipts | {"PHASE_CERTIFICATE_INJECTIVE": True},
        details={"ledger": ledger.to_report()},
    )
