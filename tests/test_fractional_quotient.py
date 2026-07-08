from fractions import Fraction

from oph_fractional.active_band import ChernBandCertificate, chern_integer_stability
from oph_fractional.compare import compare_after_freeze, demo_fractional_report
from oph_fractional.hamiltonian import PhaseCertificate, promote_hamiltonian_to_ledger
from oph_fractional.identifiability import identify_optical_sector
from oph_fractional.line_fan import LineFanPeak, fractional_optical_slope_certificate, line_fan_from_module
from oph_fractional.manybody import ManyBodyCertificate, manybody_gap_certificate
from oph_fractional.no_target_leak import no_target_leak_audit
from oph_fractional.optical_module import OpticalModuleLedger, OpticalSector
from oph_fractional.presentation import FractionalMaterialPresentation
from oph_fractional.quotient import (
    QuotientSchema,
    canonicalizer_idempotence,
    no_orbit_size_bias,
    quotient_lumpability,
    representative_invariance,
)
from oph_fractional.refinement import refinement_compatibility
from oph_fractional.report import fractional_quotient_report, write_fractional_quotient_bundle
from oph_fractional.source_law import SourceLaw, normal_form_non_selection
from oph_fractional.topological_ledger import TopologicalLedger, abelian_k_matrix_readout
from oph_fpe.measurement_pack import export_measurement_pack
from oph_fpe.physics_problem_outputs import physics_problem_outputs_report


def test_material_presentation_exposes_quotient_normal_form_receipt() -> None:
    presentation = FractionalMaterialPresentation(
        material_id="twisted_tmd_sandbox",
        regulator=12,
        representatives=("a0", "a1", "vac"),
        quotient_map={"a0": "anyon", "a1": "anyon", "vac": "vacuum"},
    )

    assert presentation.quotient_sectors == ("anyon", "vacuum")
    assert presentation.to_report()["material_quotient_normal_form_receipt"] is True


def test_quotient_receipts_detect_idempotence_invariance_and_lumpability() -> None:
    schema = QuotientSchema(
        canonical={"a0": "anyon", "a1": "anyon", "vac": "vacuum"},
        transition_kernel={
            "a0": {"a0": 0.25, "a1": 0.25, "vac": 0.5},
            "a1": {"a0": 0.5, "vac": 0.5},
            "vac": {"vac": 1.0},
        },
    )

    assert canonicalizer_idempotence(schema)["receipts"]["CANONICALIZER_IDEMPOTENCE"] is True
    assert representative_invariance(schema, lambda state: schema.canonicalize(state))["status"] == "pass"
    assert quotient_lumpability(schema)["receipts"]["QUOTIENT_LUMPABILITY"] is True


def test_bad_kernel_and_bad_observable_fail_closed() -> None:
    schema = QuotientSchema(
        canonical={"a0": "anyon", "a1": "anyon", "vac": "vacuum"},
        transition_kernel={"a0": {"a0": 1.0}, "a1": {"vac": 1.0}, "vac": {"vac": 1.0}},
    )

    invariant = representative_invariance(schema, lambda state: state)
    lumpable = quotient_lumpability(schema)

    assert invariant["fail_closed_state"] == "NOT_QUOTIENT_INVARIANT"
    assert lumpable["fail_closed_state"] == "KERNEL_NOT_LUMPABLE"


def test_source_law_must_be_frozen_before_comparison() -> None:
    source = SourceLaw(action={"sector": 0.0}, frozen=False)

    failed = compare_after_freeze(source, {"line": 1.0}, {"line": 1.1})
    passed = compare_after_freeze(source.freeze(), {"line": 1.0}, {"line": 1.1})

    assert failed["fail_closed_state"] == "SOURCE_NOT_FROZEN"
    assert passed["SOURCE_HAMILTONIAN_FROZEN"] is True
    assert normal_form_non_selection(["pf", "apf"])["NORMAL_FORM_IS_NOT_SELECTOR"] is True


def test_chern_gap_and_phase_promotion_receipts() -> None:
    band = chern_integer_stability(
        ChernBandCertificate(
            active_band_projector=True,
            chern_number=1,
            chern_stable=True,
            band_geometry_bound=0.2,
        )
    )
    gap = manybody_gap_certificate(
        ManyBodyCertificate(
            gap=0.04,
            ground_sector_degeneracy=3,
            expected_degeneracy=3,
            flux_pump_charge=1 / 3,
            hall_conductance=1 / 3,
        )
    )
    ledger = TopologicalLedger(
        sectors=("vacuum", "anyon"),
        charges={"vacuum": Fraction(0), "anyon": Fraction(1, 3)},
        edge_spectrum_certified=True,
    )
    cert = PhaseCertificate(
        source_hamiltonian_frozen=True,
        active_band_projector=True,
        chern_number=True,
        band_geometry=True,
        manybody_gap=True,
        ground_sector_degeneracy=True,
        flux_insertion_pump=True,
        hall_conductance=True,
        edge_spectrum=True,
        topological_sector_ledger=True,
        refinement_stability=True,
        no_target_leak=True,
        certificate_map_injective=True,
    )

    promoted = promote_hamiltonian_to_ledger(cert, ledger)

    assert band["receipts"]["CHERN_NUMBER"] is True
    assert gap["receipts"]["MANYBODY_GAP"] is True
    assert promoted["receipts"]["TOPOLOGICAL_SECTOR_LEDGER"] is True


def test_noninjective_phase_certificate_fails_closed() -> None:
    ledger = TopologicalLedger(sectors=("a", "b"), charges={"a": 0, "b": 0})
    cert = PhaseCertificate(
        source_hamiltonian_frozen=True,
        active_band_projector=True,
        chern_number=True,
        band_geometry=True,
        manybody_gap=True,
        ground_sector_degeneracy=True,
        flux_insertion_pump=True,
        hall_conductance=True,
        edge_spectrum=True,
        topological_sector_ledger=True,
        refinement_stability=True,
        no_target_leak=True,
        certificate_map_injective=False,
    )

    assert promote_hamiltonian_to_ledger(cert, ledger)["fail_closed_state"] == "PHASE_CERTIFICATE_NONINJECTIVE"


def test_k_matrix_readout_recovers_laughlin_charge() -> None:
    report = abelian_k_matrix_readout(((3,),), (1,), (1,))

    assert report["filling"] == "1/3"
    assert report["quasiparticle_charge_over_e"] == "1/3"


def test_optical_line_fan_neutral_fractional_exciton_boundary() -> None:
    module = OpticalModuleLedger(
        quotient_descended_operators=True,
        sectors=(
            OpticalSector(
                label="neutral_fractional_exciton",
                tau="anyon",
                total_charge=0.0,
                energy=1.6,
                intensity=0.5,
                polarization="sigma+",
                binding_derivative_bound=0.02,
            ),
        ),
    )
    fan = line_fan_from_module(module)
    peak = LineFanPeak(**fan["peaks"][0])
    slope = fractional_optical_slope_certificate(peak, binding_derivative_bound=0.02)

    assert fan["LINE_FAN_DECOMPOSITION"] is True
    assert slope["details"]["neutral_fractional_shadow"] is True


def test_optical_identifiability_and_binding_drift_fail_closed() -> None:
    peak_a = LineFanPeak("a", 1.0, 0.5, 0.0, "x", "anyon", 0.0)
    peak_b = LineFanPeak("b", 1.0, 0.5, 0.0, "x", "anyon", 0.0)

    assert identify_optical_sector([peak_a, peak_b])["fail_closed_state"] == "OPTICAL_SECTOR_AMBIGUOUS"
    assert fractional_optical_slope_certificate(peak_a, binding_derivative_bound=None)["fail_closed_state"] == "BINDING_DRIFT_UNBOUNDED"


def test_no_target_leak_and_refinement_receipts() -> None:
    clean = no_target_leak_audit(
        source_nodes=["source"],
        target_nodes=["measurement"],
        edges=[("source", "prediction")],
    )
    leaky = no_target_leak_audit(
        source_nodes=["source"],
        target_nodes=["measurement"],
        edges=[("measurement", "source")],
    )
    refinement = refinement_compatibility(0.01, max_defect=0.02)

    assert clean["receipts"]["NO_TARGET_LEAK"] is True
    assert leaky["fail_closed_state"] == "TARGET_LEAK_DETECTED"
    assert refinement["receipts"]["REFINEMENT_COMPATIBILITY"] is True


def test_orbit_size_bias_and_demo_report_are_explicit() -> None:
    schema = QuotientSchema(
        canonical={"a0": "anyon", "a1": "anyon", "vac": "vacuum"},
        orbit_sizes={"anyon": 2, "vacuum": 1},
    )

    no_bias = no_orbit_size_bias(schema, {"anyon": 0.5, "vacuum": 0.5})
    biased = no_orbit_size_bias(schema, {"anyon": 2.0, "vacuum": 1.0})
    report = demo_fractional_report()

    assert no_bias["receipts"]["NO_ORBIT_SIZE_BIAS"] is True
    assert biased["fail_closed_state"] == "ORBIT_SIZE_BIAS_DETECTED"
    assert report["schema"] == "oph_fractional_quotient_sim_report_v1"
    assert report["material_presentation"]["material_quotient_normal_form_receipt"] is True
    assert report["quotient"]["orbit_size_bias"]["receipts"]["NO_ORBIT_SIZE_BIAS"] is True
    assert report["identifiability"]["receipts"]["OPTICAL_LINE_FAN_INJECTIVE"] is True
    assert report["line_fan"]["LINE_FAN_DECOMPOSITION"] is True


def test_fractional_quotient_bundle_remains_diagnostic(tmp_path) -> None:
    report = fractional_quotient_report()
    gates = report["readiness_gates"]

    assert report["schema"] == "oph_fractional_quotient_report_v1"
    assert report["claim"] == "FRACTIONAL_QUOTIENT_SANDBOX_DIAGNOSTIC"
    assert report["first_blocked_gate"] == "MATERIAL_SPECIFIC_HAMILTONIAN_PROOF_RECEIPT"
    assert report["promotion_allowed"] is False
    assert report["material_claim"] is False
    assert gates["SIMULATOR_QUOTIENT_CORRECTNESS"] is True
    assert gates["NO_TARGET_LEAK_DAG"] is True
    assert gates["OPTICAL_LINE_FAN_INJECTIVE"] is True

    out_dir = tmp_path / "fractional"
    written = write_fractional_quotient_bundle(out_dir)
    assert written["claim"] == report["claim"]
    assert (out_dir / "fractional_quotient_report.json").is_file()
    assert (out_dir / "fractional_quotient_report.md").is_file()

    pack = export_measurement_pack([out_dir], tmp_path / "pack")
    claims = pack["claims"]
    assert claims["fractional_quotient_report_written"] is True
    assert claims["fractional_quotient_material_claim"] is False
    assert claims["fractional_quotient_simulator_correctness_receipt"] is True


def test_physics_problem_outputs_include_fractional_exciton_contract() -> None:
    outputs = physics_problem_outputs_report()["outputs"]
    fractional = outputs["fractional_exciton_quotient_sector"]

    assert fractional["computed"] is True
    assert fractional["claim"] == "FRACTIONAL_QUOTIENT_SANDBOX_DIAGNOSTIC"
    assert fractional["materialClaim"] is False
    assert fractional["receipts"]["SIMULATOR_QUOTIENT_CORRECTNESS_RECEIPT"] is True
