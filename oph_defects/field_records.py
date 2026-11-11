"""Export audited finite field records, with exact bounds and explicit domains."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
from pathlib import Path

from .quantum_evidence import bounds, digest, receipt
from .reference_repository import theory_repository

OUTPUT = Path(__file__).parent / 'viewer2/field_records.json'
NAMES = ('native_quark_spatial_spectrum', 'native_code_ground_preparation',
         'native_orthogonal_wilson_bridge', 'native_wilson_maxwell_spatial_family',
         'native_wilson_maxwell_interaction_witness', 'native_orthogonal_wilson_spatial_join',
         'native_orthogonal_wilson_charged_band')
BASE = 'model/field_record_addenda/'
REVIEWED_RESEARCH = dict(zip(NAMES, (
    '4003848f71b6da04e54955ceffb4a81179b9c659fbff56d10661ee35f41613fc',
    '447b427480b303ed01cce3ac03fa56e0121c775e7cc17780dddba2976adce63e',
    '75590a8e62090717d5b91ada6ac472498952c3b4ace19b15de815419ecdadc6c',
    '3e413171f6ad1c0df6cc4db522cb67bc4d9f5f2ae1cfc80d3f4b92fac03538ed',
    'b8519b3f2464810d07b78ca90cad8f918007cdee6339f3f710ccefa4b26a8e87',
    '44ca365fd77c85e1d820d2c2563f42187576c025e551d186c9998ad2428c7bab',
    '310d8d6b5a7491fc0088e011e45dffe02efee90c9838f220f6c99a13e58e411a')))


def scalar(value):
    value = F(value)
    if value <= 0:
        raise ValueError('Display bound must be positive')
    exponent = len(str(value.numerator))-len(str(value.denominator))
    if value < F(10)**exponent:
        exponent -= 1
    shift = 7-exponent
    scaled = value*F(10)**shift
    lower = scaled.numerator//scaled.denominator
    upper = -(-scaled.numerator//scaled.denominator)
    return {'exact': str(value), 'display': float(value),
            'lowerText': f'{lower/10**7:.7f}e{exponent:+d}',
            'upperText': f'{upper/10**7:.7f}e{exponent:+d}'}


def interval(values):
    result = bounds(values)
    result['lowerText'] = scalar(values[0])['lowerText']
    result['upperText'] = scalar(values[1])['upperText']
    return result


def load(root, name):
    relative = BASE + name + '/results.json'
    value = receipt(root, relative)
    port = value['public_port']
    if not port['exact_scientific_payload_equal'] or port['physical_status_promotion']:
        raise ValueError('Unreviewed scientific interpretation: ' + name)
    retained = root / BASE / name / 'research_reference.json'
    expected = REVIEWED_RESEARCH[name]
    if digest(retained) != expected or port['retained_research_sha256'] != expected:
        raise ValueError('Reviewed research receipt changed: ' + name)
    original = json.loads(retained.read_text())
    def science(item):
        return {k: v for k, v in item.items() if k not in ('input_sha256', 'owned_source_sha256', 'public_port')}
    if science(value) != science(original):
        raise ValueError('Reviewed scientific payload changed: ' + name)
    for path, sha in value['owned_source_sha256'].items():
        target = (root / BASE / name / path).resolve()
        if not target.is_relative_to(root) or digest(target) != sha:
            raise ValueError('Changed public proof or producer: ' + path)
    return value


def build(repository=None):
    root = Path(repository or theory_repository()).resolve()
    values = {name: load(root, name) for name in NAMES}
    quark, prep, bridge, family, witness, revised, charged = (values[n] for n in NAMES)
    if (prep['source_changed'] or prep['boundaries']['physical_mass_or_particle_identity']
            or bridge['common_free_cone']['weak_chiral_Standard_Model_representation_constructed']
            or witness['actual_source_changed'] or witness['boundaries']['Lorentz_UV_or_particle_identity']
            or revised['boundaries']['relativistic_or_UV_limit_or_SM_mass']
            or charged['boundaries']['relativistic_rest_mass_or_SM_identity']
            or charged['exact_source_controls']['eight_states_identified_as_species_or_spin_multiplicity']
            or not charged['complete_instrument']['all_failure_and_unused_outcomes_retained']
            or revised['separate_source_only_completion_option']['actual_ground_vector_or_internal_gap_or_volume_limit_proved']):
        raise ValueError('Source identity or physical scope changed')
    rows = []
    for name in ('uc', 'dc', 'Q'):
        item = quark['families'][name]
        if item['autonomous_energy_gain_or_simultaneous_joint_PVM_claimed']:
            raise ValueError('Arrival backaction was removed')
        line = item['all_atomic_and_full_H_bins'][item['best_certified_bin_index']]
        rows.append({'id': name, 'sourceChargeY': {'uc': '+2/3', 'dc': '−1/3', 'Q': '−1/6'}[name],
            'energy': interval(line['source_excitation_relative_actual_neutral_ground']),
            'probability': interval(line['original_route_then_energy_probability']),
            'records': item['coarse_original_attempt_records'], 'atomicLines': item['retained_atomic_lines'],
            'detuningLower': scalar(item['minimum_input_output_detuning_lower']),
            'zeroControl': item['exact_autonomous_disjoint_energy_control']})
    smaller = prep['smaller_positive_record_setting']; full = prep['bounds_exact']
    preparation = [
        {'copies': 2, 'pointer': smaller['pointer_M'], 'filters': smaller['filters'], 'slots': 4,
         'success': scalar(smaller['original_success_lower']),
         'groundError': scalar(smaller['conditional_ground_purification_error_upper']),
         'recordProbability': scalar(smaller['original_preparation_and_T111_probability_lower']),
         'historyError': scalar(smaller['complete_two_field_history_error_upper']),
         'capacity': interval(smaller['all_sources_and_pointers_nats_interval'])},
        {'copies': 4, 'pointer': prep['new_apparatus_setting']['pointer_M'], 'filters': 3, 'slots': 6,
         'success': scalar(full['original_all_admissions_and_filters_success_lower']),
         'groundError': scalar(full['conditional_ground_purification_error_upper']),
         'recordProbability': scalar(full['original_preparation_and_T111_probability_lower']),
         'historyError': scalar(full['complete_preparation_and_two_field_history_error_upper']),
         'capacity': interval(prep['explicit_history_budget']['all_sources_references_and_record_pointers_nats_interval'])}]
    return {'schema': 'oph.viewer.field_records.v1', 'quark': rows, 'preparation': preparation,
        'cone': {'originalAxisRatio': 2, 'revisedAxisRatio': 1,
            'mass': interval(bridge['full_physical_ground']['source_selected_mass_interval']),
            'gapLower': scalar(bridge['full_physical_ground']['full_physical_ground_gap_source_units_lower']),
            'completion': bridge['energy_alignment']['robust_declared_positive_complement_coefficient'],
            'rawSeaShiftOverMass': interval(bridge['energy_alignment']['signed_code_vacuum_shift_over_m']),
            'kineticCompressionCoefficientExcessLower': scalar(revised['bounds_exact']['compressed_squared_curvature_minus_one_lower']),
            'kineticCompressionIsSelfEnergy': revised['bulk_controls']['compression_identified_with_full_self_energy_or_particle_mass']},
        'interaction': {'contrast': scalar(witness['bounds_exact']['matched_setting_probability_contrast_lower']),
            'completeLawSeparation': scalar(witness['bounds_exact']['finite_complete_law_TV_separation_lower']),
            'cutoffError': scalar(witness['bounds_exact']['complete_finite_cap_record_error_upper']),
            'capacity': interval(witness['finite_sufficient_example']['capacity']['all_sources_references_flags_setting_and_detector_nats_interval']),
            'cellsPerAxis': witness['finite_sufficient_example']['cells_per_axis'],
            'sourceChanged': False, 'matrixEnumerated': False},
        'revisedInteraction': {'contrast': scalar(revised['bounds_exact']['matched_setting_probability_contrast_lower']),
            'completeLawSeparation': scalar(revised['bounds_exact']['complete_finite_law_TV_separation_lower']),
            'cutoffError': scalar(witness['bounds_exact']['complete_finite_cap_record_error_upper']),
            'capacity': interval(revised['finite_witness']['capacity']['all_sources_references_flags_setting_and_detector_nats_interval']),
            'cellsPerAxis': revised['finite_witness']['cells_per_axis'],
            'sourceChanged': True, 'matrixEnumerated': False,
            'groundSectorCompletion': revised['separate_source_only_completion_option']['c_N'],
            'complementSeparationOverMass': '1', 'internalGroundGapProved': False},
        'charged': {'rankPerCharge': charged['source']['active_cluster_rank_each_Q_plus_minus_one'],
            'completion': charged['source']['positive_complement_c'],
            'photonCap': charged['source']['photon_total_cap'],
            'slots': charged['clock_and_capacity']['duration_one_driven_slots'],
            'records': str(charged['complete_instrument']['all_record_labels']),
            'gapLower': scalar(charged['bounds_exact']['charged_cluster_gap_over_m_lower']),
            'complementMargin': scalar(charged['bounds_exact']['charged_native_complement_margin_over_m_lower']),
            'acceptance': scalar(charged['bounds_exact']['original_accepted_energy_pair_probability_lower']),
            'jointFidelity': scalar(charged['bounds_exact']['accepted_joint_cluster_ground_fidelity_lower']),
            'transition': interval(charged['reference_corrected_charged_transition_over_m']),
            'capacity': interval(charged['clock_and_capacity']['total_log_Hilbert_dimension_nats']),
            'speciesCount': None, 'continuousSpinProved': False, 'restMassProved': False},
        'scope': {'physicalParticleIdentities': [], 'sourceDerivedMassesGeV': [],
            'allFailuresRetained': True, 'finiteExactControlServicesDeclared': True,
            'sameAsClassicalLoadRun': False, 'interactingRelativisticLimit': False,
            'fixedSpacingVolumeLimit': True, 'efficientLocalControllerCompiled': False},
        'evidence': {n: {'path': BASE+n+'/results.json', 'sha256': digest(root/BASE/n/'results.json'),
            'researchSha256': values[n]['public_port']['retained_research_sha256'],
            'proof': 'https://github.com/FloatingPragma/observer-patch-holography/blob/main/'+BASE+n+'/PROOF.md'} for n in NAMES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = json.dumps(build(args.repository), indent=2, ensure_ascii=False, sort_keys=True)+'\n'
    if args.check:
        if OUTPUT.read_text() != text:
            raise ValueError('Field-record viewer export is stale')
    else:
        OUTPUT.write_text(text)
    print('PASS: complete field records, exact bounds and physical scope')


if __name__ == '__main__':
    main()
