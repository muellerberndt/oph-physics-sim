"""Build the viewer's finite interacting energy table from authenticated RER receipts."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
from .reference_repository import theory_repository
DEFAULT_REPOSITORY = theory_repository()
OUTPUT = HERE / 'viewer2' / 'quantum_evidence.json'
BASE = 'model/coherent_defect_wave/chiral_joint/'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(root, name):
    path = root / name
    result = json.loads(path.read_text())
    pins = result.get('input_sha256')
    if not isinstance(pins, dict) or not pins:
        raise ValueError('missing source custody: ' + name)
    for relative, expected in pins.items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file() or digest(target) != expected:
            raise ValueError('source custody changed: ' + relative)
    return result


def bounds(values):
    lo, hi = map(Fraction, values)
    if lo > hi:
        raise ValueError('reversed interval')
    return {'exact': values, 'display': [float(lo), float(hi)]}


def build(root=DEFAULT_REPOSITORY):
    root = Path(root).resolve()
    names = {key: BASE+key+'/results.json' for key in ('interacting_charged_response', 'interacting_ground_preparation')}
    source = {key: receipt(root, path) for key, path in names.items()}
    names['native_higgs_spectral_join']='model/native_higgs_spectral_join/results.json'
    higgs=receipt(root,names['native_higgs_spectral_join'])
    for name, expected in higgs['local_sha256'].items():
        target=(root/'model/native_higgs_spectral_join'/name).resolve()
        if not target.is_relative_to(root) or digest(target)!=expected:
            raise ValueError('Higgs response implementation changed: '+name)
    joined=higgs['same_H_cross_residue_and_records']
    selection=joined['positive_cross_selector']; detector=joined['complete_detector_law']
    if (not selection['same_energy_elementary_composite_G1_doublet']
            or selection['physical_SM_weak_chirality_interactions_or_GeV_identity_proved']
            or selection['maximum_elementary_E_star_common_composite_overlap_proved']
            or detector['full_setting_spin_occupation_energy_record_count']!=16
            or not detector['preparation_partners_spin_references_and_all_filter_labels_retained']
            or not detector['spectral_energy_bin_readout_is_exact_finite_service_not_QPE_implemented']):
        raise ValueError('Higgs response source, instrument or interpretation changed')
    names['native_charge_centroid']='model/native_charge_centroid/results.json'
    centroid=receipt(root,names['native_charge_centroid'])
    for name in ('bounds.py','certificate.py','PROOF.md'):
        if 'model/native_charge_centroid/'+name not in centroid['input_sha256']:
            raise ValueError('Missing charge centroid implementation custody: '+name)
    cp=centroid['actual_physical_preparation']; cc=centroid['clock']
    if (cp['extra_P0_ground_projection_used']
            or centroid['single_charged_eigenvalue_or_original_Estar_relocalized']
            or centroid['relativistic_mass_spin_or_SM_identification']
            or cc['exact_native_Eg_used_as_free_pointer_phase']
            or cc['large_source_pointer_matrix_numerically_executed']
            or not cc['all_pointer_labels_retained']
            or not cc['prior_bin_admission_remains_explicit_spectral_service']
            or not cc['common_autonomous_Htotal_on_all_branches']):
        raise ValueError('Centroid source, clock or interpretation changed')
    names['native_higgs_filtered_state']='model/native_higgs_filtered_state/results.json'
    filtered=receipt(root,names['native_higgs_filtered_state'])
    for name,expected in filtered['owned_source_sha256'].items():
        target=(root/'model/native_higgs_filtered_state'/name).resolve()
        if not target.is_relative_to(root) or digest(target)!=expected:
            raise ValueError('Filtered-state implementation changed: '+name)
    fs=filtered['filtered_state_settings'][0]; fp=filtered['physical_preparation']
    if (fs['radius_in_units_of_360_lambda']!=2
            or fs['implemented_bin_QPE_claimed']
            or fs['unconditional_A_and_B_over_2_complete_laws_are_close_claimed']
            or fp['exact_interacting_ground_projection_assumed']
            or not filtered['complete_instruments']['every_fine_preparation_record_and_partner_retained']
            or filtered['typed_free_Dirac_corollary']['interacting_native_Lorentz_or_SM_chirality_transfer']):
        raise ValueError('Filtered-state service or interpretation changed')
    names['native_weak_current_response']='model/native_weak_current_response/results.json'
    weak=receipt(root,names['native_weak_current_response'])
    for name in ('certificate.py','local.py','charged.py','bounds.py','PROOF.md'):
        if 'model/native_weak_current_response/'+name not in weak['input_sha256']:
            raise ValueError('Missing weak-current implementation custody: '+name)
    wi=weak['finite_inelastic_experiment']; wc=weak['charged_current']
    if (weak['actual_preparation']['exact_P0_added']
            or not weak['source']['same_source_clock_on_all_branches']
            or weak['scope']['weak_scattering_cross_section_or_SM_matching']
            or weak['scope']['W_boson_or_Lorentz_vector_identified']
            or wi['unknown_ground_energy_as_free_clock_phase']
            or not wi['all_additional_pointer_labels_retained']
            or not wc['current_measurement_changes_the_admitted_energy_distribution']):
        raise ValueError('Weak-current source or ordered instrument changed')
    names['native_higgs_current_response']='model/native_higgs_current_response/results.json'
    hc=receipt(root,names['native_higgs_current_response'])
    for name,expected in hc['owned_source_sha256'].items():
        target=(root/'model/native_higgs_current_response'/name).resolve()
        if not target.is_relative_to(root) or digest(target)!=expected:
            raise ValueError('Higgs current implementation changed: '+name)
    if (hc['conjugation']['W_invariant']
            or hc['scope']['weak_boson_chiral_Lorentz_interaction_or_SM_identification']
            or hc['scope']['GeV_rest_mass_or_measured_lepton_ratio']
            or not hc['source']['original_W_unchanged']
            or hc['apparatus']['complete_setting_pump_energy_current_alphabet_size']!=80
            or not hc['apparatus']['failures_and_outside_energy_bin_retained']
            or hc['preparation']['exact_interacting_ground_projection_assumed']):
        raise ValueError('Higgs current conjugation, source or complete instrument changed')
    hcr=hc['prepared_current']
    names['native_local_current_readout']='model/native_local_current_readout/results.json'
    lc=receipt(root,names['native_local_current_readout'])
    for name,expected in lc['owned_source_sha256'].items():
        target=(root/'model/native_local_current_readout'/name).resolve()
        if not target.is_relative_to(root) or digest(target)!=expected:
            raise ValueError('Local current implementation changed: '+name)
    lcs=lc['source']; lcc=lc['complete_clock_and_error']; lcr=lc['complete_prepared_records']; lcp=lc['control_resources']
    if (lcs['parent_current_response_sha256']!=digest(root/names['native_higgs_current_response'])
            or not lcs['source_hamiltonian_continues_through_final_pulse']
            or lcs['full_Hext_cancellation_used_for_final_readout']
            or not lcs['first_three_preparation_pump_bin_services_unchanged']
            or lcs['exact_interacting_ground_projection_added']
            or lcc['total_elapsed_source_time']!='3145729/1048576'
            or lcc['final_pulse_source_duration']!='1/1048576'
            or lcc['post_readout_success_selection_or_normalization_used']
            or lc['local_apparatus']['complete_local_coupling_dimension']!=400
            or lcr['complete_setting_pump_energy_current_alphabet_size']!=80
            or not lcr['all_fine_preparation_flags_and_source_outputs_retained']
            or not lcr['intervals_describe_pointer_decoded_current_means']
            or lcr['instantaneous_post_measurement_W_expectation_asserted']
            or lcr['unconditional_A_and_B_rates_equated']
            or not lcr['every_current_label_retained_on_admitted_branch']
            or not lcp['coupling_is_declared_not_compiled_from_native_C_E']
            or lcp['full_matrix_preparation_and_exact_spectral_bin_premises_removed']
            or not lcp['full_Gauss_constraints_preserved_throughout_drive']
            or not lc['scope']['local_same_source_original_W_readout_with_nonzero_response']
            or any(v for k,v in lc['scope'].items()if k!='local_same_source_original_W_readout_with_nonzero_response')):
        raise ValueError('Local current source, clock, complete instrument or scope changed')
    names['native_wilson_maxwell']='model/native_wilson_maxwell/results.json'
    wm=receipt(root,names['native_wilson_maxwell'])
    for name,expected in wm['owned_source_sha256'].items():
        target=(root/'model/native_wilson_maxwell'/name).resolve()
        if not target.is_relative_to(root) or digest(target)!=expected:
            raise ValueError('Wilson–Maxwell implementation changed: '+name)
    wmr=wm['complete_interaction_records']; wmpr=wm['actual_preparation']
    if (wm['count_instrument']['full_pointer_dimension']!=1161
            or wm['count_instrument']['target_label']!=[1,1,1]
            or wmr['source_state_is_actual_coupled_ground']
            or not wmr['all_virtual_matter_and_photon_occupations_in_Duhamel_bound']
            or not wmpr['no_initial_P0_or_code_measurement']
            or wmpr['filter_depth']!=0
            or wm['boundaries']['interacting_Lorentz_or_four_dimensional_QED_limit_proved']
            or wm['boundaries']['SM_particle_or_weak_chirality_identification']
            or wm['source_choices']['source_empirical_mass_or_coupling_calibration']):
        raise ValueError('Wilson–Maxwell source, complete count or interpretation changed')
    names['native_wilson_maxwell_refinement']='model/native_wilson_maxwell_refinement/results.json'
    wr=receipt(root,names['native_wilson_maxwell_refinement'])
    for name,expected in wr['owned_source_sha256'].items():
        target=(root/'model/native_wilson_maxwell_refinement'/name).resolve()
        if not target.is_relative_to(root) or digest(target)!=expected:
            raise ValueError('Photon-cutoff implementation changed: '+name)
    if (wr['parent_finite_source_sha256']!=digest(root/names['native_wilson_maxwell'])
            or wr['construction']['changed_parameter']!='total photon cap K only'
            or not wr['construction']['same_native_trial_state_reference_density_Gauss_and_clock']
            or wr['cap3_pair_photon_transfer']['padding_changes_parent_executed_apparatus']
            or wr['cap3_pair_photon_transfer']['finite_executed_pointer_labels']!=1161
            or wr['cap3_pair_photon_transfer']['common_overflow_padded_labels']!=1451
            or any(wr['boundaries'].values())):
        raise ValueError('Photon-cutoff comparison source, labels or scope changed')
    names['native_quark_response']='model/native_quark_response/results.json'
    qr=receipt(root,names['native_quark_response'])
    if (not qr['computed_equal_coupling_uc_dc_resolvent_equality']
            or not qr['scope']['all_local_probe_roots_and_exact_zero_residues_retained']
            or any(v for k,v in qr['scope'].items()if k!='all_local_probe_roots_and_exact_zero_residues_retained')
            or qr['same_source']['additional_exact_ground_projection']
            or not qr['same_source']['all_failure_and_no_operation_records_retained']
            or not qr['same_source']['original_physical_preparation_retained']
            or qr['fresh_replay']['numerical_eigensolver_used']
            or qr['full_probe_reference_operator']['reference_Hamiltonian']!='mu N_ref, mu=4; zero-energy alternative assessed separately'):
        raise ValueError('Quark source, reference or physical interpretation changed')
    quark_rows=[]
    for key in ('uc','dc','Q'):
        local=qr['exact_local_response'][key];full=qr['full_interacting_prepared_records'][key]
        if not full['exact_source_spectral_service_declared'] or full['efficient_native_gate_synthesis_claimed']:
            raise ValueError('Quark energy apparatus interpretation changed')
        records=full['records'];strong=max(records,key=lambda r:Fraction(r['original_attempt_probability_lower']))
        quark_rows.append({'id':key,
            'localLines':[{'energy':bounds(r['source_excitation_interval']),
                          'probability':bounds(r['probability_interval']),
                          'exactlyZeroLocalWeight':r['exactly_zero_residue']}
                          for r in local['complete_positive_records']],
            'sourceEnergy':bounds(strong['source_excitation_relative_actual_ground']),
            'originalStrongBinLower':bounds([strong['original_attempt_probability_lower']]*2),
            'allFullSourceBins':[{'energy':bounds(r['source_excitation_relative_actual_ground']),
                                 'probability':bounds(r['prepared_conditional_probability'])}
                                 for r in records],
            'outsideAllBins':bounds(full['prepared_outside_all_bins']),
            'pumpFailure':bounds(full['prepared_pump_failure_interval']),
            'preparationSuccessLower':bounds([full['preparation_success_lower']]*2),
            'referenceEnergy':full['success_reference_energy'],'massGeV':None,
            'empiricalParticle':None})
    names['native_quark_higgs_join']='model/native_quark_higgs_join/results.json'
    qh=receipt(root,names['native_quark_higgs_join'])
    qs=qh['same_source']; qa=qh['apparatus']; alignment=qh['normalized_admitted_states_and_histories']
    if (qh['source_parent']!={names['native_quark_response']:digest(root/names['native_quark_response'])}
            or qs['H']!='h_active+h_partner+4N_ref' or qs['additional_P0']
            or not qs['original_physical_preparation'] or not qs['all_complements_and_failure_records_retained']
            or qs['reference_cost_per_occupation']!=4
            or qa['full_setting_pump_energy_labels']!=8
            or qa['pump_source_duration']!=1 or qa['energy_source_duration']!=1
            or not qa['full_source_clock_on_no_operation_branches']
            or qa['efficient_native_gate_synthesis'] or qa['controller_synthesis_workspace_included']
            or alignment['settings']!=['A','C/2'] or alignment['additional_P0']
            or not alignment['original_two_source_preparation']
            or alignment['pure_inequality_source_state_and_reference_transferred']
            or alignment['unconditional_complete_laws_claimed_close']
            or not qh['scope']['actual_shared_positive_spectral_bin_contribution']
            or not qh['scope']['all_atomic_roots_and_zero_residues_retained']
            or any(value for key,value in qh['scope'].items()if key not in
                   ('actual_shared_positive_spectral_bin_contribution','all_atomic_roots_and_zero_residues_retained'))):
        raise ValueError('Quark/Higgs source, preparation, clock or interpretation changed')
    quark_higgs_rows=[]
    for key in ('uc','dc','Q'):
        state=alignment['rows'][key]; fringe=qh['full_interacting_interference_records'][key]
        if (not state['source_and_common_retained_preparation_environment_included']
                or state['different_classical_setting_label_in_compared_state']
                or state['same_history_policy_may_read_pump_setting']
                or state['original_pump_success_rates_equated']
                or not state['all_preparation_failure_records_retained']
                or state['complete_fair_setting_pump_energy_labels']!=8
                or fringe['selector_equals_maximum_elementary_residue_selector']):
            raise ValueError('Quark/Higgs conditional history domain changed')
        if len(fringe['settings'])!=2 or any(row['failed_pump_energy_hit_probability']!='0'for row in fringe['settings']):
            raise ValueError('Quark/Higgs complete interference labels changed')
        states=[]
        for channel in ('elementary','composite_B_over_2'):
            law=state['complete_setting_records'][channel]
            if law['failed_pump_energy_hit']!='0':raise ValueError('Quark/Higgs unused label changed')
            states.append({'setting':channel,
                'conditional':{name:bounds(pair)for name,pair in law['prepared_conditional'].items()},
                'originalFairSetting':{name:bounds(pair)for name,pair in law['original_attempt_joint_fair_setting'].items()}})
        quark_higgs_rows.append({'id':key,
            'sourceEnergy':bounds(fringe['source_excitation_relative_actual_ground']),
            'historyTVUpper':bounds([state['common_history_total_variation_upper']]*2),
            'originalElementarySuccessLower':bounds([state['original_attempt_elementary_in_bin_probability_lower']]*2),
            'originalCompositeSuccessLower':bounds([state['original_attempt_composite_B_over_2_in_bin_probability_lower']]*2),
            'originalInterferenceContrastLower':bounds([fringe['original_attempt_joint_contrast_lower']]*2),
            'stateRecords':states,
            'interferenceRecords':[{'sign':row['sign'],
                'conditional':{name:bounds(pair)for name,pair in row['prepared_conditional_records'].items()},
                'originalFairSetting':{name:bounds(pair)for name,pair in row['original_attempt_joint_setting_records'].items()}}
                for row in fringe['settings']]})
    names['native_wilson_maxwell_ground']='model/native_wilson_maxwell_ground/results.json'
    wg=receipt(root,names['native_wilson_maxwell_ground'])
    for name,expected in wg['owned_source_sha256'].items():
        target=(root/'model/native_wilson_maxwell_ground'/name).resolve()
        if not target.is_relative_to(root) or digest(target)!=expected:
            raise ValueError('Full-ground implementation changed: '+name)
    wp=wg['complete_repaired_ground_preparation'];ground_completion=wg['declared_onsite_completion'];wd=wg['complete_stationary_ground_dressing_record']
    if (wg['unchanged_source']['ground_in_code']
            or wg['unchanged_source']['active_ground_multiplicity']!=96
            or ground_completion['old_source_receipt_or_H_mutated']
            or ground_completion['global_scalar_energy_subtraction']!='none'
            or not ground_completion['complete_pattern_preserving_count_laws_exactly_invariant']
            or not ground_completion['selected_active_coupled_ground_is_unique']
            or ground_completion['actual_ground_radiation_or_SM_mass_claim']
            or wp['ground_energy_or_exact_ground_projector_used']
            or wp['full_source_trial_copies']!=4 or wp['finite_filter_count']!=3
            or wd['complete_count_pointer_labels']!=1161
            or not wp['all_filter_basis_outcomes_retained']
            or not wp['all_partner_sources_retained']
            or wd['ground_emission_scattering_or_radiation']
            or wg['exact_macrocell_and_Gauss_completion_control']['counterterm_is_forced_by_primitive_repair_law']
            or wg['exact_macrocell_and_Gauss_completion_control']['preexisting_global_kinetic_current_guard_removed']):
        raise ValueError('Full-ground completion, preparation or interpretation changed')
    names['native_wilson_maxwell_ground_refinement']='model/native_wilson_maxwell_ground_refinement/results.json'
    gr=receipt(root,names['native_wilson_maxwell_ground_refinement'])
    for name,expected in gr['owned_source_sha256'].items():
        target=(root/'model/native_wilson_maxwell_ground_refinement'/name).resolve()
        if not target.is_relative_to(root) or digest(target)!=expected:
            raise ValueError('Ground refinement implementation changed: '+name)
    family=gr['same_source_family']; clock=gr['fixed_complete_energy_clock']
    ground_bounds=gr['uniform_ground_bounds_exact']; prepared=gr['constructive_complete_preparation_history']; ideal=gr['constructive_ideal_ground_history']
    if (gr['parent_ground_sha256']!=digest(root/names['native_wilson_maxwell_ground'])
            or gr['parent_photon_refinement_sha256']!=digest(root/names['native_wilson_maxwell_refinement'])
            or family['changed_parameter']!='total photon cap only'
            or family['photon_caps']!='every integer K>=3'
            or not family['native_mass_reference_Gauss_density_current_and_clock_unchanged']
            or family['spatial_volume_or_spacing_refinement']
            or not clock['additional_apparatus_setting_relative_to_parent_K3_clock']
            or clock['parent_clock_or_receipt_modified'] or clock['imaginary_heat_time_is_physical_clock']
            or not clock['all_M_basis_records_and_all_noop_failure_branches_retained']
            or not clock['same_pointer_alphabet_for_all_K_and_mathematical_comparison']
            or clock['source_copies']!=4 or clock['filter_count']!=3
            or clock['pointer_M']!=315179904752653412
            or clock['source_elapsed_time_before_count']!='4, including one preparation and three filter slots'
            or clock['source_elapsed_time_after_count']!='5, including final common source factor'
            or prepared['K']!=2**96 or ideal['K']!=1024
            or prepared['matrices_enumerated_or_resource_feasibility_claimed']
            or ideal['actual_full_source_matrix_enumerated']
            or not gr['boundaries']['fixed_cube_interacting_ground_and_complete_record_limit']
            or any(value for key,value in gr['boundaries'].items()
                   if key!='fixed_cube_interacting_ground_and_complete_record_limit')):
        raise ValueError('Uniform ground source, separate apparatus, cutoff or scope changed')
    prep = source['interacting_ground_preparation']['preparation']
    filt = prep['finite_energy_agreement_filter']
    channels = source['interacting_charged_response']['same_H_interacting_channels']
    rows=[]
    for key in ('ec', 'L'):
        row=channels[key]; law=row['full_interacting_energy_readout']
        if row['scope']['physical_rest_mass_or_free_charge_dispersion'] or law['reference_energy_subtracted_silently']:
            raise ValueError('unreviewed physical promotion')
        cost=Fraction(row['source']['reference_particle_cost'])
        if cost != 4 or row['source']['source_ground_or_clock_substituted']:
            raise ValueError('source or reference changed')
        rows.append({'id':key,'name':'Charged singlet eᶜ' if key=='ec' else 'Weak lepton doublet L',
                     'sourceEnergy':bounds(law['source_excitation_energy_bin_after_subtracting_declared_reference_cost']),
                     'totalEnergy':bounds(law['total_excitation_energy_bin_relative_to_actual_initial_ground']),
                     'probabilities':{k:bounds(v) for k,v in law['conditional_probabilities'].items()},
                     'referenceEnergy':str(cost), 'empiricalParticle':None, 'massGeV':None})
    return {'schema':'oph.viewer.interacting_energy_evidence.v1',
            'source':{'cells':8,'fermionModesPerCell':30,'scalarStatesPerCell':5,'linkStates':119,
                      'hopping':prep['source']['hopping_coefficient'], 'clock':'exp(-it H_lambda)',
                      'family':'full Peter-Weyl gauge-Higgs matter source'},
            'preparation':{'successLower':float(Fraction(filt['success_probability_lower'])),
                           'traceDistanceUpper':float(Fraction(filt['conditional_one_source_trace_distance_upper'])),
                           'scheduleTerms':str(filt['explicit_integer_M']),
                           'algebraicServicesDeclared':True,'efficientNativeCircuitExecuted':False},
            'channels':rows,'allFailuresRetained':True,
            'quarkHiggs':{'families':quark_higgs_rows,
                'sameSourceAndReference':True,'originalTwoSourcePreparation':True,
                'commonRetainedEnvironmentIncluded':True,'conditionalAcceptedStates':True,
                'settingLabelExcluded':True,'allFailuresAndComplementsRetained':True,
                'unconditionalLawsEquated':False,'historyPolicyMayReadSetting':False,
                'leptonStateOrReferenceTransferred':False,'exactGroundProjection':False,
                'canonicalCompositeCAR':False,'quarkMassOrChiralityIdentified':False,
                'completeSettingLabels':8,'potentiallyNonzeroLabels':6,
                'pumpSourceDuration':qa['pump_source_duration'],'binSourceDuration':qa['energy_source_duration'],
                'sourceClock':alignment['source_clock'],'referenceEnergy':'4',
                'storageUpper':qa['retained_storage_upper'],'Mprep':str(qa['Mprep']),
                'matrixPumpAndExactBinServicesDeclared':True,'efficientNativeCompilation':False},
            'groundRefinement':{
                'allCapsFrom':3,'sourceCopies':clock['source_copies'],'filters':clock['filter_count'],
                'sameCompletedSourceFamily':True,'fixedSpatialCube':True,
                'distinctUniformPointerSetting':True,'parentCap3RecordsUnchanged':True,
                'allFailuresAndNoopRecordsRetained':True,
                'matrixServicesDeclared':True,'heatTimeIsSourceClock':False,
                'spatialOrUVContinuum':False,'matricesEnumeratedOrFeasible':False,'radiationOrSMIdentity':False,
                'uniformFullGapLower':bounds([ground_bounds['uniform_full_physical_gap_lower']]*2),
                'uniformPreparationSuccessLower':bounds([ground_bounds['uniform_complete_filter_tree_success_lower']]*2),
                'uniformPreparedCountLower':bounds([ground_bounds['uniform_physical_preparation_and_T111_joint_probability_lower']]*2),
                'uniformPreparationErrorUpper':bounds([ground_bounds['uniform_conditional_ground_trace_distance_upper']]*2),
                'idealGroundCap':ideal['K'],
                'idealGroundTwoTestErrorUpper':bounds([ground_bounds['ideal_ground_two_field_record_error_at_K1024_upper']]*2),
                'preparedCapExponent':prepared['K_dyadic_exponent'],'preparedCap':str(prepared['K']),
                'preparedCountErrorUpper':bounds([ground_bounds['fixed_pointer_complete_prep_and_count_error_upper']]*2),
                'preparedTwoTestErrorUpper':bounds([ground_bounds['fixed_pointer_complete_prep_and_two_field_tests_error_upper']]*2),
                'pointerM':str(clock['pointer_M']),'sourceTimeBeforeCount':4,'sourceTimeAfterCount':5,
                'retainedCapacityNats':bounds(prepared['full_source_reference_trial_filter_capacity_nats_interval']),
                'additionalCountPointerCapacity':prepared['additional_complete_count_pointer_capacity'],
                'completeCountLabels':str(prepared['complete_count_alphabet_at_K']),
                'rawSourceNormAtPreparedCapUpper':bounds([ground_bounds['all_cap_raw_H_norm_at_explicit_K_upper']]*2),
                'durationOneControlNorm':prepared['each_duration_one_control_norm_upper'],
                'grossPreparationWork':prepared['gross_preparation_tree_work_upper']},
            'localCurrent':{
                'elementaryMean':bounds(lcr['current_means']['elementary_W_mean_interval']),
                'compositeMean':bounds(lcr['current_means']['Higgs_composite_W_mean_interval']),
                'historyTVUpper':bounds([lcr['same_apparatus_A_B_conditional_history_distance_upper']]*2),
                'apparatusErrorUpper':bounds([lcc['all_retained_history_total_variation_error_upper']]*2),
                'meanErrorUpper':bounds([lcc['conditional_current_mean_absolute_error_upper']]*2),
                'driveNormUpper':bounds([lcp['local_final_drive_norm_upper']]*2),
                'grossSwitchWorkUpper':bounds([lcp['final_pulse_gross_switching_work_upper']]*2),
                'records':[{'value':row['W_value'],
                    'elementaryProbability':bounds(row['actual_elementary_bin_conditional_probability_interval']),
                    'compositeProbability':bounds(row['actual_composite_bin_conditional_probability_interval']),
                    'originalElementaryJointLower':bounds([row['original_elementary_attempt_joint_probability_lower']]*2),
                    'originalCompositeJointLower':bounds([row['original_composite_attempt_joint_probability_lower']]*2)}
                    for row in lcr['five_complete_current_records']],
                'totalSourceTime':lcc['total_elapsed_source_time'],
                'finalPulseSourceTime':lcc['final_pulse_source_duration'],
                'completeLabels':80,'localMatrixDimension':400,'pointerStates':5,
                'sourceKeepsEvolving':True,'allFailuresRetained':True,'sameOriginalCurrent':True,
                'localCouplingDeclared':True,'priorPreparationAndBinServicesRetained':True,
                'fullHamiltonianCancellation':False,'nativeRepairCompilationProved':False,
                'exactGroundProjection':False,'instantaneousFinalSourceMean':False,
                'SMParticleOrWeakBoson':False},
            'coupledGround':{
                'originalGroundInsideCode':False,'originalActiveGroundMultiplicity':96,
                'originalCodeAboveComplementLower':bounds([wg['bounds_exact']['original_code_above_native_triplet_margin_lower']]*2),
                'completedPhysicalGapLower':bounds([wg['bounds_exact']['repaired_full_physical_gap_lower']]*2),
                'completionCoefficient':bounds([wg['bounds_exact']['counterterm_over_m']]*2),
                'preparationSuccessLower':bounds([wp['complete_tree_success_lower']]*2),
                'preparationTraceErrorUpper':bounds([wp['conditional_main_trace_distance_upper']]*2),
                'preparedStationaryCountLower':bounds([wd['physical_preparation_and_count_joint_probability_lower']]*2),
                'capacityNats':bounds(wp['capacity_nats_interval']),
                'sourceTrials':4,'filters':3,'completeCountLabels':1161,
                'changedHamiltonianExplicit':True,'positiveMacrocellPenalty':True,
                'allFailuresPartnersRetained':True,'patternPreservingOldCountLawsInvariant':True,
                'unknownGroundProjectorOrPhaseUsed':False,'primitiveRepairForcesCompletion':False,
                'radiationOrGeVMass':False,'fullNativeSpatialLocalityProved':False},
            'quarkResponses':{'families':quark_rows,'sameFullInteractingSource':True,
                'actualCARNormalization':True,'staticColorReferenceRetained':True,
                'allPreparationPumpEnergyFailuresRetained':True,
                'ucDcEqualAtConfiguredCouplings':True,
                'sourceSelectedGroundForEveryColoredSector':False,
                'propagatingQuarkOrGeVMassIdentified':False},
            'photonRefinement':{
                'cap3ErrorUpper':bounds([wr['cap3_complete_state_and_record_bound']['complete_record_TV_upper']]*2),
                'overflowUpper':bounds([wr['cap3_complete_state_and_record_bound']['photon_overflow_probability_upper']]*2),
                'comparisonTargetLower':bounds([wr['cap3_pair_photon_transfer']['exact']['comparison_actual_target_probability_lower']]*2),
                'cutoffs':[{'cap':row['cap'],
                    'errorUpper':bounds([row['state_vector_error_upper']]*2)}
                    for row in wr['cutoff_table']if row['cap']in(1,2,3,4,8)],
                'sameSourceStateReferenceClock':True,'completeRecordBound':True,
                'fixedSpatialCube':True,'overflowRetained':True,
                'executedLabels':1161,'comparisonLabels':1451,
                'unboundedOscillatorActuallyExecuted':False,
                'spatialContinuumOrSMIdentified':False},
            'wilsonMaxwell':{
                'targetLower':bounds([wmr['exact']['actual_unconditional_target_probability_lower']]*2),
                'connectedLower':bounds([wmr['exact']['actual_unconditional_binary_connected_response_lower']]*2),
                'codeWeightLower':bounds([wmpr['global_code_probability_lower']]*2),
                'sourceEnergy':bounds(wm['clock_and_capacity']['source_selected_excitation_interval']),
                'capacityNats':bounds(wm['clock_and_capacity']['full_native_source_reference_trial_capacity_nats_interval']),
                'sites':8,'nativeConstituents':32,'transverseModes':5,'photonStates':56,'totalPhotonCap':3,
                'physicalRecords':1160,'completeRecords':1161,'targetCounts':[1,1,1],
                'coupling':wm['source_choices']['g'],'timeTimesSourceMass':wmr['exact']['dimensionless_mT'],
                'freeTargetProbability':wmr['free_target_probability'],
                'deterministicTrial':True,'allComplementRecordsRetained':True,
                'sameFullSourceClock':True,'bothIndependentCurrentMethodsAgree':True,
                'wholeOperatorEvolutionErrorIncluded':True,'exactGroundProjection':False,
                'inputIsCoupledGround':False,'continuumOrSMIdentification':False,
                'entireFockHamiltonianEnumerated':False,'controllerWorkspaceIncludedInCapacity':False},
            'higgsCurrent':{
                'elementaryMean':bounds(hcr['elementary_W_mean_interval']),
                'compositeMean':bounds(hcr['Higgs_composite_W_mean_interval']),
                'historyTVUpper':bounds([hcr['same_history_TV_upper']]*2),
                'records':[{'value':row['W_value'],
                    'elementaryProbability':bounds(row['actual_elementary_bin_conditional_probability_interval']),
                    'compositeProbability':bounds(row['actual_composite_bin_conditional_probability_interval']),
                    'originalElementaryJointLower':bounds([row['original_elementary_attempt_joint_probability_lower']]*2),
                    'originalCompositeJointLower':bounds([row['original_composite_attempt_joint_probability_lower']]*2)}
                    for row in hcr['five_complete_current_records']],
                'completeLabels':80,'allFailuresRetained':True,
                'sameOriginalCurrent':True,'currentInvariantUnderConjugation':False,
                'exactGroundProjection':False,'SMParticleOrWeakBoson':False,
                'energyRadiusHoppingNorm':hc['energy_admission']['radius_in_units_of_360lambda']},
            'weakResponse':{
                'originalInelasticLower':bounds([wi['original_attempt_nonzero_phase_probability_lower']]*2),
                'chargedConditionalMean':bounds(wc['actual_prepared_bin_conditional_W_mean_interval']),
                'currentRecords':[{'value':str(row['W_value']),'preparedProbability':bounds(row['prepared_probability_interval'])}
                                  for row in weak['ground_current']['five_complete_current_records']],
                'allFailuresAndPointerLabelsRetained':True, 'sameSourceClock':True,
                'orderedEnergyThenCurrent':True, 'measurementCanChangeEnergy':True,
                'exactGroundProjection':False,'physicalWeakCrossSection':False,'WParticleIdentified':False},
            'filteredState':{
                'sourceEnergy':bounds(fs['excitation_enclosure_relative_actual_ground']),
                'historyTVUpper':bounds([fs['actual_same_history_total_variation_upper']]*2),
                'originalElementarySuccessLower':bounds([fs['unconditional_elementary_in_bin_probability_lower']]*2),
                'originalCompositeSuccessLower':bounds([fs['unconditional_composite_B_over_2_in_bin_probability_lower']]*2),
                'sourceTrials':fp['native_trial_copies'], 'filters':fp['agreement_filter_count'],
                'allPreparationRecordsRetained':True, 'exactSpectralBinService':True,
                'commonSubsequentPolicyRequired':True, 'conditionalSuccessComparison':True,
                'exactGroundProjection':False, 'fullUnconditionedLawsEquated':False,
                'compositeCanonicalOnAllOccupations':False, 'interactingSMTransfer':False},
            'centroid':{
                'sourceEnergyMean':bounds(centroid['finite_clock_conditional_mean_source_excitation_interval']),
                'decodedMean':bounds(centroid['finite_pointer_mean_decoded_source_energy_interval']),
                'absoluteSourceAndReferenceMean':bounds(centroid['finite_pointer_mean_absolute_source_plus_reference_energy_interval']),
                'originalBinSuccessLower':bounds([cp['original_attempt_bin_success_lower']]*2),
                'reportedNeutralEnergyBaseline':str(cc['reported_neutral_energy_baseline']),
                'baselineErrorUpper':bounds([cc['actual_Eg_distance_from_reported_baseline_upper']]*2),
                'clockMeanErrorUpper':bounds([cc['energy_estimator_bias_upper_including_rational_error']]*2),
                'preparationMeanErrorUpper':bounds([cp['conditional_centroid_preparation_error_upper']]*2),
                'pointerLabels':cc['pointer_dimension'], 'rationalFactors':cc['R33_factor_count_upper'],
                'allFailuresAndPointerLabelsRetained':True, 'priorExactSpectralBinService':True,
                'extraGroundProjection':False, 'exactGroundPhaseInput':False,
                'fullSourcePointerExecuted':False, 'singleEigenvalueSelected':False,
                'restMassOrSMIdentity':False},
            'higgsConnection':{
                'elementary':'A = eᶜ†', 'composite':'B = −ε H†L',
                'equation':'[h,A] = B + 2⁻³⁴[T,A]',
                'crossResidueLower':str(joined['real_elementary_composite_cross_residue_lower_exact']),
                'compositeBinWeightLower':str(joined['composite_unnormalized_bin_weight_lower_exact']),
                'originalSettingHitContrastLower':str(joined['unconditional_full_preparation_setting_hit_contrast_lower']),
                'sourceEnergy':bounds(selection['source_only_energy_interval']),
                'settings':['(A + B/2)/2','(A − B/2)/2'],
                'detectorRecords':16, 'potentiallyNonzeroRecords':6,
                'allFailureAndComplementRecordsRetained':True,
                'usesClassicalScalarBackground':False,
                'compositeCanonicalOnAllOccupations':False,
                'sameAsMaximumElementaryResidueSelector':False,
                'weakScatteringOrSMIdentityProved':False,
                'exactFiniteSpectralService':True,
                'QPEImplementedForThisCompositeExperiment':False},
            'scope':{'sourceEnergyTransitions':True,'isolatedChargedPole':False,'physicalRestMass':False,
                     'sameAsClassicalLoadRun':False,'sourceDerivedGeV':False},
            'evidence':{key:{'path':path,'sha256':digest(root/path),
                            'proof':'https://github.com/FloatingPragma/observer-patch-holography/blob/main/'+path.replace('results.json','PROOF.md')}
                        for key,path in names.items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,default=DEFAULT_REPOSITORY)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(); text=json.dumps(build(args.repository),ensure_ascii=False,indent=2,sort_keys=True)+'\n'
    if args.check:
        if OUTPUT.read_text()!=text:raise ValueError('viewer interacting evidence is stale')
    else:OUTPUT.write_text(text)
    print('PASS: source-bound interacting energy records; no GeV or species promotion')


if __name__=='__main__':main()
