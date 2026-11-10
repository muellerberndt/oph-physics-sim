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
