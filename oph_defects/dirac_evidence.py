"""Export complete finite Dirac readouts without assigning an observed species."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
from .quantum_evidence import bounds, digest, receipt
from .reference_repository import theory_repository

SOURCE='model/native_block_dirac/results.json'
OUTPUT=Path(__file__).resolve().parent/'viewer2/dirac_evidence.json'


def build(root=None):
    root=Path(root or theory_repository()).resolve(); d=receipt(root,SOURCE)
    for name,sha in d['local_sha256'].items():
        path=(root/'model/native_block_dirac'/name).resolve()
        if not path.is_relative_to(root) or digest(path)!=sha:
            raise ValueError('Dirac implementation changed: '+name)
    scope=d['field_comparison']; member=d['finite_member']; code=d['canonical_occupation_encoding']
    if (scope['SM_weak_chirality_or_observed_mass_identity_claimed']
        or scope['interacting_continuum_or_electromagnetic_Gauss_law_claimed']
        or scope['mass_parameter_is_empirically_calibrated']
        or scope['finite_momentum_cutoff_strict_microcausality']
        or not scope['same_source_state_reference_and_clock']
        or code['complete_four_mode_Fock_code_dimension']!=16
        or code['selected_source_product_dimension']!=81
        or member['execution']['full_native_Hamiltonian_numerically_executed']
        or member['macro_side']!=32):
        raise ValueError('Unreviewed physical claim or source')
    amp=member['optional_complete_preparation_amplification']
    if not amp['failed_attempts_are_never_discarded'] or amp['independent_full_attempts_retained']!=6:
        raise ValueError('Preparation failures changed')
    eps=F(amp['complete_record_TV_to_exact_ground_example_upper'])
    trials=[]
    for key,title in [('literal_zero_separation_control','No intervening wait'),
                      ('literal_complete_active_circuit','A quarter-period wait')]:
        circuit=d[key]; rows=[]
        for r in circuit['all_sixteen_records']:
            if key=='literal_zero_separation_control':
                p=F(r['ideal_actual_source_probability']); interval=[str(p),str(p)]
            else:
                found=next(x for x in member['complete_sixteen_detector_records']
                           if all(x[k]==r[k] for k in ('first_up','first_down','second_up','second_down')))
                interval=found['ideal_probability_interval']
            lo,hi=map(F,interval)
            rows.append({'word':''.join(str(r[k]) for k in ('first_up','first_down','second_up','second_down')),
                         'exact':r['ideal_actual_source_probability'], 'ideal':bounds(interval),
                         'completeAttempt':bounds([str(max(F(0),lo-eps)),str(min(F(1),hi+eps))])})
        trials.append({'title':title,'sourceSeparation':circuit['source_elapsed_time'],'records':rows})
    return {'schema':'oph.viewer.dirac.v1',
        'source':{'constituentsPerSite':4,'cellsPerConstituent':8,'sites':32768,
                  'selectedProductStatesPerSite':81,'fockStatesPerSite':16,'fockModesPerSite':4,
                  'sourceCharge':1,'integerNativeCharge':6,'massParameter':'E*',
                  'massSourceInterval':bounds(member['mass_source_interval']),
                  'kineticLaw':'sqrt(m² I + L_a) − m I','fullSpatialRange':True},
        'dispersion':{'spacingTimesMass':bounds(member['spacing_times_E_star_interval']),
                      'testMomentumOverMass':.75,'testFrequencyOverMass':bounds(member['actual_mode_frequency_over_E_star_interval']),
                      'formula':'Ω_a² = m² + (4/a²) Σ sin²(a p_i / 2)',
                      'continuum':'ω² = m² + |p|²',
                      'fixedWindowError':'0 ≤ ω − Ω_a ≤ a² P⁴/(24m)'},
        'trials':trials,
        'preparation':{'attempts':6,'failureUpper':str(amp['all_attempts_failed_probability_upper']),
                       'completeErrorUpper':str(eps),'conditionalErrorUpper':member['native_preparation']['retained_purification_vector_error_upper'],
                       'capacityNats':bounds(amp['complete_source_preparation_reference_and_pointer_capacity_nats_interval']),
                       'capacityIncludesControllerSynthesis':False,'allOriginalFlagsRetained':True,
                       'programmableFiniteMatrixControl':True},
        'execution':{'literalActiveDimension':16,'retainedSourceAndDetectorModesDimension':256,
                     'fullNativeMatrixExecuted':False,'selectedNativeEigenvectorsExecuted':False},
        'scope':{'freeSpinHalfReadout':True,'completeCAR':True,'continuumCausalAnticommutator':True,
                 'exactFiniteBoxLorentzSymmetry':False,'interactingRelativisticLimit':False,
                 'identifiedElectron':False,'sourceDerivedGeV':False,'SMWeakChirality':False,
                 'sameAsAffineFieldClock':False,'sameAsClassicalLoadRun':False},
        'evidence':{'path':SOURCE,'sha256':digest(root/SOURCE),
                    'proof':'https://github.com/FloatingPragma/observer-patch-holography/blob/53630a945/model/native_block_dirac/PROOF.md'}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    text=json.dumps(build(a.repository),indent=2,ensure_ascii=False,sort_keys=True)+'\n'
    if a.check:
        if OUTPUT.read_text()!=text:raise ValueError('Dirac viewer evidence is stale')
    else:OUTPUT.write_text(text)
    print('PASS: native free Dirac readout, sixteen detector records and retained preparation failures')


if __name__=='__main__':main()
