"""Export source-bound finite-clock position and field observations."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
from .quantum_evidence import bounds,digest,receipt
from .reference_repository import theory_repository

OUTPUT=Path(__file__).resolve().parent/'viewer2/charged_field_evidence.json'
SOURCES={'ground':'model/affine_gauss_full_spectrum/results.json',
         'clock':'model/affine_gauss_charged_clock/results.json',
         'observations':'model/affine_gauss_clock_observables/results.json',
         'join':'model/native_abelian_field_join/results.json'}
GEOMETRY='code/electromagnetism/runtime/cone_whitney_bridge_receipt.json'


def outward(values,places=13):
    scale=10**places;out=[]
    for i,x in enumerate(map(F,values)):
        n=x.numerator*scale//x.denominator if i==0 else -(-x.numerator*scale//x.denominator)
        whole,frac=divmod(n,scale);out.append(f'{whole}.{frac:0{places}d}')
    return out


def build(root=None):
    root=Path(root or theory_repository()).resolve()
    data={k:receipt(root,v) for k,v in SOURCES.items()};g,c,o,j=(data[k] for k in ('ground','clock','observations','join'))
    if (o['schema']!='oph.affine_gauss_clock_observables.v1'
        or not o['same_actual_H_preparation_reference_and_clock']
        or o['exact_ground_projection_or_stationary_post_clock_state_claimed']
        or o['positive_joint_record_lower_bounds']!=26
        or o['scope']['mass_GeV'] is not None
        or g['physical_one_charge_dimension']!=70928
        or c['scope']['GeV_mass'] is not None
        or c['phase_apparatus']['pointer_dimension']!=2**20
        or j['scope_boundaries']['unpenalized_full_charged_ground_identified']):
        raise ValueError('Unreviewed source, detector, identity or complement claim')
    if o['parent_sha256']!=digest(root/SOURCES['clock']):raise ValueError('Clock attachment changed')
    geometry=json.loads((root/GEOMETRY).read_text())
    for path,sha in geometry['pins'].items():
        target=(root/path).resolve()
        if not target.is_relative_to(root) or digest(target)!=sha:raise ValueError('Geometry changed: '+path)
    mesh=geometry['mesh']
    if mesh['counts']!=[13,42,50,20]:raise ValueError('Carrier changed')
    records=[{'vertex':r['source_vertex'],'fieldExcited':r['transverse_nonvacuum'],
              'ground':bounds(r['actual_ground_probability_interval']),
              'afterClock':bounds(r['conditional_post_clock_probability_interval']),
              'original':bounds(r['unconditional_accepted_then_observed_interval'])}
              for r in o['joint_post_clock_outcomes']]
    law=c['finite_rational_gate_record_bounds']
    energy=g['actual_charged_ground_energy_interval']
    fidelity=F(law['accepted_source_ground_fidelity_lower']);scale=10**14
    fidelity_floor=fidelity.numerator*scale//fidelity.denominator
    return {'schema':'oph.viewer.charged_field.v1',
        'source':{'sites':13,'edges':42,'fieldStates':5456,'chargedStates':70928,
                  'fullPhysicalStates':44695552,'referenceVertex':1,'referenceEnergy':1,
                  'totalFieldOccupationCutoff':3,'clock':'exp(−it H), ħ = 1',
                  'metric':'declared counting metric, not the Whitney Hodge metric'},
        'geometry':{'vertices':mesh['vertices'],'edges':mesh['edges'],
                    'display':'orthographic layout of the declared cone; no trajectory'},
        'energy':{'total':bounds(energy),'totalOutward':outward(energy),
                  'sourceOnly':bounds(g['source_energy_after_fixed_reference_subtraction']),
                  'sourceOnlyOutward':outward(g['source_energy_after_fixed_reference_subtraction']),
                  'dressingLowering':bounds(g['energy_lowering_from_full_transverse_dressing_interval'])},
        'clock':{'qubits':20,'phaseLabels':2**20,'numberLabels':14,
                 'acceptedLabels':law['accepted_pointer_outcomes'],
                 'acceptedIndexRange':law['accepted_index_interval_inclusive'],
                 'pumpFailure':.75,'pumpSuccess':.25,
                 'acceptance':bounds(law['unconditional_accepted_record_interval']),
                 'rejectionAfterPump':bounds(law['unconditional_rejected_record_interval']),
                 'groundFidelityLower':str(fidelity),'groundFidelityFloor':f'0.{fidelity_floor:014d}',
                 'rationalGateFactors':c['finite_clock']['total_R33_factors_in_serial_binary_power_realization'],
                 'exactGroundProjection':False,'allFailuresAndLabelsRetained':True,
                 'commonAutonomousSourceEvolution':True},
        'records':records,
        'positions':[{'vertex':r['source_vertex'],'afterClock':bounds(r['conditional_post_clock_probability_interval'])}
                     for r in o['post_clock_position_marginals']],
        'fieldExcitation':{'afterClock':bounds(o['conditional_post_clock_nonvacuum_probability_interval']),
                           'original':bounds(o['unconditional_accepted_nonvacuum_probability_interval'])},
        'capacity':{'beforeJointPointer':bounds(c['capacity']['source_plus_apparatus_log_dimension_nats_interval']),
                    'jointPointerStates':26,'additionalNats':'ln(26)',
                    'formula':c['capacity']['source_plus_apparatus_log_dimension_formula']+' + ln(26)',
                    'extraSynthesisWorkspaceIncluded':False,'stateEntropy':False},
        'nativeAttachment':{'blocks':13,'cellsPerBlock':8,'actualNativeHamiltoniansRetained':True,
            'wholeBinTime':'1/4','radialArrivalLower':'3.06e-5','boundaryArrivalLower':'3.06e-5',
            'additionalGroundAdmissionService':True,'sameAsSpinlessFieldSource':False,
            'fullChargedGroundIdentified':False,'LorentzSpinIdentified':False,
            'explanation':'Odd native eᶜ probes couple to the finite field through retained conjugate references and charge ports; the selected cubic-spin channel has its own total energy and reference cost four.'},
        'scope':{'finiteBoundarySource':True,'groundProbabilitiesEqualPreparedProbabilities':False,
                 'jointMeasurementAfterFiniteClock':True,'all26OutcomesRetained':True,
                 'postMeasurementStateUnchanged':False,'standardModelIdentification':False,
                 'sourceDerivedGeV':False,'relativisticMassShell':False,'sameAsClassicalLoadRun':False},
        'evidence':{k:{'path':v,'sha256':digest(root/v),
                    'proof':'https://github.com/FloatingPragma/observer-patch-holography/blob/1bf5e5718/'+v.replace('results.json','PROOF.md')}
                    for k,v in SOURCES.items()},
        'geometryEvidence':{'path':GEOMETRY,'sha256':digest(root/GEOMETRY)}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    text=json.dumps(build(a.repository),indent=2,ensure_ascii=False,sort_keys=True)+'\n'
    if a.check:
        if OUTPUT.read_text()!=text:raise ValueError('Charged-field viewer evidence is stale')
    else:OUTPUT.write_text(text)
    print('PASS: complete finite-clock charge/field observations and declared native attachment')


if __name__=='__main__':main()
