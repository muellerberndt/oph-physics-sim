"""Export complete native massive-vector readout evidence for the viewer."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import itertools
import json
from pathlib import Path

from .quantum_evidence import bounds,digest,receipt
from .reference_repository import theory_repository

SOURCE='model/interacting_block_proca/results.json'
OUTPUT=Path(__file__).resolve().parent/'viewer2/proca_evidence.json'


def build(root=None):
    root=Path(root or theory_repository()).resolve()
    data=receipt(root,SOURCE);construction=data['construction']
    if (data['schema']!='oph.interacting_block_proca.v1'
            or construction['Standard_Model_particle_assignment'] is not None
            or construction['measured_inputs']
            or construction['reference_dimension']!=3072
            or not data['source_admission']['same_H_selected_actual_ground_reference_and_clock']
            or data['induced_spatial_and_boost_action']['finite_native_exact_boost_symmetry_claim']):
        raise ValueError('Unreviewed vector source or interpretation')
    members=[]
    for experiment,resource in zip(data['complete_actual_source_experiments'],data['explicit_source_capacities'],strict=True):
        if (experiment['spacing']!=resource['spacing'] or experiment['macro_side']!=resource['macro_side']
                or experiment['exact_supported_Fourier_modes']!=19
                or experiment['full_native_source_matrix_numerically_enumerated']
                or resource['full_source_matrix_numerically_enumerated']
                or not resource['full_native_complement_retained']):
            raise ValueError('Vector execution scope changed')
        error=F(experiment['complete_record_TV_error_upper'])
        if F(resource['complete_source_vs_finite_free_record_TV_upper_exact'])>error or error!=F(19,20*10**9):
            raise ValueError('Actual-ground and complete-record error changed')
        menu=set(itertools.product((1,-1),(0,1),(1,-1),(0,1)))
        records=[]
        for row in experiment['complete_sixteen_records']:
            key=tuple(row[k] for k in ('first_sign','first_flag','second_sign','second_flag'))
            if key not in menu:raise ValueError('Duplicate or invalid vector outcome')
            menu.remove(key)
            records.append({'a0Sign':key[0],'a0Flag':key[1],'a1Sign':key[2],'a1Flag':key[3],
                'ideal':bounds(row['ideal_finite_free_probability_interval']),
                'actual':bounds(row['actual_full_source_probability_interval'])})
        if menu:raise ValueError('Incomplete vector outcome menu')
        signals=[{'field':row['readout'],'probability':bounds(row['actual_minus_sign_probability_interval'])}
            for row in experiment['strictly_positive_actual_sign_marginals']]
        if len(signals)!=2 or any(row['probability']['display'][0]<=0 for row in signals):
            raise ValueError('Vector signal is not certified above complete error')
        capacity=resource['logical_capacity']
        if (resource['declared_readout_apparatus_dimension']!=49152
                or resource['readout_control_and_flag_qubits']!=4
                or not capacity['capacity_is_log_dimension_not_actual_state_entropy']
                or capacity['additional_gate_synthesis_or_ground_preparation_workspace_included']):
            raise ValueError('Unreviewed vector capacity accounting')
        members.append({'spacing':experiment['spacing'],'side':experiment['macro_side'],
            'sourceBoxSide':experiment['source_physical_box_side'],
            'normalizedBoxSide':experiment['unit_speed_box_side'],
            'momentumRadius':experiment['Fourier_support_radius'],
            'times':experiment['test_times_Delta_times_t'],
            'comparisonModes':experiment['exact_supported_Fourier_modes'],
            'records':records,'signals':signals,
            'completeError':{'exact':str(error),'display':float(error)},
            'anyLeakageFlagUpper':float(F(experiment['probability_of_any_Halmos_flag_one_upper'])),
            'replicasPerSite':str(resource['actual_replicas_per_macrocell_N']),
            'macroSites':resource['macro_sites'],'nativeCells':str(resource['actual_native_source_cells']),
            'referenceDimension':3072,'readoutBits':4,'apparatusDimension':49152,
            'capacity':{'nats':bounds(capacity['logHilbertDimension_nats_interval_exact']),
                'outwardDisplay':capacity['logHilbertDimension_nats_display_interval'],
                'formula':'NV(240 ln 2 + 8 ln 5 + 24 ln 119) + ln 3072 + 4 ln 2',
                'scope':capacity['carrier_scope'],'replicaOrigin':capacity['replica_number_origin'],
                'below1e122':capacity['below_10_power_122_nats'],
                'universeCapacityIdentification':False,'extraSynthesisWorkspaceIncluded':False}})
    if len(members)!=2:raise ValueError('Expected two audited vector members')
    return {'schema':'oph.viewer.native_proca.v1',
        'source':{'constituentCells':8,'canonicalComponents':3,'fieldComponents':4,'comparisonPolarizations':3,
            'sameHamiltonianStateReferenceClock':True,'referenceDimension':3072,
            'restFrequency':'actual native gap Δ','effectiveMassParameter':'Δ/c²',
            'speed':'c=Δ/√6 in declared source units','normalizedCoordinates':'τ=Δt, ξ=√6x'},
        'filters':{'temporal':'A₀(p)=i p·Π(p)/Ωₐ(p)',
            'spatial':'Aᵢ(p)=[δᵢₐ+pᵢpₐ/(Ω₀(p)+1)]Φₐ(p)',
            'units':'source-derived mass-one, speed-one coordinates',
            'timeComponentUsesCanonicalMomentum':True},
        'members':members,
        'scope':{'newObservableMap':True,'freeProcaComparison':True,'sameNativeSource':True,
            'finiteSourceExactLorentzSymmetry':False,'interactingRelativisticLimit':False,
            'unfilteredTripletAutomaticallyVector':False,'standardModelIdentification':False,
            'sourceDerivedGeV':False,'fullNativeMatrixExecuted':False,'allSixteenOutcomesRetained':True,
            'replicasDerivedFromMemoryBudget':False},
        'evidence':{'path':SOURCE,'sha256':digest(root/SOURCE),
            'proof':'https://github.com/FloatingPragma/observer-patch-holography/blob/d1d0c28907c449e6c207e41d6909c9038cf75ca7/'+SOURCE.replace('results.json','PROOF.md')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path);parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    text=json.dumps(build(args.repository),ensure_ascii=False,indent=2,sort_keys=True)+'\n'
    if args.check:
        if OUTPUT.read_text()!=text:raise ValueError('Native massive-vector evidence is stale')
    else:OUTPUT.write_text(text)
    print('PASS: same-source vector filters, complete records and explicit capacity')


if __name__=='__main__':main()
