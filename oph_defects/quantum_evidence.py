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
