"""Publish complete, source-bound finite collision records for the viewer."""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path

from .quantum_evidence import bounds, digest, receipt
from .reference_repository import theory_repository

OUTPUT = Path(__file__).resolve().parent / 'viewer2' / 'collision_evidence.json'
SOURCE = 'model/interacting_block_scattering/results.json'


def build(root=None):
    root = Path(root or theory_repository()).resolve()
    data = receipt(root, SOURCE)
    source = data['interval_certified_five_block_collision']
    if (source['physical_source_cells'] != 40 or source['exact_reducing_source_dimension'] != 1024
            or source['actual_interacting_ground_input_claim']
            or source['three_dimensional_continuum_claim']):
        raise ValueError('Collision source or interpretation changed')
    rows = []
    for frame in source['complete_time_series']:
        if frame['classical_trajectories_claim'] or not frame['uniform_marginals_follow_exact_translation_symmetry']:
            raise ValueError('Unreviewed trajectory interpretation')
        sites = []
        for site in frame['occupation_marginals']:
            components = [{'label': label, 'ideal': bounds(c['ideal_interval_exact']),
                           'prepared': bounds(c['native_preparation_interval_exact'])}
                          for label, c in zip(('α', 'β', 'γ'), site['components'], strict=True)]
            sites.append({'id': site['block'], 'components': components,
                          'sourceOnlyComponents': [
                              {'label': label, 'ideal': bounds(c['ideal_interval_exact']),
                               'prepared': bounds(c['native_preparation_interval_exact'])}
                              for label, c in zip(('α', 'β', 'γ'), site['reference_traced_source_components'], strict=True)],
                          'ground': {'ideal': bounds(site['ground_ideal_interval_exact']),
                                     'prepared': bounds(site['ground_native_preparation_interval_exact'])},
                          'higherSectorUpper': float(Fraction(site['higher_native_sector_probability_upper_exact']))})
        rows.append({'time': float(Fraction(frame['clock_Delta_times_t'])),
                     'idealHit': bounds(frame['ideal_hit_probability_interval_exact']),
                     'hit': bounds(frame['native_preparation_hit_probability_interval_exact']),
                     'complement': bounds(frame['native_preparation_complement_probability_interval_exact']),
                     'samePreparedFree': bounds(frame['same_admitted_free_preparation_error_hit_interval_exact']),
                     'samePreparedFreeComplement': bounds(frame['same_admitted_free_preparation_error_complement_interval_exact']),
                     'difference': bounds(frame['native_minus_matched_free_hit_difference_interval_exact']),
                     'sites': sites})
    prep = source['complete_native_preparation']
    return {'schema': 'oph.viewer.finite_collision.v1', 'source': {
                'blocks': 5, 'sourceCells': 40, 'reducingDimension': 1024, 'components': 3,
                'graph': source['macro_graph'], 'onsiteOverGap': source['D_over_Delta'],
                'couplingOverGap': source['t_over_Delta'], 'input': source['input'], 'output': source['output'],
                'kinematicCapacityNatsUpper': 5*(240*math.log(2)+8*math.log(5)+24*math.log(119)),
                'executedReducingCapacityNats': math.log(1024),
                'nativeComplementNumericallyEnumerated': False},
            'preparation': {'filterDepth': prep['binary_filter_depth'],
                            'trialCount': prep['constituent_native_blocks'] * prep['trials_per_constituent'],
                            'logSuccessLower': float(prep['log_success_lower_decimal']),
                            'allFailureFlagsRetained': prep['all_preparation_flags_retained'],
                            'vectorErrorSquaredUpper': float(prep['retained_purification_vector_error_squared_upper_decimal'])},
            'frames': rows, 'scope': {'finiteNeutralInteraction': True, 'quantumMarginals': True,
                'classicalTrajectories': False, 'sameAsClassicalLoadRun': False, 'standardModelIdentification': False,
                'threeDimensionalContinuum': False, 'sourceDerivedGeV': False,
                'occupationRecordsRelativeToReference': True,
                'matchedFreeUsesSameAdmittedStates': True,
                'fullNativeComplementRetained': source['full_native_complement_retained_by_invariance']},
            'capacity': source['logical_capacity'],
            'reference': source['retained_symmetry_reference'],
            'evidence': {'path': SOURCE, 'sha256': digest(root / SOURCE),
                         'proof': 'https://github.com/FloatingPragma/observer-patch-holography/blob/main/' + SOURCE.replace('results.json', 'PROOF.md')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = json.dumps(build(args.repository), ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    if args.check:
        if OUTPUT.read_text() != text:
            raise ValueError('Finite collision evidence is stale')
    else:
        OUTPUT.write_text(text)
    print('PASS: complete source-bound finite collision records')


if __name__ == '__main__':
    main()
