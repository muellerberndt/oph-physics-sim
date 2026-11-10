"""The particle table of the viewer, built from the candidate map of the theory repository.

    python -m oph_defects.particle_table            # rewrites oph_defects/particle_map.json

Sources, read at build time and recorded with their hashes:

* ``model/particle_candidate_confirmation/results.json``: eighteen named comparison targets, eleven
  protected-object entries, sixteen source excitation identifiers, every status verbatim;
* ``model/standard_model_source_map/candidates/correspondence.json``: internal gauge weights and the
  benchmark quantum numbers of the targets.

Added here and marked as such: the Z6 class ``Q6 mod 6`` of every internal component, which is the
reading that connects the flux classes of the declared register to matter components (a hypothesis of
the simulation side, not a result of the theory repository), and measured comparison masses.
Nothing in this table is an identified particle; the theory repository lists no physical particle map
and no source-derived mass.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json, pathlib
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
from .reference_repository import theory_repository
RER = theory_repository()
OUT = HERE / 'particle_map.json'

FAMILIES = [
    {'id': 'A', 'label': 'Family A: centre flux 1 and its conjugate 5 · they generate the centre', 'classes': [1, 5], 'color': '#ff7a45',
     'note': 'classes 1 and 5 are conjugates; a pair is born from two flat faces and annihilates on contact. Seen by every quark pattern and every weak doublet'},
    {'id': 'B', 'label': 'Family B: colour-centre flux 2 and its conjugate 4', 'classes': [2, 4], 'color': '#ffb347',
     'note': 'classes 2 and 4 are conjugates; produced directly by a contrast of 4 or 5, or by fusion of two class-1 fluxes. Seen by quark patterns only'},
    {'id': 'C', 'label': 'Family C: weak-centre flux 3', 'classes': [3], 'color': '#3ee6a8',
     'note': 'class 3 is its own conjugate; produced directly by a contrast of 5, or by fusion of classes 1 and 2. Seen by weak doublets only'},
]
SHORT = {1: 'Φ1', 5: 'Φ5', 2: 'Φ2', 4: 'Φ4', 3: 'Φ3'}
DESCRIBE = {'Q_L': 'quark doublet Q_L (up and down type, left)', 'u^c': 'up-type antiquark singlet u^c',
            'd^c': 'down-type antiquark singlet d^c', 'L_L': 'lepton doublet L_L (neutrino and charged lepton, left)',
            'e^c': 'charged antilepton singlet e^c'}
# measured comparison values, GeV (Particle Data Group 2024 central values; quark masses in the MS-bar
# conventions of the review). Comparison data: not an input of any run and not a prediction.
MEASURED_GEV = {'electron': 0.00051099895, 'muon': 0.1056583755, 'tau': 1.77693, 'up': 0.00216, 'down': 0.00470,
                'strange': 0.0935, 'charm': 1.2730, 'bottom': 4.183, 'top': 172.57, 'W_plus': 80.3692, 'W_minus': 80.3692,
                'Z': 91.1880, 'Higgs': 125.20, 'photon': 0.0, 'gluon': 0.0, 'electron_neutrino': None, 'muon_neutrino': None,
                'tau_neutrino': None}
MEASURED_NOTE = {'electron_neutrino': 'flavour state, no single mass; kinematic bound below 0.8 eV',
                 'muon_neutrino': 'flavour state, no single mass', 'tau_neutrino': 'flavour state, no single mass',
                 'gluon': 'massless gauge field, confined', 'up': 'MS-bar at 2 GeV', 'down': 'MS-bar at 2 GeV', 'strange': 'MS-bar at 2 GeV',
                 'charm': 'MS-bar at its own scale', 'bottom': 'MS-bar at its own scale', 'top': 'direct measurements'}


# ---------------------------------------------------------------------------- presentation: groups, markers, chart
# Colour-coded particle groups of the viewer. Colours are presentation only.
GROUPS = [
    {'id': 'quark', 'label': 'Quarks', 'color': '#ff7a45', 'blurb': 'spin 1/2 · three colours · feel the strong force'},
    {'id': 'lepton', 'label': 'Charged leptons', 'color': '#3ee6a8', 'blurb': 'spin 1/2 · electric charge −1 · no colour'},
    {'id': 'neutrino', 'label': 'Neutrinos', 'color': '#4cc3ff', 'blurb': 'spin 1/2 · neutral · only the weak force'},
    {'id': 'gluon', 'label': 'Gluons', 'color': '#ff5fb0', 'blurb': 'spin 1 · carry the strong force · eight colour states'},
    {'id': 'photon', 'label': 'Photon', 'color': '#ffe45c', 'blurb': 'spin 1 · carries electromagnetism · massless'},
    {'id': 'weak', 'label': 'W and Z bosons', 'color': '#b197ff', 'blurb': 'spin 1 · carry the weak force · massive'},
    {'id': 'higgs', 'label': 'Higgs boson', 'color': '#f4ead0', 'blurb': 'spin 0 · gives mass in the Standard Model'},
]
# Marker of each register class on the screen. A class is a flux of the centre Z6 of the gauge group: a pattern of
# charge Q (six times the hypercharge) sees flux k as the phase exp(2 pi i k Q / 6). Class 3 is the central -1 of the
# weak doublet group, classes 2 and 4 the centre of the colour group, classes 1 and 5 generate. The earlier reading
# (class k = the particle of charge k) is a duality that nothing in the construction supplies (theory session, 11-10).
CLASS_STYLE = {
    # owner's reading, 2026-11-10: a defect class is the particle with the same charges. The flux reading stays in 'detail'.
    1: {'group': 'quark', 'color': '#ff7a45', 'shape': 'disc', 'anti': False, 'short': 'qL', 'family': 'A',
        'label': 'left-quark-type candidate', 'detail': 'class 1: colour and weak charge. As a flux: seen by every quark pattern and by the lepton doublet'},
    5: {'group': 'quark', 'color': '#ff7a45', 'shape': 'ring', 'anti': True, 'short': 'q̄L', 'family': 'A',
        'label': 'left-antiquark-type candidate', 'detail': 'class 5, the conjugate of class 1: annihilates with it'},
    4: {'group': 'quark', 'color': '#ffb347', 'shape': 'disc', 'anti': False, 'short': 'qR', 'family': 'B',
        'label': 'right-quark-type candidate', 'detail': 'class 4: colour charge only. As a flux: the centre of the colour group, seen by quark patterns only'},
    2: {'group': 'quark', 'color': '#ffb347', 'shape': 'ring', 'anti': True, 'short': 'q̄R', 'family': 'B',
        'label': 'right-antiquark-type candidate', 'detail': 'class 2, the conjugate of class 4'},
    3: {'group': 'lepton', 'color': '#3ee6a8', 'rim': '#4cc3ff', 'shape': 'duo', 'anti': False, 'short': 'ℓL', 'family': 'C',
        'label': 'lepton-doublet-type candidate', 'detail': 'class 3: weak charge only; its own conjugate class; the Higgs has the same class. As a flux: the sign of a link'},
}
QUARK = {'spin': '1/2', 'statistics': 'fermion', 'colour': 'triplet: red, green or blue', 'baryon number': '1/3', 'lepton number': '0',
         'forces': 'strong, electromagnetic, weak, gravity', 'chirality': 'left-handed part in a weak doublet, right-handed part a weak singlet'}
LEPTON = {'spin': '1/2', 'statistics': 'fermion', 'colour': 'none', 'baryon number': '0', 'lepton number': '1',
          'forces': 'electromagnetic, weak, gravity', 'chirality': 'left-handed part in a weak doublet, right-handed part a weak singlet'}
NEUTRINO = {'spin': '1/2', 'statistics': 'fermion', 'colour': 'none', 'baryon number': '0', 'lepton number': '1', 'forces': 'weak, gravity',
            'chirality': 'only left-handed neutrinos and right-handed antineutrinos are observed',
            'mass': 'oscillations require nonzero mass-squared differences; a flavour state is not a mass eigenstate'}
UP = {'electric charge': '+2/3', 'weak isospin T3 (left)': '+1/2', 'hypercharge Y (left / right)': '1/6 / 2/3'}
DOWN = {'electric charge': '−1/3', 'weak isospin T3 (left)': '−1/2', 'hypercharge Y (left / right)': '1/6 / −1/3'}
CH = {'electric charge': '−1', 'weak isospin T3 (left)': '−1/2', 'hypercharge Y (left / right)': '−1/2 / −1'}
NU = {'electric charge': '0', 'weak isospin T3 (left)': '+1/2', 'hypercharge Y (left)': '−1/2'}
CHART = {   # target -> (group, row, column, symbol, antisymbol, name, generation, properties)
    'up': ('quark', 1, 1, 'u', 'ū', 'up', 1, {**QUARK, **UP, 'lifetime': 'stable inside hadrons'}),
    'charm': ('quark', 1, 2, 'c', 'c̄', 'charm', 2, {**QUARK, **UP, 'lifetime': 'decays by the weak force'}),
    'top': ('quark', 1, 3, 't', 't̄', 'top', 3, {**QUARK, **UP, 'lifetime': 'about 5 × 10⁻²⁵ s; decays before it can form a hadron'}),
    'down': ('quark', 2, 1, 'd', 'd̄', 'down', 1, {**QUARK, **DOWN, 'lifetime': 'stable inside nuclei'}),
    'strange': ('quark', 2, 2, 's', 's̄', 'strange', 2, {**QUARK, **DOWN, 'lifetime': 'decays by the weak force'}),
    'bottom': ('quark', 2, 3, 'b', 'b̄', 'bottom', 3, {**QUARK, **DOWN, 'lifetime': 'decays by the weak force'}),
    'electron': ('lepton', 3, 1, 'e<sup>−</sup>', 'e<sup>+</sup>', 'electron', 1, {**LEPTON, **CH, 'lifetime': 'stable'}),
    'muon': ('lepton', 3, 2, 'μ<sup>−</sup>', 'μ<sup>+</sup>', 'muon', 2, {**LEPTON, **CH, 'lifetime': '2.197 × 10⁻⁶ s'}),
    'tau': ('lepton', 3, 3, 'τ<sup>−</sup>', 'τ<sup>+</sup>', 'tau', 3, {**LEPTON, **CH, 'lifetime': '2.90 × 10⁻¹³ s'}),
    'electron_neutrino': ('neutrino', 4, 1, 'ν<sub>e</sub>', 'ν̄<sub>e</sub>', 'electron neutrino', 1, {**NEUTRINO, **NU, 'lifetime': 'stable'}),
    'muon_neutrino': ('neutrino', 4, 2, 'ν<sub>μ</sub>', 'ν̄<sub>μ</sub>', 'muon neutrino', 2, {**NEUTRINO, **NU, 'lifetime': 'stable'}),
    'tau_neutrino': ('neutrino', 4, 3, 'ν<sub>τ</sub>', 'ν̄<sub>τ</sub>', 'tau neutrino', 3, {**NEUTRINO, **NU, 'lifetime': 'stable'}),
    'gluon': ('gluon', 1, 4, 'g', 'g', 'gluon', 0, {'spin': '1', 'statistics': 'boson', 'electric charge': '0', 'colour': 'octet: eight colour states',
              'forces': 'carries the strong force and feels it', 'helicity states': '2', 'antiparticle': 'its own, as a set of eight', 'lifetime': 'never free: confined'}),
    'photon': ('photon', 2, 4, 'γ', 'γ', 'photon', 0, {'spin': '1', 'statistics': 'boson', 'electric charge': '0', 'colour': 'none', 'parity': '−',
               'forces': 'carries electromagnetism', 'helicity states': '2', 'antiparticle': 'its own', 'lifetime': 'stable'}),
    'Z': ('weak', 3, 4, 'Z', 'Z', 'Z boson', 0, {'spin': '1', 'statistics': 'boson', 'electric charge': '0', 'colour': 'none', 'forces': 'carries the neutral weak force',
          'polarisation states': '3', 'antiparticle': 'its own', 'width': '2.4955 GeV', 'lifetime': 'about 3 × 10⁻²⁵ s'}),
    'W_plus': ('weak', 4, 4, 'W<sup>+</sup>', 'W<sup>−</sup>', 'W boson', 0, {'spin': '1', 'statistics': 'boson', 'electric charge': '+1', 'colour': 'none',
               'forces': 'carries the charged weak force; couples to left-handed particles only', 'polarisation states': '3', 'width': '2.085 GeV', 'lifetime': 'about 3 × 10⁻²⁵ s'}),
    'W_minus': ('weak', 4, 4, 'W<sup>−</sup>', 'W<sup>+</sup>', 'W boson', 0, {'spin': '1', 'statistics': 'boson', 'electric charge': '−1', 'colour': 'none',
                'forces': 'carries the charged weak force; couples to left-handed particles only', 'polarisation states': '3', 'width': '2.085 GeV', 'lifetime': 'about 3 × 10⁻²⁵ s'}),
    'Higgs': ('higgs', 1, 5, 'H', 'H', 'Higgs boson', 0, {'spin': '0', 'statistics': 'boson', 'electric charge': '0', 'colour': 'none', 'parity': '+',
              'weak isospin': 'member of a doublet with hypercharge 1/2', 'antiparticle': 'its own', 'width': 'about 4 MeV (Standard Model value)', 'lifetime': 'about 1.6 × 10⁻²² s'}),
}
CARRIER = {   # best-guess carrier of each group in the model, and what is established about it
    'quark': 'occupation patterns with one or two of the three vector levels filled: (1,1) for the left-handed doublet, (1,2) and (1,0) for the '
             'right-handed up and down singlets. A quark pattern sees every flux class of the runs',
    'lepton': 'pattern (3,1) for the left-handed part, shared with the neutrino, and (3,0) for the right-handed part. The left-handed part sees the '
              'flux classes 1, 5 and 3; the right-handed part sees none',
    'neutrino': 'pattern (3,1), the neutral member of the left-handed lepton doublet. A right-handed partner would be (3,2), charge 0, which no flux sees',
    'gluon': 'no pattern: the eight traceless port responses on the three vector levels. Not a defect',
    'photon': 'no pattern: a combination of the conserved charge and the diagonal response on the two spinor levels. A wave, not a defect',
    'weak': 'no pattern: the responses on the two spinor levels, off-diagonal for the W, diagonal for the Z. No massive vector is assigned',
    'higgs': 'no pattern: the operator that adds or removes one quantum on the spinor levels. It changes the charge by 3 and links each left-handed pattern '
             'to its right-handed partners: (1,1) with (1,2) and (1,0), (3,1) with (3,0) and (3,2). A mass term sits on these links. Declared, no value',
}
# ---------------------------------------------------------------------------- internal patterns (best guess, revised 2026-11-10)
# Five internal levels: three vector levels and two spinor levels of a carrier (the section multiplets of the repair-law
# search; the fibre of the public program coherent-seam-repair-v1 with conserved charge diag(2,2,2,-3,-3)). A multiplet
# is an occupation pattern (n3, n2): C(3,n3) C(2,n2) states, charge 3 n2 - 2 n3 = six times the hypercharge, register
# class = that charge modulo 6, handedness = parity of n3 + n2. This is the exterior grammar of the papers on five modes.
SECTOR_NAMES = {
    (0, 0): ('conjugate of the right-handed neutrino', 'neutrino', []),
    (1, 0): ('down-type quark, right-handed', 'quark', ['down', 'strange', 'bottom']),
    (2, 0): ('up-type antiquark', 'quark', ['up', 'charm', 'top']),
    (3, 0): ('charged lepton, right-handed', 'lepton', ['electron', 'muon', 'tau']),
    (0, 1): ('antilepton doublet', 'lepton', []),
    (1, 1): ('quark doublet, left-handed', 'quark', ['up', 'down', 'charm', 'strange', 'top', 'bottom']),
    (2, 1): ('antiquark doublet', 'quark', []),
    (3, 1): ('lepton doublet, left-handed', 'lepton', ['electron', 'muon', 'tau', 'electron_neutrino', 'muon_neutrino', 'tau_neutrino']),
    (0, 2): ('charged antilepton', 'lepton', []),
    (1, 2): ('up-type quark, right-handed', 'quark', ['up', 'charm', 'top']),
    (2, 2): ('down-type antiquark', 'quark', []),
    (3, 2): ('right-handed neutrino', 'neutrino', []),
}
COLOUR_OF = {0: 'singlet', 1: 'triplet', 2: 'antitriplet', 3: 'singlet'}
WEAK_OF = {0: 'singlet', 1: 'doublet', 2: 'singlet'}


def sectors() -> list:
    from math import comb
    rows = []
    for n2 in range(3):
        for n3 in range(4):
            q = 3 * n2 - 2 * n3
            name, group, targets = SECTOR_NAMES[(n3, n2)]
            rows.append({'n3': n3, 'n2': n2, 'states': comb(3, n3) * comb(2, n2), 'six_Y': q, 'z6_class': q % 6,
                         'handedness': 'left' if (n3 + n2) % 2 == 0 else 'right', 'colour': COLOUR_OF[n3], 'weak': WEAK_OF[n2],
                         'multiplet': name, 'group': group, 'targets': targets, 'conjugate': [3 - n3, 2 - n2]})
    return rows


UPQ = [{'pattern': [1, 1], 'role': 'left-handed part, upper member of the quark doublet'}, {'pattern': [1, 2], 'role': 'right-handed part'}]
DNQ = [{'pattern': [1, 1], 'role': 'left-handed part, lower member of the quark doublet'}, {'pattern': [1, 0], 'role': 'right-handed part'}]
CHL = [{'pattern': [3, 1], 'role': 'left-handed part, lower member of the lepton doublet'}, {'pattern': [3, 0], 'role': 'right-handed part'}]
NUL = [{'pattern': [3, 1], 'role': 'left-handed, upper member of the lepton doublet'}, {'pattern': [3, 2], 'role': 'right-handed partner, not observed'}]
TARGET_SECTORS = {
    'up': UPQ, 'charm': UPQ, 'top': UPQ, 'down': DNQ, 'strange': DNQ, 'bottom': DNQ, 'electron': CHL, 'muon': CHL, 'tau': CHL,
    'electron_neutrino': NUL, 'muon_neutrino': NUL, 'tau_neutrino': NUL,
    'Higgs': [],
    'gluon': [], 'photon': [], 'Z': [], 'W_plus': [], 'W_minus': [],
}
YUKAWA_LINKS = [   # the Higgs as an operator: adds or removes one spinor-level quantum (theory session, 11-10)
    {'left': [1, 1], 'right': [1, 2], 'what': 'up-type quarks'}, {'left': [1, 1], 'right': [1, 0], 'what': 'down-type quarks'},
    {'left': [3, 1], 'right': [3, 0], 'what': 'charged leptons'}, {'left': [3, 1], 'right': [3, 2], 'what': 'neutrinos'},
]
# ---------------------------------------------------------------------------- best-guess identification (2026-11-10)
# The five levels with their electric charges: a vector level carries -1/3; the two spinor levels are the charged
# (T3 = +1/2, charge +1) and the neutral (T3 = -1/2, charge 0) member of a doublet with hypercharge 1/2.
LEVELS = [
    {'id': 'v1', 'kind': 'vector', 'label': 'vector level 1', 'colour': 'red', 'electric': '-1/3', 'six_Y': -2},
    {'id': 'v2', 'kind': 'vector', 'label': 'vector level 2', 'colour': 'green', 'electric': '-1/3', 'six_Y': -2},
    {'id': 'v3', 'kind': 'vector', 'label': 'vector level 3', 'colour': 'blue', 'electric': '-1/3', 'six_Y': -2},
    {'id': 's1', 'kind': 'spinor', 'label': 'spinor level 1, charged', 'T3': '+1/2', 'electric': '+1', 'six_Y': 3},
    {'id': 's2', 'kind': 'spinor', 'label': 'spinor level 2, neutral', 'T3': '-1/2', 'electric': '0', 'six_Y': 3},
]
# name of the state for each pattern; doublets (n2 = 1) are named by the spinor level that is filled
STATE_NAMES = {
    '0,0': {'*': ('ν̄', 'antineutrino, left-handed singlet (conjugate of a right-handed neutrino)', 'neutrino')},
    '1,0': {'*': ('d', 'down quark, right-handed', 'quark')},
    '2,0': {'*': ('ū', 'up antiquark, left-handed', 'quark')},
    '3,0': {'*': ('e⁻', 'electron, right-handed', 'lepton')},
    '0,1': {'s1': ('e⁺', 'positron, right-handed', 'lepton'), 's2': ('ν̄', 'antineutrino, right-handed', 'neutrino')},
    '1,1': {'s1': ('u', 'up quark, left-handed', 'quark'), 's2': ('d', 'down quark, left-handed', 'quark')},
    '2,1': {'s1': ('d̄', 'down antiquark, right-handed', 'quark'), 's2': ('ū', 'up antiquark, right-handed', 'quark')},
    '3,1': {'s1': ('ν', 'neutrino, left-handed', 'neutrino'), 's2': ('e⁻', 'electron, left-handed', 'lepton')},
    '0,2': {'*': ('e⁺', 'positron, left-handed', 'lepton')},
    '1,2': {'*': ('u', 'up quark, right-handed', 'quark')},
    '2,2': {'*': ('d̄', 'down antiquark, left-handed', 'quark')},
    '3,2': {'*': ('ν', 'neutrino, right-handed', 'neutrino')},
}
# objects of the executable model and their best-guess identity (audit: model/coherent_defect_wave/species/CANDIDATES.md;
# theory session: repair-law/work/FAMILY). 'in_model' says where the object exists as executed code.
MODEL_OBJECTS = [
    {'object': 'vector-type quantum (wave channels 0 to 2)', 'best_guess': 'the colour index of matter: one vector level filled is the right-handed down quark',
     'in_model': 'executed: three identical wave channels of the public walk, blind to the central link', 'confirmed': 'conserved charge −2 per level; no response to a weak-centre flux',
     'missing': 'the three copies have identical spectra; colour forces and a mass', 'status': 'labelled only', 'group': 'quark'},
    {'object': 'spinor-type quantum (wave channels 3 and 4)', 'best_guess': 'the weak doublet index of matter: one spinor level filled is the antilepton doublet',
     'in_model': 'executed: two identical wave channels of the public walk, sensitive to the central link', 'confirmed': 'conserved charge +3 per level; sign change at a flipped link; moves the flux and is changed by it',
     'missing': 'physical spin and chirality; a two-level register is not weak isospin yet', 'status': 'labelled only', 'group': 'lepton'},
    {'object': 'occupation patterns of the five levels', 'best_guess': 'one generation of quarks and leptons and its conjugate: twelve patterns, 32 states',
     'in_model': 'executed as an occupation lift with a shared link register (audit, species package); the patterns themselves are representation labels', 'confirmed': 'charges, anomaly sums zero, sum T3² / sum Q² = 3/8',
     'missing': 'a map to propagating one-fermion fields; a dynamics that keeps a pattern together', 'status': 'labelled only', 'group': 'quark'},
    {'object': 'regional flux of the link register', 'best_guess': 'a flux of the weak centre (class 3 of the big runs), not a matter particle',
     'in_model': 'executed: thirty link bits, protected with a held boundary, moved by the wave', 'confirmed': 'boundary sign −1, five positions, exact two-way interaction with the spinor type',
     'missing': 'its dispersion, statistics and rotation law; protection without a held boundary', 'status': 'protected relative to a boundary', 'group': 'weak'},
    {'object': 'the twelve port responses', 'best_guess': 'the gauge fields: eight on the vector levels (gluons), three on the spinor levels (W and the neutral weak field), one central (hypercharge)',
     'in_model': 'exact algebra; executed as port pulses on the internal factor and as the encoded controller of the audit', 'confirmed': 'dimension 12, centre 1, every carrier rotation is a word of at most six pulses',
     'missing': 'dynamics of the fields themselves; the responses must be the complete response of the law', 'status': 'algebra only', 'group': 'gluon'},
    {'object': 'the operator on the neutral spinor level', 'best_guess': 'the Higgs: the one operator that conserves colour and electric charge and links left-handed to right-handed patterns',
     'in_model': 'representation only; the audit lists the three permitted mass channels', 'confirmed': 'one amplitude gives one common mass to all four links',
     'missing': 'a value; separate masses and three generations need more structure', 'status': 'declared, no value', 'group': 'higgs'},
    {'object': 'the load wave', 'best_guess': 'no particle: the massless carrier of the conserved load',
     'in_model': 'executed at every scale: committed rule (diffusion) and coherent walk (massless branch)', 'confirmed': 'massless; speed 0.0207 mesh edges per step on the flat family, 0.0962 lattice constants per tick on the cubic family',
     'missing': 'a spin; a relation to the photon would need a transverse structure', 'status': 'no particle name', 'group': 'photon'},
    {'object': 'the conjugate triplet of the carrier (unused by the five levels)', 'best_guess': 'the space of three generations',
     'in_model': 'representation only', 'confirmed': 'a port and a cubic frame are mismatched by the golden-ratio pattern: squared overlaps 0.276, 1/2 and 0',
     'missing': 'a dynamics that selects a port for one sector and a frame for another; a breaking that splits three identical levels', 'status': 'candidate, not blind', 'group': 'neutrino'},
]
GENERATIONS = {
    'statement': 'Three generations are not in the five levels. The best candidate so far is the second triplet of the carrier, which the five levels do not use. '
                 'If it were the generation space, one kind of mass would be diagonal along a port and the other along a cubic frame of the carrier, and the two bases are mismatched by a fixed pattern.',
    'mixing': [{'angle': 'θ12', 'model': 0.276393, 'model_exact': '(5 − √5)/10', 'measured': 'about 0.31'},
               {'angle': 'θ23', 'model': 0.5, 'model_exact': '1/2', 'measured': '0.45 to 0.57'},
               {'angle': 'θ13', 'model': 0.0, 'model_exact': '0', 'measured': 'about 0.022'}],
    'caveats': 'Squared sines of the lepton mixing angles. The comparison is not blind: the measured values were known when it was made. The third angle is zero in the pattern and is not zero in nature. '
               'Three identical levels have equal spectra, so they cannot be three particles of different mass without a breaking that nothing supplies yet. Found by the theory session.',
    'status': 'candidate; no generation is derived',
}
RESPONSE_OF = {
    'Higgs': 'the operator that adds or removes one quantum on the spinor levels; it changes n2 by one and the charge by 3',

    'gluon': 'the eight traceless port responses on the three vector levels',
    'photon': 'the combination of the charge diag(2,2,2,-3,-3) and the diagonal response on the spinor levels that leaves the vacuum pattern alone',
    'Z': 'the other combination of the charge and the diagonal response on the spinor levels',
    'W_plus': 'the raising response on the two spinor levels', 'W_minus': 'the lowering response on the two spinor levels',
}
INTERNAL_MODES = {
    'levels': [{'id': 'v1', 'kind': 'vector', 'charge': -2}, {'id': 'v2', 'kind': 'vector', 'charge': -2}, {'id': 'v3', 'kind': 'vector', 'charge': -2},
               {'id': 's1', 'kind': 'spinor', 'charge': 3}, {'id': 's2', 'kind': 'spinor', 'charge': 3}],
    'statement': 'Five internal levels of a carrier: three vector levels and two spinor levels. A multiplet is an occupation pattern (n3, n2). '
                 'Its charge 3 n2 - 2 n3 is six times the hypercharge, and the parity of n3 + n2 is the handedness. The charge modulo 6 decides which centre fluxes the pattern sees. '
                 'The twelve patterns are one generation of the Standard Model and its conjugate. Generations are not separated by the pattern.',
    'origin': 'The vector triplet is the consensus space of vector agreement on pairs of faces, the spinor doublet that of spinor agreement on faces (repair-law search, 10 November). '
              'The public program coherent-seam-repair-v1 carries exactly this five-level factor on its walk and conserves the charge. '
              'The port loads compressed to it generate u(1) + su(2) + su(3).',
    'checks': 'On each of the two handedness classes the four anomaly sums vanish, and the sum of T3 squared over the sum of electric charge squared is 3/8. '
              'The sum of the twelve port responses is the charge itself, diag(2,2,2,-3,-3). Which class is called matter is a convention: nothing supplies chirality yet.',
    'higgs': 'The Higgs is not a pattern. It is the operator that adds or removes one spinor-level quantum, so it links each left-handed pattern to its right-handed partners: '
             '(1,1) with (1,2) and (1,0), (3,1) with (3,0) and (3,2). A mass term sits on these links. Of the five such operators exactly one conserves colour and electric charge, the one on the neutral spinor level. '
             'So one amplitude gives one common mass to up type, down type, charged lepton and neutrino: separate masses and three generations need structure beyond one occupation algebra. Declared, no value. '
             'The gauge fields are the twelve port responses and leave the pattern unchanged.',
    'fluxes': 'The defects of the big runs are not patterns. A register class k is a flux of the centre of the gauge group: a pattern of charge Q sees it as the phase exp(2 pi i k Q / 6). '
              'Class 3 is the central minus one of the weak group, seen by patterns with one spinor level filled; classes 2 and 4 are the centre of the colour group, seen by patterns with one or two vector levels filled; classes 1 and 5 generate.',
    'status': 'A labelling, revised best guess. No pattern is an identified particle and no mass follows from it.',
    'splits': 'An earlier version of this page read flux class k as the particle of charge k. That is a duality between fluxes and charges which the construction does not supply; particle names now sit on patterns only.',
}
REGISTER_COST = {'1': 5, '5': 5, '2': 8, '4': 8, '3': 10}


def mass_status(here: Path) -> dict:
    """What the model has today in place of rest masses: declared register costs and the band levels of the wave
    operator on the symmetric wiring (oph_defects.bloch), none assigned to a particle."""
    out = {'statement': 'No rest mass is derived from the source yet. The measured values are comparison targets.',
           'register_cost': {'table': REGISTER_COST, 'note': 'declared cost of a defect of each class in the register of these runs, in cost units; '
                             'five cost units are paid by one unit of released load potential. Declared, not derived.'}}
    f = here.parent / 'runs' / 'light' / 'bloch.json'
    if f.is_file():
        res = json.loads(f.read_text())
        hc = next(r for r in res if r['wiring'] == 'honeycomb3')
        moving = [{'level': b['level_at_zero'], 'curvature_over_massless': b['curvature_over_massless']} for b in hc['bands']
                  if not b['flat'] and b['level_at_zero'] > 1e-9]
        flat = sorted({round(b['level_at_zero'], 6) for b in hc['bands'] if b['flat']})
        out['wave_bands'] = {'wiring': 'symmetric wiring of the federation (three glued ports per carrier)', 'massless_c_squared': hc['massless_c_squared'],
                             'moving_bands': moving, 'immobile_levels': flat, 'crossing': hc.get('crossing_point'),
                             'note': 'levels of the port operator at zero wave number. On a continuous flow with pulse rate nu a band has rest frequency '
                                     'nu sqrt(level); it shares the cone of the massless band only if its curvature equals that of the massless band. '
                                     'No band is assigned to a particle.'}
    f = here.parent / 'runs' / 'light' / 'g1_bands.json'
    if f.is_file():
        g = json.loads(f.read_text())
        tb = g['transverse']['bands']
        light = next(b for b in tb if not b['flat'] and abs(b['level_at_zero']) < 1e-9)
        out['transverse_bands'] = {'massless_c_squared': light['curvature'],
                                   'moving_bands': [{'level': b['level_at_zero'], 'curvature_over_massless': b['curvature'] / light['curvature']}
                                                    for b in tb if not b['flat'] and b['level_at_zero'] > 1e-9],
                                   'immobile_levels': sorted({round(b['level_at_zero'], 6) for b in tb if b['flat'] and b['level_at_zero'] > 1e-9}),
                                   'note': 'seam and face modes of the symmetric wiring; they have no load component. The plaquette boundary inside a carrier is a declared choice.'}
    f = here.parent / 'runs' / 'light' / 'wiring_atlas.json'
    if f.is_file():
        rows = [r for r in json.loads(f.read_text()) if r['massless_anisotropy'] < 1e-4]
        out['cone_wirings'] = [{'up': sorted(r['up_distances']), 'down': sorted(r['down_distances']), 'massless_c_squared': r['massless_curvature'],
                                'half_gap': r['half_gap'], 'level': r['crossing_level'], 'slope': r['cone_slope']} for r in sorted(rows, key=lambda r: r['half_gap'])]
    f = here.parents[1] / 'repair-law' / 'work' / 'SIM' / 'bands' / 'twisted_bands.json'
    if f.is_file():
        tb = json.loads(f.read_text())
        def row(name, key, d):
            r = tb[key]; cur = r['dirs'][0]['curvature'][-1]
            return {'sector': name, 'level': r['q0'][0], 'branches': d, 'flat_branches': sum(1 for c in cur if abs(c) < 1e-7),
                    'moving_curvature': [c for c in cur if abs(c) >= 1e-7]}
        out['flux_gaps'] = {'scalar_curvature': tb['scalar']['dirs'][0]['curvature'][-1][0],
                            'rows': [row('spinor doublet, port frame', 'spinor convex (flux 1)', 2), row('spinor doublet, conjugate frame', 'spinor great (flux 7)', 2),
                                     row('vector triplet', 'vector convex helicity 1 (flux 2)', 3)],
                            'note': 'Lowest levels of the recharted port operator on the flat federation with one glue port per neighbour, in units of the kernel (relaxation rate times 2M). '
                                    'The level is the flux gap of one carrier and does not depend on the size of the federation. Flat branches do not move. '
                                    'Under the coherent walk a level becomes a frequency; the length scale of this gap is 0.14 mesh edges, a gap at the cutoff. Not assigned to a particle.'}
    f = here.parents[1] / 'repair-law' / 'work' / 'SIM' / 'bands' / 'walk_spectrum.json'
    if f.is_file():
        ws = json.loads(f.read_text())
        fam = []
        for g, label in (('bridge', 'one glue port per neighbour'), ('seam', 'a shared seam per neighbour, common frame')):
            G = ws[g]; v0 = G['massless_speed_edges_per_step']; rows = []
            for key, rr in G['sectors'].items():
                for r in rr:
                    v = (abs(r['v2']) ** 0.5) * (1 if r['v2'] >= 0 else -1)
                    rows.append({'sector': key.split('|')[0], 'rest_phase': r['omega0'], 'flat': r['flat'], 'speed_ratio': None if r['flat'] else v / v0,
                                 'compton_edges': None if r['flat'] else abs(v) / r['omega0'], 'shell_up_to_q': r['qmax_1pct']})
            fam.append({'gluing': label, 'addresses': G['addresses'], 'massless_speed': v0, 'rows': rows})
        out['family_walk'] = {'gluings': fam,
                              'note': 'A declared extension of the public walk to the federation: the quantum uses the address register of the carrier that holds it. '
                                      'Its phases follow exactly from the bands of the recharted port operator. Rest phases are in radians per walk step, speeds in mesh edges per walk step. '
                                      'A branch with a rest phase and the speed of the massless branch has the form of a mass shell in this clock. The rest phases sit at the cutoff: '
                                      'the Compton lengths are a fraction of one mesh edge. Not assigned to a particle.'}
    import math
    s5 = 5 ** 0.5
    lam = [('singlet', 1, 1.0), ('slow triplet', 3, 11 / 12 + s5 / 60), ('quintet', 5, 0.9), ('fast triplet', 3, 11 / 12 - s5 / 60)]
    out['coherent_walk'] = {'program': 'coherent-seam-repair-v1', 'dimension': 1440,
                            'levels': [{'name': n, 'states': m, 'kernel': l, 'frequency': math.acos(l)} for n, m, l in lam],
                            'two_step_return': '643/900', 'committed_two_step_return': '101/120', 'reference_success': '131/810',
                            'fibre': {'dimension': 7200, 'charge': [-2, -2, -2, 3, 3], 'central_link_amplitude_change': '-1/6750'},
                            'note': 'Public program of the audit in the theory repository (model/coherent_seam_repair). The removed seam disagreement is kept in a coherent record and the seam '
                                    'address is reflected; the step is unitary and every relaxation factor of the reference rule becomes a frequency, in radians per walk step. '
                                    'One carrier only. A frequency of a finite carrier is not a rest mass.'}
    out['integer_filling'] = {'rest_cost': 0.5, 'pair_release': 1.0, 'note': 'on a whole-number background a raised port and a lowered port are particle and antiparticle of the bare law; '
                              'each costs one half of the mismatch potential and a pair annihilates with release exactly 1 (measured: 192,528 of 192,528 pairs at level 6).'}
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args, repository: Path = RER) -> str:
    try:
        return subprocess.run(['git', '-C', str(repository), *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ''


def z6(q6) -> int:
    return int(q6) % 6


def build(rer: Path = RER) -> dict:
    res_path = rer / 'model/particle_candidate_confirmation/results.json'
    cor_path = rer / 'model/standard_model_source_map/candidates/correspondence.json'
    res = json.loads(res_path.read_text())
    cor = json.loads(cor_path.read_text())
    multiplets = []
    comp_class, comp_mult = {}, {}
    for m in cor['internal_gauge_multiplets']:
        k = z6(m['Q6'])
        multiplets.append({'label': m['comparison_label'], 'description': DESCRIBE.get(m['comparison_label'], m['comparison_label']), 'SU3': m['SU3'], 'SU2_dimension': m['SU2_dimension'], 'Y': m['Y'],
                           'Q6': m['Q6'], 'T3': m['component_T3'], 'Q': m['component_Q'], 'component_ids': m['component_ids'],
                           'convention': m['convention'], 'z6_class': k, 'z6_class_conjugate': (6 - k) % 6})
        for cid in m['component_ids']:
            comp_class[cid] = k
            comp_mult[cid] = m['comparison_label']
    bench = {t['target']: t for t in cor['target_rows']}
    targets = []
    for t in res['target_rows']:
        b = bench.get(t['target'], {})
        steps = {s['id']: s for s in t['confirmation_steps']}
        classes = sorted({comp_class[c] for c in t['source_component_ids'] if c in comp_class})
        targets.append({
            'target': t['target'], 'theory_family': t['family'], 'status': t['physical_identity_status'],
            'benchmark': b.get('benchmark'), 'best_guess': b.get('best_guess'), 'comparison_status': b.get('comparison_status'),
            'missing': b.get('missing'), 'source_component_ids': t['source_component_ids'],
            'source_multiplets': sorted({comp_mult[c] for c in t['source_component_ids'] if c in comp_mult}),
            'actual_primary_excitation_ids': t['actual_primary_excitation_ids'],
            'source_test_summary': t['source_test_summary'], 'source_derived_mass_GeV': t['source_derived_mass_GeV'],
            'source_flavor_index': t['source_flavor_index'], 'test_receipt': t['test_receipt'],
            'supporting_receipts': t['supporting_receipts'],
            'finite_source_evidence': steps.get('operator', {}).get('finite_source_evidence'),
            'required_response': steps.get('response', {}).get('requirement'),
            'required_interactions': steps.get('interactions', {}).get('requirement') or steps.get('species', {}).get('requirement'),
            'required_mass': steps.get('mass', {}).get('requirement'),
            'steps': [{'id': s['id'], 'title': s['title'], 'requirement': s['requirement'], 'status': s['physical_confirmation_status']}
                      for s in t['confirmation_steps']],
            'z6_classes_of_components': classes,                      # hypothesis of the simulation side
            'measured_mass_GeV': MEASURED_GEV.get(t['target']), 'measured_mass_note': MEASURED_NOTE.get(t['target'])})
    paths = {r['source_id']: r for r in res['protected_particle_paths']['rows']}
    protected = []
    for fam in res['labelled_catalogue']['protected_families']:
        rows = []
        for r in fam['rows']:
            p = paths.get(r['source_id'], {})
            sr = r['source_record']
            rows.append({'source_id': r['source_id'], 'label': r['label'], 'description': r['description'], 'kind': sr.get('kind'),
                         'domain': sr.get('source_domain'), 'protected_object': sr.get('protected_object'), 'topology': sr.get('topology'),
                         'evidence': r.get('evidence'), 'classification': p.get('classification'), 'result': p.get('result'),
                         'first_required': p.get('first_required'), 'remaining': p.get('remaining'), 'shortcut': p.get('shortcut'),
                         'physical_particle_id': r.get('physical_particle_id'), 'physical_rest_mass': p.get('physical_rest_mass')})
        protected.append({'id': fam['id'], 'label': fam['label'], 'rows': rows})
    excitations = []
    for fam in res['labelled_catalogue']['excitation_families']:
        rows = []
        for r in fam['rows']:
            sr = r['source_record']
            ex = sr.get('exact', {})
            rows.append({'source_id': r['source_id'], 'label': r['label'], 'description': r['description'], 'kind': sr.get('kind'),
                         'exact': {k: ex[k] for k in ('SU3', 'SU2_dimension', 'Y', 'Q6', 'T3', 'comparison_Q') if k in ex},
                         'z6_class': z6(ex['Q6']) if 'Q6' in ex else None, 'domain': sr.get('source_domain'),
                         'protected_defect_id': sr.get('protected_defect_id'), 'physical_particle_id': r.get('physical_particle_id')})
        excitations.append({'id': fam['id'], 'label': fam['label'], 'rows': rows})
    classes = []
    for k in range(6):
        ms = [m for m in multiplets if m['z6_class'] == k]
        conj = [m for m in multiplets if m['z6_class_conjugate'] == k]
        fam = next((f['id'] for f in FAMILIES if k in f['classes']), None)
        classes.append({
            'z6_class': k, 'conjugate_class': (6 - k) % 6, 'family': fam, 'short': SHORT.get(k, ''),
            'multiplets': [m['label'] for m in ms], 'conjugate_multiplets': [m['label'] for m in conj],
            'components': [c for m in ms for c in m['component_ids']],
            'targets': sorted({t['target'] for t in targets if k in t['z6_classes_of_components']}),
            'visible_to': [m['label'] for m in multiplets if (k * int(m['Q6'])) % 6],
            'invisible_to': [m['label'] for m in multiplets if not (k * int(m['Q6'])) % 6],
            'reading': ('flat face: carries no flux' if k == 0 else
                        'A flux of the centre of the gauge group. Patterns that see it: ' + (', '.join(m['label'] for m in multiplets if (k * int(m['Q6'])) % 6) or 'none')
                        + '. Patterns that do not: ' + (', '.join(m['label'] for m in multiplets if not (k * int(m['Q6'])) % 6) or 'none'))})
    return {
        'schema': 'oph.viewer.particle_table.v1',
        'built': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'status': 'Candidate labels only. The theory repository identifies no physical particle and derives no mass: '
                  'physical_particle_map = [] and mass_predictions_GeV = [] in both source files.',
        'sources': {'repository_commit': _git('rev-parse', 'HEAD', repository=rer), 'repository_dirty': bool(_git('status', '--porcelain', '--', 'model/particle_candidate_confirmation', 'model/standard_model_source_map', repository=rer)),
                    'results': {'path': 'model/particle_candidate_confirmation/results.json', 'sha256': _sha(res_path), 'content_sha256': res.get('content_sha256')},
                    'correspondence': {'path': 'model/standard_model_source_map/candidates/correspondence.json', 'sha256': _sha(cor_path)},
                    'physical_particle_map': res['physical_particle_map'], 'mass_predictions_GeV': res['mass_predictions_GeV'],
                    'measured_masses': 'Particle Data Group 2024 central values, comparison only'},
        'z6_reading': {'origin': 'simulation side, corrected by the theory session on 2026-11-10',
                       'statement': 'a flux of class k gives a component of hypercharge label Q6 the phase exp(2 pi i k Q6 / 6). A class is a flux of the centre of the gauge group, '
                                    'not the particle of charge k: class 3 is the weak centre, classes 2 and 4 the colour centre, classes 1 and 5 generate.',
                       'status': 'exact as a statement about the centre; the register rule of the runs is declared'},
        'groups': GROUPS, 'class_style': {str(k): v for k, v in CLASS_STYLE.items()},
        'chart': [{'target': t, 'group': g, 'row': r, 'column': c, 'symbol': sym, 'antisymbol': anti, 'name': name, 'generation': gen,
                   'properties': props, 'carrier': CARRIER[g]} for t, (g, r, c, sym, anti, name, gen, props) in CHART.items()],
        'mass_status': mass_status(HERE),
        'sectors': sectors(), 'target_sectors': TARGET_SECTORS, 'response_of': RESPONSE_OF, 'internal_modes': INTERNAL_MODES, 'yukawa_links': YUKAWA_LINKS,
        'levels': LEVELS, 'state_names': STATE_NAMES, 'model_objects': MODEL_OBJECTS, 'generations': GENERATIONS, 'defect_map': defect_map(),
        'families': FAMILIES, 'classes': classes, 'multiplets': multiplets, 'targets': targets,
        'protected_families': protected, 'excitation_families': excitations,
        'chronology_scope': res['labelled_catalogue'].get('chronology_scope'),
    }



# ---- the protected defects of the committed run as an observer would name them (best guess) -------------------------------
# Sources: repair-law/reports/CH.md and work/SIM/cf1/runs/H_background_from_runs.json (level 10, end of run, 20,971,520
# carriers), work/SIM/cf1/runs/H2_coherence_48.json (routing in a link-flux gas), candidates/CF1.md (mass options),
# model/coherent_defect_wave (audit: routed flux pair, fifteen-state module). A register class k of Z6 is the pair
# (k mod 3, k mod 2): a colour-centre charge and a weak-centre parity. That pair is what the known multiplets carry.
def _h2():
    f = pathlib.Path(__file__).resolve().parents[2] / 'repair-law' / 'work' / 'SIM' / 'cf1' / 'runs' / 'H2_coherence_48.json'
    try:
        d = json.loads(f.read_text()); return {s['name']: s['mean_free_path_cells'] for s in d['species']}, d['nontrivial_plaquettes_per_carrier']
    except Exception:
        return {}, {}
def defect_map():
    mfp, plaq = _h2()
    kinds = [
        {'id': 'qd', 'classes': [1, 5], 'name': 'generating flux', 'long': 'colour-centre and weak-centre flux together', 'group': 'quark', 'colour': 'yes', 'weak': 'yes',
         'guess': 'left-handed quarks (the quark doublet and its antiparticle)',
         'run': ['0.050 per carrier at 300 sweeps, falling', 'wanders by diffusion, never in a straight line; on a carrier it stays on that carrier', 'every step is to the left of the local load flow for class 1 and to the right for class 5: equal and opposite sideways drift, like opposite charges', 'ends by fusion with its own kind into class 2 or 4, or by meeting its antiparticle: the population halves every 75 sweeps'], 'law': 'needs a declared link register of order six',
         'free_path': mfp.get('quark doublets')},
        {'id': 'qs', 'classes': [2, 4], 'name': 'colour-centre flux', 'long': 'flux of the centre of the colour group', 'group': 'quark', 'colour': 'yes', 'weak': 'no',
         'guess': 'right-handed quarks and their antiparticles',
         'run': ['0.268 pairs per carrier, the same at every size', 'never alone: only in groups without net colour charge', 'made when two of class 1 or two of class 5 fuse; frozen once its faces are not neighbours', 'never moves to another carrier'], 'law': 'needs a declared link register of order six',
         'free_path': mfp.get('quark singlets')},
        {'id': 'ld', 'classes': [3], 'name': 'weak-centre flux', 'long': 'the sign of a link, its own conjugate', 'group': 'lepton', 'colour': 'no', 'weak': 'yes',
         'guess': 'left-handed leptons, and the Higgs doublet',
         'run': ['0.041 pairs per carrier; 9.1 percent of the carriers hold flux of this kind when the register is read modulo 2', 'comes in pairs, made from the mixed triples', 'frozen once made', 'the Higgs doublet has the same class: the register cannot tell the two apart'], 'law': 'the one flux of the working law: the sign of a link',
         'free_path': mfp.get('lepton doublets')},
        {'id': 'ls', 'classes': [0], 'name': 'flat face', 'long': 'no flux', 'group': 'neutrino', 'colour': 'no', 'weak': 'no',
         'guess': 'right-handed electron and neutrino',
         'run': ['carries no register class: a flat face', 'invisible to the defect register', 'in the coherent model: the four patterns that see no centre flux of any class'], 'law': 'no flux',
         'free_path': None},
    ]
    groups = [
        {'set': '{2, 4}', 'carriers': 4991967, 'share': 0.6916, 'name': 'a colour charge and its opposite', 'reading': 'the charge content of a meson', 'group': 'quark'},
        {'set': '{1, 5}', 'carriers': 1041883, 'share': 0.1443, 'name': 'a generating flux and its opposite', 'reading': 'the charge content of a left-handed quark and its antiparticle; still annihilating', 'group': 'quark'},
        {'set': '{3, 3}', 'carriers': 785555, 'share': 0.1088, 'name': 'two weak-centre fluxes', 'reading': 'the charge content of a lepton and an antilepton', 'group': 'lepton'},
        {'set': '{2, 2, 4, 4}', 'carriers': 139090, 'share': 0.0193, 'name': 'two colour pairs', 'reading': 'two neutral pairs on one carrier', 'group': 'quark'},
        {'set': '{2, 2, 2}', 'carriers': 86788, 'share': 0.0120, 'name': 'three antiquark colour charges', 'reading': 'the centre classes of an antibaryon, three antiquarks; as a flux content it is a boson', 'group': 'gluon'},
        {'set': '{4, 4, 4}', 'carriers': 86476, 'share': 0.0120, 'name': 'three quark colour charges', 'reading': 'the centre classes of a baryon, three quarks; as a flux content it is a boson, so it is not itself a baryon', 'group': 'gluon'},
        {'set': '{2, 3, 3, 4}', 'carriers': 55690, 'share': 0.0077, 'name': 'a colour pair and a weak pair', 'reading': 'two neutral groups on one carrier', 'group': 'lepton'},
        {'set': '{1, 2, 3} and {3, 4, 5}', 'carriers': 22351, 'share': 0.0031, 'name': 'mixed triple', 'reading': 'a left-handed quark, a right-handed quark and an object of class 3: the charges that meet where the Higgs gives a quark its mass', 'group': 'weak'},
    ]
    return {
        'status': 'best guess by structure; not a derivation',
        'rule': 'A register class k of Z6 is the pair (k mod 3, k mod 2): a colour-centre charge and a weak-centre parity. The known multiplets carry exactly these pairs: quark doublets (1 or 2, odd), quark singlets (1 or 2, even), lepton doublets (0, odd), lepton singlets (0, even).',
        'constraint': 'The classes on the twenty faces of a carrier sum to zero. So a colour charge never sits alone on a carrier: only pairs and triples without net charge occur. In nature this is the rule that quarks come in mesons and baryons.',
        'kinds': kinds, 'groups': groups, 'carriers': 20971520, 'empty_share': 0.6558,
        'balance': 'Baryon-like and antibaryon-like groups: 86,476 against 86,788 at level 10. Over levels 5 to 10 the two differ by 0.13 ± 0.21 percent. They are made through mirror channels at equal rates and sit at random, without clustering. The run has no excess of matter over antimatter.',
        'fusion': 'All 11.5 million fusions and splits in the stored runs obey addition modulo 6, without one exception, and the tables are the same for a class and its conjugate. Content never moves from one carrier to another.',
        'hadron_gas': 'For orientation: in a hot gas of hadrons at the temperature where quarks bind, protons and neutrons are a few percent of the pions. In the run the baryon-like groups are 1.7 percent of the meson-like groups.',
        'reactions': [['pair births', 1496469, 'a defect and its conjugate appear together'], ['annihilations', 613096, 'a defect meets its conjugate'], ['fusions', 2821014, 'two classes add to one'],
                      ['splits', 1551599, 'one class breaks into two'], ['exchanges', 1881003, 'two defects trade class'], ['moves', 30732639, 'a defect steps to a neighbouring face']],
        'h2': {'plaquettes_per_carrier': plaq, 'note': 'Routing of each kind of quantum through a static gas of link flux with the densities of the frozen classes, 96 x 48 x 48 carriers. Mean free path of the ballistic packet, relative to a free quantum.',
               'composite': 'A point object made of three quark quanta has no colour-centre charge. It reads the links exactly as a lepton doublet does and gets as far. A quark and an antiquark held apart are lost at twice the single rate in this gas, and at 0.6 to 0.8 of it in a gas of long flux lines: the arrangement of the flux matters and is not derived.'},
        'dark': {'name': 'the travelling flux pair', 'guess': 'dark matter', 'facts': ['a complete flux pair moves from carrier to carrier (audit, public program)', 'it carries no gauge charge: its five positions are gauge singlets', 'rest energy 2 source units under a declared mass rule', 'the cosmology papers name unrecovered collar charge as the dark part']},
        'tower': {'name': 'internal sectors of a carrier', 'guess': 'heavy partners at the scale of the lattice, not known particles', 'angles': [['uniform load, blank record', 0.0], ['slow triplet', 0.304708], ['quintet, recorded', 0.367208], ['quintet, blank', 0.451027], ['fast triplet', 0.496198], ['uniform load, recorded', 0.585686]]},
        'mass': ['The native idle interaction vanishes on the fifteen-state internal module. That alone supplies neither propagating massless particles nor an interacting mass spectrum.',
                 'A Higgs-type quantum memory permits invariant Yukawa couplings. This is one declared mass-generating mechanism; its interacting reference and physical parameters require calculation.',
                 'The frozen flux of the run is such a state. Read as a background it gives quarks about three times the rest phase of leptons and leaves the right-handed electron at zero. It was set by how the run was prepared: loads seeded from 0 to 5 leave 0.268 pairs per carrier, loads from 0 to 3 leave 0.008, and a start inside the settled band leaves none.'],
    }

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rer', type=Path, default=RER)
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args(argv)
    table = build(a.rer)
    a.out.write_text(json.dumps(table, indent=1, ensure_ascii=False) + '\n')
    print(f'{a.out}: {len(table["targets"])} targets, {sum(len(f["rows"]) for f in table["protected_families"])} protected objects, '
          f'{sum(len(f["rows"]) for f in table["excitation_families"])} excitations, repository {table["sources"]["repository_commit"][:9]}')
    for c in table['classes']:
        print(f'  class {c["z6_class"]}: family {c["family"]}, multiplets {c["multiplets"]}, conjugates {c["conjugate_multiplets"]}, targets {c["targets"]}')


if __name__ == '__main__':
    main()
