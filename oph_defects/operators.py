"""The Hilbert space of one carrier and its operator readouts.

    python -m oph_defects.operators

The space. Load configurations x of the twelve ports with total load two: 66 with two singly loaded ports (the
binary shell) and 12 with one doubly loaded port, dimension 78. This is the space H_78 on which the
executable-observer paper defines the native repair instruments.

The native instruments on seam e = (a, b), a the first end. With tot = x_a + x_b, lo = floor(tot / 2),
hi = tot - lo, the coin c = 0 sends (x_a, x_b) to (lo, hi) and c = 1 to (hi, lo); the current is
j = x_a - x_a' and the contrast d = x_a - x_b. The current instrument C has Kraus operators
L_(c,j) = 2^(-1/2) sum over x with current j of |y(x, c)><x|, the contrast instrument E has
M_(c,d) = 2^(-1/2) sum over x with contrast d of |y(x, c)><x|.

Readouts (all exact matrices on the 78 states):
    N_p            load of port p (diagonal); N = sum N_p is conserved by every Kraus operator
    Phi            (1/2) sum N_p^2, the mismatch potential; no Kraus branch raises it
    R_e            floor(d_e^2 / 4), the release of an attempt on e (diagonal)
    E_(c,j)        the effect of a record, L^dagger L: diagonal in the load basis for every record and for every
                   composition of records, so record statistics depend on the diagonal of the state alone
    <j>_e          mean recorded current of an attempt, (N_a - N_b) / 2: a vertex gradient, zero around every face
    T_ab           the hop |..x_a+1, x_b-1..><x|, the operator a coherent probe would need; it is not diagonal and no
                   composition of native records measures it
    g              the factor by which one attempt multiplies the coherence between "empty seam" and "one load
                   waiting on the seam": 1/2 for C, 0 for E
"""
from __future__ import annotations

import itertools

import numpy as np

from . import fedgeom as FG

SEAMS = [tuple(int(v) for v in s) for s in FG.TEMPLATE_SEAMS]


def basis():
    states = []
    for a, b in itertools.combinations(range(12), 2):
        x = [0] * 12; x[a] = x[b] = 1; states.append(tuple(x))
    for a in range(12):
        x = [0] * 12; x[a] = 2; states.append(tuple(x))
    return states, {x: i for i, x in enumerate(states)}


STATES, INDEX = basis()
DIM = len(STATES)


def step(x, e, c):
    a, b = e
    tot = x[a] + x[b]; lo = tot // 2; hi = tot - lo
    na, nb = (lo, hi) if c == 0 else (hi, lo)
    y = list(x); y[a] = na; y[b] = nb
    return tuple(y), x[a] - na, x[a] - x[b]


def kraus(e, law='C'):
    """Kraus operators of one attempt on seam e: dict record -> matrix."""
    ops = {}
    for i, x in enumerate(STATES):
        for c in (0, 1):
            y, j, d = step(x, e, c)
            key = (c, j) if law == 'C' else (c, d)
            ops.setdefault(key, np.zeros((DIM, DIM)))[INDEX[y], i] += 2 ** -0.5
    return ops


def channel(ops, rho):
    return sum(K @ rho @ K.T for K in ops.values())


def number(p):
    return np.diag([float(x[p]) for x in STATES])


def potential():
    return np.diag([0.5 * sum(v * v for v in x) for x in STATES])


def release(e):
    return np.diag([float((x[e[0]] - x[e[1]]) ** 2 // 4) for x in STATES])


def hop(a, b):
    """Moves one unit from b to a where possible."""
    T = np.zeros((DIM, DIM))
    for i, x in enumerate(STATES):
        if x[b] >= 1:
            y = list(x); y[b] -= 1; y[a] += 1
            if tuple(y) in INDEX:
                T[INDEX[tuple(y)], i] = 1.0
    return T


def is_diagonal(A, tol=1e-12):
    return bool(np.abs(A - np.diag(np.diag(A))).max() < tol)


def report() -> dict:
    out = {'dimension': DIM, 'binary_shell': 66}
    shell = np.array([max(x) == 1 for x in STATES])
    N = sum(number(p) for p in range(12)); Phi = potential()
    worst = {'trace_preserving': 0.0, 'number': 0.0, 'effects_off_diagonal': 0.0, 'potential_rise': 0.0, 'shell_leak': 0.0}
    labels_all, labels_shell = set(), set()
    for e in SEAMS:
        for law in ('C', 'E'):
            ops = kraus(e, law)
            worst['trace_preserving'] = max(worst['trace_preserving'], float(np.abs(sum(K.T @ K for K in ops.values()) - np.eye(DIM)).max()))
            for r, K in ops.items():
                worst['number'] = max(worst['number'], float(np.abs(N @ K - K @ N).max()))
                Ef = K.T @ K
                worst['effects_off_diagonal'] = max(worst['effects_off_diagonal'], float(np.abs(Ef - np.diag(np.diag(Ef))).max()))
                rows, cols = np.nonzero(K)
                worst['potential_rise'] = max(worst['potential_rise'], float((np.diag(Phi)[rows] - np.diag(Phi)[cols]).max()))
                worst['shell_leak'] = max(worst['shell_leak'], float(np.abs(K[np.ix_(~shell, shell)]).max()))
                if law == 'C':
                    labels_all.add(r)
                    if np.abs(K[:, shell]).max() > 0:
                        labels_shell.add(r)
    out['checks'] = worst
    out['current_labels_on_H78'] = sorted(labels_all); out['current_labels_on_binary_shell'] = sorted(labels_shell)
    # channel identities on one seam, as superoperators on a basis of matrix units restricted to random states
    rng = np.random.default_rng(0)
    e = SEAMS[0]
    C, E = kraus(e, 'C'), kraus(e, 'E')
    X = rng.normal(size=(DIM, DIM)) + 1j * rng.normal(size=(DIM, DIM)); rho = X @ X.conj().T; rho /= np.trace(rho)
    cC = lambda r: channel(C, r); cE = lambda r: channel(E, r)
    ident = {'E_idempotent': float(np.abs(cE(cE(rho)) - cE(rho)).max()), 'EC_equals_E': float(np.abs(cE(cC(rho)) - cE(rho)).max()),
             'CE_equals_E': float(np.abs(cC(cE(rho)) - cE(rho)).max())}
    r = rho
    for n in range(1, 6):
        r = cC(r)
        ident[f'C_power_{n}'] = float(np.abs(r - (cE(rho) + 2.0 ** (1 - n) * (cC(rho) - cE(rho)))).max())
    out['channel_identities_max_error'] = ident
    out['same_classical_transition'] = float(max(np.abs(np.diag(cC(np.diag(v))) - np.diag(cE(np.diag(v)))).max()
                                                 for v in np.eye(DIM)))
    # products of Kraus operators keep diagonal effects
    worst_prod = 0.0
    for _ in range(300):
        K = np.eye(DIM)
        for _ in range(int(rng.integers(2, 7))):
            ops = kraus(SEAMS[int(rng.integers(30))], 'C'); K = list(ops.values())[int(rng.integers(len(ops)))] @ K
        Ef = K.T @ K
        worst_prod = max(worst_prod, float(np.abs(Ef - np.diag(np.diag(Ef))).max()))
    out['effects_of_products_off_diagonal'] = worst_prod
    # mean current is a vertex gradient: <j>_e = (N_a - N_b) / 2, face sums vanish
    grad = 0.0
    for e in SEAMS:
        ops = kraus(e, 'C')
        mean_j = sum(r[1] * (K.T @ K) for r, K in ops.items())
        grad = max(grad, float(np.abs(mean_j - (number(e[0]) - number(e[1])) / 2).max()))
    out['mean_current_minus_gradient'] = grad
    # the hop is not diagonal, and the coherence factor g between "empty seam" and "one load waiting on it"
    a, b = SEAMS[0]
    T = hop(a, b)
    out['hop_is_diagonal'] = is_diagonal(T)
    spect = [p for p in range(12) if p not in (a, b)]
    x0 = [0] * 12; x0[spect[0]] = 1; x0[spect[1]] = 1                      # seam key (0, 0)
    x1 = [0] * 12; x1[b] = 1; x1[spect[0]] = 1                             # seam key (0, 1), same spectator
    unit = np.zeros((DIM, DIM)); unit[INDEX[tuple(x1)], INDEX[tuple(x0)]] = 1.0
    out['g'] = {'C': float(cC(unit)[INDEX[tuple(x1)], INDEX[tuple(x0)]]), 'E': float(cE(unit)[INDEX[tuple(x1)], INDEX[tuple(x0)]])}
    # release operator: mean release of one attempt in the uniform state on the shell, and the potential it lowers
    uni = np.diag(shell / shell.sum())
    out['uniform_shell'] = {'mean_release_per_attempt': float(np.mean([np.trace(release(e) @ uni) for e in SEAMS])),
                            'mean_potential': float(np.trace(Phi @ uni)), 'invariant_under_C': float(np.abs(np.mean([cC(uni) for _ in [0]], axis=0) - uni).max())}
    full = np.diag(np.ones(DIM) / DIM)
    out['uniform_H78'] = {'mean_release_per_attempt': float(np.mean([np.trace(release(e) @ full) for e in SEAMS])),
                          'mean_potential': float(np.trace(Phi @ full))}
    return out


def main():
    r = report()
    c = r['checks']
    print(f"H_78: dimension {r['dimension']}, binary shell {r['binary_shell']}")
    print(f"every instrument is trace preserving (max error {c['trace_preserving']:.1e}); every Kraus operator commutes with the total load ({c['number']:.1e}), "
          f"never raises the potential ({c['potential_rise']:.1f}) and maps the binary shell into itself ({c['shell_leak']:.1e})")
    print(f"every effect is diagonal in the loads (off-diagonal part {c['effects_off_diagonal']:.1e}); so are the effects of 300 random products ({r['effects_of_products_off_diagonal']:.1e})")
    print(f"current labels on H_78: {r['current_labels_on_H78']}; on the binary shell: {r['current_labels_on_binary_shell']}")
    print('channel identities of the paper, max error: ' + ', '.join(f'{k} {v:.1e}' for k, v in r['channel_identities_max_error'].items()))
    print(f"C and E give the same classical transition (max difference {r['same_classical_transition']:.1e}); mean current minus (N_a - N_b)/2: {r['mean_current_minus_gradient']:.1e}")
    print(f"the hop operator is diagonal: {r['hop_is_diagonal']}; coherence factor g of one attempt: C {r['g']['C']:.3f}, E {r['g']['E']:.3f}")
    print(f"uniform state on the shell: mean potential {r['uniform_shell']['mean_potential']:.4f}, mean release per attempt {r['uniform_shell']['mean_release_per_attempt']:.4f}; "
          f"uniform on H_78: potential {r['uniform_H78']['mean_potential']:.4f}, release per attempt {r['uniform_H78']['mean_release_per_attempt']:.5f}")
    return r


if __name__ == '__main__':
    main()
