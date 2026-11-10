"""Low spectrum of the port Laplacian for the declared gluing and for the frame-free control gluing.

    python -m oph_defects.gluing_control --levels 3,4,5

Prints 4^l lambda for the lowest modes. On a port graph with the rotation symmetry of the icosahedral mesh
the l = 1 modes form an exact triplet, l = 2 a quintet, and l = 3 splits into 3 + 4.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from . import fedgeom as FG


def laplacian(geo: FG.Geometry) -> sp.csr_matrix:
    a, b = geo.all_seam_ends()
    S, P = geo.seams, geo.ports
    B = sp.csr_matrix((np.concatenate([np.ones(S), -np.ones(S)]), (np.concatenate([np.arange(S), np.arange(S)]), np.concatenate([a, b]))), shape=(S, P))
    return (B.T @ B).tocsc()


def spectrum(level: int, gluing: str, k: int = 25) -> np.ndarray:
    geo = FG.build(level, gluing=gluing)
    L = laplacian(geo)
    vals = spl.eigsh(L, k=k, sigma=-1e-4 / 4 ** level, which='LM', return_eigenvectors=False)
    return np.sort(vals) * 4 ** level


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--levels', default='3,4,5')
    p.add_argument('--json', default='')
    a = p.parse_args(argv)
    out = {}
    for level in (int(v) for v in a.levels.split(',')):
        for gluing in ('port_pair', 'face_axis'):
            v = spectrum(level, gluing)
            out[f'{gluing}_L{level}'] = v.tolist()
            groups = [v[1:4], v[4:9], v[9:16], v[16:25]]
            print(f'level {level} {gluing:9}: zero mode {v[0]:.1e}')
            for name, g in zip(('l=1 (3)', 'l=2 (5)', 'l=3 (7)', 'l=4 (9)'), groups):
                print(f'    {name}: ' + ' '.join(f'{x:.6f}' for x in g) + f'   spread {100 * (g.max() - g.min()) / g.mean():.3f}%')
    if a.json:
        open(a.json, 'w').write(json.dumps(out))


if __name__ == '__main__':
    main()
