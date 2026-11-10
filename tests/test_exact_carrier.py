from __future__ import annotations

from fractions import Fraction

import numpy as np

from oph_exact import carrier


def test_incidence_counts_and_pairing() -> None:
    assert len(carrier.seams()) == 30
    assert len(carrier.oriented_faces()) == 20
    assert len(carrier.incidence_automorphisms()) == 120
    assert len(carrier.rotations()) == 60
    anti = carrier.antipode()
    assert all(anti[anti[p]] == p and anti[p] != p for p in range(12))
    dist = carrier.graph_distance()
    assert all(dist[p, anti[p]] == 3 for p in range(12))


def test_seam_mean_is_symmetric_conservative_retraction() -> None:
    for seam in carrier.seams():
        e = carrier.seam_mean_matrix(seam)
        assert np.allclose(e @ e, e)
        assert np.allclose(e.sum(axis=0), 1.0)
        assert np.allclose(e.sum(axis=1), 1.0)
        i, j = seam
        x = np.arange(12, dtype=float)
        y = carrier.apply_seam_mean(x, seam)
        assert y[i] == y[j] == 0.5 * (x[i] + x[j])
        assert abs(y.sum() - x.sum()) < 1e-12


def test_uniform_schedule_gives_repair_mean() -> None:
    average = sum(carrier.seam_mean_matrix(seam) for seam in carrier.seams()) / 30.0
    assert np.allclose(average, carrier.repair_mean())
    exact = carrier.repair_mean_exact()
    assert all(sum(row) == Fraction(1) for row in exact)


def test_slow_band_and_gram() -> None:
    checks = carrier.check_carrier()
    assert all(checks.values()), checks
    gram = carrier.intrinsic_gram()
    assert np.linalg.matrix_rank(gram, tol=1e-9) == 3
    eigenvalues = np.sort(np.linalg.eigvalsh(gram))[::-1]
    assert np.allclose(eigenvalues[:3], 4.0) and np.allclose(eigenvalues[3:], 0.0, atol=1e-12)


def test_normalized_kernel_has_rank_eleven_at_finite_step() -> None:
    for n in (1, 5, 30):
        kernel = carrier.normalized_response_kernel(n)
        assert np.linalg.matrix_rank(kernel, tol=1e-9) == 11
        power_form = carrier.centered_response_kernel(n)
        power_form = 12.0 * power_form / np.trace(power_form)
        assert np.allclose(kernel, power_form, atol=1e-12)
    limit = carrier.normalized_response_kernel(300)
    assert np.allclose(limit, carrier.intrinsic_gram(), atol=1e-12)


def test_integer_law_preserves_total_and_places_tie() -> None:
    assert carrier.integer_nearest_agreement(4, 2, ceiling_to_first=True) == (3, 3)
    assert carrier.integer_nearest_agreement(4, 1, ceiling_to_first=True) == (3, 2)
    assert carrier.integer_nearest_agreement(4, 1, ceiling_to_first=False) == (2, 3)
    assert carrier.integer_nearest_agreement(-3, 0, ceiling_to_first=True) == (-1, -2)


def test_antipode_commutes_with_every_rotation() -> None:
    anti = carrier.antipode()
    for perm in carrier.rotations():
        assert all(perm[anti[p]] == anti[perm[p]] for p in range(12))
