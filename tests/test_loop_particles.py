import numpy as np

from oph_fpe.microphysics.loop_particles import (
    detect_loop_particles,
    loop_mode_energy,
    shape_loop_particle_receipt,
)


def test_loop_mode_energy_detects_mode():
    amp = np.exp(2j * np.pi * np.arange(5) / 5.0)
    energy = loop_mode_energy(amp, [0, 1, 2, 3, 4], mode=1)

    assert energy > 0.99


def test_loop_particle_receipt_requires_persistence():
    tracks = [{"face_id": 0, "mode": 1, "lifetime": 8, "class_preserved": True}]
    report = shape_loop_particle_receipt(tracks, min_lifetime=8)

    assert report["passed"] is True
    assert report["standard_model_particle_claim"] is False


def test_detect_loop_particles_reports_best_mode():
    amp = np.exp(2j * np.pi * np.arange(5) / 5.0)
    particles = detect_loop_particles(amp, [[0, 1, 2, 3, 4]], energy_threshold=0.5)

    assert len(particles) == 1
    assert particles[0]["mode"] == 1
