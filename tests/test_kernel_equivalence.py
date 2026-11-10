"""The compiled sweep must reproduce the original array-based sweep exactly."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rbim.disorder import generate_disorder
from rbim.lattice import build_lattice, couplings_from_disorder
from rbim.swendsen_wang import initial_spins, sw_sweep, sw_sweep_reference

NEW_FIELDS = {"cluster_magnetization_sq_sum", "n_active_bonds"}

CASES = [
    (3, 0.4, 1.5),
    (5, 0.3, 0.7),
    (8, 0.2, 3.0),
    (12, 0.15, 1.86),
    (16, 0.08, 1.9),
    (16, 0.5, 1.0),
    (24, 0.0, 2.269),
    (24, 0.05, 1.2),
]


@pytest.mark.parametrize("L,p,temperature", CASES)
@pytest.mark.parametrize("start", ["hot", "cold"])
def test_sweep_matches_reference(L, p, temperature, start):
    lattice = build_lattice(L)
    J = couplings_from_disorder(*generate_disorder(L, p, np.random.default_rng(L)))
    rng_new = np.random.default_rng(1234)
    rng_ref = np.random.default_rng(1234)
    spins_new = initial_spins(L * L, rng_new, start)
    spins_ref = initial_spins(L * L, rng_ref, start)
    beta = 1.0 / temperature

    wrapping = 0
    for step in range(200):
        measure = step % 3 != 0
        new = sw_sweep(spins_new, lattice, J, beta, rng_new, measure=measure)
        ref = sw_sweep_reference(spins_ref, lattice, J, beta, rng_ref, measure=measure)
        assert np.array_equal(spins_new, spins_ref)
        if measure:
            new_fields = dataclasses.asdict(new)
            for name, value in dataclasses.asdict(ref).items():
                if name not in NEW_FIELDS:
                    assert new_fields[name] == value, name
            wrapping += bool(new.wraps_x or new.wraps_y)
    # Both generators must have consumed the same number of random values.
    assert rng_new.random() == rng_ref.random()


def test_new_fields_are_consistent():
    L, p, temperature = 16, 0.1, 1.9
    lattice = build_lattice(L)
    J = couplings_from_disorder(*generate_disorder(L, p, np.random.default_rng(0)))
    rng = np.random.default_rng(5)
    spins = initial_spins(L * L, rng, "hot")
    for _ in range(100):
        result = sw_sweep(spins, lattice, J, 1.0 / temperature, rng)
        assert 0 <= result.n_active_bonds <= lattice.n_bonds
        # The largest cluster is one term of the sum over clusters.
        assert result.largest_abs_magnetization**2 <= result.cluster_magnetization_sq_sum
        # A sum of squares of integers with the parity of the magnetization.
        assert result.cluster_magnetization_sq_sum >= abs(result.magnetization)
        assert result.cluster_magnetization_sq_sum <= (L * L) ** 2
