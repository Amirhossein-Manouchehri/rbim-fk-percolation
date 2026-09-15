"""Tests for FK bond activation, cluster labelling and percolation observables."""

import numpy as np
import pytest

from rbim.clusters import (
    activate_bonds,
    bond_activation_probability,
    cluster_statistics,
    label_clusters,
    spans_along,
    wraps_along,
)
from rbim.lattice import build_lattice, couplings_from_disorder
from rbim.disorder import generate_disorder


def _ferromagnetic_couplings(L):
    J_h = np.ones((L, L), dtype=np.int8)
    J_v = np.ones((L, L), dtype=np.int8)
    return couplings_from_disorder(J_h, J_v)


def test_activation_probability_limits():
    assert bond_activation_probability(0.0) == pytest.approx(0.0)
    assert bond_activation_probability(100.0) == pytest.approx(1.0)
    assert bond_activation_probability(0.5) == pytest.approx(1.0 - np.exp(-1.0))


def test_infinite_temperature_activates_nothing():
    L = 8
    lattice = build_lattice(L)
    J = _ferromagnetic_couplings(L)
    rng = np.random.default_rng(0)
    spins = np.ones(lattice.n_sites, dtype=np.int64)

    active = activate_bonds(spins, lattice, J, beta=0.0, rng=rng)
    assert not np.any(active)

    n_clusters, labels = label_clusters(active, lattice)
    assert n_clusters == lattice.n_sites
    assert not wraps_along(active, lattice, "x")
    assert not spans_along(active, lattice, "y")


def test_zero_temperature_ferromagnet_is_one_wrapping_cluster():
    L = 8
    lattice = build_lattice(L)
    J = _ferromagnetic_couplings(L)
    rng = np.random.default_rng(0)
    spins = np.ones(lattice.n_sites, dtype=np.int64)

    active = activate_bonds(spins, lattice, J, beta=50.0, rng=rng)
    assert np.all(active)

    n_clusters, labels = label_clusters(active, lattice)
    assert n_clusters == 1
    assert wraps_along(active, lattice, "x")
    assert wraps_along(active, lattice, "y")


def test_only_satisfied_bonds_are_activated():
    """The invariant that guarantees FK clusters contain no frustrated loop."""
    L = 12
    lattice = build_lattice(L)
    rng = np.random.default_rng(7)
    J_h, J_v = generate_disorder(L, 0.3, rng)
    J = couplings_from_disorder(J_h, J_v)
    spins = (rng.integers(0, 2, size=lattice.n_sites) * 2 - 1).astype(np.int64)

    active = activate_bonds(spins, lattice, J, beta=0.6, rng=rng)
    ends = lattice.bonds[active]
    assert np.all(J[active] * spins[ends[:, 0]] * spins[ends[:, 1]] > 0)


def test_cluster_statistics_are_consistent():
    L = 10
    lattice = build_lattice(L)
    rng = np.random.default_rng(3)
    J_h, J_v = generate_disorder(L, 0.15, rng)
    J = couplings_from_disorder(J_h, J_v)
    spins = (rng.integers(0, 2, size=lattice.n_sites) * 2 - 1).astype(np.int64)

    active = activate_bonds(spins, lattice, J, beta=0.5, rng=rng)
    n_clusters, labels = label_clusters(active, lattice)
    sizes, magnetizations = cluster_statistics(labels, spins, n_clusters)

    assert sizes.sum() == lattice.n_sites
    assert magnetizations.sum() == pytest.approx(spins.sum())
    # A cluster can never carry more magnetization than it has spins.
    assert np.all(np.abs(magnetizations) <= sizes + 1e-9)


def test_spanning_and_wrapping_are_distinct():
    L = 6
    lattice = build_lattice(L)
    active = np.zeros(lattice.n_bonds, dtype=bool)

    # Activate the vertical bonds of the column x = 0 except the one that closes
    # the boundary: the column becomes an open path, which spans but cannot wrap.
    for y in range(L - 1):
        n = 0 * L + y
        active[2 * n + 1] = True

    assert spans_along(active, lattice, "y")
    assert not wraps_along(active, lattice, "y")

    # Closing the ring makes it wrap as well.
    n_last = 0 * L + (L - 1)
    active[2 * n_last + 1] = True
    assert wraps_along(active, lattice, "y")
