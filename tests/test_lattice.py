"""Geometry and bookkeeping tests for the square lattice."""

import numpy as np
import pytest

from rbim.disorder import frustrated_plaquettes, frustration_density, generate_disorder
from rbim.lattice import build_lattice, couplings_from_disorder, site_index


@pytest.mark.parametrize("L", [3, 4, 8])
def test_bond_count_and_degree(L):
    lattice = build_lattice(L)
    assert lattice.n_sites == L * L
    assert lattice.bonds.shape == (2 * L * L, 2)

    # On a periodic square lattice every site has exactly four neighbours, so it
    # must appear four times among the bond endpoints.
    counts = np.bincount(lattice.bonds.reshape(-1), minlength=lattice.n_sites)
    assert np.all(counts == 4)


@pytest.mark.parametrize("L", [3, 5, 8])
def test_wrap_masks(L):
    lattice = build_lattice(L)
    # Exactly one row of horizontal bonds and one column of vertical bonds close
    # the boundary.
    assert lattice.wraps_x.sum() == L
    assert lattice.wraps_y.sum() == L
    # A bond cannot wrap in both directions.
    assert not np.any(lattice.wraps_x & lattice.wraps_y)

    # Wrapping horizontal bonds start at x = L - 1.
    starts = lattice.bonds[lattice.wraps_x, 0]
    assert np.all(starts // L == L - 1)

    # Wrapping vertical bonds start at y = L - 1.
    starts = lattice.bonds[lattice.wraps_y, 0]
    assert np.all(starts % L == L - 1)


def test_bond_ordering_matches_coupling_layout():
    L = 4
    lattice = build_lattice(L)
    J_h = np.arange(L * L, dtype=np.int8).reshape(L, L)
    J_v = -np.arange(L * L, dtype=np.int8).reshape(L, L)
    J = couplings_from_disorder(J_h, J_v)

    for x in range(L):
        for y in range(L):
            n = site_index(x, y, L)
            assert J[2 * n] == J_h[x, y]
            assert J[2 * n + 1] == J_v[x, y]
            assert lattice.bonds[2 * n, 0] == n
            assert lattice.bonds[2 * n + 1, 0] == n


def test_small_lattice_rejected():
    with pytest.raises(ValueError):
        build_lattice(2)


def test_pure_ferromagnet_has_no_frustration():
    L = 8
    J_h = np.ones((L, L), dtype=np.int8)
    J_v = np.ones((L, L), dtype=np.int8)
    assert frustration_density(J_h, J_v) == 0.0


def test_single_negative_bond_frustrates_two_plaquettes():
    L = 6
    J_h = np.ones((L, L), dtype=np.int8)
    J_v = np.ones((L, L), dtype=np.int8)
    J_h[2, 3] = -1  # bottom edge of plaquette (2, 3), top edge of plaquette (2, 2)
    frustrated = frustrated_plaquettes(J_h, J_v)
    assert frustrated.sum() == 2
    assert frustrated[2, 3]
    assert frustrated[2, 2]


def test_frustration_density_matches_expectation():
    L = 128
    p = 0.3
    rng = np.random.default_rng(12345)
    J_h, J_v = generate_disorder(L, p, rng)
    expected = 4 * p * (1 - p) * ((1 - p) ** 2 + p**2)
    assert abs(frustration_density(J_h, J_v) - expected) < 0.01
