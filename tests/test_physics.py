"""Physics validation against exactly known results for the pure 2D Ising model.

At ``p = 0`` the model reduces to the ferromagnetic square-lattice Ising model,
for which Onsager's solution provides exact benchmarks.  These tests are the
main evidence that the simulation is correct rather than merely self-consistent,
so they are worth keeping even though they are slower than unit tests.

Run only the fast tests with ``pytest -m "not slow"``.
"""

import numpy as np
import pytest

from rbim.lattice import couplings_from_disorder
from rbim.simulation import run_single_realization

TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))  # 2.269185...


def _ferromagnetic_couplings(L):
    ones = np.ones((L, L), dtype=np.int8)
    return couplings_from_disorder(ones, ones)


def onsager_magnetization(temperature: float) -> float:
    """Exact spontaneous magnetization of the infinite square-lattice Ising model."""
    if temperature >= TC_EXACT:
        return 0.0
    return float((1.0 - np.sinh(2.0 / temperature) ** -4) ** 0.125)


def test_exact_critical_temperature_value():
    assert TC_EXACT == pytest.approx(2.269185, abs=1e-5)


def test_repeated_runs_with_the_same_seed_agree():
    L = 8
    J = _ferromagnetic_couplings(L)
    kwargs = dict(n_sweeps=200, n_therm=50, n_bins=4, seed=1234)
    first = run_single_realization(L, J, 2.5, **kwargs)
    second = run_single_realization(L, J, 2.5, **kwargs)
    for name in first:
        np.testing.assert_allclose(first[name], second[name])


def test_different_seeds_disagree():
    L = 8
    J = _ferromagnetic_couplings(L)
    kwargs = dict(n_sweeps=200, n_therm=50, n_bins=4)
    first = run_single_realization(L, J, 2.5, seed=1, **kwargs)
    second = run_single_realization(L, J, 2.5, seed=2, **kwargs)
    assert not np.allclose(first["energy"], second["energy"])


@pytest.mark.slow
def test_energy_stays_within_bounds():
    L = 12
    J = _ferromagnetic_couplings(L)
    for temperature in (1.0, 2.269185, 5.0):
        run = run_single_realization(L, J, temperature, n_sweeps=400, n_therm=200, n_bins=4, seed=5)
        energy = float(np.mean(run["energy"]))
        assert -2.0 <= energy <= 0.0


@pytest.mark.slow
def test_spontaneous_magnetization_matches_onsager():
    """Below Tc the magnetization should track the exact Onsager curve."""
    L = 16
    J = _ferromagnetic_couplings(L)
    temperature = 2.0

    run = run_single_realization(
        L, J, temperature, n_sweeps=4000, n_therm=1000, n_bins=10, seed=2024
    )
    measured = float(np.mean(run["abs_magnetization"]))
    exact = onsager_magnetization(temperature)

    # A finite lattice sits slightly above the thermodynamic curve; 4% covers
    # the finite-size offset at L = 16 together with the statistical error.
    assert measured == pytest.approx(exact, abs=0.04)


@pytest.mark.slow
def test_disordered_phase_has_small_magnetization():
    L = 16
    J = _ferromagnetic_couplings(L)
    run = run_single_realization(L, J, 4.0, n_sweeps=2000, n_therm=500, n_bins=10, seed=11)
    assert float(np.mean(run["abs_magnetization"])) < 0.15


@pytest.mark.slow
def test_percolation_strength_follows_magnetization_in_the_pure_model():
    """For p = 0 the FK percolation strength equals the magnetization.

    This is the Coniglio-Klein identity, and its breakdown at p > 0 is the
    phenomenon the project is built around, so the p = 0 limit is worth
    pinning down.
    """
    L = 16
    J = _ferromagnetic_couplings(L)
    run = run_single_realization(L, J, 2.0, n_sweeps=4000, n_therm=1000, n_bins=10, seed=99)

    p_infinity = float(np.mean(run["p_infinity"]))
    abs_m = float(np.mean(run["abs_magnetization"]))
    assert p_infinity == pytest.approx(abs_m, abs=0.05)

    # The same statement seen from inside the cluster: essentially all of the
    # percolating cluster's spins point the same way.
    assert float(np.mean(run["cluster_magnetization_ratio"])) > 0.9
