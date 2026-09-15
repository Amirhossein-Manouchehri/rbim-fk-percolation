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
from rbim.simulation import derive_observables, run_single_realization, run_temperature_scan

TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))  # 2.269185...


def _ferromagnetic_couplings(L):
    ones = np.ones((L, L), dtype=np.int8)
    return couplings_from_disorder(ones, ones)


def _observables(L, temperature, **kwargs):
    """Run one realization and return its derived observables."""
    J = _ferromagnetic_couplings(L)
    result = run_single_realization(L, J, temperature, **kwargs)
    return derive_observables(result.primitives, temperature, L * L)


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
    for name, value in first.primitives.items():
        assert value == pytest.approx(second.primitives[name])


def test_different_seeds_disagree():
    L = 8
    J = _ferromagnetic_couplings(L)
    kwargs = dict(n_sweeps=200, n_therm=50, n_bins=4)
    first = run_single_realization(L, J, 2.5, seed=1, **kwargs)
    second = run_single_realization(L, J, 2.5, seed=2, **kwargs)
    assert first.primitives["energy"] != pytest.approx(second.primitives["energy"])


def test_binning_does_not_change_central_values():
    """Regression test for a biased estimator.

    Variances and moment ratios used to be evaluated inside each bin and then
    averaged, which made the Binder cumulant and the specific heat depend
    strongly on the bin count.  Central values now come from the full run, so
    the number of bins must be irrelevant.
    """
    L = 12
    J = _ferromagnetic_couplings(L)
    reference = None
    for n_bins in (2, 8, 40):
        result = run_single_realization(
            L, J, 2.3, n_sweeps=800, n_therm=200, n_bins=n_bins, seed=77
        )
        derived = derive_observables(result.primitives, 2.3, L * L)
        if reference is None:
            reference = derived
        else:
            for name, value in reference.items():
                assert derived[name] == pytest.approx(value, rel=1e-12, abs=1e-12)


@pytest.mark.slow
def test_energy_stays_within_bounds():
    for temperature in (1.0, 2.269185, 5.0):
        derived = _observables(12, temperature, n_sweeps=400, n_therm=200, n_bins=4, seed=5)
        assert -2.0 <= derived["energy"] <= 0.0


@pytest.mark.slow
def test_spontaneous_magnetization_matches_onsager():
    """Below Tc the magnetization should track the exact Onsager curve."""
    temperature = 2.0
    derived = _observables(16, temperature, n_sweeps=4000, n_therm=1000, n_bins=10, seed=2024)
    exact = onsager_magnetization(temperature)

    # A finite lattice sits slightly above the thermodynamic curve; 4% covers
    # the finite-size offset at L = 16 together with the statistical error.
    assert derived["abs_magnetization"] == pytest.approx(exact, abs=0.04)


@pytest.mark.slow
def test_disordered_phase_has_small_magnetization():
    derived = _observables(16, 4.0, n_sweeps=2000, n_therm=500, n_bins=10, seed=11)
    assert derived["abs_magnetization"] < 0.15


@pytest.mark.slow
def test_percolation_strength_follows_magnetization_in_the_pure_model():
    """For p = 0 the FK percolation strength equals the magnetization.

    This is the Coniglio-Klein identity, and its breakdown at p > 0 is the
    phenomenon the project is built around, so the p = 0 limit is worth
    pinning down.
    """
    derived = _observables(16, 2.0, n_sweeps=4000, n_therm=1000, n_bins=10, seed=99)

    assert derived["p_infinity"] == pytest.approx(derived["abs_magnetization"], abs=0.05)

    # The same statement seen from inside the cluster: essentially all of the
    # percolating cluster's spins point the same way.
    assert derived["cluster_magnetization_ratio"] > 0.9


@pytest.mark.slow
def test_binder_cumulant_brackets_its_limits():
    """U -> 2/3 deep in the ordered phase and -> 0 deep in the disordered one."""
    ordered = run_temperature_scan(
        12, 0.0, [1.2], n_realizations=1, n_sweeps=3000, n_therm=600, n_bins=10,
        seed=4, n_workers=1,
    )
    disordered = run_temperature_scan(
        12, 0.0, [5.0], n_realizations=1, n_sweeps=3000, n_therm=600, n_bins=10,
        seed=4, n_workers=1,
    )
    assert ordered["observables"]["binder"]["value"][0] == pytest.approx(2.0 / 3.0, abs=0.03)
    assert abs(disordered["observables"]["binder"]["value"][0]) < 0.1
