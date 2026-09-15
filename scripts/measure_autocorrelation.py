#!/usr/bin/env python3
"""Integrated autocorrelation times of the cluster observables across the transition.

Swendsen-Wang is known to lose efficiency once the Fortuin-Kasteleyn cluster
percolates: the update then flips a system-spanning cluster, which is close to
the global spin-flip symmetry and therefore changes nothing gauge invariant.
Since the cluster-structure observables live *inside* that regime, it matters
quantitatively how badly the dynamics slows down there, and for which
observables.

This script measures the integrated autocorrelation time

    tau_int = 1/2 + sum_{t>=1} rho(t)

with Sokal's self-consistent window, for the energy and for the cluster
observables, as a function of temperature.  The ratio of the run length to
tau_int is the number of effectively independent measurements, which is what
determines whether a given observable can be trusted at a given temperature.

Example
-------
    python scripts/measure_autocorrelation.py --L 32 --p 0.05 0.20 --sweeps 20000
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Tuple

import numpy as np

from rbim.disorder import generate_disorder
from rbim.lattice import build_lattice, couplings_from_disorder
from rbim.swendsen_wang import random_spins, sw_sweep

TRACKED = ("energy", "largest_size", "largest_abs_mag", "mag_ratio", "wrap_y")


def integrated_autocorrelation(series: np.ndarray, c: float = 6.0) -> Tuple[float, int]:
    """Return ``tau_int`` and the window used, via Sokal's automatic windowing.

    The autocorrelation function is obtained by FFT.  The window grows until it
    reaches ``c`` times the running estimate of ``tau_int``, which balances bias
    against variance; ``c = 6`` is the usual choice for exponential-ish decay.
    """
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 16 or not np.any(x):
        return 0.5, 0

    padded = 1
    while padded < 2 * n:
        padded *= 2
    spectrum = np.fft.rfft(x, padded)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum))[:n].real
    if acf[0] <= 0:
        return 0.5, 0
    acf /= acf[0]

    tau = 0.5
    for window in range(1, n):
        tau += acf[window]
        if window >= c * tau:
            return float(max(tau, 0.5)), window
    return float(max(tau, 0.5)), n - 1


def collect_series(
    L: int, J: np.ndarray, temperature: float, n_sweeps: int, n_therm: int, seed
) -> Dict[str, np.ndarray]:
    """Run one realization and return the raw time series of each observable."""
    lattice = build_lattice(L)
    rng = np.random.default_rng(seed)
    beta = 1.0 / temperature
    spins = random_spins(lattice.n_sites, rng)

    for _ in range(n_therm):
        sw_sweep(spins, lattice, J, beta, rng, measure=False)

    series = {name: np.empty(n_sweeps) for name in TRACKED}
    for i in range(n_sweeps):
        result = sw_sweep(spins, lattice, J, beta, rng, measure=True)
        largest = result.largest_size
        series["energy"][i] = result.energy
        series["largest_size"][i] = largest
        series["largest_abs_mag"][i] = result.largest_abs_magnetization
        series["mag_ratio"][i] = result.largest_abs_magnetization / largest if largest else 0.0
        series["wrap_y"][i] = float(result.wraps_y)
    return series


def scan(
    L: int,
    p: float,
    temperatures: Sequence[float],
    n_sweeps: int,
    n_therm: int,
    n_realizations: int,
    seed: int,
) -> None:
    print(f"\n=== L = {L}, p = {p} ===")
    header = f"{'T':>6} " + " ".join(f"{name[:16]:>17}" for name in TRACKED)
    print(header)
    print("-" * len(header))

    seeds = np.random.SeedSequence(seed)
    disorder_seed, run_seed = seeds.spawn(2)
    disorder_rng = np.random.default_rng(disorder_seed)
    couplings = [
        couplings_from_disorder(*generate_disorder(L, p, disorder_rng))
        for _ in range(n_realizations)
    ]
    children = run_seed.spawn(len(temperatures) * n_realizations)

    k = 0
    for temperature in temperatures:
        taus: Dict[str, List[float]] = {name: [] for name in TRACKED}
        for J in couplings:
            series = collect_series(L, J, temperature, n_sweeps, n_therm, children[k])
            k += 1
            for name in TRACKED:
                taus[name].append(integrated_autocorrelation(series[name])[0])
        row = f"{temperature:6.2f} "
        for name in TRACKED:
            tau = float(np.mean(taus[name]))
            independent = n_sweeps / (2.0 * tau)
            row += f" {tau:7.1f} ({independent:6.0f})"
        print(row)
    print("  values are tau_int (effectively independent measurements in parentheses)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--p", type=float, nargs="+", default=[0.05, 0.20])
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[2.6, 2.3, 2.1, 1.9, 1.7, 1.5, 1.3, 1.1])
    parser.add_argument("--sweeps", type=int, default=20000)
    parser.add_argument("--therm", type=int, default=2000)
    parser.add_argument("--realizations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for p in args.p:
        scan(args.L, p, args.temperatures, args.sweeps, args.therm, args.realizations, args.seed)


if __name__ == "__main__":
    main()
