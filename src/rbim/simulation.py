"""Thermalisation, binning, temperature scans and disorder averaging.

The two error bars reported by :func:`run_temperature_scan` are kept separate
on purpose.  In a disordered system the sample-to-sample fluctuation usually
dominates the statistical uncertainty, so quoting only the thermal error of a
single realization understates it badly.  Here every observable carries

``error``
    the standard error over disorder realizations, which already contains the
    thermal noise of each realization and is the number to quote;
``thermal_error``
    the mean within-realization standard error, reported for diagnostics so
    that the two contributions can be compared.

With a single realization the disorder error is undefined and the thermal
error is reported in its place, with ``n_realizations = 1`` recorded in the
metadata so the limitation stays visible.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from rbim.clusters import bond_activation_probability  # noqa: F401  (re-exported for convenience)
from rbim.disorder import generate_disorder
from rbim.lattice import build_lattice, couplings_from_disorder
from rbim.swendsen_wang import random_spins, sw_sweep

__all__ = [
    "OBSERVABLES",
    "run_single_realization",
    "run_temperature_scan",
    "save_results",
    "load_results",
]

#: Observables recorded for every bin.  Energies and magnetizations are per
#: site; ``largest_size`` and ``largest_abs_magnetization`` are deliberately
#: left extensive, because the exponent theta in ``M_C ~ S^theta`` is extracted
#: from their system-size dependence.
OBSERVABLES = (
    "energy",
    "specific_heat",
    "magnetization",
    "abs_magnetization",
    "susceptibility",
    "binder",
    "p_infinity",
    "largest_size",
    "largest_abs_magnetization",
    "cluster_magnetization_ratio",
    "second_size_fraction",
    "cluster_density_gap",
    "n_clusters_per_site",
    "wrap_x",
    "wrap_y",
    "wrap_any",
)


def _reduce_bin(
    energies: np.ndarray,
    magnetizations: np.ndarray,
    largest_sizes: np.ndarray,
    largest_mags: np.ndarray,
    second_sizes: np.ndarray,
    n_clusters: np.ndarray,
    wrap_x: np.ndarray,
    wrap_y: np.ndarray,
    n_sites: int,
    temperature: float,
    measure_percolation: bool,
) -> Dict[str, float]:
    """Collapse the per-sweep samples of one bin into scalar observables."""
    m = magnetizations / n_sites
    abs_m = np.abs(m)

    m2 = float(np.mean(m**2))
    m4 = float(np.mean(m**4))
    binder = 1.0 - m4 / (3.0 * m2**2) if m2 > 0.0 else 0.0

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(largest_sizes > 0, largest_mags / np.maximum(largest_sizes, 1), 0.0)

    values = {
        "energy": float(np.mean(energies)) / n_sites,
        "specific_heat": float(np.var(energies)) / (temperature**2 * n_sites),
        "magnetization": float(np.mean(m)),
        "abs_magnetization": float(np.mean(abs_m)),
        "susceptibility": float(np.var(np.abs(magnetizations))) / (temperature * n_sites),
        "binder": binder,
        "p_infinity": float(np.mean(largest_sizes)) / n_sites,
        "largest_size": float(np.mean(largest_sizes)),
        "largest_abs_magnetization": float(np.mean(largest_mags)),
        "cluster_magnetization_ratio": float(np.mean(ratio)),
        "second_size_fraction": float(np.mean(second_sizes)) / n_sites,
        "cluster_density_gap": float(np.mean(largest_sizes - second_sizes)) / n_sites,
        "n_clusters_per_site": float(np.mean(n_clusters)) / n_sites,
    }

    if measure_percolation:
        values["wrap_x"] = float(np.mean(wrap_x))
        values["wrap_y"] = float(np.mean(wrap_y))
        values["wrap_any"] = float(np.mean(wrap_x | wrap_y))
    else:
        values["wrap_x"] = float("nan")
        values["wrap_y"] = float("nan")
        values["wrap_any"] = float("nan")

    return values


def run_single_realization(
    L: int,
    J: np.ndarray,
    temperature: float,
    *,
    n_sweeps: int,
    n_therm: int,
    n_bins: int = 10,
    seed: Optional[int | np.random.SeedSequence] = None,
    measure_percolation: bool = True,
) -> Dict[str, np.ndarray]:
    """Simulate one disorder realization at one temperature.

    Parameters
    ----------
    L:
        Linear system size.
    J:
        Bond couplings of shape ``(2 * L * L,)``.
    temperature:
        Temperature in units of ``J / k_B``.
    n_sweeps:
        Number of measured Swendsen-Wang sweeps; split evenly into ``n_bins``
        bins, with any remainder discarded.
    n_therm:
        Number of discarded thermalisation sweeps.  No automatic equilibration
        test is performed; for the low-temperature, strongly frustrated regime
        this value should be checked explicitly, for instance by comparing hot
        and cold starts.
    n_bins:
        Number of bins used for the error estimate.
    seed:
        Seed or :class:`numpy.random.SeedSequence` for the generator.
    measure_percolation:
        Whether to evaluate the wrapping observables, which roughly triple the
        cost of a measured sweep.

    Returns
    -------
    dict
        Maps each name in :data:`OBSERVABLES` to an array of ``n_bins`` bin
        values.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    sweeps_per_bin = n_sweeps // n_bins
    if sweeps_per_bin < 1:
        raise ValueError(
            f"n_sweeps ({n_sweeps}) must be at least n_bins ({n_bins}) so that every bin "
            "contains at least one sweep."
        )

    lattice = build_lattice(L)
    rng = np.random.default_rng(seed)
    beta = 1.0 / temperature
    n_sites = lattice.n_sites

    spins = random_spins(n_sites, rng)

    for _ in range(n_therm):
        sw_sweep(spins, lattice, J, beta, rng, measure=False)

    bins: Dict[str, List[float]] = {name: [] for name in OBSERVABLES}

    energies = np.empty(sweeps_per_bin)
    magnetizations = np.empty(sweeps_per_bin)
    largest_sizes = np.empty(sweeps_per_bin)
    largest_mags = np.empty(sweeps_per_bin)
    second_sizes = np.empty(sweeps_per_bin)
    n_clusters = np.empty(sweeps_per_bin)
    wrap_x = np.zeros(sweeps_per_bin, dtype=bool)
    wrap_y = np.zeros(sweeps_per_bin, dtype=bool)

    for _ in range(n_bins):
        for i in range(sweeps_per_bin):
            result = sw_sweep(
                spins,
                lattice,
                J,
                beta,
                rng,
                measure=True,
                measure_percolation=measure_percolation,
            )
            energies[i] = result.energy
            magnetizations[i] = result.magnetization
            largest_sizes[i] = result.largest_size
            largest_mags[i] = result.largest_abs_magnetization
            second_sizes[i] = result.second_size
            n_clusters[i] = result.n_clusters
            if measure_percolation:
                wrap_x[i] = result.wraps_x
                wrap_y[i] = result.wraps_y

        reduced = _reduce_bin(
            energies,
            magnetizations,
            largest_sizes,
            largest_mags,
            second_sizes,
            n_clusters,
            wrap_x,
            wrap_y,
            n_sites,
            temperature,
            measure_percolation,
        )
        for name, value in reduced.items():
            bins[name].append(value)

    return {name: np.asarray(values) for name, values in bins.items()}


def _worker(task: tuple) -> Dict[str, np.ndarray]:
    """Process-pool entry point; arguments are packed to keep pickling simple."""
    L, J, temperature, n_sweeps, n_therm, n_bins, seed_seq, measure_percolation = task
    return run_single_realization(
        L,
        J,
        temperature,
        n_sweeps=n_sweeps,
        n_therm=n_therm,
        n_bins=n_bins,
        seed=seed_seq,
        measure_percolation=measure_percolation,
    )


def run_temperature_scan(
    L: int,
    p: float,
    temperatures: Sequence[float],
    *,
    n_realizations: int = 1,
    n_sweeps: int = 10_000,
    n_therm: Optional[int] = None,
    n_bins: int = 10,
    seed: int = 0,
    n_workers: Optional[int] = None,
    measure_percolation: bool = True,
    disorder: Optional[Iterable[tuple]] = None,
) -> Dict[str, object]:
    """Scan a range of temperatures, averaging over disorder realizations.

    Parameters
    ----------
    L, p:
        Linear size and antiferromagnetic bond concentration.
    temperatures:
        Temperatures to simulate.
    n_realizations:
        Number of independent disorder realizations.  More than one is needed
        for a meaningful error bar.
    n_sweeps, n_therm, n_bins:
        Measurement, thermalisation and binning controls.  ``n_therm`` defaults
        to one tenth of ``n_sweeps``.
    seed:
        Master seed.  Every task receives an independent child stream derived
        from it via :class:`numpy.random.SeedSequence`, so results are
        reproducible and workers are guaranteed not to share a stream.
    n_workers:
        Process-pool size; ``None`` lets the pool choose, ``1`` runs serially.
    disorder:
        Optional iterable of ``(J_h, J_v)`` pairs to use instead of generating
        fresh realizations, for example ones loaded from disk.

    Returns
    -------
    dict
        ``{"temperatures": ..., "observables": {name: {"value", "error",
        "thermal_error"}}, "metadata": {...}}`` with one entry per temperature.
    """
    temperatures = list(temperatures)
    if n_therm is None:
        n_therm = max(1, n_sweeps // 10)

    seed_sequence = np.random.SeedSequence(seed)
    disorder_seeds, run_seeds = seed_sequence.spawn(2)

    if disorder is None:
        disorder_rng = np.random.default_rng(disorder_seeds)
        realizations = [generate_disorder(L, p, disorder_rng) for _ in range(n_realizations)]
    else:
        realizations = list(disorder)[:n_realizations]
        if len(realizations) < n_realizations:
            raise ValueError(
                f"only {len(realizations)} disorder realizations supplied but "
                f"n_realizations={n_realizations}"
            )

    couplings = [couplings_from_disorder(J_h, J_v) for J_h, J_v in realizations]

    n_tasks = len(temperatures) * len(couplings)
    children = run_seeds.spawn(n_tasks)

    tasks = []
    k = 0
    for temperature in temperatures:
        for J in couplings:
            tasks.append(
                (L, J, temperature, n_sweeps, n_therm, n_bins, children[k], measure_percolation)
            )
            k += 1

    if n_workers == 1:
        raw = [_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            raw = list(pool.map(_worker, tasks))

    results: Dict[str, Dict[str, np.ndarray]] = {
        name: {
            "value": np.empty(len(temperatures)),
            "error": np.empty(len(temperatures)),
            "thermal_error": np.empty(len(temperatures)),
        }
        for name in OBSERVABLES
    }

    n_realizations_actual = len(couplings)
    for t_index in range(len(temperatures)):
        block = raw[t_index * n_realizations_actual : (t_index + 1) * n_realizations_actual]
        for name in OBSERVABLES:
            per_realization = np.array([np.mean(run[name]) for run in block])
            thermal = np.array(
                [np.std(run[name], ddof=1) / np.sqrt(len(run[name])) if len(run[name]) > 1 else 0.0
                 for run in block]
            )
            mean_thermal = float(np.sqrt(np.mean(thermal**2) / n_realizations_actual))

            value = float(np.mean(per_realization))
            if n_realizations_actual > 1:
                error = float(np.std(per_realization, ddof=1) / np.sqrt(n_realizations_actual))
            else:
                error = mean_thermal

            results[name]["value"][t_index] = value
            results[name]["error"][t_index] = error
            results[name]["thermal_error"][t_index] = mean_thermal

    metadata = {
        "L": L,
        "p": p,
        "n_realizations": n_realizations_actual,
        "n_sweeps": n_sweeps,
        "n_therm": n_therm,
        "n_bins": n_bins,
        "seed": seed,
        "measure_percolation": measure_percolation,
        "algorithm": "Swendsen-Wang cluster update on satisfied bonds",
        "binder_convention": "1 - <m^4> / (3 <m^2>^2)",
        "susceptibility_convention": "var(|M|) / (T N)",
        "error_convention": (
            "error = standard error over disorder realizations; "
            "thermal_error = mean within-realization standard error"
        ),
    }

    return {"temperatures": np.asarray(temperatures), "observables": results, "metadata": metadata}


def save_results(path: str, results: Dict[str, object]) -> None:
    """Write a scan produced by :func:`run_temperature_scan` to a ``.npz`` file."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    flat = {"temperatures": results["temperatures"]}
    for name, entry in results["observables"].items():
        for key, array in entry.items():
            flat[f"{name}__{key}"] = array
    flat["metadata_json"] = np.asarray(json.dumps(results["metadata"]))

    np.savez_compressed(path, **flat)


def load_results(path: str) -> Dict[str, object]:
    """Read a scan written by :func:`save_results`."""
    with np.load(path, allow_pickle=False) as data:
        temperatures = data["temperatures"]
        metadata = json.loads(str(data["metadata_json"]))
        observables: Dict[str, Dict[str, np.ndarray]] = {}
        for key in data.files:
            if "__" not in key:
                continue
            name, field = key.split("__", 1)
            observables.setdefault(name, {})[field] = data[key]
    return {"temperatures": temperatures, "observables": observables, "metadata": metadata}
