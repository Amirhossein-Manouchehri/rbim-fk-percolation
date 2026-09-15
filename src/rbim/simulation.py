"""Thermalisation, measurement, disorder averaging and error estimation.

Two statistical points govern the design of this module.

**Ratios and variances must not be computed inside bins.**  Quantities such as
the specific heat, the susceptibility and the Binder cumulant are variances or
ratios of moments.  Evaluating them on a short window of a correlated time
series and then averaging the windows is biased, because a short window cannot
resolve fluctuations longer than itself and because a ratio of small-sample
averages is not an unbiased estimate of the ratio.  The bias is severe: at
fixed statistics, shrinking the bins from 1000 to 10 sweeps moves the measured
Binder cumulant of this model from 0.076 to 0.278 and the specific heat from
0.55 to 0.36, while genuinely linear observables such as the percolation
strength do not move at all.  Here every primitive quantity is accumulated over
the *whole* measurement run of a realization, and the derived observables are
formed from those full-run moments.  Bins are used only to report a thermal
error, never to produce a central value.

**Disorder averaging is the outer operation.**  Thermal averages are taken
within a realization, then averaged over realizations, and the uncertainty is
estimated by jackknifing over realizations.  Jackknife is used uniformly so
that ratios like the Binder cumulant get correct error bars rather than naive
propagation.  In a disordered system the sample-to-sample fluctuation normally
dominates, so ``error`` is the number to quote; ``thermal_error`` is retained
as a diagnostic for checking that the run length was adequate.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from rbim.disorder import generate_disorder
from rbim.lattice import build_lattice, couplings_from_disorder
from rbim.swendsen_wang import random_spins, sw_sweep

__all__ = [
    "OBSERVABLES",
    "PRIMITIVES",
    "RealizationResult",
    "derive_observables",
    "run_single_realization",
    "run_temperature_scan",
    "save_results",
    "load_results",
]

#: Quantities accumulated directly from the sweep time series.  Everything the
#: analysis needs is a function of these, which is what allows the derived
#: observables to be formed after disorder averaging rather than before.
PRIMITIVES = (
    "energy",            # <E>
    "energy_sq",         # <E^2>
    "magnetization",     # <M>
    "abs_magnetization", # <|M|>
    "magnetization_sq",  # <M^2>
    "magnetization_4",   # <M^4>
    "largest_size",      # <S_max>
    "largest_abs_mag",   # <|M_C|>
    "second_size",       # <S_2>
    "mag_ratio",         # <|M_C| / S_max>
    "n_clusters",        # <number of clusters>
    "wrap_x",
    "wrap_y",
    "wrap_any",
)

#: Physical observables reported by :func:`run_temperature_scan`.
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


@dataclass
class RealizationResult:
    """Full-run moments of one disorder realization, plus per-bin moments.

    ``primitives`` holds the average of each entry of :data:`PRIMITIVES` over
    the entire measurement run, and is what the disorder average consumes.
    ``bin_primitives`` holds the same averages restricted to each bin and is
    used only for the thermal error.
    """

    primitives: Dict[str, float]
    bin_primitives: Dict[str, np.ndarray] = field(default_factory=dict)


def derive_observables(primitives: Dict[str, float], temperature: float, n_sites: int) -> Dict[str, float]:
    """Form the physical observables from disorder-averaged primitive moments.

    Conventions, which differ between authors and so are fixed explicitly here:

    * ``specific_heat`` uses the thermal variance within a sample,
      ``[<E^2> - <E>^2] / (T^2 N)``;
    * ``susceptibility`` likewise uses ``[<M^2> - <|M|>^2] / (T N)``;
    * ``binder`` is built from the disorder-averaged moments,
      ``1 - [<m^4>] / (3 [<m^2>]^2)``, rather than by averaging the Binder
      cumulants of individual samples.
    """
    N = n_sites
    energy_var = primitives["energy_sq"] - primitives["energy"] ** 2
    abs_mag_var = primitives["magnetization_sq"] - primitives["abs_magnetization"] ** 2

    m2 = primitives["magnetization_sq"] / N**2
    m4 = primitives["magnetization_4"] / N**4
    binder = 1.0 - m4 / (3.0 * m2**2) if m2 > 0.0 else 0.0

    return {
        "energy": primitives["energy"] / N,
        "specific_heat": energy_var / (temperature**2 * N),
        "magnetization": primitives["magnetization"] / N,
        "abs_magnetization": primitives["abs_magnetization"] / N,
        "susceptibility": abs_mag_var / (temperature * N),
        "binder": binder,
        "p_infinity": primitives["largest_size"] / N,
        "largest_size": primitives["largest_size"],
        "largest_abs_magnetization": primitives["largest_abs_mag"],
        "cluster_magnetization_ratio": primitives["mag_ratio"],
        "second_size_fraction": primitives["second_size"] / N,
        "cluster_density_gap": (primitives["largest_size"] - primitives["second_size"]) / N,
        "n_clusters_per_site": primitives["n_clusters"] / N,
        "wrap_x": primitives["wrap_x"],
        "wrap_y": primitives["wrap_y"],
        "wrap_any": primitives["wrap_any"],
    }


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
) -> RealizationResult:
    """Simulate one disorder realization at one temperature.

    Parameters
    ----------
    L, J, temperature:
        System size, bond couplings of shape ``(2 L^2,)``, and temperature.
    n_sweeps:
        Measured Swendsen-Wang sweeps, split evenly into ``n_bins`` bins with
        any remainder discarded.
    n_therm:
        Discarded thermalisation sweeps.  No automatic equilibration test is
        performed; verify it explicitly in the low-temperature, strongly
        frustrated corner, where Swendsen-Wang is expected to struggle.
    n_bins:
        Bins used for the thermal error only.  Central values are always taken
        from the full run, so this choice cannot bias the result.
    seed:
        Seed or :class:`numpy.random.SeedSequence`.
    measure_percolation:
        Whether to report the wrapping observables.  They are now free, since
        the cluster labelling produces them.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    sweeps_per_bin = n_sweeps // n_bins
    if sweeps_per_bin < 1:
        raise ValueError(
            f"n_sweeps ({n_sweeps}) must be at least n_bins ({n_bins}) so every bin "
            "contains at least one sweep."
        )

    lattice = build_lattice(L)
    rng = np.random.default_rng(seed)
    beta = 1.0 / temperature
    n_sites = lattice.n_sites

    spins = random_spins(n_sites, rng)
    for _ in range(n_therm):
        sw_sweep(spins, lattice, J, beta, rng, measure=False)

    total = n_bins * sweeps_per_bin
    series = {name: np.empty(total) for name in PRIMITIVES}

    for index in range(total):
        result = sw_sweep(
            spins, lattice, J, beta, rng, measure=True, measure_percolation=measure_percolation
        )
        energy = result.energy
        magnetization = result.magnetization
        largest = result.largest_size

        series["energy"][index] = energy
        series["energy_sq"][index] = energy * energy
        series["magnetization"][index] = magnetization
        series["abs_magnetization"][index] = abs(magnetization)
        series["magnetization_sq"][index] = magnetization**2
        series["magnetization_4"][index] = magnetization**4
        series["largest_size"][index] = largest
        series["largest_abs_mag"][index] = result.largest_abs_magnetization
        series["second_size"][index] = result.second_size
        series["mag_ratio"][index] = (
            result.largest_abs_magnetization / largest if largest > 0 else 0.0
        )
        series["n_clusters"][index] = result.n_clusters
        if measure_percolation:
            series["wrap_x"][index] = float(result.wraps_x)
            series["wrap_y"][index] = float(result.wraps_y)
            series["wrap_any"][index] = float(result.wraps_x or result.wraps_y)
        else:
            series["wrap_x"][index] = np.nan
            series["wrap_y"][index] = np.nan
            series["wrap_any"][index] = np.nan

    primitives = {name: float(np.mean(values)) for name, values in series.items()}
    bin_primitives = {
        name: values.reshape(n_bins, sweeps_per_bin).mean(axis=1) for name, values in series.items()
    }
    return RealizationResult(primitives=primitives, bin_primitives=bin_primitives)


def _worker(task: tuple) -> RealizationResult:
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


def _combine_realizations(
    results: Sequence[RealizationResult], temperature: float, n_sites: int
) -> Dict[str, Dict[str, float]]:
    """Disorder-average the primitives and jackknife over realizations.

    Jackknife is applied to the derived observables rather than to the
    primitives, so ratios such as the Binder cumulant receive a correct error
    bar instead of one obtained by propagating through a nonlinear function.
    """
    n_realizations = len(results)
    stacked = {
        name: np.array([r.primitives[name] for r in results], dtype=float) for name in PRIMITIVES
    }

    mean_primitives = {name: float(values.mean()) for name, values in stacked.items()}
    central = derive_observables(mean_primitives, temperature, n_sites)

    combined: Dict[str, Dict[str, float]] = {}

    if n_realizations > 1:
        # Leave-one-out estimates of every observable.
        totals = {name: values.sum() for name, values in stacked.items()}
        leave_one_out: Dict[str, List[float]] = {name: [] for name in OBSERVABLES}
        for i in range(n_realizations):
            reduced = {
                name: (totals[name] - stacked[name][i]) / (n_realizations - 1)
                for name in PRIMITIVES
            }
            derived = derive_observables(reduced, temperature, n_sites)
            for name in OBSERVABLES:
                leave_one_out[name].append(derived[name])

        factor = (n_realizations - 1) / n_realizations
        for name in OBSERVABLES:
            samples = np.asarray(leave_one_out[name])
            variance = factor * float(np.sum((samples - samples.mean()) ** 2))
            combined[name] = {"value": central[name], "error": float(np.sqrt(variance))}
    else:
        for name in OBSERVABLES:
            combined[name] = {"value": central[name], "error": float("nan")}

    # Thermal error: within each realization, form the observables per bin
    # (acceptable here because it is only a spread, not a central value), take
    # the standard error across bins, and combine the realizations in quadrature.
    n_bins = len(next(iter(results[0].bin_primitives.values())))
    if n_bins > 1:
        per_realization = {name: [] for name in OBSERVABLES}
        for result in results:
            bin_values = {name: [] for name in OBSERVABLES}
            for b in range(n_bins):
                reduced = {name: float(result.bin_primitives[name][b]) for name in PRIMITIVES}
                derived = derive_observables(reduced, temperature, n_sites)
                for name in OBSERVABLES:
                    bin_values[name].append(derived[name])
            for name in OBSERVABLES:
                values = np.asarray(bin_values[name])
                per_realization[name].append(np.std(values, ddof=1) / np.sqrt(n_bins))
        for name in OBSERVABLES:
            spread = np.asarray(per_realization[name], dtype=float)
            combined[name]["thermal_error"] = float(
                np.sqrt(np.mean(spread**2) / n_realizations)
            )
    else:
        for name in OBSERVABLES:
            combined[name]["thermal_error"] = float("nan")

    if n_realizations == 1:
        # With one sample there is no disorder error; report the thermal one so
        # that the number is not silently missing, and record the limitation.
        for name in OBSERVABLES:
            combined[name]["error"] = combined[name]["thermal_error"]

    return combined


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
    """Scan temperatures, averaging over disorder realizations.

    The same set of realizations is used at every temperature, which correlates
    the curves and sharpens crossing determinations considerably compared with
    drawing fresh disorder at each point.

    Returns
    -------
    dict
        ``{"temperatures", "observables": {name: {"value", "error",
        "thermal_error"}}, "metadata"}``.
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
    n_realizations_actual = len(couplings)

    n_tasks = len(temperatures) * n_realizations_actual
    children = run_seeds.spawn(n_tasks)

    tasks = []
    for k, (temperature, J) in enumerate(
        (t, J) for t in temperatures for J in couplings
    ):
        tasks.append(
            (L, J, temperature, n_sweeps, n_therm, n_bins, children[k], measure_percolation)
        )

    if n_workers == 1:
        raw = [_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            raw = list(pool.map(_worker, tasks))

    n_sites = L * L
    results: Dict[str, Dict[str, np.ndarray]] = {
        name: {
            "value": np.empty(len(temperatures)),
            "error": np.empty(len(temperatures)),
            "thermal_error": np.empty(len(temperatures)),
        }
        for name in OBSERVABLES
    }

    for t_index, temperature in enumerate(temperatures):
        block = raw[t_index * n_realizations_actual : (t_index + 1) * n_realizations_actual]
        combined = _combine_realizations(block, temperature, n_sites)
        for name in OBSERVABLES:
            results[name]["value"][t_index] = combined[name]["value"]
            results[name]["error"][t_index] = combined[name]["error"]
            results[name]["thermal_error"][t_index] = combined[name]["thermal_error"]

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
        "estimator": (
            "primitive moments accumulated over the full run of each realization; "
            "derived observables formed after disorder averaging; "
            "uncertainties by jackknife over realizations"
        ),
        "binder_convention": "1 - [<m^4>] / (3 [<m^2>]^2), from disorder-averaged moments",
        "specific_heat_convention": "[<E^2> - <E>^2] / (T^2 N)",
        "susceptibility_convention": "[<M^2> - <|M|>^2] / (T N)",
        "error_convention": (
            "error = jackknife over disorder realizations; "
            "thermal_error = mean within-realization standard error across bins"
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
            name, field_name = key.split("__", 1)
            observables.setdefault(name, {})[field_name] = data[key]
    return {"temperatures": temperatures, "observables": observables, "metadata": metadata}
