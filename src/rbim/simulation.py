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

import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from rbim.disorder import generate_disorder
from rbim.lattice import build_lattice, couplings_from_disorder
from rbim.swendsen_wang import initial_spins, sw_sweep

__all__ = [
    "OBSERVABLES",
    "PRIMITIVES",
    "RealizationResult",
    "derive_observables",
    "run_single_realization",
    "run_temperature_scan",
    "save_results",
    "load_results",
    "remove_checkpoints",
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
    "second_abs_mag",    # <|M_2|>, second-largest cluster
    "largest_mag_sq",    # <M_C^2>
    "cluster_mag_sq_sum",  # <sum_C M_C^2>, equal to <M^2> in expectation
    "active_bonds",      # <number of active FK bonds>
    # The last two are not time-series averages: they are squares of the
    # full-run means of one realization, needed to form the connected
    # (within-sample) susceptibility and specific heat after disorder averaging.
    "energy_mean_sq",          # <E>^2
    "abs_magnetization_mean_sq",  # <|M|>^2
)

#: Primitives that are squares of run means rather than means of a series.
_MEAN_SQUARE_PRIMITIVES = {
    "energy_mean_sq": "energy",
    "abs_magnetization_mean_sq": "abs_magnetization",
}

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
    # Added after the first production scans.  Files written before then do not
    # contain these keys; everything above is unchanged in name and definition.
    "magnetization_sq",
    "second_abs_magnetization",
    "largest_magnetization_sq",
    "cluster_magnetization_sq_sum",
    "largest_cluster_m2_share",
    "fk_bond_density",
    "susceptibility_connected",
    "specific_heat_connected",
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

    Square brackets denote the disorder average and angle brackets the thermal
    average.  Conventions, which differ between authors, are fixed here:

    * ``specific_heat`` is ``([<E^2>] - [<E>]^2) / (T^2 N)`` and
      ``susceptibility`` is ``([<M^2>] - [<|M|>]^2) / (T N)``.  Both subtract
      the square of a *disorder-averaged* mean, so they contain the
      sample-to-sample variance of ``<E>`` and ``<|M|>`` on top of the thermal
      fluctuation.  They are kept under these names, unchanged, so that files
      from earlier runs remain comparable.
    * ``specific_heat_connected`` and ``susceptibility_connected`` subtract
      the square inside the disorder average, ``[<E^2> - <E>^2] / (T^2 N)``
      and ``[<M^2> - <|M|>^2] / (T N)``.  These are the within-sample thermal
      quantities.
    * ``magnetization_sq`` is ``[<M^2>]`` itself (extensive).
    * ``largest_magnetization_sq`` is ``[<M_C^2>]`` for the largest cluster and
      ``cluster_magnetization_sq_sum`` is ``[<sum_C M_C^2>]``, whose
      expectation equals ``[<M^2>]`` because clusters flip independently.
      ``largest_cluster_m2_share`` is the ratio of the two.
    * ``fk_bond_density`` is the fraction of bonds that are active.
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
        "magnetization_sq": primitives["magnetization_sq"],
        "second_abs_magnetization": primitives["second_abs_mag"],
        "largest_magnetization_sq": primitives["largest_mag_sq"],
        "cluster_magnetization_sq_sum": primitives["cluster_mag_sq_sum"],
        "largest_cluster_m2_share": (
            primitives["largest_mag_sq"] / primitives["cluster_mag_sq_sum"]
            if primitives["cluster_mag_sq_sum"] > 0.0
            else 0.0
        ),
        "fk_bond_density": primitives["active_bonds"] / (2 * N),
        "susceptibility_connected": (
            primitives["magnetization_sq"] - primitives["abs_magnetization_mean_sq"]
        )
        / (temperature * N),
        "specific_heat_connected": (
            primitives["energy_sq"] - primitives["energy_mean_sq"]
        )
        / (temperature**2 * N),
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
    start: str = "hot",
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
    start:
        ``"hot"`` for a random initial configuration or ``"cold"`` for a fully
        aligned one.  Below the ordering transition the two approach
        equilibrium from opposite sides, so running both and comparing is the
        way to establish that a result is equilibrated.
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

    spins = initial_spins(n_sites, rng, start)
    for _ in range(n_therm):
        sw_sweep(spins, lattice, J, beta, rng, measure=False)

    total = n_bins * sweeps_per_bin
    series = {name: np.empty(total) for name in PRIMITIVES if name not in _MEAN_SQUARE_PRIMITIVES}

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
        series["second_abs_mag"][index] = result.second_abs_magnetization
        series["largest_mag_sq"][index] = result.largest_abs_magnetization**2
        series["cluster_mag_sq_sum"][index] = result.cluster_magnetization_sq_sum
        series["active_bonds"][index] = result.n_active_bonds
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
    for name, source in _MEAN_SQUARE_PRIMITIVES.items():
        primitives[name] = primitives[source] ** 2
        bin_primitives[name] = bin_primitives[source] ** 2
    return RealizationResult(primitives=primitives, bin_primitives=bin_primitives)


def _worker(task: tuple) -> RealizationResult:
    """Process-pool entry point; arguments are packed to keep pickling simple."""
    L, J, temperature, n_sweeps, n_therm, n_bins, seed_seq, measure_percolation, start = task
    return run_single_realization(
        L,
        J,
        temperature,
        n_sweeps=n_sweeps,
        n_therm=n_therm,
        n_bins=n_bins,
        seed=seed_seq,
        measure_percolation=measure_percolation,
        start=start,
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
    start: str = "hot",
    disorder: Optional[Iterable[tuple]] = None,
    checkpoint_dir: Optional[str] = None,
    progress: bool = False,
) -> Dict[str, object]:
    """Scan temperatures, averaging over disorder realizations.

    The same set of realizations is used at every temperature, which correlates
    the curves and sharpens crossing determinations considerably compared with
    drawing fresh disorder at each point.

    Parameters
    ----------
    checkpoint_dir:
        When given, the per-realization results of each temperature are written
        there as soon as that temperature finishes, and a later call with the
        same arguments reuses them instead of recomputing.  A checkpoint written
        with different run parameters or different disorder is refused rather
        than mixed in.  Seeds are assigned per (temperature, realization), so a
        resumed run gives exactly the result of an uninterrupted one.
    progress:
        Print one line per finished temperature.

    Returns
    -------
    dict
        ``{"temperatures", "observables": {name: {"value", "error",
        "thermal_error"}}, "metadata", "per_realization"}``, where
        ``per_realization`` holds ``names`` (the entries of
        :data:`PRIMITIVES`) and ``values`` of shape
        ``(n_temperatures, n_realizations, n_primitives)``.
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
            (L, J, temperature, n_sweeps, n_therm, n_bins, children[k], measure_percolation, start)
        )

    fingerprint = _run_fingerprint(
        L, p, n_realizations_actual, n_sweeps, n_therm, n_bins, seed,
        measure_percolation, start, couplings,
    )
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    raw: List[Optional[RealizationResult]] = [None] * n_tasks
    pending: List[int] = []
    for t_index, temperature in enumerate(temperatures):
        block = None
        if checkpoint_dir is not None:
            block = _load_checkpoint(
                _checkpoint_path(checkpoint_dir, t_index), fingerprint, temperature
            )
        if block is None:
            pending.append(t_index)
        else:
            raw[t_index * n_realizations_actual : (t_index + 1) * n_realizations_actual] = block
            if progress:
                print(f"  T={temperature:.6g}: loaded from checkpoint", flush=True)

    def _run_block(t_index: int, mapper) -> None:
        lo = t_index * n_realizations_actual
        hi = lo + n_realizations_actual
        block = list(mapper(tasks[lo:hi]))
        raw[lo:hi] = block
        if checkpoint_dir is not None:
            _save_checkpoint(
                _checkpoint_path(checkpoint_dir, t_index), block, fingerprint, temperatures[t_index]
            )
        if progress:
            print(
                f"  T={temperatures[t_index]:.6g}: done "
                f"({len(temperatures) - pending.index(t_index) - 1} temperature(s) left)",
                flush=True,
            )

    if pending:
        if n_workers == 1:
            for t_index in pending:
                _run_block(t_index, lambda block_tasks: map(_worker, block_tasks))
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                workers = n_workers or os.cpu_count() or 1
                chunksize = max(1, n_realizations_actual // (8 * workers))
                for t_index in pending:
                    _run_block(
                        t_index,
                        lambda block_tasks: pool.map(_worker, block_tasks, chunksize=chunksize),
                    )

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
        "start": start,
        "algorithm": "Swendsen-Wang cluster update on satisfied bonds",
        "estimator": (
            "primitive moments accumulated over the full run of each realization; "
            "derived observables formed after disorder averaging; "
            "uncertainties by jackknife over realizations"
        ),
        "binder_convention": "1 - [<m^4>] / (3 [<m^2>]^2), from disorder-averaged moments",
        "specific_heat_convention": (
            "([<E^2>] - [<E>]^2) / (T^2 N); includes sample-to-sample variance of <E>"
        ),
        "susceptibility_convention": (
            "([<M^2>] - [<|M|>]^2) / (T N); includes sample-to-sample variance of <|M|>"
        ),
        "specific_heat_connected_convention": "[<E^2> - <E>^2] / (T^2 N)",
        "susceptibility_connected_convention": "[<M^2> - <|M|>^2] / (T N)",
        "format_version": 2,
        "error_convention": (
            "error = jackknife over disorder realizations; "
            "thermal_error = mean within-realization standard error across bins"
        ),
    }

    per_realization = np.array(
        [
            [[r.primitives[name] for name in PRIMITIVES] for r in raw[
                t_index * n_realizations_actual : (t_index + 1) * n_realizations_actual
            ]]
            for t_index in range(len(temperatures))
        ],
        dtype=float,
    )

    return {
        "temperatures": np.asarray(temperatures),
        "observables": results,
        "metadata": metadata,
        "per_realization": {"names": list(PRIMITIVES), "values": per_realization},
    }


def _run_fingerprint(
    L, p, n_realizations, n_sweeps, n_therm, n_bins, seed, measure_percolation, start, couplings
) -> str:
    """Identify a run by everything that determines its per-realization results."""
    digest = hashlib.sha256()
    for J in couplings:
        digest.update(np.ascontiguousarray(J).tobytes())
    return json.dumps(
        {
            "L": int(L),
            "p": float(p),
            "n_realizations": int(n_realizations),
            "n_sweeps": int(n_sweeps),
            "n_therm": int(n_therm),
            "n_bins": int(n_bins),
            "seed": int(seed),
            "measure_percolation": bool(measure_percolation),
            "start": str(start),
            "disorder_sha256": digest.hexdigest(),
            "primitives": list(PRIMITIVES),
        },
        sort_keys=True,
    )


def _checkpoint_path(checkpoint_dir: str, t_index: int) -> str:
    return os.path.join(checkpoint_dir, f"T{t_index:04d}.npz")


def _save_checkpoint(
    path: str, block: Sequence[RealizationResult], fingerprint: str, temperature: float
) -> None:
    """Write one temperature's per-realization results, atomically."""
    primitives = np.array([[r.primitives[name] for name in PRIMITIVES] for r in block], dtype=float)
    bins = np.array([[r.bin_primitives[name] for name in PRIMITIVES] for r in block], dtype=float)
    temporary = path + ".tmp.npz"
    np.savez_compressed(
        temporary,
        primitives=primitives,
        bin_primitives=bins,
        temperature=np.asarray(float(temperature)),
        fingerprint=np.asarray(fingerprint),
    )
    os.replace(temporary, path)


def _load_checkpoint(
    path: str, fingerprint: str, temperature: float
) -> Optional[List[RealizationResult]]:
    """Read one temperature's checkpoint, or return ``None`` if there is none."""
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["fingerprint"]) != fingerprint or float(data["temperature"]) != float(temperature):
            raise ValueError(
                f"checkpoint {path} was written by a run with different parameters, "
                "temperatures or disorder; remove the checkpoint directory or choose another."
            )
        primitives = data["primitives"]
        bins = data["bin_primitives"]
    return [
        RealizationResult(
            primitives={name: float(primitives[r, k]) for k, name in enumerate(PRIMITIVES)},
            bin_primitives={name: bins[r, k].copy() for k, name in enumerate(PRIMITIVES)},
        )
        for r in range(primitives.shape[0])
    ]


def save_results(
    path: str, results: Dict[str, object], *, per_realization_path: Optional[str] = None
) -> None:
    """Write a scan produced by :func:`run_temperature_scan` to a ``.npz`` file.

    The summary file holds the disorder-averaged observables and is small.
    When ``per_realization_path`` is given, the full-run moments of every
    realization at every temperature are written there as well, so that the
    analysis can be redone (different estimators, bootstrap, subsets) without
    rerunning the simulation.  That file is tens of megabytes for a production
    scan and is not meant to be tracked in git.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    flat = {"temperatures": results["temperatures"]}
    for name, entry in results["observables"].items():
        for key, array in entry.items():
            flat[f"{name}__{key}"] = array
    flat["metadata_json"] = np.asarray(json.dumps(results["metadata"]))

    np.savez_compressed(path, **flat)

    if per_realization_path is not None:
        per_realization = results.get("per_realization")
        if per_realization is None:
            raise ValueError("results carry no per-realization data to save")
        os.makedirs(os.path.dirname(os.path.abspath(per_realization_path)), exist_ok=True)
        np.savez_compressed(
            per_realization_path,
            temperatures=results["temperatures"],
            names=np.asarray(per_realization["names"]),
            values=per_realization["values"],
            metadata_json=np.asarray(json.dumps(results["metadata"])),
        )


def remove_checkpoints(checkpoint_dir: str) -> None:
    """Delete a checkpoint directory once its scan has been saved."""
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)


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
