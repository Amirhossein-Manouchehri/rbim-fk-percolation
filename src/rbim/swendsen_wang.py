"""A single Swendsen-Wang update and the observables measured alongside it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from rbim._kernel import activate_label_wrap, flip_and_measure
from rbim.clusters import (
    activate_bonds,
    bond_activation_probability,
    cluster_statistics,
    label_and_wrap,
)
from rbim.lattice import Lattice

__all__ = [
    "SweepResult",
    "energy",
    "magnetization",
    "random_spins",
    "aligned_spins",
    "initial_spins",
    "sw_sweep",
    "sw_sweep_reference",
]


@dataclass(frozen=True)
class SweepResult:
    """Observables of a single Swendsen-Wang sweep.

    Attributes
    ----------
    energy, magnetization:
        Extensive energy and signed magnetization after the update.
    n_clusters:
        Number of Fortuin-Kasteleyn clusters, counting isolated sites.
    largest_size, largest_abs_magnetization:
        Size of the largest cluster and the magnitude of the net magnetization
        it carries.  Their ratio measures how much of the cluster's internal
        order survives projection onto the uniform direction.
    second_size, second_abs_magnetization:
        The same quantities for the second-largest cluster.
    wraps_x, wraps_y:
        Whether a cluster winds around the torus along each axis.  ``None``
        when percolation measurement was switched off.
    cluster_magnetization_sq_sum:
        Sum over all clusters of the squared cluster magnetization.  Because
        clusters flip independently, its average equals ``<M^2>`` exactly, so
        the ratio ``largest_abs_magnetization**2 / cluster_magnetization_sq_sum``
        is the share of ``<M^2>`` carried by the largest cluster.
    n_active_bonds:
        Number of active Fortuin-Kasteleyn bonds in the configuration the
        clusters were built from.
    """

    energy: float
    magnetization: float
    n_clusters: int
    largest_size: int
    largest_abs_magnetization: float
    second_size: int
    second_abs_magnetization: float
    wraps_x: Optional[bool]
    wraps_y: Optional[bool]
    cluster_magnetization_sq_sum: float = float("nan")
    n_active_bonds: int = -1


def random_spins(n_sites: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random ``+-1`` spin configuration of length ``n_sites``."""
    return (rng.integers(0, 2, size=n_sites, dtype=np.int64) * 2 - 1)


def aligned_spins(n_sites: int) -> np.ndarray:
    """Return the fully aligned configuration, for a cold start."""
    return np.ones(n_sites, dtype=np.int64)


def initial_spins(n_sites: int, rng: np.random.Generator, start: str = "hot") -> np.ndarray:
    """Return a starting configuration.

    A hot start is random, a cold start fully aligned.  Running both and
    comparing is the standard equilibration test: below the transition the two
    approach equilibrium from opposite sides, so results that disagree are not
    equilibrated no matter how small the statistical error looks.
    """
    if start == "hot":
        return random_spins(n_sites, rng)
    if start == "cold":
        return aligned_spins(n_sites)
    raise ValueError(f"start must be 'hot' or 'cold', got {start!r}")


def energy(spins: np.ndarray, lattice: Lattice, J: np.ndarray) -> float:
    """Total energy ``-sum_<ij> J_ij s_i s_j``; each bond contributes once."""
    ends = lattice.bonds
    return -float(np.sum(J * spins[ends[:, 0]] * spins[ends[:, 1]]))


def magnetization(spins: np.ndarray) -> float:
    """Total signed magnetization ``sum_i s_i``."""
    return float(np.sum(spins))


def _two_largest(sizes: np.ndarray) -> Tuple[int, Optional[int]]:
    """Indices of the largest and second-largest entries of ``sizes``."""
    if sizes.size == 1:
        return 0, None
    candidates = np.argpartition(sizes, -2)[-2:]
    first, second = candidates[0], candidates[1]
    if sizes[first] < sizes[second]:
        first, second = second, first
    return int(first), int(second)


def sw_sweep(
    spins: np.ndarray,
    lattice: Lattice,
    J: np.ndarray,
    beta: float,
    rng: np.random.Generator,
    *,
    measure: bool = True,
    measure_percolation: bool = True,
) -> Optional[SweepResult]:
    """Perform one Swendsen-Wang update of ``spins`` in place.

    Satisfied bonds are activated with probability ``1 - exp(-2 beta)``, the
    active-bond graph is decomposed into clusters, and each cluster is flipped
    independently with probability one half.

    The work is done by two compiled kernels (:mod:`rbim._kernel`).  The random
    numbers are drawn here, in the same order as in
    :func:`sw_sweep_reference`, so the two functions give identical results
    from identical generator states.

    Parameters
    ----------
    spins:
        Spin configuration, modified in place.
    lattice, J, beta, rng:
        Geometry, couplings, inverse temperature and random generator.
    measure:
        When false the update is performed without computing any observables,
        which is what thermalisation sweeps want.
    measure_percolation:
        When false the wrapping flags are not reported.

    Returns
    -------
    SweepResult or None
        ``None`` when ``measure`` is false.
    """
    L = lattice.L
    p_add = bond_activation_probability(beta)
    n_clusters, labels, wrapped_x, wrapped_y, n_active = activate_label_wrap(
        spins, J, rng.random(lattice.n_bonds), p_add, L
    )
    sizes, magnetizations, total_energy, total_magnetization, sum_sq = flip_and_measure(
        spins, labels, rng.random(n_clusters), J, L, n_clusters, measure
    )

    if not measure:
        return None

    first, second = _two_largest(sizes)

    if second is None:
        second_size = 0
        second_abs_magnetization = 0.0
    else:
        second_size = int(sizes[second])
        second_abs_magnetization = abs(float(magnetizations[second]))

    if measure_percolation:
        wx: Optional[bool] = bool(wrapped_x)
        wy: Optional[bool] = bool(wrapped_y)
    else:
        wx = wy = None

    return SweepResult(
        energy=float(total_energy),
        magnetization=float(total_magnetization),
        n_clusters=int(n_clusters),
        largest_size=int(sizes[first]),
        largest_abs_magnetization=abs(float(magnetizations[first])),
        second_size=second_size,
        second_abs_magnetization=second_abs_magnetization,
        wraps_x=wx,
        wraps_y=wy,
        cluster_magnetization_sq_sum=float(sum_sq),
        n_active_bonds=int(n_active),
    )


def sw_sweep_reference(
    spins: np.ndarray,
    lattice: Lattice,
    J: np.ndarray,
    beta: float,
    rng: np.random.Generator,
    *,
    measure: bool = True,
    measure_percolation: bool = True,
) -> Optional[SweepResult]:
    """Original array-based sweep, kept as the reference implementation.

    :func:`sw_sweep` must reproduce this function exactly for the same random
    generator state.  It does not report the two newer fields.

    Perform one Swendsen-Wang update of ``spins`` in place.

    Satisfied bonds are activated with probability ``1 - exp(-2 beta)``, the
    active-bond graph is decomposed into clusters, and each cluster is flipped
    independently with probability one half.

    Parameters
    ----------
    spins:
        Spin configuration, modified in place.
    lattice, J, beta, rng:
        Geometry, couplings, inverse temperature and random generator.
    measure:
        When false the update is performed without computing any observables,
        which is what thermalisation sweeps want.
    measure_percolation:
        When false the wrapping flags are not reported.  They now come free
        with the cluster labelling, so this only affects what is returned.

    Returns
    -------
    SweepResult or None
        ``None`` when ``measure`` is false.
    """
    active = activate_bonds(spins, lattice, J, beta, rng)
    n_clusters, labels, wrapped_x, wrapped_y = label_and_wrap(active, lattice)

    flip = rng.random(n_clusters) < 0.5
    spins[flip[labels]] *= -1

    if not measure:
        return None

    sizes, magnetizations = cluster_statistics(labels, spins, n_clusters)
    first, second = _two_largest(sizes)

    if second is None:
        second_size = 0
        second_abs_magnetization = 0.0
    else:
        second_size = int(sizes[second])
        second_abs_magnetization = abs(float(magnetizations[second]))

    if measure_percolation:
        wx: Optional[bool] = bool(wrapped_x)
        wy: Optional[bool] = bool(wrapped_y)
    else:
        wx = wy = None

    return SweepResult(
        energy=energy(spins, lattice, J),
        magnetization=magnetization(spins),
        n_clusters=int(n_clusters),
        largest_size=int(sizes[first]),
        largest_abs_magnetization=abs(float(magnetizations[first])),
        second_size=second_size,
        second_abs_magnetization=second_abs_magnetization,
        wraps_x=wx,
        wraps_y=wy,
    )
