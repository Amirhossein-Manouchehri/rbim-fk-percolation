"""A single Swendsen-Wang update and the observables measured alongside it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from rbim.clusters import activate_bonds, cluster_statistics, label_clusters, wraps_along
from rbim.lattice import Lattice

__all__ = ["SweepResult", "energy", "magnetization", "random_spins", "sw_sweep"]


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


def random_spins(n_sites: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random ``+-1`` spin configuration of length ``n_sites``."""
    return (rng.integers(0, 2, size=n_sites, dtype=np.int64) * 2 - 1)


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
        When false the wrapping tests are skipped.  They require two extra
        cluster labellings and dominate the cost of a measured sweep.

    Returns
    -------
    SweepResult or None
        ``None`` when ``measure`` is false.
    """
    active = activate_bonds(spins, lattice, J, beta, rng)
    n_clusters, labels = label_clusters(active, lattice)

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
        wx: Optional[bool] = wraps_along(active, lattice, "x")
        wy: Optional[bool] = wraps_along(active, lattice, "y")
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
