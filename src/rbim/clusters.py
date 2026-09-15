"""Fortuin-Kasteleyn cluster construction and percolation observables.

For the +-J model the Fortuin-Kasteleyn-Coniglio-Klein construction activates
only *satisfied* bonds, those with ``J_ij s_i s_j > 0``, each independently
with probability ``1 - exp(-2 beta |J|)``.  Unsatisfied bonds are never
activated.

A consequence worth keeping in mind when interpreting the observables below:
around any closed loop the product of ``s_i s_j`` is ``+1``, because every spin
appears twice.  If every bond of the loop is satisfied then the product of the
couplings around it must also be ``+1``, so the loop is unfrustrated.  An FK
cluster therefore never contains a frustrated loop: it is gauge-equivalent to a
pure ferromagnet and its spins are locked rigidly relative to one another up to
a global flip.  The physical magnetization it carries is the projection of that
internal pattern onto the uniform direction, which is what
:func:`cluster_statistics` measures.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from rbim.lattice import Lattice

__all__ = [
    "bond_activation_probability",
    "activate_bonds",
    "label_clusters",
    "label_and_wrap",
    "cluster_statistics",
    "wraps_along",
    "spans_along",
]


def bond_activation_probability(beta: float) -> float:
    """Return ``1 - exp(-2 beta)``, the FK activation probability for ``|J| = 1``.

    ``numpy.expm1`` is used so that the small-``beta`` limit stays accurate.
    """
    return float(-np.expm1(-2.0 * beta))


def activate_bonds(
    spins: np.ndarray,
    lattice: Lattice,
    J: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the set of active FK bonds for the current spin configuration.

    Parameters
    ----------
    spins:
        Spin configuration of shape ``(N,)`` with entries ``+-1``.
    lattice:
        Lattice geometry.
    J:
        Bond couplings of shape ``(2 * N,)`` in the ordering of
        :func:`rbim.lattice.build_lattice`.
    beta:
        Inverse temperature.
    rng:
        Random generator; passed in explicitly so that runs are reproducible.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(2 * N,)``, true for active bonds.
    """
    ends = lattice.bonds
    satisfied = (J * spins[ends[:, 0]] * spins[ends[:, 1]]) > 0
    p_add = bond_activation_probability(beta)
    return satisfied & (rng.random(lattice.n_bonds) < p_add)


def label_clusters(active: np.ndarray, lattice: Lattice) -> Tuple[int, np.ndarray]:
    """Label the connected components of the active-bond graph.

    Returns
    -------
    (n_clusters, labels):
        Number of clusters and an array of shape ``(N,)`` giving the cluster
        index of every site.  Isolated sites count as clusters of size one.
    """
    ends = lattice.bonds[active]
    n_sites = lattice.n_sites
    data = np.ones(ends.shape[0], dtype=np.int8)
    graph = csr_matrix((data, (ends[:, 0], ends[:, 1])), shape=(n_sites, n_sites))
    return connected_components(graph, directed=False)


def cluster_statistics(
    labels: np.ndarray, spins: np.ndarray, n_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the size and the net magnetization of every cluster.

    The magnetization of a cluster is the signed sum of its spins.  Its
    magnitude is invariant under the Swendsen-Wang flip, since flipping a whole
    cluster only changes the overall sign, so it may be measured either before
    or after the update.

    Returns
    -------
    (sizes, magnetizations):
        Integer sizes and floating-point signed magnetizations, both of shape
        ``(n_clusters,)``.
    """
    sizes = np.bincount(labels, minlength=n_clusters)
    magnetizations = np.bincount(
        labels, weights=spins.astype(np.float64), minlength=n_clusters
    )
    return sizes, magnetizations


def wraps_along(active: np.ndarray, lattice: Lattice, axis: str) -> bool:
    """Test whether some cluster winds around the torus along ``axis``.

    Delegates to :func:`label_and_wrap`, which examines every fundamental cycle
    of the active-bond graph and is exact.  An earlier version of this function
    used the cheaper test of deleting the boundary-closing bonds and asking
    whether any deleted bond rejoined a cluster to itself; that only detects
    cycles crossing the boundary once and under-reports wrapping for large,
    multiply connected clusters.  It is kept out of the codebase deliberately.

    Note that wrapping along ``x`` and wrapping along ``y`` are distinct
    observables.  They share a transition point but their universal values at
    criticality differ, so the choice must be stated when comparing against
    published crossing values.
    """
    _, _, wrapped_x, wrapped_y = label_and_wrap(active, lattice)
    if axis == "x":
        return bool(wrapped_x)
    if axis == "y":
        return bool(wrapped_y)
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def label_and_wrap(active: np.ndarray, lattice: Lattice) -> Tuple[int, np.ndarray, bool, bool]:
    """Label the clusters and test wrapping along both axes in a single pass.

    Traverses the active bonds once, using a union-find that carries the
    displacement of every site to the root of its tree, so that the winding of
    each fundamental cycle falls out while the clusters are being built.  This
    replaced three separate connected-component searches, which used to account
    for about three quarters of the cost of a measured sweep.

    Returns
    -------
    (n_clusters, labels, wraps_x, wraps_y)
    """
    from rbim._unionfind import label_and_wrap_unionfind

    return label_and_wrap_unionfind(active, lattice)


def spans_along(active: np.ndarray, lattice: Lattice, axis: str) -> bool:
    """Test whether some cluster spans the lattice along ``axis``.

    The boundary-closing bonds along ``axis`` are removed, making that
    direction open while the transverse direction stays periodic, and a cluster
    spans when it contains a site on each of the two open faces.  This is a
    different observable from :func:`wraps_along` and the two should not be
    mixed within one analysis.
    """
    wrap_mask = lattice.wrap_mask(axis)
    _, labels = label_clusters(active & ~wrap_mask, lattice)

    L = lattice.L
    sites = np.arange(lattice.n_sites)
    coordinate = sites // L if axis == "x" else sites % L

    low = np.unique(labels[coordinate == 0])
    high = np.unique(labels[coordinate == L - 1])
    return bool(np.intersect1d(low, high, assume_unique=True).size > 0)
