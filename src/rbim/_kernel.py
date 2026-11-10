"""Compiled kernels for one Swendsen-Wang sweep.

The sweep is split into two compiled steps so that the random numbers can stay
with ``numpy.random.Generator`` in the caller.  That keeps the random stream
identical to the original, array-based implementation: one call for the bond
uniforms, one call for the cluster-flip uniforms, in that order.  A run with a
given seed therefore reproduces the earlier code exactly, which is how this
module is verified (``tests/test_kernel_equivalence.py``).

The lattice geometry is computed inline from the conventions of
:mod:`rbim.lattice`: site ``n = x * L + y``; bond ``2 n`` joins ``n`` to its
neighbour in ``+x`` (site ``n + L``, wrapped) and bond ``2 n + 1`` joins ``n``
to its neighbour in ``+y`` (site ``n + 1``, wrapped within the row).

``numba`` compiles the kernels when it is installed.  Without it the same code
runs as plain Python, far slower but with identical results.
"""

from __future__ import annotations

import numpy as np

from rbim._unionfind import njit

__all__ = ["activate_label_wrap", "flip_and_measure"]


@njit(cache=True)
def activate_label_wrap(spins, J, uniforms, p_add, L):
    """Activate bonds, label the clusters and detect wrapping.

    A bond is active when it is satisfied (``J s_a s_b > 0``) and its uniform
    is below ``p_add``.  Clusters are found by a flood fill that gives every
    site its position relative to the cluster's seed in *unwrapped*
    coordinates.  When an active bond joins two sites already in the cluster,
    the two positions must differ by exactly that bond's step in both
    coordinates; a mismatch in ``x`` (``y``) means the bond closes a loop that
    winds around the torus along ``x`` (``y``).  Every active bond is
    examined, so every fundamental cycle is tested and the result is exact.

    Returns ``(n_clusters, labels, wraps_x, wraps_y, n_active)``.  Seeds are
    taken in increasing site order, so labels are numbered by first appearance
    when scanning sites: the numbering is a function of the partition alone.
    """
    n_sites = L * L
    active = np.empty(2 * n_sites, dtype=np.bool_)
    n_active = 0
    for a in range(n_sites):
        b = a + L
        if b >= n_sites:
            b -= n_sites
        on = (uniforms[2 * a] < p_add) & (J[2 * a] * spins[a] * spins[b] > 0)
        active[2 * a] = on
        n_active += on
        b = a + 1
        if (b % L) == 0:
            b -= L
        on = (uniforms[2 * a + 1] < p_add) & (J[2 * a + 1] * spins[a] * spins[b] > 0)
        active[2 * a + 1] = on
        n_active += on

    labels = np.full(n_sites, -1, dtype=np.int64)
    px = np.zeros(n_sites, dtype=np.int32)
    py = np.zeros(n_sites, dtype=np.int32)
    stack = np.empty(n_sites, dtype=np.int32)
    wrap_x = False
    wrap_y = False
    n_clusters = 0

    for seed in range(n_sites):
        if labels[seed] != -1:
            continue
        labels[seed] = n_clusters
        px[seed] = 0
        py[seed] = 0
        stack[0] = seed
        top = 1
        while top > 0:
            top -= 1
            a = stack[top]
            ax = px[a]
            ay = py[a]
            y = a % L

            # +x neighbour through bond 2a
            if active[2 * a]:
                b = a + L
                if b >= n_sites:
                    b -= n_sites
                if labels[b] == -1:
                    labels[b] = n_clusters
                    px[b] = ax + 1
                    py[b] = ay
                    stack[top] = b
                    top += 1
                else:
                    if px[b] != ax + 1:
                        wrap_x = True
                    if py[b] != ay:
                        wrap_y = True
            # -x neighbour through that neighbour's bond
            b = a - L
            if b < 0:
                b += n_sites
            if active[2 * b]:
                if labels[b] == -1:
                    labels[b] = n_clusters
                    px[b] = ax - 1
                    py[b] = ay
                    stack[top] = b
                    top += 1
                else:
                    if px[b] != ax - 1:
                        wrap_x = True
                    if py[b] != ay:
                        wrap_y = True
            # +y neighbour through bond 2a+1
            if active[2 * a + 1]:
                if y == L - 1:
                    b = a - (L - 1)
                else:
                    b = a + 1
                if labels[b] == -1:
                    labels[b] = n_clusters
                    px[b] = ax
                    py[b] = ay + 1
                    stack[top] = b
                    top += 1
                else:
                    if px[b] != ax:
                        wrap_x = True
                    if py[b] != ay + 1:
                        wrap_y = True
            # -y neighbour through that neighbour's bond
            if y == 0:
                b = a + (L - 1)
            else:
                b = a - 1
            if active[2 * b + 1]:
                if labels[b] == -1:
                    labels[b] = n_clusters
                    px[b] = ax
                    py[b] = ay - 1
                    stack[top] = b
                    top += 1
                else:
                    if px[b] != ax:
                        wrap_x = True
                    if py[b] != ay - 1:
                        wrap_y = True
        n_clusters += 1

    return n_clusters, labels, wrap_x, wrap_y, n_active


@njit(cache=True)
def flip_and_measure(spins, labels, flip_uniforms, J, L, n_clusters, measure):
    """Flip clusters whose uniform is below one half, then measure.

    ``spins`` is modified in place.  When ``measure`` is false only the flip is
    done and empty arrays are returned.

    Returns ``(sizes, magnetizations, energy, magnetization, sum_sq)`` where
    ``sizes`` and ``magnetizations`` are per cluster, ``energy`` is
    ``-sum J s s`` over bonds, and ``sum_sq`` is the sum over clusters of the
    squared cluster magnetization.
    """
    n_sites = L * L
    for i in range(n_sites):
        if flip_uniforms[labels[i]] < 0.5:
            spins[i] = -spins[i]

    if not measure:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            0.0,
            0.0,
            0.0,
        )

    sizes = np.zeros(n_clusters, dtype=np.int64)
    mags = np.zeros(n_clusters, dtype=np.float64)
    total = 0
    for i in range(n_sites):
        c = labels[i]
        sizes[c] += 1
        mags[c] += spins[i]
        total += spins[i]

    bond_sum = 0
    for a in range(n_sites):
        b = a + L
        if b >= n_sites:
            b -= n_sites
        bond_sum += J[2 * a] * spins[a] * spins[b]
        if (a % L) == L - 1:
            b = a - (L - 1)
        else:
            b = a + 1
        bond_sum += J[2 * a + 1] * spins[a] * spins[b]

    sum_sq = 0.0
    for c in range(n_clusters):
        sum_sq += mags[c] * mags[c]

    return sizes, mags, -float(bond_sum), float(total), sum_sq
