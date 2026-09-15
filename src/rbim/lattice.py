"""Square-lattice geometry for the two-dimensional random-bond Ising model.

Conventions
-----------
Sites of an ``L x L`` lattice are indexed by a single integer

    n = x * L + y,      0 <= x, y < L,

so ``x`` is the slow index and ``y`` the fast one.  Each site owns exactly two
bonds, which fixes the length of every bond array to ``2 * N`` with
``N = L * L``:

    bond ``2 * n``      horizontal, connects ``(x, y) -> (x + 1, y)``
    bond ``2 * n + 1``  vertical,   connects ``(x, y) -> (x, y + 1)``

Both lattice directions are periodic.  The bonds that close the periodic
boundary are recorded explicitly in ``Lattice.wraps_x`` / ``Lattice.wraps_y``
rather than being re-derived from site indices, which keeps the spanning and
wrapping observables unambiguous.

This ordering matches the ``J_h`` / ``J_v`` disorder files used by the earlier
simulations, so existing realizations remain readable: ``J_h[x, y]`` is the
coupling on bond ``2 * n`` and ``J_v[x, y]`` the coupling on bond ``2 * n + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Lattice", "build_lattice", "site_index", "couplings_from_disorder"]


@dataclass(frozen=True)
class Lattice:
    """Bond structure of a periodic ``L x L`` square lattice.

    Attributes
    ----------
    L:
        Linear system size.
    bonds:
        Integer array of shape ``(2 * N, 2)`` holding the two endpoint site
        indices of every bond.
    wraps_x:
        Boolean mask of shape ``(2 * N,)``, true for the horizontal bonds that
        close the periodic boundary in the ``x`` direction.
    wraps_y:
        Boolean mask of shape ``(2 * N,)``, true for the vertical bonds that
        close the periodic boundary in the ``y`` direction.
    """

    L: int
    bonds: np.ndarray
    wraps_x: np.ndarray
    wraps_y: np.ndarray

    @property
    def n_sites(self) -> int:
        return self.L * self.L

    @property
    def n_bonds(self) -> int:
        return 2 * self.n_sites

    def wrap_mask(self, axis: str) -> np.ndarray:
        """Return the wrap mask for ``axis``, which must be ``'x'`` or ``'y'``."""
        if axis == "x":
            return self.wraps_x
        if axis == "y":
            return self.wraps_y
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def site_index(x: int | np.ndarray, y: int | np.ndarray, L: int):
    """Map lattice coordinates to the flat site index ``n = x * L + y``."""
    return x * L + y


def build_lattice(L: int) -> Lattice:
    """Construct the periodic square lattice of linear size ``L``.

    Parameters
    ----------
    L:
        Linear system size.  Must be at least 3; for ``L <= 2`` a site is its
        own periodic neighbour in one or both directions and the bond list
        would contain duplicates.
    """
    if L < 3:
        raise ValueError(
            f"L must be at least 3 (got {L}); smaller lattices produce duplicate "
            "bonds under periodic boundary conditions."
        )

    x = np.repeat(np.arange(L), L)
    y = np.tile(np.arange(L), L)
    n = x * L + y
    right = ((x + 1) % L) * L + y
    up = x * L + ((y + 1) % L)

    n_bonds = 2 * L * L
    bonds = np.empty((n_bonds, 2), dtype=np.int64)
    bonds[0::2, 0] = n
    bonds[0::2, 1] = right
    bonds[1::2, 0] = n
    bonds[1::2, 1] = up

    wraps_x = np.zeros(n_bonds, dtype=bool)
    wraps_y = np.zeros(n_bonds, dtype=bool)
    wraps_x[0::2] = x == L - 1
    wraps_y[1::2] = y == L - 1

    return Lattice(L=L, bonds=bonds, wraps_x=wraps_x, wraps_y=wraps_y)


def couplings_from_disorder(J_h: np.ndarray, J_v: np.ndarray) -> np.ndarray:
    """Interleave horizontal and vertical coupling arrays into bond order.

    Parameters
    ----------
    J_h, J_v:
        Arrays of shape ``(L, L)`` holding the horizontal and vertical
        couplings, indexed as ``J[x, y]``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2 * N,)`` ordered to match :func:`build_lattice`.
    """
    if J_h.shape != J_v.shape:
        raise ValueError(f"J_h and J_v must have the same shape, got {J_h.shape} and {J_v.shape}")
    if J_h.ndim != 2 or J_h.shape[0] != J_h.shape[1]:
        raise ValueError(f"couplings must be square 2D arrays, got shape {J_h.shape}")

    L = J_h.shape[0]
    J = np.empty(2 * L * L, dtype=np.int8)
    J[0::2] = J_h.reshape(-1)
    J[1::2] = J_v.reshape(-1)
    return J
