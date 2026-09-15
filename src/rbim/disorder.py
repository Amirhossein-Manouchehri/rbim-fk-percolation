"""Generation, storage and frustration analysis of +-J bond disorder.

A disorder realization is a pair of ``(L, L)`` integer arrays ``J_h`` and
``J_v`` holding the horizontal and vertical couplings, each equal to ``-1``
with probability ``p`` and ``+1`` otherwise.  Realizations are stored as
``.npz`` archives, which keeps them readable by the earlier simulation code.

A plaquette is *frustrated* when the product of its four couplings is
negative; no spin configuration can satisfy all four of its bonds.  The
frustrated plaquettes are a property of the disorder alone, independent of
temperature and of the spin configuration, and they are the objects that a
minimum-weight perfect matching pairs up when computing ground states.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "generate_disorder",
    "save_disorder",
    "load_disorder",
    "frustrated_plaquettes",
    "frustration_density",
    "default_disorder_filename",
]


def generate_disorder(
    L: int, p: float, rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw one realization of +-J bond disorder.

    Parameters
    ----------
    L:
        Linear system size.
    p:
        Probability that a given bond is antiferromagnetic (``J = -1``).
    rng:
        Random generator.  A fresh default generator is used when omitted,
        which makes the result irreproducible; pass an explicit generator for
        anything that needs to be repeatable.

    Returns
    -------
    (J_h, J_v):
        Horizontal and vertical couplings, each of shape ``(L, L)`` and dtype
        ``int8``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must lie in [0, 1], got {p}")
    if rng is None:
        rng = np.random.default_rng()

    J_h = np.where(rng.random((L, L)) < p, -1, 1).astype(np.int8)
    J_v = np.where(rng.random((L, L)) < p, -1, 1).astype(np.int8)
    return J_h, J_v


def default_disorder_filename(L: int, p: float, index: int = 0) -> str:
    """Return the canonical file name for a stored realization."""
    return f"disorder_L={L}_p={p:.3f}_r={index:04d}.npz"


def save_disorder(
    path: str,
    J_h: np.ndarray,
    J_v: np.ndarray,
    *,
    p: Optional[float] = None,
    seed: Optional[int] = None,
) -> None:
    """Write a realization to ``path``, creating parent directories as needed.

    The nominal disorder strength and the seed are stored alongside the
    couplings so that a realization can always be traced back to how it was
    produced.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    metadata = {}
    if p is not None:
        metadata["p"] = np.asarray(p)
    if seed is not None:
        metadata["seed"] = np.asarray(seed)

    np.savez_compressed(path, J_h=J_h, J_v=J_v, **metadata)


def load_disorder(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a realization written by :func:`save_disorder`.

    Files produced by the earlier simulation code are also accepted, since they
    use the same ``J_h`` / ``J_v`` keys.
    """
    with np.load(path) as data:
        if "J_h" not in data or "J_v" not in data:
            raise KeyError(
                f"{path} does not contain 'J_h' and 'J_v' arrays; found {list(data.keys())}"
            )
        J_h = data["J_h"].astype(np.int8)
        J_v = data["J_v"].astype(np.int8)
    return J_h, J_v


def frustrated_plaquettes(J_h: np.ndarray, J_v: np.ndarray) -> np.ndarray:
    """Locate the frustrated plaquettes of a disorder realization.

    The plaquette labelled ``(x, y)`` has its lower-left corner at site
    ``(x, y)`` and is bounded by

        bottom  J_h[x, y]                left   J_v[x, y]
        top     J_h[x, (y + 1) % L]      right  J_v[(x + 1) % L, y]

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(L, L)``, true where the product of the four
        couplings is negative.
    """
    L = J_h.shape[0]
    bottom = J_h.astype(np.int64)
    top = np.roll(J_h, -1, axis=1).astype(np.int64)
    left = J_v.astype(np.int64)
    right = np.roll(J_v, -1, axis=0).astype(np.int64)
    return (bottom * top * left * right) < 0


def frustration_density(J_h: np.ndarray, J_v: np.ndarray) -> float:
    """Fraction of plaquettes that are frustrated.

    For uncorrelated +-J disorder the expectation is
    ``4 p (1 - p) [(1 - p)^2 + p^2]``, which peaks at ``1/2`` for ``p = 1/2``.
    """
    return float(frustrated_plaquettes(J_h, J_v).mean())
