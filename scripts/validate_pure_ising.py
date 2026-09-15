#!/usr/bin/env python3
"""Recover the exact critical temperature of the pure 2D Ising model.

At ``p = 0`` the model reduces to the ferromagnetic square-lattice Ising model
with the exactly known critical point

    T_c = 2 / ln(1 + sqrt(2)) = 2.269185...

Two independent crossings are extracted and both must land on that value:

* the **Binder cumulant** ``U = 1 - <m^4> / (3 <m^2>^2)``, a purely magnetic
  quantity, and
* the **FK wrapping probability**, a purely geometric quantity.

Their coincidence at ``p = 0`` is the Coniglio-Klein identity.  It is exactly
this coincidence that fails for ``p > 0``, which is the phenomenon the rest of
the project is built to measure, so reproducing it here checks the magnetic and
the percolation machinery against each other as well as against Onsager.

Example
-------
    python scripts/validate_pure_ising.py --sizes 8 16 24 --sweeps 8000
"""

from __future__ import annotations

import argparse
import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

from rbim.simulation import run_temperature_scan

TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))


def crossing(
    temperatures: np.ndarray, curve_a: np.ndarray, curve_b: np.ndarray
) -> float | None:
    """Temperature where two curves intersect, by linear interpolation.

    Returns ``None`` when the difference does not change sign on the grid.
    """
    difference = np.asarray(curve_a) - np.asarray(curve_b)
    signs = np.sign(difference)
    for i in range(len(difference) - 1):
        if signs[i] == 0:
            return float(temperatures[i])
        if signs[i] * signs[i + 1] < 0:
            t0, t1 = temperatures[i], temperatures[i + 1]
            d0, d1 = difference[i], difference[i + 1]
            return float(t0 - d0 * (t1 - t0) / (d1 - d0))
    return None


def pairwise_crossings(
    temperatures: np.ndarray, curves: Dict[int, np.ndarray]
) -> List[Tuple[int, int, float]]:
    """All pairwise crossings between the curves, keyed by system size."""
    found = []
    for La, Lb in itertools.combinations(sorted(curves), 2):
        location = crossing(temperatures, curves[La], curves[Lb])
        if location is not None:
            found.append((La, Lb, location))
    return found


def report(name: str, temperatures: np.ndarray, curves: Dict[int, np.ndarray]) -> float | None:
    crossings = pairwise_crossings(temperatures, curves)
    print(f"\n{name} crossings")
    if not crossings:
        print("  none found on this temperature grid")
        return None
    for La, Lb, location in crossings:
        print(f"  L={La:3d} vs L={Lb:3d}:  T = {location:.4f}   ({location - TC_EXACT:+.4f} from exact)")
    estimate = float(np.mean([c[2] for c in crossings]))
    print(f"  mean: T = {estimate:.4f}   ({estimate - TC_EXACT:+.4f} from exact, "
          f"{100 * abs(estimate - TC_EXACT) / TC_EXACT:.2f}%)")
    return estimate


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24], help="system sizes")
    parser.add_argument("--t-min", type=float, default=2.15)
    parser.add_argument("--t-max", type=float, default=2.40)
    parser.add_argument("--n-temps", type=int, default=11)
    parser.add_argument("--sweeps", type=int, default=8000)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    temperatures = np.linspace(args.t_min, args.t_max, args.n_temps)
    print(f"exact T_c = {TC_EXACT:.6f}")
    print(f"sizes {args.sizes}, {args.n_temps} temperatures in [{args.t_min}, {args.t_max}], "
          f"{args.sweeps} sweeps")

    binder: Dict[int, np.ndarray] = {}
    wrapping: Dict[int, np.ndarray] = {}

    for L in args.sizes:
        # p = 0 means every bond is ferromagnetic, so one "realization" is the
        # whole ensemble and disorder averaging is unnecessary.
        results = run_temperature_scan(
            L,
            0.0,
            temperatures,
            n_realizations=1,
            n_sweeps=args.sweeps,
            n_bins=args.bins,
            seed=args.seed + L,
            n_workers=args.workers,
        )
        binder[L] = results["observables"]["binder"]["value"]
        wrapping[L] = results["observables"]["wrap_y"]["value"]
        print(f"  L={L:3d} done")

    binder_tc = report("Binder cumulant", temperatures, binder)
    wrapping_tc = report("FK wrapping probability", temperatures, wrapping)

    if binder_tc is not None and wrapping_tc is not None:
        print(
            f"\nConiglio-Klein check: magnetic and geometric estimates differ by "
            f"{abs(binder_tc - wrapping_tc):.4f}"
        )


if __name__ == "__main__":
    main()
