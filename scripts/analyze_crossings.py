#!/usr/bin/env python3
"""Locate T_1 and T_2 from a pair of system sizes and suggest production ranges.

``T_1`` is read off the crossing of the Fortuin-Kasteleyn wrapping probability
between two sizes, ``T_2`` from the crossing of the Binder cumulant.  Two sizes
give only a first estimate — the crossings drift with size — but that is enough
to decide where a production scan should put its temperatures.

A missing Binder crossing is a result rather than a failure: above the critical
disorder concentration there is no finite-temperature ferromagnetic transition,
so the curves should stop crossing.

Example
-------
    python scripts/analyze_crossings.py --dir results/survey --small 16 --large 32
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence

import numpy as np

from rbim.simulation import load_results


def crossings(temperatures: np.ndarray, curve_a: Sequence[float], curve_b: Sequence[float]) -> List[float]:
    """All temperatures where two curves intersect, by linear interpolation."""
    difference = np.asarray(curve_a, dtype=float) - np.asarray(curve_b, dtype=float)
    found: List[float] = []
    for i in range(len(difference) - 1):
        d0, d1 = difference[i], difference[i + 1]
        if d0 == 0.0:
            found.append(float(temperatures[i]))
        elif d0 * d1 < 0:
            t0, t1 = temperatures[i], temperatures[i + 1]
            found.append(float(t0 - d0 * (t1 - t0) / (d1 - d0)))
    return found


def pick(found: List[float]) -> Optional[float]:
    """Choose the most plausible crossing when interpolation finds several."""
    if not found:
        return None
    return max(found)  # spurious crossings from noise sit at low T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="results/survey")
    parser.add_argument("--small", type=int, default=16)
    parser.add_argument("--large", type=int, default=32)
    parser.add_argument("--p", type=float, nargs="+",
                        default=[0.02, 0.05, 0.08, 0.10, 0.15, 0.20])
    parser.add_argument("--pad", type=float, default=0.20,
                        help="half-width of the suggested production window")
    args = parser.parse_args()

    print(f"{'p':>6} {'T1 (wrap)':>11} {'T2 (Binder)':>13} {'window':>8}   "
          f"{'P_inf':>8} {'|m|':>7}   suggested production range")
    print("-" * 100)

    for p in args.p:
        small_path = os.path.join(args.dir, f"L{args.small}_p{p:.2f}.npz")
        large_path = os.path.join(args.dir, f"L{args.large}_p{p:.2f}.npz")
        if not (os.path.exists(small_path) and os.path.exists(large_path)):
            print(f"{p:6.2f}   missing files for this p")
            continue

        small = load_results(small_path)
        large = load_results(large_path)
        temperatures = large["temperatures"]

        t1 = pick(crossings(temperatures,
                            small["observables"]["wrap_y"]["value"],
                            large["observables"]["wrap_y"]["value"]))
        t2 = pick(crossings(temperatures,
                            small["observables"]["binder"]["value"],
                            large["observables"]["binder"]["value"]))

        # Separation of geometry from magnetism, sampled just below T_1.
        p_inf = m_abs = float("nan")
        if t1 is not None:
            probe = int(np.argmin(np.abs(temperatures - (t1 - 0.15))))
            p_inf = large["observables"]["p_infinity"]["value"][probe]
            m_abs = large["observables"]["abs_magnetization"]["value"][probe]

        window = f"{t1 - t2:8.3f}" if (t1 is not None and t2 is not None) else "       -"
        t1_text = f"{t1:11.3f}" if t1 is not None else "          -"
        t2_text = f"{t2:13.3f}" if t2 is not None else "     none    "

        lo = min(x for x in (t1, t2) if x is not None) - args.pad if t1 or t2 else None
        hi = max(x for x in (t1, t2) if x is not None) + args.pad if t1 or t2 else None
        suggestion = f"--t-min {lo:.2f} --t-max {hi:.2f}" if lo is not None else "-"

        print(f"{p:6.2f} {t1_text} {t2_text} {window}   "
              f"{p_inf:8.3f} {m_abs:7.3f}   {suggestion}")

    print("\nP_inf and |m| are sampled 0.15 below T_1; a large gap between them is the")
    print("intermediate regime.  'none' under T_2 means the Binder curves stopped")
    print("crossing, i.e. no finite-temperature ferromagnetic transition at that p.")


if __name__ == "__main__":
    main()
