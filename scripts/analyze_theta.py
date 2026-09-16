#!/usr/bin/env python3
"""Extract T_1, the equilibration floor, and the exponent theta from a production scan.

The exponent is defined by ``<|M_C|> ~ <S_max>^theta``, where ``S_max`` is the
size of the largest Fortuin-Kasteleyn cluster and ``M_C`` the net magnetization
it carries.  It measures how much of the percolating cluster's internal order
survives projection onto the uniform direction: ``theta = 1`` when the cluster
is internally aligned, ``theta = 1/2`` when its internal sign pattern is
effectively a random walk and the cluster carries no coherent moment.

Three points of care, each of which changes the answer:

*Equilibration.*  In equilibrium ``<|m|>`` must fall monotonically as the
temperature rises.  Where it does not, the run has failed to equilibrate, and
the lowest temperature above that turnover is reported as the usable floor.

*Errors on both axes.*  Both ``S_max`` and ``|M_C|`` carry uncertainty, so the
fit uses the effective-variance method rather than weighting on ``y`` alone,
and reports the residual-inflated error alongside the propagated one.  A ratio
between them much larger than one means the data has curvature that a single
power law does not capture.

*Whether a single exponent exists at all.*  The slope between the two largest
sizes is reported next to the four-size fit.  If they disagree, small-size
corrections dominate and no single ``theta`` is meaningful at that temperature,
however small the error bar looks.

Sizes are statistically independent here because each ``(L, p)`` run draws its
own disorder, so ordinary propagation across sizes is valid; temperatures
within one size share realizations and are correlated, so the ``theta(T)``
curve should not be fitted point-by-point as if independent.

Example
-------
    python scripts/analyze_theta.py --dir results/prod --csv results/theta.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rbim.simulation import load_results


def crossings(temperatures: np.ndarray, a: Sequence[float], b: Sequence[float]) -> List[float]:
    """Temperatures where two curves intersect, by linear interpolation."""
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    found = []
    for i in range(len(difference) - 1):
        d0, d1 = difference[i], difference[i + 1]
        if d0 * d1 < 0:
            t0, t1 = temperatures[i], temperatures[i + 1]
            found.append(float(t0 - d0 * (t1 - t0) / (d1 - d0)))
    return found


def equilibration_floor(temperatures: np.ndarray, abs_m: Sequence[float]) -> float:
    """Lowest temperature whose data is trustworthy.

    With temperatures in ascending order, equilibrium requires ``<|m|>`` to be
    non-increasing.  Any point where it rises with temperature signals that the
    colder run is further from equilibrium than the warmer one; everything at
    or below the highest such point is discarded.
    """
    m = np.asarray(abs_m, dtype=float)
    rising = [i for i in range(len(m) - 1) if m[i] < m[i + 1]]
    return float(temperatures[max(rising) + 1]) if rising else float(temperatures[0])


def power_law_slope(
    S: np.ndarray, dS: np.ndarray, M: np.ndarray, dM: np.ndarray
) -> Tuple[float, float, float, float]:
    """Fit ``log M = theta log S + c`` with uncertainty on both axes.

    Returns
    -------
    (theta, error, inflated_error, chi2_per_dof)
        ``error`` propagates the measured uncertainties; ``inflated_error``
        additionally scales by the goodness of fit, and is the honest one to
        quote when the reduced chi-squared exceeds one.
    """
    x, y = np.log(S), np.log(M)
    sx, sy = dS / S, dM / M
    slope = float(np.polyfit(x, y, 1)[0])

    weights = np.ones_like(x)
    intercept = 0.0
    determinant = 1.0
    for _ in range(50):
        weights = 1.0 / np.maximum(sy**2 + (slope * sx) ** 2, 1e-30)
        sum_w = weights.sum()
        sum_x = (weights * x).sum()
        sum_y = (weights * y).sum()
        sum_xx = (weights * x * x).sum()
        sum_xy = (weights * x * y).sum()
        determinant = sum_w * sum_xx - sum_x**2
        if determinant == 0:
            break
        updated = (sum_w * sum_xy - sum_x * sum_y) / determinant
        intercept = (sum_xx * sum_y - sum_x * sum_xy) / determinant
        if abs(updated - slope) < 1e-13:
            slope = updated
            break
        slope = updated

    error = float(np.sqrt(weights.sum() / determinant)) if determinant else float("nan")
    residuals = y - (slope * x + intercept)
    dof = max(len(x) - 2, 1)
    chi2 = float((weights * residuals**2).sum() / dof)
    return float(slope), error, error * float(np.sqrt(max(chi2, 1.0))), chi2


def pair_slope(S: np.ndarray, M: np.ndarray, i: int, j: int) -> float:
    """Local exponent between two sizes, free of small-size contamination."""
    return float((np.log(M[j]) - np.log(M[i])) / (np.log(S[j]) - np.log(S[i])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="results/prod")
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 24, 32, 48, 64])
    parser.add_argument("--p", type=float, nargs="+",
                        default=[0.02, 0.05, 0.08, 0.10, 0.15, 0.20])
    parser.add_argument("--csv", default=None, help="write the full theta(T) table here")
    args = parser.parse_args()

    rows: List[str] = ["p,L_list,T,theta,error,inflated_error,chi2,theta_top_pair,"
                       "p_infinity,abs_m,wrap_y,usable"]

    print(f"{'p':>6} {'sizes':>18} {'T1':>8} {'floor':>7} {'theta(T1)':>18} "
          f"{'theta_min':>18} {'at T':>7} {'theta(cold)':>12}")
    print("-" * 110)

    for p in args.p:
        loaded: Dict[int, dict] = {}
        for L in args.sizes:
            path = os.path.join(args.dir, f"L{L}_p{p:.2f}.npz")
            if os.path.exists(path):
                loaded[L] = load_results(path)
        if len(loaded) < 3:
            print(f"{p:6.2f}   only {len(loaded)} size(s) present, skipping")
            continue

        sizes = sorted(loaded)
        temperatures = loaded[sizes[-1]]["temperatures"]

        # T_1 from the crossing between the two largest sizes, which carries the
        # smallest finite-size correction.
        top_a, top_b = sizes[-2], sizes[-1]
        found = crossings(temperatures,
                          loaded[top_a]["observables"]["wrap_y"]["value"],
                          loaded[top_b]["observables"]["wrap_y"]["value"])
        t1 = max(found) if found else float("nan")

        floor = max(
            equilibration_floor(temperatures, loaded[L]["observables"]["abs_magnetization"]["value"])
            for L in sizes
        )

        thetas: List[Optional[float]] = []
        for i, T in enumerate(temperatures):
            S = np.array([loaded[L]["observables"]["largest_size"]["value"][i] for L in sizes])
            dS = np.array([loaded[L]["observables"]["largest_size"]["error"][i] for L in sizes])
            M = np.array([loaded[L]["observables"]["largest_abs_magnetization"]["value"][i] for L in sizes])
            dM = np.array([loaded[L]["observables"]["largest_abs_magnetization"]["error"][i] for L in sizes])

            if np.any(S <= 0) or np.any(M <= 0):
                thetas.append(None)
                continue

            theta, err, inflated, chi2 = power_law_slope(S, dS, M, dM)
            top = pair_slope(S, M, len(sizes) - 2, len(sizes) - 1)
            thetas.append(theta)

            big = loaded[sizes[-1]]["observables"]
            rows.append(
                f"{p},{'|'.join(map(str, sizes))},{T:.5f},{theta:.5f},{err:.5f},{inflated:.5f},"
                f"{chi2:.3f},{top:.5f},{big['p_infinity']['value'][i]:.5f},"
                f"{big['abs_magnetization']['value'][i]:.5f},{big['wrap_y']['value'][i]:.5f},"
                f"{int(T >= floor)}"
            )

        usable = [(i, t) for i, t in enumerate(thetas)
                  if t is not None and temperatures[i] >= floor]
        if not usable:
            print(f"{p:6.2f}   no usable temperatures above the equilibration floor")
            continue

        def theta_at(target: float) -> str:
            i = min(usable, key=lambda it: abs(temperatures[it[0]] - target))[0]
            S = np.array([loaded[L]["observables"]["largest_size"]["value"][i] for L in sizes])
            dS = np.array([loaded[L]["observables"]["largest_size"]["error"][i] for L in sizes])
            M = np.array([loaded[L]["observables"]["largest_abs_magnetization"]["value"][i] for L in sizes])
            dM = np.array([loaded[L]["observables"]["largest_abs_magnetization"]["error"][i] for L in sizes])
            th, _, inf, _ = power_law_slope(S, dS, M, dM)
            return f"{th:.4f}+-{inf:.4f}"

        i_min = min(usable, key=lambda it: it[1])[0]
        coldest = min(usable, key=lambda it: temperatures[it[0]])[0]

        print(f"{p:6.2f} {'|'.join(map(str, sizes)):>18} {t1:8.4f} {floor:7.3f} "
              f"{theta_at(t1):>18} {theta_at(temperatures[i_min]):>18} "
              f"{temperatures[i_min]:7.3f} {thetas[coldest]:12.4f}")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w") as handle:
            handle.write("\n".join(rows) + "\n")
        print(f"\nfull table written to {args.csv}")
        print("columns: theta with propagated and residual-inflated errors, chi2/dof, the")
        print("slope between the two largest sizes alone, and a usable flag from the")
        print("equilibration floor.  Quote the inflated error; if theta and theta_top_pair")
        print("disagree, a single exponent is not well defined at that temperature.")


if __name__ == "__main__":
    main()
