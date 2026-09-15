#!/usr/bin/env python3
"""Run a temperature scan of the 2D +-J random-bond Ising model.

Example
-------
    python scripts/run_sweep.py --L 32 --p 0.05 --t-min 1.0 --t-max 3.0 --n-temps 40 \\
        --realizations 32 --sweeps 20000 --out results/L32_p0.05.npz
"""

from __future__ import annotations

import argparse
import glob
import time
from typing import List, Optional, Tuple

import numpy as np

from rbim.disorder import load_disorder
from rbim.simulation import run_temperature_scan, save_results


def _load_disorder_directory(pattern: str, n_realizations: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no disorder files matched {pattern!r}")
    if len(paths) < n_realizations:
        raise ValueError(
            f"{len(paths)} disorder files matched {pattern!r} but --realizations={n_realizations}"
        )
    return [load_disorder(path) for path in paths[:n_realizations]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, required=True, help="linear system size")
    parser.add_argument("--p", type=float, required=True, help="antiferromagnetic bond concentration")
    parser.add_argument("--t-min", type=float, default=1.0, help="lowest temperature")
    parser.add_argument("--t-max", type=float, default=3.0, help="highest temperature")
    parser.add_argument("--n-temps", type=int, default=30, help="number of temperatures")
    parser.add_argument("--realizations", type=int, default=1, help="number of disorder realizations")
    parser.add_argument("--sweeps", type=int, default=10_000, help="measured sweeps per realization")
    parser.add_argument("--therm", type=int, default=None, help="thermalisation sweeps (default: sweeps // 10)")
    parser.add_argument("--bins", type=int, default=10, help="number of bins")
    parser.add_argument("--seed", type=int, default=0, help="master seed")
    parser.add_argument("--workers", type=int, default=None, help="process pool size (1 runs serially)")
    parser.add_argument("--no-percolation", action="store_true", help="skip the wrapping observables")
    parser.add_argument("--start", choices=["hot", "cold"], default="hot",
                        help="initial configuration; run both and compare to test equilibration")
    parser.add_argument(
        "--disorder-glob",
        type=str,
        default=None,
        help="optional glob of stored .npz realizations to use instead of generating new ones",
    )
    parser.add_argument("--out", type=str, required=True, help="output .npz path")
    args = parser.parse_args()

    temperatures = np.linspace(args.t_min, args.t_max, args.n_temps)

    disorder: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    if args.disorder_glob is not None:
        disorder = _load_disorder_directory(args.disorder_glob, args.realizations)

    print(
        f"L={args.L}  p={args.p}  {args.n_temps} temperatures in "
        f"[{args.t_min}, {args.t_max}]  {args.realizations} realization(s)  "
        f"{args.sweeps} sweeps"
    )

    start = time.time()
    results = run_temperature_scan(
        args.L,
        args.p,
        temperatures,
        n_realizations=args.realizations,
        n_sweeps=args.sweeps,
        n_therm=args.therm,
        n_bins=args.bins,
        seed=args.seed,
        n_workers=args.workers,
        measure_percolation=not args.no_percolation,
        start=args.start,
        disorder=disorder,
    )
    elapsed = time.time() - start

    save_results(args.out, results)
    print(f"finished in {elapsed:.1f} s; wrote {args.out}")

    if args.realizations == 1:
        print(
            "warning: a single disorder realization was used, so the reported error is "
            "thermal only and does not include sample-to-sample fluctuation."
        )


if __name__ == "__main__":
    main()
