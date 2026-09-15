#!/usr/bin/env python3
"""Generate and store +-J disorder realizations.

Example
-------
    python scripts/generate_disorder.py --L 32 --p 0.05 --realizations 64 --out data/disorder
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from rbim.disorder import (
    default_disorder_filename,
    frustration_density,
    generate_disorder,
    save_disorder,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, required=True, help="linear system size")
    parser.add_argument("--p", type=float, required=True, help="antiferromagnetic bond concentration")
    parser.add_argument("--realizations", type=int, default=1, help="number of realizations to generate")
    parser.add_argument("--out", type=str, default="data/disorder", help="output directory")
    parser.add_argument("--seed", type=int, default=0, help="master seed")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seeds = np.random.SeedSequence(args.seed).spawn(args.realizations)

    densities = []
    for index, child in enumerate(seeds):
        rng = np.random.default_rng(child)
        J_h, J_v = generate_disorder(args.L, args.p, rng)
        densities.append(frustration_density(J_h, J_v))

        path = os.path.join(args.out, default_disorder_filename(args.L, args.p, index))
        save_disorder(path, J_h, J_v, p=args.p, seed=args.seed)

    expected = 4 * args.p * (1 - args.p) * ((1 - args.p) ** 2 + args.p**2)
    print(f"wrote {args.realizations} realization(s) for L={args.L}, p={args.p} to {args.out}")
    print(f"frustrated plaquette density: {np.mean(densities):.4f} (expected {expected:.4f})")


if __name__ == "__main__":
    main()
