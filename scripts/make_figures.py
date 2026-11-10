#!/usr/bin/env python3
"""Produce the publication figures from a completed production scan.

Four figures are written, each as both PDF (for LaTeX) and PNG (for slides):

fig1_theta_loglog
    Mean cluster magnetization against mean cluster size, log-log, one series
    per disorder concentration, with a straight-line fit whose slope is theta.
    This is the primary figure: the slope drops from ~1 at weak disorder, where
    the percolating cluster is internally aligned, to 1/2 at strong disorder,
    where its moment is a random walk in its own size.

fig2_amplitude
    The amplitude A = <|M_C|> / sqrt(<S>) against the gauge correlation length
    xi = -1/ln(1-2p).  The prediction A proportional to xi follows from the
    gauge argument; the fitted slope is the only free number.

fig3_theta_vs_T
    theta as a function of temperature at each disorder concentration, from the
    two largest lattice sizes, with the values 1 and 1/2 marked.

fig4_phase_boundary
    The Fortuin-Kasteleyn percolation temperature against disorder, anchored at
    p = 0 by the exact Ising value.

Errors on theta combine a parametric bootstrap over the independent size-runs
with a systematic term from the drift between consecutive-size slopes; the
systematic dominates and is the number to quote.

Example
-------
    python scripts/make_figures.py --dir results/prod --out results/figures
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rbim.simulation import load_results

SIZES = (16, 24, 32, 48, 64)
COLORS = {
    "0.02": "#2a78d6", "0.05": "#eb6834", "0.08": "#1baf7a",
    "0.10": "#eda100", "0.15": "#e87ba4", "0.20": "#6250d6",
}
MARKERS = {"0.02": "o", "0.05": "^", "0.08": "s", "0.10": "D", "0.15": "v", "0.20": "*"}


def style() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": 0.8, "lines.linewidth": 1.2,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })


def gauge_length(p: float) -> float:
    """Gauge correlation length xi = -1/ln(1-2p), in bonds."""
    return -1.0 / np.log(1.0 - 2.0 * p)


def weighted_slope(x, y, sx, sy) -> Tuple[float, float]:
    """Straight-line fit with uncertainty on both axes (effective variance)."""
    slope = float(np.polyfit(x, y, 1)[0])
    intercept = 0.0
    for _ in range(60):
        w = 1.0 / np.maximum(sy**2 + (slope * sx) ** 2, 1e-30)
        sw, sxw, syw = w.sum(), (w * x).sum(), (w * y).sum()
        sxx, sxy = (w * x * x).sum(), (w * x * y).sum()
        det = sw * sxx - sxw**2
        new = (sw * sxy - sxw * syw) / det
        intercept = (sxx * syw - sxw * sxy) / det
        if abs(new - slope) < 1e-14:
            slope = new
            break
        slope = new
    return slope, intercept


def load_sizes(directory: str, p: str) -> Dict[int, dict]:
    found = {}
    for L in SIZES:
        path = os.path.join(directory, f"L{L}_p{p}.npz")
        if os.path.exists(path):
            found[L] = load_results(path)
    return found


def theta_at(runs: Dict[int, dict], index: int, rng) -> Tuple[float, float, float, dict]:
    """Return theta, its bootstrap error, the consecutive-pair systematic, and the raw points."""
    sizes = sorted(runs)
    S = np.array([runs[L]["observables"]["largest_size"]["value"][index] for L in sizes])
    dS = np.array([runs[L]["observables"]["largest_size"]["error"][index] for L in sizes])
    M = np.array([runs[L]["observables"]["largest_abs_magnetization"]["value"][index] for L in sizes])
    dM = np.array([runs[L]["observables"]["largest_abs_magnetization"]["error"][index] for L in sizes])

    x, y = np.log(S), np.log(M)
    slope, intercept = weighted_slope(x, y, dS / S, dM / M)

    draws = []
    for _ in range(4000):
        Sb, Mb = S + rng.normal(0, dS), M + rng.normal(0, dM)
        if np.any(Sb <= 0) or np.any(Mb <= 0):
            continue
        draws.append(weighted_slope(np.log(Sb), np.log(Mb), dS / Sb, dM / Mb)[0])
    stat = float(np.std(draws, ddof=1))

    pairs = [(y[j + 1] - y[j]) / (x[j + 1] - x[j]) for j in range(len(sizes) - 1)]
    systematic = float(abs(pairs[-1] - pairs[-2])) if len(pairs) > 1 else float("nan")

    return slope, stat, systematic, {
        "sizes": sizes, "S": S, "dS": dS, "M": M, "dM": dM, "intercept": intercept
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="results/prod")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--p", nargs="+", default=["0.02", "0.05", "0.08", "0.10", "0.15", "0.20"])
    parser.add_argument("--T1", nargs="+", type=float,
                        default=[2.139, 2.007, 1.948, 1.916, 1.863, 1.835],
                        help="reference temperature per p; figures are evaluated here")
    args = parser.parse_args()

    style()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    data = {p: load_sizes(args.dir, p) for p in args.p}
    reference = dict(zip(args.p, args.T1))

    # ---------------- figure 1: the log-log fits ----------------
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    summary: List[str] = []
    amps: List[Tuple[float, float, bool]] = []

    for p in args.p:
        runs = data[p]
        if len(runs) < 3:
            continue
        T = runs[max(runs)]["temperatures"]
        i = int(np.argmin(abs(T - reference[p])))
        theta, stat, systematic, raw = theta_at(runs, i, rng)
        total = float(np.hypot(stat, systematic))

        S, M, dM = raw["S"], raw["M"], raw["dM"]
        ax.errorbar(S, M, yerr=dM, fmt=MARKERS[p], color=COLORS[p], ms=6,
                    capsize=2, lw=0, elinewidth=1,
                    label=rf"$p={float(p):.2f}$,  $\theta={theta:.3f}\pm{total:.3f}$")
        span = np.array([S.min() * 0.8, S.max() * 1.25])
        ax.plot(span, np.exp(raw["intercept"]) * span**theta, "-", color=COLORS[p], lw=1.1, alpha=0.9)

        A = float(np.mean(M / np.sqrt(S)))
        converged = systematic < 0.02
        amps.append((gauge_length(float(p)), A, converged))
        summary.append(f"p={p}  T={T[i]:.3f}  theta={theta:.4f} +- {total:.4f} "
                       f"(stat {stat:.4f}, sys {systematic:.4f})  A={A:.3f}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"mean largest-cluster size  $\langle S\rangle$")
    ax.set_ylabel(r"mean cluster magnetization  $\langle |M_C|\rangle$")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    for path in (f"{args.out}/fig1_theta_loglog.pdf", f"{args.out}/fig1_theta_loglog.png"):
        fig.savefig(path)
    plt.close(fig)

    # ---------------- figure 2: amplitude against gauge length ----------------
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    xi_c = np.array([a[0] for a in amps if a[2]])
    A_c = np.array([a[1] for a in amps if a[2]])
    xi_u = np.array([a[0] for a in amps if not a[2]])
    A_u = np.array([a[1] for a in amps if not a[2]])

    if len(xi_c) >= 2:
        slope = float((A_c * xi_c).sum() / (xi_c**2).sum())     # fit through the origin
        grid = np.linspace(0, max(xi_c.max(), 1) * 1.15, 50)
        ax.plot(grid, slope * grid, "--", color="#52514e", lw=1,
                label=rf"$A={slope:.2f}\,\xi$")
    ax.plot(xi_c, A_c, "o", color="#2a78d6", ms=7, label=r"converged $\theta=1/2$")
    if len(xi_u):
        ax.plot(xi_u, A_u, "^", mfc="none", mec="#898781", ms=7, label="not converged")
    ax.set_xlabel(r"gauge correlation length  $\xi=-1/\ln(1-2p)$")
    ax.set_ylabel(r"amplitude  $A=\langle |M_C|\rangle/\sqrt{\langle S\rangle}$")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    for path in (f"{args.out}/fig2_amplitude.pdf", f"{args.out}/fig2_amplitude.png"):
        fig.savefig(path)
    plt.close(fig)

    # ---------------- figure 3: theta against temperature ----------------
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for p in args.p:
        runs = data[p]
        if len(runs) < 3:
            continue
        big, second = sorted(runs)[-1], sorted(runs)[-2]
        T = runs[big]["temperatures"]
        S1 = np.array(runs[second]["observables"]["largest_size"]["value"])
        S2 = np.array(runs[big]["observables"]["largest_size"]["value"])
        M1 = np.array(runs[second]["observables"]["largest_abs_magnetization"]["value"])
        M2 = np.array(runs[big]["observables"]["largest_abs_magnetization"]["value"])
        theta = (np.log(M2) - np.log(M1)) / (np.log(S2) - np.log(S1))
        ax.plot(T, theta, MARKERS[p] + "-", color=COLORS[p], ms=4, lw=1,
                label=rf"$p={float(p):.2f}$")
    ax.axhline(1.0, color="#898781", ls=":", lw=1)
    ax.axhline(0.5, color="#898781", ls=":", lw=1)
    ax.set_xlabel(r"temperature  $T$")
    ax.set_ylabel(r"$\theta$  (two largest sizes)")
    ax.set_ylim(0.42, 1.08)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    for path in (f"{args.out}/fig3_theta_vs_T.pdf", f"{args.out}/fig3_theta_vs_T.png"):
        fig.savefig(path)
    plt.close(fig)

    # ---------------- figure 4: the percolation line ----------------
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    p_axis = np.array([0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20])
    t1 = np.array([2.268, 2.192, 2.139, 2.072, 2.007, 1.948, 1.916, 1.863, 1.835])
    ax.plot(p_axis, t1, "o-", color="#2a78d6", ms=5, lw=1.1, label=r"$T_1$  (FK percolation)")
    ax.plot(0.0, 2.269185, "*", color="#0b0b0b", ms=13, label=r"exact Ising $T_c$")
    ax.plot([0.02, 0.05], [2.086, 1.843], "^--", mfc="none", color="#eb6834", ms=7,
            lw=1, label=r"$T_2$  (ferro, where measurable)")
    ax.set_xlabel(r"antiferromagnetic bond fraction  $p$")
    ax.set_ylabel(r"temperature  $T$")
    ax.legend(frameon=False, fontsize=9)
    for path in (f"{args.out}/fig4_phase_boundary.pdf", f"{args.out}/fig4_phase_boundary.png"):
        fig.savefig(path)
    plt.close(fig)

    with open(os.path.join(args.out, "summary.txt"), "w") as handle:
        handle.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"\nfigures written to {args.out}/ (pdf + png)")


if __name__ == "__main__":
    main()
