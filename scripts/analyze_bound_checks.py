#!/usr/bin/env python3
"""Checks on theta = 1/2 that need no new runs.

1. Cluster-estimator bound.  SW clusters flip independently, so
   <M^2> = <sum_C M_C^2> >= <M_C,max^2> >= <|M_C,max|>^2, and by Jensen the same
   holds after disorder averaging.  We report
       r_strict = [<|M_C|>]^2 / [<M^2>]          (must be <= 1, no assumption)
       r_gauss  = (pi/2) [<|M_C|>]^2 / [<M^2>]    (fraction of <M^2> carried by
                                                  the largest cluster, if M_C
                                                  is Gaussian)
   [<M^2>] is recovered exactly from the stored observables:
       [<M^2>] = chi T N + (N [<|m|>])^2
   because both were formed from disorder-averaged primitives.
2. Amplitude A = [<|M_C|>] / sqrt([<S>]) against T at fixed p.
3. Crossover: effective theta (pairwise slope) and A/xi_g against L/xi_g.
4. FK active-bond density rho = f (1 - e^{-2 beta}), f = satisfied fraction
   = (1 - e/2)/2 with e = E/N (two bonds per site); exact in expectation.
   rho = 1/2 at the pure-Ising T_c and at the Nishimori-line FK point
   (p = 1/4, T = 2/ln 3); tested here at T_1(p) in between.

Writes figures to results/figures/checks/ and a CSV per check.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rbim.simulation import load_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD = os.path.join(ROOT, "results", "prod")
OUT = os.path.join(ROOT, "results", "figures", "checks")
P_VALUES = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
SIZES = [16, 24, 32, 48, 64]


def xi_gauge(p: float) -> float:
    return -1.0 / np.log(1.0 - 2.0 * p)


def crossing(T, a, b):
    d = np.asarray(a) - np.asarray(b)
    out = [T[i] - d[i] * (T[i + 1] - T[i]) / (d[i + 1] - d[i])
           for i in range(len(d) - 1) if d[i] * d[i + 1] < 0]
    return max(out) if out else np.nan


def floor_T(T, absm):
    rising = [i for i in range(len(absm) - 1) if absm[i] < absm[i + 1]]
    return T[max(rising) + 1] if rising else T[0]


def load(p):
    data = {}
    for L in SIZES:
        path = os.path.join(PROD, f"L{L}_p{p:.2f}.npz")
        if os.path.exists(path):
            data[L] = load_results(path)
    return data


def obs(r, key):
    return np.asarray(r["observables"][key]["value"], float), np.asarray(r["observables"][key]["error"], float)


def main():
    os.makedirs(OUT, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(P_VALUES)))
    rows_bound, rows_A, rows_cross, rows_rho = [], [], [], []

    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 4))
    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    fig3, ax3 = plt.subplots(1, 2, figsize=(10, 4))

    for c, p in zip(colors, P_VALUES):
        data = load(p)
        Ls = sorted(data)
        T = np.asarray(data[Ls[-1]]["temperatures"])
        xi = xi_gauge(p)
        t1 = crossing(T, obs(data[Ls[-2]], "wrap_y")[0], obs(data[Ls[-1]], "wrap_y")[0])
        floor = max(floor_T(T, obs(data[L], "abs_magnetization")[0]) for L in Ls)
        ok = T >= floor

        S = {}; MC = {}
        for L in Ls:
            N = L * L
            chi, _ = obs(data[L], "susceptibility")
            absm, _ = obs(data[L], "abs_magnetization")
            M2 = chi * T * N + (N * absm) ** 2
            s, ds = obs(data[L], "largest_size")
            mc, dmc = obs(data[L], "largest_abs_magnetization")
            S[L], MC[L] = (s, ds), (mc, dmc)
            r_strict = mc**2 / M2
            for i in range(len(T)):
                rows_bound.append((p, L, T[i], r_strict[i], np.pi / 2 * r_strict[i], int(ok[i])))
            if L == Ls[-1]:
                ax1[0].plot(T[ok] / t1, (np.pi / 2 * r_strict)[ok], "o", ms=4, color=c, label=f"p={p:.2f}, L={L}")
                ax1[0].plot(T[~ok] / t1, (np.pi / 2 * r_strict)[~ok], "o", ms=4, mfc="none", color=c)
                ax1[1].plot(T[ok] / t1, r_strict[ok], "o", ms=4, color=c)

            A = mc / np.sqrt(s)
            dA = A * np.sqrt((dmc / mc) ** 2 + (0.5 * ds / s) ** 2)
            for i in range(len(T)):
                rows_A.append((p, L, T[i], A[i], dA[i], A[i] / xi, int(ok[i])))
            if L == Ls[-1]:
                ax2.errorbar(T[ok] / t1, A[ok] / xi, dA[ok] / xi, fmt="o", ms=4, color=c, label=f"p={p:.2f}, L={L}")
                ax2.plot(T[~ok] / t1, A[~ok] / xi, "o", ms=4, mfc="none", color=c)

        # crossover at the grid temperature nearest T1
        i1 = int(np.argmin(np.abs(T - t1)))
        for a, b in zip(Ls[:-1], Ls[1:]):
            th = (np.log(MC[b][0][i1]) - np.log(MC[a][0][i1])) / (np.log(S[b][0][i1]) - np.log(S[a][0][i1]))
            x = np.sqrt(a * b) / xi
            rows_cross.append((p, a, b, T[i1], x, th))
            ax3[0].plot(x, th, "o", color=c, label=f"p={p:.2f}" if a == Ls[0] else None)
        for L in Ls:
            A = MC[L][0][i1] / np.sqrt(S[L][0][i1])
            ax3[1].plot(L / xi, A / xi, "o", color=c, label=f"p={p:.2f}" if L == Ls[0] else None)

        # FK active-bond density at T1, interpolated on the largest size
        e, de = obs(data[Ls[-1]], "energy")
        f = (1.0 - e / 2.0) / 2.0
        rho = f * (1.0 - np.exp(-2.0 / T))
        rho_t1 = float(np.interp(t1, T, rho))
        drho = float(np.interp(t1, T, de / 4.0 * (1.0 - np.exp(-2.0 / T))))
        rows_rho.append((p, Ls[-1], t1, rho_t1, drho))

    # Reference points for rho: pure Ising and Nishimori-line FK point
    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))

    ax1[0].axhline(1, color="k", lw=0.8, ls="--")
    ax1[0].set(xlabel=r"$T/T_1(p)$", ylabel=r"$(\pi/2)[\langle|M_C|\rangle]^2/[\langle M^2\rangle]$",
               title="share of $M^2$ in largest cluster (Gaussian $M_C$)")
    ax1[1].axhline(1, color="k", lw=0.8, ls="--")
    ax1[1].set(xlabel=r"$T/T_1(p)$", ylabel=r"$[\langle|M_C|\rangle]^2/[\langle M^2\rangle]$", title="strict bound: must be $\\leq 1$")
    ax1[0].legend(fontsize=7)
    fig1.tight_layout(); fig1.savefig(os.path.join(OUT, "check1_bound.pdf")); fig1.savefig(os.path.join(OUT, "check1_bound.png"), dpi=150)

    ax2.axvline(1, color="gray", lw=0.8, ls=":")
    ax2.set(xlabel=r"$T/T_1(p)$", ylabel=r"$A/\xi_g$", title=r"amplitude vs temperature (open: below equilibration floor)")
    ax2.legend(fontsize=7)
    fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "check2_A_vs_T.pdf")); fig2.savefig(os.path.join(OUT, "check2_A_vs_T.png"), dpi=150)

    ax3[0].axhline(0.5, color="k", lw=0.8, ls="--"); ax3[0].axhline(48 / 91, color="r", lw=0.8, ls=":")
    ax3[0].set(xscale="log", xlabel=r"$\sqrt{L_aL_b}/\xi_g$", ylabel=r"pairwise $\theta_{\rm eff}$ at $T\approx T_1$")
    ax3[1].set(xscale="log", xlabel=r"$L/\xi_g$", ylabel=r"$A/\xi_g$ at $T\approx T_1$")
    ax3[0].legend(fontsize=7)
    fig3.tight_layout(); fig3.savefig(os.path.join(OUT, "check3_crossover.pdf")); fig3.savefig(os.path.join(OUT, "check3_crossover.png"), dpi=150)

    def write(name, header, rows):
        with open(os.path.join(OUT, name), "w") as fh:
            fh.write(header + "\n")
            for r in rows:
                fh.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r) + "\n")

    write("check1_bound.csv", "p,L,T,r_strict,r_gauss,usable", rows_bound)
    write("check2_amplitude.csv", "p,L,T,A,dA,A_over_xi,usable", rows_A)
    write("check3_crossover.csv", "p,La,Lb,T,sqrtLaLb_over_xi,theta_pair", rows_cross)
    write("check4_fk_density.csv", "p,L,T1,rho_active,err", rows_rho)

    print("check 1  max r_strict over usable points:",
          f"{max(r[3] for r in rows_bound if r[5]):.4f}  (bound: <= 1)")
    print("check 1  r_gauss at T~T1, largest L:")
    print("check 4  FK active-bond density at T1 (pure Ising Tc and Nishimori FK point give exactly 1/2):")
    for r in rows_rho:
        print(f"   p={r[0]:.2f} L={r[1]} T1={r[2]:.4f} rho={r[3]:.4f} +- {r[4]:.4f}")
    print(f"   reference: p=0 at Tc={Tc:.4f}; p=0.25 at T=2/ln3={2/np.log(3):.4f}")
    print("check 3  pairwise theta vs L/xi_g:")
    for r in rows_cross:
        print(f"   p={r[0]:.2f} {r[1]}/{r[2]} x={r[4]:6.2f} theta={r[5]:.4f}")


if __name__ == "__main__":
    main()
