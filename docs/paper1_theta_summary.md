# Paper 1 (FK percolation, θ = 1/2): status summary

Unpublished. Not yet discussed with Shtengel. All numbers come from `results/prod/` in this repo.

## 1. Definitions (full version: `docs/theta_derivation.pdf`)

- **Model**: 2D ±J RBIM, J = −1 with probability p, open/periodic as in `src/rbim/lattice.py`.
- **FK clusters**: built by Swendsen–Wang. A bond is active if it is satisfied (J s_i s_j > 0) and passes the test rand < 1 − e^{−2β}.
- **T₁(p)**: FK percolation temperature. It is the crossing in L of the wrapping probability R_y(T, L): the probability that a cluster wraps the torus in the y direction. Wrapping is detected exactly with a union-find that tracks displacements. The estimate quoted is the crossing of the two largest sizes.
- **θ**: defined by ⟨|M_C|⟩ ∝ ⟨S⟩^θ across L at fixed (p, T). Here S is the size of the largest FK cluster and M_C = Σ_{i∈C} s_i is its net magnetization.
  - θ = 1 means the cluster is internally aligned (pure Ising, where Coniglio–Klein gives m = P_∞).
  - θ = 1/2 means the cluster's projection onto the uniform direction is random-walk-like.
  - **θ is NOT the rate at which the gap T₁ − T₂ opens with p.**
- **Mechanism (derivation)**: an FK cluster contains no frustrated loop. It is therefore gauge-equivalent to a ferromagnet, s_i = ε σ_i with fixed gauge signs σ_i. So M_C = ε Σ σ_i. The gauge signs are correlated over the gauge length ξ_g = −1/ln(1−2p), which gives M_C ~ ξ_g √S. That is θ = 1/2 with amplitude A ≡ ⟨|M_C|⟩/√⟨S⟩ ∝ ξ_g.
  - Caveat stated in the PDF: the result depends on conditioning on the cluster, i.e. on the cluster shape decorrelating from the gauge pattern beyond ξ_g. This is a physical argument, not a theorem.

## 2. Figures (`results/figures/`, PDF + PNG)

| File | Content |
|---|---|
| `fig1_theta_loglog` | ⟨\|M_C\|⟩ vs ⟨S⟩ at T ≈ T₁, log-log, with a slope-1/2 guide |
| `fig2_amplitude` | A vs ξ_g (linear, A ≈ 1.12 ξ_g) |
| `fig3_theta_vs_T` | θ(T) per p, above the equilibration floor only |
| `fig4_phase_boundary` | T₁(p) |

## 3. Tables

**Production parameters, every (L, p)**: 10,000 disorder realizations; 2,000 SW sweeps after 500 thermalisation sweeps; 20 bins; 20 temperatures per p in a window around T₁; hot start; seed 1. L = 16, 24, 32, 48 for all p, plus L = 64 for p = 0.05 and 0.08. Errors are jackknife over realizations. Observables are built from full-run primitive moments, so they don't depend on the bin count.

**T₁(p)**: crossing of the two largest sizes. The statistical error comes from 2,000 resamplings of R_y within its jackknife errors. Successive pairwise crossings show the finite-size drift.

| p | sizes | T₁ (top pair) | stat. err | 16/24 | 24/32 | 32/48 | 48/64 |
|---|---|---|---|---|---|---|---|
| 0.02 | 16–48 | 2.1391 | 0.0007 | 2.1439 | 2.1413 | 2.1391 | – |
| 0.05 | 16–64 | 2.0171 | 0.0005 | 1.9905 | 1.9981 | 2.0071 | 2.0171 |
| 0.08 | 16–64 | 1.9495 | 0.0004 | 1.9239 | 1.9424 | 1.9483 | 1.9495 |
| 0.10 | 16–48 | 1.9160 | 0.0003 | 1.9053 | 1.9153 | 1.9160 | – |
| 0.15 | 16–48 | 1.8631 | 0.0003 | 1.8628 | 1.8629 | 1.8631 | – |
| 0.20 | 16–48 | 1.8350 | 0.0002 | 1.8355 | 1.8347 | 1.8350 | – |

How converged these are:
- p ≥ 0.08: converged to about 10⁻³.
- p = 0.02: still drifting by about 0.002 per size step.
- p = 0.05: **not converged**. The crossing still rises by about 0.01 per size step, so the drift, not the statistical error, sets the uncertainty.

**θ at T ≈ T₁** (`results/figures/summary.txt`; error = stat ⊕ sys, sys from the spread of pairwise slopes):

| p | ξ_g | θ | A |
|---|---|---|---|
| 0.02 | 24.5 | 0.94 ± 0.02 | 21.1 |
| 0.05 | 9.5 | 0.67 ± 0.09 | 13.8 |
| 0.08 | 5.7 | 0.513 ± 0.003 | 6.87 |
| 0.10 | 4.5 | 0.502 ± 0.008 | 5.07 |
| 0.15 | 2.8 | 0.499 ± 0.004 | 3.11 |
| 0.20 | 2.0 | 0.4998 ± 0.0008 | 2.19 |

Reading of the table: θ = 1/2 once L ≫ ξ_g. For p ≤ 0.05 the sizes L ≤ 64 are only a few ξ_g, and θ is still crossing over from 1 (expected from the argument, not yet shown to converge). Full θ(T) table: `python scripts/analyze_theta.py --dir results/prod --csv results/theta.csv`.

## 4. Run setup and validation

- `scripts/run_sweep.py` → `rbim.simulation.run_temperature_scan`. Run metadata is stored inside each `.npz` (`load_results(path)['metadata']`).
- Validation (`tests/`, 29 tests):
  - Onsager magnetization.
  - Coniglio–Klein m = P_∞ at p = 0.
  - Exact T_c recovered to 0.04% from both the Binder and wrapping crossings (`scripts/validate_pure_ising.py`).
  - Wrapping detector checked against a brute-force oracle on 1,200 configurations.
  - Observables independent of bin count.
- **Equilibration**:
  - Hot/cold-start test at L = 32, p = 0.08: agreement at T = 1.6, disagreement of 12–16σ at T = 1.4 (`results/eq/`).
  - Production windows were restricted to about T₁ − 0.3. `analyze_theta.py` additionally drops points below the ⟨|m|⟩ turnover.
  - Geometric observables (R_y, P_∞, S) stay reliable even where |m| is not.

## 5. AF-pinning measurement

**One-off measurement, not yet in the code.** At L = 64, p = 0.05, T = 1.0, after 3,000 SW sweeps, SW sits in a stuck state:
- |m| = 0.19 versus ≈ 0.98 in equilibrium;
- the largest cluster covers 99.7% of the lattice yet carries |M_C|/S = 0.19;
- the domain walls inside it run along AF bonds: 31% of wall bonds are AF against a 5.6% background (5.5× enrichment), and 40% of all AF bonds sit on a wall.

Interpretation: a wall placed on AF bonds is both cheap and made of satisfied (FK-active) bonds. It therefore lives inside the spanning cluster, where SW cannot remove it.

This describes a **non-equilibrium** SW state at a single (L, p, T). It belongs in an algorithm/methods discussion, not in an equilibrium claim.

## Checks on existing data (`scripts/analyze_bound_checks.py` → `results/figures/checks/`)

- **Bound.** Swendsen–Wang clusters flip independently, so ⟨M²⟩ = ⟨Σ_C M_C²⟩. It follows that [⟨|M_C|⟩]² ≤ [⟨M²⟩]. The data obey this at every point (max ratio 0.9989). Finite χ together with a percolating cluster therefore forces θ ≤ 1/2, and θ ≤ 48/91 at T₁.
  - Below T₁ at p ≥ 0.15, (π/2)[⟨|M_C|⟩]²/[⟨M²⟩] ≈ 0.8–1.0, so the largest cluster carries essentially all of ⟨M²⟩.
  - The Gaussian factor π/2 overshoots (ratio > 1) at low T for p ≤ 0.10, because M_C is not Gaussian there.
- **A is not constant in T.** At p = 0.20, A/ξ_g goes from 1.31 (T = 1.50) to 1.12 (T₁) to 1.02 (T = 2.18). At p = 0.10, from 2.33 to 1.13 (T₁) to 0.89. "A ≈ 1.12 ξ_g" is a statement at T₁ only.
- **A at T₁ is not universal either.** A/ξ_g ≈ 1.11–1.13 for p ≥ 0.10, ≈ 1.21 at p = 0.08, and still rising (1.69) at p = 0.05.
- **Crossover.** The pairwise θ_eff falls from 0.96 to 0.50 as L/ξ_g goes from 1 to 10, with approximate collapse; p = 0.05 sits slightly above the others.
- **FK active-bond density at T₁**, computed exactly from the energy as ρ = f(1 − e^{−2β}). It is exactly 1/2 at p = 0 (T_c) and at the Nishimori-line FK point (p = 1/4, T = 2/ln 3).

  | p | 0.02 | 0.05 | 0.08 | 0.10 | 0.15 | 0.20 |
  |---|---|---|---|---|---|---|
  | ρ(T₁) | 0.5152 | 0.5146 | 0.5106 | 0.5087 | 0.5045 | 0.5017 |

  So T₁(p) is **not** a constant-density line in between. The error is set by T₁ (about ±0.001). The p = 0.05 value moves down slightly if T₁ keeps drifting up.
- **Not stored, would need a rerun:** ⟨M_C²⟩, Σ_C M_C², active-bond counts (not needed, since they are exact from the energy), and per-realization data.

## What is missing for a paper

1. L = 64 for p = 0.02, 0.10, 0.15, 0.20. L ≥ 96 at p ≤ 0.05, where L/ξ_g is still small.
2. Proper extrapolation of T₁(L → ∞) using 2D percolation ν = 4/3. The p = 0.05 drift needs this most.
3. A universality check at T₁: ν from the slope of R_y, and the fractal dimension from S ~ L^{91/48}.
4. Hot/cold checks beyond the single point L = 32, p = 0.08.
5. T₂(p): not measurable reliably with SW below T₁. Take it from the literature, or state the paper is about T₁ only.
6. AF-pinning as a proper observable over many realizations, if it goes into the paper.
7. Literature closest to θ: arXiv:0704.1626 (M ~ √V droplet picture), Fajen–Hartmann–Young PRE 102, 012131 (3D, T₁ > T₂), Münster–Weigel PRE 107, 054103 and arXiv:2511.05748. Not yet checked against Yamaguchi's conjectured thresholds.
