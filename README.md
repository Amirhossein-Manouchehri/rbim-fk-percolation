# FK cluster percolation in the 2D ±J random-bond Ising model

Swendsen-Wang cluster Monte Carlo for the two-dimensional random-bond Ising model,

```
H = - Σ_<ij> J_ij s_i s_j ,      J_ij = +1 with probability 1 - p
                                        -1 with probability p
```

built to study the **Fortuin-Kasteleyn (FK) cluster percolation transition** and, in
particular, the internal structure of the percolating cluster.

## The physics

For any `p > 0` the FK clusters — the clusters built from *satisfied* bonds, the ones
Swendsen-Wang flips — start to percolate at a temperature `T₁` that is strictly above the
magnetic ordering temperature `T₂`. Between them the system has a system-spanning cluster
but no magnetization.

The reason is geometric. Around any closed loop the product of `s_i s_j` is `+1`, since
every spin appears twice. If every bond of the loop is satisfied then the product of the
couplings around it must also be `+1` — so the loop is unfrustrated. **An FK cluster can
therefore never contain a frustrated loop.** It is gauge-equivalent to a pure ferromagnet
and its spins are locked rigidly relative to one another, up to a global flip. What it
contributes to the physical magnetization is the projection of that internal pattern onto
the uniform direction, and under frustration the projection largely cancels.

In the pure model (`p = 0`) the Coniglio-Klein identity makes the magnetization equal to
the percolation strength, `m = P_∞`. This code measures how that identity breaks as
disorder is switched on, through the size and net magnetization of the largest cluster.

## Installation

```bash
git clone <repository-url>
cd rbim-fk-percolation
pip install -e ".[dev]"
```

## Quick start

Generate disorder realizations, then run a temperature scan:

```bash
python scripts/generate_disorder.py --L 32 --p 0.05 --realizations 32 --out data/disorder

python scripts/run_sweep.py --L 32 --p 0.05 \
    --t-min 1.0 --t-max 3.0 --n-temps 40 \
    --realizations 32 --sweeps 20000 \
    --disorder-glob "data/disorder/disorder_L=32_p=0.050_*.npz" \
    --out results/L32_p0.05.npz
```

Or from Python:

```python
import numpy as np
from rbim import run_temperature_scan

results = run_temperature_scan(
    L=32, p=0.05,
    temperatures=np.linspace(1.0, 3.0, 40),
    n_realizations=32, n_sweeps=20_000, seed=0,
)
p_inf = results["observables"]["p_infinity"]["value"]
```

## Observables

Per-site unless noted. Definitions are fixed here because several have more than one
convention in the literature.

| Name | Meaning |
|---|---|
| `energy`, `specific_heat` | `E/N` and `C = var(E)/(T² N)` |
| `magnetization`, `abs_magnetization` | `⟨m⟩` and `⟨|m|⟩` |
| `susceptibility` | `var(|M|)/(T N)` |
| `binder` | `1 - ⟨m⁴⟩ / (3⟨m²⟩²)` |
| `p_infinity` | largest-cluster fraction, `⟨S_max⟩/N` |
| `largest_size`, `largest_abs_magnetization` | `⟨S_max⟩` and `⟨|M_C|⟩`, kept **extensive** so the exponent θ in `M_C ~ S^θ` can be fitted against system size |
| `cluster_magnetization_ratio` | `⟨|M_C|/S_max⟩`, the internal alignment of the percolating cluster |
| `second_size_fraction`, `cluster_density_gap` | second-largest cluster and `(S₁ - S₂)/N` |
| `wrap_x`, `wrap_y`, `wrap_any` | probability that a cluster winds around the torus |

**Wrapping versus spanning.** These are different observables: `wraps_along` removes the
boundary-closing bonds along an axis and asks whether an active removed bond rejoins a
cluster to itself, while `spans_along` asks whether a cluster touches both open faces.
They share a transition point but not the universal value at criticality, so results must
state which one was used. Wrapping along `x` and along `y` are likewise distinct.

**Error bars.** Each observable carries `error`, the standard error across disorder
realizations, and `thermal_error`, the mean within-realization error. In a disordered
system the sample-to-sample fluctuation usually dominates, so `error` is the one to quote.
With a single realization the disorder error is undefined and the thermal error is
reported in its place.

## Validation

### Recovering the exact critical temperature

At `p = 0` the model reduces to the ferromagnetic square-lattice Ising model, whose critical
point is known exactly: `T_c = 2/ln(1+√2) = 2.269185`. Running

```bash
python scripts/validate_pure_ising.py --sizes 8 16 24 --sweeps 8000
```

extracts `T_c` twice over, from a magnetic observable and from a geometric one:

| crossing | L=8 vs 16 | L=8 vs 24 | L=16 vs 24 |
|---|---|---|---|
| Binder cumulant | 2.2491 | 2.2596 | 2.2627 |
| FK wrapping probability | 2.2623 | 2.2657 | 2.2674 |

Both drift monotonically toward the exact value as the sizes grow, which is the expected
finite-size behaviour; the largest pair reaches `2.2674`, within 0.08% of exact. The two
estimates also agree with each other to `0.008`, which is the Coniglio-Klein identity: at
zero disorder the percolation transition and the magnetic transition are the same
transition. Its breakdown at `p > 0` is what the rest of the project measures, so this
agreement is the baseline against which that breakdown is defined.

### Test suite

- the spontaneous magnetization at `T = 2.0` against Onsager's `(1 - sinh⁻⁴(2/T))^{1/8}`;
- the Coniglio-Klein identity `m = P_∞` at `p = 0`;
- reproducibility under a fixed seed, and independence across seeds;
- geometric invariants: bond counts, lattice degree, frustrated-plaquette density against
  `4p(1-p)[(1-p)² + p²]`, and the fact that only satisfied bonds are ever activated.

```bash
pytest                  # everything
pytest -m "not slow"    # unit tests only
```

## Project layout

```
src/rbim/
  lattice.py        square-lattice geometry and bond bookkeeping
  disorder.py       disorder generation, storage, frustrated plaquettes
  clusters.py       FK bond activation, labelling, wrapping and spanning
  swendsen_wang.py  one cluster update and its measurement
  simulation.py     thermalisation, binning, temperature scans, disorder averaging
scripts/            command-line entry points
tests/              unit tests and physics validation
```

## Notes and limitations

- Thermalisation is a fixed number of discarded sweeps with no automatic equilibration
  test. Swendsen-Wang is known to lose efficiency once the FK cluster percolates, which is
  precisely the regime of interest, so thermalisation should be verified independently —
  for example by comparing hot and cold starts — before trusting low-temperature results.
- Random streams come from a single master seed via `numpy.random.SeedSequence`, spawned
  per task, so parallel workers never share a stream and runs are reproducible.
- Disorder files use the same `J_h` / `J_v` `.npz` layout as the earlier simulations, so
  existing realizations can be read directly.

## License

MIT.
