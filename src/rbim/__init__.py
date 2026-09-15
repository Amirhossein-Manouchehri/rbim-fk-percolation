"""Fortuin-Kasteleyn cluster percolation in the 2D +-J random-bond Ising model.

The package provides a Swendsen-Wang cluster Monte Carlo simulation of the
two-dimensional random-bond Ising model,

    H = - sum_<ij> J_ij s_i s_j,     J_ij = +1 with probability 1 - p
                                            -1 with probability p,

together with the geometric observables needed to study the Fortuin-Kasteleyn
(FK) cluster percolation transition and the structure of the percolating
cluster.

Submodules
----------
lattice      Square-lattice geometry and bond bookkeeping.
disorder     Generation, storage and frustration analysis of bond disorder.
clusters     FK bond activation, cluster labelling, spanning and wrapping.
swendsen_wang  A single Swendsen-Wang update and the per-sweep measurement.
simulation   Thermalisation, binning, temperature scans, disorder averaging.
"""

from rbim.lattice import Lattice, build_lattice, couplings_from_disorder
from rbim.disorder import (
    generate_disorder,
    load_disorder,
    save_disorder,
    frustrated_plaquettes,
    frustration_density,
)
from rbim.clusters import (
    activate_bonds,
    label_clusters,
    cluster_statistics,
    wraps_along,
    spans_along,
)
from rbim.swendsen_wang import SweepResult, sw_sweep
from rbim.simulation import run_single_realization, run_temperature_scan

__version__ = "0.1.0"

__all__ = [
    "Lattice",
    "build_lattice",
    "couplings_from_disorder",
    "generate_disorder",
    "load_disorder",
    "save_disorder",
    "frustrated_plaquettes",
    "frustration_density",
    "activate_bonds",
    "label_clusters",
    "cluster_statistics",
    "wraps_along",
    "spans_along",
    "SweepResult",
    "sw_sweep",
    "run_single_realization",
    "run_temperature_scan",
]
