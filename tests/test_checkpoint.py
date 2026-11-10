"""Checkpointed, resumed and parallel scans must equal an uninterrupted serial scan."""

from __future__ import annotations

import os

import numpy as np
import pytest

from rbim.simulation import OBSERVABLES, load_results, run_temperature_scan, save_results

KWARGS = dict(n_realizations=4, n_sweeps=60, n_therm=20, n_bins=10, seed=4)
TEMPERATURES = [1.7, 1.9, 2.1]


def _same(a, b):
    for name in OBSERVABLES:
        for key in ("value", "error", "thermal_error"):
            if not np.array_equal(a["observables"][name][key], b["observables"][name][key], equal_nan=True):
                return False
    return np.array_equal(a["per_realization"]["values"], b["per_realization"]["values"])


def test_resume_reproduces_uninterrupted_run(tmp_path):
    reference = run_temperature_scan(12, 0.08, TEMPERATURES, n_workers=1, **KWARGS)

    directory = str(tmp_path / "ckpt")
    first = run_temperature_scan(12, 0.08, TEMPERATURES, n_workers=1, checkpoint_dir=directory, **KWARGS)
    assert _same(reference, first)

    # Simulate a job killed after the first temperature.
    os.remove(os.path.join(directory, "T0001.npz"))
    os.remove(os.path.join(directory, "T0002.npz"))
    resumed = run_temperature_scan(12, 0.08, TEMPERATURES, n_workers=1, checkpoint_dir=directory, **KWARGS)
    assert _same(reference, resumed)


def test_checkpoint_from_other_run_is_refused(tmp_path):
    directory = str(tmp_path / "ckpt")
    run_temperature_scan(12, 0.08, TEMPERATURES, n_workers=1, checkpoint_dir=directory, **KWARGS)
    other = dict(KWARGS, n_sweeps=70)
    with pytest.raises(ValueError):
        run_temperature_scan(12, 0.08, TEMPERATURES, n_workers=1, checkpoint_dir=directory, **other)


def test_per_realization_file_and_summary_roundtrip(tmp_path):
    results = run_temperature_scan(12, 0.08, TEMPERATURES, n_workers=1, **KWARGS)
    summary = str(tmp_path / "scan.npz")
    per_realization = str(tmp_path / "scan.real.npz")
    save_results(summary, results, per_realization_path=per_realization)

    loaded = load_results(summary)
    assert set(loaded["observables"]) == set(OBSERVABLES)
    with np.load(per_realization) as data:
        assert data["values"].shape == (len(TEMPERATURES), KWARGS["n_realizations"], len(data["names"]))
        # The disorder average of a per-realization primitive is the summary value.
        names = list(data["names"])
        largest = data["values"][:, :, names.index("largest_size")].mean(axis=1)
    assert np.allclose(largest, loaded["observables"]["largest_size"]["value"])


def test_connected_susceptibility_definition():
    results = run_temperature_scan(12, 0.08, TEMPERATURES, n_workers=1, **KWARGS)
    names = results["per_realization"]["names"]
    values = results["per_realization"]["values"]
    m2 = values[:, :, names.index("magnetization_sq")]
    abs_m = values[:, :, names.index("abs_magnetization")]
    n_sites = 12 * 12
    expected = (m2 - abs_m**2).mean(axis=1) / (np.asarray(TEMPERATURES) * n_sites)
    assert np.allclose(results["observables"]["susceptibility_connected"]["value"], expected)
    # Subtracting the square of the disorder-averaged mean can only give more.
    assert np.all(
        results["observables"]["susceptibility"]["value"]
        >= results["observables"]["susceptibility_connected"]["value"] - 1e-12
    )
