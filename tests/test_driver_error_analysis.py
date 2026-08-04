import numpy as np
import pytest

from trot import driver


def test_gamma_is_primary_while_both_energy_errors_are_computed():
    rng = np.random.default_rng(14)
    energy = rng.normal(size=1000)
    weights = np.exp(0.1 * rng.normal(size=energy.size))

    analysis = driver._analyze_energy_errors(
        energy,
        weights,
        error_method="gamma",
    )

    assert analysis.error_method == "gamma"
    assert analysis.stderr == analysis.gamma["se_gamma"]
    assert analysis.blocking["se_star"] is not None
    assert analysis.gamma["window_found"]


def test_blocking_can_be_selected_explicitly_for_legacy_comparisons():
    rng = np.random.default_rng(17)
    energy = rng.normal(size=1000)
    weights = np.ones(energy.size)

    analysis = driver._analyze_energy_errors(
        energy,
        weights,
        error_method="blocking",
    )

    assert analysis.error_method == "blocking"
    assert analysis.stderr == analysis.blocking["se_star"]
    assert np.isfinite(analysis.gamma["se_gamma"])


def test_short_gamma_analysis_is_unavailable_without_silent_fallback():
    analysis = driver._analyze_energy_errors(
        np.asarray([1.0, 2.0, 3.0]),
        np.ones(3),
        error_method="gamma",
    )

    assert np.isnan(analysis.stderr)
    assert not analysis.reliable
    assert analysis.blocking["se_star"] is None
    assert "at least four samples" in analysis.gamma["warnings"][0]


def test_energy_error_analysis_rejects_unknown_primary_method():
    with pytest.raises(ValueError, match="error_method"):
        driver._analyze_energy_errors(
            np.arange(10.0),
            np.ones(10),
            error_method="unknown",
        )
