import numpy as np
import pytest

from trot.stat_utils import (
    _autocovariance_fft,
    _pick_plateau_with_status,
    gamma_analysis_components,
    gamma_analysis_ratio,
)


def _stationary_ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty(n)
    values[0] = rng.normal()
    innovations = np.sqrt(1.0 - phi**2) * rng.normal(size=n - 1)
    for i in range(1, n):
        values[i] = phi * values[i - 1] + innovations[i - 1]
    return values


def test_autocovariance_fft_matches_direct_calculation():
    rng = np.random.default_rng(12)
    values = rng.normal(size=31)
    centered = values - values.mean()

    expected = np.asarray(
        [
            np.dot(centered[: values.size - lag], centered[lag:]) / (values.size - lag)
            for lag in range(9)
        ]
    )

    np.testing.assert_allclose(_autocovariance_fft(values, 8), expected, atol=1.0e-14)


def test_blocking_plateau_status_distinguishes_plateau_from_fallback():
    block_sizes = np.asarray([1, 2, 4, 8, 16])
    n_blocks = np.asarray([320, 160, 80, 40, 20])

    plateau = _pick_plateau_with_status(
        block_sizes,
        np.asarray([1.0, 1.3, 1.3, 1.3, 1.3]),
        n_blocks,
    )
    fallback = _pick_plateau_with_status(
        block_sizes,
        np.asarray([1.0, 1.1, 1.2, 1.3, 1.4]),
        n_blocks,
    )

    assert plateau[3:] == (True, "plateau")
    assert fallback[3:] == (False, "near_maximum_fallback")


def test_gamma_ratio_is_invariant_to_rescaling_weights():
    rng = np.random.default_rng(77)
    energy = rng.normal(size=4096)
    weights = np.exp(0.25 * rng.normal(size=energy.size))

    result = gamma_analysis_ratio(energy, weights, print_q=False)
    rescaled = gamma_analysis_ratio(energy, 7.0 * weights, print_q=False)

    assert result["window_found"]
    assert result["reliable"]
    assert 0.35 < result["tau_int"] < 0.75
    np.testing.assert_allclose(result["mu"], rescaled["mu"], atol=1.0e-15)
    np.testing.assert_allclose(result["se_gamma"], rescaled["se_gamma"], rtol=1.0e-14)
    np.testing.assert_allclose(
        result["influence"], rescaled["influence"], rtol=1.0e-14, atol=1.0e-15
    )


def test_gamma_method_recovers_ar1_autocorrelation_scale():
    n = 20_000
    phi = 0.8
    energy = _stationary_ar1(n, phi, seed=123)
    expected_tau = (1.0 + phi) / (2.0 * (1.0 - phi))
    expected_se = np.sqrt(2.0 * expected_tau / n)

    result = gamma_analysis_ratio(energy, np.ones(n), print_q=False)

    assert result["window_found"]
    assert result["reliable"]
    assert result["window"] > expected_tau
    assert result["tau_int"] == pytest.approx(expected_tau, rel=0.20)
    assert result["se_gamma"] == pytest.approx(expected_se, rel=0.15)


def test_component_gamma_matches_explicit_delta_method_projection():
    n = 4096
    x = _stationary_ar1(n, 0.7, seed=91)
    weights = np.exp(0.15 * _stationary_ar1(n, 0.3, seed=92))
    components = np.column_stack((1.5 + 0.2 * x, 2.0 - 0.1 * x))

    def combine(h0, values):
        return h0 + values[..., 0] * values[..., 1]

    result = gamma_analysis_components(
        0.25,
        weights,
        components,
        combine,
        print_q=False,
    )
    mean_components = np.sum(weights[:, None] * components, axis=0) / np.sum(weights)
    component_influence = (
        weights[:, None]
        * (components - mean_components[None, :])
        / np.mean(weights)
    )
    expected_influence = (
        mean_components[1] * component_influence[:, 0]
        + mean_components[0] * component_influence[:, 1]
    )
    expected = gamma_analysis_ratio(
        expected_influence,
        np.ones(n),
        print_q=False,
    )

    np.testing.assert_allclose(result["mean_components"], mean_components)
    np.testing.assert_allclose(result["mu"], combine(0.25, mean_components))
    np.testing.assert_allclose(result["influence"], expected_influence, rtol=2.0e-7)
    np.testing.assert_allclose(result["se_gamma"], expected["se_gamma"], rtol=2.0e-7)
    assert result["window"] == expected["window"]
    assert result["reliable"]


def test_component_gamma_supports_complex_weights_and_components():
    rng = np.random.default_rng(47)
    weights = np.exp(0.1 * rng.normal(size=1024) + 0.02j * rng.normal(size=1024))
    components = np.column_stack(
        (
            1.2 + 0.1 * rng.normal(size=1024) + 0.03j * rng.normal(size=1024),
            -0.7 + 0.2 * rng.normal(size=1024) + 0.04j * rng.normal(size=1024),
        )
    )

    def combine(h0, values):
        return h0 + values[..., 0] * values[..., 1]

    result = gamma_analysis_components(
        0.5,
        weights,
        components,
        combine,
        print_q=False,
    )
    rescaled = gamma_analysis_components(
        0.5,
        (3.0 - 2.0j) * weights,
        components,
        combine,
        print_q=False,
    )

    np.testing.assert_allclose(result["mu"], rescaled["mu"], atol=1.0e-14)
    np.testing.assert_allclose(
        result["mean_components"],
        rescaled["mean_components"],
        atol=1.0e-14,
    )
    np.testing.assert_allclose(result["se_gamma"], rescaled["se_gamma"], rtol=1.0e-13)
    np.testing.assert_allclose(
        result["influence"],
        rescaled["influence"],
        rtol=1.0e-12,
        atol=3.0e-15,
    )


def test_gamma_method_flags_an_artificially_short_window_search():
    energy = _stationary_ar1(2000, 0.98, seed=5)

    result = gamma_analysis_ratio(energy, np.ones(energy.size), max_lag=2, print_q=False)

    assert result["window"] == 2
    assert not result["window_found"]
    assert not result["reliable"]
    assert "automatic window was not found" in result["warnings"][0]


def test_gamma_method_handles_a_degenerate_ratio_series():
    weights = np.linspace(0.5, 2.0, 100)

    result = gamma_analysis_ratio(np.full(100, -1.25), weights, print_q=False)

    assert result["mu"] == pytest.approx(-1.25)
    assert result["se_gamma"] == 0.0
    assert result["tau_int"] is None
    assert not result["reliable"]


@pytest.mark.parametrize(
    "energy,weights,message",
    [
        (np.ones(3), np.ones(3), "at least four"),
        (np.ones(4), np.ones(3), "same number"),
        (np.asarray([1.0, 2.0, np.nan, 4.0]), np.ones(4), "finite"),
        (np.arange(4.0), np.zeros(4), "sum\\(wt\\)"),
    ],
)
def test_gamma_method_rejects_invalid_input(energy, weights, message):
    with pytest.raises(ValueError, match=message):
        gamma_analysis_ratio(energy, weights, print_q=False)
