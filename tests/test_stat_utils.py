import numpy as np
import pytest

from trot.stat_utils import _autocovariance_fft, gamma_analysis_ratio


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
