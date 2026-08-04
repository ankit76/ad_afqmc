import numpy as np

from examples.statistics.analyze_afqmc_error_ensemble import _bootstrap_ratio_intervals


def test_bca_ratio_interval_corrects_reciprocal_standard_deviation_bias():
    means = np.random.default_rng(115).normal(size=64)
    reported_errors = np.ones(means.size)
    seed = 7319
    n_bootstrap = 10_000

    interval = _bootstrap_ratio_intervals(
        means,
        {"test": reported_errors},
        n_bootstrap=n_bootstrap,
        seed=seed,
    )["test"]

    # This reproduces the former raw-percentile construction.  Its lower
    # endpoint is shifted above one for these otherwise calibrated data.
    indices = np.random.default_rng(seed).integers(
        0,
        means.size,
        size=(n_bootstrap, means.size),
    )
    percentile_interval = np.percentile(
        1.0 / np.std(means[indices], axis=1, ddof=1),
        [2.5, 97.5],
    )

    assert percentile_interval[0] > 1.0
    assert interval[0] < 1.0 < interval[1]
