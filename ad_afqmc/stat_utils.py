from functools import partial

import numpy as np

print = partial(print, flush=True)


def blocking_analysis(weights, energies, neql=0, printQ=False, writeBlockedQ=False):
    nSamples = weights.shape[0] - neql
    weights = weights[neql:]
    energies = energies[neql:]
    weightedEnergies = np.multiply(weights, energies)
    meanEnergy = weightedEnergies.sum() / weights.sum()
    if printQ:
        print(f"#\n# Mean: {meanEnergy:.8e}")
        print("# Block size    # of blocks         Mean                Error")
    blockSizes = np.array([1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 1000, 10000])
    prevError = 0.0
    plateauError = None
    for i in blockSizes[blockSizes < nSamples / 2.0]:
        nBlocks = nSamples // i
        blockedWeights = np.zeros(nBlocks)
        blockedEnergies = np.zeros(nBlocks)
        for j in range(nBlocks):
            blockedWeights[j] = weights[j * i : (j + 1) * i].sum()
            blockedEnergies[j] = (
                weightedEnergies[j * i : (j + 1) * i].sum() / blockedWeights[j]
            )
        v1 = blockedWeights.sum()
        v2 = (blockedWeights**2).sum()
        mean = np.multiply(blockedWeights, blockedEnergies).sum() / v1
        error = (
            np.multiply(blockedWeights, (blockedEnergies - mean) ** 2).sum()
            / (v1 - v2 / v1)
            / (nBlocks - 1)
        ) ** 0.5
        if writeBlockedQ:
            np.savetxt(
                f"samples_blocked_{i}.dat",
                np.stack((blockedWeights, blockedEnergies)).T,
            )
        if printQ:
            print(f"  {i:5d}           {nBlocks:6d}       {mean:.8e}       {error:.6e}")
        if error < 1.05 * prevError and plateauError is None:
            plateauError = max(error, prevError)
        prevError = error

    if printQ:
        if plateauError is not None:
            print(f"# Stocahstic error estimate: {plateauError:.6e}\n#")

    return meanEnergy, plateauError


def reject_outliers(data, obs, m=10.0, min_threshold=1e-5):
    target = data[:, obs]
    median_val = np.median(target)
    d = np.abs(target - median_val)
    mdev = np.median(d)
    q1, q3 = np.percentile(target, [25, 75])
    iqr = q3 - q1
    normalized_iqr = iqr / 1.349
    dispersion = max(mdev, normalized_iqr, min_threshold)
    s = d / dispersion
    mask = s < m
    return data[mask], mask


def jackknife_ratios(num: np.ndarray, denom: np.ndarray):
    r"""Jackknife estimation of standard deviation of the ratio of means.

    Parameters
    ----------
    num : :class:`np.ndarray
        Numerator samples.
    denom : :class:`np.ndarray`
        Denominator samples.

    Returns
    -------
    mean : :class:`np.ndarray`
        Ratio of means.
    sigma : :class:`np.ndarray`
        Standard deviation of the ratio of means.
    """
    n_samples = num.size
    num_mean = np.mean(num)
    denom_mean = np.mean(denom)
    mean = num_mean / denom_mean
    mean_num_all = (num_mean * n_samples - num) / (n_samples - 1)
    mean_denom_all = (denom_mean * n_samples - denom) / (n_samples - 1)
    jackknife_estimates = (mean_num_all / mean_denom_all).real
    mean = np.mean(jackknife_estimates)
    sigma = np.sqrt((n_samples - 1) * np.var(jackknife_estimates))
    return mean, sigma
