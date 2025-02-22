from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import vmap

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


def reject_outliers(data, obs, m=10.0):
    d = np.abs(data[:, obs] - np.median(data[:, obs]))
    mdev = np.median(d) + 1.0e-10
    s = d / mdev if mdev else 0.0
    return data[s < m], s < m


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
    num_mean = jnp.mean(num)
    denom_mean = jnp.mean(denom)
    mean = num_mean / denom_mean

    idx = jnp.arange(n_samples)

    jacki = lambda i: ((num_mean * n_samples - num[i]) / (denom_mean * n_samples - denom[i])).real
    jackknife_estimates = vmap(jacki, in_axes=0)(idx)
    return jnp.mean(jackknife_estimates), jnp.sqrt((n_samples - 1) * np.var(jackknife_estimates))


    jackknife_estimates = np.zeros(n_samples, dtype=num.dtype)

    for i in range(n_samples):
        mean_num_i = (num_mean * n_samples - num[i]) / (n_samples - 1)
        mean_denom_i = (denom_mean * n_samples - denom[i]) / (n_samples - 1)
        jackknife_estimates[i] = (mean_num_i / mean_denom_i).real
    mean = np.mean(jackknife_estimates)
    sigma = np.sqrt((n_samples - 1) * np.var(jackknife_estimates))
    return mean, sigma
