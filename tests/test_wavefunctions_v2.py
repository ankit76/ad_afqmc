import os
import numpy as np

from ad_afqmc import config
config.setup_jax()
from jax import numpy as jnp
from jax import scipy as jsp
from ad_afqmc import pyscf_interface, wavefunctions, wavefunctions_v2

# -----------------------------------------------------------------------------
# RHF wavefunction.
seed = 102
np.random.seed(seed)
norb, nelec, nchol = 10, (5, 5), 5
rhf = wavefunctions_v2.rhf(norb, nelec)

wave_data = {}
wave_data["mo_coeff"] = jnp.eye(norb)[:, : nelec[0]]
wave_data["rdm1"] = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
walker = jnp.array(np.random.rand(norb, nelec[0])) + \
            1.0j * jnp.array(np.random.rand(norb, nelec[0]))

ham_data = {}
ham_data["h0"] = np.random.rand(1)[0]
ham_data["rot_h1"] = jnp.array(np.random.rand(nelec[0], norb))
ham_data["rot_chol"] = jnp.array(np.random.rand(nchol, nelec[0], norb))
ham_data["h1"] = jnp.array([np.random.rand(norb, norb)] * 2)
ham_data["chol"] = jnp.array(np.random.rand(nchol, norb, norb))
ham_data["ene0"] = 0.0

ham_data["normal_ordering_term"] = -0.5 * jnp.einsum(
    "gik,gjk->ij",
    ham_data["chol"].reshape(-1, norb, norb),
    ham_data["chol"].reshape(-1, norb, norb),
    optimize="optimal",
)

# -----------------------------------------------------------------------------
# UHF wavefunction.
nelec_sp = (3, 2)
uhf = wavefunctions_v2.uhf(norb, nelec_sp)

wave_data_u = {}
wave_data_u["mo_coeff"] = [
    jnp.array(np.random.rand(norb, nelec_sp[0])),
    jnp.array(np.random.rand(norb, nelec_sp[1])),
]
wave_data_u["rdm1"] = jnp.array(
    [
        jnp.array(wave_data_u["mo_coeff"][0] @ wave_data_u["mo_coeff"][0].T),
        jnp.array(wave_data_u["mo_coeff"][1] @ wave_data_u["mo_coeff"][1].T),
    ]
)

walker_up = jnp.array(np.random.rand(norb, nelec_sp[0])) + \
                1.0j * jnp.array(np.random.rand(norb, nelec_sp[0]))
walker_dn = jnp.array(np.random.rand(norb, nelec_sp[1])) + \
                1.0j * jnp.array(np.random.rand(norb, nelec_sp[1]))

n_walkers = 10
walkers_u = [jnp.array([walker_up] * n_walkers), jnp.array([walker_dn] * n_walkers)]

ham_data_u = {}
ham_data_u["h0"] = ham_data["h0"]
ham_data_u["rot_h1"] = [
    ham_data["rot_h1"][: nelec_sp[0], :],
    ham_data["rot_h1"][: nelec_sp[1], :],
]
ham_data_u["rot_chol"] = [
    ham_data["rot_chol"][:, : nelec_sp[0], :],
    ham_data["rot_chol"][:, : nelec_sp[1], :],
]
ham_data_u["h1"] = ham_data["h1"]  # jnp.array([ham_data["h1"], ham_data["h1"]])
ham_data_u["chol"] = ham_data["chol"]
ham_data_u["ene0"] = ham_data["ene0"]

# -----------------------------------------------------------------------------
# NOCI wavefunction.
ndets = 5
noci = wavefunctions.noci(norb, nelec_sp, ndets)
dets = [
    jnp.array(np.random.rand(ndets, norb, nelec_sp[0])),
    jnp.array(np.random.rand(ndets, norb, nelec_sp[1])),
]
ci_coeffs = jnp.array(np.random.randn(ndets))

wave_data_noci = {}
wave_data_noci["ci_coeffs_dets"] = [ci_coeffs, dets]
wave_data_noci["rdm1"] = wave_data_u["rdm1"]

ham_data_noci = {}
ham_data_noci["h0"] = ham_data["h0"]
ham_data_noci["rot_h1"] = [
    jnp.array(np.random.rand(ndets, nelec_sp[0], norb)),
    jnp.array(np.random.rand(ndets, nelec_sp[1], norb)),
]
ham_data_noci["rot_chol"] = [
    jnp.array(np.random.rand(ndets, nchol, nelec_sp[0], norb)),
    jnp.array(np.random.rand(ndets, nchol, nelec_sp[1], norb)),
]

# -----------------------------------------------------------------------------
# GHF wavefunction.
nocc = sum(nelec_sp)
ghf = wavefunctions_v2.ghf(norb, nelec_sp)
ghf_legacy = wavefunctions.ghf(norb, nelec_sp)

wave_data_g = {}
wave_data_g["mo_coeff"] = jnp.array(np.random.rand(2 * norb, nocc))

_wave_data_g = {} # GHF wave_data from UHF.
_wave_data_g["mo_coeff"] = jsp.linalg.block_diag(*wave_data_u["mo_coeff"])

n_walkers = 10
walker_g = jsp.linalg.block_diag(walker_up, walker_dn)
walkers_g = jnp.array([walker_g] * n_walkers)

ham_data_g = {}
ham_data_g["h0"] = ham_data["h0"]
ham_data_g["rot_h1"] = jnp.array(np.random.rand(nocc, 2 * norb))
ham_data_g["rot_chol"] = jnp.array(np.random.rand(nchol, nocc, 2 * norb))
ham_data_g["h1"] = jnp.array([ham_data["h1"], ham_data["h1"]])
ham_data_g["chol"] = ham_data["chol"]
ham_data_g["ene0"] = ham_data["ene0"]

# -----------------------------------------------------------------------------
# UHF wavefunction for CPMC.
uhf_cpmc = wavefunctions_v2.uhf_cpmc(norb, nelec_sp) # nelec_sp = (3, 2) from above.
uhf_cpmc_legacy = wavefunctions.uhf_cpmc(norb, nelec_sp)

# -----------------------------------------------------------------------------
# GHF wavefunction for CPMC.
ghf_cpmc = wavefunctions_v2.ghf_cpmc(norb, nelec_sp) # nelec_sp = (3, 2) from above.
ghf_cpmc_legacy = wavefunctions.ghf_cpmc(norb, nelec_sp)

# -----------------------------------------------------------------------------
# GHF tests.
def test_ghf_overlap():
    overlap = ghf._calc_overlap(walker_up, walker_dn, wave_data_g)
    overlap_legacy = ghf_legacy._calc_overlap(walker_up, walker_dn, wave_data_g)
    np.testing.assert_allclose(jnp.real(overlap), -0.7645032356687913)
    np.testing.assert_allclose(jnp.real(overlap), jnp.real(overlap_legacy))


def test_ghf_overlap_restricted():
    overlap = ghf._calc_overlap_restricted(walker_g, wave_data_g)
    np.testing.assert_allclose(jnp.real(overlap), -0.7645032356687913)


def test_ghf_overlap_batch():
    overlaps = ghf.calc_overlap(walkers_u, wave_data_g)
    overlaps_legacy = ghf_legacy.calc_overlap(walkers_u, wave_data_g)
    np.testing.assert_allclose(jnp.real(overlaps), -0.7645032356687913)
    np.testing.assert_allclose(jnp.real(overlaps), jnp.real(overlaps_legacy))


def test_ghf_overlap_restricted_batch():
    overlaps = ghf.calc_overlap(walkers_g, wave_data_g)
    np.testing.assert_allclose(jnp.real(overlaps), -0.7645032356687913)


def test_ghf_green():
    green = ghf._calc_green(walker_up, walker_dn, wave_data_g)
    green_legacy = ghf_legacy._calc_green(walker_up, walker_dn, wave_data_g)

    overlap_mat = wave_data_g["mo_coeff"].T.conj() @ walker_g
    inv = jsp.linalg.inv(overlap_mat)
    green_ref = (walker_g @ inv).T

    assert green.shape == (nocc, 2 * norb)
    np.testing.assert_allclose(green, green_ref)
    np.testing.assert_allclose(green, green_legacy)


def test_ghf_green_restricted():
    green = ghf._calc_green_restricted(walker_g, wave_data_g)
    green_ref = ghf._calc_green(walker_up, walker_dn, wave_data_g)
    green_legacy = ghf_legacy._calc_green(walker_up, walker_dn, wave_data_g)

    assert green.shape == (nocc, 2 * norb)
    np.testing.assert_allclose(green, green_ref)
    np.testing.assert_allclose(green, green_legacy)


def test_ghf_force_bias():
    force_bias = ghf._calc_force_bias(walker_up, walker_dn, ham_data_g, wave_data_g)
    force_bias_legacy = ghf_legacy._calc_force_bias(walker_up, walker_dn, ham_data_g, wave_data_g)
    
    green = ghf._calc_green(walker_up, walker_dn, wave_data_g)
    force_bias_ref = np.einsum('gij,ij->g', ham_data_g["rot_chol"], green)
    
    assert force_bias.shape == (nchol,)
    np.testing.assert_allclose(force_bias, force_bias_ref)
    np.testing.assert_allclose(force_bias, force_bias_legacy)


def test_ghf_force_bias_restricted():
    force_bias = ghf._calc_force_bias_restricted(walker_g, ham_data_g, wave_data_g)
    force_bias_ref = ghf._calc_force_bias(walker_up, walker_dn, ham_data_g, wave_data_g)

    assert force_bias.shape == (nchol,)
    np.testing.assert_allclose(force_bias, force_bias_ref)


def test_ghf_force_bias_batch():
    force_bias = ghf.calc_force_bias(walkers_u, ham_data_g, wave_data_g)
    force_bias_legacy = ghf_legacy.calc_force_bias(walkers_u, ham_data_g, wave_data_g)
    
    assert force_bias.shape == (n_walkers, nchol,)
    np.testing.assert_allclose(force_bias, force_bias_legacy)

    green = ghf._calc_green(walker_up, walker_dn, wave_data_g)
    force_bias_ref = np.einsum('gij,ij->g', ham_data_g["rot_chol"], green)
    
    for iw in range(n_walkers):
        np.testing.assert_allclose(force_bias[iw], force_bias_ref)


def test_ghf_force_bias_restricted_batch():
    force_bias = ghf.calc_force_bias(walkers_g, ham_data_g, wave_data_g)
    force_bias_ref = ghf.calc_force_bias(walkers_u, ham_data_g, wave_data_g)

    assert force_bias.shape == (n_walkers, nchol,)
    np.testing.assert_allclose(force_bias, force_bias_ref)


def test_ghf_energy():
    energy = ghf._calc_energy(walker_up, walker_dn, ham_data_g, wave_data_g)
    energy_legacy = ghf_legacy._calc_energy(walker_up, walker_dn, ham_data_g, wave_data_g)
    
    np.testing.assert_allclose(jnp.real(energy), 47.91857449460195)
    np.testing.assert_allclose(jnp.real(energy), jnp.real(energy_legacy))


def test_ghf_energy_restricted():
    energy = ghf._calc_energy_restricted(walker_g, ham_data_g, wave_data_g)
    energy_ref = ghf._calc_energy(walker_up, walker_dn, ham_data_g, wave_data_g)
    
    np.testing.assert_allclose(jnp.real(energy), jnp.real(energy_ref))


def test_ghf_rdm1():
    rdm1 = ghf._calc_rdm1(wave_data_g)
    rdm1_legacy = ghf_legacy._calc_rdm1(wave_data_g)
    
    np.testing.assert_allclose(rdm1, rdm1_legacy)


def test_ghf_rdm1_restricted():
    rdm1 = ghf._calc_rdm1_restricted(wave_data_g)
    rdm1_ref = wave_data_g["mo_coeff"] @ wave_data_g["mo_coeff"].T.conj()
    
    np.testing.assert_allclose(rdm1, rdm1_ref)


# -----------------------------------------------------------------------------
# UHF-CPMC tests.
def test_uhf_cpmc_green_full():
    green = uhf_cpmc._calc_green_full(walker_up, walker_dn, wave_data_u)
    green_legacy = uhf_cpmc_legacy.calc_full_green(walker_up, walker_dn, wave_data_u)

    assert green[0].shape == (norb, norb)
    assert green[1].shape == (norb, norb)
    assert np.allclose(green, green_legacy)

def test_uhf_cpmc_green_full_batch():
    greens = uhf_cpmc.calc_green_full(walkers_u, wave_data_u)
    greens_legacy = uhf_cpmc_legacy.calc_full_green_vmap(walkers_u, wave_data_u)

    assert greens.shape == (n_walkers, 2, norb, norb)
    assert np.allclose(greens, greens_legacy)

def test_uhf_cpmc_overlap_ratio():
    u = 4.0
    dt = 0.005
    gamma = jnp.arccosh(jnp.exp(dt * u / 2))
    const = jnp.exp(-dt * u / 2)
    hs_constant = const * jnp.array(
        [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
    )

    walker_up = jnp.array(np.random.rand(norb, nelec_sp[0])) + \
                    1.0j * jnp.array(np.random.rand(norb, nelec_sp[0]))
    walker_dn = jnp.array(np.random.rand(norb, nelec_sp[1])) + \
                    1.0j * jnp.array(np.random.rand(norb, nelec_sp[1]))

    green = uhf_cpmc._calc_green_full(walker_up, walker_dn, wave_data_u)
    assert green[0].shape == (norb, norb)
    assert green[1].shape == (norb, norb)

    wick_ratio = uhf_cpmc._calc_overlap_ratio(
        green,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    wick_ratio_legacy = uhf_cpmc_legacy.calc_overlap_ratio(
        green,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )

    overlap_0 = uhf_cpmc._calc_overlap(walker_up, walker_dn, wave_data_u)
    new_walker_0 = walker_up.at[3, :].mul(hs_constant[0, 0])
    new_walker_1 = walker_dn.at[3, :].mul(hs_constant[0, 1])
    ratio = uhf_cpmc._calc_overlap(new_walker_0, new_walker_1, wave_data_u) / overlap_0

    np.testing.assert_allclose(ratio, wick_ratio)
    np.testing.assert_allclose(wick_ratio, wick_ratio_legacy)


def test_uhf_cpmc_overlap_ratio_batch():
    u = 4.0
    dt = 0.005
    gamma = jnp.arccosh(jnp.exp(dt * u / 2))
    const = jnp.exp(-dt * u / 2)
    hs_constant = const * jnp.array(
        [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
    )

    n_walkers = 10
    walkers_u = [jnp.array([walker_up] * n_walkers), jnp.array([walker_dn] * n_walkers)]
    
    greens = uhf_cpmc.calc_green_full(walkers_u, wave_data_u)
    assert greens.shape == (n_walkers, 2, norb, norb)

    wick_ratios = uhf_cpmc.calc_overlap_ratio(
        greens,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    wick_ratios_legacy = uhf_cpmc_legacy.calc_overlap_ratio_vmap(
        greens,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    np.testing.assert_allclose(wick_ratios, wick_ratios_legacy)

    overlaps_0 = uhf_cpmc.calc_overlap(walkers_u, wave_data_u)
    
    for iw in range(n_walkers):
        new_walker_0 = walkers_u[0][iw].at[3, :].mul(hs_constant[0, 0])
        new_walker_1 = walkers_u[1][iw].at[3, :].mul(hs_constant[0, 1])
        ratio = uhf_cpmc._calc_overlap(new_walker_0, new_walker_1, wave_data_u) / overlaps_0[iw]
        np.testing.assert_allclose(ratio, wick_ratios[iw])

    
def test_uhf_cpmc_update_green():
    u = 4.0
    dt = 0.005
    gamma = jnp.arccosh(jnp.exp(dt * u / 2))
    const = jnp.exp(-dt * u / 2)
    hs_constant = const * jnp.array(
        [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
    )

    walker_up = jnp.array(np.random.rand(norb, nelec_sp[0])) + \
                    1.0j * jnp.array(np.random.rand(norb, nelec_sp[0]))
    walker_dn = jnp.array(np.random.rand(norb, nelec_sp[1])) + \
                    1.0j * jnp.array(np.random.rand(norb, nelec_sp[1]))

    green = uhf_cpmc._calc_green_full(walker_up, walker_dn, wave_data_u)
    assert green[0].shape == (norb, norb)
    assert green[1].shape == (norb, norb)

    new_walker_0 = walker_up.at[3, :].mul(hs_constant[0, 0])
    new_walker_1 = walker_dn.at[3, :].mul(hs_constant[0, 1])
    
    ratio = uhf_cpmc._calc_overlap_ratio(
        green,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    
    new_green = uhf_cpmc._calc_green_full(new_walker_0, new_walker_1, wave_data_u)
    new_green_wick = uhf_cpmc._update_green(
        green,
        ratio,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    new_green_wick_legacy = uhf_cpmc_legacy.update_greens_function(
        green,
        ratio,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )

    assert np.allclose(new_green, new_green_wick)
    assert np.allclose(new_green_wick, new_green_wick_legacy)


def test_uhf_cpmc_update_green_batch():
    u = 4.0
    dt = 0.005
    gamma = jnp.arccosh(jnp.exp(dt * u / 2))
    const = jnp.exp(-dt * u / 2)
    hs_constant = const * jnp.array(
        [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
    )
    n_walkers = 10
    walkers_u = [jnp.array([walker_up] * n_walkers), jnp.array([walker_dn] * n_walkers)]
    update_constants = jnp.array([hs_constant[0] - 1] * n_walkers)

    greens = uhf_cpmc.calc_green_full(walkers_u, wave_data_u)
    assert greens.shape == (n_walkers, 2, norb, norb)
    
    ratios = uhf_cpmc.calc_overlap_ratio(
        greens,
        jnp.array([[0, 3], [1, 3]]),
        jnp.array(hs_constant[0] - 1),
    )
    
    new_walkers_0 = []
    new_walkers_1 = []

    for iw in range(n_walkers):
        new_walker_0 = walkers_u[0][iw].at[3, :].mul(hs_constant[0, 0])
        new_walker_1 = walkers_u[1][iw].at[3, :].mul(hs_constant[0, 1])
        new_walkers_0.append(new_walker_0)
        new_walkers_1.append(new_walker_1)

    new_walkers = [jnp.array(new_walkers_0), jnp.array(new_walkers_1)]
    new_greens = uhf_cpmc.calc_green_full(new_walkers, wave_data_u)
    
    new_greens_wick = uhf_cpmc.update_green(
        greens,
        ratios,
        jnp.array([[0, 3], [1, 3]]),
        update_constants,
    )
    new_greens_wick_legacy = uhf_cpmc_legacy.update_greens_function_vmap(
        greens,
        ratios,
        jnp.array([[0, 3], [1, 3]]),
        update_constants,
    )
    
    np.testing.assert_allclose(new_greens, new_greens_wick)
    np.testing.assert_allclose(new_greens, new_greens_wick_legacy)


# -----------------------------------------------------------------------------
# GHF-CPMC tests.
def test_ghf_cpmc_green_full():
    green = ghf_cpmc._calc_green_full(walker_up, walker_dn, _wave_data_g)
    green_ref = jsp.linalg.block_diag(
                    *uhf_cpmc._calc_green_full(walker_up, walker_dn, wave_data_u)
                )

    assert green.shape == (2*norb, 2*norb)
    np.testing.assert_allclose(green, green_ref)


def test_ghf_cpmc_green_full_batch():
    greens = ghf_cpmc.calc_green_full(walkers_u, _wave_data_g)
    greens_ref = uhf_cpmc.calc_green_full(walkers_u, wave_data_u)
    assert greens.shape == (n_walkers, 2*norb, 2*norb)

    for iw in range(n_walkers):
        np.testing.assert_allclose(
                greens[iw], jsp.linalg.block_diag(*greens_ref[iw]))


def test_ghf_cpmc_green_full_restricted():
    green = ghf_cpmc._calc_green_full_restricted(walker_g, _wave_data_g)
    green_ref = jsp.linalg.block_diag(
                    *uhf_cpmc._calc_green_full(walker_up, walker_dn, wave_data_u)
                )

    assert green.shape == (2*norb, 2*norb)
    np.testing.assert_allclose(green, green_ref)


def test_ghf_cpmc_green_full_restricted_batch():
    greens = ghf_cpmc.calc_green_full(walkers_g, _wave_data_g)
    greens_ref = uhf_cpmc.calc_green_full(walkers_u, wave_data_u)
    assert greens.shape == (n_walkers, 2*norb, 2*norb)

    for iw in range(n_walkers):
        np.testing.assert_allclose(
                greens[iw], jsp.linalg.block_diag(*greens_ref[iw]))


def test_ghf_cpmc_overlap_ratio():
    u = 4.0
    dt = 0.005
    gamma = jnp.arccosh(jnp.exp(dt * u / 2))
    const = jnp.exp(-dt * u / 2)
    hs_constant = const * jnp.array(
        [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
    )
    walker_g = jsp.linalg.block_diag(walker_up, walker_dn)
    
    # Test 1: GHF from UHF.
    green = ghf_cpmc._calc_green_full_restricted(walker_g, _wave_data_g)
    _green_ref = uhf_cpmc._calc_green_full(walker_up, walker_dn, wave_data_u)
    green_ref = jsp.linalg.block_diag(*_green_ref)
    assert green.shape == (2*norb, 2*norb)
    np.testing.assert_allclose(green, green_ref)

    wick_ratio = ghf_cpmc._calc_overlap_ratio(
        green,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    wick_ratio_ref = uhf_cpmc._calc_overlap_ratio(
        _green_ref,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    np.testing.assert_allclose(wick_ratio, wick_ratio_ref)

    # Test 2: GHF
    green = ghf_cpmc._calc_green_full_restricted(walker_g, wave_data_g)

    wick_ratio = ghf_cpmc._calc_overlap_ratio(
        green,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )

    overlap_0 = ghf_cpmc._calc_overlap_restricted(walker_g, wave_data_g)
    new_walker = walker_g.at[3, :].mul(hs_constant[0, 0])
    new_walker = new_walker.at[norb + 3, :].mul(hs_constant[0, 1])
    ratio = ghf_cpmc._calc_overlap_restricted(new_walker, wave_data_g) / overlap_0

    np.testing.assert_allclose(ratio, wick_ratio)


def test_ghf_cpmc_overlap_ratio_batch():
    u = 4.0
    dt = 0.005
    gamma = jnp.arccosh(jnp.exp(dt * u / 2))
    const = jnp.exp(-dt * u / 2)
    hs_constant = const * jnp.array(
        [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
    )
    n_walkers = 10
    walkers_g = jnp.array([walker_g] * n_walkers)
    
    # Test 1: GHF from UHF.
    greens = ghf_cpmc.calc_green_full(walkers_g, _wave_data_g)
    _greens_ref = uhf_cpmc.calc_green_full(walkers_u, wave_data_u)
    greens_ref = jnp.array(
        [jsp.linalg.block_diag(*_green_ref) for _green_ref in _greens_ref]
    )
    assert greens.shape == (n_walkers, 2*norb, 2*norb)
    np.testing.assert_allclose(greens, greens_ref)
    
    wick_ratios = ghf_cpmc.calc_overlap_ratio(
        greens,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    wick_ratios_ref = uhf_cpmc.calc_overlap_ratio(
        _greens_ref,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    np.testing.assert_allclose(wick_ratios, wick_ratios_ref)

    # Test 2: GHF
    greens = ghf_cpmc.calc_green_full(walkers_g, wave_data_g)

    wick_ratios = ghf_cpmc.calc_overlap_ratio(
        greens,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )

    overlaps_0 = ghf_cpmc.calc_overlap(walkers_g, wave_data_g)

    for iw in range(n_walkers):
        new_walker = walkers_g[iw].at[3, :].mul(hs_constant[0, 0])
        new_walker = new_walker.at[norb + 3, :].mul(hs_constant[0, 1])
        ratio = ghf_cpmc._calc_overlap_restricted(new_walker, wave_data_g) / overlaps_0[iw]
        np.testing.assert_allclose(ratio, wick_ratios[iw])


def test_ghf_cpmc_update_green():
    u = 4.0
    dt = 0.005
    gamma = jnp.arccosh(jnp.exp(dt * u / 2))
    const = jnp.exp(-dt * u / 2)
    hs_constant = const * jnp.array(
        [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
    )
    walker_g = jsp.linalg.block_diag(walker_up, walker_dn)
    
    # Test 1: GHF from UHF.
    green = ghf_cpmc._calc_green_full_restricted(walker_g, _wave_data_g)
    assert green.shape == (2*norb, 2*norb)
    
    ratio = ghf_cpmc._calc_overlap_ratio(
        green,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    
    new_green_wick = ghf_cpmc._update_green(
        green,
        ratio,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )

    green_legacy = uhf_cpmc._calc_green_full(walker_up, walker_dn, wave_data_u)
    _new_green_wick_legacy = uhf_cpmc_legacy.update_greens_function(
        green_legacy,
        ratio,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    new_green_wick_legacy = jsp.linalg.block_diag(*_new_green_wick_legacy)
    np.testing.assert_allclose(new_green_wick, new_green_wick_legacy)

    # Test 2: GHF.
    green = ghf_cpmc._calc_green_full_restricted(walker_g, wave_data_g)

    new_walker = walker_g.at[3, :].mul(hs_constant[0, 0])
    new_walker = new_walker.at[norb + 3, :].mul(hs_constant[0, 1])
    
    ratio = ghf_cpmc._calc_overlap_ratio(
        green,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    
    new_green = ghf_cpmc._calc_green_full_restricted(new_walker, wave_data_g)

    new_green_wick = ghf_cpmc._update_green(
        green,
        ratio,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    np.testing.assert_allclose(new_green, new_green_wick)


def test_ghf_cpmc_update_green_batch():
    u = 4.0
    dt = 0.005
    gamma = jnp.arccosh(jnp.exp(dt * u / 2))
    const = jnp.exp(-dt * u / 2)
    hs_constant = const * jnp.array(
        [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
    )
    n_walkers = 10
    walkers_g = jnp.array([walker_g] * n_walkers)
    update_constants = jnp.array([hs_constant[0] - 1] * n_walkers)

    # Test 1: GHF from UHF.
    greens = ghf_cpmc.calc_green_full(walkers_g, _wave_data_g)
    assert greens.shape == (n_walkers, 2*norb, 2*norb)
    
    ratios = ghf_cpmc.calc_overlap_ratio(
        greens,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    
    new_greens_wick = ghf_cpmc.update_green(
        greens,
        ratios,
        jnp.array([[0, 3], [1, 3]]),
        update_constants,
    )

    greens_legacy = uhf_cpmc.calc_green_full(walkers_u, wave_data_u)
    _new_greens_wick_legacy = uhf_cpmc_legacy.update_greens_function_vmap(
        greens_legacy,
        ratios,
        jnp.array([[0, 3], [1, 3]]),
        update_constants,
    )

    for iw in range(n_walkers):
        new_green_wick_legacy = jsp.linalg.block_diag(*_new_greens_wick_legacy[iw])
        np.testing.assert_allclose(new_greens_wick[iw], new_green_wick_legacy)

    # Test 2: GHF.
    greens = ghf_cpmc.calc_green_full(walkers_g, wave_data_g)
    ratios = ghf_cpmc.calc_overlap_ratio(
        greens,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    new_greens_wick = ghf_cpmc.update_green(
        greens,
        ratios,
        jnp.array([[0, 3], [1, 3]]),
        update_constants,
    )
    
    for iw in range(n_walkers):
        new_walker = walkers_g[iw].at[3, :].mul(hs_constant[0, 0])
        new_walker = new_walker.at[norb + 3, :].mul(hs_constant[0, 1])
        new_green = ghf_cpmc._calc_green_full_restricted(new_walker, wave_data_g)
        np.testing.assert_allclose(new_green, new_greens_wick[iw])


if __name__ == "__main__":
    test_ghf_overlap()
    test_ghf_green()
    test_ghf_force_bias()
    test_ghf_energy()
    test_ghf_rdm1()

    test_ghf_overlap_restricted()
    test_ghf_green_restricted()
    test_ghf_force_bias_restricted()
    test_ghf_energy_restricted()
    test_ghf_rdm1_restricted()

    test_ghf_overlap_batch()
    test_ghf_overlap_restricted_batch()
    test_ghf_force_bias_batch()
    test_ghf_force_bias_restricted_batch()

    test_uhf_cpmc_green_full()
    test_uhf_cpmc_overlap_ratio()
    test_uhf_cpmc_update_green()
    test_uhf_cpmc_green_full_batch()
    test_uhf_cpmc_overlap_ratio_batch()
    test_uhf_cpmc_update_green_batch()

    test_ghf_cpmc_green_full()
    test_ghf_cpmc_green_full_restricted()
    test_ghf_cpmc_overlap_ratio()
    test_ghf_cpmc_update_green()

    test_ghf_cpmc_green_full_batch()
    test_ghf_cpmc_green_full_restricted_batch()
    test_ghf_cpmc_overlap_ratio_batch()
    test_ghf_cpmc_update_green_batch()
