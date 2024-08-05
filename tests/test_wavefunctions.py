import os

import numpy as np

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import numpy as jnp

from ad_afqmc import pyscf_interface, wavefunctions

seed = 102
np.random.seed(seed)
norb, nelec, nchol = 10, (5, 5), 5
rhf = wavefunctions.rhf(norb, nelec)
wave_data = {}
wave_data["mo_coeff"] = jnp.eye(norb)[:, : nelec[0]]
wave_data["rdm1"] = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
walker = jnp.array(np.random.rand(norb, nelec[0])) + 1.0j * jnp.array(
    np.random.rand(norb, nelec[0])
)
ham_data = {}
ham_data["h0"] = np.random.rand(
    1,
)[0]
ham_data["rot_h1"] = jnp.array(np.random.rand(nelec[0], norb))
ham_data["rot_chol"] = jnp.array(np.random.rand(nchol, nelec[0], norb))
ham_data["h1"] = jnp.array(np.random.rand(norb, norb))
ham_data["chol"] = jnp.array(np.random.rand(nchol, norb, norb))
ham_data["ene0"] = 0.0


ham_data["normal_ordering_term"] = -0.5 * jnp.einsum(
    "gik,gjk->ij",
    ham_data["chol"].reshape(-1, norb, norb),
    ham_data["chol"].reshape(-1, norb, norb),
    optimize="optimal",
)

multislater = wavefunctions.multislater(norb, nelec, max_excitation=6)
path = os.path.dirname(os.path.abspath(__file__))
Acre, Ades, Bcre, Bdes, coeff, ref_det = pyscf_interface.get_excitations(
    fname=path + "/dets.bin", max_excitation=6, ndets=10
)  # reads dets.bin
ref_det = jnp.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 2])
wave_data_multislater = {
    "Acre": Acre,
    "Ades": Ades,
    "Bcre": Bcre,
    "Bdes": Bdes,
    "coeff": coeff,
    "ref_det": ref_det,
}

nelec_sp = (3, 2)
uhf = wavefunctions.uhf(norb, nelec_sp)
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
walker_up, walker_dn = jnp.array(np.random.rand(norb, nelec_sp[0])) + 1.0j * jnp.array(
    np.random.rand(norb, nelec_sp[0])
), jnp.array(np.random.rand(norb, nelec_sp[1])) + 1.0j * jnp.array(
    np.random.rand(norb, nelec_sp[1])
)
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
ham_data_u["h1"] = jnp.array([ham_data["h1"], ham_data["h1"]])
ham_data_u["chol"] = ham_data["chol"]
ham_data_u["ene0"] = ham_data["ene0"]

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


nelec_sp = (3, 2)
ghf = wavefunctions.ghf(norb, nelec_sp)
wave_data_g = {}
wave_data_g["mo_coeff"] = jnp.array(np.random.rand(2 * norb, nelec_sp[0] + nelec_sp[1]))
wave_data_g["rdm1"] = wave_data_u["rdm1"]
ham_data_g = {}
ham_data_g["h0"] = ham_data["h0"]
ham_data_g["rot_h1"] = jnp.array(np.random.rand(nelec_sp[0] + nelec_sp[1], 2 * norb))
ham_data_g["rot_chol"] = jnp.array(
    np.random.rand(nchol, nelec_sp[0] + nelec_sp[1], 2 * norb)
)
ham_data_g["h1"] = jnp.array([ham_data["h1"], ham_data["h1"]])
ham_data_g["chol"] = ham_data["chol"]
ham_data_g["ene0"] = ham_data["ene0"]

uhf_cpmc = wavefunctions.uhf_cpmc(norb, nelec_sp)


def test_rhf_overlap():
    overlap = rhf._calc_overlap_restricted(walker, wave_data)
    assert np.allclose(jnp.real(overlap), -0.10794844182417201)


def test_rhf_green():
    green = rhf._calc_green(walker, wave_data)
    assert green.shape == (nelec[0], norb)
    assert np.allclose(jnp.real(jnp.sum(green)), 12.181348093111438)


def test_rhf_force_bias():
    force_bias = rhf._calc_force_bias_restricted(walker, ham_data, wave_data)
    assert force_bias.shape == (nchol,)
    assert np.allclose(jnp.real(jnp.sum(force_bias)), 66.13455680423321)


def test_rhf_energy():
    energy = rhf._calc_energy_restricted(walker, ham_data, wave_data)
    assert np.allclose(jnp.real(energy), 217.79874063608622)


def test_rhf_optimize_orbs():
    wave_data_opt = rhf.optimize(ham_data, wave_data)
    orbs = wave_data_opt["mo_coeff"]
    assert orbs.shape == (norb, norb)
    assert np.allclose(jnp.sum(orbs), 2.9662577668717933)


def test_uhf_overlap():
    overlap = uhf._calc_overlap(walker_up, walker_dn, wave_data_u)
    assert np.allclose(jnp.real(overlap), -0.4029825074695857)


def test_uhf_green():
    green = uhf._calc_green(walker_up, walker_dn, wave_data_u)
    assert green[0].shape == (nelec_sp[0], norb)
    assert green[1].shape == (nelec_sp[1], norb)
    assert np.allclose(
        jnp.real(jnp.sum(green[0]) + jnp.sum(green[1])), 3.6324117896217394
    )


def test_uhf_force_bias():
    force_bias = uhf._calc_force_bias(walker_up, walker_dn, ham_data_u, wave_data_u)
    assert force_bias.shape == (nchol,)
    assert np.allclose(jnp.real(jnp.sum(force_bias)), 10.441272099672341)


def test_uhf_energy():
    energy = uhf._calc_energy(
        walker_up,
        walker_dn,
        ham_data_u,
        wave_data_u,
    )
    assert np.allclose(jnp.real(energy), -1.7203463308366032)


def test_uhf_optimize_orbs():
    wave_data_opt = uhf.optimize(ham_data_u, wave_data_u)
    orbs = wave_data_opt["mo_coeff"]
    assert orbs[0].shape == (norb, norb)
    assert orbs[1].shape == (norb, norb)
    assert np.allclose(jnp.sum(orbs[0]) + jnp.sum(orbs[1]), 7.402931219898609)


def test_ghf_overlap():
    overlap = ghf._calc_overlap(walker_up, walker_dn, wave_data_g)
    assert np.allclose(jnp.real(overlap), -0.7645032356687913)


def test_ghf_green():
    green = ghf._calc_green(walker_up, walker_dn, wave_data_g)
    assert green.shape == (nelec_sp[0] + nelec_sp[1], 2 * norb)


def test_ghf_force_bias():
    force_bias = ghf._calc_force_bias(walker_up, walker_dn, ham_data_g, wave_data_g)
    assert force_bias.shape == (nchol,)


def test_ghf_energy():
    energy = ghf._calc_energy(
        walker_up,
        walker_dn,
        ham_data_g,
        wave_data_g,
    )
    assert np.allclose(jnp.real(energy), 47.91857449460195)


def test_noci_overlap():
    overlap = noci._calc_overlap(walker_up, walker_dn, wave_data_noci)
    # print(overlap)
    # assert np.allclose(jnp.real(overlap), -0.4029825074695857)


def test_noci_green():
    green_0, green_1, _ = noci._calc_green(walker_up, walker_dn, wave_data_noci)
    assert green_0.shape == (ndets, nelec_sp[0], norb)
    assert green_1.shape == (ndets, nelec_sp[1], norb)
    # assert np.allclose(jnp.real(jnp.sum(green[0])+jnp.sum(green[1])), 3.6324117896217394)


def test_noci_force_bias():
    force_bias = noci._calc_force_bias(
        walker_up, walker_dn, ham_data_noci, wave_data_noci
    )
    assert force_bias.shape == (nchol,)
    # assert np.allclose(jnp.real(jnp.sum(force_bias)), 10.441272099672341)


def test_noci_energy():
    energy = noci._calc_energy(
        walker_up,
        walker_dn,
        ham_data_noci,
        wave_data_noci,
    )
    # assert np.allclose(jnp.real(energy), -1.7203463308366032)


def test_noci_get_rdm1():
    rdm1 = noci.get_rdm1(wave_data_noci)
    assert rdm1.shape == (2, norb, norb)


def test_uhf_cpmc():
    u = 4.0
    dt = 0.005
    gamma = jnp.arccosh(jnp.exp(dt * u / 2))
    const = jnp.exp(-dt * u / 2)
    hs_constant = const * jnp.array(
        [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
    )
    green = uhf_cpmc.calc_full_green(walker_up, walker_dn, wave_data_u)
    assert green[0].shape == (norb, norb)
    assert green[1].shape == (norb, norb)
    wick_ratio = uhf_cpmc.calc_overlap_ratio(
        green,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    overlap_0 = uhf_cpmc._calc_overlap(walker_up, walker_dn, wave_data_u)
    new_walker_0 = walker_up.at[3, :].mul(hs_constant[0, 0])
    new_walker_1 = walker_dn.at[3, :].mul(hs_constant[0, 1])
    ratio = uhf_cpmc._calc_overlap(new_walker_0, new_walker_1, wave_data_u) / overlap_0
    assert np.allclose(ratio, wick_ratio)

    new_green = uhf_cpmc.calc_full_green(new_walker_0, new_walker_1, wave_data_u)
    new_green_wick = uhf_cpmc.update_greens_function(
        green,
        ratio,
        jnp.array([[0, 3], [1, 3]]),
        hs_constant[0] - 1,
    )
    assert np.allclose(new_green, new_green_wick)


def test_multislater():
    # slow test due to compilation time for fb and energy
    overlap = multislater._calc_overlap_restricted(walker, wave_data_multislater)
    assert np.allclose(jnp.real(overlap), -0.10468995287669804)

    green = multislater._calc_green_restricted(walker, wave_data_multislater)
    assert green.shape == (nelec[0], norb)
    assert np.allclose(jnp.real(jnp.sum(green)), 12.181348093111438)

    force_bias = multislater._calc_force_bias_restricted(
        walker, ham_data, wave_data_multislater
    )
    assert force_bias.shape == (nchol,)
    assert np.allclose(jnp.real(jnp.sum(force_bias)), 63.76752581694084)

    energy = multislater._calc_energy_restricted(
        walker, ham_data, wave_data_multislater
    )
    assert np.allclose(jnp.real(energy), 204.41045113133066)


if __name__ == "__main__":
    test_rhf_overlap()
    test_rhf_green()
    test_rhf_force_bias()
    test_rhf_energy()
    test_rhf_optimize_orbs()
    test_uhf_overlap()
    test_uhf_green()
    test_uhf_force_bias()
    test_uhf_energy()
    test_uhf_optimize_orbs()
    test_ghf_overlap()
    test_ghf_green()
    test_ghf_force_bias()
    test_ghf_energy()
    test_noci_overlap()
    test_noci_green()
    test_noci_force_bias()
    test_noci_energy()
    test_noci_get_rdm1()
    test_uhf_cpmc()
    test_multislater()
