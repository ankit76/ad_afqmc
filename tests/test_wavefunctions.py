import os

import numpy as np

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import numpy as jnp

from ad_afqmc import wavefunctions

seed = 102
np.random.seed(seed)
norb, nelec, nchol = 10, 5, 5
rhf = wavefunctions.rhf(norb, nelec)
walker = jnp.array(np.random.rand(norb, nelec)) + 1.0j * jnp.array(
    np.random.rand(norb, nelec)
)
ham_data = {}
ham_data["h0"] = np.random.rand(
    1,
)[0]
ham_data["rot_h1"] = jnp.array(np.random.rand(nelec, norb))
ham_data["rot_chol"] = jnp.array(np.random.rand(nchol, nelec, norb))
ham_data["h1"] = jnp.array(np.random.rand(norb, norb))
ham_data["chol"] = jnp.array(np.random.rand(nchol, norb, norb))
ham_data["ene0"] = 0.0

nelec_sp = (3, 2)
uhf = wavefunctions.uhf(norb, nelec_sp)
wave_data = [
    jnp.array(np.random.rand(norb, nelec_sp[0])),
    jnp.array(np.random.rand(norb, nelec_sp[1])),
]
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
wave_data_noci = [ci_coeffs, dets]
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
wave_data_g = jnp.array(np.random.rand(2 * norb, nelec_sp[0] + nelec_sp[1]))
ham_data_g = {}
ham_data_g["h0"] = ham_data["h0"]
ham_data_g["rot_h1"] = jnp.array(np.random.rand(nelec_sp[0] + nelec_sp[1], 2 * norb))
ham_data_g["rot_chol"] = jnp.array(
    np.random.rand(nchol, nelec_sp[0] + nelec_sp[1], 2 * norb)
)
ham_data_g["h1"] = jnp.array([ham_data["h1"], ham_data["h1"]])
ham_data_g["chol"] = ham_data["chol"]
ham_data_g["ene0"] = ham_data["ene0"]


def test_rhf_overlap():
    overlap = rhf.calc_overlap(walker)
    assert np.allclose(jnp.real(overlap), -0.10794844182417201)


def test_rhf_green():
    green = rhf.calc_green(walker)
    assert green.shape == (nelec, norb)
    assert np.allclose(jnp.real(jnp.sum(green)), 12.181348093111438)


def test_rhf_force_bias():
    force_bias = rhf.calc_force_bias(walker, ham_data["rot_chol"])
    assert force_bias.shape == (nchol,)
    assert np.allclose(jnp.real(jnp.sum(force_bias)), 66.13455680423321)


def test_rhf_energy():
    energy = rhf.calc_energy(
        ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"], walker
    )
    assert np.allclose(jnp.real(energy), 217.79874063608622)


def test_rhf_optimize_orbs():
    orbs = rhf.optimize_orbs(ham_data)
    assert orbs.shape == (norb, norb)
    assert np.allclose(jnp.sum(orbs), 2.9662577668717933)


def test_uhf_overlap():
    overlap = uhf.calc_overlap(walker_up, walker_dn, wave_data)
    assert np.allclose(jnp.real(overlap), -0.4029825074695857)

def test_uhf_update_inv_overlap_mat():
    # Update matrix.
    site = np.random.randint(0, min(nelec_sp))
    dB_diag = jnp.zeros(norb)
    dB_diag = dB_diag.at[site].set(1)
    dB = jnp.diag(dB_diag)
    B = jnp.eye(norb) + dB
    print(f'Updating site {site}')
    
    # Current inverse overlap matrix.
    inv_overlap_mat = uhf.calc_inv_overlap_mat(walker_up, walker_dn, wave_data)

    # Update.
    walker_up_new = jnp.einsum("ij,jk->ik", B, walker_up)
    walker_dn_new = jnp.einsum("ij,jk->ik", B, walker_dn)

    dwalker_up = walker_up_new - walker_up
    dwalker_dn = walker_dn_new - walker_dn

    # Check that only 1 row is changed.
    assert np.count_nonzero(np.linalg.norm(dwalker_up, axis=1)) == 1
    assert np.count_nonzero(np.linalg.norm(dwalker_dn, axis=1)) == 1

    # New inverse overlap matrix.
    ref_inv_overlap_mat_new = uhf.calc_inv_overlap_mat(walker_up_new, walker_dn_new, wave_data)
    
    # Test.
    inv_overlap_mat_new = uhf.update_inv_overlap_mat(walker_up, walker_dn, wave_data, inv_overlap_mat, (1, 1), site)

    np.testing.assert_equal(inv_overlap_mat_new[0].shape, ref_inv_overlap_mat_new[0].shape)
    np.testing.assert_equal(inv_overlap_mat_new[1].shape, ref_inv_overlap_mat_new[1].shape)
    np.testing.assert_allclose(inv_overlap_mat_new[0], ref_inv_overlap_mat_new[0], atol=1e-15)
    np.testing.assert_allclose(inv_overlap_mat_new[1], ref_inv_overlap_mat_new[1], atol=1e-15)

def test_uhf_update_overlap():
    # Update matrix.
    site = np.random.randint(0, min(nelec_sp))
    dB_diag = jnp.zeros(norb)
    dB_diag = dB_diag.at[site].set(1)
    dB = jnp.diag(dB_diag)
    B = jnp.eye(norb) + dB
    print(f'Updating site {site}')
    
    # Current overlap.
    overlap = uhf.calc_overlap(walker_up, walker_dn, wave_data)

    # Current inverse overlap matrix.
    inv_overlap_mat = uhf.calc_inv_overlap_mat(walker_up, walker_dn, wave_data)

    # Update.
    walker_up_new = jnp.einsum("ij,jk->ik", B, walker_up)
    walker_dn_new = jnp.einsum("ij,jk->ik", B, walker_dn)

    dwalker_up = walker_up_new - walker_up
    dwalker_dn = walker_dn_new - walker_dn

    # Check that only 1 row is changed.
    assert np.count_nonzero(np.linalg.norm(dwalker_up, axis=1)) == 1
    assert np.count_nonzero(np.linalg.norm(dwalker_dn, axis=1)) == 1

    # New overlap.
    ref_overlap_new = uhf.calc_overlap(walker_up_new, walker_dn_new, wave_data)
    
    # Test.
    overlap_new = uhf.update_overlap(walker_up, walker_dn, wave_data, inv_overlap_mat, overlap, (1, 1), site)

    np.testing.assert_allclose(overlap_new, ref_overlap_new, atol=1e-15)


def test_uhf_green():
    green = uhf.calc_green(walker_up, walker_dn, wave_data)
    assert green[0].shape == (nelec_sp[0], norb)
    assert green[1].shape == (nelec_sp[1], norb)
    assert np.allclose(
        jnp.real(jnp.sum(green[0]) + jnp.sum(green[1])), 3.6324117896217394
    )


def test_uhf_force_bias():
    force_bias = uhf.calc_force_bias(
        walker_up, walker_dn, ham_data_u["rot_chol"], wave_data
    )
    assert force_bias.shape == (nchol,)
    assert np.allclose(jnp.real(jnp.sum(force_bias)), 10.441272099672341)


def test_uhf_energy():
    energy = uhf.calc_energy(
        ham_data_u["h0"],
        ham_data_u["rot_h1"],
        ham_data_u["rot_chol"],
        walker_up,
        walker_dn,
        wave_data,
    )
    assert np.allclose(jnp.real(energy), -1.7203463308366032)


def test_uhf_optimize_orbs():
    orbs = uhf.optimize_orbs(ham_data_u, wave_data)
    assert orbs[0].shape == (norb, norb)
    assert orbs[1].shape == (norb, norb)
    assert np.allclose(jnp.sum(orbs[0]) + jnp.sum(orbs[1]), 7.402931219898609)


def test_ghf_overlap():
    overlap = ghf.calc_overlap(walker_up, walker_dn, wave_data_g)
    print("overlap: ", overlap)
    assert np.allclose(jnp.real(overlap), -0.7645032356687913)


def test_ghf_green():
    green = ghf.calc_green(walker_up, walker_dn, wave_data_g)
    assert green.shape == (nelec_sp[0] + nelec_sp[1], 2 * norb)


def test_ghf_force_bias():
    force_bias = ghf.calc_force_bias(
        walker_up, walker_dn, ham_data_g["rot_chol"], wave_data_g
    )
    assert force_bias.shape == (nchol,)


def test_ghf_energy():
    energy = ghf.calc_energy(
        ham_data_g["h0"],
        ham_data_g["rot_h1"],
        ham_data_g["rot_chol"],
        walker_up,
        walker_dn,
        wave_data_g,
    )
    print("energy: ", energy)
    assert np.allclose(jnp.real(energy), 47.91857449460195)


def test_noci_overlap():
    overlap = noci.calc_overlap(walker_up, walker_dn, wave_data_noci)
    # print(overlap)
    # assert np.allclose(jnp.real(overlap), -0.4029825074695857)


def test_noci_green():
    green_0, green_1, _ = noci.calc_green(walker_up, walker_dn, wave_data_noci)
    assert green_0.shape == (ndets, nelec_sp[0], norb)
    assert green_1.shape == (ndets, nelec_sp[1], norb)
    # assert np.allclose(jnp.real(jnp.sum(green[0])+jnp.sum(green[1])), 3.6324117896217394)


def test_noci_force_bias():
    force_bias = noci.calc_force_bias(
        walker_up, walker_dn, ham_data_noci["rot_chol"], wave_data_noci
    )
    assert force_bias.shape == (nchol,)
    # assert np.allclose(jnp.real(jnp.sum(force_bias)), 10.441272099672341)


def test_noci_energy():
    energy = noci.calc_energy(
        ham_data_noci["h0"],
        ham_data_noci["rot_h1"],
        ham_data_noci["rot_chol"],
        walker_up,
        walker_dn,
        wave_data_noci,
    )
    # assert np.allclose(jnp.real(energy), -1.7203463308366032)


def test_noci_get_rdm1():
    rdm1 = noci.get_rdm1(wave_data_noci)
    assert rdm1.shape == (norb, norb)


if __name__ == "__main__":
    test_rhf_overlap()
    test_rhf_green()
    test_rhf_force_bias()
    test_rhf_energy()
    test_rhf_optimize_orbs()
    test_uhf_overlap()
    test_uhf_update_inv_overlap_mat()
    test_uhf_update_overlap()
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
