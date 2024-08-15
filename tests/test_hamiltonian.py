import numpy as np

from ad_afqmc import config

config.setup_jax()
from jax import numpy as jnp

from ad_afqmc import hamiltonian, propagation, wavefunctions

seed = 102
np.random.seed(seed)
norb, nelec, nchol = 10, (5, 5), 5
ham_handler = hamiltonian.hamiltonian(norb)
trial = wavefunctions.rhf(norb, nelec)
ham_data = {}
ham_data["h0"] = np.random.rand(
    1,
)[0]
ham_data["h1"] = jnp.array(np.random.rand(norb, norb))
ham_data["h1"] = jnp.array([ham_data["h1"], ham_data["h1"]])
ham_data["chol"] = jnp.array(np.random.rand(nchol, norb * norb))
ham_data["ene0"] = 0.0
wave_data = {}
wave_data["mo_coeff"] = jnp.eye(norb)[:, : nelec[0]]
wave_data["rdm1"] = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
mo_coeff = jnp.array(np.random.rand(norb, norb))
prop_r = propagation.propagator_restricted(dt=0.005)

nelec_sp = 5, 4
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
trial_u = wavefunctions.uhf(norb, nelec_sp)
ham_data_u = {}
ham_data_u["h0"] = np.random.rand(
    1,
)[0]
ham_data_u["h1"] = jnp.array(np.random.rand(2, norb, norb))
ham_data_u["chol"] = jnp.array(np.random.rand(nchol, norb * norb))
ham_data_u["ene0"] = 0.0
prop_u = propagation.propagator_unrestricted(dt=0.005)

ndets = 5
ci_coeffs = jnp.array(np.random.randn(ndets))
dets = [
    jnp.array(np.random.rand(ndets, norb, nelec_sp[0])),
    jnp.array(np.random.rand(ndets, norb, nelec_sp[1])),
]
wave_data_noci = {}
wave_data_noci["ci_coeffs_dets"] = [ci_coeffs, dets]
wave_data_noci["rdm1"] = wave_data_u["rdm1"]
trial_noci = wavefunctions.noci(norb, nelec_sp, ndets)

nelec_sp = 5, 4
wave_data_g = {}
wave_data_g["mo_coeff"] = jnp.array(np.random.rand(2 * norb, nelec_sp[0] + nelec_sp[1]))
wave_data_g["rdm1"] = wave_data_u["rdm1"]
trial_g = wavefunctions.ghf(norb, nelec_sp)
ham_data_g = {}
ham_data_g["h0"] = np.random.rand(
    1,
)[0]
ham_data_g["h1"] = jnp.array(np.random.rand(2, norb, norb))
ham_data_g["chol"] = jnp.array(np.random.rand(nchol, norb * norb))
ham_data_g["ene0"] = 0.0


def test_rot_orbs():
    ham_rot = ham_handler.rotate_orbs(ham_data, mo_coeff)
    assert ham_rot["h1"].shape == (2, norb, norb)
    assert ham_rot["chol"].shape == (nchol, norb * norb)


def test_rot_ham():
    rot_ham = ham_handler.build_measurement_intermediates(ham_data, trial, wave_data)
    assert rot_ham["rot_h1"].shape == (nelec[0], norb)
    assert rot_ham["rot_chol"].shape == (nchol, nelec[0], norb)


def test_prop_ham():
    prop_ham = ham_handler.build_propagation_intermediates(
        ham_data, prop_r, trial, wave_data
    )
    assert prop_ham["mf_shifts"].shape == (nchol,)
    assert prop_ham["exp_h1"].shape == (norb, norb)


def test_rot_ham_u():
    rot_ham = ham_handler.build_measurement_intermediates(
        ham_data_u, trial_u, wave_data_u
    )
    assert rot_ham["rot_h1"][0].shape == (nelec_sp[0], norb)
    assert rot_ham["rot_h1"][1].shape == (nelec_sp[1], norb)
    assert rot_ham["rot_chol"][0].shape == (nchol, nelec_sp[0], norb)
    assert rot_ham["rot_chol"][1].shape == (nchol, nelec_sp[1], norb)


def test_prop_ham_u():
    prop_ham = ham_handler.build_propagation_intermediates(
        ham_data_u, prop_u, trial_u, wave_data_u
    )
    assert prop_ham["mf_shifts"].shape == (nchol,)
    assert prop_ham["exp_h1"].shape == (2, norb, norb)


def test_rot_ham_g():
    rot_ham = ham_handler.build_measurement_intermediates(
        ham_data_g, trial_g, wave_data_g
    )
    assert rot_ham["rot_h1"].shape == (nelec_sp[0] + nelec_sp[1], 2 * norb)
    assert rot_ham["rot_chol"].shape == (nchol, nelec_sp[0] + nelec_sp[1], 2 * norb)


def test_prop_ham_g():
    prop_ham = ham_handler.build_propagation_intermediates(
        ham_data_g, prop_u, trial_g, wave_data_g
    )
    assert prop_ham["mf_shifts"].shape == (nchol,)
    assert prop_ham["exp_h1"].shape == (2, norb, norb)


def test_rot_ham_noci():
    rot_ham = ham_handler.build_measurement_intermediates(
        ham_data_u, trial_noci, wave_data_noci
    )
    assert rot_ham["rot_h1"][0].shape == (ndets, nelec_sp[0], norb)
    assert rot_ham["rot_h1"][1].shape == (ndets, nelec_sp[1], norb)
    assert rot_ham["rot_chol"][0].shape == (ndets, nchol, nelec_sp[0], norb)
    assert rot_ham["rot_chol"][1].shape == (ndets, nchol, nelec_sp[1], norb)


def test_prop_ham_noci():
    prop_ham = ham_handler.build_propagation_intermediates(
        ham_data_u, prop_u, trial_noci, wave_data_noci
    )
    assert prop_ham["mf_shifts"].shape == (nchol,)
    assert prop_ham["exp_h1"].shape == (2, norb, norb)


if __name__ == "__main__":
    test_rot_orbs()
    test_rot_ham()
    test_prop_ham()
    test_rot_ham_u()
    test_prop_ham_u()
    test_rot_ham_g()
    test_prop_ham_g()
    test_rot_ham_noci()
    test_prop_ham_noci()
