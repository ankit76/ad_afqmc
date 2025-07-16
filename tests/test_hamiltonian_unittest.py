import unittest

import numpy as np
from jax import numpy as jnp

from ad_afqmc import config

config.setup_jax()
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
prop_r = propagation.propagator_afqmc(dt=0.005)

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
prop_u = propagation.propagator_afqmc(dt=0.005, walker_type="unrestricted")

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


class TestHamiltonian(unittest.TestCase):

    # Trick to avoid it running with pytest
    __test__ = False

    def test_rot_orbs(self):
        ham_rot = ham_handler.rotate_orbs(ham_data, mo_coeff)
        self.assertTrue(ham_rot["h1"].shape == (2, norb, norb))
        self.assertTrue(ham_rot["chol"].shape == (nchol, norb * norb))

    def test_rot_ham(self):
        rot_ham = ham_handler.build_measurement_intermediates(
            ham_data, trial, wave_data
        )
        self.assertTrue(rot_ham["rot_h1"].shape == (nelec[0], norb))
        self.assertTrue(rot_ham["rot_chol"].shape == (nchol, nelec[0], norb))

    def test_prop_ham(self):
        prop_ham = ham_handler.build_propagation_intermediates(
            ham_data, prop_r, trial, wave_data
        )
        self.assertTrue(prop_ham["mf_shifts"].shape == (nchol,))
        self.assertTrue(prop_ham["exp_h1"].shape == (norb, norb))

    def test_rot_ham_u(self):
        rot_ham = ham_handler.build_measurement_intermediates(
            ham_data_u, trial_u, wave_data_u
        )
        self.assertTrue(rot_ham["rot_h1"][0].shape == (nelec_sp[0], norb))
        self.assertTrue(rot_ham["rot_h1"][1].shape == (nelec_sp[1], norb))
        self.assertTrue(rot_ham["rot_chol"][0].shape == (nchol, nelec_sp[0], norb))
        self.assertTrue(rot_ham["rot_chol"][1].shape == (nchol, nelec_sp[1], norb))

    def test_prop_ham_u(self):
        prop_ham = ham_handler.build_propagation_intermediates(
            ham_data_u, prop_u, trial_u, wave_data_u
        )
        self.assertTrue(prop_ham["mf_shifts"].shape == (nchol,))
        self.assertTrue(prop_ham["exp_h1"].shape == (2, norb, norb))

    def test_rot_ham_g(self):
        rot_ham = ham_handler.build_measurement_intermediates(
            ham_data_g, trial_g, wave_data_g
        )
        self.assertTrue(
            rot_ham["rot_h1"].shape == (nelec_sp[0] + nelec_sp[1], 2 * norb)
        )
        self.assertTrue(
            rot_ham["rot_chol"].shape == (nchol, nelec_sp[0] + nelec_sp[1], 2 * norb)
        )

    def test_prop_ham_g(self):
        prop_ham = ham_handler.build_propagation_intermediates(
            ham_data_g, prop_u, trial_g, wave_data_g
        )
        self.assertTrue(prop_ham["mf_shifts"].shape == (nchol,))
        self.assertTrue(prop_ham["exp_h1"].shape == (2, norb, norb))

    def test_rot_ham_noci(self):
        rot_ham = ham_handler.build_measurement_intermediates(
            ham_data_u, trial_noci, wave_data_noci
        )
        self.assertTrue(rot_ham["rot_h1"][0].shape == (ndets, nelec_sp[0], norb))
        self.assertTrue(rot_ham["rot_h1"][1].shape == (ndets, nelec_sp[1], norb))
        self.assertTrue(
            rot_ham["rot_chol"][0].shape == (ndets, nchol, nelec_sp[0], norb)
        )
        self.assertTrue(
            rot_ham["rot_chol"][1].shape == (ndets, nchol, nelec_sp[1], norb)
        )

    def test_prop_ham_noci(self):
        prop_ham = ham_handler.build_propagation_intermediates(
            ham_data_u, prop_u, trial_noci, wave_data_noci
        )
        self.assertTrue(prop_ham["mf_shifts"].shape == (nchol,))
        self.assertTrue(prop_ham["exp_h1"].shape == (2, norb, norb))


if __name__ == "__main__":
    unittest.main()
