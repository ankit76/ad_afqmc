import numpy as np

from ad_afqmc import config

config.setup_jax()
from jax import numpy as jnp
from jax import scipy as jsp
from jax import random

from ad_afqmc import hamiltonian, propagation, wavefunctions

seed = 102
np.random.seed(seed)
norb, nelec, nchol = 10, (5, 5), 5

# -----------------------------------------------------------------------------
# RHF.
# -----------------------------------------------------------------------------
ham_handler = hamiltonian.hamiltonian(norb)
trial = wavefunctions.rhf(norb, nelec)
prop_handler = propagation.propagator_restricted(n_walkers=10, n_batch=5)

wave_data = {}
wave_data["mo_coeff"] = jnp.eye(norb)[:, : nelec[0]]
wave_data["rdm1"] = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)

ham_data = {}
ham_data["h0"] = np.random.rand(1,)[0]

# Use symmetric matrices in tests.
h1 = jnp.array(np.random.rand(norb, norb))
h1_symm = (h1 + h1.T) / 2.0

ham_data["h1"] = h1_symm
ham_data["h1"] = jnp.array([ham_data["h1"], ham_data["h1"]])
ham_data["chol"] = jnp.array(np.random.rand(nchol, norb * norb))
ham_data["ene0"] = 0.0
ham_data = ham_handler.build_propagation_intermediates(
    ham_data, prop_handler, trial, wave_data
)
ham_data = ham_handler.build_measurement_intermediates(ham_data, trial, wave_data)

prop_data = prop_handler.init_prop_data(trial, wave_data, ham_data)
prop_data["key"] = random.PRNGKey(seed)
prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

# -----------------------------------------------------------------------------
# UHF.
# -----------------------------------------------------------------------------
nelec_sp = (5, 4)
trial_u = wavefunctions.uhf(norb, nelec_sp)
prop_handler_u = propagation.propagator_unrestricted(n_walkers=10, n_batch=5)

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

ham_data_u = {}
ham_data_u["h0"] = np.random.rand(1,)[0]

# Use symmetric matrices in tests.
h1 = jnp.array(np.random.rand(2, norb, norb))
h1_symm = (h1 + h1.transpose(0, 2, 1)) / 2.0
ham_data_u["h1"] = h1_symm
ham_data_u["chol"] = jnp.array(np.random.rand(nchol, norb * norb))
ham_data_u["ene0"] = 0.0
ham_data_u = ham_handler.build_propagation_intermediates(
    ham_data_u, prop_handler_u, trial_u, wave_data_u
)
ham_data_u = ham_handler.build_measurement_intermediates(
    ham_data_u, trial_u, wave_data_u
)

# -----------------------------------------------------------------------------
# GHF.
# -----------------------------------------------------------------------------
nelec_sp = (5, 4)
nelec = sum(nelec_sp)
trial_g = wavefunctions.ghf(norb, nelec_sp)
prop_handler_g = propagation.propagator_general(n_walkers=10, n_batch=5)

wave_data_g = {}
wave_data_g["mo_coeff"] = jnp.array(np.random.rand(2 * norb, nelec))
wave_data_g["rdm1"] = jnp.array(
                        wave_data_g["mo_coeff"] @ wave_data_g["mo_coeff"].T)

ham_data_g = {}
ham_data_g["h0"] = np.random.rand(1,)[0]
ham_data_g["h1"] = jnp.array(np.random.rand(2, norb, norb))
ham_data_g["chol"] = jnp.array(np.random.rand(nchol, norb * norb))
ham_data_g["ene0"] = 0.0
ham_data_g = ham_handler.build_propagation_intermediates(
    ham_data_g, prop_handler_g, trial_g, wave_data_g
)
ham_data_g = ham_handler.build_measurement_intermediates(
    ham_data_g, trial_g, wave_data_g
)

prop_data_u = prop_handler_u.init_prop_data(trial_u, wave_data_u, ham_data_u)
prop_data_u["key"] = random.PRNGKey(seed)
prop_data_u["overlaps"] = trial_u.calc_overlap(prop_data_u["walkers"], wave_data_u)

prop_data_g = prop_handler_g.init_prop_data(trial_g, wave_data_g, ham_data_g)
prop_data_g["key"] = random.PRNGKey(seed)
prop_data_g["overlaps"] = trial_g.calc_overlap(prop_data_g["walkers"], wave_data_g)

# -----------------------------------------------------------------------------
# GHF from UHF.
# -----------------------------------------------------------------------------
mo_coeff_u = wave_data_u["mo_coeff"]
mo_coeff_g2 = np.zeros(wave_data_g["mo_coeff"].shape, dtype=mo_coeff_u[0].dtype)
mo_coeff_g2[: norb, : nelec_sp[0]] = mo_coeff_u[0]
mo_coeff_g2[norb :, nelec_sp[0] :] = mo_coeff_u[1]
mo_coeff_g2 = jnp.array(mo_coeff_g2)

wave_data_g2 = {}
wave_data_g2["mo_coeff"] = mo_coeff_g2
wave_data_g2["rdm1"] = jnp.array(
                        wave_data_g2["mo_coeff"] @ wave_data_g2["mo_coeff"].T)

ham_data_g2 = {}
ham_data_g2["h0"] = ham_data_u["h0"].copy()
ham_data_g2["h1"] = ham_data_u["h1"].copy()
ham_data_g2["chol"] = ham_data_u["chol"].copy()
ham_data_g2["ene0"] = ham_data_u["ene0"]
ham_data_g2 = ham_handler.build_propagation_intermediates(
    ham_data_g2, prop_handler_g, trial_g, wave_data_g2
)
ham_data_g2 = ham_handler.build_measurement_intermediates(
    ham_data_g2, trial_g, wave_data_g2
)

walkers_u = prop_data_u["walkers"]
n_walkers = walkers_u[0].shape[0]
walkers_g2 = np.zeros(prop_data_g["walkers"].shape, dtype=walkers_u[0].dtype)

for iw in range(n_walkers):
    walkers_g2[iw, : norb, : nelec_sp[0]] = walkers_u[0][iw]
    walkers_g2[iw, norb :, nelec_sp[0] :] = walkers_u[1][iw]

walkers_g2 = jnp.array(walkers_g2)

prop_data_g2 = prop_handler_g.init_prop_data(
    trial_g, wave_data_g2, ham_data_g2, init_walkers=walkers_g2
)
prop_data_g2["key"] = random.PRNGKey(seed)
prop_data_g2["overlaps"] = trial_g.calc_overlap(prop_data_g2["walkers"], wave_data_g2)

fields = random.normal(
    random.PRNGKey(seed), shape=(prop_handler.n_walkers, nchol)
)

# -----------------------------------------------------------------------------
# CPMC.
# -----------------------------------------------------------------------------
prop_handler_cpmc_u = propagation.propagator_cpmc_unrestricted(n_walkers=10)
prop_handler_cpmc_g = propagation.propagator_cpmc_general(n_walkers=10)
prop_handler_cpmc_slow = propagation.propagator_cpmc_slow(n_walkers=10)

neighbors = tuple((i, (i + 1) % norb) for i in range(norb))
prop_handler_cpmc_nn = propagation.propagator_cpmc_nn(n_walkers=10, neighbors=neighbors)
prop_handler_cpmc_nn_slow = propagation.propagator_cpmc_nn_slow(
    n_walkers=10, neighbors=neighbors
)


# -----------------------------------------------------------------------------
# Tests.
# -----------------------------------------------------------------------------
def test_stochastic_reconfiguration_local():
    prop_data_new = prop_handler.stochastic_reconfiguration_local(prop_data)
    assert prop_data_new["walkers"].shape == prop_data["walkers"].shape
    assert prop_data_new["weights"].shape == prop_data["weights"].shape


def test_propagate():
    prop_data_new = prop_handler.propagate(
        trial, ham_data, prop_data, fields, wave_data
    )
    assert prop_data_new["walkers"].shape == prop_data["walkers"].shape
    assert prop_data_new["weights"].shape == prop_data["weights"].shape
    assert prop_data_new["overlaps"].shape == prop_data["overlaps"].shape


def test_stochastic_reconfiguration_local_u():
    prop_data_new = prop_handler_u.stochastic_reconfiguration_local(prop_data_u)
    assert prop_data_new["walkers"][0].shape == prop_data_u["walkers"][0].shape
    assert prop_data_new["walkers"][1].shape == prop_data_u["walkers"][1].shape
    assert prop_data_new["weights"].shape == prop_data_u["weights"].shape


def test_propagate_u():
    prop_data_new = prop_handler_u.propagate(
        trial_u, ham_data_u, prop_data_u, fields, wave_data_u
    )
    assert prop_data_new["walkers"][0].shape == prop_data_u["walkers"][0].shape
    assert prop_data_new["walkers"][1].shape == prop_data_u["walkers"][1].shape
    assert prop_data_new["weights"].shape == prop_data_u["weights"].shape
    assert prop_data_new["overlaps"].shape == prop_data_u["overlaps"].shape


def test_propagate_free_u():
    prop_data_new = prop_handler_u.propagate_free(
        trial_u, ham_data_u, prop_data_u, fields, wave_data_u
    )
    assert prop_data_new["walkers"][0].shape == prop_data_u["walkers"][0].shape
    assert prop_data_new["walkers"][1].shape == prop_data_u["walkers"][1].shape
    assert prop_data_new["weights"].shape == prop_data_u["weights"].shape
    assert prop_data_new["overlaps"].shape == prop_data_u["overlaps"].shape


def test_propagate_cpmc():
    trial_cpmc_u = wavefunctions.uhf_cpmc(norb, nelec_sp)
    ham_data_u["u"] = 4.0
    prop_data_cpmc = prop_handler_cpmc_u.init_prop_data(
        trial_cpmc_u, wave_data_u, ham_data_u
    )
    prop_data_new = prop_handler_cpmc_u.propagate(
        trial_cpmc_u, ham_data_u, prop_data_cpmc, fields, wave_data_u
    )
    assert prop_data_new["walkers"][0].shape == prop_data_u["walkers"][0].shape
    assert prop_data_new["walkers"][1].shape == prop_data_u["walkers"][1].shape
    assert prop_data_new["weights"].shape == prop_data_u["weights"].shape
    assert prop_data_new["overlaps"].shape == prop_data_u["overlaps"].shape
    prop_data_new_slow = prop_handler_cpmc_slow.propagate(
        trial_cpmc_u, ham_data_u, prop_data_cpmc, fields, wave_data_u
    )
    assert np.allclose(prop_data_new_slow["walkers"][0], prop_data_new["walkers"][0])
    assert np.allclose(prop_data_new_slow["walkers"][1], prop_data_new["walkers"][1])
    assert np.allclose(prop_data_new_slow["weights"], prop_data_new["weights"])
    assert np.allclose(prop_data_new_slow["overlaps"], prop_data_new["overlaps"])


def test_propagate_one_body_cpmc_g():
    trial_cpmc_u = wavefunctions.uhf_cpmc(norb, nelec_sp)
    trial_cpmc_g = wavefunctions.ghf_cpmc(norb, nelec_sp)

    ham_data_u["u"] = 4.0
    ham_data_g["u"] = 4.0
    ham_data_g2["u"] = 4.0

    prop_data_cpmc_u = prop_handler_cpmc_u.init_prop_data(
        trial_cpmc_u, wave_data_u, ham_data_u
    )
    prop_data_cpmc_g = prop_handler_cpmc_g.init_prop_data(
        trial_cpmc_g, wave_data_g, ham_data_g
    )
    prop_data_cpmc_g2 = prop_handler_cpmc_g.init_prop_data(
        trial_cpmc_g, wave_data_g2, ham_data_g2
    )
    
    prop_data_new_u = prop_handler_cpmc_u.propagate_one_body(
        trial_cpmc_u, ham_data_u, prop_data_cpmc_u, wave_data_u
    )
    prop_data_new_g = prop_handler_cpmc_g.propagate_one_body(
        trial_cpmc_g, ham_data_g, prop_data_cpmc_g, wave_data_g
    )
    prop_data_new_g2 = prop_handler_cpmc_g.propagate_one_body(
        trial_cpmc_g, ham_data_g2, prop_data_cpmc_g2, wave_data_g2
    )

    assert prop_data_new_g["walkers"].shape == prop_data_g["walkers"].shape
    assert prop_data_new_g["weights"].shape == prop_data_g["weights"].shape
    assert prop_data_new_g["overlaps"].shape == prop_data_g["overlaps"].shape
    
    np.testing.assert_allclose(
        prop_data_cpmc_u["walkers"][0], 
        prop_data_cpmc_g2["walkers"][:, : norb, : nelec_sp[0]]
    )
    np.testing.assert_allclose(
        prop_data_cpmc_u["walkers"][1], 
        prop_data_cpmc_g2["walkers"][:, norb :, nelec_sp[0] :]
        )
    np.testing.assert_allclose(
        prop_data_new_u["walkers"][0], 
        prop_data_new_g2["walkers"][:, : norb, : nelec_sp[0]]
    )
    assert np.allclose(
        prop_data_new_u["walkers"][1], 
        prop_data_new_g2["walkers"][:, norb :, nelec_sp[0] :]
        )
    assert np.allclose(prop_data_new_u["overlaps"], prop_data_new_g2["overlaps"])
    assert np.allclose(prop_data_new_u["weights"], prop_data_new_g2["weights"])


def test_propagate_cpmc_g():
    trial_cpmc_u = wavefunctions.uhf_cpmc(norb, nelec_sp)
    trial_cpmc_g = wavefunctions.ghf_cpmc(norb, nelec_sp)

    ham_data_u["u"] = 4.0
    ham_data_g["u"] = 4.0
    ham_data_g2["u"] = 4.0

    prop_data_cpmc_u = prop_handler_cpmc_u.init_prop_data(
        trial_cpmc_u, wave_data_u, ham_data_u
    )
    prop_data_cpmc_g = prop_handler_cpmc_g.init_prop_data(
        trial_cpmc_g, wave_data_g, ham_data_g
    )
    prop_data_cpmc_g2 = prop_handler_cpmc_g.init_prop_data(
        trial_cpmc_g, wave_data_g2, ham_data_g2
    )
    
    prop_data_new_u = prop_handler_cpmc_u.propagate(
        trial_cpmc_u, ham_data_u, prop_data_cpmc_u, fields, wave_data_u
    )
    prop_data_new_g = prop_handler_cpmc_g.propagate(
        trial_cpmc_g, ham_data_g, prop_data_cpmc_g, fields, wave_data_g
    )
    prop_data_new_g2 = prop_handler_cpmc_g.propagate(
        trial_cpmc_g, ham_data_g2, prop_data_cpmc_g2, fields, wave_data_g2
    )

    assert prop_data_new_g["walkers"].shape == prop_data_g["walkers"].shape
    assert prop_data_new_g["weights"].shape == prop_data_g["weights"].shape
    assert prop_data_new_g["overlaps"].shape == prop_data_g["overlaps"].shape
    
    np.testing.assert_allclose(
        prop_data_cpmc_u["walkers"][0], 
        prop_data_cpmc_g2["walkers"][:, : norb, : nelec_sp[0]]
    )
    np.testing.assert_allclose(
        prop_data_cpmc_u["walkers"][1], 
        prop_data_cpmc_g2["walkers"][:, norb :, nelec_sp[0] :]
        )
    np.testing.assert_allclose(
        prop_data_new_u["walkers"][0], 
        prop_data_new_g2["walkers"][:, : norb, : nelec_sp[0]]
    )
    assert np.allclose(
        prop_data_new_u["walkers"][1], 
        prop_data_new_g2["walkers"][:, norb :, nelec_sp[0] :]
        )
    assert np.allclose(prop_data_new_u["overlaps"], prop_data_new_g2["overlaps"])
    assert np.allclose(prop_data_new_u["weights"], prop_data_new_g2["weights"])


def test_propagate_cpmc_nn():
    trial_cpmc_u = wavefunctions.uhf_cpmc(norb, nelec_sp)
    ham_data_u["u"] = 4.0
    ham_data_u["u_1"] = 1.0
    prop_data_cpmc = prop_handler_cpmc_nn.init_prop_data(
        trial_cpmc_u, wave_data_u, ham_data_u
    )
    prop_data_cpmc["key"] = random.PRNGKey(seed)
    prop_data_new = prop_handler_cpmc_nn.propagate(
        trial_cpmc_u, ham_data_u, prop_data_cpmc, fields, wave_data_u
    )
    assert prop_data_new["walkers"][0].shape == prop_data_u["walkers"][0].shape
    assert prop_data_new["walkers"][1].shape == prop_data_u["walkers"][1].shape
    assert prop_data_new["weights"].shape == prop_data_u["weights"].shape
    assert prop_data_new["overlaps"].shape == prop_data_u["overlaps"].shape
    prop_data_cpmc["key"] = random.PRNGKey(seed)
    prop_data_new_slow = prop_handler_cpmc_nn_slow.propagate(
        trial_cpmc_u, ham_data_u, prop_data_cpmc, fields, wave_data_u
    )
    assert np.allclose(prop_data_new_slow["walkers"][0], prop_data_new["walkers"][0])
    assert np.allclose(prop_data_new_slow["walkers"][1], prop_data_new["walkers"][1])
    assert np.allclose(prop_data_new_slow["weights"], prop_data_new["weights"])
    assert np.allclose(prop_data_new_slow["overlaps"], prop_data_new["overlaps"])


if __name__ == "__main__":
    test_stochastic_reconfiguration_local()
    test_propagate()
    test_stochastic_reconfiguration_local_u()
    test_propagate_u()
    test_propagate_free_u()
    test_propagate_cpmc()
    test_propagate_one_body_cpmc_g()
    test_propagate_cpmc_g()
    test_propagate_cpmc_nn()
