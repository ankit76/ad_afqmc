import os

import numpy as np
import pytest

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import numpy as jnp
from jax import random

from ad_afqmc import hamiltonian, propagation, wavefunctions

seed = 102
np.random.seed(seed)
norb, nelec, nchol = 10, (5, 5), 5

ham_handler = hamiltonian.hamiltonian(norb)
trial = wavefunctions.rhf(norb, nelec)
prop_handler = propagation.propagator_restricted(n_walkers=7)

wave_data = {}
wave_data["mo_coeff"] = jnp.eye(norb)[:, : nelec[0]]
wave_data["rdm1"] = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)

ham_data = {}
ham_data["h0"] = np.random.rand(
    1,
)[0]
ham_data["h1"] = jnp.array(np.random.rand(norb, norb))
ham_data["chol"] = jnp.array(np.random.rand(nchol, norb * norb))
ham_data["ene0"] = 0.0
ham_data = ham_handler.build_propagation_intermediates(
    ham_data, prop_handler, trial, wave_data
)
ham_data = ham_handler.build_measurement_intermediates(ham_data, trial, wave_data)

prop_data = prop_handler.init_prop_data(trial, wave_data, ham_data)
prop_data["key"] = random.PRNGKey(seed)
prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

nelec_sp = (5, 4)
trial_u = wavefunctions.uhf(norb, nelec_sp)
prop_handler_u = propagation.propagator_unrestricted(n_walkers=7)

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
ham_data_u["h0"] = np.random.rand(
    1,
)[0]
ham_data_u["h1"] = jnp.array(np.random.rand(2, norb, norb))
ham_data_u["chol"] = jnp.array(np.random.rand(nchol, norb * norb))
ham_data_u["ene0"] = 0.0
ham_data_u = ham_handler.build_propagation_intermediates(
    ham_data_u, prop_handler_u, trial_u, wave_data_u
)
ham_data_u = ham_handler.build_measurement_intermediates(
    ham_data_u, trial_u, wave_data_u
)

prop_data_u = prop_handler_u.init_prop_data(trial_u, wave_data_u, ham_data_u)
prop_data_u["key"] = random.PRNGKey(seed)
prop_data_u["overlaps"] = trial_u.calc_overlap(prop_data_u["walkers"], wave_data_u)

fields = random.normal(
    random.PRNGKey(seed), shape=(prop_handler.n_walkers, ham_data["chol"].shape[0])
)

prop_handler_cpmc = propagation.propagator_cpmc(n_walkers=7)
prop_handler_cpmc_slow = propagation.propagator_cpmc_slow(n_walkers=7)

neighbors = tuple((i, (i + 1) % norb) for i in range(norb))
prop_handler_cpmc_nn = propagation.propagator_cpmc_nn(n_walkers=7, neighbors=neighbors)
prop_handler_cpmc_nn_slow = propagation.propagator_cpmc_nn_slow(
    n_walkers=7, neighbors=neighbors
)


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
    prop_data_cpmc = prop_handler_cpmc.init_prop_data(
        trial_cpmc_u, wave_data_u, ham_data_u
    )
    prop_data_new = prop_handler_cpmc.propagate(
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
    test_propagate_cpmc_nn()
