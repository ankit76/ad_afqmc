import os
import pytest
import numpy as np
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
from jax import numpy as jnp
from ad_afqmc import hamiltonian, propagation, wavefunctions

seed = 102
np.random.seed(seed)
norb, nelec, nchol = 10, 5, 5
prop_handler = propagation.propagator()
ham_handler = hamiltonian.hamiltonian(norb, nelec, nchol)
ham = { }
ham['h0'] = np.random.rand(1,)[0]
ham['h1'] = jnp.array(np.random.rand(norb, norb))
ham['chol'] = jnp.array(np.random.rand(nchol, norb * norb))
ham = ham_handler.prop_ham(ham, prop_handler.dt)
ham = ham_handler.rot_ham(ham)
prop = { }
n_walkers = 6
prop['walkers'] = jnp.array(np.random.rand(n_walkers, norb, nelec)) + 1.j * jnp.array(np.random.rand(n_walkers, norb, nelec))
prop['fields'] = jnp.array(np.random.rand(n_walkers, nchol)) + 1.j * jnp.array(np.random.rand(n_walkers, nchol))
trial = wavefunctions.rhf()
prop['weights'] = jnp.array(np.random.rand(n_walkers,))
prop['overlaps'] = jnp.array(np.random.rand(n_walkers,)) + 1.j * jnp.array(np.random.rand(n_walkers,))
prop['pop_control_ene_shift'] = 0.0
prop['e_estimate'] = 0.0

def test_apply_propagator():
  new_walkers = prop_handler.apply_propagator_vmap(ham, prop['walkers'], prop['fields'])
  #assert np.allclose(jnp.real(overlap), -0.10794844182417201)

def test_propagate():
  prop_new = prop_handler.propagate(trial, ham, prop)
  #assert np.allclose(jnp.real(overlap), -0.10794844182417201)

if __name__ == "__main__":
  test_apply_propagator()
  test_propagate()
