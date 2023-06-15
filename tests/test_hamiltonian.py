import os
import pytest
import numpy as np
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
from jax import numpy as jnp
from ad_afqmc import hamiltonian

seed = 102
np.random.seed(seed)
norb, nelec, nchol = 10, 5, 5
ham_handler = hamiltonian.hamiltonian(norb, nelec, nchol)
ham = { }
ham['h0'] = np.random.rand(1,)[0]
ham['h1'] = jnp.array(np.random.rand(norb, norb))
ham['chol'] = jnp.array(np.random.rand(nchol, norb * norb))
mo_coeff = jnp.array(np.random.rand(norb, norb))

def test_rot_orbs():
  _ = ham_handler.rot_orbs(ham, mo_coeff)
  #assert np.allclose(jnp.real(overlap), -0.10794844182417201)

def test_rot_ham():
  _ = ham_handler.rot_ham(ham)
  #assert np.allclose(jnp.real(overlap), -0.10794844182417201)

def test_prop_ham():
  _ = ham_handler.prop_ham(ham, 0.005)
  #assert np.allclose(jnp.real(overlap), -0.10794844182417201)

if __name__ == "__main__":
  test_rot_orbs()
  test_rot_ham()
  test_prop_ham()

