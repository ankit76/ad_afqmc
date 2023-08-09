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

nelec_sp = 5, 4
wave_data = [jnp.array(np.random.rand(norb, nelec_sp[0])),
             jnp.array(np.random.rand(norb, nelec_sp[1]))]
ham_handler_u = hamiltonian.hamiltonian_uhf(norb, nelec_sp, nchol)
ham_u = { }
ham_u['h0'] = np.random.rand(1,)[0]
ham_u['h1'] = jnp.array(np.random.rand(2, norb, norb))
ham_u['chol'] = jnp.array(np.random.rand(nchol, norb * norb))

def test_rot_orbs():
  ham_rot = ham_handler.rot_orbs(ham, mo_coeff)
  assert ham_rot['h1'].shape == (norb, norb)
  assert ham_rot['chol'].shape == (nchol, norb * norb)

def test_rot_ham():
  rot_ham = ham_handler.rot_ham(ham)
  assert rot_ham['rot_h1'].shape == (nelec, norb)
  assert rot_ham['rot_chol'].shape == (nchol, nelec, norb)

def test_prop_ham():
  prop_ham = ham_handler.prop_ham(ham, 0.005)
  assert prop_ham['mf_shifts'].shape == (nchol,)
  assert prop_ham['exp_h1'].shape == (norb, norb)

def test_rot_ham_u():
  rot_ham = ham_handler_u.rot_ham(ham_u, wave_data)
  assert rot_ham['rot_h1'][0].shape == (nelec_sp[0], norb)
  assert rot_ham['rot_h1'][1].shape == (nelec_sp[1], norb)
  assert rot_ham['rot_chol'][0].shape == (nchol, nelec_sp[0], norb)
  assert rot_ham['rot_chol'][1].shape == (nchol, nelec_sp[1], norb)

def test_prop_ham_u():
  prop_ham = ham_handler_u.prop_ham(ham_u, 0.005, wave_data)
  assert prop_ham['mf_shifts'].shape == (nchol,)
  assert prop_ham['exp_h1'].shape == (2, norb, norb)

if __name__ == "__main__":
  test_rot_orbs()
  test_rot_ham()
  test_prop_ham()
  test_rot_ham_u()
  test_prop_ham_u()

