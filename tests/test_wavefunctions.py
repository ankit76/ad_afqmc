import os
import pytest
import numpy as np
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
from jax import numpy as jnp
from ad_afqmc import wavefunctions

seed = 102
np.random.seed(seed)
rhf = wavefunctions.rhf()
norb, nelec, nchol = 10, 5, 5
walker = jnp.array(np.random.rand(norb, nelec)) + 1.j * jnp.array(np.random.rand(norb, nelec))
h0 = np.random.rand(1,)[0]
rot_h1 = jnp.array(np.random.rand(nelec, norb))
rot_chol = jnp.array(np.random.rand(nchol, nelec, norb))
h1 = jnp.array(np.random.rand(norb, norb))
chol = jnp.array(np.random.rand(nchol, norb, norb))

def test_rhf_overlap():
  overlap = rhf.calc_overlap(walker)
  assert np.allclose(jnp.real(overlap), -0.10794844182417201)

def test_rhf_green():
  green = rhf.calc_green(walker)
  assert np.allclose(jnp.real(jnp.sum(green)), 12.181348093111438)

def test_rhf_force_bias():
  force_bias = rhf.calc_force_bias(walker, rot_chol)
  assert np.allclose(jnp.real(jnp.sum(force_bias)), 66.13455680423321)

def test_rhf_energy():
  energy = rhf.calc_energy(h0, rot_h1, rot_chol, walker)
  assert np.allclose(jnp.real(energy), 217.79874063608622)

def test_rhf_optimize_orbs():
  orbs = rhf.optimize_orbs(h1, chol, nelec)
  assert np.allclose(jnp.sum(orbs), 3.014929093892039)

if __name__ == "__main__":
  test_rhf_overlap()
  test_rhf_green()
  test_rhf_force_bias()
  test_rhf_energy()
  test_rhf_optimize_orbs()

