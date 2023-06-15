import os, time
import numpy as np
import scipy as sp
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
#os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax, jit, custom_jvp, vmap, random, vjp, jvp, checkpoint, dtypes
from dataclasses import dataclass

from functools import partial
print = partial(print, flush=True)

@dataclass
class hamiltonian():
  norb: int
  nelec: int
  nchol: int

  @partial(jit, static_argnums=(0,))
  def rot_orbs(self, ham, mo_coeff):
    ham['h1'] = mo_coeff.T.dot(ham['h1']).dot(mo_coeff)
    ham['chol'] = jnp.einsum('gij,jp->gip', ham['chol'].reshape(-1, self.norb, self.norb), mo_coeff)
    ham['chol'] = jnp.einsum('qi,gip->gqp', mo_coeff.T, ham['chol']).reshape(-1, self.norb * self.norb)
    return ham

  @partial(jit, static_argnums=(0,))
  def rot_ham(self, ham):
    ham['rot_h1'] = ham['h1'][:self.nelec, :].copy()
    ham['rot_chol'] = ham['chol'].reshape(-1, self.norb, self.norb)[:, :self.nelec, :].copy()
    return ham

  @partial(jit, static_argnums=(0,))
  def prop_ham(self, ham, dt):
    ham['mf_shifts'] = 2.j * vmap(lambda x: jnp.sum(jnp.diag(x.reshape(self.norb, self.norb))[:self.nelec]))(ham['chol'])
    ham['h0_prop'] = - ham['h0'] - jnp.sum(ham['mf_shifts']**2) / 2.
    v0 = 0.5 * jnp.einsum('gik,gjk->ij', ham['chol'].reshape(-1, self.norb, self.norb), ham['chol'].reshape(-1, self.norb, self.norb), optimize='optimal')
    h1_mod = ham['h1'] - v0
    h1_mod = h1_mod - jnp.real(1.j * jnp.einsum('g,gik->ik', ham['mf_shifts'], ham['chol'].reshape(-1, self.norb, self.norb)))
    ham['exp_h1'] = jsp.linalg.expm(-dt * h1_mod / 2.)
    return ham

  def __hash__(self):
    return hash((self.norb, self.nelec, self.nchol))
