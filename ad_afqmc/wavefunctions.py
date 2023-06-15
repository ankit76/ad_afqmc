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
from typing import Sequence, Tuple, Callable, Any
from ad_afqmc import linalg_utils

from functools import partial
print = partial(print, flush=True)

# we assume afqmc is performed in the rhf orbital basis
@dataclass
class rhf():
  nelec: int
  n_opt_iter: int = 30

  #@checkpoint
  @partial(jit, static_argnums=0)
  def calc_overlap(self, walker, wave_data=None):
    return jnp.linalg.det(walker[:walker.shape[1], :])**2

  @partial(jit, static_argnums=0)
  def calc_overlap_vmap(self, walkers, wave_data=None):
    return vmap(self.calc_overlap, in_axes=(0, None))(walkers, wave_data)

  #@checkpoint
  @partial(jit, static_argnums=0)
  def calc_green(self, walker, wave_data=None):
    return (walker.dot(jnp.linalg.inv(walker[:walker.shape[1], :]))).T

  @partial(jit, static_argnums=0)
  def calc_green_vmap(self, walkers, wave_data=None):
    return vmap(self.calc_green, in_axes=(0, None))(walkers, wave_data)

  #@checkpoint
  @partial(jit, static_argnums=0)
  def calc_force_bias(self, walker, rot_chol, wave_data=None):
    green_walker = self.calc_green(walker, wave_data)
    fb = 2. * jnp.einsum('gij,ij->g', rot_chol, green_walker, optimize='optimal')
    return fb

  @partial(jit, static_argnums=0)
  def calc_force_bias_vmap(self, walkers, ham, wave_data=None):
    return vmap(self.calc_force_bias, in_axes=(0, None, None))(walkers, ham['rot_chol'], wave_data)

  #@checkpoint
  @partial(jit, static_argnums=0)
  def calc_energy(self, h0, rot_h1, rot_chol, walker, wave_data=None):
    ene0 = h0
    green_walker = self.calc_green(walker, wave_data)
    ene1 = 2. * jnp.sum(green_walker * rot_h1)
    f = jnp.einsum('gij,jk->gik', rot_chol, green_walker.T, optimize='optimal')
    c = vmap(jnp.trace)(f)
    exc = jnp.sum(vmap(lambda x: x * x.T)(f))
    ene2 = 2. * jnp.sum(c * c) - exc
    return ene2 + ene1 + ene0

  @partial(jit, static_argnums=0)
  def calc_energy_vmap (self, ham, walkers, wave_data=None):
    return vmap(self.calc_energy, in_axes=(None, None, None, 0, None))(ham['h0'], ham['rot_h1'], ham['rot_chol'], walkers, wave_data)

  #@checkpoint
  @partial(jit, static_argnums=0)
  def optimize_orbs(self, ham_data, dm0=None):
    h1 = ham_data['h1']
    h2 = ham_data['chol']
    h2 = h2.reshape((h2.shape[0], h1.shape[0], h1.shape[0]))
    nelec = self.nelec
    h1 = (h1 + h1.T) / 2.
    #nelec = nelec_proxy.shape[0]
    
    def scanned_fun(carry, x):
      dm = carry
      f = jnp.einsum('gij,ik->gjk', h2, dm)
      c = vmap(jnp.trace)(f)
      vj = jnp.einsum('g,gij->ij', c, h2)
      vk = jnp.einsum('glj,gjk->lk', f, h2)
      vhf = vj - 0.5 * vk
      fock = h1 + vhf
      mo_energy, mo_coeff = linalg_utils._eigh(fock)
      idx = jnp.argmax(abs(mo_coeff.real), axis=0)
      mo_coeff = jnp.where(mo_coeff[idx, jnp.arange(len(mo_energy))].real < 0, -mo_coeff, mo_coeff)
      e_idx = jnp.argsort(mo_energy)
      e_sort = mo_energy[e_idx]
      nmo = mo_energy.size
      mo_occ = jnp.zeros(nmo)
      nocc = nelec // 2
      mo_occ = mo_occ.at[e_idx[:nocc]].set(2)
      mocc = mo_coeff[:, jnp.nonzero(mo_occ, size=nocc)[0]]
      dm = (mocc * mo_occ[jnp.nonzero(mo_occ, size=nocc)[0]]).dot(mocc.T)
      return dm, mo_coeff

    norb = h1.shape[0]
    dm0 = 2 * jnp.eye(norb, nelec//2).dot(jnp.eye(norb, nelec//2).T)
    _, mo_coeff = lax.scan(scanned_fun, dm0, None, length=self.n_opt_iter)

    return mo_coeff[-1]

  def __hash__(self):
    return hash((self.n_opt_iter,))

#@dataclass
#class uhf():
#  orbs: Any
#
#  #@checkpoint
#  @jit
#  def calc_overlap(self, walker):
#    return jnp.linalg.det(walker[:walker.shape[1], :])**2
#
#  def calc_overlap_vmap(self, walkers):
#    return vmap(self.calc_overlap)(walkers)
#
#  #@checkpoint
#  @jit
#  def calc_green(self, walker):
#    return (walker.dot(jnp.linalg.inv(walker[:walker.shape[1], :]))).T
#
#  def calc_green_vmap(self, walkers):
#    return vmap(self.calc_green)(walkers)
#
#  #@checkpoint
#  @jit
#  def calc_force_bias(self, walker, rot_chol):
#    green_walker = self.calc_green(walker)
#    fb = 2. * jnp.einsum('gij,ij->g', rot_chol, green_walker, optimize='optimal')
#    return fb
#
#  def calc_force_bias_vmap(self, walkers, rot_chol):
#    return vmap(self.calc_force_bias, in_axes = (0, None))(walkers, rot_chol)
#
#  #@checkpoint
#  @jit
#  def calc_energy(self, h0, rot_h1, rot_chol, walker):
#    ene0 = h0
#    green_walker = self.calc_green(walker)
#    ene1 = 2. * jnp.sum(green_walker * rot_h1)
#    f = jnp.einsum('gij,jk->gik', rot_chol, green_walker.T, optimize='optimal')
#    c = vmap(jnp.trace)(f)
#    exc = jnp.sum(vmap(lambda x: x * x.T)(f))
#    ene2 = 2. * jnp.sum(c * c) - exc
#    return ene2 + ene1 + ene0
#
#  def calc_energy_vmap (self, h0, rot_h1, rot_chol, walkers):
#    return vmap(self.calc_energy, in_axes = (None, None, None, 0))(h0, rot_h1, rot_chol, walkers)
#
#  #@checkpoint
#  @jit
#  def optimize_orbs(self, h1, h2, nelec_proxy, dm0):
#    nelec = nelec_proxy.shape
#    def scanned_fun(carry, x):
#      dm = carry
#      f_up = jnp.einsum('gij,ik->gjk', h2, dm[0])
#      c_up = vmap(jnp.trace)(f_up)
#      vj_up = jnp.einsum('g,gij->ij', c_up, h2)
#      vk_up = jnp.einsum('glj,gjk->lk', f_up, h2)
#      f_dn = jnp.einsum('gij,ik->gjk', h2, dm[1])
#      c_dn = vmap(jnp.trace)(f_dn)
#      vj_dn = jnp.einsum('g,gij->ij', c_dn, h2)
#      vk_dn = jnp.einsum('glj,gjk->lk', f_dn, h2)
#      fock_up = h1 + vj_up + vj_dn - vk_up
#      fock_dn = h1 + vj_up + vj_dn - vk_dn
#      mo_energy_up, mo_coeff_up = linalg_utils._eigh(fock_up)
#      mo_energy_dn, mo_coeff_dn = linalg_utils._eigh(fock_dn)
#
#      nmo = mo_energy_up.size
#
#      idx_up = jnp.argmax(abs(mo_coeff_up.real), axis=0)
#      mo_coeff_up = jnp.where(mo_coeff_up[idx_up, jnp.arange(len(mo_energy_up))].real < 0, -mo_coeff_up, mo_coeff_up)
#      e_idx_up = jnp.argsort(mo_energy_up)
#      e_sort_up = mo_energy_up[e_idx_up]
#      mo_occ_up = jnp.zeros(nmo)
#      nocc_up = nelec[0]
#      mo_occ_up = mo_occ_up.at[e_idx_up[:nocc_up]].set(1)
#      mocc_up = mo_coeff_up[:, jnp.nonzero(mo_occ_up, size=nocc_up)[0]]
#      dm_up = (mocc_up * mo_occ_up[jnp.nonzero(mo_occ_up, size=nocc_up)[0]]).dot(mocc_up.T)
#
#      idx_dn = jnp.argmax(abs(mo_coeff_dn.real), axis=0)
#      mo_coeff_dn = jnp.where(mo_coeff_dn[idx_dn, jnp.arange(len(mo_energy_dn))].real < 0, -mo_coeff_dn, mo_coeff_dn)
#      e_idx_dn = jnp.argsort(mo_energy_dn)
#      e_sort_dn = mo_energy_dn[e_idx_dn]
#      mo_occ_dn = jnp.zeros(nmo)
#      nocc_dn = nelec[1]
#      mo_occ_dn = mo_occ_dn.at[e_idx_dn[:nocc_dn]].set(1)
#      mocc_dn = mo_coeff_dn[:, jnp.nonzero(mo_occ_dn, size=nocc_dn)[0]]
#      dm_dn = (mocc_dn * mo_occ_dn[jnp.nonzero(mo_occ_dn, size=nocc_dn)[0]]).dot(mocc_dn.T)
#
#      return jnp.array([ dm_up, dm_dn ]), jnp.array([ mo_coeff_up, mo_coeff_dn ])
#
#    _, mo_coeff = lax.scan(scanned_fun, dm0, None, length=30)
#
#    return mo_coeff[-1]
#
#  def __hash__(self):
#    return hash((orbs,))
