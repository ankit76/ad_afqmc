import os, time, math
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
class propagator():
  dt: float = 0.01
  n_prop_steps: int = 50
  n_ene_blocks: int = 50
  n_sr_blocks: int = 1
  n_blocks: int = 50
  ad_q: bool = True

  def __post_init__(self):
    if not self.ad_q:
      self.n_ene_blocks = 5
      self.n_sr_blocks = 10

  # defining this separately because calculating vhs for a batch seems to be faster
  #@checkpoint
  @partial(jit, static_argnums=(0,))
  def apply_propagator(self, exp_h1, vhs_i, walker_i):
    walker_i = exp_h1.dot(walker_i)
    def scanned_fun(carry, x):
      carry = vhs_i.dot(carry)
      return carry, carry
    _, vhs_n_walker = lax.scan(scanned_fun, walker_i, jnp.arange(1, 6))
    walker_i = walker_i + jnp.sum(jnp.stack([ vhs_n_walker[n] / math.factorial(n+1) for n in range(5) ]), axis=0)
    walker_i = exp_h1.dot(walker_i)
    return walker_i

  #@checkpoint
  @partial(jit, static_argnums=(0,))
  def apply_propagator_vmap(self, ham, walkers, fields):
    vhs = 1.j * jnp.sqrt(self.dt) * fields.dot(ham['chol']).reshape(walkers.shape[0], walkers.shape[1], walkers.shape[1])
    return vmap(self.apply_propagator, in_axes = (None, 0, 0))(ham['exp_h1'], vhs, walkers)

  # defining this separately because of possible divergences in derivatives
  #@custom_jvp
  def calc_imp_fun(self, exponent_1, exponent_2, overlaps_new, overlaps_old):
    imp_fun = jnp.exp(exponent_1) * overlaps_new / overlaps_old
    theta = jnp.angle(jnp.exp(exponent_2) * overlaps_new / overlaps_old)
    #imp_fun_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)
    return imp_fun, theta
    #return imp_fun
    #return imp_fun_phaseless

  #@calc_imp_fun.defjvp
  #def _imp_fun_jvp(primals, tangents):
  #  primals_out, tangents_out = jvp(calc_imp_fun0, primals, tangents)
  #  return primals_out, (jnp.clip(tangents_out[0], -1000., 1000.), jnp.clip(tangents_out[1], -1000., 1000.))
  #  #return primals_out, (0. * primals_out[0], 0. * primals_out[1])

  @partial(jit, static_argnums=(0,1))
  def propagate(self, trial, ham, prop, fields):
    force_bias = trial.calc_force_bias_vmap(prop['walkers'], ham)
    field_shifts = -jnp.sqrt(self.dt) * (1.j * force_bias - ham['mf_shifts'])
    shifted_fields = fields - field_shifts
    shift_term = jnp.sum(shifted_fields * ham['mf_shifts'], axis=1)
    fb_term = jnp.sum(fields * field_shifts - field_shifts * field_shifts / 2., axis=1)

    prop['walkers'] = self.apply_propagator_vmap(ham, prop['walkers'], shifted_fields)

    overlaps_new = trial.calc_overlap_vmap(prop['walkers'])
    imp_fun, theta = self.calc_imp_fun(-jnp.sqrt(self.dt) * shift_term + fb_term + self.dt * (prop['pop_control_ene_shift'] + ham['h0_prop']), -jnp.sqrt(self.dt) * shift_term, overlaps_new, prop['overlaps'])
    imp_fun_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)
    imp_fun_phaseless = jnp.where(jnp.isnan(imp_fun_phaseless), 0., imp_fun_phaseless)
    imp_fun_phaseless = jnp.where(imp_fun_phaseless < 1.e-3, 0., imp_fun_phaseless) # type: ignore
    imp_fun_phaseless = jnp.where(imp_fun_phaseless > 100., 0., imp_fun_phaseless)
    prop['weights'] = imp_fun_phaseless * prop['weights']
    prop['weights'] = jnp.where(prop['weights'] > 100., 0., prop['weights'])
    prop['pop_control_ene_shift'] = prop['e_estimate'] - 0.1 * jnp.array(jnp.log(jnp.sum(prop['weights']) / prop['walkers'].shape[0]) / self.dt) # type: ignore
    prop['overlaps'] = overlaps_new
    return prop

  def __hash__(self):
    return hash((self.dt, self.n_prop_steps, self.n_ene_blocks, self.n_sr_blocks, self.n_blocks))

if __name__ == "__main__":
  prop = propagator()
  nelec = 3
  norb = 6
  nchol = 6
  nwalkers = 5
  h0 = 0.
  key = random.PRNGKey(0)
  key, subkey = random.split(key)
  h1 = random.normal(subkey, (norb, norb))
  key, subkey = random.split(key)
  walkers = random.normal(subkey, (nwalkers, norb, nelec)) + 0.j
  h0_prop = 0.
  exp_h1 = jsp.linalg.expm(-prop.dt * h1 / 2.)
  key, subkey = random.split(key)
  chol = random.normal(subkey, (nchol, norb, norb))
  chol = chol.reshape(nchol, norb * norb)
  key, subkey = random.split(key)
  fields = random.normal(subkey, shape=(nwalkers, nchol))
  new_walkers = prop.apply_propagator_vmap(exp_h1, chol, walkers, fields)
