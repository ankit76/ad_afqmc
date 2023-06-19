import os, time
import numpy as np
from numpy.random import Generator, MT19937, PCG64
import scipy as sp
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
#os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax, jit, custom_jvp, vmap, random, vjp, jvp, checkpoint, dtypes
from mpi4py import MPI
from ad_afqmc import linalg_utils, sr
#import stat_utils

from functools import partial
print = partial(print, flush=True)

@partial(jit, static_argnums=(3,4))
def _step_scan(prop_data, fields, ham_data, propagator, trial):
  prop_data = propagator.propagate(trial, ham_data, prop_data, fields)
  return prop_data, fields

@partial(jit, static_argnums=(3,4))
def _block_scan(prop_data, _x, ham_data, propagator, trial):
  prop_data['key'], subkey = random.split(prop_data['key'])
  fields = random.normal(subkey, shape=(propagator.n_steps, prop_data['walkers'].shape[0], ham_data['chol'].shape[0]))
  _step_scan_wrapper = lambda x, y: _step_scan(x, y, ham_data, propagator, trial)
  prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
  prop_data['n_killed_walkers'] += prop_data['weights'].size - jnp.count_nonzero(prop_data['weights'])
  prop_data['walkers'] = linalg_utils.qr_vmap(prop_data['walkers'])
  prop_data['overlaps'] = trial.calc_overlap_vmap(prop_data['walkers'])
  energy_samples = jnp.real(trial.calc_energy_vmap(ham_data, prop_data['walkers']))
  energy_samples = jnp.where(jnp.abs(energy_samples - prop_data['pop_control_ene_shift']) > jnp.sqrt(2./propagator.dt), prop_data['pop_control_ene_shift'], energy_samples)
  block_weight = jnp.sum(prop_data['weights'])
  block_energy = jnp.sum(energy_samples * prop_data['weights']) / block_weight
  prop_data['pop_control_ene_shift'] = 0.9 * prop_data['pop_control_ene_shift'] + 0.1 * block_energy
  return prop_data, (block_energy, block_weight)

@partial(jit, static_argnums=(3,4))
def _sr_block_scan(prop_data, _x, ham_data, propagator, trial):
  _block_scan_wrapper = lambda x, y: _block_scan(x, y, ham_data, propagator, trial)
  prop_data, (block_energy, block_weight) = lax.scan(_block_scan_wrapper, prop_data, None, length=propagator.n_blocks)
  prop_data['key'], subkey = random.split(prop_data['key'])
  zeta = random.uniform(subkey)
  prop_data['walkers'], prop_data['weights'] = sr.stochastic_reconfiguration(prop_data['walkers'], prop_data['weights'], zeta)
  prop_data['overlaps'] = trial.calc_overlap_vmap(prop_data['walkers'])
  return prop_data, (block_energy, block_weight)

@partial(jit, static_argnums=(2,3))
def _ad_block(prop_data, ham_data, propagator, trial):
  _sr_block_scan_wrapper = lambda x, y: _sr_block_scan(x, y, ham_data, propagator, trial)

  prop_data['overlaps'] = trial.calc_overlap_vmap(prop_data['walkers'] )
  prop_data['n_killed_walkers'] = 0
  prop_data['pop_control_ene_shift'] = prop_data['e_estimate']
  prop_data, (block_energy, block_weight) = lax.scan(_sr_block_scan_wrapper, prop_data, None, length=propagator.n_sr_blocks)
  prop_data['n_killed_walkers'] /= (propagator.n_blocks * prop_data['walkers'].shape[0])
  return prop_data, (block_energy, block_weight)

@partial(jit, static_argnums=(0,4,6))
def propagate_phaseless_ad(ham, ham_data, coupling, observable_op, propagator, prop_data, trial):
  ham_data['h1'] = ham_data['h1'] + coupling * observable_op
  mo_coeff = trial.optimize_orbs(ham_data)
  ham_data = ham.rot_orbs(ham_data, mo_coeff)
  ham_data = ham.rot_ham(ham_data)
  ham_data = ham.prop_ham(ham_data, propagator.dt)

  prop_data, (block_energy, block_weight) = _ad_block(prop_data, ham_data, propagator, trial)
  return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data

@partial(jit, static_argnums=(0, 2, 4))
def propagate_phaseless(ham, ham_data, propagator, prop_data, trial):
  mo_coeff = trial.optimize_orbs(ham_data)
  ham_data = ham.rot_orbs(ham_data, mo_coeff)
  ham_data = ham.rot_ham(ham_data)
  ham_data = ham.prop_ham(ham_data, propagator.dt)

  def _sr_block_scan_wrapper(x, y): return _sr_block_scan(x, y, ham_data, propagator, trial)

  prop_data['overlaps'] = trial.calc_overlap_vmap(prop_data['walkers'])
  prop_data['n_killed_walkers'] = 0
  prop_data, (block_energy, block_weight) = lax.scan(_sr_block_scan_wrapper, prop_data, None, length=propagator.n_sr_blocks)
  prop_data['n_killed_walkers'] /= (propagator.n_blocks * prop_data['walkers'].shape[0])

  return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data
