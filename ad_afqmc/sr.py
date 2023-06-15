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

# this uses numpy but is only called once after each block
def stochastic_reconfiguration_np(walkers, weights, zeta):
  nwalkers = walkers.shape[0]
  walkers = np.array(walkers)
  weights = np.array(weights)
  walkers_new = 0. * walkers
  cumulative_weights = np.cumsum(np.abs(weights))
  total_weight = cumulative_weights[-1]
  average_weight = total_weight / nwalkers
  weights_new = np.ones(nwalkers) * average_weight
  for i in range(nwalkers):
    z = (i + zeta) / nwalkers
    new_i = np.searchsorted(cumulative_weights, z * total_weight)
    walkers_new[i] = walkers[new_i].copy()
  return jnp.array(walkers_new), jnp.array(weights_new)


#@checkpoint
@jit
def stochastic_reconfiguration(walkers, weights, zeta):
  nwalkers = walkers.shape[0]
  cumulative_weights = jnp.cumsum(jnp.abs(weights))
  total_weight = cumulative_weights[-1]
  average_weight = total_weight / nwalkers
  weights = jnp.ones(nwalkers) * average_weight
  z = total_weight * (jnp.arange(nwalkers) + zeta) / nwalkers
  indices = vmap(jnp.searchsorted, in_axes = (None, 0))(cumulative_weights, z)
  walkers = walkers[indices]
  return walkers, weights

# this uses numpy but is only called once after each block
def stochastic_reconfiguration_mpi(walkers, weights, zeta, comm):
  size = comm.Get_size()
  rank = comm.Get_rank()
  nwalkers = walkers.shape[0]
  walkers = np.array(walkers)
  weights = np.array(weights)
  walkers_new = 0. * walkers
  weights_new = 0. * weights
  global_buffer_walkers = None
  global_buffer_walkers_new = None
  global_buffer_weights = None
  global_buffer_weights_new = None
  if rank == 0:
    global_buffer_walkers = np.zeros((nwalkers * size, walkers.shape[1], walkers.shape[2]), dtype=walkers.dtype)
    global_buffer_walkers_new = np.zeros((nwalkers * size, walkers.shape[1], walkers.shape[2]), dtype=walkers.dtype)
    global_buffer_weights = np.zeros(nwalkers * size, dtype=weights.dtype)
    global_buffer_weights_new = np.zeros(nwalkers * size, dtype=weights.dtype)

  comm.Gather(walkers, global_buffer_walkers, root=0)
  comm.Gather(weights, global_buffer_weights, root=0)

  if rank == 0:
    cumulative_weights = np.cumsum(np.abs(global_buffer_weights))
    total_weight = cumulative_weights[-1]
    average_weight = total_weight / nwalkers / size
    global_buffer_weights_new = (np.ones(nwalkers * size) * average_weight).astype(weights.dtype)
    for i in range(nwalkers * size):
      z = (i + zeta) / nwalkers / size
      new_i = np.searchsorted(cumulative_weights, z * total_weight)
      global_buffer_walkers_new[i] = global_buffer_walkers[new_i].copy()

  comm.Scatter(global_buffer_walkers_new, walkers_new, root=0)
  comm.Scatter(global_buffer_weights_new, weights_new, root=0)
  return jnp.array(walkers_new), jnp.array(weights_new)
