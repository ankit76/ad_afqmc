import os

import numpy as np

os.environ[
    "XLA_FLAGS"
] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
# os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
from jax import jit, vmap


# this uses numpy but is only called once after each block
def stochastic_reconfiguration_np(walkers, weights, zeta):
    nwalkers = walkers.shape[0]
    walkers = np.array(walkers)
    weights = np.array(weights)
    walkers_new = 0.0 * walkers
    cumulative_weights = np.cumsum(np.abs(weights))
    total_weight = cumulative_weights[-1]
    average_weight = total_weight / nwalkers
    weights_new = np.ones(nwalkers) * average_weight
    for i in range(nwalkers):
        z = (i + zeta) / nwalkers
        new_i = np.searchsorted(cumulative_weights, z * total_weight)
        walkers_new[i] = walkers[new_i].copy()
    return jnp.array(walkers_new), jnp.array(weights_new)


# @checkpoint
@jit
def stochastic_reconfiguration(walkers, weights, zeta):
    nwalkers = walkers.shape[0]
    cumulative_weights = jnp.cumsum(jnp.abs(weights))
    total_weight = cumulative_weights[-1]
    average_weight = total_weight / nwalkers
    weights = jnp.ones(nwalkers) * average_weight
    z = total_weight * (jnp.arange(nwalkers) + zeta) / nwalkers
    indices = vmap(jnp.searchsorted, in_axes=(None, 0))(cumulative_weights, z)
    walkers = walkers[indices]
    return walkers, weights


@jit
def stochastic_reconfiguration_uhf(walkers, weights, zeta):
    nwalkers = walkers[0].shape[0]
    cumulative_weights = jnp.cumsum(jnp.abs(weights))
    total_weight = cumulative_weights[-1]
    average_weight = total_weight / nwalkers
    weights = jnp.ones(nwalkers) * average_weight
    z = total_weight * (jnp.arange(nwalkers) + zeta) / nwalkers
    indices = vmap(jnp.searchsorted, in_axes=(None, 0))(cumulative_weights, z)
    walkers[0] = walkers[0][indices]
    walkers[1] = walkers[1][indices]
    return walkers, weights


# this uses numpy but is only called once after each block
def stochastic_reconfiguration_mpi(walkers, weights, zeta, comm):
    size = comm.Get_size()
    rank = comm.Get_rank()
    nwalkers = walkers.shape[0]
    walkers = np.array(walkers)
    weights = np.array(weights)
    walkers_new = 0.0 * walkers
    weights_new = 0.0 * weights
    global_buffer_walkers = None
    global_buffer_walkers_new = None
    global_buffer_weights = None
    global_buffer_weights_new = None
    if rank == 0:
        global_buffer_walkers = np.zeros(
            (nwalkers * size, walkers.shape[1], walkers.shape[2]), dtype=walkers.dtype
        )
        global_buffer_walkers_new = np.zeros(
            (nwalkers * size, walkers.shape[1], walkers.shape[2]), dtype=walkers.dtype
        )
        global_buffer_weights = np.zeros(nwalkers * size, dtype=weights.dtype)
        global_buffer_weights_new = np.zeros(nwalkers * size, dtype=weights.dtype)

    comm.Gather(walkers, global_buffer_walkers, root=0)
    comm.Gather(weights, global_buffer_weights, root=0)

    if rank == 0:
        cumulative_weights = np.cumsum(np.abs(global_buffer_weights))
        total_weight = cumulative_weights[-1]
        average_weight = total_weight / nwalkers / size
        global_buffer_weights_new = (np.ones(nwalkers * size) * average_weight).astype(
            weights.dtype
        )
        for i in range(nwalkers * size):
            z = (i + zeta) / nwalkers / size
            new_i = np.searchsorted(cumulative_weights, z * total_weight)
            global_buffer_walkers_new[i] = global_buffer_walkers[new_i].copy()

    comm.Scatter(global_buffer_walkers_new, walkers_new, root=0)
    comm.Scatter(global_buffer_weights_new, weights_new, root=0)
    return jnp.array(walkers_new), jnp.array(weights_new)


def stochastic_reconfiguration_mpi_uhf(walkers, weights, zeta, comm):
    size = comm.Get_size()
    rank = comm.Get_rank()
    nwalkers = walkers[0].shape[0]
    walkers[0] = np.array(walkers[0])
    walkers[1] = np.array(walkers[1])
    weights = np.array(weights)
    walkers_new_up = 0.0 * walkers[0]
    walkers_new_dn = 0.0 * walkers[1]
    weights_new = 0.0 * weights
    global_buffer_walkers_up = None
    global_buffer_walkers_new_up = None
    global_buffer_walkers_dn = None
    global_buffer_walkers_new_dn = None
    global_buffer_weights = None
    global_buffer_weights_new = None
    if rank == 0:
        global_buffer_walkers_up = np.zeros(
            (nwalkers * size, walkers[0].shape[1], walkers[0].shape[2]),
            dtype=walkers[0].dtype,
        )
        global_buffer_walkers_new_up = np.zeros(
            (nwalkers * size, walkers[0].shape[1], walkers[0].shape[2]),
            dtype=walkers[0].dtype,
        )
        global_buffer_walkers_dn = np.zeros(
            (nwalkers * size, walkers[1].shape[1], walkers[1].shape[2]),
            dtype=walkers[0].dtype,
        )
        global_buffer_walkers_new_dn = np.zeros(
            (nwalkers * size, walkers[1].shape[1], walkers[1].shape[2]),
            dtype=walkers[0].dtype,
        )
        global_buffer_weights = np.zeros(nwalkers * size, dtype=weights.dtype)
        global_buffer_weights_new = np.zeros(nwalkers * size, dtype=weights.dtype)

    comm.Gather(walkers[0], global_buffer_walkers_up, root=0)
    comm.Gather(walkers[1], global_buffer_walkers_dn, root=0)
    comm.Gather(weights, global_buffer_weights, root=0)

    if rank == 0:
        cumulative_weights = np.cumsum(np.abs(global_buffer_weights))
        total_weight = cumulative_weights[-1]
        average_weight = total_weight / nwalkers / size
        global_buffer_weights_new = (np.ones(nwalkers * size) * average_weight).astype(
            weights.dtype
        )
        for i in range(nwalkers * size):
            z = (i + zeta) / nwalkers / size
            new_i = np.searchsorted(cumulative_weights, z * total_weight)
            global_buffer_walkers_new_up[i] = global_buffer_walkers_up[new_i].copy()
            global_buffer_walkers_new_dn[i] = global_buffer_walkers_dn[new_i].copy()

    comm.Scatter(global_buffer_walkers_new_up, walkers_new_up, root=0)
    comm.Scatter(global_buffer_walkers_new_dn, walkers_new_dn, root=0)
    comm.Scatter(global_buffer_weights_new, weights_new, root=0)
    walkers_new = [jnp.array(walkers_new_up), jnp.array(walkers_new_dn)]
    return walkers_new, jnp.array(weights_new)
