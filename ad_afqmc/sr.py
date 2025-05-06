import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap


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
def stochastic_reconfiguration_0(walkers, weights, zeta):
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
def stochastic_reconfiguration_uhf_0(walkers, weights, zeta):
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


@jit
def stochastic_reconfiguration(walkers, weights, zeta):
    """Stochastic reconfiguration implementation for pmap"""
    nwalkers_per_device = weights.shape[0]
    n_devices = jax.device_count()
    total_walkers = nwalkers_per_device * n_devices
    device_id = lax.axis_index("device")

    # Gather weights and walkers from all devices
    all_weights = lax.all_gather(weights, axis_name="device")
    all_walkers = lax.all_gather(walkers, axis_name="device")

    # Reshape to remove the device dimension
    all_weights = all_weights.reshape(-1)
    all_walkers = all_walkers.reshape(total_walkers, *walkers.shape[1:])

    # Compute global cumulative weights
    cumulative_weights = jnp.cumsum(jnp.abs(all_weights))
    total_weight = cumulative_weights[-1]
    average_weight = total_weight / total_walkers

    # Use a fixed-size array for this device's indices
    local_indices = jnp.arange(nwalkers_per_device)

    # Calculate the global indices for this device directly
    global_indices = local_indices + device_id * nwalkers_per_device

    # Calculate z values for this device's portion
    z_values = total_weight * (global_indices + zeta) / total_walkers

    # Find indices in the global walker array
    new_indices = jax.vmap(jnp.searchsorted, in_axes=(None, 0))(
        cumulative_weights, z_values
    )

    # Get new walkers for this device
    new_walkers = jnp.take(all_walkers, new_indices, axis=0)
    new_weights = jnp.ones_like(weights) * average_weight

    return new_walkers, new_weights


@jit
def stochastic_reconfiguration_uhf(walkers, weights, zeta):
    """Stochastic reconfiguration implementation for UHF walkers with pmap"""
    nwalkers_per_device = weights.shape[0]
    n_devices = jax.device_count()
    total_walkers = nwalkers_per_device * n_devices
    device_id = lax.axis_index("device")

    # Gather weights and walkers from all devices
    all_weights = lax.all_gather(weights, axis_name="device")
    all_walkers_up = lax.all_gather(walkers[0], axis_name="device")
    all_walkers_down = lax.all_gather(walkers[1], axis_name="device")

    # Reshape to remove the device dimension
    all_weights = all_weights.reshape(-1)
    all_walkers_up = all_walkers_up.reshape(total_walkers, *walkers[0].shape[1:])
    all_walkers_down = all_walkers_down.reshape(total_walkers, *walkers[1].shape[1:])

    # Compute global cumulative weights
    cumulative_weights = jnp.cumsum(jnp.abs(all_weights))
    total_weight = cumulative_weights[-1]
    average_weight = total_weight / total_walkers

    # Use a fixed-size array for this device's indices
    local_indices = jnp.arange(nwalkers_per_device)

    # Calculate the global indices for this device directly
    global_indices = local_indices + device_id * nwalkers_per_device

    # Calculate z values for this device's portion
    z_values = total_weight * (global_indices + zeta) / total_walkers

    # Find indices in the global walker array
    new_indices = jax.vmap(jnp.searchsorted, in_axes=(None, 0))(
        cumulative_weights, z_values
    )

    # Get new walkers for this device
    new_walkers_up = jnp.take(all_walkers_up, new_indices, axis=0)
    new_walkers_down = jnp.take(all_walkers_down, new_indices, axis=0)
    new_weights = jnp.ones_like(weights) * average_weight

    return [new_walkers_up, new_walkers_down], new_weights


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
        assert global_buffer_weights is not None
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
        assert global_buffer_weights is not None
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
