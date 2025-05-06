from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

# This module contains utility functions for multi-GPU execution in AFQMC simulations.


def prepare_dict_for_devices(
    data_dict: Dict[str, Any],
    shard: bool = False,
    shard_keys: Optional[List[str]] = None,
    random_key_name: str = "key",
) -> Dict[str, Any]:
    """
    Prepare a dictionary for multi-GPU execution by replicating or sharding arrays.

    Args:
        data_dict: Dictionary containing arrays and other values
        shard: Whether to shard large arrays across devices
        shard_keys: List of specific keys to shard (if None, auto-detect based on array size)
        random_key_name: Name of the key for random number generation

    Returns:
        Dictionary with arrays properly distributed across devices
    """
    devices = jax.devices()
    n_devices = len(devices)
    result_dict = {}

    if shard_keys is None:
        shard_keys = []

    for key, value in data_dict.items():
        # Special handling for random keys
        if key == random_key_name:
            if n_devices > 1:
                split_keys = jax.random.split(value, n_devices)
            else:  # to keep the same behavior for single device (for testing)
                split_keys = [value]
            result_dict[key] = jax.device_put_sharded(
                [split_keys[i] for i in range(n_devices)], devices
            )
            continue

        # Handle JAX arrays
        if isinstance(value, (jnp.ndarray, np.ndarray)):
            if shard and (
                key in shard_keys
                or (
                    len(value.shape) > 0
                    and value.shape[0] >= n_devices
                    and value.shape[0] % n_devices == 0
                )
            ):
                # Shard arrays with sufficient first dimension
                chunks = np.split(np.array(value), n_devices, axis=0)
                result_dict[key] = jax.device_put_sharded(chunks, devices)
            else:
                # Replicate arrays that shouldn't be sharded
                result_dict[key] = jax.device_put_replicated(value, devices)

        # Handle lists of arrays
        elif isinstance(value, list) and all(
            isinstance(item, (jnp.ndarray, np.ndarray)) for item in value
        ):
            if shard and key in shard_keys:
                # Shard each array in the list
                sharded_list = []
                for arr in value:
                    chunks = np.split(np.array(arr), n_devices, axis=0)
                    sharded_list.append(jax.device_put_sharded(chunks, devices))
                result_dict[key] = sharded_list
            else:
                # Replicate each array in the list
                result_dict[key] = [
                    jax.device_put_replicated(arr, devices) for arr in value
                ]

        # Handle non-array values
        else:
            result_dict[key] = value

    return result_dict


def sync_device_values(
    prop_data: Dict[str, Any],
    sync_keys: List[str],
    axis_name: str = "device",
    reduce_op: str = "mean",
) -> Dict[str, Any]:
    """
    Synchronize values across devices for specified keys.

    Args:
        prop_data: Dictionary with device-distributed values
        sync_keys: List of keys to synchronize
        axis_name: Name of the pmapped axis for communication
        reduce_op: Operation to perform ('mean', 'sum', 'min', 'max')

    Returns:
        Dictionary with synchronized values
    """
    prop_data = dict(prop_data)  # Create a copy to avoid modifying the original

    for key in prop_data:
        if key in sync_keys and isinstance(prop_data[key], jnp.ndarray):
            # Get values from all devices
            all_values = lax.all_gather(prop_data[key], axis_name=axis_name)

            # Apply the specified reduction operation
            if reduce_op == "mean":
                prop_data[key] = jnp.mean(all_values, axis=0)
            elif reduce_op == "sum":
                prop_data[key] = jnp.sum(all_values, axis=0)
            elif reduce_op == "min":
                prop_data[key] = jnp.min(all_values, axis=0)
            elif reduce_op == "max":
                prop_data[key] = jnp.max(all_values, axis=0)
            else:
                raise ValueError(f"Unsupported reduce_op: {reduce_op}")

    return prop_data


def collect_results_for_analysis(block_energy_n, prop_data, weights_key="weights"):
    """
    Collect results from all devices for analysis on CPU.

    Args:
        block_energy_n: Energy results from pmapped function
        prop_data: Dictionary with device-distributed values
        weights_key: Key for weight data in prop_data

    Returns:
        Tuple of (block_energy_n, block_weight_n) as numpy arrays
    """
    # Convert to numpy arrays
    energy_values = np.array(block_energy_n)
    weights = np.array(prop_data[weights_key])

    # Calculate weight for each device
    weights_per_device = np.sum(weights, axis=1)

    # Calculate total energy and weight
    block_energy_cpu = np.sum(energy_values * weights_per_device) / np.sum(
        weights_per_device
    )
    block_weight_cpu = np.array([np.sum(weights_per_device)], dtype="float32")

    return block_energy_cpu, block_weight_cpu
