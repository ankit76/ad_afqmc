import os

os.environ[
    "XLA_FLAGS"
] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
from functools import partial

# os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
from jax import checkpoint, jit, lax, random

# import stat_utils

print = partial(print, flush=True)


@partial(jit, static_argnums=(3, 4))
def _step_scan(prop_data, fields, ham_data, propagator, trial, wave_data):
    prop_data = propagator.propagate(trial, ham_data, prop_data, fields, wave_data)
    return prop_data, fields


@partial(jit, static_argnums=(3, 4))
def _step_scan_free(prop_data, fields, ham_data, propagator, trial, wave_data):
    prop_data = propagator.propagate_free(trial, ham_data, prop_data, fields, wave_data)
    return prop_data, fields


@partial(jit, static_argnums=(3, 4))
def _block_scan(prop_data, _x, ham_data, propagator, trial, wave_data):
    prop_data["key"], subkey = random.split(prop_data["key"])
    fields = random.normal(
        subkey,
        shape=(
            propagator.n_prop_steps,
            propagator.n_walkers,
            ham_data["chol"].shape[0],
        ),
    )
    _step_scan_wrapper = lambda x, y: _step_scan(
        x, y, ham_data, propagator, trial, wave_data
    )
    prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
    prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
        prop_data["weights"]
    )
    prop_data = propagator.orthonormalize_walkers(prop_data)
    prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
    energy_samples = jnp.real(
        trial.calc_energy_vmap(ham_data, prop_data["walkers"], wave_data)
    )
    energy_samples = jnp.where(
        jnp.abs(energy_samples - prop_data["pop_control_ene_shift"])
        > jnp.sqrt(2.0 / propagator.dt),
        prop_data["pop_control_ene_shift"],
        energy_samples,
    )
    block_weight = jnp.sum(prop_data["weights"])
    block_energy = jnp.sum(energy_samples * prop_data["weights"]) / block_weight
    prop_data["pop_control_ene_shift"] = (
        0.9 * prop_data["pop_control_ene_shift"] + 0.1 * block_energy
    )


@partial(jit, static_argnums=(3, 4))
def _block_scan_free(prop_data, _x, ham_data, propagator, trial, wave_data):
    prop_data["key"], subkey = random.split(prop_data["key"])
    fields = random.normal(
        subkey,
        shape=(
            propagator.n_prop_steps,
            propagator.n_walkers,
            ham_data["chol"].shape[0],
        ),
    )
    _step_scan_wrapper = lambda x, y: _step_scan_free(
        x, y, ham_data, propagator, trial, wave_data
    )
    prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
    energy_samples = trial.calc_energy_vmap(ham_data, prop_data["walkers"], wave_data)
    # energy_samples = jnp.where(jnp.abs(energy_samples - ham_data['ene0']) > jnp.sqrt(2./propagator.dt), ham_data['ene0'], energy_samples)
    block_energy = jnp.sum(energy_samples * prop_data["overlaps"]) / jnp.sum(
        prop_data["overlaps"]
    )
    block_weight = jnp.sum(prop_data["overlaps"])
    return prop_data, (prop_data, block_energy, block_weight)


@partial(jit, static_argnums=(3, 4))
def _sr_block_scan(prop_data, _x, ham_data, propagator, trial, wave_data):
    _block_scan_wrapper = lambda x, y: _block_scan(
        x, y, ham_data, propagator, trial, wave_data
    )
    prop_data, (block_energy, block_weight) = lax.scan(
        _block_scan_wrapper, prop_data, None, length=propagator.n_ene_blocks
    )
    prop_data = propagator.stochastic_reconfiguration_local(prop_data)
    prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
    return prop_data, (block_energy, block_weight)


@partial(jit, static_argnums=(2, 3))
def _ad_block(prop_data, ham_data, propagator, trial, wave_data):
    _sr_block_scan_wrapper = lambda x, y: _sr_block_scan(
        x, y, ham_data, propagator, trial, wave_data
    )

    prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    prop_data, (block_energy, block_weight) = lax.scan(
        checkpoint(_sr_block_scan_wrapper),
        prop_data,
        None,
        length=propagator.n_sr_blocks,
    )
    prop_data["n_killed_walkers"] /= (
        propagator.n_sr_blocks * propagator.n_ene_blocks * propagator.n_walkers
    )
    return prop_data, (block_energy, block_weight)


@partial(jit, static_argnums=(0, 4, 6))
def propagate_phaseless_ad(
    ham, ham_data, coupling, observable_op, propagator, prop_data, trial, wave_data
):
    ham_data["h1"] = ham_data["h1"] + coupling * observable_op
    wave_data = trial.optimize_orbs(ham_data, wave_data)
    ham_data = ham.rot_orbs(ham_data, wave_data)
    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)

    prop_data, (block_energy, block_weight) = _ad_block(
        prop_data, ham_data, propagator, trial, wave_data
    )
    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data


@partial(jit, static_argnums=(0, 4, 6))
def propagate_phaseless_ad_nosr(
    ham, ham_data, coupling, observable_op, propagator, prop_data, trial, wave_data
):
    ham_data["h1"] = ham_data["h1"] + coupling * observable_op
    wave_data = trial.optimize_orbs(ham_data, wave_data)
    ham_data = ham.rot_orbs(ham_data, wave_data)
    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)

    prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    _block_scan_wrapper = lambda x, y: _block_scan(
        x, y, ham_data, propagator, trial, wave_data
    )
    prop_data, (block_energy, block_weight) = lax.scan(
        checkpoint(_block_scan_wrapper), prop_data, None, length=propagator.n_ene_blocks
    )
    prop_data["n_killed_walkers"] /= (
        propagator.n_sr_blocks * propagator.n_ene_blocks * propagator.n_walkers
    )
    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data


@partial(jit, static_argnums=(0, 4, 6))
def propagate_phaseless_ad_norot(
    ham, ham_data, coupling, observable_op, propagator, prop_data, trial, wave_data
):
    ham_data["h1"] = ham_data["h1"] + coupling * observable_op
    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)

    prop_data, (block_energy, block_weight) = _ad_block(
        prop_data, ham_data, propagator, trial, wave_data
    )
    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data


@partial(jit, static_argnums=(0, 4, 6))
def propagate_phaseless_ad_nosr_norot(
    ham, ham_data, coupling, observable_op, propagator, prop_data, trial, wave_data
):
    ham_data["h1"] = ham_data["h1"] + coupling * observable_op
    ham_data = ham.rot_ham(ham_data, wave_data)
    ham_data = ham.prop_ham(ham_data, propagator.dt, trial, wave_data)

    def _block_scan_wrapper(x, y):
        return _block_scan(x, y, ham_data, propagator, trial, wave_data)

    prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    prop_data, (block_energy, block_weight) = lax.scan(
        checkpoint(_block_scan_wrapper), prop_data, None, length=propagator.n_ene_blocks
    )
    prop_data["n_killed_walkers"] /= (
        propagator.n_sr_blocks * propagator.n_ene_blocks * propagator.n_walkers
    )
    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data


@partial(jit, static_argnums=(0, 2, 4))
def propagate_phaseless(ham, ham_data, propagator, prop_data, trial, wave_data):
    def _sr_block_scan_wrapper(x, y):
        return _sr_block_scan(x, y, ham_data, propagator, trial, wave_data)

    prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
    prop_data["n_killed_walkers"] = 0
    prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
    prop_data, (block_energy, block_weight) = lax.scan(
        _sr_block_scan_wrapper, prop_data, None, length=propagator.n_sr_blocks
    )
    prop_data["n_killed_walkers"] /= (
        propagator.n_sr_blocks * propagator.n_ene_blocks * propagator.n_walkers
    )
    return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data


@partial(jit, static_argnums=(0, 2, 4))
def propagate_free(ham, ham_data, propagator, prop_data, trial, wave_data):
    def _block_scan_free_wrapper(x, y):
        return _block_scan_free(x, y, ham_data, propagator, trial, wave_data)

    prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
    prop_data, (prop_data_tr, block_energy, block_weight) = lax.scan(
        _block_scan_free_wrapper, prop_data, None, length=propagator.n_blocks
    )
    return prop_data_tr, block_energy, block_weight, prop_data["key"]
