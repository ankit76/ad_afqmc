import os

# os.environ["XLA_FLAGS"] = (
#     "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# )
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence, Tuple

from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from jax import checkpoint, jit, lax, random

# import stat_utils
from ad_afqmc import linalg_utils
from ad_afqmc.hamiltonian import hamiltonian
from ad_afqmc.propagation import propagator
from ad_afqmc.wavefunctions import wave_function

print = partial(print, flush=True)


@dataclass
class sampler:
    n_prop_steps: int = 50
    n_ene_blocks: int = 50
    n_sr_blocks: int = 1
    n_blocks: int = 50

    @partial(jit, static_argnums=(0, 4, 5))
    def _step_scan(
        self,
        prop_data: dict,
        fields: jnp.ndarray,
        ham_data: dict,
        prop: propagator,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[dict, jnp.ndarray]:
        """Phaseless propagation scan function over steps."""
        prop_data = prop.propagate(trial, ham_data, prop_data, fields, wave_data)
        return prop_data, fields

    @partial(jit, static_argnums=(0, 4, 5))
    def _step_scan_free(
        self,
        prop_data: dict,
        fields: jnp.ndarray,
        ham_data: dict,
        prop: propagator,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[dict, jnp.ndarray]:
        """Free propagation scan function over steps."""
        prop_data = prop.propagate_free(trial, ham_data, prop_data, fields, wave_data)
        return prop_data, fields

    @partial(jit, static_argnums=(0, 4, 5))
    def _block_scan(
        self,
        prop_data: dict,
        _x: Any,
        ham_data: dict,
        prop: propagator,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Block scan function. Propagation and energy calculation."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                prop.n_walkers,
                ham_data["chol"].shape[0],
            ),
        )
        _step_scan_wrapper = lambda x, y: self._step_scan(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
        prop_data["n_killed_walkers"] += prop_data["weights"].size - jnp.count_nonzero(
            prop_data["weights"]
        )
        prop_data = prop.orthonormalize_walkers(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        energy_samples = jnp.real(
            trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        )
        energy_samples = jnp.where(
            jnp.abs(energy_samples - prop_data["pop_control_ene_shift"])
            > jnp.sqrt(2.0 / prop.dt),
            prop_data["pop_control_ene_shift"],
            energy_samples,
        )
        block_weight = jnp.sum(prop_data["weights"])
        block_energy = jnp.sum(energy_samples * prop_data["weights"]) / block_weight
        prop_data["pop_control_ene_shift"] = (
            0.9 * prop_data["pop_control_ene_shift"] + 0.1 * block_energy
        )
        return prop_data, (block_energy, block_weight)

    @partial(jit, static_argnums=(0, 4, 5))
    def _block_scan_free(
        self,
        prop_data: dict,
        _x: Any,
        ham_data: dict,
        prop: propagator,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[dict, jnp.ndarray, jnp.ndarray]]:
        """Block scan function for free propagation."""
        prop_data["key"], subkey = random.split(prop_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                self.n_prop_steps,
                prop.n_walkers,
                ham_data["chol"].shape[0],
            ),
        )
        _step_scan_wrapper = lambda x, y: self._step_scan_free(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, _ = lax.scan(_step_scan_wrapper, prop_data, fields)
        energy_samples = trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        # energy_samples = jnp.where(jnp.abs(energy_samples - ham_data['ene0']) > jnp.sqrt(2./propagator.dt), ham_data['ene0'],     energy_samples)
        block_energy = jnp.sum(energy_samples * prop_data["overlaps"]) / jnp.sum(
            prop_data["overlaps"]
        )
        block_weight = jnp.sum(prop_data["overlaps"])
        return prop_data, (prop_data, block_energy, block_weight)

    @partial(jit, static_argnums=(0, 4, 5))
    def _sr_block_scan(
        self,
        prop_data: dict,
        _x: Any,
        ham_data: dict,
        prop: propagator,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jnp.ndarray, jnp.ndarray]]:
        _block_scan_wrapper = lambda x, y: self._block_scan(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, (block_energy, block_weight) = lax.scan(
            _block_scan_wrapper, prop_data, None, length=self.n_ene_blocks
        )
        prop_data = prop.stochastic_reconfiguration_local(prop_data)
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data, (block_energy, block_weight)

    @partial(jit, static_argnums=(0, 3, 4))
    def _ad_block(
        self,
        prop_data: dict,
        ham_data: dict,
        prop: propagator,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[dict, Tuple[jnp.ndarray, jnp.ndarray]]:
        _sr_block_scan_wrapper = lambda x, y: self._sr_block_scan(
            x, y, ham_data, prop, trial, wave_data
        )

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] = 0
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data, (block_energy, block_weight) = lax.scan(
            checkpoint(_sr_block_scan_wrapper),
            prop_data,
            None,
            length=self.n_sr_blocks,
        )
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )
        return prop_data, (block_energy, block_weight)

    @partial(jit, static_argnums=(0, 1, 5, 7))
    def propagate_phaseless_ad(
        self,
        ham: hamiltonian,
        ham_data: dict,
        coupling: float,
        observable_op: Sequence,
        prop: propagator,
        prop_data: dict,
        trial: wave_function,
        wave_data: Any,
    ) -> Tuple[jnp.ndarray, dict]:
        ham_data["h1"] = ham_data["h1"] + coupling * observable_op
        wave_data = trial.optimize(ham_data, wave_data)
        # ham_data = ham.rot_orbs(ham_data, wave_data)
        ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
        ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)

        prop_data, (block_energy, block_weight) = self._ad_block(
            prop_data, ham_data, prop, trial, wave_data
        )
        return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data

    # for 2-rdm
    @partial(jit, static_argnums=(0, 1, 5, 7))
    def propagate_phaseless_ad_1(
        self,
        ham: hamiltonian,
        ham_data: dict,
        coupling: float,
        observable_op: Sequence,
        prop: propagator,
        prop_data: dict,
        trial: wave_function,
        wave_data: dict,
    ):
        # modify ham_data
        observable_op = (
            observable_op
            + jnp.transpose(observable_op, (2, 3, 0, 1))
            + jnp.transpose(observable_op, (1, 0, 3, 2))
            + jnp.transpose(observable_op, (3, 2, 1, 0))
        ) / 4.0
        norb = ham.norb
        ham_data["chol"] = linalg_utils.modified_cholesky(
            observable_op.reshape(norb**2, norb**2), norb, ham.nchol
        )
        wave_data = trial.optimize(ham_data, wave_data)
        # ham_data = ham.rot_orbs(ham_data, wave_data)
        ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
        ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)

        prop_data, (block_energy, block_weight) = self._ad_block(
            prop_data, ham_data, prop, trial, wave_data
        )
        return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data

    @partial(jit, static_argnums=(0, 1, 5, 7))
    def propagate_phaseless_ad_nosr(
        self,
        ham: hamiltonian,
        ham_data: dict,
        coupling: float,
        observable_op: Sequence,
        prop: propagator,
        prop_data: dict,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[jnp.ndarray, dict]:
        ham_data["h1"] = ham_data["h1"] + coupling * observable_op
        wave_data = trial.optimize(ham_data, wave_data)
        # ham_data = ham.rot_orbs(ham_data, wave_data)
        ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
        ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] = 0
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        _block_scan_wrapper = lambda x, y: self._block_scan(
            x, y, ham_data, prop, trial, wave_data
        )
        prop_data, (block_energy, block_weight) = lax.scan(
            checkpoint(_block_scan_wrapper),
            prop_data,
            None,
            length=self.n_ene_blocks,
        )
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )
        return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data

    @partial(jit, static_argnums=(0, 1, 5, 7))
    def propagate_phaseless_ad_norot(
        self,
        ham: hamiltonian,
        ham_data: dict,
        coupling: float,
        observable_op: Sequence,
        prop: propagator,
        prop_data: dict,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[jnp.ndarray, dict]:
        ham_data["h1"] = ham_data["h1"] + coupling * observable_op
        ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
        ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)

        prop_data, (block_energy, block_weight) = self._ad_block(
            prop_data, ham_data, prop, trial, wave_data
        )
        return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data

    @partial(jit, static_argnums=(0, 1, 5, 7))
    def propagate_phaseless_ad_nosr_norot(
        self,
        ham: hamiltonian,
        ham_data: dict,
        coupling: float,
        observable_op: Sequence,
        prop: propagator,
        prop_data: dict,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[jnp.ndarray, dict]:
        ham_data["h1"] = ham_data["h1"] + coupling * observable_op
        ham_data = ham.rot_ham(ham_data, wave_data)
        ham_data = ham.prop_ham(ham_data, prop.dt, trial, wave_data)

        def _block_scan_wrapper(x, y):
            return self._block_scan(x, y, ham_data, prop, trial, wave_data)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] = 0
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data, (block_energy, block_weight) = lax.scan(
            checkpoint(_block_scan_wrapper),
            prop_data,
            None,
            length=self.n_ene_blocks,
        )
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )
        return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data

    @partial(jit, static_argnums=(0, 1, 3, 5))
    def propagate_phaseless(
        self,
        ham: hamiltonian,
        ham_data: dict,
        prop: propagator,
        prop_data: dict,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[jnp.ndarray, dict]:
        def _sr_block_scan_wrapper(x, y):
            return self._sr_block_scan(x, y, ham_data, prop, trial, wave_data)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["n_killed_walkers"] = 0
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data, (block_energy, block_weight) = lax.scan(
            _sr_block_scan_wrapper, prop_data, None, length=self.n_sr_blocks
        )
        prop_data["n_killed_walkers"] /= (
            self.n_sr_blocks * self.n_ene_blocks * prop.n_walkers
        )
        return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), prop_data

    @partial(jit, static_argnums=(0, 1, 3, 5))
    def propagate_free(
        self,
        ham: hamiltonian,
        ham_data: dict,
        prop: propagator,
        prop_data: dict,
        trial: wave_function,
        wave_data: dict,
    ) -> Tuple[jnp.ndarray, dict]:
        def _block_scan_free_wrapper(x, y):
            return self._block_scan_free(x, y, ham_data, prop, trial, wave_data)

        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data, (prop_data_tr, block_energy, block_weight) = lax.scan(
            _block_scan_free_wrapper, prop_data, None, length=self.n_blocks
        )
        return prop_data_tr, block_energy, block_weight, prop_data["key"]

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
