import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, lax, random, vmap

from ad_afqmc import linalg_utils, sr, wavefunctions
from ad_afqmc.wavefunctions import wave_function


@dataclass
class propagator(ABC):
    """Abstract base class for propagator classes.
    Contains methods for propagation, orthogonalization, and reconfiguration.

    Attributes:
        dt: time step
        n_walkers: number of walkers
        phaseless_epsilon: the minimum overlap with trail allowed
    """

    dt: float = 0.01
    n_walkers: int = 50
    n_exp_terms: int = 6
    phaseless_epsilon: float = 0.

    @abstractmethod
    def init_prop_data(
        self,
        trial: wave_function,
        wave_data: Any,
        ham_data: dict,
        init_walkers: Optional[Union[jax.Array, List]] = None,
    ) -> dict:
        """Initialize propagation data. If walkers are not provided they are generated
        using the trial.

        Args:
            trial: trial wave function handler
            wave_data: dictionary containing the wave function data
            ham_data: dictionary containing the Hamiltonian data
            init_walkers: initial walkers

        Returns:
            prop_data: dictionary containing the propagation data
        """
        pass

    @abstractmethod
    def stochastic_reconfiguration_local(self, prop_data: dict) -> dict:
        """Perform stochastic reconfiguration locally on a process. Jax friendly."""
        pass

    @abstractmethod
    def stochastic_reconfiguration_global(self, prop_data: dict, comm: Any) -> dict:
        """Perform stochastic reconfiguration globally across processes using MPI. Not jax friendly."""
        pass

    @abstractmethod
    def orthonormalize_walkers(self, prop_data: dict) -> dict:
        """Orthonormalize walkers."""
        pass

    # defining this separately because calculating vhs for a batch seems to be faster
    @partial(jit, static_argnums=(0,))
    def _apply_trotprop_det(
        self, exp_h1: jax.Array, vhs_i: jax.Array, walker_i: jax.Array
    ) -> jax.Array:
        """Apply the Trotterized propagator to a det."""
        walker_i = exp_h1.dot(walker_i)

        def scanned_fun(carry, x):
            carry = vhs_i.dot(carry)
            return carry, carry

        _, vhs_n_walker = lax.scan(
            scanned_fun, walker_i, jnp.arange(1, self.n_exp_terms)
        )
        walker_i = walker_i + jnp.sum(
            jnp.stack(
                [
                    vhs_n_walker[n] / math.factorial(n + 1)
                    for n in range(self.n_exp_terms - 1)
                ]
            ),
            axis=0,
        )
        walker_i = exp_h1.dot(walker_i)
        return walker_i

    def _apply_trotprop(
        self, ham_data: dict, walkers: Sequence, fields: jax.Array
    ) -> jax.Array:
        """Apply the Trotterized propagator to a batch of walkers."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    @partial(jit, static_argnums=(0, 1))
    def propagate(
        self,
        trial: wave_function,
        ham_data: dict,
        prop_data: dict,
        fields: jax.Array,
        wave_data: dict,
    ) -> dict:
        """Phaseless AFQMC propagation.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            fields: auxiliary fields
            wave_data: wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        force_bias = trial.calc_force_bias(prop_data["walkers"], ham_data, wave_data)
        field_shifts = -jnp.sqrt(self.dt) * (1.0j * force_bias - ham_data["mf_shifts"])
        shifted_fields = fields - field_shifts
        shift_term = jnp.sum(shifted_fields * ham_data["mf_shifts"], axis=1)
        fb_term = jnp.sum(
            fields * field_shifts - field_shifts * field_shifts / 2.0, axis=1
        )

        prop_data["walkers"] = self._apply_trotprop(
            ham_data, prop_data["walkers"], shifted_fields
        )

        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        #####
        overlaps_new = jnp.where(
            jnp.abs(overlaps_new) < self.phaseless_epsilon, 0.0, overlaps_new
        )
        ###
        imp_fun = (
            jnp.exp(
                -jnp.sqrt(self.dt) * shift_term
                + fb_term
                + self.dt * (prop_data["pop_control_ene_shift"] + ham_data["h0_prop"])
            )
            * overlaps_new
            / prop_data["overlaps"]
        )
        theta = jnp.angle(
            jnp.exp(-jnp.sqrt(self.dt) * shift_term)
            * overlaps_new
            / prop_data["overlaps"]
        )
        imp_fun_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)
        imp_fun_phaseless = jnp.array(
            jnp.where(jnp.isnan(imp_fun_phaseless), 0.0, imp_fun_phaseless)
        )
        imp_fun_phaseless = jnp.where(
            imp_fun_phaseless < 1.0e-3, 0.0, imp_fun_phaseless
        )
        imp_fun_phaseless = jnp.where(imp_fun_phaseless > 100.0, 0.0, imp_fun_phaseless)
        prop_data["weights"] = imp_fun_phaseless * prop_data["weights"]
        prop_data["weights"] = jnp.array(
            jnp.where(prop_data["weights"] > 100.0, 0.0, prop_data["weights"])
        )
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        prop_data["overlaps"] = overlaps_new
        return prop_data

    def propagate_free(
        self,
        trial: wave_function,
        ham_data: dict,
        prop_data: dict,
        fields: jax.Array,
        wave_data: dict,
    ) -> dict:
        """Free projection AFQMC propagation.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            fields: auxiliary fields
            wave_data: wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        raise NotImplementedError(
            "Free projection not implemented for this propagator."
        )

    def _build_propagation_intermediates(
        self, ham_data: dict, trial: wave_function, wave_data: dict
    ) -> dict:
        """Build intermediates for propagation."""
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class propagator_restricted(propagator):
    """Propagator for walkers with the same alpha and beta dets."""

    dt: float = 0.01
    n_walkers: int = 50
    n_exp_terms: int = 6
    n_batch: int = 1

    def init_prop_data(
        self,
        trial: wave_function,
        wave_data: dict,
        ham_data: dict,
        init_walkers: Optional[jax.Array] = None,
    ) -> dict:
        prop_data = {}
        prop_data["weights"] = jnp.ones(self.n_walkers)
        if init_walkers is not None:
            prop_data["walkers"] = init_walkers
        else:
            prop_data["walkers"] = trial.get_init_walkers(
                wave_data, self.n_walkers, restricted=True
            )
        energy_samples = jnp.real(
            trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        )
        e_estimate = jnp.array(jnp.sum(energy_samples) / self.n_walkers)
        prop_data["e_estimate"] = e_estimate
        prop_data["pop_control_ene_shift"] = e_estimate
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        return prop_data

    @partial(jit, static_argnums=(0,))
    def stochastic_reconfiguration_local(self, prop_data: dict) -> dict:
        prop_data["key"], subkey = random.split(prop_data["key"])
        zeta = random.uniform(subkey)
        prop_data["walkers"], prop_data["weights"] = sr.stochastic_reconfiguration(
            prop_data["walkers"], prop_data["weights"], zeta
        )
        return prop_data

    def stochastic_reconfiguration_global(self, prop_data: dict, comm: Any) -> dict:
        prop_data["key"], subkey = random.split(prop_data["key"])
        zeta = random.uniform(subkey)
        prop_data["walkers"], prop_data["weights"] = sr.stochastic_reconfiguration_mpi(
            prop_data["walkers"], prop_data["weights"], zeta, comm
        )
        return prop_data

    def orthonormalize_walkers(self, prop_data: dict) -> dict:
        prop_data["walkers"], _ = linalg_utils.qr_vmap(prop_data["walkers"])
        return prop_data

    @partial(jit, static_argnums=(0,))
    def _apply_trotprop(
        self, ham_data: dict, walkers: jax.Array, fields: jax.Array
    ) -> jax.Array:
        """Apply the propagator to a batch of walkers."""
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, batch):
            field_batch, walker_batch = batch
            vhs = (
                1.0j
                * jnp.sqrt(self.dt)
                * field_batch.dot(ham_data["chol"]).reshape(
                    batch_size, walkers.shape[1], walkers.shape[1]
                )
            )
            walkers_new = vmap(self._apply_trotprop_det, in_axes=(None, 0, 0))(
                ham_data["exp_h1"],
                vhs,
                walker_batch,
            )
            return carry, walkers_new

        _, walkers_new = lax.scan(
            scanned_fun,
            None,
            (
                fields.reshape(self.n_batch, batch_size, -1),
                walkers.reshape(
                    self.n_batch, batch_size, walkers.shape[1], walkers.shape[2]
                ),
            ),
        )
        walkers = walkers_new.reshape(n_walkers, walkers.shape[1], walkers.shape[2])
        return walkers

    @partial(jit, static_argnums=(0, 2))
    def _build_propagation_intermediates(
        self, ham_data: dict, trial: wave_function, wave_data: dict
    ) -> dict:
        rdm1 = wave_data["rdm1"]
        rdm1 = rdm1[0] + rdm1[1]
        ham_data["mf_shifts"] = 1.0j * vmap(
            lambda x: jnp.sum(x.reshape(trial.norb, trial.norb) * rdm1)
        )(ham_data["chol"])
        ham_data["mf_shifts_fp"] = ham_data["mf_shifts"] / 2.0 / trial.nelec[0]
        ham_data["h0_prop"] = (
            -ham_data["h0"] - jnp.sum(ham_data["mf_shifts"] ** 2) / 2.0
        )
        ham_data["h0_prop_fp"] = [
            (ham_data["h0_prop"] + ham_data["ene0"]) / trial.nelec[0],
            (ham_data["h0_prop"] + ham_data["ene0"]) / trial.nelec[0],
        ]
        v0 = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham_data["chol"].reshape(-1, trial.norb, trial.norb),
            ham_data["chol"].reshape(-1, trial.norb, trial.norb),
            optimize="optimal",
        )
        h1_mod = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0 - v0
        mf_shifts_r = (1.0j * ham_data["mf_shifts"]).real
        h1_mod = h1_mod - jnp.einsum(
            "g,gik->ik",
            mf_shifts_r,
            ham_data["chol"].reshape(-1, trial.norb, trial.norb),
        )
        ham_data["exp_h1"] = jsp.linalg.expm(-self.dt * h1_mod / 2.0)
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class propagator_unrestricted(propagator_restricted):
    def init_prop_data(
        self,
        trial: wave_function,
        wave_data: dict,
        ham_data: dict,
        init_walkers: Optional[Sequence] = None,
    ) -> dict:
        prop_data = {}
        prop_data["weights"] = jnp.ones(self.n_walkers)
        if init_walkers is not None:
            prop_data["walkers"] = init_walkers
        else:
            prop_data["walkers"] = trial.get_init_walkers(
                wave_data, self.n_walkers, restricted=False
            )
        energy_samples = jnp.real(
            trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
        )
        e_estimate = jnp.array(jnp.sum(energy_samples) / self.n_walkers)
        prop_data["e_estimate"] = e_estimate
        prop_data["pop_control_ene_shift"] = e_estimate
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["normed_overlaps"] = prop_data["overlaps"]
        prop_data["norms"] = jnp.ones(self.n_walkers) + 0.0j
        return prop_data

    @partial(jit, static_argnums=(0,))
    def _apply_trotprop(
        self, ham_data: dict, walkers: Sequence, fields: jax.Array
    ) -> List:
        n_walkers = walkers[0].shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, batch):
            field_batch, walker_batch_0, walker_batch_1 = batch
            vhs = (
                1.0j
                * jnp.sqrt(self.dt)
                * field_batch.dot(ham_data["chol"]).reshape(
                    batch_size, walkers[0].shape[1], walkers[0].shape[1]
                )
            )
            walkers_new_0 = vmap(self._apply_trotprop_det, in_axes=(None, 0, 0))(
                ham_data["exp_h1"][0], vhs, walker_batch_0
            )
            walkers_new_1 = vmap(self._apply_trotprop_det, in_axes=(None, 0, 0))(
                ham_data["exp_h1"][1], vhs, walker_batch_1
            )
            return carry, [walkers_new_0, walkers_new_1]

        _, walkers_new = lax.scan(
            scanned_fun,
            None,
            (
                fields.reshape(self.n_batch, batch_size, -1),
                walkers[0].reshape(
                    self.n_batch, batch_size, walkers[0].shape[1], walkers[0].shape[2]
                ),
                walkers[1].reshape(
                    self.n_batch, batch_size, walkers[1].shape[1], walkers[1].shape[2]
                ),
            ),
        )
        walkers = [
            walkers_new[0].reshape(n_walkers, walkers[0].shape[1], walkers[0].shape[2]),
            walkers_new[1].reshape(n_walkers, walkers[1].shape[1], walkers[1].shape[2]),
        ]
        return walkers

    @partial(jit, static_argnums=(0,))
    def stochastic_reconfiguration_local(self, prop_data: dict) -> dict:
        prop_data["key"], subkey = random.split(prop_data["key"])
        zeta = random.uniform(subkey)
        prop_data["walkers"], prop_data["weights"] = sr.stochastic_reconfiguration_uhf(
            prop_data["walkers"], prop_data["weights"], zeta
        )
        return prop_data

    def stochastic_reconfiguration_global(self, prop_data: dict, comm: Any) -> dict:
        prop_data["key"], subkey = random.split(prop_data["key"])
        zeta = random.uniform(subkey)
        (
            prop_data["walkers"],
            prop_data["weights"],
        ) = sr.stochastic_reconfiguration_mpi_uhf(
            prop_data["walkers"], prop_data["weights"], zeta, comm
        )
        return prop_data

    def orthonormalize_walkers(self, prop_data: dict) -> dict:
        prop_data["walkers"], _ = linalg_utils.qr_vmap_uhf(prop_data["walkers"])
        return prop_data

    def _orthogonalize_walkers(self, prop_data: dict) -> Tuple:
        prop_data["walkers"], norms = linalg_utils.qr_vmap_uhf(prop_data["walkers"])
        return prop_data, norms

    @partial(jit, static_argnums=(0))
    def _multiply_constant(self, walkers: List, constants: jax.Array) -> Sequence:
        walkers[0] = constants[0].reshape(-1, 1, 1) * walkers[0]
        walkers[1] = constants[1].reshape(-1, 1, 1) * walkers[1]
        return walkers

    @partial(jit, static_argnums=(0, 1))
    def propagate_free(
        self,
        trial: wave_function,
        ham_data,
        prop_data: dict,
        fields: jax.Array,
        wave_data: Sequence,
    ) -> dict:
        shift_term = jnp.einsum("wg,sg->sw", fields, ham_data["mf_shifts_fp"])
        constants = jnp.einsum(
            "sw,s->sw",
            jnp.exp(-jnp.sqrt(self.dt) * shift_term),
            jnp.exp(self.dt * ham_data["h0_prop_fp"]),
        )
        prop_data["walkers"] = self._apply_trotprop(
            ham_data, prop_data["walkers"], fields
        )
        prop_data["walkers"] = self._multiply_constant(prop_data["walkers"], constants)
        prop_data, norms = self._orthogonalize_walkers(prop_data)
        prop_data["norms"] *= norms[0] * norms[1]
        prop_data["overlaps"] = (
            trial.calc_overlap(prop_data["walkers"], wave_data) * prop_data["norms"]
        )
        normed_walkers, _ = linalg_utils.qr_vmap_uhf(prop_data["walkers"])
        prop_data["normed_overlaps"] = trial.calc_overlap(normed_walkers, wave_data)
        return prop_data

    @partial(jit, static_argnums=(0, 2))
    def _build_propagation_intermediates(
        self, ham_data: dict, trial: wave_function, wave_data: dict
    ) -> dict:
        rdm1 = wave_data["rdm1"]
        rdm1 = rdm1[0] + rdm1[1]
        ham_data["mf_shifts"] = 1.0j * vmap(
            lambda x: jnp.sum(x.reshape(trial.norb, trial.norb) * rdm1)
        )(ham_data["chol"])
        ham_data["mf_shifts_fp"] = jnp.stack(
            (
                ham_data["mf_shifts"] / trial.nelec[0] / 2.0,
                ham_data["mf_shifts"] / trial.nelec[1] / 2.0,
            )
        )
        ham_data["h0_prop"] = (
            -ham_data["h0"] - jnp.sum(ham_data["mf_shifts"] ** 2) / 2.0
        )
        ham_data["h0_prop_fp"] = jnp.stack(
            (
                (ham_data["h0_prop"] + ham_data["ene0"]) / trial.nelec[0] / 2.0,
                (ham_data["h0_prop"] + ham_data["ene0"]) / trial.nelec[1] / 2.0,
            )
        )
        v0 = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham_data["chol"].reshape(-1, trial.norb, trial.norb),
            ham_data["chol"].reshape(-1, trial.norb, trial.norb),
            optimize="optimal",
        )
        mf_shifts_r = (1.0j * ham_data["mf_shifts"]).real
        v1 = jnp.einsum(
            "g,gik->ik",
            mf_shifts_r,
            ham_data["chol"].reshape(-1, trial.norb, trial.norb),
        )
        h1_mod = ham_data["h1"] - jnp.array([v0 + v1, v0 + v1])
        ham_data["exp_h1"] = jnp.array(
            [
                jsp.linalg.expm(-self.dt * h1_mod[0] / 2.0),
                jsp.linalg.expm(-self.dt * h1_mod[1] / 2.0),
            ]
        )
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


class propagator_cpmc(propagator_unrestricted):
    """CPMC propagator for the Hubbard model with on-site interactions."""

    def init_prop_data(
        self,
        trial: wavefunctions.wave_function_cpmc,
        wave_data: dict,
        ham_data: dict,
        init_walkers: Optional[Sequence] = None,
    ) -> dict:
        prop_data = super().init_prop_data(trial, wave_data, ham_data, init_walkers)
        prop_data["walkers"][0] = prop_data["walkers"][0].real
        prop_data["walkers"][1] = prop_data["walkers"][1].real
        prop_data["overlaps"] = prop_data["overlaps"].real
        prop_data["greens"] = trial.calc_full_green_vmap(
            prop_data["walkers"], wave_data
        )
        gamma = jnp.arccosh(jnp.exp(self.dt * ham_data["u"] / 2))
        const = jnp.exp(-self.dt * ham_data["u"] / 2)
        prop_data["hs_constant"] = const * jnp.array(
            [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
        )
        return prop_data

    @partial(jit, static_argnums=(0, 1))
    def propagate_one_body(
        self,
        trial: wavefunctions.wave_function_cpmc,
        ham_data: dict,
        prop_data: dict,
        wave_data: dict,
    ) -> dict:
        prop_data["walkers"][0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"][0]
        )
        prop_data["walkers"][1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"][1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        prop_data["weights"] = jnp.where(
            prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"]
        )
        prop_data["overlaps"] = overlaps_new
        prop_data["greens"] = trial.calc_full_green_vmap(
            prop_data["walkers"], wave_data
        )
        return prop_data

    @partial(jit, static_argnums=(0, 1))
    def propagate(
        self,
        trial: wavefunctions.wave_function_cpmc,
        ham_data: dict,
        prop_data: dict,
        gaussian_rns: jnp.array,
        wave_data: dict,
    ) -> dict:
        """
        Propagate the walkers using the CPMC algorithm.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            gaussian_rns: Gaussian random numbers (these are converted to uniform)
            wave_data: dictionary containing the wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        # one body
        prop_data = self.propagate_one_body(trial, ham_data, prop_data, wave_data)

        # two body
        # TODO: define separate sampler that feeds uniform_rns instead of gaussian_rns
        uniform_rns = (jsp.special.erf(gaussian_rns / 2**0.5) + 1) / 2

        # iterate over sites
        def scanned_fun(carry, x):
            # field 1
            ratio_0 = trial.calc_overlap_ratio_vmap(
                carry["greens"],
                jnp.array([[0, x], [1, x]]),
                prop_data["hs_constant"][0] - 1,
            )
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            ratio_1 = trial.calc_overlap_ratio_vmap(
                carry["greens"],
                jnp.array([[0, x], [1, x]]),
                prop_data["hs_constant"][1] - 1,
            )

            ratio_1 = jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1)

            # normalize
            prob_0 = ratio_0.real / 2.0
            prob_1 = ratio_1.real / 2.0
            norm = prob_0 + prob_1
            prob_0 /= norm

            # update
            rns = uniform_rns[:, x]
            mask = rns < prob_0
            constants = jnp.where(
                mask.reshape(-1, 1),
                prop_data["hs_constant"][0],
                prop_data["hs_constant"][1],
            )
            new_walkers_up = (
                carry["walkers"][0].at[:, x, :].mul(constants[:, 0].reshape(-1, 1))
            )
            new_walkers_dn = (
                carry["walkers"][1].at[:, x, :].mul(constants[:, 1].reshape(-1, 1))
            )
            carry["walkers"] = [new_walkers_up, new_walkers_dn]
            ratios = jnp.where(mask, ratio_0, ratio_1)
            update_constants = constants - 1
            carry["greens"] = trial.update_greens_function_vmap(
                carry["greens"],
                ratios,
                jnp.array([[0, x], [1, x]]),
                update_constants,
            )
            carry["overlaps"] = ratios * carry["overlaps"]
            carry["weights"] *= norm
            return carry, x

        prop_data, _ = lax.scan(scanned_fun, prop_data, jnp.arange(trial.norb))

        # one body
        prop_data = self.propagate_one_body(trial, ham_data, prop_data, wave_data)

        prop_data["weights"] *= jnp.exp(self.dt * (prop_data["pop_control_ene_shift"]))
        prop_data["weights"] = jnp.where(
            prop_data["weights"] > 100.0, 0.0, prop_data["weights"]
        )
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        return prop_data


class propagator_cpmc_slow(propagator_cpmc, propagator_unrestricted):
    """CPMC propagator for the Hubbard model with on-site interactions."""

    @partial(jit, static_argnums=(0, 1))
    def propagate(
        self,
        trial: wavefunctions.wave_function,
        ham_data: dict,
        prop_data: dict,
        gaussian_rns: jnp.array,
        wave_data: dict,
    ) -> dict:
        """
        Propagate the walkers using the CPMC algorithm.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            gaussian_rns: Gaussian random numbers (these are converted to uniform)
            wave_data: dictionary containing the wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        # one body
        prop_data["walkers"][0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"][0]
        )
        prop_data["walkers"][1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"][1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        prop_data["weights"] = jnp.where(
            prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"]
        )
        prop_data["overlaps"] = overlaps_new

        # two body
        # TODO: define separate sampler that feeds uniform_rns instead of gaussian_rns
        uniform_rns = (jsp.special.erf(gaussian_rns / 2**0.5) + 1) / 2

        # iterate over sites
        # TODO: fast update
        def scanned_fun(carry, x):
            # field 1
            new_walkers_0_up = (
                carry["walkers"][0].at[:, x, :].mul(prop_data["hs_constant"][0, 0])
            )
            new_walkers_0_dn = (
                carry["walkers"][1].at[:, x, :].mul(prop_data["hs_constant"][0, 1])
            )
            overlaps_new_0 = trial.calc_overlap(
                [new_walkers_0_up, new_walkers_0_dn], wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            new_walkers_1_up = (
                carry["walkers"][0].at[:, x, :].mul(prop_data["hs_constant"][1, 0])
            )
            new_walkers_1_dn = (
                carry["walkers"][1].at[:, x, :].mul(prop_data["hs_constant"][1, 1])
            )
            overlaps_new_1 = trial.calc_overlap(
                [new_walkers_1_up, new_walkers_1_dn], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.array(jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1))

            # normalize
            norm = ratio_0 + ratio_1
            ratio_0 /= norm

            # update
            rns = uniform_rns[:, x]
            mask_0 = (rns < ratio_0).reshape(-1, 1, 1)
            new_walkers_up = jnp.where(mask_0, new_walkers_0_up, new_walkers_1_up)
            new_walkers_dn = jnp.where(mask_0, new_walkers_0_dn, new_walkers_1_dn)
            new_walkers = [new_walkers_up, new_walkers_dn]
            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, overlaps_new_0, overlaps_new_1)
            carry["walkers"] = new_walkers
            carry["weights"] *= norm
            carry["overlaps"] = overlaps_new
            return carry, x

        prop_data, _ = lax.scan(scanned_fun, prop_data, jnp.arange(trial.norb))

        # one body
        prop_data["walkers"][0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"][0]
        )
        prop_data["walkers"][1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"][1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        prop_data["weights"] = jnp.array(
            jnp.where(prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"])
        )
        prop_data["overlaps"] = overlaps_new

        prop_data["weights"] *= jnp.exp(self.dt * (prop_data["pop_control_ene_shift"]))
        prop_data["weights"] = jnp.where(
            prop_data["weights"] > 100.0, 0.0, prop_data["weights"]
        )
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        return prop_data

    @partial(jit, static_argnums=(0, 1))
    def propagate_1(
        self,
        trial: wavefunctions.wave_function,
        ham_data: dict,
        prop_data: dict,
        gaussian_rns: jnp.array,
        wave_data: dict,
    ) -> dict:
        """
        Propagate the walkers using the CPMC algorithm (no importance sampling).

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            gaussian_rns: Gaussian random numbers (these are converted to uniform)
            wave_data: dictionary containing the wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        # one body
        prop_data["walkers"][0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"][0]
        )
        prop_data["walkers"][1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"][1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        # prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        # prop_data["weights"] = jnp.where(
        #     prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"]
        # )
        prop_data["overlaps"] = overlaps_new

        # two body
        # TODO: define separate sampler that feeds uniform_rns instead of gaussian_rns
        uniform_rns = (jsp.special.erf(gaussian_rns / 2**0.5) + 1) / 2

        # iterate over sites
        # TODO: fast update
        def scanned_fun(carry, x):
            # field 1
            new_walkers_0_up = (
                carry["walkers"][0].at[:, x, :].mul(prop_data["hs_constant"][0, 0])
            )
            new_walkers_0_dn = (
                carry["walkers"][1].at[:, x, :].mul(prop_data["hs_constant"][0, 1])
            )
            overlaps_new_0 = trial.calc_overlap(
                [new_walkers_0_up, new_walkers_0_dn], wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            new_walkers_1_up = (
                carry["walkers"][0].at[:, x, :].mul(prop_data["hs_constant"][1, 0])
            )
            new_walkers_1_dn = (
                carry["walkers"][1].at[:, x, :].mul(prop_data["hs_constant"][1, 1])
            )
            overlaps_new_1 = trial.calc_overlap(
                [new_walkers_1_up, new_walkers_1_dn], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real
            ratio_1 = jnp.array(jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1))

            # normalize
            norm = ratio_0 + ratio_1
            ratio_0 /= norm

            # update
            rns = uniform_rns[:, x]
            mask_0 = (rns < 0.5).reshape(-1, 1, 1)
            new_walkers_up = jnp.where(mask_0, new_walkers_0_up, new_walkers_1_up)
            new_walkers_dn = jnp.where(mask_0, new_walkers_0_dn, new_walkers_1_dn)
            new_walkers = [new_walkers_up, new_walkers_dn]
            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, overlaps_new_0, overlaps_new_1)
            carry["walkers"] = new_walkers
            # carry["weights"] *= (overlaps_new / carry["overlaps"]).real
            carry["overlaps"] = overlaps_new
            return carry, x

        prop_data, _ = lax.scan(
            scanned_fun, prop_data, jnp.arange(ham_data["chol"].shape[0])
        )  # TODO: chol will be removed from ham_data for hubbard

        # one body
        prop_data["walkers"][0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"][0]
        ) * jnp.exp(self.dt * (prop_data["e_estimate"]) / 2)
        prop_data["walkers"][1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"][1]
        ) * jnp.exp(self.dt * (prop_data["e_estimate"]) / 2)
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        # prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        # prop_data["weights"] = jnp.where(
        #     prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"]
        # )
        prop_data["overlaps"] = overlaps_new
        prop_data["weights"] = overlaps_new.real
        # prop_data["weights"] *= jnp.exp(self.dt * (prop_data["e_estimate"]))
        # prop_data["weights"] *= jnp.exp(self.dt * (prop_data["pop_control_ene_shift"]))
        # prop_data["weights"] = jnp.where(
        #     prop_data["weights"] > 100.0, 0.0, prop_data["weights"]
        # )
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        return prop_data


@dataclass
class propagator_cpmc_nn(propagator_cpmc, propagator_unrestricted):
    neighbors: Optional[tuple] = None

    def init_prop_data(
        self,
        trial: wavefunctions.wave_function_cpmc,
        wave_data: dict,
        ham_data: dict,
        init_walkers: Optional[Sequence] = None,
    ) -> dict:
        prop_data = super().init_prop_data(trial, wave_data, ham_data, init_walkers)
        prop_data["greens"] = trial.calc_full_green_vmap(
            prop_data["walkers"], wave_data
        )
        gamma = jnp.arccosh(jnp.exp(self.dt * ham_data["u"] / 2))
        const = jnp.exp(-self.dt * ham_data["u"] / 2)
        prop_data["hs_constant_onsite"] = const * jnp.array(
            [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
        )
        gamma_1 = jnp.arccosh(jnp.exp(self.dt * ham_data["u_1"] / 2))
        const_1 = jnp.exp(-self.dt * ham_data["u_1"] / 2)
        prop_data["hs_constant_nn"] = const_1 * jnp.array(
            [
                [jnp.exp(gamma_1), jnp.exp(-gamma_1)],
                [jnp.exp(-gamma_1), jnp.exp(gamma_1)],
            ]
        )
        return prop_data

    @partial(jit, static_argnums=(0, 1))
    def propagate(
        self,
        trial: wavefunctions.wave_function_cpmc,
        ham_data: dict,
        prop_data: dict,
        _gaussian_rns: jnp.array,
        wave_data: dict,
    ) -> dict:
        """
        Propagate the walkers using the CPMC algorithm.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            _gaussian_rns: Gaussian random numbers (not used, need to define cpmc sampler)
            wave_data: dictionary containing the wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        # one body
        prop_data = self.propagate_one_body(trial, ham_data, prop_data, wave_data)

        # two body
        # on site
        prop_data["key"], subkey = random.split(prop_data["key"])
        uniform_rns = random.uniform(subkey, shape=(self.n_walkers, trial.norb))

        # iterate over sites
        def scanned_fun(carry, x):
            # field 1
            ratio_0 = trial.calc_overlap_ratio_vmap(
                carry["greens"],
                jnp.array([[0, x], [1, x]]),
                prop_data["hs_constant_onsite"][0] - 1,
            )
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            ratio_1 = trial.calc_overlap_ratio_vmap(
                carry["greens"],
                jnp.array([[0, x], [1, x]]),
                prop_data["hs_constant_onsite"][1] - 1,
            )

            ratio_1 = jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1)

            # normalize
            prob_0 = ratio_0.real / 2.0
            prob_1 = ratio_1.real / 2.0
            norm = prob_0 + prob_1
            prob_0 /= norm

            # update
            rns = uniform_rns[:, x]
            mask = rns < prob_0
            constants = jnp.where(
                mask.reshape(-1, 1),
                prop_data["hs_constant_onsite"][0],
                prop_data["hs_constant_onsite"][1],
            )
            new_walkers_up = (
                carry["walkers"][0].at[:, x, :].mul(constants[:, 0].reshape(-1, 1))
            )
            new_walkers_dn = (
                carry["walkers"][1].at[:, x, :].mul(constants[:, 1].reshape(-1, 1))
            )
            carry["walkers"] = [new_walkers_up, new_walkers_dn]
            ratios = jnp.where(mask, ratio_0, ratio_1)
            update_constants = constants - 1
            carry["greens"] = trial.update_greens_function_vmap(
                carry["greens"],
                ratios,
                jnp.array([[0, x], [1, x]]),
                update_constants,
            )
            carry["overlaps"] = ratios * carry["overlaps"]
            carry["weights"] *= norm
            return carry, x

        prop_data, _ = lax.scan(scanned_fun, prop_data, jnp.arange(trial.norb))

        prop_data["key"], subkey = random.split(prop_data["key"])
        uniform_rns_1 = random.uniform(
            subkey, shape=(self.n_walkers, 4, jnp.array(self.neighbors).shape[0])
        )

        # neighbors
        def scanned_fun_1(carry, x):
            site_i = jnp.array(self.neighbors)[x][0]
            site_j = jnp.array(self.neighbors)[x][1]
            # up up
            # field 1
            ratio_0 = trial.calc_overlap_ratio_vmap(
                carry["greens"],
                jnp.array([[0, site_i], [0, site_j]]),
                prop_data["hs_constant_nn"][0] - 1,
            )
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            ratio_1 = trial.calc_overlap_ratio_vmap(
                carry["greens"],
                jnp.array([[0, site_i], [0, site_j]]),
                prop_data["hs_constant_nn"][1] - 1,
            )

            ratio_1 = jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1)

            # normalize
            prob_0 = ratio_0.real / 2.0
            prob_1 = ratio_1.real / 2.0
            norm = prob_0 + prob_1
            prob_0 /= norm

            # update
            rns = uniform_rns_1[:, 0, x]
            mask = rns < prob_0
            constants = jnp.where(
                mask.reshape(-1, 1),
                prop_data["hs_constant_nn"][0],
                prop_data["hs_constant_nn"][1],
            )
            new_walkers_up = (
                carry["walkers"][0].at[:, site_i, :].mul(constants[:, 0].reshape(-1, 1))
            )
            new_walkers_up = new_walkers_up.at[:, site_j, :].mul(
                constants[:, 1].reshape(-1, 1)
            )
            carry["walkers"] = [new_walkers_up, carry["walkers"][1]]
            ratios = jnp.where(mask, ratio_0, ratio_1)
            update_constants = constants - 1
            carry["greens"] = trial.update_greens_function_vmap(
                carry["greens"],
                ratios,
                jnp.array([[0, site_i], [0, site_j]]),
                update_constants,
            )
            carry["overlaps"] = ratios * carry["overlaps"]
            carry["weights"] *= norm

            # up dn
            ratio_0 = trial.calc_overlap_ratio_vmap(
                carry["greens"],
                jnp.array([[0, site_i], [1, site_j]]),
                prop_data["hs_constant_nn"][0] - 1,
            )
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            ratio_1 = trial.calc_overlap_ratio_vmap(
                carry["greens"],
                jnp.array([[0, site_i], [1, site_j]]),
                prop_data["hs_constant_nn"][1] - 1,
            )

            ratio_1 = jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1)

            # normalize
            prob_0 = ratio_0.real / 2.0
            prob_1 = ratio_1.real / 2.0
            norm = prob_0 + prob_1
            prob_0 /= norm

            # update
            rns = uniform_rns_1[:, 1, x]
            mask = rns < prob_0
            constants = jnp.where(
                mask.reshape(-1, 1),
                prop_data["hs_constant_nn"][0],
                prop_data["hs_constant_nn"][1],
            )
            new_walkers_up = (
                carry["walkers"][0].at[:, site_i, :].mul(constants[:, 0].reshape(-1, 1))
            )
            new_walkers_dn = (
                carry["walkers"][1].at[:, site_j, :].mul(constants[:, 1].reshape(-1, 1))
            )
            carry["walkers"] = [new_walkers_up, new_walkers_dn]
            ratios = jnp.where(mask, ratio_0, ratio_1)
            update_constants = constants - 1
            carry["greens"] = trial.update_greens_function_vmap(
                carry["greens"],
                ratios,
                jnp.array([[0, site_i], [1, site_j]]),
                update_constants,
            )
            carry["overlaps"] = ratios * carry["overlaps"]
            carry["weights"] *= norm

            # dn up
            ratio_0 = trial.calc_overlap_ratio_vmap(
                carry["greens"],
                jnp.array([[1, site_i], [0, site_j]]),
                prop_data["hs_constant_nn"][0] - 1,
            )
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            ratio_1 = trial.calc_overlap_ratio_vmap(
                carry["greens"],
                jnp.array([[1, site_i], [0, site_j]]),
                prop_data["hs_constant_nn"][1] - 1,
            )

            ratio_1 = jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1)

            # normalize
            prob_0 = ratio_0.real / 2.0
            prob_1 = ratio_1.real / 2.0
            norm = prob_0 + prob_1
            prob_0 /= norm

            # update
            rns = uniform_rns_1[:, 2, x]
            mask = rns < prob_0
            constants = jnp.where(
                mask.reshape(-1, 1),
                prop_data["hs_constant_nn"][0],
                prop_data["hs_constant_nn"][1],
            )
            new_walkers_dn = (
                carry["walkers"][1].at[:, site_i, :].mul(constants[:, 0].reshape(-1, 1))
            )
            new_walkers_up = (
                carry["walkers"][0].at[:, site_j, :].mul(constants[:, 1].reshape(-1, 1))
            )
            carry["walkers"] = [new_walkers_up, new_walkers_dn]
            ratios = jnp.where(mask, ratio_0, ratio_1)
            update_constants = constants - 1
            carry["greens"] = trial.update_greens_function_vmap(
                carry["greens"],
                ratios,
                jnp.array([[1, site_i], [0, site_j]]),
                update_constants,
            )
            carry["overlaps"] = ratios * carry["overlaps"]
            carry["weights"] *= norm

            # dn dn
            # field 1
            ratio_0 = trial.calc_overlap_ratio_vmap(
                carry["greens"],
                jnp.array([[1, site_i], [1, site_j]]),
                prop_data["hs_constant_nn"][0] - 1,
            )
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            ratio_1 = trial.calc_overlap_ratio_vmap(
                carry["greens"],
                jnp.array([[1, site_i], [1, site_j]]),
                prop_data["hs_constant_nn"][1] - 1,
            )

            ratio_1 = jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1)

            # normalize
            prob_0 = ratio_0.real / 2.0
            prob_1 = ratio_1.real / 2.0
            norm = prob_0 + prob_1
            prob_0 /= norm

            # update
            rns = uniform_rns_1[:, 3, x]
            mask = rns < prob_0
            constants = jnp.where(
                mask.reshape(-1, 1),
                prop_data["hs_constant_nn"][0],
                prop_data["hs_constant_nn"][1],
            )
            new_walkers_dn = (
                carry["walkers"][1].at[:, site_i, :].mul(constants[:, 0].reshape(-1, 1))
            )
            new_walkers_dn = new_walkers_dn.at[:, site_j, :].mul(
                constants[:, 1].reshape(-1, 1)
            )
            carry["walkers"] = [carry["walkers"][0], new_walkers_dn]
            ratios = jnp.where(mask, ratio_0, ratio_1)
            update_constants = constants - 1
            carry["greens"] = trial.update_greens_function_vmap(
                carry["greens"],
                ratios,
                jnp.array([[1, site_i], [1, site_j]]),
                update_constants,
            )
            carry["overlaps"] = ratios * carry["overlaps"]
            carry["weights"] *= norm

            return carry, x

        prop_data, _ = lax.scan(
            scanned_fun_1, prop_data, jnp.arange(jnp.array(self.neighbors).shape[0])
        )

        # one body
        prop_data = self.propagate_one_body(trial, ham_data, prop_data, wave_data)

        prop_data["weights"] *= jnp.exp(self.dt * (prop_data["pop_control_ene_shift"]))
        prop_data["weights"] = jnp.where(
            prop_data["weights"] > 100.0, 0.0, prop_data["weights"]
        )
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        return prop_data

    def __hash__(self) -> int:
        return hash((self.dt, self.n_walkers, self.neighbors))


@dataclass
class propagator_cpmc_nn_slow(propagator_unrestricted):
    neighbors: Optional[tuple] = None

    def init_prop_data(
        self,
        trial: wavefunctions.wave_function_cpmc,
        wave_data: dict,
        ham_data: dict,
        init_walkers: Optional[Sequence] = None,
    ) -> dict:
        prop_data = super().init_prop_data(trial, wave_data, ham_data, init_walkers)
        prop_data["greens"] = trial.calc_full_green_vmap(
            prop_data["walkers"], wave_data
        )
        gamma = jnp.arccosh(jnp.exp(self.dt * ham_data["u"] / 2))
        const = jnp.exp(-self.dt * ham_data["u"] / 2)
        prop_data["hs_constant_onsite"] = const * jnp.array(
            [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
        )
        gamma_1 = jnp.arccosh(jnp.exp(self.dt * ham_data["u_1"] / 2))
        const_1 = jnp.exp(-self.dt * ham_data["u_1"] / 2)
        prop_data["hs_constant_nn"] = const_1 * jnp.array(
            [
                [jnp.exp(gamma_1), jnp.exp(-gamma_1)],
                [jnp.exp(-gamma_1), jnp.exp(gamma_1)],
            ]
        )
        return prop_data

    @partial(jit, static_argnums=(0, 1))
    def propagate(
        self,
        trial: wavefunctions.wave_function_cpmc,
        ham_data: dict,
        prop_data: dict,
        _gaussian_rns: jnp.array,
        wave_data: dict,
    ) -> dict:
        """
        Propagate the walkers using the CPMC algorithm.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            gaussian_rns: Gaussian random numbers (these are converted to uniform)
            wave_data: dictionary containing the wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        # one body
        prop_data["walkers"][0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"][0]
        )
        prop_data["walkers"][1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"][1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        prop_data["weights"] = jnp.where(
            prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"]
        )
        prop_data["overlaps"] = overlaps_new

        # two body
        # on site
        prop_data["key"], subkey = random.split(prop_data["key"])
        uniform_rns = random.uniform(subkey, shape=(self.n_walkers, trial.norb))

        # iterate over sites
        # TODO: fast update
        def scanned_fun(carry, x):
            # field 1
            new_walkers_0_up = (
                carry["walkers"][0]
                .at[:, x, :]
                .mul(prop_data["hs_constant_onsite"][0, 0])
            )
            new_walkers_0_dn = (
                carry["walkers"][1]
                .at[:, x, :]
                .mul(prop_data["hs_constant_onsite"][0, 1])
            )
            overlaps_new_0 = trial.calc_overlap(
                [new_walkers_0_up, new_walkers_0_dn], wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            new_walkers_1_up = (
                carry["walkers"][0]
                .at[:, x, :]
                .mul(prop_data["hs_constant_onsite"][1, 0])
            )
            new_walkers_1_dn = (
                carry["walkers"][1]
                .at[:, x, :]
                .mul(prop_data["hs_constant_onsite"][1, 1])
            )
            overlaps_new_1 = trial.calc_overlap(
                [new_walkers_1_up, new_walkers_1_dn], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.array(jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1))

            # normalize
            norm = ratio_0 + ratio_1
            ratio_0 /= norm

            # update
            rns = uniform_rns[:, x]
            mask_0 = (rns < ratio_0).reshape(-1, 1, 1)
            new_walkers_up = jnp.where(mask_0, new_walkers_0_up, new_walkers_1_up)
            new_walkers_dn = jnp.where(mask_0, new_walkers_0_dn, new_walkers_1_dn)
            new_walkers = [new_walkers_up, new_walkers_dn]
            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, overlaps_new_0, overlaps_new_1)
            carry["walkers"] = new_walkers
            carry["weights"] *= norm
            carry["overlaps"] = overlaps_new
            return carry, x

        prop_data, _ = lax.scan(scanned_fun, prop_data, jnp.arange(trial.norb))

        prop_data["key"], subkey = random.split(prop_data["key"])
        uniform_rns_1 = random.uniform(
            subkey, shape=(self.n_walkers, 4, jnp.array(self.neighbors).shape[0])
        )

        # neighbors
        def scanned_fun_1(carry, x):
            site_i = jnp.array(self.neighbors)[x][0]
            site_j = jnp.array(self.neighbors)[x][1]
            # up up
            # field 1
            new_walkers_0_up = (
                carry["walkers"][0]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][0, 0])
            )
            new_walkers_0_up = new_walkers_0_up.at[:, site_j, :].mul(
                prop_data["hs_constant_nn"][0, 1]
            )
            overlaps_new_0 = trial.calc_overlap(
                [new_walkers_0_up, carry["walkers"][1]], wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            new_walkers_1_up = (
                carry["walkers"][0]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][1, 0])
            )
            new_walkers_1_up = new_walkers_1_up.at[:, site_j, :].mul(
                prop_data["hs_constant_nn"][1, 1]
            )
            overlaps_new_1 = trial.calc_overlap(
                [new_walkers_1_up, carry["walkers"][1]], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.array(jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1))

            # normalize
            norm = ratio_0 + ratio_1
            ratio_0 /= norm

            # update
            rns = uniform_rns_1[:, 0, x]
            mask_0 = (rns < ratio_0).reshape(-1, 1, 1)
            new_walkers_up = jnp.where(mask_0, new_walkers_0_up, new_walkers_1_up)
            new_walkers = [new_walkers_up, carry["walkers"][1]]
            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, overlaps_new_0, overlaps_new_1)
            carry["walkers"] = new_walkers
            carry["weights"] *= norm
            carry["overlaps"] = overlaps_new

            # up dn
            # field 1
            new_walkers_0_up = (
                carry["walkers"][0]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][0, 0])
            )
            new_walkers_0_dn = (
                carry["walkers"][1]
                .at[:, site_j, :]
                .mul(prop_data["hs_constant_nn"][0, 1])
            )
            overlaps_new_0 = trial.calc_overlap(
                [new_walkers_0_up, new_walkers_0_dn], wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            new_walkers_1_up = (
                carry["walkers"][0]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][1, 0])
            )
            new_walkers_1_dn = (
                carry["walkers"][1]
                .at[:, site_j, :]
                .mul(prop_data["hs_constant_nn"][1, 1])
            )
            overlaps_new_1 = trial.calc_overlap(
                [new_walkers_1_up, new_walkers_1_dn], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.array(jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1))

            # normalize
            norm = ratio_0 + ratio_1
            ratio_0 /= norm

            # update
            rns = uniform_rns_1[:, 1, x]
            mask_0 = (rns < ratio_0).reshape(-1, 1, 1)
            new_walkers_up = jnp.where(mask_0, new_walkers_0_up, new_walkers_1_up)
            new_walkers_dn = jnp.where(mask_0, new_walkers_0_dn, new_walkers_1_dn)
            new_walkers = [new_walkers_up, new_walkers_dn]
            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, overlaps_new_0, overlaps_new_1)
            carry["walkers"] = new_walkers
            carry["weights"] *= norm
            carry["overlaps"] = overlaps_new

            # dn up
            # field 1
            new_walkers_0_dn = (
                carry["walkers"][1]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][0, 0])
            )
            new_walkers_0_up = (
                carry["walkers"][0]
                .at[:, site_j, :]
                .mul(prop_data["hs_constant_nn"][0, 1])
            )
            overlaps_new_0 = trial.calc_overlap(
                [new_walkers_0_up, new_walkers_0_dn], wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            new_walkers_1_dn = (
                carry["walkers"][1]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][1, 0])
            )
            new_walkers_1_up = (
                carry["walkers"][0]
                .at[:, site_j, :]
                .mul(prop_data["hs_constant_nn"][1, 1])
            )
            overlaps_new_1 = trial.calc_overlap(
                [new_walkers_1_up, new_walkers_1_dn], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.array(jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1))

            # normalize
            norm = ratio_0 + ratio_1
            ratio_0 /= norm

            # update
            rns = uniform_rns_1[:, 2, x]
            mask_0 = (rns < ratio_0).reshape(-1, 1, 1)
            new_walkers_up = jnp.where(mask_0, new_walkers_0_up, new_walkers_1_up)
            new_walkers_dn = jnp.where(mask_0, new_walkers_0_dn, new_walkers_1_dn)
            new_walkers = [new_walkers_up, new_walkers_dn]
            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, overlaps_new_0, overlaps_new_1)
            carry["walkers"] = new_walkers
            carry["weights"] *= norm
            carry["overlaps"] = overlaps_new

            # dn dn
            # field 1
            new_walkers_0_dn = (
                carry["walkers"][1]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][0, 0])
            )
            new_walkers_0_dn = new_walkers_0_dn.at[:, site_j, :].mul(
                prop_data["hs_constant_nn"][0, 1]
            )
            overlaps_new_0 = trial.calc_overlap(
                [carry["walkers"][0], new_walkers_0_dn], wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            new_walkers_1_dn = (
                carry["walkers"][1]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][1, 0])
            )
            new_walkers_1_dn = new_walkers_1_dn.at[:, site_j, :].mul(
                prop_data["hs_constant_nn"][1, 1]
            )
            overlaps_new_1 = trial.calc_overlap(
                [carry["walkers"][0], new_walkers_1_dn], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.array(jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1))

            # normalize
            norm = ratio_0 + ratio_1
            ratio_0 /= norm

            # update
            rns = uniform_rns_1[:, 3, x]
            mask_0 = (rns < ratio_0).reshape(-1, 1, 1)
            new_walkers_dn = jnp.where(mask_0, new_walkers_0_dn, new_walkers_1_dn)
            new_walkers = [carry["walkers"][0], new_walkers_dn]
            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, overlaps_new_0, overlaps_new_1)
            carry["walkers"] = new_walkers
            carry["weights"] *= norm
            carry["overlaps"] = overlaps_new

            return carry, x

        prop_data, _ = lax.scan(
            scanned_fun_1, prop_data, jnp.arange(jnp.array(self.neighbors).shape[0])
        )

        # one body
        prop_data["walkers"][0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"][0]
        )
        prop_data["walkers"][1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"][1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        prop_data["weights"] = jnp.array(
            jnp.where(prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"])
        )
        prop_data["overlaps"] = overlaps_new

        prop_data["weights"] *= jnp.exp(self.dt * (prop_data["pop_control_ene_shift"]))
        prop_data["weights"] = jnp.where(
            prop_data["weights"] > 100.0, 0.0, prop_data["weights"]
        )
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        return prop_data

    def __hash__(self) -> int:
        return hash((self.dt, self.n_walkers, self.neighbors))


class propagator_cpmc_continuous(propagator_unrestricted):

    @partial(jit, static_argnums=(0, 1))
    def propagate(
        self,
        trial: wavefunctions.wave_function_cpmc,
        ham_data: dict,
        prop_data: dict,
        gaussian_rns: jnp.array,
        wave_data: dict,
    ) -> dict:
        """
        Propagate the walkers using the CPMC algorithm.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            gaussian_rns: Gaussian random numbers
            wave_data: dictionary containing the wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        # one body
        prop_data["walkers"][0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"][0]
        )
        prop_data["walkers"][1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"][1]
        )

        # two body
        green_diag = trial.calc_green_diagonal_vmap(prop_data["walkers"], wave_data)
        force_bias = (
            0.0 * ham_data["hs_constant"] * (green_diag[:, 0] - green_diag[:, 1])
        )
        shifted_fields = gaussian_rns - force_bias
        fb_term = 0.0 * jnp.sum(
            gaussian_rns * force_bias - force_bias * force_bias / 2.0, axis=1
        )
        prop_data["walkers"][0] = jnp.einsum(
            "wi,wij->wij",
            jnp.exp(shifted_fields * ham_data["hs_constant"]),
            prop_data["walkers"][0],
            optimize="optimal",
        )
        prop_data["walkers"][1] = jnp.einsum(
            "wi,wij->wij",
            jnp.exp(-shifted_fields * ham_data["hs_constant"]),
            prop_data["walkers"][1],
            optimize="optimal",
        )

        # one body
        prop_data["walkers"][0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"][0]
        )
        prop_data["walkers"][1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"][1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)

        imp_fun = (
            jnp.exp(self.dt * (prop_data["pop_control_ene_shift"]) + fb_term)
            * overlaps_new
            / prop_data["overlaps"]
        )
        prop_data["weights"] *= imp_fun.real
        prop_data["weights"] = jnp.array(
            jnp.where(prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"])
        )
        prop_data["overlaps"] = overlaps_new
        prop_data["weights"] = jnp.where(
            prop_data["weights"] > 100.0, 0.0, prop_data["weights"]
        )
        # prop_data["weights"] = overlaps_new.real
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        return prop_data
