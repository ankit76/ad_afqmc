import math
import os

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Sequence

# os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, lax, random, vmap

from ad_afqmc import linalg_utils, sr, wavefunctions

print = partial(print, flush=True)


@dataclass
class propagator:
    dt: float = 0.01
    n_prop_steps: int = 50
    n_ene_blocks: int = 50
    n_sr_blocks: int = 1
    n_blocks: int = 50
    ad_q: bool = True
    n_walkers: int = 50
    n_exp_terms: int = 6

    def init_prop_data(self, trial, wave_data, ham, ham_data, init_walkers=None):
        prop_data = {}
        prop_data["weights"] = jnp.ones(self.n_walkers)
        if init_walkers is not None:
            prop_data["walkers"] = init_walkers
        else:
            prop_data["walkers"] = jnp.stack(
                [jnp.eye(ham.norb, ham.nelec) + 0.0j for _ in range(self.n_walkers)]
            )
        energy_samples = jnp.real(
            trial.calc_energy_vmap(prop_data["walkers"], ham_data, wave_data)
        )
        e_estimate = jnp.array(jnp.sum(energy_samples) / self.n_walkers)
        prop_data["e_estimate"] = e_estimate
        prop_data["pop_control_ene_shift"] = e_estimate
        prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
        return prop_data

    @partial(jit, static_argnums=(0,))
    def stochastic_reconfiguration_local(self, prop_data):
        prop_data["key"], subkey = random.split(prop_data["key"])
        zeta = random.uniform(subkey)
        prop_data["walkers"], prop_data["weights"] = sr.stochastic_reconfiguration(
            prop_data["walkers"], prop_data["weights"], zeta
        )
        return prop_data

    def stochastic_reconfiguration_global(self, prop_data, comm):
        prop_data["key"], subkey = random.split(prop_data["key"])
        zeta = random.uniform(subkey)
        prop_data["walkers"], prop_data["weights"] = sr.stochastic_reconfiguration_mpi(
            prop_data["walkers"], prop_data["weights"], zeta, comm
        )
        return prop_data

    def orthonormalize_walkers(self, prop_data):
        prop_data["walkers"], _ = linalg_utils.qr_vmap(prop_data["walkers"])
        return prop_data

    # defining this separately because calculating vhs for a batch seems to be faster
    @partial(jit, static_argnums=(0,))
    def apply_propagator(self, exp_h1, vhs_i, walker_i):
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

    @partial(jit, static_argnums=(0,))
    def apply_propagator_vmap(self, ham, walkers, fields):
        vhs = (
            1.0j
            * jnp.sqrt(self.dt)
            * fields.dot(ham["chol"]).reshape(
                walkers.shape[0], walkers.shape[1], walkers.shape[1]
            )
        )
        return vmap(self.apply_propagator, in_axes=(None, 0, 0))(
            ham["exp_h1"], vhs, walkers
        )

    # defining this separately because of possible divergences in derivatives
    # @custom_jvp
    def calc_imp_fun(self, exponent_1, exponent_2, overlaps_new, overlaps_old):
        imp_fun = jnp.exp(exponent_1) * overlaps_new / overlaps_old
        theta = jnp.angle(jnp.exp(exponent_2) * overlaps_new / overlaps_old)
        # imp_fun_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)
        return imp_fun, theta
        # return imp_fun
        # return imp_fun_phaseless

    # @calc_imp_fun.defjvp
    # def _imp_fun_jvp(primals, tangents):
    #  primals_out, tangents_out = jvp(calc_imp_fun0, primals, tangents)
    #  return primals_out, (jnp.clip(tangents_out[0], -1000., 1000.), jnp.clip(tangents_out[1], -1000., 1000.))
    #  #return primals_out, (0. * primals_out[0], 0. * primals_out[1])

    @partial(jit, static_argnums=(0, 1))
    def propagate(self, trial, ham, prop, fields, wave_data):
        force_bias = trial.calc_force_bias_vmap(prop["walkers"], ham, wave_data)
        field_shifts = -jnp.sqrt(self.dt) * (1.0j * force_bias - ham["mf_shifts"])
        shifted_fields = fields - field_shifts
        shift_term = jnp.sum(shifted_fields * ham["mf_shifts"], axis=1)
        fb_term = jnp.sum(
            fields * field_shifts - field_shifts * field_shifts / 2.0, axis=1
        )

        prop["walkers"] = self.apply_propagator_vmap(
            ham, prop["walkers"], shifted_fields
        )

        overlaps_new = trial.calc_overlap_vmap(prop["walkers"], wave_data)
        imp_fun, theta = self.calc_imp_fun(
            -jnp.sqrt(self.dt) * shift_term
            + fb_term
            + self.dt * (prop["pop_control_ene_shift"] + ham["h0_prop"]),
            -jnp.sqrt(self.dt) * shift_term,
            overlaps_new,
            prop["overlaps"],
        )
        imp_fun_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)
        imp_fun_phaseless = jnp.where(
            jnp.isnan(imp_fun_phaseless), 0.0, imp_fun_phaseless
        )
        imp_fun_phaseless = jnp.where(imp_fun_phaseless < 1.0e-3, 0.0, imp_fun_phaseless)  # type: ignore
        imp_fun_phaseless = jnp.where(imp_fun_phaseless > 100.0, 0.0, imp_fun_phaseless)
        prop["weights"] = imp_fun_phaseless * prop["weights"]
        prop["weights"] = jnp.where(prop["weights"] > 100.0, 0.0, prop["weights"])
        prop["pop_control_ene_shift"] = prop["e_estimate"] - 0.1 * jnp.array(jnp.log(jnp.sum(prop["weights"]) / self.n_walkers) / self.dt)  # type: ignore
        prop["overlaps"] = overlaps_new
        return prop

    def __hash__(self):
        return hash(
            (
                self.dt,
                self.n_prop_steps,
                self.n_ene_blocks,
                self.n_sr_blocks,
                self.n_blocks,
                self.n_walkers,
            )
        )


@dataclass
class propagator_uhf(propagator):
    def init_prop_data(self, trial, wave_data, ham, ham_data, init_walkers=None):
        prop_data = {}
        prop_data["weights"] = jnp.ones(self.n_walkers)
        if init_walkers is not None:
            prop_data["walkers"] = init_walkers
        else:
            if isinstance(trial, wavefunctions.noci):
                walkers_up = jnp.stack(
                    [
                        wave_data[1][0][0][:, : ham.nelec[0]] + 1.0e-10j
                        for _ in range(self.n_walkers)
                    ]
                )
                walkers_dn = jnp.stack(
                    [
                        wave_data[1][1][0][:, : ham.nelec[1]] + 1.0e-10j
                        for _ in range(self.n_walkers)
                    ]
                )
            elif isinstance(trial, wavefunctions.ghf):
                rdm = wave_data @ wave_data.conj().T
                _, nat_up = jnp.linalg.eigh(rdm[: ham.norb, : ham.norb])
                _, nat_dn = jnp.linalg.eigh(rdm[ham.norb :, ham.norb :])
                walkers_up = jnp.stack(
                    [nat_up[:, -ham.nelec[0] :] + 0.0j for _ in range(self.n_walkers)]
                )
                walkers_dn = jnp.stack(
                    [nat_dn[:, -ham.nelec[1] :] + 0.0j for _ in range(self.n_walkers)]
                )
            else:
                walkers_up = jnp.stack(
                    [
                        wave_data[0][:, : ham.nelec[0]] + 0.0j
                        for _ in range(self.n_walkers)
                    ]
                )
                walkers_dn = jnp.stack(
                    [
                        wave_data[1][:, : ham.nelec[1]] + 0.0j
                        for _ in range(self.n_walkers)
                    ]
                )
            prop_data["walkers"] = [walkers_up, walkers_dn]
        energy_samples = jnp.real(
            trial.calc_energy_vmap(prop_data["walkers"], ham_data, wave_data)
        )
        e_estimate = jnp.array(jnp.sum(energy_samples) / self.n_walkers)
        prop_data["e_estimate"] = e_estimate
        prop_data["pop_control_ene_shift"] = e_estimate
        prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
        prop_data["normed_overlaps"] = prop_data["overlaps"]
        prop_data["norms"] = jnp.ones(self.n_walkers) + 0.0j
        return prop_data

    @partial(jit, static_argnums=(0,))
    def apply_propagator_vmap(self, ham, walkers, fields):
        vhs = (
            1.0j
            * jnp.sqrt(self.dt)
            * fields.dot(ham["chol"]).reshape(
                walkers[0].shape[0], walkers[0].shape[1], walkers[0].shape[1]
            )
        )
        walkers[0] = vmap(self.apply_propagator, in_axes=(None, 0, 0))(
            ham["exp_h1"][0], vhs, walkers[0]
        )
        walkers[1] = vmap(self.apply_propagator, in_axes=(None, 0, 0))(
            ham["exp_h1"][1], vhs, walkers[1]
        )
        return walkers

    @partial(jit, static_argnums=(0,))
    def stochastic_reconfiguration_local(self, prop_data):
        prop_data["key"], subkey = random.split(prop_data["key"])
        zeta = random.uniform(subkey)
        prop_data["walkers"], prop_data["weights"] = sr.stochastic_reconfiguration_uhf(
            prop_data["walkers"], prop_data["weights"], zeta
        )
        return prop_data

    def stochastic_reconfiguration_global(self, prop_data, comm):
        prop_data["key"], subkey = random.split(prop_data["key"])
        zeta = random.uniform(subkey)
        (
            prop_data["walkers"],
            prop_data["weights"],
        ) = sr.stochastic_reconfiguration_mpi_uhf(
            prop_data["walkers"], prop_data["weights"], zeta, comm
        )
        return prop_data

    def orthonormalize_walkers(self, prop_data):
        prop_data["walkers"], _ = linalg_utils.qr_vmap_uhf(prop_data["walkers"])
        return prop_data

    def orthogonalize_walkers(self, prop_data):
        prop_data["walkers"], norms = linalg_utils.qr_vmap_uhf(prop_data["walkers"])
        return prop_data, norms

    @partial(jit, static_argnums=(0))
    def multiply_constant(self, walkers, constants):
        walkers[0] = constants[0].reshape(-1, 1, 1) * walkers[0]
        walkers[1] = constants[1].reshape(-1, 1, 1) * walkers[1]
        return walkers

    @partial(jit, static_argnums=(0, 1))
    def propagate_free(self, trial, ham, prop, fields, wave_data):
        # jax.debug.print('ham:\n{}', ham)
        # jax.debug.print('prop:\n{}', prop)
        # jax.debug.print('fields:\n{}', fields)
        shift_term = jnp.einsum("wg,sg->sw", fields, ham["mf_shifts_fp"])
        # jax.debug.print('shift_term:\n{}', shift_term)
        constants = jnp.einsum(
            "sw,s->sw",
            jnp.exp(-jnp.sqrt(self.dt) * shift_term),
            jnp.exp(self.dt * ham["h0_prop_fp"]),
        )
        # jax.debug.print('constants:\n{}', constants)
        prop["walkers"] = self.apply_propagator_vmap(ham, prop["walkers"], fields)
        # jax.debug.print('walkers:\n{}', prop['walkers'])
        prop["walkers"] = self.multiply_constant(prop["walkers"], constants)
        # jax.debug.print('walkers after multi:\n{}', prop['walkers'])
        prop, norms = self.orthogonalize_walkers(prop)
        prop["norms"] *= norms[0] * norms[1]
        prop["overlaps"] = (
            trial.calc_overlap_vmap(prop["walkers"], wave_data) * prop["norms"]
        )
        normed_walkers, _ = linalg_utils.qr_vmap_uhf(prop["walkers"])
        prop["normed_overlaps"] = trial.calc_overlap_vmap(normed_walkers, wave_data)
        return prop

    def __hash__(self):
        return hash(
            (
                self.dt,
                self.n_prop_steps,
                self.n_ene_blocks,
                self.n_sr_blocks,
                self.n_blocks,
                self.n_walkers,
            )
        )


class propagator_cpmc(propagator_uhf):
    """CPMC propagator for the Hubbard model with on-site interactions."""

    def init_prop_data(
        self,
        trial: wavefunctions.wave_function_cpmc,
        wave_data: dict,
        ham: Any,
        ham_data: dict,
        init_walkers: Optional[Sequence] = None,
    ) -> dict:
        prop_data = super().init_prop_data(
            trial, wave_data, ham, ham_data, init_walkers
        )
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
        overlaps_new = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
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


class propagator_cpmc_slow(propagator_cpmc, propagator_uhf):
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
        overlaps_new = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
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
            overlaps_new_0 = trial.calc_overlap_vmap(
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
            overlaps_new_1 = trial.calc_overlap_vmap(
                [new_walkers_1_up, new_walkers_1_dn], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1)

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
        overlaps_new = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
        prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        prop_data["weights"] = jnp.where(
            prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"]
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
        overlaps_new = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
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
            overlaps_new_0 = trial.calc_overlap_vmap(
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
            overlaps_new_1 = trial.calc_overlap_vmap(
                [new_walkers_1_up, new_walkers_1_dn], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real
            ratio_1 = jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1)

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
        overlaps_new = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
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
class propagator_cpmc_nn(propagator_cpmc, propagator_uhf):
    neighbors: tuple = None

    def init_prop_data(
        self,
        trial: wavefunctions.wave_function_cpmc,
        wave_data: dict,
        ham: Any,
        ham_data: dict,
        init_walkers: Optional[Sequence] = None,
    ) -> dict:
        prop_data = super().init_prop_data(
            trial, wave_data, ham, ham_data, init_walkers
        )
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
        return hash(
            (
                self.dt,
                self.n_prop_steps,
                self.n_ene_blocks,
                self.n_sr_blocks,
                self.n_blocks,
                self.n_walkers,
                self.neighbors,
            )
        )


@dataclass
class propagator_cpmc_nn_slow(propagator_uhf):
    neighbors: tuple = None

    def init_prop_data(
        self,
        trial: wavefunctions.wave_function_cpmc,
        wave_data: dict,
        ham: Any,
        ham_data: dict,
        init_walkers: Optional[Sequence] = None,
    ) -> dict:
        prop_data = self.__super__.init_prop_data(
            trial, wave_data, ham, ham_data, init_walkers
        )
        prop_data["greens"] = trial.calc_green_vmap(walkers, wave_data)
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
        overlaps_new = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
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
            overlaps_new_0 = trial.calc_overlap_vmap(
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
            overlaps_new_1 = trial.calc_overlap_vmap(
                [new_walkers_1_up, new_walkers_1_dn], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1)

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
            overlaps_new_0 = trial.calc_overlap_vmap(
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
            overlaps_new_1 = trial.calc_overlap_vmap(
                [new_walkers_1_up, carry["walkers"][1]], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1)

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
            overlaps_new_0 = trial.calc_overlap_vmap(
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
            overlaps_new_1 = trial.calc_overlap_vmap(
                [new_walkers_1_up, new_walkers_1_dn], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1)

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
            overlaps_new_0 = trial.calc_overlap_vmap(
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
            overlaps_new_1 = trial.calc_overlap_vmap(
                [new_walkers_1_up, new_walkers_1_dn], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1)

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
            overlaps_new_0 = trial.calc_overlap_vmap(
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
            overlaps_new_1 = trial.calc_overlap_vmap(
                [carry["walkers"][0], new_walkers_1_dn], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1)

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
        overlaps_new = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
        prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        prop_data["weights"] = jnp.where(
            prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"]
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
        return hash(
            (
                self.dt,
                self.n_prop_steps,
                self.n_ene_blocks,
                self.n_sr_blocks,
                self.n_blocks,
                self.n_walkers,
                self.neighbors,
            )
        )


class propagator_cpmc_continuous(propagator_uhf):

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
            optimize=True,
        )
        prop_data["walkers"][1] = jnp.einsum(
            "wi,wij->wij",
            jnp.exp(-shifted_fields * ham_data["hs_constant"]),
            prop_data["walkers"][1],
            optimize=True,
        )

        # one body
        prop_data["walkers"][0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"][0]
        )
        prop_data["walkers"][1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"][1]
        )
        overlaps_new = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)

        imp_fun = (
            jnp.exp(self.dt * (prop_data["pop_control_ene_shift"]) + fb_term)
            * overlaps_new
            / prop_data["overlaps"]
        )
        prop_data["weights"] *= imp_fun.real
        prop_data["weights"] = jnp.where(
            prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"]
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


if __name__ == "__main__":
    prop = propagator()
    nelec = 3
    norb = 6
    nchol = 6
    nwalkers = 5
    h0 = 0.0
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    h1 = random.normal(subkey, (norb, norb))
    key, subkey = random.split(key)
    walkers = random.normal(subkey, (nwalkers, norb, nelec)) + 0.0j
    h0_prop = 0.0
    exp_h1 = jsp.linalg.expm(-prop.dt * h1 / 2.0)
    key, subkey = random.split(key)
    chol = random.normal(subkey, (nchol, norb, norb))
    chol = chol.reshape(nchol, norb * norb)
    key, subkey = random.split(key)
    fields = random.normal(subkey, shape=(nwalkers, nchol))
    new_walkers = prop.apply_propagator_vmap(exp_h1, chol, walkers, fields)
