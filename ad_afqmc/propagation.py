import math
import os

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
from dataclasses import dataclass
from functools import partial

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

    def __post_init__(self):
        if not self.ad_q:
            self.n_ene_blocks = 5
            self.n_sr_blocks = 10

    def init_prop_data(self, trial, wave_data, ham, ham_data):
        prop_data = {}
        prop_data["weights"] = jnp.ones(self.n_walkers)
        prop_data["walkers"] = jnp.stack(
            [jnp.eye(ham.norb, ham.nelec) + 0.0j for _ in range(self.n_walkers)]
        )
        energy_samples = jnp.real(
            trial.calc_energy_vmap(ham_data, prop_data["walkers"], wave_data)
        )
        e_estimate = jnp.array(jnp.sum(energy_samples) / self.n_walkers)
        prop_data["e_estimate"] = e_estimate
        prop_data["pop_control_ene_shift"] = e_estimate
        prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
        prop_data["inv_overlap_mats"] = trial.calc_inv_overlap_mat_vmap(prop_data["walkers"], wave_data)
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
    def init_prop_data(self, trial, wave_data, ham, ham_data):
        prop_data = {}
        prop_data["weights"] = jnp.ones(self.n_walkers)
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
                [wave_data[0][:, : ham.nelec[0]] + 0.0j for _ in range(self.n_walkers)]
            )
            walkers_dn = jnp.stack(
                [wave_data[1][:, : ham.nelec[1]] + 0.0j for _ in range(self.n_walkers)]
            )
        prop_data["walkers"] = [walkers_up, walkers_dn]
        energy_samples = jnp.real(
            trial.calc_energy_vmap(ham_data, prop_data["walkers"], wave_data)
        )
        e_estimate = jnp.array(jnp.sum(energy_samples) / self.n_walkers)
        prop_data["e_estimate"] = e_estimate
        prop_data["pop_control_ene_shift"] = e_estimate
        prop_data["overlaps"] = trial.calc_overlap_vmap(prop_data["walkers"], wave_data)
        prop_data["inv_overlap_mats"] = trial.calc_inv_overlap_mat_vmap(prop_data["walkers"], wave_data)
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
    @partial(jit, static_argnums=(0, 1))
    def propagate(
        self,
        trial: wavefunctions.wave_function,
        ham_data: dict,
        prop_data: dict,
        gaussian_rns: jnp.array,
        wave_data: jnp.array,
    ) -> dict:
        """
        Propagate the walkers using the CPMC algorithm.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            gaussian_rns: Gaussian random numbers (these are converted to uniform)
            wave_data: array containing the wave function coefficient matrix

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        if isinstance(trial, wavefunctions.ghf):
            wave_data = [wave_data[: self.norb], wave_data[self.norb :]]

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
        prop_data["inv_overlap_mats"] = trial.calc_inv_overlap_mat_vmap(
                                            prop_data["walkers"], wave_data)

        # two body
        # TODO: define separate sampler that feeds uniform_rns instead of gaussian_rns
        uniform_rns = (jsp.special.erf(gaussian_rns / 2**0.5) + 1) / 2

        # iterate over sites
        # TODO: fast update
        def scanned_fun(carry, x):
            # Constant for updating overlap matrix.
            delta = ham_data["hs_constant"] - 1

            # field 1
            new_walkers_0_up = (
                carry["walkers"][0].at[:, x, :].mul(ham_data["hs_constant"][0, 0])
            )
            new_walkers_0_dn = (
                carry["walkers"][1].at[:, x, :].mul(ham_data["hs_constant"][0, 1])
            )

            new_inv_overlap_mats_0 = trial.update_inv_overlap_mat_vmap( 
                carry["walkers"], wave_data, carry["inv_overlap_mats"], delta[0], x
            )

            new_overlaps_0 = trial.update_overlap_vmap(
                carry["walkers"], wave_data, carry["inv_overlap_mats"], carry["overlaps"], delta[0], x
            )

            ratio_0 = (new_overlaps_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            new_walkers_1_up = (
                carry["walkers"][0].at[:, x].mul(ham_data["hs_constant"][1, 0])
            )
            new_walkers_1_dn = (
                carry["walkers"][1].at[:, x].mul(ham_data["hs_constant"][1, 1])
            )
            
            new_inv_overlap_mats_1 = trial.update_inv_overlap_mat_vmap( 
                carry["walkers"], wave_data, carry["inv_overlap_mats"], delta[1], x
            )

            new_overlaps_1 = trial.update_overlap_vmap(
                carry["walkers"], wave_data, carry["inv_overlap_mats"], carry["overlaps"], delta[1], x
            )
            
            ratio_1 = (new_overlaps_1 / carry["overlaps"]).real / 2.0
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

            new_inv_overlap_mats_up = jnp.where(
                    mask_0, new_inv_overlap_mats_0[0], new_inv_overlap_mats_1[0])
            new_inv_overlap_mats_dn = jnp.where(
                    mask_0, new_inv_overlap_mats_0[1], new_inv_overlap_mats_1[1])
            new_inv_overlap_mats = [new_inv_overlap_mats_up, new_inv_overlap_mats_dn]

            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, new_overlaps_0, new_overlaps_1)
            
            carry["walkers"] = new_walkers
            carry["weights"] *= norm # NOTE why this update?
            carry["overlaps"] = overlaps_new
            carry["inv_overlap_mats"] = new_inv_overlap_mats
            return carry, x
        
        # Scan over lattice sites.
        prop_data, _ = lax.scan(
            scanned_fun, prop_data, jnp.arange(ham_data["chol"].shape[0])
        )  # TODO: chol will be removed from ham_data for hubbard

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
    def propagate_fast(
        self,
        trial: wavefunctions.wave_function,
        ham_data: dict,
        prop_data: dict,
        gaussian_rns: jnp.array,
        wave_data: jnp.array,
    ) -> dict:
        """
        Propagate the walkers using the CPMC algorithm.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            gaussian_rns: Gaussian random numbers (these are converted to uniform)
            wave_data: array containing the wave function coefficient matrix

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        if isinstance(trial, wavefunctions.ghf):
            wave_data = [wave_data[: self.norb], wave_data[self.norb :]]

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
        prop_data["inv_overlap_mats"] = trial.calc_inv_overlap_mat_vmap(
                                            prop_data["walkers"], wave_data)

        # two body
        # TODO: define separate sampler that feeds uniform_rns instead of gaussian_rns
        uniform_rns = (jsp.special.erf(gaussian_rns / 2**0.5) + 1) / 2

        # iterate over sites
        # TODO: fast update
        def scanned_fun(carry, x):
            # Constant for updating overlap matrix.
            delta = ham_data["hs_constant"] - 1

            # field 1
            new_walkers_0_up = (
                carry["walkers"][0].at[:, x, :].mul(ham_data["hs_constant"][0, 0])
            )
            new_walkers_0_dn = (
                carry["walkers"][1].at[:, x, :].mul(ham_data["hs_constant"][0, 1])
            )

            new_inv_overlap_mats_0_up = trial.update_inv_overlap_mat_vmap( 
                carry["walkers"][0][:, x], wave_data[0][x, : trial.nelec[0]], 
                carry["inv_overlap_mats"][0], delta[0, 0]
            )
            new_inv_overlap_mats_0_dn = trial.update_inv_overlap_mat_vmap( 
                carry["walkers"][1][:, x], wave_data[1][x, : trial.nelec[1]], 
                carry["inv_overlap_mats"][1], delta[0, 1]
            )

            # Update spin up first.
            new_overlaps_0 = trial.update_overlap_vmap(
                carry["walkers"][0][:, x], wave_data[0][x, : trial.nelec[0]], 
                carry["inv_overlap_mats"][0], carry["overlaps"], delta[0, 0]
            )

            # Update spin down.
            new_overlaps_0 = trial.update_overlap_vmap(
                carry["walkers"][1][:, x], wave_data[1][x, : trial.nelec[1]], 
                carry["inv_overlap_mats"][1], new_overlaps_0, delta[0, 1]
            )

            ratio_0 = (new_overlaps_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            new_walkers_1_up = (
                carry["walkers"][0].at[:, x].mul(ham_data["hs_constant"][1, 0])
            )
            new_walkers_1_dn = (
                carry["walkers"][1].at[:, x].mul(ham_data["hs_constant"][1, 1])
            )
            
            new_inv_overlap_mats_1_up = trial.update_inv_overlap_mat_vmap( 
                carry["walkers"][0][:, x], wave_data[0][x, : trial.nelec[0]], 
                carry["inv_overlap_mats"][0], delta[1, 0]
            )
            new_inv_overlap_mats_1_dn = trial.update_inv_overlap_mat_vmap( 
                carry["walkers"][1][:, x], wave_data[1][x, : trial.nelec[1]], 
                carry["inv_overlap_mats"][1], delta[1, 1]
            )
            
            # Update spin up first.
            new_overlaps_1 = trial.update_overlap_vmap(
                carry["walkers"][0][:, x], wave_data[0][x, : trial.nelec[0]], 
                carry["inv_overlap_mats"][0], carry["overlaps"], delta[1, 0]
            )

            # Update spin down.
            new_overlaps_1 = trial.update_overlap_vmap(
                carry["walkers"][1][:, x], wave_data[1][x, : trial.nelec[1]], 
                carry["inv_overlap_mats"][1], new_overlaps_1, delta[1, 1]
            )

            ratio_1 = (new_overlaps_1 / carry["overlaps"]).real / 2.0
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

            new_inv_overlap_mats_up = jnp.where(
                    mask_0, new_inv_overlap_mats_0_up, new_inv_overlap_mats_1_up)
            new_inv_overlap_mats_dn = jnp.where(
                    mask_0, new_inv_overlap_mats_0_dn, new_inv_overlap_mats_1_dn)
            new_inv_overlap_mats = [new_inv_overlap_mats_up, new_inv_overlap_mats_dn]

            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, new_overlaps_0, new_overlaps_1)
            
            carry["walkers"] = new_walkers
            carry["weights"] *= norm # NOTE why this update?
            carry["overlaps"] = overlaps_new
            carry["inv_overlap_mats"] = new_inv_overlap_mats
            return carry, x
        
        # Scan over lattice sites.
        prop_data, _ = lax.scan(
            scanned_fun, prop_data, jnp.arange(ham_data["chol"].shape[0])
        )  # TODO: chol will be removed from ham_data for hubbard

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
    def propagate_slow(
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
                carry["walkers"][0].at[:, x, :].mul(ham_data["hs_constant"][0, 0])
            )
            new_walkers_0_dn = (
                carry["walkers"][1].at[:, x, :].mul(ham_data["hs_constant"][0, 1])
            )
            overlaps_new_0 = trial.calc_overlap_vmap(
                [new_walkers_0_up, new_walkers_0_dn], wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            new_walkers_1_up = (
                carry["walkers"][0].at[:, x, :].mul(ham_data["hs_constant"][1, 0])
            )
            new_walkers_1_dn = (
                carry["walkers"][1].at[:, x, :].mul(ham_data["hs_constant"][1, 1])
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

        prop_data, _ = lax.scan(
            scanned_fun, prop_data, jnp.arange(ham_data["chol"].shape[0])
        )  # TODO: chol will be removed from ham_data for hubbard

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
                carry["walkers"][0].at[:, x, :].mul(ham_data["hs_constant"][0, 0])
            )
            new_walkers_0_dn = (
                carry["walkers"][1].at[:, x, :].mul(ham_data["hs_constant"][0, 1])
            )
            overlaps_new_0 = trial.calc_overlap_vmap(
                [new_walkers_0_up, new_walkers_0_dn], wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)

            # field 2
            new_walkers_1_up = (
                carry["walkers"][0].at[:, x, :].mul(ham_data["hs_constant"][1, 0])
            )
            new_walkers_1_dn = (
                carry["walkers"][1].at[:, x, :].mul(ham_data["hs_constant"][1, 1])
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
