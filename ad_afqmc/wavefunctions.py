import os

import numpy as np

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence, Tuple

# os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
from jax import jit, lax, vmap

from ad_afqmc import linalg_utils

print = partial(print, flush=True)


class wave_function(ABC):
    """Abstract class for wave functions."""

    @abstractmethod
    def calc_overlap_vmap(self, walkers: Sequence, wave_data: Any) -> jnp.array:
        """Calculate the overlap between the walkers and the wave function.

        Args:
            walkers :
                The walkers.
            wave_data : Any
                The trial wave function data.

        Returns:
            jnp.array: The overlaps.
        """
        pass

    @abstractmethod
    def calc_green_vmap(self, walkers: Sequence, wave_data: Any) -> jnp.array:
        """Calculate the greens function.

        Args:
            walkers :
                The walkers.
            wave_data : Any
                The trial wave function data.

        Returns:
            jnp.array: The greens function (< psi_T | a_i^dagger a_j | walker > / < psi_T | walker >).
            In case of some trials this returns only a part of the greens function.
            The other parts are stored in rotated hamiltonian integrals to avoid recomputation.
        """
        pass

    @abstractmethod
    def calc_force_bias_vmap(
        self, walkers: Sequence, ham_data: dict, wave_data: Any
    ) -> jnp.array:
        """Calculate the force bias.

        Args:
            walkers :
                The walkers.
            ham : Any
                The hamiltonian data.
            wave_data : Any
                The trial wave function data.

        Returns:
            jnp.array: The force biases.
        """
        pass

    @abstractmethod
    def calc_energy_vmap(
        self, ham_data: dict, walkers: Sequence, wave_data: Any
    ) -> jnp.array:
        """Calculate the energy.

        Args:
            ham : Any
                The hamiltonian data.
            walkers :
                The walkers.
            wave_data : Any
                The trial wave function data.

        Returns:
            jnp.array: The walker energies.
        """
        pass


# we assume afqmc is performed in the rhf orbital basis
@dataclass
class rhf(wave_function):
    norb: int
    nelec: (
        int  # this is the number of electrons of each spin, so nelec = total_nelec // 2
    )
    n_opt_iter: int = 30

    @partial(jit, static_argnums=0)
    def calc_overlap(self, walker, wave_data=None):
        return jnp.linalg.det(walker[: walker.shape[1], :]) ** 2

    @partial(jit, static_argnums=0)
    def calc_overlap_vmap(self, walkers, wave_data=None):
        return vmap(self.calc_overlap, in_axes=(0, None))(walkers, wave_data)

    def calc_inv_overlap_mat_vmap(self, walkers, wave_data):
        pass
    
    @partial(jit, static_argnums=0)
    def calc_green(self, walker, wave_data=None):
        return (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T

    @partial(jit, static_argnums=0)
    def calc_green_vmap(self, walkers, wave_data=None):
        return vmap(self.calc_green, in_axes=(0, None))(walkers, wave_data)

    @partial(jit, static_argnums=0)
    def calc_1rdm(self, walker, wave_data=None):  # shouldnt be here
        rdm1 = (
            walker.dot(jnp.linalg.inv(walker.T.conj().dot(walker))).dot(walker.T.conj())
        ).T
        return rdm1

    @partial(jit, static_argnums=0)
    def calc_1rdm_vmap(self, walkers, wave_data=None):
        return vmap(self.calc_1rdm, in_axes=(0, None))(walkers, wave_data)

    @partial(jit, static_argnums=0)
    def calc_force_bias(self, walker, rot_chol, wave_data=None):
        green_walker = self.calc_green(walker, wave_data)
        fb = 2.0 * jnp.einsum("gij,ij->g", rot_chol, green_walker, optimize="optimal")
        return fb

    @partial(jit, static_argnums=0)
    def calc_force_bias_vmap(self, walkers, ham_data, wave_data=None):
        return vmap(self.calc_force_bias, in_axes=(0, None, None))(
            walkers, ham_data["rot_chol"], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_energy(self, h0, rot_h1, rot_chol, walker, wave_data=None):
        ene0 = h0
        green_walker = self.calc_green(walker, wave_data)
        ene1 = 2.0 * jnp.sum(green_walker * rot_h1)
        f = jnp.einsum("gij,jk->gik", rot_chol, green_walker.T, optimize="optimal")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = 2.0 * jnp.sum(c * c) - exc
        return ene2 + ene1 + ene0

    @partial(jit, static_argnums=0)
    def calc_energy_vmap(self, ham_data, walkers, wave_data=None):
        return vmap(self.calc_energy, in_axes=(None, None, None, 0, None))(
            ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"], walkers, wave_data
        )

    def get_rdm1(self, wave_data):
        rdm1 = 2 * np.eye(self.norb, self.nelec).dot(np.eye(self.norb, self.nelec).T)
        return rdm1

    @partial(jit, static_argnums=0)
    def optimize_orbs(self, ham_data, wave_data=None):
        h1 = ham_data["h1"]
        h2 = ham_data["chol"]
        h2 = h2.reshape((h2.shape[0], h1.shape[0], h1.shape[0]))
        nelec = self.nelec
        h1 = (h1 + h1.T) / 2.0

        def scanned_fun(carry, x):
            dm = carry
            f = jnp.einsum("gij,ik->gjk", h2, dm)
            c = vmap(jnp.trace)(f)
            vj = jnp.einsum("g,gij->ij", c, h2)
            vk = jnp.einsum("glj,gjk->lk", f, h2)
            vhf = vj - 0.5 * vk
            fock = h1 + vhf
            mo_energy, mo_coeff = linalg_utils._eigh(fock)
            idx = jnp.argmax(abs(mo_coeff.real), axis=0)
            mo_coeff = jnp.where(
                mo_coeff[idx, jnp.arange(len(mo_energy))].real < 0, -mo_coeff, mo_coeff
            )
            e_idx = jnp.argsort(mo_energy)
            nmo = mo_energy.size
            mo_occ = jnp.zeros(nmo)
            nocc = nelec
            mo_occ = mo_occ.at[e_idx[:nocc]].set(2)
            mocc = mo_coeff[:, jnp.nonzero(mo_occ, size=nocc)[0]]
            dm = (mocc * mo_occ[jnp.nonzero(mo_occ, size=nocc)[0]]).dot(mocc.T)
            return dm, mo_coeff

        norb = h1.shape[0]
        dm0 = 2 * jnp.eye(norb, nelec).dot(jnp.eye(norb, nelec).T)
        _, mo_coeff = lax.scan(scanned_fun, dm0, None, length=self.n_opt_iter)

        return mo_coeff[-1]

    def __hash__(self):
        return hash(
            (
                self.norb,
                self.nelec,
                self.n_opt_iter,
            )
        )


@dataclass
class uhf(wave_function):
    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30

    @partial(jit, static_argnums=0)
    def calc_overlap(self, walker_up, walker_dn, wave_data):
        return jnp.linalg.det(
            wave_data[0][:, : self.nelec[0]].T.conj() @ walker_up
        ) * jnp.linalg.det(wave_data[1][:, : self.nelec[1]].T.conj() @ walker_dn)

    def calc_overlap_vmap(self, walkers, wave_data):
        return vmap(self.calc_overlap, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_inv_overlap_mat(self, walker_up, walker_dn, wave_data):
        return [jnp.linalg.inv(wave_data[0][:, : self.nelec[0]].T.conj() @ walker_up),
                jnp.linalg.inv(wave_data[1][:, : self.nelec[1]].T.conj() @ walker_dn)]
    
    def calc_inv_overlap_mat_vmap(self, walkers, wave_data):
        return vmap(self.calc_inv_overlap_mat, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )
    
    @partial(jit, static_argnums=0)
    def update_inv_overlap_mat(self, walker_up, walker_dn, wave_data, inv_overlap_mat, delta, i):
        vup = walker_up[i] * delta[0]
        vdn = walker_dn[i] * delta[1]

        # Sherman-Morrison formula.
        inv_overlap_mat[0] = linalg_utils.sherman_morrison(
                inv_overlap_mat[0], wave_data[0][i, : self.nelec[0]].conj(), vup)
        inv_overlap_mat[1] = linalg_utils.sherman_morrison(
                inv_overlap_mat[1], wave_data[1][i, : self.nelec[1]].conj(), vdn)
        return inv_overlap_mat
    
    def update_inv_overlap_mat_vmap(self, walkers, wave_data, inv_overlap_mats, delta, i):
        return vmap(self.update_inv_overlap_mat, in_axes=(0, 0, None, 0, None, None))(
            walkers[0], walkers[1], wave_data, inv_overlap_mats, delta, i
        )

    @partial(jit, static_argnums=0)
    def update_overlap(self, walker_up, walker_dn, wave_data, inv_overlap_mat, overlap, delta, i):
        vup = walker_up[i] * delta[0]
        vdn = walker_dn[i] * delta[1]

        # Matrix determinant lemma.
        # Update spin up first.
        overlap = linalg_utils.mat_det_lemma(
                overlap, inv_overlap_mat[0], wave_data[0][i, : self.nelec[0]].conj(), vup)

        # Update spin down.
        overlap = linalg_utils.mat_det_lemma(
                overlap, inv_overlap_mat[1], wave_data[1][i, : self.nelec[1]].conj(), vdn)
        return overlap

    def update_overlap_vmap(self, walkers, wave_data, inv_overlap_mats, overlaps, delta, i):
        return vmap(self.update_overlap, in_axes=(0, 0, None, 0, 0, None, None))(
            walkers[0], walkers[1], wave_data, inv_overlap_mats, overlaps, delta, i
        )

    @partial(jit, static_argnums=0)
    def calc_green(self, walker_up, walker_dn, wave_data):
        green_up = (
            walker_up.dot(
                jnp.linalg.inv(wave_data[0][:, : self.nelec[0]].T.dot(walker_up))
            )
        ).T
        green_dn = (
            walker_dn.dot(
                jnp.linalg.inv(wave_data[1][:, : self.nelec[1]].T.dot(walker_dn))
            )
        ).T
        return [green_up, green_dn]

    def calc_green_vmap(self, walkers, wave_data):
        return vmap(self.calc_green, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_force_bias(self, walker_up, walker_dn, rot_chol, wave_data):
        green_walker = self.calc_green(walker_up, walker_dn, wave_data)
        fb_up = jnp.einsum(
            "gij,ij->g", rot_chol[0], green_walker[0], optimize="optimal"
        )
        fb_dn = jnp.einsum(
            "gij,ij->g", rot_chol[1], green_walker[1], optimize="optimal"
        )
        return fb_up + fb_dn

    def calc_force_bias_vmap(self, walkers, ham_data, wave_data):
        return vmap(self.calc_force_bias, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data["rot_chol"], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_energy(self, h0, rot_h1, rot_chol, walker_up, walker_dn, wave_data):
        ene0 = h0
        green_walker = self.calc_green(walker_up, walker_dn, wave_data)
        ene1 = jnp.sum(green_walker[0] * rot_h1[0]) + jnp.sum(
            green_walker[1] * rot_h1[1]
        )
        f_up = jnp.einsum(
            "gij,jk->gik", rot_chol[0], green_walker[0].T, optimize="optimal"
        )
        f_dn = jnp.einsum(
            "gij,jk->gik", rot_chol[1], green_walker[1].T, optimize="optimal"
        )
        c_up = vmap(jnp.trace)(f_up)
        c_dn = vmap(jnp.trace)(f_dn)
        exc_up = jnp.sum(vmap(lambda x: x * x.T)(f_up))
        exc_dn = jnp.sum(vmap(lambda x: x * x.T)(f_dn))
        ene2 = (
            jnp.sum(c_up * c_up)
            + jnp.sum(c_dn * c_dn)
            + 2.0 * jnp.sum(c_up * c_dn)
            - exc_up
            - exc_dn
        ) / 2.0

        return ene2 + ene1 + ene0

    def calc_energy_vmap(self, ham_data, walkers, wave_data):
        return vmap(self.calc_energy, in_axes=(None, None, None, 0, 0, None))(
            ham_data["h0"],
            ham_data["rot_h1"],
            ham_data["rot_chol"],
            walkers[0],
            walkers[1],
            wave_data,
        )

    def get_rdm1(self, wave_data):
        dm_up = (wave_data[0][:, : self.nelec[0]]).dot(
            wave_data[0][:, : self.nelec[0]].T
        )
        dm_dn = (wave_data[1][:, : self.nelec[1]]).dot(
            wave_data[1][:, : self.nelec[1]].T
        )
        return jnp.array([dm_up, dm_dn])

    @partial(jit, static_argnums=0)
    def optimize_orbs(self, ham_data, wave_data):
        h1 = ham_data["h1"]
        h1 = h1.at[0].set((h1[0] + h1[0].T) / 2.0)
        h1 = h1.at[1].set((h1[1] + h1[1].T) / 2.0)
        h2 = ham_data["chol"]
        h2 = h2.reshape((h2.shape[0], h1.shape[1], h1.shape[1]))
        nelec = self.nelec

        def scanned_fun(carry, x):
            dm = carry
            f_up = jnp.einsum("gij,ik->gjk", h2, dm[0])
            c_up = vmap(jnp.trace)(f_up)
            vj_up = jnp.einsum("g,gij->ij", c_up, h2)
            vk_up = jnp.einsum("glj,gjk->lk", f_up, h2)
            f_dn = jnp.einsum("gij,ik->gjk", h2, dm[1])
            c_dn = vmap(jnp.trace)(f_dn)
            vj_dn = jnp.einsum("g,gij->ij", c_dn, h2)
            vk_dn = jnp.einsum("glj,gjk->lk", f_dn, h2)
            fock_up = h1[0] + vj_up + vj_dn - vk_up
            fock_dn = h1[1] + vj_up + vj_dn - vk_dn
            mo_energy_up, mo_coeff_up = linalg_utils._eigh(fock_up)
            mo_energy_dn, mo_coeff_dn = linalg_utils._eigh(fock_dn)

            nmo = mo_energy_up.size

            idx_up = jnp.argmax(abs(mo_coeff_up.real), axis=0)
            mo_coeff_up = jnp.where(
                mo_coeff_up[idx_up, jnp.arange(len(mo_energy_up))].real < 0,
                -mo_coeff_up,
                mo_coeff_up,
            )
            e_idx_up = jnp.argsort(mo_energy_up)
            mo_occ_up = jnp.zeros(nmo)
            nocc_up = nelec[0]
            mo_occ_up = mo_occ_up.at[e_idx_up[:nocc_up]].set(1)
            mocc_up = mo_coeff_up[:, jnp.nonzero(mo_occ_up, size=nocc_up)[0]]
            dm_up = (mocc_up * mo_occ_up[jnp.nonzero(mo_occ_up, size=nocc_up)[0]]).dot(
                mocc_up.T
            )

            idx_dn = jnp.argmax(abs(mo_coeff_dn.real), axis=0)
            mo_coeff_dn = jnp.where(
                mo_coeff_dn[idx_dn, jnp.arange(len(mo_energy_dn))].real < 0,
                -mo_coeff_dn,
                mo_coeff_dn,
            )
            e_idx_dn = jnp.argsort(mo_energy_dn)
            mo_occ_dn = jnp.zeros(nmo)
            nocc_dn = nelec[1]
            mo_occ_dn = mo_occ_dn.at[e_idx_dn[:nocc_dn]].set(1)
            mocc_dn = mo_coeff_dn[:, jnp.nonzero(mo_occ_dn, size=nocc_dn)[0]]
            dm_dn = (mocc_dn * mo_occ_dn[jnp.nonzero(mo_occ_dn, size=nocc_dn)[0]]).dot(
                mocc_dn.T
            )

            return jnp.array([dm_up, dm_dn]), jnp.array([mo_coeff_up, mo_coeff_dn])

        dm_up = (wave_data[0][:, : nelec[0]]).dot(wave_data[0][:, : nelec[0]].T)
        dm_dn = (wave_data[1][:, : nelec[1]]).dot(wave_data[1][:, : nelec[1]].T)
        dm0 = jnp.array([dm_up, dm_dn])
        _, mo_coeff = lax.scan(scanned_fun, dm0, None, length=self.n_opt_iter)

        return mo_coeff[-1]

    def __hash__(self):
        return hash(
            (
                self.norb,
                self.nelec,
                self.n_opt_iter,
            )
        )

@dataclass
class ghf(wave_function):
    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30

    @partial(jit, static_argnums=0)
    def calc_overlap(self, walker_up, walker_dn, wave_data):
        return jnp.linalg.det(
            jnp.hstack(
                [
                    wave_data[: self.norb].T.conj() @ walker_up,
                    wave_data[self.norb :].T.conj() @ walker_dn,
                ]
            )
        )
    
    def calc_overlap_vmap(self, walkers, wave_data):
        return vmap(self.calc_overlap, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_inv_overlap_mat(self, walker_up, walker_dn, wave_data):
        return jnp.linalg.inv(
            jnp.hstack(
                [
                    wave_data[: self.norb].T.conj() @ walker_up,
                    wave_data[self.norb : ].T.conj() @ walker_dn,
                ]
            )
        )
    
    def calc_inv_overlap_mat_vmap(self, walkers, wave_data):
        return vmap(self.calc_inv_overlap_mat, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )
    
    @partial(jit, static_argnums=0)
    def update_inv_overlap_mat(self, walker_up, walker_dn, wave_data, inv_overlap_mat, delta, i):
        # Reshape UHF walker wavefunction to GHF form.
        walker = jnp.block([
                    [walker_up, jnp.zeros((self.norb, self.nelec[1]))],
                    [jnp.zeros((self.norb, self.nelec[0])), walker_dn]
                ])
        v = walker[i] * delta

        # Sherman-Morrison formula.
        inv_overlap_mat = linalg_utils.sherman_morrison(inv_overlap_mat, wave_data[i].conj(), v)
        return inv_overlap_mat
    
    def update_inv_overlap_mat_vmap(self, walkers, wave_data, inv_overlap_mats, delta, i):
        return vmap(self.update_inv_overlap_mat, in_axes=(0, 0, None, 0, None, None))(
            walkers[0], walkers[1], wave_data, inv_overlap_mats, delta, i
        )

    @partial(jit, static_argnums=0)
    def update_overlap(self, walker_up, walker_dn, wave_data, inv_overlap_mat, overlap, delta, i):
        walker = jnp.block([
                    [walker_up, jnp.zeros((self.norb, self.nelec[1]))],
                    [jnp.zeros((self.norb, self.nelec[0])), walker_dn]
                ])
        v = walker[i] * delta

        # Matrix determinant lemma.
        overlap = linalg_utils.mat_det_lemma(overlap, inv_overlap_mat, wave_data[i].conj(), v)
        return overlap

    def update_overlap_vmap(self, walkers, wave_data, inv_overlap_mats, overlaps, delta, i):
        return vmap(self.update_overlap, in_axes=(0, 0, None, 0, 0, None, None))(
            walkers[0], walkers[1], wave_data, inv_overlap_mats, overlaps, delta, i
        )

    @partial(jit, static_argnums=0)
    def calc_green(self, walker_up, walker_dn, wave_data):
        overlap_mat = jnp.hstack(
            [
                wave_data[: self.norb].T @ walker_up,
                wave_data[self.norb :].T @ walker_dn,
            ]
        )
        inv = jnp.linalg.inv(overlap_mat)
        green = (
            jnp.vstack(
                [walker_up @ inv[: self.nelec[0]], walker_dn @ inv[self.nelec[0] :]]
            )
        ).T
        return green

    def calc_green_vmap(self, walkers, wave_data):
        return vmap(self.calc_green, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_force_bias(self, walker_up, walker_dn, rot_chol, wave_data):
        green_walker = self.calc_green(walker_up, walker_dn, wave_data)
        fb = jnp.einsum("gij,ij->g", rot_chol, green_walker, optimize="optimal")
        return fb

    def calc_force_bias_vmap(self, walkers, ham_data, wave_data):
        return vmap(self.calc_force_bias, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data["rot_chol"], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_energy(self, h0, rot_h1, rot_chol, walker_up, walker_dn, wave_data):
        ene0 = h0
        green_walker = self.calc_green(walker_up, walker_dn, wave_data)
        ene1 = jnp.sum(green_walker * rot_h1)
        f = jnp.einsum("gij,jk->gik", rot_chol, green_walker.T, optimize="optimal")
        coul = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = (jnp.sum(coul * coul) - exc) / 2.0
        return ene2 + ene1 + ene0

    def calc_energy_vmap(self, ham_data, walkers, wave_data):
        return vmap(self.calc_energy, in_axes=(None, None, None, 0, 0, None))(
            ham_data["h0"],
            ham_data["rot_h1"],
            ham_data["rot_chol"],
            walkers[0],
            walkers[1],
            wave_data,
        )

    def get_rdm1(self, wave_data):
        dm = (
            wave_data[:, : self.nelec[0] + self.nelec[1]]
            @ wave_data[:, : self.nelec[0] + self.nelec[1]].T
        )
        dm_up = dm[: self.norb, : self.norb]
        dm_dn = dm[self.norb :, self.norb :]
        return jnp.array([dm_up, dm_dn])

    # not implemented
    @partial(jit, static_argnums=0)
    def optimize_orbs(self, ham_data, wave_data):
        return wave_data

    def __hash__(self):
        return hash(
            (
                self.norb,
                self.nelec,
                self.n_opt_iter,
            )
        )


@dataclass
class noci(wave_function):
    norb: int
    nelec: Tuple[int, int]
    ndets: int

    @partial(jit, static_argnums=0)
    def calc_overlap_single_det(self, walker_up, walker_dn, trial_up, trial_dn):
        return jnp.linalg.det(
            trial_up[:, : self.nelec[0]].T @ walker_up
        ) * jnp.linalg.det(trial_dn[:, : self.nelec[1]].T @ walker_dn)

    @partial(jit, static_argnums=0)
    def calc_overlap(self, walker_up, walker_dn, wave_data):
        ci_coeffs = wave_data[0]
        dets = wave_data[1]
        overlaps = vmap(self.calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        return jnp.sum(ci_coeffs * overlaps)

    @partial(jit, static_argnums=0)
    def calc_overlap_vmap(self, walkers, wave_data):
        return vmap(self.calc_overlap, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_green_single_det(self, walker_up, walker_dn, trial_up, trial_dn):
        green_up = (
            walker_up.dot(jnp.linalg.inv(trial_up[:, : self.nelec[0]].T.dot(walker_up)))
        ).T
        green_dn = (
            walker_dn.dot(jnp.linalg.inv(trial_dn[:, : self.nelec[1]].T.dot(walker_dn)))
        ).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def calc_green(self, walker_up, walker_dn, wave_data):
        ci_coeffs = wave_data[0]
        dets = wave_data[1]
        overlaps = vmap(self.calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        overlap = jnp.sum(ci_coeffs * overlaps)
        up_greens, dn_greens = vmap(
            self.calc_green_single_det, in_axes=(None, None, 0, 0)
        )(walker_up, walker_dn, dets[0], dets[1])
        return up_greens, dn_greens, overlaps

    def calc_green_vmap(self, walkers, wave_data):
        pass
    
    @partial(jit, static_argnums=0)
    def calc_force_bias(self, walker_up, walker_dn, rot_chol, wave_data):
        ci_coeffs = wave_data[0]
        dets = wave_data[1]
        up_greens, dn_greens, overlaps = self.calc_green(
            walker_up, walker_dn, wave_data
        )
        overlap = jnp.sum(ci_coeffs * overlaps)
        fb_up = (
            jnp.einsum(
                "ngij,nij,n->g",
                rot_chol[0],
                up_greens,
                ci_coeffs * overlaps,
                optimize="optimal",
            )
            / overlap
        )
        fb_dn = (
            jnp.einsum(
                "ngij,nij,n->g",
                rot_chol[1],
                dn_greens,
                ci_coeffs * overlaps,
                optimize="optimal",
            )
            / overlap
        )
        return fb_up + fb_dn

    @partial(jit, static_argnums=0)
    def calc_force_bias_vmap(self, walkers, ham_data, wave_data):
        return vmap(self.calc_force_bias, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data["rot_chol"], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_energy_single_det(
        self,
        h0,
        rot_h1_up,
        rot_h1_dn,
        rot_chol_up,
        rot_chol_dn,
        walker_up,
        walker_dn,
        trial_up,
        trial_dn,
    ):
        ene0 = h0
        green_walker = self.calc_green_single_det(
            walker_up, walker_dn, trial_up, trial_dn
        )
        ene1 = jnp.sum(green_walker[0] * rot_h1_up) + jnp.sum(
            green_walker[1] * rot_h1_dn
        )
        f_up = jnp.einsum(
            "gij,jk->gik", rot_chol_up, green_walker[0].T, optimize="optimal"
        )
        f_dn = jnp.einsum(
            "gij,jk->gik", rot_chol_dn, green_walker[1].T, optimize="optimal"
        )
        c_up = vmap(jnp.trace)(f_up)
        c_dn = vmap(jnp.trace)(f_dn)
        exc_up = jnp.sum(vmap(lambda x: x * x.T)(f_up))
        exc_dn = jnp.sum(vmap(lambda x: x * x.T)(f_dn))
        ene2 = (
            jnp.sum(c_up * c_up)
            + jnp.sum(c_dn * c_dn)
            + 2.0 * jnp.sum(c_up * c_dn)
            - exc_up
            - exc_dn
        ) / 2.0

        return ene2 + ene1 + ene0

    @partial(jit, static_argnums=0)
    def calc_energy(self, h0, rot_h1, rot_chol, walker_up, walker_dn, wave_data):
        ci_coeffs = wave_data[0]
        dets = wave_data[1]
        overlaps = vmap(self.calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        overlap = jnp.sum(ci_coeffs * overlaps)
        energies = vmap(
            self.calc_energy_single_det, in_axes=(None, 0, 0, 0, 0, None, None, 0, 0)
        )(
            h0,
            rot_h1[0],
            rot_h1[1],
            rot_chol[0],
            rot_chol[1],
            walker_up,
            walker_dn,
            dets[0],
            dets[1],
        )
        ene = jnp.sum(ci_coeffs * overlaps * energies) / overlap
        return ene

    @partial(jit, static_argnums=0)
    def calc_energy_vmap(self, ham_data, walkers, wave_data):
        return vmap(self.calc_energy, in_axes=(None, None, None, 0, 0, None))(
            ham_data["h0"],
            ham_data["rot_h1"],
            ham_data["rot_chol"],
            walkers[0],
            walkers[1],
            wave_data,
        )

    @partial(jit, static_argnums=0)
    def get_trans_rdm1_single_det(self, sd_0_up, sd_0_dn, sd_1_up, sd_1_dn):
        dm_up = (
            (sd_0_up[:, : self.nelec[0]])
            .dot(
                jnp.linalg.inv(
                    sd_1_up[:, : self.nelec[0]].T.dot(sd_0_up[:, : self.nelec[0]])
                )
            )
            .dot(sd_1_up[:, : self.nelec[0]].T)
        )
        dm_dn = (
            (sd_0_dn[:, : self.nelec[1]])
            .dot(
                jnp.linalg.inv(
                    sd_1_dn[:, : self.nelec[1]].T.dot(sd_0_dn[:, : self.nelec[1]])
                )
            )
            .dot(sd_1_dn[:, : self.nelec[1]].T)
        )
        return [dm_up, dm_dn]

    @partial(jit, static_argnums=0)
    def get_rdm1(self, wave_data):
        ci_coeffs = wave_data[0]
        dets = wave_data[1]
        overlaps = vmap(
            vmap(self.calc_overlap_single_det, in_axes=(None, None, 0, 0)),
            in_axes=(0, 0, None, None),
        )(dets[0], dets[1], dets[0], dets[1])
        overlap = jnp.sum(jnp.outer(ci_coeffs, ci_coeffs) * overlaps)
        up_rdm1s, dn_rdm1s = vmap(
            vmap(self.get_trans_rdm1_single_det, in_axes=(0, 0, None, None)),
            in_axes=(None, None, 0, 0),
        )(dets[0], dets[1], dets[0], dets[1])
        up_rdm1 = (
            jnp.einsum(
                "hg,hgij->ij", jnp.outer(ci_coeffs, ci_coeffs) * overlaps, up_rdm1s
            )
            / overlap
        )
        dn_rdm1 = (
            jnp.einsum(
                "hg,hgij->ij", jnp.outer(ci_coeffs, ci_coeffs) * overlaps, dn_rdm1s
            )
            / overlap
        )
        return up_rdm1 + dn_rdm1

    # not implemented
    @partial(jit, static_argnums=0)
    def optimize_orbs(self, ham_data, wave_data):
        return wave_data

    def __hash__(self):
        return hash(
            (
                self.norb,
                self.nelec,
                self.ndets,
            )
        )
