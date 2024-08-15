import os

import numpy as np

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, singledispatchmethod
from typing import Any, List, Sequence, Tuple

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import jax.scipy as jsp
from jax import config, jit, jvp, lax, vjp, vmap

from ad_afqmc import linalg_utils

print = partial(print, flush=True)
config.update("jax_enable_x64", True)


class wave_function(ABC):
    """Base class for wave functions. Contains methods for wave function measurements.

    The measurement methods support two types of walker batches:

    1) unrestricted: walkers is a list ([up, down]). up and down are jnp.ndarrays of shapes
    (nwalkers, norb, nelec[sigma]). In this case the _calc_<property> method is mapped over.

    2) restricted (up and down dets are assumed to be the same): walkers is a jnp.ndarray of shape
    (nwalkers, max(nelec[0], nelec[1])). In this case the _calc_<property>_restricted method is mapped over. By default
    this method is defined to call _calc_<property>. For certain trial states, one can override
    it for computational efficiency.

    A minimal implementation of a wave function should define the _calc_<property> methods for
    property = overlap, force_bias, energy.

    The wave function data is stored in a separate wave_data dictionary. Its structure depends on the
    wave function type and is described in the corresponding class. It may contain "rdm1" which is a
    one-body spin RDM (2, norb, norb). If it is not provided, wave function specific methods are called.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
    """

    norb: int
    nelec: Tuple[int, int]

    @singledispatchmethod
    def calc_overlap(self, walkers, wave_data: dict) -> jnp.ndarray:
        """Calculate the overlap < psi_t | walker > for a batch of walkers.

        Args:
            walkers : list or jnp.ndarray
                The batched walkers.
            wave_data : dict
                The trial wave function data.

        Returns:
            jnp.ndarray: The overlap of the walkers with the trial wave function.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_overlap.register
    def _(self, walkers: list, wave_data: dict) -> jnp.ndarray:
        return vmap(self._calc_overlap, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @calc_overlap.register
    def _(self, walkers: jnp.ndarray, wave_data: dict) -> jnp.ndarray:
        return vmap(self._calc_overlap_restricted, in_axes=(0, None))(
            walkers, wave_data
        )

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(
        self, walker: jnp.ndarray, wave_data: dict
    ) -> jnp.ndarray:
        """Overlap for a single restricted walker."""
        return self._calc_overlap(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], wave_data
        )

    def _calc_overlap(
        self, walker_up: jnp.ndarray, walker_dn: jnp.ndarray, wave_data: dict
    ) -> jnp.ndarray:
        """Overlap for a single walker."""
        raise NotImplementedError("Overlap not defined")

    @singledispatchmethod
    def calc_force_bias(self, walkers, ham_data: dict, wave_data: dict) -> jnp.ndarray:
        """Calculate the force bias < psi_T | chol | walker > / < psi_T | walker > for a batch of walkers.

        Args:
            walkers : list or jnp.ndarray
                The batched walkers.
            ham_data : dict
                The hamiltonian data.
            wave_data : dict
                The trial wave function data.

        Returns:
            jnp.ndarray: The force bias.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_force_bias.register
    def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jnp.ndarray:
        return vmap(self._calc_force_bias, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data
        )

    @calc_force_bias.register
    def _(self, walkers: jnp.ndarray, ham_data: dict, wave_data: dict) -> jnp.ndarray:
        return vmap(self._calc_force_bias_restricted, in_axes=(0, None, None))(
            walkers, ham_data, wave_data
        )

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jnp.ndarray, ham_data: dict, wave_data: dict
    ) -> jnp.ndarray:
        """Force bias for a single restricted walker."""
        return self._calc_force_bias(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], ham_data, wave_data
        )

    def _calc_force_bias(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        ham_data: dict,
        wave_data: dict,
    ) -> jnp.ndarray:
        """Force bias for a single walker."""
        raise NotImplementedError("Force bias not definedr")

    @singledispatchmethod
    def calc_energy(self, walkers, ham_data: dict, wave_data: dict) -> jnp.ndarray:
        """Calculate the energy < psi_T | H | walker > / < psi_T | walker > for a batch of walkers.

        Args:
            walkers : list or jnp.ndarray
                The batched walkers.
            ham_data : dict
                The hamiltonian data.
            wave_data : dict
                The trial wave function data.

        Returns:
            jnp.ndarray: The energy.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_energy.register
    def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jnp.ndarray:
        return vmap(self._calc_energy, in_axes=(0, 0, None, None))(
            walkers[0], walkers[1], ham_data, wave_data
        )

    @calc_energy.register
    def _(self, walkers: jnp.ndarray, ham_data: dict, wave_data: dict) -> jnp.ndarray:
        return vmap(self._calc_energy_restricted, in_axes=(0, None, None))(
            walkers, ham_data, wave_data
        )

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jnp.ndarray, ham_data: dict, wave_data: dict
    ) -> jnp.ndarray:
        """Energy for a single restricted walker."""
        return self._calc_energy(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], ham_data, wave_data
        )

    def _calc_energy(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        ham_data: dict,
        wave_data: dict,
    ) -> jnp.ndarray:
        """Energy for a single walker."""
        raise NotImplementedError("Energy not defined")

    def get_rdm1(self, wave_data: dict) -> jnp.ndarray:
        """Returns the one-body spin reduced density matrix of the trial.
        Used for calculating mean-field shift and as a default value in cases of large
        deviations in observable samples. If wave_data contains "rdm1" this value is used,
        calls otherwise _calc_rdm1.

        Args:
            wave_data : The trial wave function data.

        Returns:
            rdm1: The one-body spin reduced density matrix (2, norb, norb).
        """
        if "rdm1" in wave_data:
            return jnp.array(wave_data["rdm1"])
        else:
            return self._calc_rdm1(wave_data)

    def _calc_rdm1(self, wave_data: dict) -> jnp.ndarray:
        """Calculate the one-body spin reduced density matrix. Exact or approximate rdm1
        of the trial state.

        Args:
            wave_data : The trial wave function data.

        Returns:
            rdm1: The one-body spin reduced density matrix (2, norb, norb).
        """
        raise NotImplementedError(
            "One-body spin RDM not found in wave_data and not implemented for this trial."
        )

    def get_init_walkers(
        self, wave_data: dict, n_walkers: int, restricted: bool = False
    ) -> Sequence:
        """Get the initial walkers. Uses the rdm1 natural orbitals.

        Args:
            wave_data: The trial wave function data.
            n_walkers: The number of walkers.
            restricted: Whether the walkers should be restricted.

        Returns:
            walkers: The initial walkers.
                If restricted, a single jnp.ndarray of shape (nwalkers, norb, nelec[0]).
                If unrestricted, a list of two jnp.ndarrays each of shape (nwalkers, norb, nelec[sigma]).
        """
        rdm1 = self.get_rdm1(wave_data)
        natorbs_up = jnp.linalg.eigh(rdm1[0])[1][:, ::-1][:, : self.nelec[0]]
        natorbs_dn = jnp.linalg.eigh(rdm1[1])[1][:, ::-1][:, : self.nelec[1]]
        if restricted:
            if self.nelec[0] >= self.nelec[1]:
                return jnp.array([natorbs_up + 0.0j] * n_walkers)
            else:
                return jnp.array([natorbs_dn + 0.0j] * n_walkers)
        else:
            return [
                jnp.array([natorbs_up + 0.0j] * n_walkers),
                jnp.array([natorbs_dn + 0.0j] * n_walkers),
            ]

    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Build intermediates for measurements in ham_data. This method is called by the hamiltonian class.

        Args:
            ham_data: The hamiltonian data.
            wave_data: The trial wave function data.

        Returns:
            ham_data: The updated Hamiltonian data.
        """
        return ham_data

    def optimize(self, ham_data: dict, wave_data: dict) -> dict:
        """Optimize the wave function parameters.

        Args:
            ham_data: The hamiltonian data.
            wave_data: The trial wave function data.

        Returns:
            wave_data: The updated trial wave function data.
        """
        return wave_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


class wave_function_cpmc(wave_function):
    """This is used in CPMC. Not as well tested and supported as ab initio currently."""

    @abstractmethod
    def calc_green_diagonal_vmap(self, walkers: Sequence, wave_data: dict) -> jnp.array:
        """Calculate the diagonal elements of the greens function.

        Args:
            walkers: The walkers. (mapped over)
            wave_data: The trial wave function data.

        Returns:
            diag_green: The diagonal elements of the greens function.
        """
        pass

    @abstractmethod
    def calc_full_green_vmap(self, walkers: Sequence, wave_data: dict) -> jnp.array:
        """Calculate the greens function.

        Args:
            walkers: The walkers. (mapped over)
            wave_data: The trial wave function data.

        Returns:
            green: The greens function.
        """
        pass

    @abstractmethod
    def calc_overlap_ratio_vmap(
        self, greens: Sequence, update_indices: Sequence, update_constants: jnp.array
    ) -> jnp.array:
        """Calculate the overlap ratio.

        Args:
            greens: The greens functions. (mapped over)
            update_indices: Proposed update indices.
            constants: Proposed update constants.

        Returns:
            overlap_ratios: The overlap ratios.
        """
        pass

    @abstractmethod
    def update_greens_function_vmap(
        self,
        greens: Sequence,
        ratios: Sequence,
        update_indices: Sequence,
        update_constants: jnp.array,
    ) -> jnp.array:
        """Update the greens function.

        Args:
            greens: The old greens functions. (mapped over)
            ratios: The overlap ratios. (mapped over)
            indices: Where to update.
            constants: What to update with. (mapped over)

        Returns:
            green: The updated greens functions.
        """
        pass


# we assume afqmc is performed in the rhf orbital basis
@dataclass
class rhf(wave_function):
    """Class for the restricted Hartree-Fock wave function.

    The corresponding wave_data should contain "mo_coeff", a jnp.ndarray of shape (norb, nelec).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations.
    """

    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30

    def __post_init__(self):
        assert (
            self.nelec[0] == self.nelec[1]
        ), "RHF requires equal number of up and down electrons."

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(
        self, walker: jnp.ndarray, wave_data: dict
    ) -> jnp.ndarray:
        return jnp.linalg.det(wave_data["mo_coeff"].T.conj() @ walker) ** 2

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jnp.ndarray, walker_dn: jnp.ndarray, wave_data: dict
    ) -> jnp.ndarray:
        return jnp.linalg.det(
            wave_data["mo_coeff"].T.conj() @ walker_up
        ) * jnp.linalg.det(wave_data["mo_coeff"].T.conj() @ walker_dn)

    @partial(jit, static_argnums=0)
    def _calc_green(self, walker: jnp.ndarray, wave_data: dict) -> jnp.ndarray:
        """Calculates the half green's function.

        Args:
            walker: The walker.
            wave_data: The trial wave function data.

        Returns:
            green: The half green's function.
        """
        return (walker.dot(jnp.linalg.inv(wave_data["mo_coeff"].T.conj() @ walker))).T

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: Sequence, ham_data: dict, wave_data: dict
    ) -> jnp.ndarray:
        green_walker = self._calc_green(walker, wave_data)
        fb = 2.0 * jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"], green_walker, optimize="optimal"
        )
        return fb

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        ham_data: dict,
        wave_data: dict,
    ) -> jnp.ndarray:
        green_walker_up = self._calc_green(walker_up, wave_data)
        green_walker_dn = self._calc_green(walker_dn, wave_data)
        green_walker = green_walker_up + green_walker_dn
        fb = jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"], green_walker, optimize="optimal"
        )
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jnp.ndarray, ham_data: dict, wave_data: dict
    ) -> jnp.ndarray:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker = self._calc_green(walker, wave_data)
        ene1 = 2.0 * jnp.sum(green_walker * rot_h1)
        f = jnp.einsum("gij,jk->gik", rot_chol, green_walker.T, optimize="optimal")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = 2.0 * jnp.sum(c * c) - exc
        return ene2 + ene1 + ene0

    @partial(jit, static_argnums=0)
    def _calc_energy(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        ham_data: dict,
        wave_data: dict,
    ) -> jnp.ndarray:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker_up = self._calc_green(walker_up, wave_data)
        green_walker_dn = self._calc_green(walker_dn, wave_data)
        green_walker = green_walker_up + green_walker_dn
        ene1 = jnp.sum(green_walker * rot_h1)
        f = jnp.einsum("gij,jk->gik", rot_chol, green_walker.T, optimize="optimal")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = jnp.sum(c * c) - exc
        return ene2 + ene1 + ene0

    def _calc_rdm1(self, wave_data: dict) -> jnp.ndarray:
        rdm1 = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
        return rdm1

    @partial(jit, static_argnums=0)
    def optimize(self, ham_data: dict, wave_data: dict) -> dict:
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        h2 = ham_data["chol"]
        h2 = h2.reshape((h2.shape[0], h1.shape[0], h1.shape[0]))
        nelec = self.nelec[0]
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

        dm0 = 2 * wave_data["mo_coeff"] @ wave_data["mo_coeff"].T.conj()
        _, mo_coeff = lax.scan(scanned_fun, dm0, None, length=self.n_opt_iter)
        wave_data["mo_coeff"] = mo_coeff[-1][:, :nelec]
        return wave_data

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ (
            (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        )
        ham_data["rot_chol"] = jnp.einsum(
            "pi,gij->gpj",
            wave_data["mo_coeff"].T.conj(),
            ham_data["chol"].reshape(-1, self.norb, self.norb),
        )
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class uhf(wave_function):
    """Class for the unrestricted Hartree-Fock wave function.

    The corresponding wave_data contains "mo_coeff", a list of two jnp.ndarrays of shape (norb, nelec[sigma]).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations.
    """

    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        wave_data: dict,
    ) -> complex:
        return jnp.linalg.det(
            wave_data["mo_coeff"][0].T.conj() @ walker_up
        ) * jnp.linalg.det(wave_data["mo_coeff"][1].T.conj() @ walker_dn)

    @partial(jit, static_argnums=0)
    def _calc_green(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        wave_data: dict,
    ) -> list:
        """Calculates the half green's function.

        Args:
            walker_up: The walker for spin up.
            walker_dn: The walker for spin down.
            wave_data: The trial wave function data.

        Returns:
            green: The half green's function for spin up and spin down.
        """
        green_up = (
            walker_up.dot(jnp.linalg.inv(wave_data["mo_coeff"][0].T.conj() @ walker_up))
        ).T
        green_dn = (
            walker_dn.dot(jnp.linalg.inv(wave_data["mo_coeff"][1].T.conj() @ walker_dn))
        ).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        ham_data: dict,
        wave_data: dict,
    ) -> jnp.ndarray:
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
        fb_up = jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"][0], green_walker[0], optimize="optimal"
        )
        fb_dn = jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"][1], green_walker[1], optimize="optimal"
        )
        return fb_up + fb_dn

    @partial(jit, static_argnums=0)
    def _calc_energy(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        ham_data: dict,
        wave_data: dict,
    ) -> jnp.ndarray:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
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

    def _calc_rdm1(self, wave_data: dict) -> jnp.ndarray:
        dm_up = wave_data["mo_coeff"][0] @ wave_data["mo_coeff"][0].T.conj()
        dm_dn = wave_data["mo_coeff"][1] @ wave_data["mo_coeff"][1].T.conj()
        return jnp.array([dm_up, dm_dn])

    @partial(jit, static_argnums=0)
    def optimize(self, ham_data: dict, wave_data: dict) -> dict:
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

        dm0 = self._calc_rdm1(wave_data)
        _, mo_coeff = lax.scan(scanned_fun, dm0, None, length=self.n_opt_iter)

        wave_data["mo_coeff"] = [
            mo_coeff[-1][0][:, : nelec[0]],
            mo_coeff[-1][1][:, : nelec[1]],
        ]
        return wave_data

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        ham_data["rot_h1"] = [
            wave_data["mo_coeff"][0].T.conj() @ ham_data["h1"][0],
            wave_data["mo_coeff"][1].T.conj() @ ham_data["h1"][1],
        ]
        ham_data["rot_chol"] = [
            jnp.einsum(
                "pi,gij->gpj",
                wave_data["mo_coeff"][0].T.conj(),
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            ),
            jnp.einsum(
                "pi,gij->gpj",
                wave_data["mo_coeff"][1].T.conj(),
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            ),
        ]
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class uhf_cpmc(uhf, wave_function_cpmc):

    @partial(jit, static_argnums=0)
    def calc_green_diagonal(
        self, walker_up: jnp.array, walker_dn: jnp.array, wave_data: dict
    ) -> jnp.array:
        green_up = (
            walker_up
            @ jnp.linalg.inv(wave_data["mo_coeff"][0].T.dot(walker_up))
            @ wave_data["mo_coeff"][0].T
        ).diagonal()
        green_dn = (
            walker_dn
            @ jnp.linalg.inv(wave_data["mo_coeff"][1].T.dot(walker_dn))
            @ wave_data["mo_coeff"][1].T
        ).diagonal()
        return jnp.array([green_up, green_dn])

    @partial(jit, static_argnums=0)
    def calc_green_diagonal_vmap(self, walkers: Sequence, wave_data: dict) -> jnp.array:
        return vmap(self.calc_green_diagonal, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_overlap_ratio(
        self, green: jnp.array, update_indices: jnp.array, update_constants: jnp.array
    ) -> float:
        spin_i, i = update_indices[0]
        spin_j, j = update_indices[1]
        ratio = (1 + update_constants[0] * green[spin_i, i, i]) * (
            1 + update_constants[1] * green[spin_j, j, j]
        ) - (spin_i == spin_j) * update_constants[0] * update_constants[1] * (
            green[spin_i, i, j] * green[spin_j, j, i]
        )
        return ratio

    @partial(jit, static_argnums=0)
    def calc_overlap_ratio_vmap(
        self, greens: jnp.array, update_indices: jnp.array, update_constants: jnp.array
    ) -> jnp.array:
        return vmap(self.calc_overlap_ratio, in_axes=(0, None, None))(
            greens, update_indices, update_constants
        )

    @partial(jit, static_argnums=0)
    def calc_full_green(
        self, walker_up: jnp.array, walker_dn: jnp.array, wave_data: dict
    ) -> jnp.array:
        green_up = (
            walker_up
            @ jnp.linalg.inv(wave_data["mo_coeff"][0].T.dot(walker_up))
            @ wave_data["mo_coeff"][0].T
        ).T
        green_dn = (
            walker_dn
            @ jnp.linalg.inv(wave_data["mo_coeff"][1].T.dot(walker_dn))
            @ wave_data["mo_coeff"][1].T
        ).T
        return jnp.array([green_up, green_dn])

    @partial(jit, static_argnums=0)
    def calc_full_green_vmap(self, walkers: Sequence, wave_data: Any) -> jnp.array:
        return vmap(self.calc_full_green, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def update_greens_function(
        self,
        green: jnp.array,
        ratio: float,
        update_indices: jnp.array,
        update_constants: jnp.array,
    ) -> jnp.array:
        spin_i, i = update_indices[0]
        spin_j, j = update_indices[1]
        sg_i = green[spin_i, i].at[i].add(-1)
        sg_j = green[spin_j, j].at[j].add(-1)
        g_ii = green[spin_i, i, i]
        g_jj = green[spin_j, j, j]
        g_ij = (spin_i == spin_j) * green[spin_i, i, j]
        g_ji = (spin_i == spin_j) * green[spin_j, j, i]
        green = green.at[spin_i, :, :].add(
            (update_constants[0] / ratio)
            * jnp.outer(
                green[spin_i, :, i],
                update_constants[1] * (g_ij * sg_j - g_jj * sg_i) - sg_i,
            )
        )
        green = green.at[spin_j, :, :].add(
            (update_constants[1] / ratio)
            * jnp.outer(
                green[spin_j, :, j],
                update_constants[0] * (g_ji * sg_i - g_ii * sg_j) - sg_j,
            )
        )
        return green

    @partial(jit, static_argnums=0)
    def update_greens_function_vmap(
        self, greens, ratios, update_indices, update_constants
    ):
        return vmap(self.update_greens_function, in_axes=(0, 0, None, 0))(
            greens, ratios, update_indices, update_constants
        )

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class ghf(wave_function):
    """Class for the generalized Hartree-Fock wave function.

    The corresponding wave_data should contain "mo_coeff", a jnp.ndarray of shape (2 * norb, nelec[0] + nelec[1]).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations
    """

    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jnp.ndarray, walker_dn: jnp.ndarray, wave_data: dict
    ) -> jnp.ndarray:
        return jnp.linalg.det(
            jnp.hstack(
                [
                    wave_data["mo_coeff"][: self.norb].T @ walker_up,
                    wave_data["mo_coeff"][self.norb :].T @ walker_dn,
                ]
            )
        )

    @partial(jit, static_argnums=0)
    def _calc_green(
        self, walker_up: jnp.ndarray, walker_dn: jnp.ndarray, wave_data: dict
    ) -> jnp.ndarray:
        overlap_mat = jnp.hstack(
            [
                wave_data["mo_coeff"][: self.norb].T @ walker_up,
                wave_data["mo_coeff"][self.norb :].T @ walker_dn,
            ]
        )
        inv = jnp.linalg.inv(overlap_mat)
        green = (
            jnp.vstack(
                [walker_up @ inv[: self.nelec[0]], walker_dn @ inv[self.nelec[0] :]]
            )
        ).T
        return green

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        ham_data: dict,
        wave_data: dict,
    ) -> jnp.ndarray:
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
        fb = jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"], green_walker, optimize="optimal"
        )
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        ham_data: dict,
        wave_data: dict,
    ) -> jnp.ndarray:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
        ene1 = jnp.sum(green_walker * rot_h1)
        f = jnp.einsum("gij,jk->gik", rot_chol, green_walker.T, optimize="optimal")
        coul = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = (jnp.sum(coul * coul) - exc) / 2.0
        return ene2 + ene1 + ene0

    def _calc_rdm1(self, wave_data: dict) -> jnp.ndarray:
        dm = (
            wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]]
            @ wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T
        )
        dm_up = dm[: self.norb, : self.norb]
        dm_dn = dm[self.norb :, self.norb :]
        return jnp.array([dm_up, dm_dn])

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ jnp.block(
            [
                [ham_data["h1"][0], jnp.zeros_like(ham_data["h1"][1])],
                [jnp.zeros_like(ham_data["h1"][0]), ham_data["h1"][1]],
            ]
        )
        ham_data["rot_chol"] = vmap(
            lambda x: jnp.hstack(
                [
                    wave_data["mo_coeff"].T.conj()[:, : self.norb] @ x,
                    wave_data["mo_coeff"].T.conj()[:, self.norb :] @ x,
                ]
            ),
            in_axes=(0),
        )(ham_data["chol"].reshape(-1, self.norb, self.norb))
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class ghf_cpmc(ghf, wave_function_cpmc):

    @partial(jit, static_argnums=0)
    def calc_green_diagonal(
        self, walker_up: jnp.array, walker_dn: jnp.array, wave_data: dict
    ) -> jnp.array:
        walker_ghf = jsp.linalg.block_diag(walker_up, walker_dn)
        overlap_mat = (
            wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T @ walker_ghf
        )
        inv = jnp.linalg.inv(overlap_mat)
        green = (
            walker_ghf
            @ inv
            @ wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T
        ).diagonal()
        return jnp.array([green[: self.norb], green[self.norb :]])

    @partial(jit, static_argnums=0)
    def calc_green_diagonal_vmap(self, walkers: Sequence, wave_data: dict) -> jnp.array:
        return vmap(self.calc_green_diagonal, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_overlap_ratio(
        self, green: jnp.array, update_indices: jnp.array, update_constants: jnp.array
    ) -> float:
        spin_i, i = update_indices[0]
        spin_j, j = update_indices[1]
        i = i + (spin_i == 1) * self.norb
        j = j + (spin_j == 1) * self.norb
        ratio = (1 + update_constants[0] * green[i, i]) * (
            1 + update_constants[1] * green[j, j]
        ) - update_constants[0] * update_constants[1] * (green[i, j] * green[j, i])
        return ratio

    @partial(jit, static_argnums=0)
    def calc_overlap_ratio_vmap(
        self, greens: jnp.array, update_indices: jnp.array, update_constants: jnp.array
    ) -> jnp.array:
        return vmap(self.calc_overlap_ratio, in_axes=(0, None, None))(
            greens, update_indices, update_constants
        )

    @partial(jit, static_argnums=0)
    def calc_full_green(
        self, walker_up: jnp.array, walker_dn: jnp.array, wave_data: dict
    ) -> jnp.array:
        walker_ghf = jsp.linalg.block_diag(walker_up, walker_dn)
        green = (
            walker_ghf
            @ jnp.linalg.inv(
                wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T @ walker_ghf
            )
            @ wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T
        ).T
        return green

    @partial(jit, static_argnums=0)
    def calc_full_green_vmap(self, walkers: Sequence, wave_data: dict) -> jnp.array:
        return vmap(self.calc_full_green, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def update_greens_function(
        self,
        green: jnp.array,
        ratio: float,
        update_indices: jnp.array,
        update_constants: jnp.array,
    ) -> jnp.array:
        spin_i, i = update_indices[0]
        spin_j, j = update_indices[1]
        i = i + (spin_i == 1) * self.norb
        j = j + (spin_j == 1) * self.norb
        sg_i = green[i].at[i].add(-1)
        sg_j = green[j].at[j].add(-1)
        green += (update_constants[0] / ratio) * jnp.outer(
            green[:, i],
            update_constants[1] * (green[i, j] * sg_j - green[j, j] * sg_i) - sg_i,
        ) + (update_constants[1] / ratio) * jnp.outer(
            green[:, j],
            update_constants[0] * (green[j, i] * sg_i - green[i, i] * sg_j) - sg_j,
        )
        return green

    @partial(jit, static_argnums=0)
    def update_greens_function_vmap(
        self, greens, ratios, update_indices, update_constants
    ):
        return vmap(self.update_greens_function, in_axes=(0, 0, None, 0))(
            greens, ratios, update_indices, update_constants
        )

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class noci(wave_function):
    """Class for the NOCI wave function.

    The corresponding wave_data should contain "ci_coeffs_dets", a list [ci_coeffs, dets]
    where ci_coeffs is a jnp.ndarray of shape (ndets) and dets is a list [dets_up, dets_dn]
    with each being a jnp.ndarray of shape (ndets, norb, nelec[sigma]), both ci_coeffs and dets
    are assumed to be real. The measurement methods make use of half-rotated integrals
    which are stored in ham_data (rot_h1 and rot_chol for each det).

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        ndets: Number of determinants in the NOCI expansion.
    """

    norb: int
    nelec: Tuple[int, int]
    ndets: int

    @partial(jit, static_argnums=0)
    def _calc_overlap_single_det(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        trial_up: jnp.ndarray,
        trial_dn: jnp.ndarray,
    ) -> jnp.ndarray:
        """Calculate the overlap with a single determinant in the NOCI trial."""
        return jnp.linalg.det(
            trial_up[:, : self.nelec[0]].T.conj() @ walker_up
        ) * jnp.linalg.det(trial_dn[:, : self.nelec[1]].T.conj() @ walker_dn)

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jnp.ndarray, walker_dn: jnp.ndarray, wave_data: dict
    ) -> jnp.ndarray:
        ci_coeffs = wave_data["ci_coeffs_dets"][0]
        dets = wave_data["ci_coeffs_dets"][1]
        overlaps = vmap(self._calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        return jnp.sum(ci_coeffs * overlaps)

    @partial(jit, static_argnums=0)
    def _calc_green_single_det(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        trial_up: jnp.ndarray,
        trial_dn: jnp.ndarray,
    ) -> jnp.ndarray:
        """Calculate the half greens function with a single determinant in the NOCI trial."""
        green_up = (
            walker_up.dot(jnp.linalg.inv(trial_up[:, : self.nelec[0]].T.dot(walker_up)))
        ).T
        green_dn = (
            walker_dn.dot(jnp.linalg.inv(trial_dn[:, : self.nelec[1]].T.dot(walker_dn)))
        ).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_green(
        self, walker_up: jnp.ndarray, walker_dn: jnp.ndarray, wave_data: dict
    ) -> jnp.ndarray:
        """Calculate the half greens function for the full trial."""
        dets = wave_data["ci_coeffs_dets"][1]
        overlaps = vmap(self._calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        up_greens, dn_greens = vmap(
            self._calc_green_single_det, in_axes=(None, None, 0, 0)
        )(walker_up, walker_dn, dets[0], dets[1])
        return up_greens, dn_greens, overlaps

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        ham_data: dict,
        wave_data: dict,
    ) -> jnp.ndarray:
        ci_coeffs = wave_data["ci_coeffs_dets"][0]
        up_greens, dn_greens, overlaps = self._calc_green(
            walker_up, walker_dn, wave_data
        )
        overlap = jnp.sum(ci_coeffs * overlaps)
        fb_up = (
            jnp.einsum(
                "ngij,nij,n->g",
                ham_data["rot_chol"][0],
                up_greens,
                ci_coeffs * overlaps,
                optimize="optimal",
            )
            / overlap
        )
        fb_dn = (
            jnp.einsum(
                "ngij,nij,n->g",
                ham_data["rot_chol"][1],
                dn_greens,
                ci_coeffs * overlaps,
                optimize="optimal",
            )
            / overlap
        )
        return fb_up + fb_dn

    @partial(jit, static_argnums=0)
    def _calc_energy_single_det(
        self,
        h0: float,
        rot_h1_up: jnp.ndarray,
        rot_h1_dn: jnp.ndarray,
        rot_chol_up: jnp.ndarray,
        rot_chol_dn: jnp.ndarray,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        trial_up: jnp.ndarray,
        trial_dn: jnp.ndarray,
    ) -> jnp.ndarray:
        """Calculate the energy with a single determinant in the NOCI trial."""
        ene0 = h0
        green_walker = self._calc_green_single_det(
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
    def _calc_energy(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        ham_data: dict,
        wave_data: dict,
    ) -> jnp.ndarray:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ci_coeffs = wave_data["ci_coeffs_dets"][0]
        dets = wave_data["ci_coeffs_dets"][1]
        overlaps = vmap(self._calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        overlap = jnp.sum(ci_coeffs * overlaps)
        energies = vmap(
            self._calc_energy_single_det, in_axes=(None, 0, 0, 0, 0, None, None, 0, 0)
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
    def _get_trans_rdm1_single_det(self, sd_0_up, sd_0_dn, sd_1_up, sd_1_dn) -> list:
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
    def _calc_rdm1(self, wave_data: dict) -> jnp.ndarray:
        ci_coeffs = wave_data["ci_coeffs_dets"][0]
        dets = wave_data["ci_coeffs_dets"][1]
        overlaps = vmap(
            vmap(self._calc_overlap_single_det, in_axes=(None, None, 0, 0)),
            in_axes=(0, 0, None, None),
        )(dets[0], dets[1], dets[0], dets[1])
        overlap = jnp.sum(jnp.outer(ci_coeffs, ci_coeffs) * overlaps)
        up_rdm1s, dn_rdm1s = vmap(
            vmap(self._get_trans_rdm1_single_det, in_axes=(0, 0, None, None)),
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
        return jnp.array([up_rdm1, dn_rdm1])

    @partial(jit, static_argnums=(0,))
    def _rot_orbs_single_det(
        self, ham_data: dict, trial_up: jnp.ndarray, trial_dn: jnp.ndarray
    ) -> Tuple:
        rot_h1 = [
            trial_up.T.conj() @ ham_data["h1"][0],
            trial_dn.T.conj() @ ham_data["h1"][1],
        ]
        rot_chol = [
            jnp.einsum(
                "pi,gij->gpj",
                trial_up.T,
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            ),
            jnp.einsum(
                "pi,gij->gpj",
                trial_dn.T,
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            ),
        ]
        return rot_h1, rot_chol

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        trial = wave_data["ci_coeffs_dets"][1]
        ham_data["rot_h1"], ham_data["rot_chol"] = vmap(
            self._rot_orbs_single_det, in_axes=(None, 0, 0)
        )(ham_data, trial[0], trial[1])
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class wave_function_auto(wave_function):
    """This wave function only requires the definition of overlap functions.
    It evaluates force bias and local energy by differentiating various overlaps
    (single derivatives with AD and double with FD)."""

    def __post_init__(self):
        """eps is the finite difference step size in local energy calculations."""
        if not hasattr(self, "eps"):
            self.eps = 1.0e-4

    @partial(jit, static_argnums=0)
    def _overlap_with_rot_sd_restricted(
        self,
        x_gamma: jnp.ndarray,
        walker: jnp.ndarray,
        chol: jnp.ndarray,
        wave_data: dict,
    ) -> jnp.ndarray:
        """Helper function for calculating force bias using AD,
        evaluates < psi_T | exp(x_gamma * chol) | walker > to linear order"""
        x_chol = jnp.einsum(
            "gij,g->ij", chol.reshape(-1, self.norb, self.norb), x_gamma
        )
        walker_1 = walker + x_chol.dot(walker)
        return self._calc_overlap_restricted(walker_1, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jnp.ndarray, ham_data: dict, wave_data: dict
    ) -> jnp.ndarray:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker > by differentiating
        < psi_T | exp(x_gamma * chol) | walker > / < psi_T | walker >"""
        val, grad = vjp(
            self._overlap_with_rot_sd_restricted,
            jnp.zeros((ham_data["chol"].shape[0],)) + 0.0j,
            walker,
            ham_data["chol"],
            wave_data,
        )
        return grad(1.0 + 0.0j)[0] / val

    @partial(jit, static_argnums=0)
    def _overlap_with_rot_sd(
        self,
        x_gamma: jnp.ndarray,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        chol: jnp.ndarray,
        wave_data: Any,
    ) -> jnp.ndarray:
        """Helper function for calculating force bias using AD,
        evaluates < psi_T | exp(x_gamma * chol) | walker > to linear order"""
        x_chol = jnp.einsum(
            "gij,g->ij", chol.reshape(-1, self.norb, self.norb), x_gamma
        )
        walker_up_1 = walker_up + x_chol.dot(walker_up)
        walker_dn_1 = walker_dn + x_chol.dot(walker_dn)
        return self._calc_overlap(walker_up_1, walker_dn_1, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        ham_data: dict,
        wave_data: Any,
    ) -> jnp.ndarray:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker > by differentiating
        < psi_T | exp(x_gamma * chol) | walker > / < psi_T | walker >"""
        val, grad = vjp(
            self._overlap_with_rot_sd,
            jnp.zeros((ham_data["chol"].shape[0],)) + 0.0j,
            walker_up,
            walker_dn,
            ham_data["chol"],
            wave_data,
        )
        return grad(1.0 + 0.0j)[0] / val

    @partial(jit, static_argnums=0)
    def _overlap_with_single_rot_restricted(
        self, x: float, h1: jnp.ndarray, walker: jnp.ndarray, wave_data: dict
    ) -> jnp.ndarray:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * h1) | walker > to linear order"""
        walker_2 = walker + x * h1.dot(walker)
        return self._calc_overlap_restricted(walker_2, wave_data)

    @partial(jit, static_argnums=0)
    def _overlap_with_double_rot_restricted(
        self, x: float, chol_i: jnp.ndarray, walker: jnp.ndarray, wave_data: dict
    ) -> jnp.ndarray:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * chol_i) | walker > to quadratic order"""
        walker2 = (
            walker
            + x * chol_i.dot(walker)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
        )
        return self._calc_overlap_restricted(walker2, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self,
        walker: jnp.ndarray,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        """Calculates local energy using AD and finite difference for the two body term"""

        h0, h1, chol, v0 = (
            ham_data["h0"],
            (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0,
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            ham_data["normal_ordering_term"],
        )

        x = 0.0
        # one body
        f1 = lambda a: self._overlap_with_single_rot_restricted(
            a, h1 + v0, walker, wave_data
        )
        overlap, d_overlap = jvp(f1, [x], [1.0])

        # two body
        eps = self.eps

        # carry: [eps, walker, wave_data]
        def scanned_fun(carry, x):
            eps, walker, wave_data = carry
            return carry, self._overlap_with_double_rot_restricted(
                eps, x, walker, wave_data
            )

        _, overlap_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
        _, overlap_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
        _, overlap_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
        d_2_overlap = (overlap_p - 2.0 * overlap_0 + overlap_m) / eps / eps

        return (d_overlap + jnp.sum(d_2_overlap) / 2.0) / overlap + h0

    @partial(jit, static_argnums=0)
    def _overlap_with_single_rot(
        self,
        x: float,
        h1: jnp.ndarray,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        wave_data: Any,
    ) -> jnp.ndarray:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * h1) | walker > to linear order"""
        walker_up_1 = walker_up + x * h1[0].dot(walker_up)
        walker_dn_1 = walker_dn + x * h1[1].dot(walker_dn)
        return self._calc_overlap(walker_up_1, walker_dn_1, wave_data)

    @partial(jit, static_argnums=0)
    def _overlap_with_double_rot(
        self,
        x: float,
        chol_i: jnp.ndarray,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        wave_data: Any,
    ) -> jnp.ndarray:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * chol_i) | walker > to quadratic order"""
        walker_up_1 = (
            walker_up
            + x * chol_i.dot(walker_up)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker_up))
        )
        walker_dn_1 = (
            walker_dn
            + x * chol_i.dot(walker_dn)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker_dn))
        )
        return self._calc_overlap(walker_up_1, walker_dn_1, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_energy(
        self,
        walker_up: jnp.ndarray,
        walker_dn: jnp.ndarray,
        ham_data: dict,
        wave_data: Any,
    ) -> jnp.ndarray:
        """Calculates local energy using AD and finite difference for the two body term"""

        h0, h1, chol, v0 = (
            ham_data["h0"],
            ham_data["h1"],
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            ham_data["normal_ordering_term"],
        )

        x = 0.0
        # one body
        f1 = lambda a: self._overlap_with_single_rot(
            a, h1 + v0, walker_up, walker_dn, wave_data
        )
        val1, dx1 = jvp(f1, [x], [1.0])

        # two body
        # vmap_fun = vmap(
        #     self._overlap_with_double_rot, in_axes=(None, 0, None, None, None)
        # )

        eps = self.eps

        # carry: [eps, walker, wave_data]
        def scanned_fun(carry, chol_i):
            eps, walker_up, walker_dn, wave_data = carry
            return carry, self._overlap_with_double_rot(
                eps, chol_i, walker_up, walker_dn, wave_data
            )

        _, overlap_p = lax.scan(
            scanned_fun, (eps, walker_up, walker_dn, wave_data), chol
        )
        _, overlap_0 = lax.scan(
            scanned_fun, (0.0, walker_up, walker_dn, wave_data), chol
        )
        _, overlap_m = lax.scan(
            scanned_fun, (-1.0 * eps, walker_up, walker_dn, wave_data), chol
        )
        d_2_overlap = (overlap_p - 2.0 * overlap_0 + overlap_m) / eps / eps

        # dx2 = (
        #     (
        #         vmap_fun(eps, chol, walker_up, walker_dn, wave_data)
        #         - 2.0 * vmap_fun(zero, chol, walker_up, walker_dn, wave_data)
        #         + vmap_fun(-1.0 * eps, chol, walker_up, walker_dn, wave_data)
        #     )
        #     / eps
        #     / eps
        # )

        return (dx1 + jnp.sum(d_2_overlap) / 2.0) / val1 + h0

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        v0 = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            optimize="optimal",
        )
        ham_data["normal_ordering_term"] = -v0
        return ham_data


@dataclass
class multislater(wave_function_auto):
    """Multislater wave function implemented using the auto class.

    We work in the orbital basis of the wave function.
    Associated wave_data consists of excitation indices and ci coefficients:
        Acre: Alpha creation indices.
        Ades: Alpha destruction indices.
        Bcre: Beta creation indices.
        Bdes: Beta destruction indices.
        coeff: Coefficients of the determinants.
        ref_det: Reference determinant.
    """

    norb: int
    nelec: Tuple[int, int]
    max_excitation: int  # maximum of sum of alpha and beta excitation ranks
    eps: float = 1.0e-4  # finite difference step size in local energy calculations

    @partial(jit, static_argnums=0)
    def _det_overlap(
        self, green: jnp.ndarray, cre: jnp.ndarray, des: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.linalg.det(green[jnp.ix_(cre, des)])

    @partial(jit, static_argnums=0)
    def _calc_green_restricted(
        self, walker: jnp.ndarray, wave_data: dict
    ) -> jnp.ndarray:
        ref_det = wave_data["ref_det"][0]
        return (
            walker.dot(
                jnp.linalg.inv(walker[jnp.nonzero(ref_det, size=self.nelec[0])[0], :])
            )
        ).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jnp.ndarray, wave_data: dict) -> complex:
        """Calclulates < psi_T | walker > efficiently using Wick's theorem"""
        Acre, Ades, Bcre, Bdes, coeff, ref_det = (
            wave_data["Acre"],
            wave_data["Ades"],
            wave_data["Bcre"],
            wave_data["Bdes"],
            wave_data["coeff"],
            wave_data["ref_det"],
        )
        green = self._calc_green_restricted(walker, wave_data)

        # overlap with the reference determinant
        overlap_0 = (
            jnp.linalg.det(walker[jnp.nonzero(ref_det[0], size=self.nelec[0])[0], :])
            ** 2
        )

        # overlap / overlap_0
        overlap = coeff[(0, 0)] + 0.0j

        for i in range(1, self.max_excitation + 1):
            overlap += vmap(self._det_overlap, in_axes=(None, 0, 0))(
                green, Acre[(i, 0)], Ades[(i, 0)]
            ).dot(coeff[(i, 0)])
            overlap += vmap(self._det_overlap, in_axes=(None, 0, 0))(
                green, Bcre[(0, i)], Bdes[(0, i)]
            ).dot(coeff[(0, i)])

            for j in range(1, self.max_excitation - i + 1):
                overlap_a = vmap(self._det_overlap, in_axes=(None, 0, 0))(
                    green, Acre[(i, j)], Ades[(i, j)]
                )
                overlap_b = vmap(self._det_overlap, in_axes=(None, 0, 0))(
                    green, Bcre[(i, j)], Bdes[(i, j)]
                )
                overlap += (overlap_a * overlap_b) @ coeff[(i, j)]

        return (overlap * overlap_0)[0]

    @partial(jit, static_argnums=0)
    def _calc_green(
        self, walker_up: jnp.ndarray, walker_dn: jnp.ndarray, wave_data: dict
    ) -> list:
        ref_det = wave_data["ref_det"]
        green_up = (
            walker_up.dot(
                jnp.linalg.inv(
                    walker_up[jnp.nonzero(ref_det[0], size=self.nelec[0])[0], :]
                )
            )
        ).T
        green_dn = (
            walker_dn.dot(
                jnp.linalg.inv(
                    walker_dn[jnp.nonzero(ref_det[1], size=self.nelec[1])[0], :]
                )
            )
        ).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jnp.ndarray, walker_dn: jnp.ndarray, wave_data: dict
    ) -> complex:
        """Calclulates < psi_T | walker > efficiently using Wick's theorem"""
        Acre, Ades, Bcre, Bdes, coeff, ref_det = (
            wave_data["Acre"],
            wave_data["Ades"],
            wave_data["Bcre"],
            wave_data["Bdes"],
            wave_data["coeff"],
            wave_data["ref_det"],
        )
        green = self._calc_green(walker_up, walker_dn, wave_data)

        # overlap with the reference determinant
        overlap_0 = jnp.linalg.det(
            walker_up[jnp.nonzero(ref_det[0], size=self.nelec[0])[0], :]
        ) * jnp.linalg.det(walker_dn[jnp.nonzero(ref_det[1], size=self.nelec[1])[0], :])

        # overlap / overlap_0
        overlap = coeff[(0, 0)] + 0.0j

        for i in range(1, self.max_excitation + 1):
            overlap += vmap(self._det_overlap, in_axes=(None, 0, 0))(
                green[0], Acre[(i, 0)], Ades[(i, 0)]
            ).dot(coeff[(i, 0)])
            overlap += vmap(self._det_overlap, in_axes=(None, 0, 0))(
                green[1], Bcre[(0, i)], Bdes[(0, i)]
            ).dot(coeff[(0, i)])

            for j in range(1, self.max_excitation - i + 1):
                overlap_a = vmap(self._det_overlap, in_axes=(None, 0, 0))(
                    green[0], Acre[(i, j)], Ades[(i, j)]
                )
                overlap_b = vmap(self._det_overlap, in_axes=(None, 0, 0))(
                    green[1], Bcre[(i, j)], Bdes[(i, j)]
                )
                overlap += (overlap_a * overlap_b) @ coeff[(i, j)]

        return (overlap * overlap_0)[0]

    def _calc_rdm1(self, wave_data: dict) -> jnp.ndarray:
        """Spatial 1RDM of the reference det"""
        ref_det = wave_data["ref_det"]
        orb_mat = np.eye(self.norb)
        orbs_up = orb_mat[:, ref_det[0] > 0]
        orbs_dn = orb_mat[:, ref_det[1] > 0]
        rdm_up = orbs_up @ orbs_up.T
        rdm_dn = orbs_dn @ orbs_dn.T
        return jnp.array([rdm_up, rdm_dn])

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class CISD(wave_function_auto):
    """This class contains functions for the CISD wavefunction
    |0> + c(ia) |ia> + c(ia jb) |ia jb>

    . The wave_data need to store the coefficient C(ia) and C(ia jb)
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations

    @partial(jit, static_argnums=0)
    def _calc_green_restricted(self, walker: jnp.ndarray) -> jnp.ndarray:
        return (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jnp.ndarray, wave_data: dict) -> complex:
        nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
        GF = self._calc_green_restricted(walker)
        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        o1 = jnp.einsum("ia,ia", ci1, GF[:, nocc:])
        o2 = 2 * jnp.einsum(
            "iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:]
        ) - jnp.einsum("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:])
        return (1.0 + 2 * o1 + o2) * o0

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class UCISD(wave_function_auto):
    """This class contains functions for the CISD wavefunction
    |0> + c(ia) |ia> + c(ia jb) |ia jb>

    . The wave_data need to store the coefficient C(ia) and C(ia jb)
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations

    @partial(jit, static_argnums=0)
    def _calc_green(
        self, walker_up: jnp.ndarray, walker_dn: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        green_up = (walker_up.dot(jnp.linalg.inv(walker_up[: walker_up.shape[1], :]))).T
        green_dn = (walker_dn.dot(jnp.linalg.inv(walker_dn[: walker_dn.shape[1], :]))).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jnp.ndarray, walker_dn: jnp.ndarray, wave_data: dict
    ) -> complex:

        noccA, ci1A, ci2AA = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        noccB, ci1B, ci2BB = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2AB = wave_data["ci2AB"]
        moA, moB = wave_data["mo_coeff"][0], wave_data["mo_coeff"][1]

        walker_dn_B = moB.T.dot(
            walker_dn[:, :noccB]
        )  # put walker_dn in the basis of alpha reference

        GFA, GFB = self._calc_green(walker_up, walker_dn_B)

        o0 = jnp.linalg.det(walker_up[:noccA, :]) * jnp.linalg.det(
            walker_dn_B[:noccB, :]
        )

        o1 = jnp.einsum("ia,ia", ci1A, GFA[:, noccA:]) + jnp.einsum(
            "ia,ia", ci1B, GFB[:, noccB:]
        )

        # AA
        o2 = 0.25 * jnp.einsum("iajb, ia, jb", ci2AA, GFA[:, noccA:], GFA[:, noccA:])
        o2 -= 0.25 * jnp.einsum("iajb, ib, ja", ci2AA, GFA[:, noccA:], GFA[:, noccA:])

        # BB
        o2 += 0.25 * jnp.einsum("iajb, ia, jb", ci2BB, GFB[:, noccB:], GFB[:, noccB:])
        o2 -= 0.25 * jnp.einsum("iajb, ib, ja", ci2BB, GFB[:, noccB:], GFB[:, noccB:])

        # AB
        o2 += jnp.einsum("iajb, ia, jb", ci2AB, GFA[:, noccA:], GFB[:, noccB:])

        return (1.0 + o1 + o2) * o0

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class CISD_THC(wave_function_auto):
    """This class contains functions for the CISD wavefunction
    |0> + c(ia) |ia> + c(ia jb) |ia jb>

    . The wave_data need to store the coefficient C(ia) and C(ia jb)  in the THC format
    i.e. C(i,a,j,b) = X1(P,i) X2(P,a) V(P,Q) X1(P,j) X2(P,b)
    """

    norb: int
    nelec: int
    eps: float = 1.0e-4  # finite difference step size in local energy calculations

    @partial(jit, static_argnums=0)
    def _calc_green_restricted(self, walker: jnp.ndarray) -> jnp.ndarray:
        return (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jnp.ndarray, wave_data: dict) -> complex:
        nocc, ci1, Xocc, Xvirt, VKL = (
            walker.shape[1],
            wave_data["ci1"],
            wave_data["Xocc"],
            wave_data["Xvirt"],
            wave_data["VKL"],
        )
        GF = self._calc_green_restricted(walker)

        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2

        o1 = jnp.einsum("ia,ia", ci1, GF[:, nocc:])

        # A = jnp.einsum('ia,Pi,Pa->P', GF[:,nocc:], Xocc, Xvirt)
        # o2 = 2*jnp.einsum('P,PQ,Q', A, VKL, A)

        gv = GF[:, nocc:] @ Xvirt.T
        A = jnp.einsum("Pi,iP->P", Xocc, gv)
        o2 = 2 * (A @ VKL).dot(A)

        B = Xocc @ gv
        o2 -= jnp.sum(B * B.T * VKL)

        return (1.0 + 2 * o1 + o2) * o0

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
