from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, singledispatchmethod
from typing import Any, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import jit, jvp, lax, vjp, vmap
from jax._src.typing import DTypeLike

from ad_afqmc import linalg_utils


class wave_function(ABC):
    """Base class for wave functions. Contains methods for wave function measurements.

    The measurement methods support two types of walker batches:

    1) unrestricted: walkers is a list ([up, down]). up and down are jax.Arrays of shapes
    (nwalkers, norb, nelec[sigma]). In this case the _calc_<property> method is mapped over.

    2) restricted (up and down dets are assumed to be the same): walkers is a jax.Array of shape
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
        n_batch: Number of batches used in scan.
    """

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @singledispatchmethod
    def calc_overlap(self, walkers, wave_data: dict) -> jax.Array:
        """Calculate the overlap < psi_t | walker > for a batch of walkers.

        Args:
            walkers : list or jax.Array
                The batched walkers.
            wave_data : dict
                The trial wave function data.

        Returns:
            jax.Array: The overlap of the walkers with the trial wave function.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_overlap.register
    def _(self, walkers: list, wave_data: dict) -> jax.Array:
        n_walkers = walkers[0].shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            walker_batch_0, walker_batch_1 = walker_batch
            overlap_batch = vmap(self._calc_overlap, in_axes=(0, 0, None))(
                walker_batch_0, walker_batch_1, wave_data
            )
            return carry, overlap_batch

        _, overlaps = lax.scan(
            scanned_fun,
            None,
            (
                walkers[0].reshape(self.n_batch, batch_size, self.norb, self.nelec[0]),
                walkers[1].reshape(self.n_batch, batch_size, self.norb, self.nelec[1]),
            ),
        )
        return overlaps.reshape(n_walkers)

    @calc_overlap.register
    def _(self, walkers: jax.Array, wave_data: dict) -> jax.Array:
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            overlap_batch = vmap(self._calc_overlap_restricted, in_axes=(0, None))(
                walker_batch, wave_data
            )
            return carry, overlap_batch

        _, overlaps = lax.scan(
            scanned_fun, None, walkers.reshape(self.n_batch, batch_size, self.norb, -1)
        )
        return overlaps.reshape(n_walkers)

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        """Overlap for a single restricted walker."""
        return self._calc_overlap(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], wave_data
        )

    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        """Overlap for a single walker."""
        raise NotImplementedError("Overlap not defined")

    @singledispatchmethod
    def calc_force_bias(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
        """Calculate the force bias < psi_T | chol | walker > / < psi_T | walker > for a batch of walkers.

        Args:
            walkers : list or jax.Array
                The batched walkers.
            ham_data : dict
                The hamiltonian data.
            wave_data : dict
                The trial wave function data.

        Returns:
            jax.Array: The force bias.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_force_bias.register
    def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        n_walkers = walkers[0].shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            walker_batch_0, walker_batch_1 = walker_batch
            fb_batch = vmap(self._calc_force_bias, in_axes=(0, 0, None, None))(
                walker_batch_0, walker_batch_1, ham_data, wave_data
            )
            return carry, fb_batch

        _, fbs = lax.scan(
            scanned_fun,
            None,
            (
                walkers[0].reshape(self.n_batch, batch_size, self.norb, self.nelec[0]),
                walkers[1].reshape(self.n_batch, batch_size, self.norb, self.nelec[1]),
            ),
        )
        fbs = jnp.concatenate(fbs, axis=0)
        return fbs.reshape(n_walkers, -1)

    @calc_force_bias.register
    def _(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            fb_batch = vmap(self._calc_force_bias_restricted, in_axes=(0, None, None))(
                walker_batch, ham_data, wave_data
            )
            return carry, fb_batch

        _, fbs = lax.scan(
            scanned_fun, None, walkers.reshape(self.n_batch, batch_size, self.norb, -1)
        )
        return fbs.reshape(n_walkers, -1)

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Force bias for a single restricted walker."""
        return self._calc_force_bias(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], ham_data, wave_data
        )

    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        """Force bias for a single walker."""
        raise NotImplementedError("Force bias not definedr")

    @singledispatchmethod
    def calc_energy(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
        """Calculate the energy < psi_T | H | walker > / < psi_T | walker > for a batch of walkers.

        Args:
            walkers : list or jax.Array
                The batched walkers.
            ham_data : dict
                The hamiltonian data.
            wave_data : dict
                The trial wave function data.

        Returns:
            jax.Array: The energy.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_energy.register
    def _(self, walkers: list, ham_data: dict, wave_data: dict) -> jax.Array:
        n_walkers = walkers[0].shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            walker_batch_0, walker_batch_1 = walker_batch
            energy_batch = vmap(self._calc_energy, in_axes=(0, 0, None, None))(
                walker_batch_0, walker_batch_1, ham_data, wave_data
            )
            return carry, energy_batch

        _, energies = lax.scan(
            scanned_fun,
            None,
            (
                walkers[0].reshape(self.n_batch, batch_size, self.norb, self.nelec[0]),
                walkers[1].reshape(self.n_batch, batch_size, self.norb, self.nelec[1]),
            ),
        )
        return energies.reshape(n_walkers)

    @calc_energy.register
    def _(self, walkers: jax.Array, ham_data: dict, wave_data: dict) -> jax.Array:
        n_walkers = walkers.shape[0]
        batch_size = n_walkers // self.n_batch

        def scanned_fun(carry, walker_batch):
            energy_batch = vmap(self._calc_energy_restricted, in_axes=(0, None, None))(
                walker_batch, ham_data, wave_data
            )
            return carry, energy_batch

        _, energies = lax.scan(
            scanned_fun,
            None,
            walkers.reshape(self.n_batch, batch_size, self.norb, -1),
        )
        return energies.reshape(n_walkers)

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Energy for a single restricted walker."""
        return self._calc_energy(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], ham_data, wave_data
        )

    def _calc_energy(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        """Energy for a single walker."""
        raise NotImplementedError("Energy not defined")

    def get_rdm1(self, wave_data: dict) -> jax.Array:
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

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
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
    ) -> Union[Sequence, jax.Array]:
        """Get the initial walkers. Uses the rdm1 natural orbitals.

        Args:
            wave_data: The trial wave function data.
            n_walkers: The number of walkers.
            restricted: Whether the walkers should be restricted.

        Returns:
            walkers: The initial walkers.
                If restricted, a single jax.Array of shape (nwalkers, norb, nelec[0]).
                If unrestricted, a list of two jax.Arrays each of shape (nwalkers, norb, nelec[sigma]).
        """
        rdm1 = self.get_rdm1(wave_data)
        natorbs_up = jnp.linalg.eigh(rdm1[0])[1][:, ::-1][:, : self.nelec[0]]
        natorbs_dn = jnp.linalg.eigh(rdm1[1])[1][:, ::-1][:, : self.nelec[1]]
        if restricted:
            if self.nelec[0] == self.nelec[1]:
                det_overlap = np.linalg.det(
                    natorbs_up[:, : self.nelec[0]].T @ natorbs_dn[:, : self.nelec[1]]
                )
                if (
                    np.abs(det_overlap) > 1e-3
                ):  # probably should scale this threshold with number of electrons
                    return jnp.array([natorbs_up + 0.0j] * n_walkers)
                else:
                    overlaps = np.array(
                        [
                            natorbs_up[:, i].T @ natorbs_dn[:, i]
                            for i in range(self.nelec[0])
                        ]
                    )
                    new_vecs = natorbs_up[:, : self.nelec[0]] + np.einsum(
                        "ij,j->ij", natorbs_dn[:, : self.nelec[1]], np.sign(overlaps)
                    )
                    new_vecs = np.linalg.qr(new_vecs)[0]
                    det_overlap = np.linalg.det(
                        new_vecs.T @ natorbs_up[:, : self.nelec[0]]
                    ) * np.linalg.det(new_vecs.T @ natorbs_dn[:, : self.nelec[1]])
                    if np.abs(det_overlap) > 1e-3:
                        return jnp.array([new_vecs + 0.0j] * n_walkers)
                    else:
                        raise ValueError(
                            "Cannot find a set of RHF orbitals with good trial overlap."
                        )
            else:
                # bring the dn orbital projection onto up space to the front
                dn_proj = natorbs_up.T.conj() @ natorbs_dn
                proj_orbs = jnp.linalg.qr(dn_proj, mode="complete")[0]
                orbs = natorbs_up @ proj_orbs
                return jnp.array([orbs + 0.0j] * n_walkers)
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
        self, greens: Sequence, update_indices: jax.Array, update_constants: jnp.array
    ) -> jax.Array:
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
        ratios: jax.Array,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> jax.Array:
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

    The corresponding wave_data should contain "mo_coeff", a jax.Array of shape (norb, nelec).
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
    n_batch: int = 1

    def __post_init__(self):
        assert (
            self.nelec[0] == self.nelec[1]
        ), "RHF requires equal number of up and down electrons."

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        return jnp.linalg.det(wave_data["mo_coeff"].T.conj() @ walker) ** 2

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        return jnp.linalg.det(
            wave_data["mo_coeff"].T.conj() @ walker_up
        ) * jnp.linalg.det(wave_data["mo_coeff"].T.conj() @ walker_dn)

    @partial(jit, static_argnums=0)
    def _calc_green(self, walker: jax.Array, wave_data: dict) -> jax.Array:
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
    ) -> jax.Array:
        green_walker = self._calc_green(walker, wave_data)
        fb = 2.0 * jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"], green_walker, optimize="optimal"
        )
        return fb

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        green_walker_up = self._calc_green(walker_up, wave_data)
        green_walker_dn = self._calc_green(walker_dn, wave_data)
        green_walker = green_walker_up + green_walker_dn
        fb = jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"], green_walker, optimize="optimal"
        )
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
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
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker_up = self._calc_green(walker_up, wave_data)
        green_walker_dn = self._calc_green(walker_dn, wave_data)
        green_walker = [green_walker_up, green_walker_dn]
        ene1 = jnp.sum((green_walker[0] + green_walker[1]) * rot_h1)
        f_up = jnp.einsum(
            "gij,jk->gik", rot_chol, green_walker[0].T, optimize="optimal"
        )
        f_dn = jnp.einsum(
            "gij,jk->gik", rot_chol, green_walker[1].T, optimize="optimal"
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

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
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

    The corresponding wave_data contains "mo_coeff", a list of two jax.Arrays of shape (norb, nelec[sigma]).
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
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> complex:
        return jnp.linalg.det(
            wave_data["mo_coeff"][0].T.conj() @ walker_up
        ) * jnp.linalg.det(wave_data["mo_coeff"][1].T.conj() @ walker_dn)

    @partial(jit, static_argnums=0)
    def _calc_green(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
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
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
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
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
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

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
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
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
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
    def calc_green_diagonal_vmap(self, walkers: Sequence, wave_data: dict) -> jax.Array:
        return vmap(self.calc_green_diagonal, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def calc_overlap_ratio(
        self, green: jax.Array, update_indices: jax.Array, update_constants: jax.Array
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
        self, greens: jax.Array, update_indices: jax.Array, update_constants: jax.Array
    ) -> jax.Array:
        return vmap(self.calc_overlap_ratio, in_axes=(0, None, None))(
            greens, update_indices, update_constants
        )

    @partial(jit, static_argnums=0)
    def calc_full_green(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
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
    def calc_full_green_vmap(self, walkers: Sequence, wave_data: Any) -> jax.Array:
        return vmap(self.calc_full_green, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @partial(jit, static_argnums=0)
    def update_greens_function(
        self,
        green: jax.Array,
        ratio: float,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> jax.Array:
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

    The corresponding wave_data should contain "mo_coeff", a jax.Array of shape (2 * norb, nelec[0] + nelec[1]).
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
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        return jnp.linalg.det(
            jnp.hstack(
                [
                    wave_data["mo_coeff"][: self.norb].T.conj() @ walker_up,
                    wave_data["mo_coeff"][self.norb :].T.conj() @ walker_dn,
                ]
            )
        )

    @partial(jit, static_argnums=0)
    def _calc_green(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        overlap_mat = jnp.hstack(
            [
                wave_data["mo_coeff"][: self.norb].T.conj() @ walker_up,
                wave_data["mo_coeff"][self.norb :].T.conj() @ walker_dn,
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
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
        fb = jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"], green_walker, optimize="optimal"
        )
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
        ene1 = jnp.sum(green_walker * rot_h1)
        f = jnp.einsum("gij,jk->gik", rot_chol, green_walker.T, optimize="optimal")
        coul = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = (jnp.sum(coul * coul) - exc) / 2.0
        return ene2 + ene1 + ene0

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        dm = (
            wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]]
            @ wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T.conj()
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
            wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T.conj()
            @ walker_ghf
        )
        inv = jnp.linalg.inv(overlap_mat)
        green = (
            walker_ghf
            @ inv
            @ wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T.conj()
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
                wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T.conj()
                @ walker_ghf
            )
            @ wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T.conj()
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
    where ci_coeffs is a jax.Array of shape (ndets) and dets is a list [dets_up, dets_dn]
    with each being a jax.Array of shape (ndets, norb, nelec[sigma]), both ci_coeffs and dets
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
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap_single_det(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        trial_up: jax.Array,
        trial_dn: jax.Array,
    ) -> jax.Array:
        """Calculate the overlap with a single determinant in the NOCI trial."""
        return jnp.linalg.det(
            trial_up[:, : self.nelec[0]].T.conj() @ walker_up
        ) * jnp.linalg.det(trial_dn[:, : self.nelec[1]].T.conj() @ walker_dn)

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        ci_coeffs = wave_data["ci_coeffs_dets"][0]
        dets = wave_data["ci_coeffs_dets"][1]
        overlaps = vmap(self._calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        return jnp.sum(ci_coeffs * overlaps)

    @partial(jit, static_argnums=0)
    def _calc_green_single_det(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        trial_up: jax.Array,
        trial_dn: jax.Array,
    ) -> List:
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
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> Tuple:
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
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
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
        rot_h1_up: jax.Array,
        rot_h1_dn: jax.Array,
        rot_chol_up: jax.Array,
        rot_chol_dn: jax.Array,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        trial_up: jax.Array,
        trial_dn: jax.Array,
    ) -> jax.Array:
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
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
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
    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
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
        self, ham_data: dict, trial_up: jax.Array, trial_dn: jax.Array
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
        x_gamma: jax.Array,
        walker: jax.Array,
        chol: jax.Array,
        wave_data: dict,
    ) -> jax.Array:
        """Helper function for calculating force bias using AD,
        evaluates < psi_T | exp(x_gamma * chol) | walker > to linear order"""
        x_chol = jnp.einsum(
            "gij,g->ij", chol.reshape(-1, self.norb, self.norb), x_gamma
        )
        walker_1 = walker + x_chol.dot(walker)
        return self._calc_overlap_restricted(walker_1, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
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
        x_gamma: jax.Array,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        chol: jax.Array,
        wave_data: Any,
    ) -> jax.Array:
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
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: Any,
    ) -> jax.Array:
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
        self, x: float, h1: jax.Array, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * h1) | walker > to linear order"""
        walker_2 = walker + x * h1.dot(walker)
        return self._calc_overlap_restricted(walker_2, wave_data)

    @partial(jit, static_argnums=0)
    def _overlap_with_double_rot_restricted(
        self, x: float, chol_i: jax.Array, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
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
        walker: jax.Array,
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
        h1: jax.Array,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: Any,
    ) -> jax.Array:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * h1) | walker > to linear order"""
        walker_up_1 = walker_up + x * h1[0].dot(walker_up)
        walker_dn_1 = walker_dn + x * h1[1].dot(walker_dn)
        return self._calc_overlap(walker_up_1, walker_dn_1, wave_data)

    @partial(jit, static_argnums=0)
    def _overlap_with_double_rot(
        self,
        x: float,
        chol_i: jax.Array,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: Any,
    ) -> jax.Array:
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
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: Any,
    ) -> jax.Array:
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
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _det_overlap(
        self, green: jax.Array, cre: jax.Array, des: jax.Array
    ) -> jax.Array:
        return jnp.linalg.det(green[jnp.ix_(cre, des)])

    @partial(jit, static_argnums=0)
    def _calc_green_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        ref_det = wave_data["ref_det"][0]
        return (
            walker.dot(
                jnp.linalg.inv(walker[jnp.nonzero(ref_det, size=self.nelec[0])[0], :])
            )
        ).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
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
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
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
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
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

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
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
class CIS(wave_function_auto):
    """This class contains functions for the excited state CIS wavefunction c(ia) |ia>.
    The wave_data need to store the coefficient C(ia).
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1 = walker.shape[1], wave_data["ci1"]
        gf = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        o1 = jnp.einsum("ia,ia", ci1, gf[:, nocc:])
        return 2 * o1 * o0

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
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_green_restricted(self, walker: jax.Array) -> jax.Array:
        return (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
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
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_green(
        self, walker_up: jax.Array, walker_dn: jax.Array
    ) -> List[jax.Array]:

        green_up = (walker_up.dot(jnp.linalg.inv(walker_up[: walker_up.shape[1], :]))).T
        green_dn = (walker_dn.dot(jnp.linalg.inv(walker_dn[: walker_dn.shape[1], :]))).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
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
        o2 = 0.5 * jnp.einsum("iajb, ia, jb", ci2AA, GFA[:, noccA:], GFA[:, noccA:])
        # o2 -= 0.25 * jnp.einsum("iajb, ib, ja", ci2AA, GFA[:, noccA:], GFA[:, noccA:])

        # BB
        o2 += 0.5 * jnp.einsum("iajb, ia, jb", ci2BB, GFB[:, noccB:], GFB[:, noccB:])
        # o2 -= 0.25 * jnp.einsum("iajb, ib, ja", ci2BB, GFB[:, noccB:], GFB[:, noccB:])

        # AB
        o2 += jnp.einsum("iajb, ia, jb", ci2AB, GFA[:, noccA:], GFB[:, noccB:])

        return (1.0 + o1 + o2) * o0

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class GCISD(wave_function_auto):
    """This class contains functions for the GCISD wavefunction
    |0> + c(ia) |ia> + c(ia jb) |ia jb>
    The wave_data need to store the coefficients C(ia) and C(ia jb) and the GHF mo coefficients
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_green(self, walker: jax.Array) -> jax.Array:
        return (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        nocc, ci1, ci2 = (
            self.nelec[0] + self.nelec[1],
            wave_data["ci1"],
            wave_data["ci2"],
        )
        walker = jnp.block(
            [
                [walker_up, jnp.zeros_like(walker_dn)],
                [jnp.zeros_like(walker_up), walker_dn],
            ]
        )
        walker = wave_data["mo_coeff"].T @ walker
        GF = self._calc_green(walker)
        o0 = jnp.linalg.det(walker[: walker.shape[1], :])
        o1 = jnp.einsum("ia,ia", ci1, GF[:, nocc:])
        o2 = jnp.einsum("iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:]) - jnp.einsum(
            "iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:]
        )
        return (1.0 + o1 + o2 / 4.0) * o0

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
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_green_restricted(self, walker: jax.Array) -> jax.Array:
        return (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
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


@dataclass
class cis(wave_function):
    """A manual implementation of the excited state CIS wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1 = walker.shape[1], wave_data["ci1"]
        gf = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        o1 = jnp.einsum("ia,ia", ci1, gf[:, nocc:])
        return 2 * o1 * o0

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
        ci1 = wave_data["ci1"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]
        lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")

        ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
        ci1gp = jnp.einsum("pt,it->pi", ci1, greenp, optimize="optimal")
        gci1gp = jnp.einsum("pj,pi->ij", green, ci1gp, optimize="optimal")
        fb_1_1 = 4 * ci1g * lg
        fb_1_2 = -2 * jnp.einsum("gij,ij->g", chol, gci1gp, optimize="optimal")
        fb_1 = fb_1_1 + fb_1_2

        # overlap
        overlap_1 = 2 * ci1g
        overlap = overlap_1

        return fb_1 / overlap

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci1 = wave_data["ci1"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        # rot_chol = ham_data["rot_chol"]
        rot_chol = chol[:, : self.nelec[0], :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = jnp.einsum("pj,pj->", h1[:nocc, :], green)

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
        e1_1_1 = 4 * ci1g * hg
        gpci1 = greenp @ ci1.T
        ci1_green = gpci1 @ green
        e1_1_2 = -2 * jnp.einsum("ij,ij->", h1, ci1_green, optimize="optimal")
        e1_1 = e1_1_1 + e1_1_2

        e1 = e1_1

        # two body energy
        lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
        # lg1 = jnp.einsum("gpj,pk->gjk", rot_chol, green, optimize="optimal")
        lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
        e2_0_1 = 2 * lg @ lg
        e2_0_2 = -jnp.sum(vmap(lambda x: x * x.T)(lg1))
        e2_0 = e2_0_1 + e2_0_2

        # single excitations
        e2_1_1 = 2 * e2_0 * ci1g
        lci1g = jnp.einsum("gij,ij->g", chol, ci1_green, optimize="optimal")
        e2_1_2 = -2 * (lci1g @ lg)
        # lci1g1 = jnp.einsum("gij,jk->gik", chol, ci1_green, optimize="optimal")
        # glgpci1 = jnp.einsum(("gpi,iq->gpq"), gl, gpci1, optimize="optimal")
        ci1g1 = ci1 @ green_occ.T
        # e2_1_3 = jnp.einsum("gpq,gpq->", glgpci1, lg1, optimize="optimal")
        e2_1_3_1 = jnp.einsum("gpq,gqr,rp->", lg1, lg1, ci1g1, optimize="optimal")
        lci1g = jnp.einsum("gip,qi->gpq", ham_data["lci1"], green, optimize="optimal")
        e2_1_3_2 = -jnp.einsum("gpq,gqp->", lci1g, lg1, optimize="optimal")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        e2 = e2_1

        # overlap
        overlap_1 = 2 * ci1g  # jnp.einsum("ia,ia", ci1, green_occ)
        overlap = overlap_1
        return (e1 + e2) / overlap + e0

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["lci1"] = jnp.einsum(
            "git,pt->gip",
            ham_data["chol"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["ci1"],
            optimize="optimal",
        )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class cisd(wave_function):
    """A manual implementation of the CISD wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
        GF = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        o1 = jnp.einsum("ia,ia", ci1, GF[:, nocc:])
        o2 = 2 * jnp.einsum(
            "iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:]
        ) - jnp.einsum("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:])
        return (1.0 + 2 * o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]
        lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")

        # ref
        fb_0 = 2 * lg

        # single excitations
        ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
        ci1gp = jnp.einsum("pt,it->pi", ci1, greenp, optimize="optimal")
        gci1gp = jnp.einsum("pj,pi->ij", green, ci1gp, optimize="optimal")
        fb_1_1 = 4 * ci1g * lg
        fb_1_2 = -2 * jnp.einsum("gij,ij->g", chol, gci1gp, optimize="optimal")
        fb_1 = fb_1_1 + fb_1_2

        # double excitations
        ci2g_c = jnp.einsum("ptqu,pt->qu", ci2, green_occ)
        ci2g_e = jnp.einsum("ptqu,pu->qt", ci2, green_occ)
        cisd_green_c = (greenp @ ci2g_c.T) @ green
        cisd_green_e = (greenp @ ci2g_e.T) @ green
        cisd_green = -4 * cisd_green_c + 2 * cisd_green_e
        ci2g = 4 * ci2g_c - 2 * ci2g_e
        gci2g = jnp.einsum("qu,qu->", ci2g, green_occ, optimize="optimal")
        fb_2_1 = lg * gci2g
        fb_2_2 = jnp.einsum("gij,ij->g", chol, cisd_green, optimize="optimal")
        fb_2 = fb_2_1 + fb_2_2

        # overlap
        overlap_1 = 2 * ci1g
        overlap_2 = gci2g / 2.0
        overlap = 1.0 + overlap_1 + overlap_2

        return (fb_0 + fb_1 + fb_2) / overlap

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        # rot_chol = ham_data["rot_chol"]
        rot_chol = chol[:, : self.nelec[0], :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = jnp.einsum("pj,pj->", h1[:nocc, :], green)

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = 2 * hg

        # single excitations
        ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
        e1_1_1 = 4 * ci1g * hg
        gpci1 = greenp @ ci1.T
        ci1_green = gpci1 @ green
        e1_1_2 = -2 * jnp.einsum("ij,ij->", h1, ci1_green, optimize="optimal")
        e1_1 = e1_1_1 + e1_1_2

        # double excitations
        ci2g_c = jnp.einsum("ptqu,pt->qu", ci2, green_occ)
        ci2g_e = jnp.einsum("ptqu,pu->qt", ci2, green_occ)
        ci2_green_c = (greenp @ ci2g_c.T) @ green
        ci2_green_e = (greenp @ ci2g_e.T) @ green
        ci2_green = 2 * ci2_green_c - ci2_green_e
        ci2g = 2 * ci2g_c - ci2g_e
        gci2g = jnp.einsum("qu,qu->", ci2g, green_occ, optimize="optimal")
        e1_2_1 = 2 * hg * gci2g
        e1_2_2 = -2 * jnp.einsum("ij,ij->", h1, ci2_green, optimize="optimal")
        e1_2 = e1_2_1 + e1_2_2
        e1 = e1_0 + e1_1 + e1_2

        # two body energy
        # ref
        lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
        # lg1 = jnp.einsum("gpj,pk->gjk", rot_chol, green, optimize="optimal")
        lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
        e2_0_1 = 2 * lg @ lg
        e2_0_2 = -jnp.sum(vmap(lambda x: x * x.T)(lg1))
        e2_0 = e2_0_1 + e2_0_2

        # single excitations
        e2_1_1 = 2 * e2_0 * ci1g
        lci1g = jnp.einsum("gij,ij->g", chol, ci1_green, optimize="optimal")
        e2_1_2 = -2 * (lci1g @ lg)
        # lci1g1 = jnp.einsum("gij,jk->gik", chol, ci1_green, optimize="optimal")
        # glgpci1 = jnp.einsum(("gpi,iq->gpq"), gl, gpci1, optimize="optimal")
        ci1g1 = ci1 @ green_occ.T
        # e2_1_3 = jnp.einsum("gpq,gpq->", glgpci1, lg1, optimize="optimal")
        e2_1_3_1 = jnp.einsum("gpq,gqr,rp->", lg1, lg1, ci1g1, optimize="optimal")
        lci1g = jnp.einsum("gip,qi->gpq", ham_data["lci1"], green, optimize="optimal")
        e2_1_3_2 = -jnp.einsum("gpq,gqp->", lci1g, lg1, optimize="optimal")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g = jnp.einsum("gij,ij->g", chol, ci2_green, optimize="optimal")
        e2_2_2_1 = -lci2g @ lg

        # lci2g1 = jnp.einsum("gij,jk->gik", chol, ci2_green, optimize="optimal")
        # lci2_green = jnp.einsum("gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal")
        # e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")
        def scanned_fun(carry, x):
            chol_i, rot_chol_i = x
            gl_i = jnp.einsum("pj,ji->pi", green, chol_i, optimize="optimal")
            lci2_green_i = jnp.einsum(
                "pi,ji->pj", rot_chol_i, ci2_green, optimize="optimal"
            )
            carry[0] += 0.5 * jnp.einsum(
                "pi,pi->", gl_i, lci2_green_i, optimize="optimal"
            )
            glgp_i = jnp.einsum("pi,it->pt", gl_i, greenp, optimize="optimal").astype(
                jnp.complex64
            )
            l2ci2_1 = jnp.einsum(
                "pt,qu,ptqu->",
                glgp_i,
                glgp_i,
                ci2.astype(jnp.float32),
                optimize="optimal",
            )
            l2ci2_2 = jnp.einsum(
                "pu,qt,ptqu->",
                glgp_i,
                glgp_i,
                ci2.astype(jnp.float32),
                optimize="optimal",
            )
            carry[1] += 2 * l2ci2_1 - l2ci2_2
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0, 0.0], (chol, rot_chol))
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        # glgp = jnp.einsum("pi,gij,jt->gpt", green, chol, greenp, optimize="optimal")
        # l2 = jnp.einsum("gpt,gqu->ptqu", glgp, glgp, optimize="optimal")
        # l2ci2_1 = jnp.einsum("ptqu,ptqu->", l2, ci2, optimize="optimal")
        # l2ci2_2 = jnp.einsum("puqt,ptqu->", l2, ci2, optimize="optimal")

        # glgp = jnp.einsum("gpi,it->gpt", gl, greenp, optimize="optimal")
        # l2ci2_1 = jnp.einsum("gpt,gqu,ptqu->g", glgp, glgp, ci2, optimize="optimal")
        # l2ci2_2 = jnp.einsum("gpu,gqt,ptqu->g", glgp, glgp, ci2, optimize="optimal")
        # e2_2_3 = 2 * l2ci2_1.sum() - l2ci2_2.sum()
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        e2 = e2_0 + e2_1 + e2_2

        # overlap
        overlap_1 = 2 * ci1g  # jnp.einsum("ia,ia", ci1, green_occ)
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2
        return (e1 + e2) / overlap + e0

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["lci1"] = jnp.einsum(
            "git,pt->gip",
            ham_data["chol"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["ci1"],
            optimize="optimal",
        )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class cisd_faster(cisd):
    """A manual implementation of the CISD wave function.

    Faster than cisd, but the energy function builds some large intermediates, O(XMN),
    so memory usage is high.
    """

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = jnp.einsum("pj,pj->", h1[:nocc, :], green)

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = 2 * hg

        # single excitations
        ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
        e1_1_1 = 4 * ci1g * hg
        gpci1 = greenp @ ci1.T
        ci1_green = gpci1 @ green
        e1_1_2 = -2 * jnp.einsum("ij,ij->", h1, ci1_green, optimize="optimal")
        e1_1 = e1_1_1 + e1_1_2

        # double excitations
        ci2g_c = jnp.einsum("ptqu,pt->qu", ci2, green_occ)
        ci2g_e = jnp.einsum("ptqu,pu->qt", ci2, green_occ)
        ci2_green_c = (greenp @ ci2g_c.T) @ green
        ci2_green_e = (greenp @ ci2g_e.T) @ green
        ci2_green = 2 * ci2_green_c - ci2_green_e
        ci2g = 2 * ci2g_c - ci2g_e
        gci2g = jnp.einsum("qu,qu->", ci2g, green_occ, optimize="optimal")
        e1_2_1 = 2 * hg * gci2g
        e1_2_2 = -2 * jnp.einsum("ij,ij->", h1, ci2_green, optimize="optimal")
        e1_2 = e1_2_1 + e1_2_2
        e1 = e1_0 + e1_1 + e1_2

        # two body energy
        # ref
        lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
        lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
        e2_0_1 = 2 * lg @ lg
        e2_0_2 = -jnp.sum(vmap(lambda x: x * x.T)(lg1))
        e2_0 = e2_0_1 + e2_0_2

        # single excitations
        e2_1_1 = 2 * e2_0 * ci1g
        lci1g = jnp.einsum("gij,ij->g", chol, ci1_green, optimize="optimal")
        e2_1_2 = -2 * (lci1g @ lg)
        gl = jnp.einsum("pj,gji->gpi", green, chol, optimize="optimal")
        ci1g1 = ci1 @ green_occ.T
        e2_1_3_1 = jnp.einsum("gpq,gqr,rp->", lg1, lg1, ci1g1, optimize="optimal")
        lci1g = jnp.einsum("gip,qi->gpq", ham_data["lci1"], green, optimize="optimal")
        e2_1_3_2 = -jnp.einsum("gpq,gqp->", lci1g, lg1, optimize="optimal")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g = jnp.einsum("gij,ij->g", chol, ci2_green, optimize="optimal")
        e2_2_2_1 = -lci2g @ lg
        lci2_green = jnp.einsum("gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal")
        e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")
        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        glgp = jnp.einsum("gpi,it->gpt", gl, greenp, optimize="optimal").astype(
            jnp.complex64
        )
        l2ci2_1 = jnp.einsum(
            "gpt,gqu,ptqu->g", glgp, glgp, ci2.astype(jnp.float32), optimize="optimal"
        )
        l2ci2_2 = jnp.einsum(
            "gpu,gqt,ptqu->g", glgp, glgp, ci2.astype(jnp.float32), optimize="optimal"
        )
        e2_2_3 = 2 * l2ci2_1.sum() - l2ci2_2.sum()
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3
        e2 = e2_0 + e2_1 + e2_2

        # overlap
        overlap_1 = 2 * ci1g
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2
        return (e1 + e2) / overlap + e0

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class ucisd(wave_function):
    """A manual implementation of the UCISD wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1
    mixed_real_dtype: DTypeLike = jnp.float32
    mixed_complex_dtype: DTypeLike = jnp.complex64

    @partial(jit, static_argnums=0)
    def _calc_overlap(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        noccA, ci1A, ci2AA = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        noccB, ci1B, ci2BB = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2AB = wave_data["ci2AB"]
        _, moB = wave_data["mo_coeff"][0], wave_data["mo_coeff"][1]
        walker_dn_B = moB.T.dot(walker_dn[:, :noccB])
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[: walker_up.shape[1], :]))).T
        green_b = (
            walker_dn_B.dot(jnp.linalg.inv(walker_dn_B[: walker_dn_B.shape[1], :]))
        ).T
        green_a, green_b = green_a[:, noccA:], green_b[:, noccB:]
        o0 = jnp.linalg.det(walker_up[:noccA, :]) * jnp.linalg.det(
            walker_dn_B[:noccB, :]
        )
        o1 = jnp.einsum("ia,ia", ci1A, green_a) + jnp.einsum("ia,ia", ci1B, green_b)
        o2 = (
            0.5 * jnp.einsum("iajb, ia, jb", ci2AA, green_a, green_a)
            + 0.5 * jnp.einsum("iajb, ia, jb", ci2BB, green_b, green_b)
            + jnp.einsum("iajb, ia, jb", ci2AB, green_a, green_b)
        )
        return (1.0 + o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _calc_force_bias(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
        nocc_a, ci1_a, ci2_aa = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        nocc_b, ci1_b, ci2_bb = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2_ab = wave_data["ci2AB"]
        walker_dn_b = wave_data["mo_coeff"][1].T.dot(walker_dn[:, :nocc_b])
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T
        green_b = (walker_dn_b.dot(jnp.linalg.inv(walker_dn_b[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:].copy()
        green_occ_b = green_b[:, nocc_b:].copy()
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(self.norb - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(self.norb - nocc_b)))

        chol_a = ham_data["chol"].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol_b"].reshape(-1, self.norb, self.norb)
        rot_chol_a = chol_a[:, : self.nelec[0], :]
        rot_chol_b = chol_b[:, : self.nelec[1], :]
        lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a, optimize="optimal")
        lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b, optimize="optimal")
        lg = lg_a + lg_b

        # ref
        fb_0 = lg_a + lg_b

        # single excitations
        ci1g_a = jnp.einsum("pt,pt->", ci1_a, green_occ_a, optimize="optimal")
        ci1g_b = jnp.einsum("pt,pt->", ci1_b, green_occ_b, optimize="optimal")
        ci1g = ci1g_a + ci1g_b
        fb_1_1 = ci1g * lg
        ci1gp_a = jnp.einsum("pt,it->pi", ci1_a, greenp_a, optimize="optimal")
        ci1gp_b = jnp.einsum("pt,it->pi", ci1_b, greenp_b, optimize="optimal")
        gci1gp_a = jnp.einsum("pj,pi->ij", green_a, ci1gp_a, optimize="optimal")
        gci1gp_b = jnp.einsum("pj,pi->ij", green_b, ci1gp_b, optimize="optimal")
        fb_1_2 = -jnp.einsum(
            "gij,ij->g", chol_a, gci1gp_a, optimize="optimal"
        ) - jnp.einsum("gij,ij->g", chol_b, gci1gp_b, optimize="optimal")
        fb_1 = fb_1_1 + fb_1_2

        # double excitations
        ci2g_a = jnp.einsum("ptqu,pt->qu", ci2_aa, green_occ_a)
        ci2g_b = jnp.einsum("ptqu,pt->qu", ci2_bb, green_occ_b)
        ci2g_ab_a = jnp.einsum("ptqu,qu->pt", ci2_ab, green_occ_b)
        ci2g_ab_b = jnp.einsum("ptqu,pt->qu", ci2_ab, green_occ_a)
        gci2g_a = 0.5 * jnp.einsum("qu,qu->", ci2g_a, green_occ_a, optimize="optimal")
        gci2g_b = 0.5 * jnp.einsum("qu,qu->", ci2g_b, green_occ_b, optimize="optimal")
        gci2g_ab = jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a, optimize="optimal")
        gci2g = gci2g_a + gci2g_b + gci2g_ab
        fb_2_1 = lg * gci2g
        ci2_green_a = (greenp_a @ (ci2g_a + ci2g_ab_a).T) @ green_a
        ci2_green_b = (greenp_b @ (ci2g_b + ci2g_ab_b).T) @ green_b
        fb_2_2_a = -jnp.einsum("gij,ij->g", chol_a, ci2_green_a, optimize="optimal")
        fb_2_2_b = -jnp.einsum("gij,ij->g", chol_b, ci2_green_b, optimize="optimal")
        fb_2_2 = fb_2_2_a + fb_2_2_b
        fb_2 = fb_2_1 + fb_2_2

        # overlap
        overlap_1 = ci1g
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2

        return (fb_0 + fb_1 + fb_2) / overlap

    @partial(jit, static_argnums=0)
    def _calc_energy(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        nocc_a, ci1_a, ci2_aa = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        nocc_b, ci1_b, ci2_bb = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2_ab = wave_data["ci2AB"]
        walker_dn_b = wave_data["mo_coeff"][1].T.dot(walker_dn[:, :nocc_b])
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T
        green_b = (walker_dn_b.dot(jnp.linalg.inv(walker_dn_b[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:].copy()
        green_occ_b = green_b[:, nocc_b:].copy()
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(self.norb - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(self.norb - nocc_b)))

        chol_a = ham_data["chol"].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol_b"].reshape(-1, self.norb, self.norb)
        rot_chol_a = chol_a[:, :nocc_a, :]
        rot_chol_b = chol_b[:, :nocc_b, :]
        h1_a = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        h1_b = ham_data["h1_b"]
        hg_a = jnp.einsum("pj,pj->", h1_a[:nocc_a, :], green_a)
        hg_b = jnp.einsum("pj,pj->", h1_b[:nocc_b, :], green_b)
        hg = hg_a + hg_b

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = hg

        # single excitations
        ci1g_a = jnp.einsum("pt,pt->", ci1_a, green_occ_a, optimize="optimal")
        ci1g_b = jnp.einsum("pt,pt->", ci1_b, green_occ_b, optimize="optimal")
        ci1g = ci1g_a + ci1g_b
        e1_1_1 = ci1g * hg
        gpci1_a = greenp_a @ ci1_a.T
        gpci1_b = greenp_b @ ci1_b.T
        ci1_green_a = gpci1_a @ green_a
        ci1_green_b = gpci1_b @ green_b
        e1_1_2 = -(
            jnp.einsum("ij,ij->", h1_a, ci1_green_a, optimize="optimal")
            + jnp.einsum("ij,ij->", h1_b, ci1_green_b, optimize="optimal")
        )
        e1_1 = e1_1_1 + e1_1_2

        # double excitations
        ci2g_a = jnp.einsum("ptqu,pt->qu", ci2_aa, green_occ_a) / 4
        ci2g_b = jnp.einsum("ptqu,pt->qu", ci2_bb, green_occ_b) / 4
        ci2g_ab_a = jnp.einsum("ptqu,qu->pt", ci2_ab, green_occ_b)
        ci2g_ab_b = jnp.einsum("ptqu,pt->qu", ci2_ab, green_occ_a)
        gci2g_a = jnp.einsum("qu,qu->", ci2g_a, green_occ_a, optimize="optimal")
        gci2g_b = jnp.einsum("qu,qu->", ci2g_b, green_occ_b, optimize="optimal")
        gci2g_ab = jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a, optimize="optimal")
        gci2g = 2 * (gci2g_a + gci2g_b) + gci2g_ab
        e1_2_1 = hg * gci2g
        ci2_green_a = (greenp_a @ ci2g_a.T) @ green_a
        ci2_green_ab_a = (greenp_a @ ci2g_ab_a.T) @ green_a
        ci2_green_b = (greenp_b @ ci2g_b.T) @ green_b
        ci2_green_ab_b = (greenp_b @ ci2g_ab_b.T) @ green_b
        e1_2_2_a = -jnp.einsum(
            "ij,ij->", h1_a, 4 * ci2_green_a + ci2_green_ab_a, optimize="optimal"
        )
        e1_2_2_b = -jnp.einsum(
            "ij,ij->", h1_b, 4 * ci2_green_b + ci2_green_ab_b, optimize="optimal"
        )
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2

        e1 = e1_0 + e1_1 + e1_2

        # two body energy
        # ref
        lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a, optimize="optimal")
        lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b, optimize="optimal")
        e2_0_1 = ((lg_a + lg_b) @ (lg_a + lg_b)) / 2.0
        lg1_a = jnp.einsum("gpj,qj->gpq", rot_chol_a, green_a, optimize="optimal")
        lg1_b = jnp.einsum("gpj,qj->gpq", rot_chol_b, green_b, optimize="optimal")
        e2_0_2 = (
            -(
                jnp.sum(vmap(lambda x: x * x.T)(lg1_a))
                + jnp.sum(vmap(lambda x: x * x.T)(lg1_b))
            )
            / 2.0
        )
        e2_0 = e2_0_1 + e2_0_2

        # single excitations
        e2_1_1 = e2_0 * ci1g
        lci1g_a = jnp.einsum("gij,ij->g", chol_a, ci1_green_a, optimize="optimal")
        lci1g_b = jnp.einsum("gij,ij->g", chol_b, ci1_green_b, optimize="optimal")
        e2_1_2 = -((lci1g_a + lci1g_b) @ (lg_a + lg_b))
        ci1g1_a = ci1_a @ green_occ_a.T
        ci1g1_b = ci1_b @ green_occ_b.T
        e2_1_3_1 = jnp.einsum(
            "gpq,gqr,rp->", lg1_a, lg1_a, ci1g1_a, optimize="optimal"
        ) + jnp.einsum("gpq,gqr,rp->", lg1_b, lg1_b, ci1g1_b, optimize="optimal")
        lci1g_a = jnp.einsum(
            "gip,qi->gpq", ham_data["lci1_a"], green_a, optimize="optimal"
        )
        lci1g_b = jnp.einsum(
            "gip,qi->gpq", ham_data["lci1_b"], green_b, optimize="optimal"
        )
        e2_1_3_2 = -jnp.einsum(
            "gpq,gqp->", lci1g_a, lg1_a, optimize="optimal"
        ) - jnp.einsum("gpq,gqp->", lci1g_b, lg1_b, optimize="optimal")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + e2_1_2 + e2_1_3

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g_a = jnp.einsum(
            "gij,ij->g",
            chol_a,
            8 * ci2_green_a + 2 * ci2_green_ab_a,
            optimize="optimal",
        )
        lci2g_b = jnp.einsum(
            "gij,ij->g",
            chol_b,
            8 * ci2_green_b + 2 * ci2_green_ab_b,
            optimize="optimal",
        )
        e2_2_2_1 = -((lci2g_a + lci2g_b) @ (lg_a + lg_b)) / 2.0

        def scanned_fun(carry, x):
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
            gl_a_i = jnp.einsum("pj,ji->pi", green_a, chol_a_i, optimize="optimal")
            gl_b_i = jnp.einsum("pj,ji->pi", green_b, chol_b_i, optimize="optimal")
            lci2_green_a_i = jnp.einsum(
                "pi,ji->pj",
                rot_chol_a_i,
                8 * ci2_green_a + 2 * ci2_green_ab_a,
                optimize="optimal",
            )
            lci2_green_b_i = jnp.einsum(
                "pi,ji->pj",
                rot_chol_b_i,
                8 * ci2_green_b + 2 * ci2_green_ab_b,
                optimize="optimal",
            )
            carry[0] += 0.5 * (
                jnp.einsum("pi,pi->", gl_a_i, lci2_green_a_i, optimize="optimal")
                + jnp.einsum("pi,pi->", gl_b_i, lci2_green_b_i, optimize="optimal")
            )
            glgp_a_i = jnp.einsum(
                "pi,it->pt", gl_a_i, greenp_a, optimize="optimal"
            ).astype(self.mixed_complex_dtype)
            glgp_b_i = jnp.einsum(
                "pi,it->pt", gl_b_i, greenp_b, optimize="optimal"
            ).astype(self.mixed_complex_dtype)
            l2ci2_a = 0.5 * jnp.einsum(
                "pt,qu,ptqu->",
                glgp_a_i,
                glgp_a_i,
                ci2_aa.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2ci2_b = 0.5 * jnp.einsum(
                "pt,qu,ptqu->",
                glgp_b_i,
                glgp_b_i,
                ci2_bb.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2ci2_ab = jnp.einsum(
                "pt,qu,ptqu->",
                glgp_a_i,
                glgp_b_i,
                ci2_ab.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            carry[1] += l2ci2_a + l2ci2_b + l2ci2_ab
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(
            scanned_fun, [0.0, 0.0], (chol_a, rot_chol_a, chol_b, rot_chol_b)
        )
        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        e2 = e2_0 + e2_1 + e2_2

        # overlap
        overlap_1 = ci1g  # jnp.einsum("ia,ia", ci1, green_occ)
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2
        return (e1 + e2) / overlap + e0

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        mo_coeff_b = wave_data["mo_coeff"][1]
        ham_data["h1_b"] = mo_coeff_b.T @ ham_data["h1"][1] @ mo_coeff_b
        ham_data["chol_b"] = jnp.einsum(
            "pi,gij,jq->gpq",
            mo_coeff_b.T,
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            mo_coeff_b,
        )
        ham_data["lci1_a"] = jnp.einsum(
            "git,pt->gip",
            ham_data["chol"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["ci1A"],
            optimize="optimal",
        )
        ham_data["lci1_b"] = jnp.einsum(
            "git,pt->gip",
            ham_data["chol_b"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[1] :],
            wave_data["ci1B"],
            optimize="optimal",
        )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class cisd_eom_t_auto(wave_function_auto):
    """(r1 + r2 + r1 c1 + r1 c2 + r2 c1) |0>, ad implementation"""

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1e-4
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2, r1, r2 = (
            walker.shape[1],
            wave_data["ci1"],
            wave_data["ci2"],
            wave_data["r1"],
            wave_data["r2"],
        )
        green = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        green_occ = green[:, nocc:]
        # r1 terms
        # r1 1
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        r1_1 = r1g
        # r1 c1
        c1g = 2 * jnp.einsum("pt,pt", ci1, green_occ)
        r1_c1_1 = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = ci1 @ green_occ.T
        r1_c1_2 = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1_c1 = r1_c1_1 + r1_c1_2
        # r1 c2
        c2g2 = 2 * jnp.einsum("ptqu,pt,qu", ci2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", ci2, green_occ, green_occ
        )
        r1_c2_1 = r1g * c2g2
        c2g_1 = jnp.einsum("ptqu,qu->pt", ci2, green_occ)
        c2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", ci2, green_occ)
        gc2_g = (c2g_1 - c2g_2) @ green_occ.T
        r1_c2_2 = -4 * jnp.einsum("pq,qp", r1_g, gc2_g)
        r1_c2 = r1_c2_1 + r1_c2_2

        # r2 terms
        # r2 1
        r2g2 = 2 * jnp.einsum("ptqu,pt,qu", r2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", r2, green_occ, green_occ
        )
        r2_1 = r2g2
        # r2 c1
        r2_c1_1 = r2g2 * c1g
        r2g_1 = jnp.einsum("ptqu,qu->pt", r2, green_occ)
        r2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", r2, green_occ)
        gr2_g = (r2g_1 - r2g_2) @ green_occ.T
        r2_c1_2 = -4 * jnp.einsum("pq,qp", gr2_g, c1_g)
        r2_c1 = r2_c1_1 + r2_c1_2

        # # r2 c2
        # r2_c2_1 = r2g2 * c2g2
        # r2_c2_2 = -8 * jnp.einsum("pq,qp", gr2_g, gc2_g)
        # r2_int = jnp.einsum("ptqu,rt,su->prqs", r2, green_occ, green_occ)
        # c2_int = jnp.einsum("ptqu,rt,su->prqs", ci2, green_occ, green_occ)
        # r2_c2_3 = 2 * jnp.einsum("prqs,rpsq", r2_int, c2_int) - jnp.einsum(
        #     "prqs,rqsp", r2_int, c2_int
        # )
        # r2_c2 = r2_c2_1 + r2_c2_2 + r2_c2_3

        overlap_0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        return (r1_1 + r1_c1 + r1_c2 + r2_1 + r2_c1) * overlap_0
        # return (r1_1 + r1_c1 + r1_c2 + r2_1 + r2_c1 + r2_c2) * overlap_0

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class cisd_eom_auto(wave_function_auto):
    """(r1 + r2) (1 + c1 + c2) |0>, ad implementation"""

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1e-4
    n_batch: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2, r1, r2 = (
            walker.shape[1],
            wave_data["ci1"],
            wave_data["ci2"],
            wave_data["r1"],
            wave_data["r2"],
        )
        green = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        green_occ = green[:, nocc:]
        # r1 terms
        # r1 1
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        r1_1 = r1g
        # r1 c1
        c1g = 2 * jnp.einsum("pt,pt", ci1, green_occ)
        r1_c1_1 = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = ci1 @ green_occ.T
        r1_c1_2 = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1_c1 = r1_c1_1 + r1_c1_2
        # r1 c2
        c2g2 = 2 * jnp.einsum("ptqu,pt,qu", ci2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", ci2, green_occ, green_occ
        )
        r1_c2_1 = r1g * c2g2
        c2g_1 = jnp.einsum("ptqu,qu->pt", ci2, green_occ)
        c2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", ci2, green_occ)
        gc2_g = (c2g_1 - c2g_2) @ green_occ.T
        r1_c2_2 = -4 * jnp.einsum("pq,qp", r1_g, gc2_g)
        r1_c2 = r1_c2_1 + r1_c2_2

        # r2 terms
        # r2 1
        r2g2 = 2 * jnp.einsum("ptqu,pt,qu", r2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", r2, green_occ, green_occ
        )
        r2_1 = r2g2
        # r2 c1
        r2_c1_1 = r2g2 * c1g
        r2g_1 = jnp.einsum("ptqu,qu->pt", r2, green_occ)
        r2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", r2, green_occ)
        gr2_g = (r2g_1 - r2g_2) @ green_occ.T
        r2_c1_2 = -4 * jnp.einsum("pq,qp", gr2_g, c1_g)
        r2_c1 = r2_c1_1 + r2_c1_2

        # r2 c2
        r2_c2_1 = r2g2 * c2g2
        r2_c2_2 = -8 * jnp.einsum("pq,qp", gr2_g, gc2_g)
        r2_int = jnp.einsum("ptqu,rt,su->prqs", r2, green_occ, green_occ)
        c2_int = jnp.einsum("ptqu,rt,su->prqs", ci2, green_occ, green_occ)
        r2_c2_3 = 2 * jnp.einsum("prqs,rpsq", r2_int, c2_int) - jnp.einsum(
            "prqs,rqsp", r2_int, c2_int
        )
        r2_c2 = r2_c2_1 + r2_c2_2 + r2_c2_3

        overlap_0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        return (r1_1 + r1_c1 + r1_c2 + r2_1 + r2_c1 + r2_c2) * overlap_0

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class cisd_eom_t(wave_function):
    """(r1 + r2 + r1 c1 + r1 c2 + r2 c1) |0>, manual implementation"""

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1
    mixed_real_dtype: DTypeLike = jnp.float32
    mixed_complex_dtype: DTypeLike = jnp.complex64

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2, r1, r2 = (
            walker.shape[1],
            wave_data["ci1"],
            wave_data["ci2"],
            wave_data["r1"],
            wave_data["r2"],
        )
        green = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        green_occ = green[:, nocc:]
        # r1 terms
        # r1 1
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        r1_1 = r1g
        # r1 c1
        c1g = 2 * jnp.einsum("pt,pt", ci1, green_occ)
        r1_c1_1 = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = ci1 @ green_occ.T
        r1_c1_2 = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1_c1 = r1_c1_1 + r1_c1_2
        # r1 c2
        c2g2 = 2 * jnp.einsum("ptqu,pt,qu", ci2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", ci2, green_occ, green_occ
        )
        r1_c2_1 = r1g * c2g2
        c2g_1 = jnp.einsum("ptqu,qu->pt", ci2, green_occ)
        c2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", ci2, green_occ)
        gc2_g = (c2g_1 - c2g_2) @ green_occ.T
        r1_c2_2 = -4 * jnp.einsum("pq,qp", r1_g, gc2_g)
        r1_c2 = r1_c2_1 + r1_c2_2

        # r2 terms
        # r2 1
        r2g2 = 2 * jnp.einsum("ptqu,pt,qu", r2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", r2, green_occ, green_occ
        )
        r2_1 = r2g2
        # r2 c1
        r2_c1_1 = r2g2 * c1g
        r2g_1 = jnp.einsum("ptqu,qu->pt", r2, green_occ)
        r2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", r2, green_occ)
        gr2_g = (r2g_1 - r2g_2) @ green_occ.T
        r2_c1_2 = -4 * jnp.einsum("pq,qp", gr2_g, c1_g)
        r2_c1 = r2_c1_1 + r2_c1_2

        overlap_0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        return (r1_1 + r1_c1 + r1_c2 + r2_1 + r2_c1) * overlap_0
        # return (r1_1 + r1_c1 + r1_c2 + r2_1 + r2_c1 + r2_c2) * overlap_0

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        c1, c2, r1, r2 = (
            jnp.array(wave_data["ci1"]),
            jnp.array(wave_data["ci2"]),
            jnp.array(wave_data["r1"]),
            jnp.array(wave_data["r2"]),
        )
        nocc = self.nelec[0]
        nvirt = self.norb - nocc
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(nvirt)))
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]

        # r1
        # 2: spin
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        # 2: spin
        lg = 2.0 * jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
        fb_r1_1 = r1g * lg
        g_r1_gp = green.T @ (r1 @ greenp.T)
        # 2: spin
        fb_r1_2 = -2 * jnp.einsum("gij,ji", chol, g_r1_gp)
        fb_r1 = fb_r1_1 + fb_r1_2

        # r1 c1
        # 2: spin
        c1g = 2 * jnp.einsum("pt,pt", c1, green_occ)
        r1c1_c = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = c1 @ green_occ.T
        # 2: spin
        r1c1_e = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1c1 = r1c1_c + r1c1_e
        fb_r1c1_1 = r1c1 * lg
        r1_c1 = r1 * c1g + r1g * c1 - r1_g @ c1 - c1_g @ r1
        g_r1_c1_gp = green.T @ (r1_c1 @ greenp.T)
        # 2: spin
        fb_r1c1_2 = -2 * jnp.einsum("gij,ji", chol, g_r1_c1_gp)
        fb_r1c1 = fb_r1c1_1 + fb_r1c1_2

        # r2
        # 2: spin, 0.5: r2, 2: permutation
        r2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", r2, green_occ)
        # 0.5: r2, 2: permutation
        r2g_e = jnp.einsum("ptqu,pu->qt", r2, green_occ)
        r2g = r2g_c - r2g_e
        # 2: spin, 0.5: no permuation
        r2g2 = jnp.einsum("qu,qu", r2g, green_occ, optimize="optimal")
        fb_r2_1 = lg * r2g2
        g_r2g_gp = green.T @ (r2g @ greenp.T)
        # 2: spin
        fb_r2_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r2g_gp)
        fb_r2 = fb_r2_1 + fb_r2_2

        # r2 c1
        r2c1_c = r2g2 * c1g
        r2g_g = r2g @ green_occ.T
        # 2: spin
        r2c1_e = -2.0 * jnp.einsum("pq,qp", r2g_g, c1_g, optimize="optimal")
        r2c1 = r2c1_c + r2c1_e
        fb_r2c1_1 = lg * r2c1
        r2_c1 = r2g * c1g + r2g2 * c1 - r2g_g @ c1 - c1_g @ r2g
        g_c1_g = green_occ.T @ c1_g
        # 2: spin, 2: permutation, 0.5: r2
        r2_c1 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_c1_g, r2, optimize="optimal")
        # 2: permutation, 0.5: r2
        r2_c1 += jnp.einsum("tq,ptqu->pu", g_c1_g, r2, optimize="optimal")
        g_r2_c1_gp = green.T @ (r2_c1 @ greenp.T)
        # 2: spin
        fb_r2c1_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r2_c1_gp)
        fb_r2c1 = fb_r2c1_1 + fb_r2c1_2

        # r1 c2
        # 2: spin, 0.5: c2, 2: permutation
        c2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", c2, green_occ)
        # 0.5: c2, 2: permutation
        c2g_e = jnp.einsum("ptqu,pu->qt", c2, green_occ)
        c2g = c2g_c - c2g_e
        # 2: spin, 0.5: no permuation
        c2g2 = jnp.einsum("qu,qu", c2g, green_occ, optimize="optimal")
        r1c2_c = r1g * c2g2
        c2g_g = c2g @ green_occ.T
        # 2: spin
        r1c2_e = -2.0 * jnp.einsum("pq,qp", r1_g, c2g_g, optimize="optimal")
        r1c2 = r1c2_c + r1c2_e
        fb_r1c2_1 = lg * r1c2
        r1_c2 = r1 * c2g2 + r1g * c2g - r1_g @ c2g - c2g_g @ r1
        g_r1_g = green_occ.T @ r1_g
        # 2: spin, 2: permutation, 0.5: c2
        r1_c2 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_r1_g, c2, optimize="optimal")
        # 2: permutation, 0.5: c2
        r1_c2 += jnp.einsum("tq,ptqu->pu", g_r1_g, c2, optimize="optimal")
        g_r1_c2_gp = green.T @ (r1_c2 @ greenp.T)
        # 2: spin
        fb_r1c2_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r1_c2_gp)
        fb_r1c2 = fb_r1c2_1 + fb_r1c2_2

        overlap = r1g + r1c1 + r2g2 + r2c1 + r1c2
        fb = (fb_r1 + fb_r1c1 + fb_r2 + fb_r2c1 + fb_r1c2) / overlap
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        c1, c2, r1, r2 = (
            jnp.array(wave_data["ci1"]),
            jnp.array(wave_data["ci2"]),
            jnp.array(wave_data["r1"]),
            jnp.array(wave_data["r2"]),
        )
        nocc = self.nelec[0]
        nvirt = self.norb - nocc
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(nvirt)))
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # r1
        # 2: spin
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        # 2: spin
        h1g = 2 * jnp.einsum("pt,pt", h1[:nocc, :], green)
        e1_r1_1 = r1g * h1g
        gp_h_g = greenp.T @ (h1 @ green.T)
        # 2: spin
        e1_r1_2 = -2 * jnp.einsum("tp,pt", gp_h_g, r1)
        e1_r1 = e1_r1_1 + e1_r1_2

        # r1 c1
        # 2: spin
        c1g = 2 * jnp.einsum("pt,pt", c1, green_occ)
        r1c1_c = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = c1 @ green_occ.T
        # 2: spin
        r1c1_e = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1c1 = r1c1_c + r1c1_e
        e1_r1c1_1 = r1c1 * h1g

        r1_c1 = r1 * c1g + r1g * c1 - r1_g @ c1 - c1_g @ r1
        # 2: spin
        e1_r1c1_2 = -2 * jnp.einsum("tp,pt", gp_h_g, r1_c1)
        e1_r1c1 = e1_r1c1_1 + e1_r1c1_2

        # r2
        # 2: spin, 0.5: r2, 2: permutation
        r2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", r2, green_occ)
        # 0.5: r2, 2: permutation
        r2g_e = jnp.einsum("ptqu,pu->qt", r2, green_occ)
        r2g = r2g_c - r2g_e
        # 2: spin, 0.5: no permuation
        r2g2 = jnp.einsum("qu,qu", r2g, green_occ, optimize="optimal")
        e1_r2_1 = h1g * r2g2
        # 2: spin
        e1_r2_2 = -2.0 * jnp.einsum("pt,tp", r2g, gp_h_g, optimize="optimal")
        e1_r2 = e1_r2_1 + e1_r2_2

        # r2 c1
        r2c1_c = r2g2 * c1g
        r2g_g = r2g @ green_occ.T
        # 2: spin
        r2c1_e = -2.0 * jnp.einsum("pq,qp", r2g_g, c1_g, optimize="optimal")
        r2c1 = r2c1_c + r2c1_e
        e1_r2c1_1 = h1g * r2c1
        r2_c1 = r2g * c1g + r2g2 * c1 - r2g_g @ c1 - c1_g @ r2g
        g_c1_g = green_occ.T @ c1_g
        # 2: spin, 2: permutation, 0.5: r2
        r2_c1 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_c1_g, r2, optimize="optimal")
        # 2: permutation, 0.5: r2
        r2_c1 += jnp.einsum("tq,ptqu->pu", g_c1_g, r2, optimize="optimal")
        # 2: spin
        e1_r2c1_2 = -2.0 * jnp.einsum("tp,pt", gp_h_g, r2_c1, optimize="optimal")
        e1_r2c1 = e1_r2c1_1 + e1_r2c1_2

        # r1 c2
        # 2: spin, 0.5: c2, 2: permutation
        c2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", c2, green_occ)
        # 0.5: c2, 2: permutation
        c2g_e = jnp.einsum("ptqu,pu->qt", c2, green_occ)
        c2g = c2g_c - c2g_e
        # 2: spin, 0.5: no permuation
        c2g2 = jnp.einsum("qu,qu", c2g, green_occ, optimize="optimal")
        r1c2_c = r1g * c2g2
        c2g_g = c2g @ green_occ.T
        # 2: spin
        r1c2_e = -2.0 * jnp.einsum("pq,qp", r1_g, c2g_g, optimize="optimal")
        r1c2 = r1c2_c + r1c2_e
        e1_r1c2_1 = h1g * r1c2
        r1_c2 = r1 * c2g2 + r1g * c2g - r1_g @ c2g - c2g_g @ r1
        g_r1_g = green_occ.T @ r1_g
        # 2: spin, 2: permutation, 0.5: c2
        r1_c2 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_r1_g, c2, optimize="optimal")
        # 2: permutation, 0.5: c2
        r1_c2 += jnp.einsum("tq,ptqu->pu", g_r1_g, c2, optimize="optimal")
        # 2: spin
        e1_r1c2_2 = -2.0 * jnp.einsum("tp,pt", gp_h_g, r1_c2, optimize="optimal")
        e1_r1c2 = e1_r1c2_1 + e1_r1c2_2

        e1 = e1_r1 + e1_r1c1 + e1_r2 + e1_r2c1 + e1_r1c2  # + e1_r2c2

        # 2 body energy
        # 2: spin
        lg = 2.0 * jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
        l_g = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
        # 0.5: coulomb
        l2g2_c = 0.5 * (lg @ lg)
        l2g2_e = -jnp.sum(vmap(lambda x: x * x.T)(l_g))
        l2g2 = l2g2_c + l2g2_e

        # doing this first to build intermediates
        # r2
        e2_r2_1 = l2g2 * r2g2

        # carry: [e2_r2_, e2_c2_, e2_r1c1_, e2_r2c1_, e2_r1c2_, l2g]
        def loop_over_chol(carry, x):
            chol_i, lg_i, l_g_i = x
            # build intermediate
            gp_l_g_i = greenp.T @ (chol_i @ green.T)
            # 0.5: coulomb, 2: permutation
            l2g_i_c = gp_l_g_i * lg_i
            l2g_i_e = gp_l_g_i @ l_g_i
            l2g_i = l2g_i_c - l2g_i_e
            carry[5] += l2g_i

            gp_l_g_i = gp_l_g_i.astype(self.mixed_complex_dtype)
            # evaluate energy
            # r2
            # 4: spin, 2: permutation, 0.5: r2, 0.5: coulomb
            l2r2_c = 2.0 * jnp.einsum(
                "tp,uq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 2: spin, 2: permutation, 0.5: r2, 0.5: coulomb
            l2r2_e = jnp.einsum(
                "up,tq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2r2 = l2r2_c - l2r2_e
            carry[0] += l2r2

            # c2
            # 4: spin, 2: permutation, 0.5: c2, 0.5: coulomb
            l2c2_c = 2.0 * jnp.einsum(
                "tp,uq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 2: spin, 2: permutation, 0.5: c2, 0.5: coulomb
            l2c2_e = jnp.einsum(
                "up,tq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2c2 = l2c2_c - l2c2_e
            carry[1] += l2c2

            # r1 c1
            # 2: spin
            lr1 = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, r1, optimize="optimal")
            lc1 = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, c1, optimize="optimal")
            # 2: permutation, 0.5: coulomb
            l2r1c1_c = lr1 * lc1
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r1c1_e = 2.0 * jnp.einsum(
                "up,tq,pt,qu",
                gp_l_g_i,
                gp_l_g_i,
                r1.astype(self.mixed_real_dtype),
                c1.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2r1c1 = l2r1c1_c - l2r1c1_e
            carry[2] += l2r1c1

            # r2 c1
            # 2: spin
            lr2g = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, r2g, optimize="optimal")
            # 2: permutation, 0.5: coulomb
            l2r2c1_1_c = lc1 * lr2g
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c1_1_e = 2.0 * jnp.einsum(
                "up,tq,pt,qu",
                gp_l_g_i,
                gp_l_g_i,
                r2g.astype(self.mixed_complex_dtype),
                c1.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2r2c1_1 = l2r2c1_1_c - l2r2c1_1_e

            # 2: spin, 0.5: r2, 2: permutation
            lr2_c = 2.0 * jnp.einsum(
                "tp,ptqu->qu",
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 0.5: r2, 2: permutation
            lr2_e = jnp.einsum(
                "tp,puqt->qu",
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            lr2 = lr2_c - lr2_e
            lr2_c1 = (lr2 @ green_occ.T) @ c1
            c1_lr2 = c1_g @ lr2
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c1_2 = -2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, lr2_c1 + c1_lr2, optimize="optimal"
            )
            l2r2c1 = l2r2c1_1 + l2r2c1_2
            carry[3] += l2r2c1

            # r1 c2
            # 2: spin
            lc2g = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, c2g, optimize="optimal")
            # 2: permutation, 0.5: coulomb
            l2r1c2_1_c = lr1 * lc2g
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r1c2_1_e = 2.0 * jnp.einsum(
                "up,tq,pt,qu",
                gp_l_g_i,
                gp_l_g_i,
                r1.astype(self.mixed_real_dtype),
                c2g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            l2r1c2_1 = l2r1c2_1_c - l2r1c2_1_e

            # 2: spin, 0.5: c2, 2: permutation
            lc2_c = 2.0 * jnp.einsum(
                "tp,ptqu->qu",
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 0.5: c2, 2: permutation
            lc2_e = jnp.einsum(
                "tp,puqt->qu",
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            lc2 = lc2_c - lc2_e
            lc2_r1 = (lc2 @ green_occ.T) @ r1
            r1_lc2 = r1_g @ lc2
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r1c2_2 = -2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, lc2_r1 + r1_lc2, optimize="optimal"
            )
            l2r1c2 = l2r1c2_1 + l2r1c2_2
            carry[4] += l2r1c2

            return carry, 0.0

        l2g = jnp.zeros((nvirt, nocc)) + 0.0j
        [e2_r2_3, e2_c2_3, e2_r1c1_3, e2_r2c1_3, e2_r1c2_3, l2g], _ = lax.scan(
            loop_over_chol, [0.0j, 0.0j, 0.0j, 0.0j, 0.0j, l2g], (chol, lg, l_g)
        )
        e2_r2_2 = -2.0 * jnp.einsum("tp,pt->", l2g, r2g, optimize="optimal")
        e2_r2 = e2_r2_1 + e2_r2_2 + e2_r2_3

        # r1
        e2_r1_1 = l2g2 * r1g
        # 2: spin
        e2_r1_2 = -2.0 * jnp.einsum("tp,pt->", l2g, r1, optimize="optimal")
        e2_r1 = e2_r1_1 + e2_r1_2 + e2_r1c1_3

        # r1 c1
        e2_r1c1_1 = l2g2 * r1c1
        e2_r1c1_2 = -2.0 * jnp.einsum("tp,pt", l2g, r1_c1, optimize="optimal")
        e2_r1c1 = e2_r1c1_1 + e2_r1c1_2

        # r2 c1
        e2_r2c1_1 = l2g2 * r2c1
        e2_r2c1_2 = -2.0 * jnp.einsum("tp,pt", l2g, r2_c1, optimize="optimal")
        e2_r2c1_3 += e2_r2_3 * c1g
        e2_r2c1 = e2_r2c1_1 + e2_r2c1_2 + e2_r2c1_3

        # r1 c2
        e2_r1c2_1 = l2g2 * r1c2
        e2_r1c2_2 = -2.0 * jnp.einsum("tp,pt", l2g, r1_c2, optimize="optimal")
        e2_r1c2_3 += r1g * e2_c2_3
        e2_r1c2 = e2_r1c2_1 + e2_r1c2_2 + e2_r1c2_3

        e2 = e2_r1 + e2_r2 + e2_r1c1 + e2_r2c1 + e2_r1c2

        overlap = r1g + r1c1 + r2g2 + r2c1 + r1c2  # + r2c2
        return (e1 + e2) / overlap + e0

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class cisd_eom(wave_function):
    """(r1 + r2) (1 + c1 + c2) |0>, manual implementation

    Mixed precision is only used for N^6 scaling terms in the energy calculation. Can be turned off by setting
    mixed_real_dtype and mixed_complex_dtype to jnp.float64 and jnp.complex128 respectively.

    Attributes:
        norb: number of orbitals
        nelec: number of electrons as tuple (alpha, beta)
        n_batch: number of walkers in a batch
        mixed_real_dtype: real dtype of the mixed precision
        mixed_complex_dtype: complex dtype of the mixed precision
    """

    norb: int
    nelec: Tuple[int, int]
    n_batch: int = 1
    mixed_real_dtype: DTypeLike = jnp.float32
    mixed_complex_dtype: DTypeLike = jnp.complex64

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2, r1, r2 = (
            walker.shape[1],
            wave_data["ci1"],
            wave_data["ci2"],
            wave_data["r1"],
            wave_data["r2"],
        )
        green = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        green_occ = green[:, nocc:]
        # r1 terms
        # r1 1
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        r1_1 = r1g
        # r1 c1
        c1g = 2 * jnp.einsum("pt,pt", ci1, green_occ)
        r1_c1_1 = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = ci1 @ green_occ.T
        r1_c1_2 = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1_c1 = r1_c1_1 + r1_c1_2
        # r1 c2
        c2g2 = 2 * jnp.einsum("ptqu,pt,qu", ci2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", ci2, green_occ, green_occ
        )
        r1_c2_1 = r1g * c2g2
        c2g_1 = jnp.einsum("ptqu,qu->pt", ci2, green_occ)
        c2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", ci2, green_occ)
        gc2_g = (c2g_1 - c2g_2) @ green_occ.T
        r1_c2_2 = -4 * jnp.einsum("pq,qp", r1_g, gc2_g)
        r1_c2 = r1_c2_1 + r1_c2_2

        # r2 terms
        # r2 1
        r2g2 = 2 * jnp.einsum("ptqu,pt,qu", r2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", r2, green_occ, green_occ
        )
        r2_1 = r2g2
        # r2 c1
        r2_c1_1 = r2g2 * c1g
        r2g_1 = jnp.einsum("ptqu,qu->pt", r2, green_occ)
        r2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", r2, green_occ)
        gr2_g = (r2g_1 - r2g_2) @ green_occ.T
        r2_c1_2 = -4 * jnp.einsum("pq,qp", gr2_g, c1_g)
        r2_c1 = r2_c1_1 + r2_c1_2

        # r2 c2
        r2_c2_1 = r2g2 * c2g2
        r2_c2_2 = -8 * jnp.einsum("pq,qp", gr2_g, gc2_g)
        r2_int = jnp.einsum("ptqu,rt,su->prqs", r2, green_occ, green_occ)
        c2_int = jnp.einsum("ptqu,rt,su->prqs", ci2, green_occ, green_occ)
        r2_c2_3 = 2 * jnp.einsum("prqs,rpsq", r2_int, c2_int) - jnp.einsum(
            "prqs,rqsp", r2_int, c2_int
        )
        r2_c2 = r2_c2_1 + r2_c2_2 + r2_c2_3

        overlap_0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        return (r1_1 + r1_c1 + r1_c2 + r2_1 + r2_c1 + r2_c2) * overlap_0

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        c1, c2, r1, r2 = (
            jnp.array(wave_data["ci1"]),
            jnp.array(wave_data["ci2"]),
            jnp.array(wave_data["r1"]),
            jnp.array(wave_data["r2"]),
        )
        nocc = self.nelec[0]
        nvirt = self.norb - nocc
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(nvirt)))
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]

        # r1
        # 2: spin
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        # 2: spin
        lg = 2.0 * jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
        fb_r1_1 = r1g * lg
        g_r1_gp = green.T @ (r1 @ greenp.T)
        # 2: spin
        fb_r1_2 = -2 * jnp.einsum("gij,ji", chol, g_r1_gp)
        fb_r1 = fb_r1_1 + fb_r1_2

        # r1 c1
        # 2: spin
        c1g = 2 * jnp.einsum("pt,pt", c1, green_occ)
        r1c1_c = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = c1 @ green_occ.T
        # 2: spin
        r1c1_e = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1c1 = r1c1_c + r1c1_e
        fb_r1c1_1 = r1c1 * lg
        r1_c1 = r1 * c1g + r1g * c1 - r1_g @ c1 - c1_g @ r1
        g_r1_c1_gp = green.T @ (r1_c1 @ greenp.T)
        # 2: spin
        fb_r1c1_2 = -2 * jnp.einsum("gij,ji", chol, g_r1_c1_gp)
        fb_r1c1 = fb_r1c1_1 + fb_r1c1_2

        # r2
        # 2: spin, 0.5: r2, 2: permutation
        r2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", r2, green_occ)
        # 0.5: r2, 2: permutation
        r2g_e = jnp.einsum("ptqu,pu->qt", r2, green_occ)
        r2g = r2g_c - r2g_e
        # 2: spin, 0.5: no permuation
        r2g2 = jnp.einsum("qu,qu", r2g, green_occ, optimize="optimal")
        fb_r2_1 = lg * r2g2
        g_r2g_gp = green.T @ (r2g @ greenp.T)
        # 2: spin
        fb_r2_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r2g_gp)
        fb_r2 = fb_r2_1 + fb_r2_2

        # r2 c1
        r2c1_c = r2g2 * c1g
        r2g_g = r2g @ green_occ.T
        # 2: spin
        r2c1_e = -2.0 * jnp.einsum("pq,qp", r2g_g, c1_g, optimize="optimal")
        r2c1 = r2c1_c + r2c1_e
        fb_r2c1_1 = lg * r2c1
        r2_c1 = r2g * c1g + r2g2 * c1 - r2g_g @ c1 - c1_g @ r2g
        g_c1_g = green_occ.T @ c1_g
        # 2: spin, 2: permutation, 0.5: r2
        r2_c1 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_c1_g, r2, optimize="optimal")
        # 2: permutation, 0.5: r2
        r2_c1 += jnp.einsum("tq,ptqu->pu", g_c1_g, r2, optimize="optimal")
        g_r2_c1_gp = green.T @ (r2_c1 @ greenp.T)
        # 2: spin
        fb_r2c1_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r2_c1_gp)
        fb_r2c1 = fb_r2c1_1 + fb_r2c1_2

        # r1 c2
        # 2: spin, 0.5: c2, 2: permutation
        c2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", c2, green_occ)
        # 0.5: c2, 2: permutation
        c2g_e = jnp.einsum("ptqu,pu->qt", c2, green_occ)
        c2g = c2g_c - c2g_e
        # 2: spin, 0.5: no permuation
        c2g2 = jnp.einsum("qu,qu", c2g, green_occ, optimize="optimal")
        r1c2_c = r1g * c2g2
        c2g_g = c2g @ green_occ.T
        # 2: spin
        r1c2_e = -2.0 * jnp.einsum("pq,qp", r1_g, c2g_g, optimize="optimal")
        r1c2 = r1c2_c + r1c2_e
        fb_r1c2_1 = lg * r1c2
        r1_c2 = r1 * c2g2 + r1g * c2g - r1_g @ c2g - c2g_g @ r1
        g_r1_g = green_occ.T @ r1_g
        # 2: spin, 2: permutation, 0.5: c2
        r1_c2 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_r1_g, c2, optimize="optimal")
        # 2: permutation, 0.5: c2
        r1_c2 += jnp.einsum("tq,ptqu->pu", g_r1_g, c2, optimize="optimal")
        g_r1_c2_gp = green.T @ (r1_c2 @ greenp.T)
        # 2: spin
        fb_r1c2_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r1_c2_gp)
        fb_r1c2 = fb_r1c2_1 + fb_r1c2_2

        # r2 c2
        r2c2_c = r2g2 * c2g2
        # 2: spin
        r2c2_e_1 = -2.0 * jnp.einsum("pq,qp", r2g_g, c2g_g, optimize="optimal")
        # 0.5: r2
        r2_g = 0.5 * jnp.einsum("ptqu,rt->prqu", r2, green_occ)
        r2_g_g = jnp.einsum("prqu,su->prqs", r2_g, green_occ)
        # del r2_g
        # 0.5: c2
        c2_g = 0.5 * jnp.einsum("ptqu,rt->prqu", c2, green_occ)
        c2_g_g = jnp.einsum("prqu,su->prqs", c2_g, green_occ)
        # del c2_g
        # 4: spin, 2: permutation
        r2c2_e_2_c = 8.0 * jnp.einsum("prqs,rpsq", r2_g_g, c2_g_g, optimize="optimal")
        # 2: spin, 2: permutation
        r2c2_e_2_e = -4.0 * jnp.einsum("prqs,rqsp", r2_g_g, c2_g_g, optimize="optimal")
        r2c2_e_2 = r2c2_e_2_c + r2c2_e_2_e
        r2c2 = r2c2_c + r2c2_e_1 + r2c2_e_2
        fb_r2c2_1 = lg * r2c2
        r2_c2 = r2g2 * c2g + r2g * c2g2 - r2g_g @ c2g - c2g_g @ r2g
        # 2: spin, 2: permutation
        r2_c2 -= 4.0 * jnp.einsum("pr,rpqu->qu", r2g_g, c2_g, optimize="optimal")
        r2_c2 -= 4.0 * jnp.einsum("pr,rpqu->qu", c2g_g, r2_g, optimize="optimal")
        # 2: permutation
        r2_c2 += 2.0 * jnp.einsum("pr,qpru->qu", r2g_g, c2_g, optimize="optimal")
        r2_c2 += 2.0 * jnp.einsum("pr,qpru->qu", c2g_g, r2_g, optimize="optimal")
        # 2: spin, 4: permutation
        r2_c2 += 8.0 * jnp.einsum("pqrs,qpst->rt", r2_g_g, c2_g, optimize="optimal")
        r2_c2 += 8.0 * jnp.einsum("pqrs,qpst->rt", c2_g_g, r2_g, optimize="optimal")
        # 4: permutation
        r2_c2 -= 4.0 * jnp.einsum("pqrs,spqt->rt", r2_g_g, c2_g, optimize="optimal")
        r2_c2 -= 4.0 * jnp.einsum("pqrs,spqt->rt", c2_g_g, r2_g, optimize="optimal")
        g_r2_c2_gp = green.T @ (r2_c2 @ greenp.T)
        # 2: spin
        fb_r2c2_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r2_c2_gp)
        fb_r2c2 = fb_r2c2_1 + fb_r2c2_2

        overlap = r1g + r1c1 + r2g2 + r2c1 + r1c2 + r2c2
        fb = (fb_r1 + fb_r1c1 + fb_r2 + fb_r2c1 + fb_r1c2 + fb_r2c2) / overlap
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        c1, c2, r1, r2 = (
            jnp.array(wave_data["ci1"]),
            jnp.array(wave_data["ci2"]),
            jnp.array(wave_data["r1"]),
            jnp.array(wave_data["r2"]),
        )
        nocc = self.nelec[0]
        nvirt = self.norb - nocc
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(nvirt)))
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # r1
        # 2: spin
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        # 2: spin
        h1g = 2 * jnp.einsum("pt,pt", h1[:nocc, :], green)
        e1_r1_1 = r1g * h1g
        gp_h_g = greenp.T @ (h1 @ green.T)
        # 2: spin
        e1_r1_2 = -2 * jnp.einsum("tp,pt", gp_h_g, r1)
        e1_r1 = e1_r1_1 + e1_r1_2

        # r1 c1
        # 2: spin
        c1g = 2 * jnp.einsum("pt,pt", c1, green_occ)
        r1c1_c = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = c1 @ green_occ.T
        # 2: spin
        r1c1_e = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1c1 = r1c1_c + r1c1_e
        e1_r1c1_1 = r1c1 * h1g

        r1_c1 = r1 * c1g + r1g * c1 - r1_g @ c1 - c1_g @ r1
        # 2: spin
        e1_r1c1_2 = -2 * jnp.einsum("tp,pt", gp_h_g, r1_c1)
        e1_r1c1 = e1_r1c1_1 + e1_r1c1_2

        # r2
        # 2: spin, 0.5: r2, 2: permutation
        r2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", r2, green_occ)
        # 0.5: r2, 2: permutation
        r2g_e = jnp.einsum("ptqu,pu->qt", r2, green_occ)
        r2g = r2g_c - r2g_e
        # 2: spin, 0.5: no permuation
        r2g2 = jnp.einsum("qu,qu", r2g, green_occ, optimize="optimal")
        e1_r2_1 = h1g * r2g2
        # 2: spin
        e1_r2_2 = -2.0 * jnp.einsum("pt,tp", r2g, gp_h_g, optimize="optimal")
        e1_r2 = e1_r2_1 + e1_r2_2

        # r2 c1
        r2c1_c = r2g2 * c1g
        r2g_g = r2g @ green_occ.T
        # 2: spin
        r2c1_e = -2.0 * jnp.einsum("pq,qp", r2g_g, c1_g, optimize="optimal")
        r2c1 = r2c1_c + r2c1_e
        e1_r2c1_1 = h1g * r2c1
        r2_c1 = r2g * c1g + r2g2 * c1 - r2g_g @ c1 - c1_g @ r2g
        g_c1_g = green_occ.T @ c1_g
        # 2: spin, 2: permutation, 0.5: r2
        r2_c1 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_c1_g, r2, optimize="optimal")
        # 2: permutation, 0.5: r2
        r2_c1 += jnp.einsum("tq,ptqu->pu", g_c1_g, r2, optimize="optimal")
        # 2: spin
        e1_r2c1_2 = -2.0 * jnp.einsum("tp,pt", gp_h_g, r2_c1, optimize="optimal")
        e1_r2c1 = e1_r2c1_1 + e1_r2c1_2

        # r1 c2
        # 2: spin, 0.5: c2, 2: permutation
        c2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", c2, green_occ)
        # 0.5: c2, 2: permutation
        c2g_e = jnp.einsum("ptqu,pu->qt", c2, green_occ)
        c2g = c2g_c - c2g_e
        # 2: spin, 0.5: no permuation
        c2g2 = jnp.einsum("qu,qu", c2g, green_occ, optimize="optimal")
        r1c2_c = r1g * c2g2
        c2g_g = c2g @ green_occ.T
        # 2: spin
        r1c2_e = -2.0 * jnp.einsum("pq,qp", r1_g, c2g_g, optimize="optimal")
        r1c2 = r1c2_c + r1c2_e
        e1_r1c2_1 = h1g * r1c2
        r1_c2 = r1 * c2g2 + r1g * c2g - r1_g @ c2g - c2g_g @ r1
        g_r1_g = green_occ.T @ r1_g
        # 2: spin, 2: permutation, 0.5: c2
        r1_c2 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_r1_g, c2, optimize="optimal")
        # 2: permutation, 0.5: c2
        r1_c2 += jnp.einsum("tq,ptqu->pu", g_r1_g, c2, optimize="optimal")
        # 2: spin
        e1_r1c2_2 = -2.0 * jnp.einsum("tp,pt", gp_h_g, r1_c2, optimize="optimal")
        e1_r1c2 = e1_r1c2_1 + e1_r1c2_2

        # r2 c2
        r2c2_c = r2g2 * c2g2
        # 2: spin
        r2c2_e_1 = -2.0 * jnp.einsum("pq,qp", r2g_g, c2g_g, optimize="optimal")
        # 0.5: r2
        r2_g = 0.5 * jnp.einsum("ptqu,rt->prqu", r2, green_occ)
        r2_g_g = jnp.einsum("prqu,su->prqs", r2_g, green_occ)
        # 0.5: c2
        c2_g = 0.5 * jnp.einsum("ptqu,rt->prqu", c2, green_occ, optimize="optimal")
        c2_g_g = jnp.einsum("prqu,su->prqs", c2_g, green_occ, optimize="optimal")
        # 4: spin, 2: permutation
        r2c2_e_2_c = 8.0 * jnp.einsum("prqs,rpsq", r2_g_g, c2_g_g, optimize="optimal")
        # 2: spin, 2: permutation
        r2c2_e_2_e = -4.0 * jnp.einsum("prqs,rqsp", r2_g_g, c2_g_g, optimize="optimal")
        r2c2_e_2 = r2c2_e_2_c + r2c2_e_2_e
        r2c2 = r2c2_c + r2c2_e_1 + r2c2_e_2
        e1_r2c2_1 = h1g * r2c2
        r2_c2 = r2g2 * c2g + r2g * c2g2 - r2g_g @ c2g - c2g_g @ r2g
        # 2: spin, 2: permutation
        r2_c2 -= 4.0 * jnp.einsum("pr,rpqu->qu", r2g_g, c2_g, optimize="optimal")
        r2_c2 -= 4.0 * jnp.einsum("pr,rpqu->qu", c2g_g, r2_g, optimize="optimal")
        # 2: permutation
        r2_c2 += 2.0 * jnp.einsum("pr,qpru->qu", r2g_g, c2_g, optimize="optimal")
        r2_c2 += 2.0 * jnp.einsum("pr,qpru->qu", c2g_g, r2_g, optimize="optimal")
        # 2: spin, 4: permutation
        r2_c2 += 8.0 * jnp.einsum("pqrs,qpst->rt", r2_g_g, c2_g, optimize="optimal")
        r2_c2 += 8.0 * jnp.einsum("pqrs,qpst->rt", c2_g_g, r2_g, optimize="optimal")
        # 4: permutation
        r2_c2 -= 4.0 * jnp.einsum("pqrs,spqt->rt", r2_g_g, c2_g, optimize="optimal")
        r2_c2 -= 4.0 * jnp.einsum("pqrs,spqt->rt", c2_g_g, r2_g, optimize="optimal")
        e1_r2c2_2 = -2.0 * jnp.einsum("tp,pt", gp_h_g, r2_c2, optimize="optimal")
        e1_r2c2 = e1_r2c2_1 + e1_r2c2_2

        e1 = e1_r1 + e1_r1c1 + e1_r2 + e1_r2c1 + e1_r1c2 + e1_r2c2

        # 2 body energy
        # 2: spin
        lg = 2.0 * jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
        l_g = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
        # 0.5: coulomb
        l2g2_c = 0.5 * (lg @ lg)
        l2g2_e = -jnp.sum(vmap(lambda x: x * x.T)(l_g))
        l2g2 = l2g2_c + l2g2_e

        # doing this first to build intermediates
        # r2
        e2_r2_1 = l2g2 * r2g2

        # carry: [e2_r2_, e2_c2_, e2_r1c1_, e2_r2c1_, e2_r1c2_, e2_r2c2_, l2g]
        def loop_over_chol(carry, x):
            chol_i, lg_i, l_g_i = x
            # build intermediate
            gp_l_g_i = greenp.T @ (chol_i @ green.T)
            # 0.5: coulomb, 2: permutation
            l2g_i_c = gp_l_g_i * lg_i
            l2g_i_e = gp_l_g_i @ l_g_i
            l2g_i = l2g_i_c - l2g_i_e
            carry[6] += l2g_i

            gp_l_g_i = gp_l_g_i.astype(self.mixed_complex_dtype)
            # evaluate energy
            # r2
            # 4: spin, 2: permutation, 0.5: r2, 0.5: coulomb
            l2r2_c = 2.0 * jnp.einsum(
                "tp,uq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 2: spin, 2: permutation, 0.5: r2, 0.5: coulomb
            l2r2_e = jnp.einsum(
                "up,tq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2r2 = l2r2_c - l2r2_e
            carry[0] += l2r2

            # c2
            # 4: spin, 2: permutation, 0.5: c2, 0.5: coulomb
            l2c2_c = 2.0 * jnp.einsum(
                "tp,uq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 2: spin, 2: permutation, 0.5: c2, 0.5: coulomb
            l2c2_e = jnp.einsum(
                "up,tq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2c2 = l2c2_c - l2c2_e
            carry[1] += l2c2

            # r1 c1
            # 2: spin
            lr1 = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, r1, optimize="optimal")
            lc1 = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, c1, optimize="optimal")
            # 2: permutation, 0.5: coulomb
            l2r1c1_c = lr1 * lc1
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r1c1_e = 2.0 * jnp.einsum(
                "up,tq,pt,qu",
                gp_l_g_i,
                gp_l_g_i,
                r1.astype(self.mixed_real_dtype),
                c1.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2r1c1 = l2r1c1_c - l2r1c1_e
            carry[2] += l2r1c1

            # r2 c1
            # 2: spin
            lr2g = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, r2g, optimize="optimal")
            # 2: permutation, 0.5: coulomb
            l2r2c1_1_c = lc1 * lr2g
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c1_1_e = 2.0 * jnp.einsum(
                "up,tq,pt,qu",
                gp_l_g_i,
                gp_l_g_i,
                r2g.astype(self.mixed_complex_dtype),
                c1.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2r2c1_1 = l2r2c1_1_c - l2r2c1_1_e

            # 2: spin, 0.5: r2, 2: permutation
            lr2_c = -2.0 * jnp.einsum(
                "tp,ptqu->qu",
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 0.5: r2, 2: permutation
            lr2_e = jnp.einsum(
                "tp,puqt->qu",
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            lr2 = lr2_c + lr2_e
            lr2_c1 = (lr2 @ green_occ.T) @ c1
            c1_lr2 = c1_g @ lr2
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c1_2 = 2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, lr2_c1 + c1_lr2, optimize="optimal"
            )
            l2r2c1 = l2r2c1_1 + l2r2c1_2
            carry[3] += l2r2c1

            # r1 c2
            # 2: spin
            lc2g = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, c2g, optimize="optimal")
            # 2: permutation, 0.5: coulomb
            l2r1c2_1_c = lr1 * lc2g
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r1c2_1_e = 2.0 * jnp.einsum(
                "up,tq,pt,qu",
                gp_l_g_i,
                gp_l_g_i,
                r1.astype(self.mixed_real_dtype),
                c2g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            l2r1c2_1 = l2r1c2_1_c - l2r1c2_1_e

            # 2: spin, 0.5: c2, 2: permutation
            lc2_c = -2.0 * jnp.einsum(
                "tp,ptqu->qu",
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 0.5: c2, 2: permutation
            lc2_e = jnp.einsum(
                "tp,puqt->qu",
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            lc2 = lc2_c + lc2_e
            lc2_r1 = (lc2 @ green_occ.T) @ r1
            r1_lc2 = r1_g @ lc2
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r1c2_2 = 2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, lc2_r1 + r1_lc2, optimize="optimal"
            )
            l2r1c2 = l2r1c2_1 + l2r1c2_2
            carry[4] += l2r1c2

            # r2 c2
            # 2: permutation, 0.5: coulomb
            l2r2c2_1_c = lr2g * lc2g
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c2_1_e = -2.0 * jnp.einsum(
                "up,tq,pt,qu",
                gp_l_g_i,
                gp_l_g_i,
                r2g.astype(self.mixed_complex_dtype),
                c2g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            l2r2c2_1 = l2r2c2_1_c + l2r2c2_1_e

            lr2_g = lr2 @ green_occ.T
            lr2_c2g = lr2_g @ c2g
            # 2: spin, 2: permutaion, 0.5: coulomb
            l2r2c2_2_1 = 2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, lr2_c2g, optimize="optimal"
            )
            c2g_lr2 = c2g_g @ lr2
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c2_2_2 = 2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, c2g_lr2, optimize="optimal"
            )
            lc2_g = lc2 @ green_occ.T
            lc2_r2g = lc2_g @ r2g
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c2_2_3 = 2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, lc2_r2g, optimize="optimal"
            )
            r2g_lc2 = r2g_g @ lc2
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c2_2_4 = 2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, r2g_lc2, optimize="optimal"
            )
            l2r2c2_2 = l2r2c2_2_1 + l2r2c2_2_2 + l2r2c2_2_3 + l2r2c2_2_4

            # 4: spin, 0.5: coulomb, 0.5: r2, 4: permutation
            l2r2c2_3_1_1_c = 4.0 * jnp.einsum(
                "vp,wq,rvsw,prqs",
                gp_l_g_i,
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                c2_g_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            # 2: spin, 0.5: coulomb, 0.5; r2, 4: permutation
            l2r2c2_3_1_1_e = -2.0 * jnp.einsum(
                "vp,wq,svrw,prqs",
                gp_l_g_i,
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                c2_g_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            l2r2c2_3_1_1 = l2r2c2_3_1_1_c + l2r2c2_3_1_1_e
            # 4: spin, 0.5: coulomb, 0.5: c2, 4: permutation
            l2r2c2_3_1_2_c = 4.0 * jnp.einsum(
                "vp,wq,rvsw,prqs",
                gp_l_g_i,
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                r2_g_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            # 2: spin, 0.5: coulomb, 0.5: c2, 4: permutation
            l2r2c2_3_1_2_e = -2.0 * jnp.einsum(
                "vp,wq,svrw,prqs",
                gp_l_g_i,
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                r2_g_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            l2r2c2_3_1_2 = l2r2c2_3_1_2_c + l2r2c2_3_1_2_e
            l2r2c2_3_1 = l2r2c2_3_1_1 + l2r2c2_3_1_2

            # 4: spin, 8: permutation, 0.5: coulomb
            l2r2c2_3_2 = 16.0 * jnp.einsum(
                "vq,us,rpsv,prqu",
                gp_l_g_i,
                gp_l_g_i,
                r2_g.astype(self.mixed_complex_dtype),
                c2_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )

            # 2: spin, 8: permutation, 0.5: coulomb
            l2r2c2_3_3_1 = -8.0 * jnp.einsum(
                "vq,us,sprv,prqu",
                gp_l_g_i,
                gp_l_g_i,
                r2_g.astype(self.mixed_complex_dtype),
                c2_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            l2r2c2_3_3_2 = -8.0 * jnp.einsum(
                "vq,us,sprv,prqu",
                gp_l_g_i,
                gp_l_g_i,
                c2_g.astype(self.mixed_complex_dtype),
                r2_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            l2r2c2_3_3 = l2r2c2_3_3_1 + l2r2c2_3_3_2

            # 4: spin, 8: permutation, 0.5: coulomb
            l2r2c2_3_4 = 16.0 * jnp.einsum(
                "vp,us,sqrv,prqu",
                gp_l_g_i,
                gp_l_g_i,
                r2_g.astype(self.mixed_complex_dtype),
                c2_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )

            l2r2c2_3 = l2r2c2_3_1 + l2r2c2_3_2 + l2r2c2_3_3 + l2r2c2_3_4

            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c2_4 = -2.0 * jnp.einsum("pq,qp", lr2_g, lc2_g, optimize="optimal")

            l2r2c2 = l2r2c2_1 + l2r2c2_2 + l2r2c2_3 + l2r2c2_4
            carry[5] += l2r2c2

            return carry, 0.0

        l2g = jnp.zeros((nvirt, nocc)) + 0.0j
        [e2_r2_3, e2_c2_3, e2_r1c1_3, e2_r2c1_3, e2_r1c2_3, e2_r2c2_3, l2g], _ = (
            lax.scan(
                loop_over_chol,
                [0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, l2g],
                (chol, lg, l_g),
            )
        )
        e2_r2_2 = -2.0 * jnp.einsum("tp,pt->", l2g, r2g, optimize="optimal")
        e2_r2 = e2_r2_1 + e2_r2_2 + e2_r2_3

        # r1
        e2_r1_1 = l2g2 * r1g
        # 2: spin
        e2_r1_2 = -2.0 * jnp.einsum("tp,pt->", l2g, r1, optimize="optimal")
        e2_r1 = e2_r1_1 + e2_r1_2 + e2_r1c1_3

        # r1 c1
        e2_r1c1_1 = l2g2 * r1c1
        e2_r1c1_2 = -2.0 * jnp.einsum("tp,pt", l2g, r1_c1, optimize="optimal")
        e2_r1c1 = e2_r1c1_1 + e2_r1c1_2

        # r2 c1
        e2_r2c1_1 = l2g2 * r2c1
        e2_r2c1_2 = -2.0 * jnp.einsum("tp,pt", l2g, r2_c1, optimize="optimal")
        e2_r2c1_3 += e2_r2_3 * c1g
        e2_r2c1 = e2_r2c1_1 + e2_r2c1_2 + e2_r2c1_3

        # r1 c2
        e2_r1c2_1 = l2g2 * r1c2
        e2_r1c2_2 = -2.0 * jnp.einsum("tp,pt", l2g, r1_c2, optimize="optimal")
        e2_r1c2_3 += r1g * e2_c2_3
        e2_r1c2 = e2_r1c2_1 + e2_r1c2_2 + e2_r1c2_3

        # r2 c2
        e2_r2c2_1 = l2g2 * r2c2
        e2_r2c2_2 = -2.0 * jnp.einsum("tp,pt", l2g, r2_c2, optimize="optimal")
        e2_r2c2_3 += r2g2 * e2_c2_3
        e2_r2c2_3 += c2g2 * e2_r2_3
        e2_r2c2 = e2_r2c2_1 + e2_r2c2_2 + e2_r2c2_3

        e2 = e2_r1 + e2_r2 + e2_r1c1 + e2_r2c1 + e2_r1c2 + e2_r2c2

        overlap = r1g + r1c1 + r2g2 + r2c1 + r1c2 + r2c2
        return (e1 + e2) / overlap + e0

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))
