from dataclasses import dataclass
from functools import partial
from typing import Sequence

import jax.numpy as jnp
from jax import jit

from ad_afqmc.propagation import propagator
from ad_afqmc.wavefunctions import wave_function


@dataclass
class hamiltonian:
    """Class for ab initio Hamiltonians. Contains methods for building intermediates.
    This class is fairly light, could be useful in the future for different kinds of Hamiltonians.

    Attributes:
        norb (int): Number of spatial orbitals.
    """

    norb: int

    @partial(jit, static_argnums=(0,))
    def rotate_orbs(self, ham_data: dict, mo_coeff: Sequence) -> dict:
        """Rotate the Hamiltonian to the molecular orbital basis defined by mo_coeff.

        Args:
            ham_data (dict): Hamiltonian data.
            mo_coeff (Sequence): Molecular orbital coefficients.

        Returns:
            dict: Rotated Hamiltonian data.
        """
        ham_data["h1"] = (
            ham_data["h1"].at[0].set(mo_coeff.T.dot(ham_data["h1"][0]).dot(mo_coeff))
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set(mo_coeff.T.dot(ham_data["h1"][1]).dot(mo_coeff))
        )
        ham_data["chol"] = jnp.einsum(
            "gij,jp->gip", ham_data["chol"].reshape(-1, self.norb, self.norb), mo_coeff
        )
        ham_data["chol"] = jnp.einsum(
            "qi,gip->gqp", mo_coeff.T, ham_data["chol"]
        ).reshape(-1, self.norb * self.norb)
        return ham_data

    @partial(jit, static_argnums=(0, 2))
    def build_measurement_intermediates(
        self, ham_data: dict, trial: wave_function, wave_data: dict
    ) -> dict:
        """Calculates and stores intermediates used in various measurements. A wrapper around the trial method.

        Args:
            ham_data (dict): Hamiltonian data.
            trial (wave_function): Trial wavefunction.
            wave_data (dict): Wavefunction data.

        Returns:
            dict: Hamiltonian data with intermediates.
        """
        ham_data = trial._build_measurement_intermediates(ham_data, wave_data)
        return ham_data

    @partial(jit, static_argnums=(0, 2, 3))
    def build_propagation_intermediates(
        self, ham_data: dict, prop: propagator, trial: wave_function, wave_data: dict
    ) -> dict:
        """Stores various intermediates used in propagation.

        Args:
            ham_data (dict): Hamiltonian data.
            prop (propagator): Propagator.
            trial (wave_function): Trial wavefunction.
            wave_data (Any): Wavefunction data.

        Returns:
            dict: Prepared Hamiltonian data.
        """
        ham_data = prop._build_propagation_intermediates(ham_data, trial, wave_data)
        return ham_data

    def __hash__(self) -> int:
        return hash(self.norb)
