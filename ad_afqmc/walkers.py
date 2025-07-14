from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

import jax
from jax import lax, tree_util, vmap

from ad_afqmc import linalg_utils, sr


class walker_batch(ABC):
    """Base class for walkers with batched iteration support."""

    @abstractmethod
    def apply_chunked(
        self, apply_fn: Callable, n_chunks: int, *args, **kwargs
    ) -> jax.Array:
        """Apply a function to all walkers in sequential chunks.

        Args:
            apply_fn: Function to apply to individual walker(s)
            n_chunks: Number of sequential chunks for memory-efficient processing
            *args: Additional arguments to pass to apply_fn
            **kwargs: Additional keyword arguments to pass to apply_fn

        Returns:
            Results from applying function to all walkers
        """
        pass

    @property
    @abstractmethod
    def n_walkers(self) -> int:
        """Total number of walkers."""
        pass

    @abstractmethod
    def stochastic_reconfiguration_local(
        self, weights: jax.Array, zeta: float
    ) -> Tuple["walker_batch", jax.Array]:
        """Perform stochastic reconfiguration locally on a process. Jax friendly."""
        pass

    @abstractmethod
    def stochastic_reconfiguration_global(
        self, weights: jax.Array, zeta: float, comm: Any
    ) -> Tuple["walker_batch", jax.Array]:
        """Perform stochastic reconfiguration globally across processes using MPI. Not jax friendly."""
        pass

    @abstractmethod
    def orthonormalize(self) -> "walker_batch":
        """Orthonormalize walkers."""
        pass


@dataclass
class RHFWalkers(walker_batch):
    """Wrapper for RHF walkers"""

    data: jax.Array

    @property
    def n_walkers(self) -> int:
        return self.data.shape[0]

    def apply_chunked(
        self, apply_fn: Callable, n_chunks: int, *args, **kwargs
    ) -> jax.Array:
        """Apply function to RHF walkers in sequential chunks."""
        chunk_size = self.n_walkers // n_chunks

        def scanned_fun(carry, walker_chunk):
            result_chunk = vmap(apply_fn, in_axes=(0, *[None] * len(args)))(
                walker_chunk, *args, **kwargs
            )
            return carry, result_chunk

        _, results = lax.scan(
            scanned_fun,
            None,
            self.data.reshape(n_chunks, chunk_size, *self.data.shape[1:]),
        )

        return results.reshape(self.n_walkers, *results.shape[2:])

    def stochastic_reconfiguration_local(
        self, weights: jax.Array, zeta
    ) -> Tuple["RHFWalkers", jax.Array]:
        new_data, new_weights = sr.stochastic_reconfiguration_restricted(
            self.data, weights, zeta
        )
        return self.__class__(new_data), new_weights

    def stochastic_reconfiguration_global(
        self, weights: jax.Array, zeta: float, comm: Any
    ) -> Tuple["RHFWalkers", jax.Array]:
        new_data, new_weights = sr.stochastic_reconfiguration_mpi_restricted(
            self.data, weights, zeta, comm
        )
        return self.__class__(new_data), new_weights

    def orthonormalize(self) -> "RHFWalkers":
        new_data, _ = linalg_utils.qr_vmap_restricted(self.data)
        return self.__class__(new_data)


# Register with JAX as a PyTree node
tree_util.register_pytree_node(
    RHFWalkers,
    lambda x: ([x.data], None),  # flatten: return children, aux_data
    lambda aux, children: RHFWalkers(
        *children
    ),  # unflatten: use children to reconstruct
)


@dataclass
class UHFWalkers(walker_batch):
    """Wrapper for RHF walkers"""

    data: List[jax.Array]

    @property
    def n_walkers(self) -> int:
        return self.data[0].shape[0]

    def apply_chunked(
        self, apply_fn: Callable, n_chunks: int, *args, **kwargs
    ) -> jax.Array:
        """Apply function to UHF walkers in sequential chunks."""
        chunk_size = self.n_walkers // n_chunks

        def scanned_fun(carry, walker_chunk):
            walker_chunk_up, walker_chunk_dn = walker_chunk
            result_chunk = vmap(apply_fn, in_axes=(0, 0, *[None] * len(args)))(
                walker_chunk_up, walker_chunk_dn, *args, **kwargs
            )
            return carry, result_chunk

        _, results = lax.scan(
            scanned_fun,
            None,
            (
                self.data[0].reshape(n_chunks, chunk_size, *self.data[0].shape[1:]),
                self.data[1].reshape(n_chunks, chunk_size, *self.data[1].shape[1:]),
            ),
        )

        return results.reshape(self.n_walkers, *results.shape[2:])

    def stochastic_reconfiguration_local(
        self, weights: jax.Array, zeta: float
    ) -> Tuple["UHFWalkers", jax.Array]:
        new_data, new_weights = sr.stochastic_reconfiguration_unrestricted(
            self.data, weights, zeta
        )
        return self.__class__(new_data), new_weights

    def stochastic_reconfiguration_global(
        self, weights: jax.Array, zeta: float, comm: Any
    ) -> Tuple["UHFWalkers", jax.Array]:
        new_data, new_weights = sr.stochastic_reconfiguration_mpi_unrestricted(
            self.data, weights, zeta, comm
        )
        return self.__class__(new_data), new_weights

    def orthonormalize(self) -> "UHFWalkers":
        new_up, _ = linalg_utils.qr_vmap_restricted(self.data[0])
        new_dn, _ = linalg_utils.qr_vmap_restricted(self.data[1])
        return UHFWalkers([new_up, new_dn])


tree_util.register_pytree_node(
    UHFWalkers,
    lambda x: (x.data, None),  # flatten: list of arrays, no aux
    lambda aux, children: UHFWalkers(
        list(children)
    ),  # unflatten: rewrap list into UHFWalkers
)


class GHFWalkers(RHFWalkers):
    """Wrapper for GHF walkers, implementation same as RHF."""

    pass


tree_util.register_pytree_node(
    GHFWalkers,
    lambda x: ([x.data], None),  # flatten: return children, aux_data
    lambda aux, children: GHFWalkers(
        *children
    ),  # unflatten: use children to reconstruct
)
