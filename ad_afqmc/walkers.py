from abc import ABC
from dataclasses import dataclass
from typing import List

import jax
from jax import tree_util


class walker_batch(ABC):
    """Base class for walkers

    Functions to be added.
    """


@dataclass
class RHFWalkers(walker_batch):
    """Wrapper for RHF walkers"""

    data: jax.Array


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


# Register UHFWalkers as a PyTree
tree_util.register_pytree_node(
    UHFWalkers,
    lambda x: (x.data, None),  # flatten: list of arrays, no aux
    lambda aux, children: UHFWalkers(
        list(children)
    ),  # unflatten: rewrap list into UHFWalkers
)


@dataclass
class GHFWalkers(walker_batch):
    """Wrapper for GHF walkers"""

    data: jax.Array


# Register with JAX as a PyTree node
tree_util.register_pytree_node(
    GHFWalkers,
    lambda x: ([x.data], None),  # flatten: return children, aux_data
    lambda aux, children: GHFWalkers(
        *children
    ),  # unflatten: use children to reconstruct
)
