from dataclasses import dataclass
import jax
from typing import List
from jax import tree_util

@dataclass
class RHFWalkers:
    """Wrapper for RHF walkers"""

    data: jax.Array

    # def __post_init__(self):
    #     assert (
    #         len(self.data.shape) == 3
    #     ), f"RHF walkers must be 3D, got {self.data.shape}"

# Register with JAX as a PyTree node
tree_util.register_pytree_node(
    RHFWalkers,
    lambda x: ([x.data], None),             # flatten: return children, aux_data
    lambda aux, children: RHFWalkers(*children),  # unflatten: use children to reconstruct
)

@dataclass
class UHFWalkers:
    """Wrapper for RHF walkers"""

    data: List[jax.Array]

    # def __post_init__(self):
    #     assert (
    #         len(self.data[0].shape) == 3 and len(self.data[1].shape) == 3
    #     ), f"UHF walkers must be 3D, got {self.data[0].shape}, {self.data[1].shape}"

# Register UHFWalkers as a PyTree
tree_util.register_pytree_node(
    UHFWalkers,
    lambda x: (x.data, None),                  # flatten: list of arrays, no aux
    lambda aux, children: UHFWalkers(list(children)) # unflatten: rewrap list into UHFWalkers
)

@dataclass
class GHFWalkers:
    """Wrapper for GHF walkers"""

    data: jax.Array

    # def __post_init__(self):
    #     assert (
    #         len(self.data.shape) == 3
    #     ), f"GHF walkers must be 3D, got {self.data.shape}"

# Register with JAX as a PyTree node
tree_util.register_pytree_node(
    GHFWalkers,
    lambda x: ([x.data], None),             # flatten: return children, aux_data
    lambda aux, children: GHFWalkers(*children),  # unflatten: use children to reconstruct
)
