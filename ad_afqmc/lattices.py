from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence

import numpy as np
from jax import jit, lax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class


class lattice(ABC):
    """Abstract base class for lattice objects."""

    @abstractmethod
    def get_site_num(self, pos: Sequence) -> int:
        """Return the site number corresponding to a given position."""
        pass

    @abstractmethod
    def get_nearest_neighbors(self, pos: Sequence) -> Sequence:
        """Return the nearest neighbors of a site."""
        pass

    @abstractmethod
    def __hash__(self):
        pass


@dataclass
@register_pytree_node_class
class one_dimensional_chain(lattice):
    n_sites: int
    shape: Optional[tuple] = None
    sites: Optional[Sequence] = None
    bonds: Optional[Sequence] = None
    hop_signs: Sequence = (1.0, -1.0)
    coord_num: int = 2

    def __post_init__(self):
        self.shape = (self.n_sites,)
        self.sites = tuple([(i,) for i in range(self.n_sites)])
        self.bonds = (
            tuple(
                [
                    (
                        0,
                        i,
                    )
                    for i in range(self.n_sites)
                ]
            )
            if self.n_sites > 2
            else tuple([(0, 0)])
        )

    def get_bond_mode_distance(self, bond, mode):
        neigboring_sites = self.get_neighboring_sites(bond)
        dist_1 = self.get_distance(neigboring_sites[0], mode[1:])
        dist_2 = self.get_distance(neigboring_sites[1], mode[1:])
        lr = (dist_1 < dist_2) * 1.0 - (dist_1 > dist_2) * 1.0
        return jnp.min(jnp.array([dist_1, dist_2])), lr

    def get_site_num(self, pos):
        return pos[0]

    def get_marshall_sign(self, walker):
        if isinstance(walker, list):
            # TODO: this is a bit hacky
            walker = walker[0]
        walker_a = walker[::2]
        return (-1) ** jnp.sum(jnp.where(walker_a > 0, 1, 0))

    def get_symm_fac(self, pos, k):
        return (
            jnp.exp(2 * jnp.pi * 1.0j * k[0] * pos[0] / self.n_sites)
            if k is not None
            else 1.0
        )

    @partial(jit, static_argnums=(0,))
    def get_neighboring_bonds(self, pos):
        return jnp.array(
            [
                (
                    0,
                    (pos[0] - 1) % self.n_sites,
                ),
                (
                    0,
                    pos[0],
                ),
            ]
        )

    # ordering is used in the ssh model
    @partial(jit, static_argnums=(0,))
    def get_nearest_neighbors(self, pos):
        return jnp.array(
            [((pos[0] - 1) % self.n_sites,), ((pos[0] + 1) % self.n_sites,)]
        )

    def get_nearest_neighbor_modes(self, pos):
        return (
            jnp.array(
                [
                    (
                        0,
                        (pos[0] - 1) % self.n_sites,
                    ),
                    (
                        0,
                        (pos[0] + 1) % self.n_sites,
                    ),
                ]
            )
            if self.n_sites > 2
            else jnp.array(
                [
                    (
                        0,
                        1 - pos[0],
                    )
                ]
            )
        )

    @partial(jit, static_argnums=(0,))
    def get_neighboring_sites(self, bond):
        return [(bond[1] % self.n_sites,), ((bond[1] + 1) % self.n_sites,)]

    def get_neighboring_modes(self, bond):
        return [
            (
                bond[0],
                bond[1] % self.n_sites,
            ),
            (
                bond[0],
                (bond[1] + 1) % self.n_sites,
            ),
        ]

    def get_distance(self, pos_1, pos_2):
        return jnp.min(
            jnp.array(
                [
                    jnp.abs(pos_1[0] - pos_2[0]),
                    self.n_sites - jnp.abs(pos_1[0] - pos_2[0]),
                ]
            )
        )

    def get_bond_distance(self, pos_1, pos_2):
        return jnp.min(
            jnp.array(
                [
                    jnp.abs(pos_1[1] - pos_2[1]),
                    self.n_sites - jnp.abs(pos_1[1] - pos_2[1]),
                ]
            )
        )

    def create_adjacency_matrix(self):
        n_sites = self.n_sites
        h = np.zeros((n_sites, n_sites), dtype=int)

        for r in range(n_sites):
            neighbors = self.get_nearest_neighbors((r,))
            for nr in neighbors:
                h[r, nr] = 1
                h[nr, r] = 1
        return h

    def __hash__(self):
        return hash((self.n_sites, self.shape, self.sites, self.bonds))

    def tree_flatten(self):
        return (), (self.n_sites, self.shape, self.sites, self.bonds, self.coord_num)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


@dataclass
@register_pytree_node_class
class two_dimensional_grid(lattice):
    l_x: int
    l_y: int
    shape: Optional[tuple] = None
    shell_distances: Optional[Sequence] = None
    bond_shell_distances: Optional[Sequence] = None
    sites: Optional[Sequence] = None
    bonds: Optional[Sequence] = None
    n_sites: int = 0
    hop_signs: Sequence = (-1.0, -1.0, 1.0, 1.0)
    coord_num: int = 4

    def __post_init__(self):
        self.shape = (self.l_y, self.l_x)
        self.n_sites = self.l_x * self.l_y
        distances = []
        for x in range(self.l_x // 2 + 1):
            for y in range(self.l_y // 2 + 1):
                dist = x**2 + y**2
                distances.append(dist)
        distances = [*set(distances)]
        distances.sort()
        self.shell_distances = tuple(distances)
        self.sites = tuple(
            [(i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y)]
        )

        bond_distances = []
        for x in range(self.l_x + 1):
            for y in range(self.l_y + 1):
                if x % 2 == y % 2:
                    dist = x**2 + y**2
                    bond_distances.append(dist)
        bond_distances = [*set(bond_distances)]
        bond_distances.sort()
        self.bond_shell_distances = tuple(bond_distances)
        # if (self.l_x == 2) & (self.l_y == 2):
        #     self.bonds = ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0))
        # elif self.l_x == 2:
        #     self.bonds = tuple(
        #         [(0, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y)]
        #         + [(1, i, 0) for i in range(self.l_y)]
        #     )
        # elif self.l_y == 2:
        #     self.bonds = tuple(
        #         [(0, 0, i) for i in self.l_x]
        #         + [(1, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y)]
        #     )
        # else:
        self.bonds = tuple(
            [(0, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y)]
            + [(1, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y)]
        )

    def get_site_num(self, pos):
        return pos[1] + self.l_x * pos[0]

    def get_marshall_sign(self, walker):
        if isinstance(walker, list):
            # TODO: this is a bit hacky
            walker = walker[0]
        walker_a = walker[::2, ::2]
        return (-1) ** jnp.sum(jnp.where(walker_a > 0, 1, 0))

    def get_symm_fac(self, pos, k):
        return (
            jnp.exp(2 * jnp.pi * 1.0j * k[0] * pos[0] / self.l_x)
            * jnp.exp(2 * jnp.pi * 1.0j * k[1] * pos[1] / self.l_y)
            if k is not None
            else 1.0
        )

    @partial(jit, static_argnums=(0,))
    def get_distance(self, pos_1, pos_2):
        dist_y = jnp.min(
            jnp.array(
                [jnp.abs(pos_1[0] - pos_2[0]), self.l_y - jnp.abs(pos_1[0] - pos_2[0])]
            )
        )
        dist_x = jnp.min(
            jnp.array(
                [jnp.abs(pos_1[1] - pos_2[1]), self.l_x - jnp.abs(pos_1[1] - pos_2[1])]
            )
        )
        dist = dist_x**2 + dist_y**2
        shell_number = jnp.searchsorted(jnp.array(self.shell_distances), dist)
        return shell_number

    @partial(jit, static_argnums=(0,))
    def get_bond_distance(self, pos_1, pos_2):
        shifted_pos_1 = 2 * jnp.array(pos_1[1:])
        shifted_pos_1 = shifted_pos_1.at[pos_1[0]].add(1)
        shifted_pos_2 = 2 * jnp.array(pos_2[1:])
        shifted_pos_2 = shifted_pos_2.at[pos_2[0]].add(1)
        dist_y = jnp.min(
            jnp.array(
                [
                    jnp.abs(shifted_pos_1[0] - shifted_pos_2[0]),
                    2 * self.l_y - jnp.abs(shifted_pos_1[0] - shifted_pos_2[0]),
                ]
            )
        )
        dist_x = jnp.min(
            jnp.array(
                [
                    jnp.abs(shifted_pos_1[1] - shifted_pos_2[1]),
                    2 * self.l_x - jnp.abs(shifted_pos_1[1] - shifted_pos_2[1]),
                ]
            )
        )
        dist = dist_x**2 + dist_y**2
        shell_number = jnp.searchsorted(jnp.array(self.bond_shell_distances), dist)
        return shell_number

    def get_neighboring_bonds(self, pos):
        down = (0, *pos)
        right = (1, *pos)
        up = (0, (pos[0] - 1) % self.l_y, pos[1])
        left = (1, pos[0], (pos[1] - 1) % self.l_x)
        neighbors = []
        if (self.l_x == 2) & (self.l_y == 2):
            neighbors = [(0, 0, pos[1]), (1, pos[0], 0)]
        elif self.l_x == 2:
            neighbors = [right if pos[1] == 0 else left, down, up]
        elif self.l_y == 2:
            neighbors = [right, down if pos[0] == 0 else up, left]
        else:
            neighbors = [right, down, left, up]
        return jnp.array(neighbors)

    @partial(jit, static_argnums=(0,))
    def get_neighboring_sites(self, bond):
        # neighbors = [ ]
        # if bond[0] == 0:
        #  neighbors = [ (bond[1], bond[2]), ((bond[1] + 1) % self.l_y, bond[2]) ]
        # else:
        #  neighbors = [ (bond[1], bond[2]), (bond[1], (bond[2] + 1) % self.l_x) ]
        neighbors = lax.cond(
            bond[0] == 0,
            lambda x: [(bond[1], bond[2]), ((bond[1] + 1) % self.l_y, bond[2])],
            lambda x: [(bond[1], bond[2]), (bond[1], (bond[2] + 1) % self.l_x)],
            0,
        )
        return jnp.array(neighbors)

    @partial(jit, static_argnums=(0,))
    def get_neighboring_modes(self, bond):
        neighbors = lax.cond(
            bond[0] == 0,
            lambda x: [(0, bond[1], bond[2]), (0, (bond[1] + 1) % self.l_y, bond[2])],
            lambda x: [(1, bond[1], bond[2]), (1, bond[1], (bond[2] + 1) % self.l_x)],
            0,
        )
        return jnp.array(neighbors)

    @partial(jit, static_argnums=(0,))
    def get_nearest_neighbors(self, pos):
        right = (pos[0], (pos[1] + 1) % self.l_x)
        down = ((pos[0] + 1) % self.l_y, pos[1])
        left = (pos[0], (pos[1] - 1) % self.l_x)
        up = ((pos[0] - 1) % self.l_y, pos[1])
        return jnp.array([right, down, left, up])

    # used in the ssh model
    @partial(jit, static_argnums=(0,))
    def get_nearest_neighbor_modes(self, pos):
        right = (1, pos[0], (pos[1] + 1) % self.l_x)
        down = (0, (pos[0] + 1) % self.l_y, pos[1])
        left = (1, pos[0], (pos[1] - 1) % self.l_x)
        up = (0, (pos[0] - 1) % self.l_y, pos[1])
        neighbors = [right, down, left, up]
        return jnp.array(neighbors)

    # used in the ssh model
    @partial(jit, static_argnums=(0,))
    def get_bond_mode_distance(self, bond, mode):
        neighboring_sites = self.get_neighboring_sites(bond)
        dist_1 = self.get_distance(neighboring_sites[0], mode[1:])
        dist_2 = self.get_distance(neighboring_sites[1], mode[1:])
        # evaluate both parallel and perpendicular cases
        # parallel
        dist_bond_1 = jnp.min(
            jnp.array(
                [
                    jnp.abs(neighboring_sites[0][mode[0]] - mode[1:][mode[0]]),
                    jnp.array(self.shape)[mode[0]]
                    - jnp.abs(neighboring_sites[0][mode[0]] - mode[1:][mode[0]]),
                ]
            )
        )
        dist_bond_2 = jnp.min(
            jnp.array(
                [
                    jnp.abs(neighboring_sites[1][mode[0]] - mode[1:][mode[0]]),
                    jnp.array(self.shape)[mode[0]]
                    - jnp.abs(neighboring_sites[1][mode[0]] - mode[1:][mode[0]]),
                ]
            )
        )
        lr_bond = (dist_bond_1 < dist_bond_2) * 1.0 - (dist_bond_1 > dist_bond_2) * 1.0
        # perpendicular
        dist_site_1 = (
            neighboring_sites[1 - (dist_1 < dist_2)][mode[0]] - mode[1:][mode[0]]
        ) % jnp.array(self.shape)[mode[0]]
        dist_site_2 = (
            mode[1:][mode[0]] - neighboring_sites[1 - (dist_1 < dist_2)][mode[0]]
        ) % jnp.array(self.shape)[mode[0]]
        lr_site = (dist_site_1 < dist_site_2) * 1.0 - (dist_site_1 > dist_site_2) * 1.0
        lr = lr_bond * (bond[0] == mode[0]) + lr_site * (bond[0] != mode[0])
        return jnp.min(jnp.array([dist_1, dist_2])), lr

    def create_adjacency_matrix(self):
        width, height = self.l_x, self.l_y
        size = width * height
        h = np.zeros((size, size), dtype=int)

        for r in range(width):
            for q in range(height):
                i = q * width + r
                neighbors = self.get_nearest_neighbors((q, r))
                for nq, nr in neighbors:
                    if 0 <= nq < height and 0 <= nr < width:  # Check bounds
                        j = nq * width + nr
                        h[i, j] = 1
                        h[j, i] = 1
        return h

    def __hash__(self):
        return hash(
            (
                self.l_x,
                self.l_y,
                self.shape,
                self.shell_distances,
                self.bond_shell_distances,
                self.sites,
                self.bonds,
                self.coord_num,
            )
        )

    def tree_flatten(self):
        return (), (
            self.l_x,
            self.l_y,
            self.shape,
            self.shell_distances,
            self.bond_shell_distances,
            self.sites,
            self.bonds,
            self.coord_num,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


@dataclass
@register_pytree_node_class
class triangular_grid(lattice):
    l_x: int  # height
    l_y: int  # width
    shape: tuple = ()
    sites: Sequence = ()
    n_sites: int = 0
    coord_num: int = 6
    open_x: bool = False
    open_x_y: bool = False

    def __post_init__(self):
        self.shape = (self.l_y, self.l_x)
        self.n_sites = self.l_x * self.l_y
        self.sites = tuple(
            [(i // self.l_y, i % self.l_y) for i in range(self.l_x * self.l_y)]
        )

    def get_site_num(self, pos):
        return pos[1] + self.l_y * pos[0]

    def get_site_coordinate(self, pos):
        """
        Returns the real space coordinate of the site specified by `pos`,
        assuming a primitive lattice vector of unit length.
        """
        theta = np.pi / 3.0
        lattice_vecs = np.array([[np.cos(theta), 1.0], [np.sin(theta), 0.0]])
        if self.open_x:
            L2, L1 = lattice_vecs.T
            L3 = L2 - L1
            Ly = [L2, L3]
            coords = np.zeros(2)

            for i in range(1, pos[0] + 1):
                coords += Ly[(i - 1) % 2]

            coords += pos[1] * L1

        else:  # PBC and OBC.
            coords = pos @ lattice_vecs.T

        return coords

    # @partial(jit, static_argnums=(0,))
    def get_nearest_neighbors(self, pos):
        if self.open_x:
            n1 = (pos[0], (pos[1] + 1))
            n3 = (pos[0], (pos[1] - 1))
            if pos[0] % 2 == 1:
                n5 = ((pos[0] + 1) % self.l_x, (pos[1] + 1))
                n6 = ((pos[0] - 1) % self.l_x, (pos[1] + 1))
            else:
                n5 = ((pos[0] + 1) % self.l_x, (pos[1] - 1))
                n6 = ((pos[0] - 1) % self.l_x, (pos[1] - 1))
            n2 = ((pos[0] + 1) % self.l_x, pos[1])
            n4 = ((pos[0] - 1) % self.l_x, pos[1])
        elif self.open_x_y:
            n1 = (pos[0], (pos[1] + 1))
            n3 = (pos[0], (pos[1] - 1))
            if pos[0] % 2 == 1:
                n5 = ((pos[0] + 1), (pos[1] + 1))
                n6 = ((pos[0] - 1), (pos[1] + 1))
            else:
                n5 = ((pos[0] + 1), (pos[1] - 1))
                n6 = ((pos[0] - 1), (pos[1] - 1))
            # n5 = ((pos[0] + 1), (pos[1] + 1))
            # n6 = ((pos[0] - 1), (pos[1] - 1))
            n2 = ((pos[0] + 1), pos[1])
            n4 = ((pos[0] - 1), pos[1])
        else:
            n1 = (pos[0], (pos[1] + 1) % self.l_y)
            n3 = (pos[0], (pos[1] - 1) % self.l_y)
            n5 = ((pos[0] + 1) % self.l_x, (pos[1] + 1) % self.l_y)
            n6 = ((pos[0] - 1) % self.l_x, (pos[1] - 1) % self.l_y)
            n2 = ((pos[0] + 1) % self.l_x, pos[1])
            n4 = ((pos[0] - 1) % self.l_x, pos[1])

        return jnp.array([n1, n2, n3, n4, n5, n6])

    def create_adjacency_matrix(self):
        width, height = self.l_y, self.l_x
        size = width * height
        h = np.zeros((size, size), dtype=int)

        for r in range(width):
            for q in range(height):
                i = q * width + r
                neighbors = self.get_nearest_neighbors((q, r))
                for nq, nr in neighbors:
                    if 0 <= nq < height and 0 <= nr < width:  # Check bounds
                        j = nq * width + nr
                        h[i, j] = 1
                        h[j, i] = 1
        return h

    def __hash__(self):
        return hash(
            (
                self.l_x,
                self.l_y,
                self.shape,
                self.sites,
                self.coord_num,
            )
        )

    def tree_flatten(self):
        return (), (
            self.l_x,
            self.l_y,
            self.shape,
            self.sites,
            self.coord_num,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


@dataclass
@register_pytree_node_class
class three_dimensional_grid(lattice):
    l_x: int
    l_y: int
    l_z: int
    shape: Optional[tuple] = None
    shell_distances: Optional[Sequence] = None
    sites: Optional[Sequence] = None
    bonds: Optional[Sequence] = None
    n_sites: Optional[int] = None
    coord_num: int = 6

    def __post_init__(self):
        self.shape = (self.l_z, self.l_y, self.l_x)
        self.n_sites = self.l_x * self.l_y * self.l_z
        distances = []
        for x in range(self.l_x // 2 + 1):
            for y in range(self.l_y // 2 + 1):
                for z in range(self.l_z // 2 + 1):
                    dist = x**2 + y**2 + z**2
                    distances.append(dist)
        distances = [*set(distances)]
        distances.sort()
        self.shell_distances = tuple(distances)
        self.sites = tuple(
            [
                (
                    i // (self.l_x * self.l_y),
                    (i % (self.l_x * self.l_y)) // self.l_x,
                    (i % (self.l_x * self.l_y)) % self.l_x,
                )
                for i in range(self.l_x * self.l_y * self.l_z)
            ]
        )
        # TODO: fix bonds
        # self.bonds = tuple([ (i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y) ])

    def get_site_num(self, pos):
        return pos[2] + self.l_x * pos[1] + (self.l_x * self.l_y) * pos[0]

    # does not work
    def get_symm_fac(self, pos, k):
        return 1.0
        # return jnp.exp(2 * jnp.pi * 1.j * k[0] * pos[0] / self.n_sites) * jnp.exp(2 * jnp.pi * 1.j * k[1] * pos[1] / self.n_sites) if k is not None else 1.

    def get_distance(self, pos_1, pos_2):
        dist_z = jnp.min(
            jnp.array(
                [jnp.abs(pos_1[0] - pos_2[0]), self.l_z - jnp.abs(pos_1[0] - pos_2[0])]
            )
        )
        dist_y = jnp.min(
            jnp.array(
                [jnp.abs(pos_1[1] - pos_2[1]), self.l_y - jnp.abs(pos_1[1] - pos_2[1])]
            )
        )
        dist_x = jnp.min(
            jnp.array(
                [jnp.abs(pos_1[2] - pos_2[2]), self.l_x - jnp.abs(pos_1[2] - pos_2[2])]
            )
        )
        dist = dist_x**2 + dist_y**2 + dist_z**2
        shell_number = jnp.searchsorted(jnp.array(self.shell_distances), dist)
        return shell_number

    def get_bond_distance(self, pos_1, pos_2):
        # dist_y = jnp.min(jnp.array([jnp.abs(pos_1[0] - pos_2[0]), self.l_y - jnp.abs(pos_1[0] - pos_2[0])]))
        # dist_x = jnp.min(jnp.array([jnp.abs(pos_1[1] - pos_2[1]), self.l_x - jnp.abs(pos_1[1] - pos_2[1])]))
        # dist = dist_x**2 + dist_y**2
        # shell_number = jnp.searchsorted(jnp.array(self.shell_distances), dist)
        # return shell_number
        return 0

    # TODO: fix this
    def get_neighboring_bonds(self, pos):
        return jnp.array([pos])

    # ignoring side length 1 and 2 special cases
    def get_nearest_neighbors(self, pos):
        right = (pos[0], (pos[1] + 1) % self.l_y, pos[2])
        down = ((pos[0] + 1) % self.l_z, pos[1], pos[2])
        left = (pos[0], (pos[1] - 1) % self.l_y, pos[2])
        up = ((pos[0] - 1) % self.l_z, pos[1], pos[2])
        front = (pos[0], pos[1], (pos[2] + 1) % self.l_x)
        back = (pos[0], pos[1], (pos[2] - 1) % self.l_x)
        neighbors = [right, down, left, up, front, back]
        return jnp.array(neighbors)

    def __hash__(self):
        return hash(
            (
                self.l_x,
                self.l_y,
                self.l_z,
                self.shape,
                self.shell_distances,
                self.sites,
                self.bonds,
                self.coord_num,
            )
        )

    def tree_flatten(self):
        return (), (
            self.l_x,
            self.l_y,
            self.l_z,
            self.shape,
            self.shell_distances,
            self.sites,
            self.bonds,
            self.coord_num,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)
