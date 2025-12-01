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
    n_sites: Optional[int] = None
    hop_signs: Sequence = (-1.0, -1.0, 1.0, 1.0)
    coord_num: int = 4
    boundary: str = "pbc"

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
        if self.boundary == "pbc":
            right = (pos[0], (pos[1] + 1) % self.l_x)
            down = ((pos[0] + 1) % self.l_y, pos[1])
            left = (pos[0], (pos[1] - 1) % self.l_x)
            up = ((pos[0] - 1) % self.l_y, pos[1])
        else:  # obc
            right = (pos[0], pos[1] + 1)
            down = (pos[0] + 1, pos[1])
            left = (pos[0], pos[1] - 1)
            up = (pos[0] - 1, pos[1])

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
    boundary: str = "pbc"

    def __post_init__(self):
        self.shape = (self.l_x, self.l_y)
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

        if (self.boundary == "xc") or (self.boundary == "oxc"):
            L2, L1 = lattice_vecs.T
            L3 = L2 - L1
            Ly = [L2, L3]
            coords = np.zeros(2)

            for i in range(1, pos[0] + 1):
                coords += Ly[(i - 1) % 2]

            coords += pos[1] * L1

        elif self.boundary == "yc":
            theta = np.pi / 6.0
            lattice_vecs = np.array([[0.0, np.cos(theta)], [1.0, np.sin(theta)]])
            L1, L2 = lattice_vecs.T
            L3 = L2 - L1
            Lx = [L3, L2]
            coords = np.zeros(2)

            for i in range(1, pos[1] + 1):
                coords += Lx[(i - 1) % 2]

            coords += pos[0] * L1

        else:  # PBC and OBC.
            coords = pos @ lattice_vecs.T

        return coords

    # @partial(jit, static_argnums=(0,))
    def get_nearest_neighbors(self, pos):
        if self.boundary == "xc":
            n1 = (pos[0], pos[1] + 1)
            n3 = (pos[0], pos[1] - 1)
            if pos[0] % 2 == 1:
                n5 = ((pos[0] + 1) % self.l_x, pos[1] + 1)
                n6 = ((pos[0] - 1) % self.l_x, pos[1] + 1)
            else:
                n5 = ((pos[0] + 1) % self.l_x, pos[1] - 1)
                n6 = ((pos[0] - 1) % self.l_x, pos[1] - 1)
            n2 = ((pos[0] + 1) % self.l_x, pos[1])
            n4 = ((pos[0] - 1) % self.l_x, pos[1])

        elif self.boundary == "oxc":  # open-XC.
            n1 = (pos[0], pos[1] + 1)
            n3 = (pos[0], pos[1] - 1)
            if pos[0] % 2 == 1:
                n5 = (pos[0] + 1, pos[1] + 1)
                n6 = (pos[0] - 1, pos[1] + 1)
            else:
                n5 = (pos[0] + 1, pos[1] - 1)
                n6 = (pos[0] - 1, pos[1] - 1)
            n2 = (pos[0] + 1, pos[1])
            n4 = (pos[0] - 1, pos[1])

        elif self.boundary == "yc":
            n1 = (pos[0], pos[1] + 1)
            n3 = (pos[0], pos[1] - 1)
            if pos[1] % 2 == 1:
                n5 = ((pos[0] - 1) % self.l_x, pos[1] + 1)
                n6 = ((pos[0] - 1) % self.l_x, pos[1] - 1)
            else:
                n5 = ((pos[0] + 1) % self.l_x, pos[1] + 1)
                n6 = ((pos[0] + 1) % self.l_x, pos[1] - 1)
            n2 = ((pos[0] + 1) % self.l_x, pos[1])
            n4 = ((pos[0] - 1) % self.l_x, pos[1])

        elif self.boundary == "pbc":
            n1 = (pos[0], (pos[1] + 1) % self.l_y)
            n3 = (pos[0], (pos[1] - 1) % self.l_y)
            n5 = ((pos[0] + 1) % self.l_x, (pos[1] + 1) % self.l_y)
            n6 = ((pos[0] - 1) % self.l_x, (pos[1] - 1) % self.l_y)
            n2 = ((pos[0] + 1) % self.l_x, pos[1])
            n4 = ((pos[0] - 1) % self.l_x, pos[1])

        elif self.boundary == "obc":
            n1 = (pos[0], pos[1] + 1)
            n3 = (pos[0], pos[1] - 1)
            n5 = (pos[0] + 1, pos[1] - 1)
            n6 = (pos[0] - 1, pos[1] + 1)
            n2 = (pos[0] + 1, pos[1])
            n4 = (pos[0] - 1, pos[1])

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

    def get_neel_guess(self):
        boundary = self.boundary
        sites_0 = []
        sites_1 = []
        sites_2 = []
        for site in self.sites:
            x, y = site
            site_n = self.get_site_num(site)
            if boundary == "xc" or boundary == "oxc":
                sub = (y + 2 * (x & 1)) % 3
            elif boundary == "yc":
                sub = (x + (y & 1)) % 3
            else:
                raise ValueError(
                    "Neel guess only implemented for XC or OXC or YC boundaries."
                )
            if sub == 0:
                sites_0.append(site_n)
            elif sub == 1:
                sites_1.append(site_n)
            else:
                sites_2.append(site_n)
        spinor_0 = np.array([1, 0])
        spinor_0_dm = np.outer(spinor_0, spinor_0)
        theta = 2 * np.pi / 3
        spinor_1 = np.array([np.cos(theta / 2), np.sin(theta / 2)])
        spinor_1_dm = np.outer(spinor_1, spinor_1)
        spinor_2 = np.array([np.cos(theta / 2), -np.sin(theta / 2)])
        spinor_2_dm = np.outer(spinor_2, spinor_2)
        dm_init = np.zeros((2 * self.n_sites, 2 * self.n_sites))
        for i in sites_0:
            dm_init[i, i] = spinor_0_dm[0, 0]
            dm_init[i + self.n_sites, i + self.n_sites] = spinor_0_dm[1, 1]
            dm_init[i, i + self.n_sites] = spinor_0_dm[0, 1]
            dm_init[i + self.n_sites, i] = spinor_0_dm[1, 0]
        for i in sites_1:
            dm_init[i, i] = spinor_1_dm[0, 0]
            dm_init[i + self.n_sites, i + self.n_sites] = spinor_1_dm[1, 1]
            dm_init[i, i + self.n_sites] = spinor_1_dm[0, 1]
            dm_init[i + self.n_sites, i] = spinor_1_dm[1, 0]
        for i in sites_2:
            dm_init[i, i] = spinor_2_dm[0, 0]
            dm_init[i + self.n_sites, i + self.n_sites] = spinor_2_dm[1, 1]
            dm_init[i, i + self.n_sites] = spinor_2_dm[0, 1]
            dm_init[i + self.n_sites, i] = spinor_2_dm[1, 0]
        return dm_init

    def get_chiral_guess(self, tilt=np.pi / 3):
        boundary = self.boundary
        if boundary not in ("xc", "oxc", "yc"):
            raise ValueError(
                "Chiral guess only implemented for 'xc', 'oxc' or 'yc' boundaries."
            )
        sites_0, sites_1, sites_2 = [], [], []
        for x, y in self.sites:
            site_n = self.get_site_num((x, y))
            if boundary == "xc" or boundary == "oxc":
                sub = (y + 2 * (x & 1)) % 3
            elif boundary == "yc":
                sub = (x + (y & 1)) % 3
            if sub == 0:
                sites_0.append(site_n)
            elif sub == 1:
                sites_1.append(site_n)
            else:
                sites_2.append(site_n)

        def spinor(theta, phi):
            up = np.cos(theta / 2.0)
            dn = np.exp(1j * phi) * np.sin(theta / 2.0)
            return np.array([up, dn])

        s0 = spinor(tilt, 0.0)
        s1 = spinor(tilt, 2.0 * np.pi / 3.0)
        s2 = spinor(tilt, 4.0 * np.pi / 3.0)

        rho0 = np.outer(s0, s0.conj())
        rho1 = np.outer(s1, s1.conj())
        rho2 = np.outer(s2, s2.conj())

        dm = np.zeros((2 * self.n_sites, 2 * self.n_sites), dtype=complex)

        def put_block(i, rho):
            dm[i, i] = rho[0, 0]
            dm[i, i + self.n_sites] = rho[0, 1]
            dm[i + self.n_sites, i] = rho[1, 0]
            dm[i + self.n_sites, i + self.n_sites] = rho[1, 1]

        for i in sites_0:
            put_block(i, rho0)
        for i in sites_1:
            put_block(i, rho1)
        for i in sites_2:
            put_block(i, rho2)

        return dm

    def get_stripe_guess(self, axis="x"):
        """
        Simple collinear stripe state.
        axis="x": stripes along x (rows alternate up/down, uniform in y)
        axis="y": stripes along y (columns alternate up/down, uniform in x)
        """
        sites_up = []
        sites_dn = []

        for x, y in self.sites:
            site_n = self.get_site_num((x, y))
            if axis == "x":
                sub = x & 1
            else:
                sub = y & 1
            if sub == 0:
                sites_up.append(site_n)
            else:
                sites_dn.append(site_n)

        dm = np.zeros((2 * self.n_sites, 2 * self.n_sites), dtype=complex)

        for i in sites_up:
            dm[i, i] = 1.0

        for i in sites_dn:
            dm[i + self.n_sites, i + self.n_sites] = 1.0

        return dm

    def plot_observables(
        self,
        spin_rdm1,
        arrow_scale=1.5,
        annotate_sites=True,
        cmap="viridis",
        size=None,
        show_currents=False,
        current_scale=0.6,
        current_threshold=1e-3,
        edge_only=False,
        h1=None,
        t=1.0,
        current_color="magenta",
    ):
        """
        Plot charge density (color) and spin expectation (arrows) on a triangular lattice,
        and optionally charge currents on bonds.
        """
        import matplotlib.pyplot as plt

        n_sites = self.n_sites
        nx, ny = self.l_x, self.l_y

        boundary = self.boundary
        if boundary not in ("xc", "oxc", "yc"):
            raise ValueError(
                "plot_spin_charge only implemented for 'xc', 'oxc', or 'yc'."
            )

        r_uu = spin_rdm1[:n_sites, :n_sites]
        r_dd = spin_rdm1[n_sites:, n_sites:]
        r_ud = spin_rdm1[:n_sites, n_sites:]
        r_du = spin_rdm1[n_sites:, :n_sites]

        sz = (np.diag(r_uu) - np.diag(r_dd)) / 2.0
        charge = np.diag(r_uu) + np.diag(r_dd)
        sx = (np.diag(r_ud) + np.diag(r_du)) / 2.0
        sy = -0.5j * (np.diag(r_ud) - np.diag(r_du))

        sz = sz.real.reshape(nx, ny)
        sx = sx.real.reshape(nx, ny)
        sy = sy.real.reshape(nx, ny)
        charge = charge.real.reshape(nx, ny)  # type: ignore

        P = r_uu + r_dd

        coords = []
        xs, ys, cs = [], [], []
        for i, (x, y) in enumerate(self.sites):
            X, Y = self.get_site_coordinate((x, y))
            coords.append((X, Y))
            xs.append(X)
            ys.append(Y)
            cs.append(charge[x, y])
        xs = np.array(xs)
        ys = np.array(ys)
        cs = np.array(cs)

        def get_figsize(fig_height_pt, ratio):
            inches_per_pt = 1.0 / 72.0
            fig_height = fig_height_pt * inches_per_pt
            fig_width = fig_height / ratio
            return [fig_width, fig_height]

        if size is None:
            size = 100 * nx
        fig, ax = plt.subplots(figsize=get_figsize(size, ratio=nx / ny))

        def is_edge_site(x, y):
            if boundary in ("xc", "oxc"):  # PBC in x, open in y
                return (y == 0) or (y == ny - 1)
            elif boundary == "yc":  # PBC in y, open in x
                return (x == 0) or (x == nx - 1)
            else:
                return False

        for i, (x, y) in enumerate(self.sites):
            Xi, Yi = coords[i]
            for nn in self.get_nearest_neighbors((x, y)):
                xn, yn = int(nn[0]), int(nn[1])
                if not (0 <= xn < nx and 0 <= yn < ny):
                    continue
                j = self.get_site_num((xn, yn))
                if j <= i:
                    continue

                Xj, Yj = coords[j]

                is_pbc = (abs(xn - x) > 1) or (abs(yn - y) > 1)
                if is_pbc:
                    ax.plot([Xi, Xj], [Yi, Yj], "k--", lw=0.03, zorder=0)
                else:
                    ax.plot([Xi, Xj], [Yi, Yj], "k-", lw=0.6, zorder=1)

                if show_currents and not is_pbc:
                    # bond hopping
                    tij = h1[i, j] if h1 is not None else t
                    if tij == 0:
                        continue

                    Jij = 2.0 * np.imag(tij * P[i, j])  # spin-summed charge current

                    if abs(Jij) < current_threshold:
                        continue

                    if edge_only and not (is_edge_site(x, y) or is_edge_site(xn, yn)):
                        continue

                    # arrow along the bond, centered at the midpoint
                    xm = 0.5 * (Xi + Xj)
                    ym = 0.5 * (Yi + Yj)
                    dx = Xj - Xi
                    dy = Yj - Yi
                    L = np.hypot(dx, dy)
                    if L == 0:
                        continue
                    ux, uy = dx / L, dy / L

                    direction = np.sign(Jij) if Jij != 0 else 1.0
                    length = current_scale * abs(Jij)

                    ax.arrow(
                        xm,
                        ym,
                        direction * length * ux,
                        direction * length * uy,
                        head_width=0.08,
                        length_includes_head=True,
                        zorder=2.5,
                        color=current_color,
                    )

        sc = ax.scatter(
            xs,
            ys,
            c=cs,
            cmap=cmap,
            s=30 * (size / 100),
            edgecolors="k",
            linewidths=0.3,
            zorder=2,
        )

        for i, (x, y) in enumerate(self.sites):
            X, Y = coords[i]
            ax.arrow(
                X,
                Y,
                arrow_scale * sx[x, y],  # type: ignore
                arrow_scale * sz[x, y],  # type: ignore
                head_width=0.1,
                length_includes_head=True,
                zorder=3,
            )
            if annotate_sites:
                ax.text(
                    X + 0.1,
                    Y + 0.1,
                    f"{i}",
                    color="red",
                    zorder=4,
                )

        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("charge density")

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(False)

        plt.tight_layout()
        plt.show()

        return fig, ax

    def build_pg_ops_xc(self):
        """
        Full space-group ops for triangular XC cylinder (PBC in x, OBC in y)
        with a 2-row unit cell along x.

        Returns
        -------
        pg_ops  : list of (N, N) arrays
            All symmetry operators: {E, C2, G, C2G} * {T2^n}, n = 0..nx/2-1.
            Total number of ops = 2 * nx.
        pg_chars: list of ints
            Characters for the totally symmetric irrep (all +1).
        """
        nx = self.l_x
        ny = self.l_y

        if nx % 2 != 0:
            raise ValueError("nx must be even to have a 2-row unit cell along x.")

        n_sites = nx * ny
        E = np.eye(n_sites, dtype=float)

        def idx(x, y):
            return (x % nx) * ny + y

        # --- C2: inversion ---
        # (x, y) -> (nx-1-x, ny-1-y)  <=>  i -> N-1-i in row-major
        perm_c2 = np.arange(n_sites - 1, -1, -1)
        U_c2 = E[:, perm_c2]

        # --- Glide G: (x, y) -> (x+1, ny-1-y) ---
        perm_g = np.empty(n_sites, dtype=int)
        for x in range(nx):
            for y in range(ny):
                i = idx(x, y)
                perm_g[i] = idx((x + 1) % nx, ny - 1 - y)
        U_g = E[:, perm_g]

        # --- T2^n: translation by 2n rows: (x, y) -> (x+2n, y) ---
        def U_t2_power(n):
            perm = np.empty(n_sites, dtype=int)
            for x in range(nx):
                for y in range(ny):
                    i = idx(x, y)
                    perm[i] = idx((x + 2 * n) % nx, y)
            return E[:, perm]

        # Base (point-group) ops without translations
        base_ops = [E, U_c2, U_g, U_c2 @ U_g]

        pg_ops = []
        pg_chars = []

        # n = 0: just the base ops
        for U in base_ops:
            if not any(np.array_equal(U, V) for V in pg_ops):
                pg_ops.append(U)
                pg_chars.append(1)

        # n = 1 .. nx/2-1: T2^n and its products with base ops
        maxn = nx // 2
        for n in range(1, maxn):
            U_tn = U_t2_power(n)
            for U0 in base_ops:
                U = U0 @ U_tn
                if not any(np.array_equal(U, V) for V in pg_ops):
                    pg_ops.append(U)
                    pg_chars.append(1)

        return pg_ops, pg_chars

    def build_pg_ops_yc(self):
        """
        Full space-group ops for triangular YC cylinder (PBC in x, OBC in y)
        with a 1-row unit cell along x.

        Returns
        -------
        pg_ops  : list of (N, N) arrays
            All symmetry operators: {E, C2, G, C2G} * {T^n}, n = 0..nx-1.
            Total number of ops = 4 * nx.
        pg_chars: list of ints
            Characters for the totally symmetric irrep (all +1).
        """
        nx = self.l_x
        ny = self.l_y
        n_sites = nx * ny

        E = np.eye(n_sites, dtype=float)

        def idx(x, y):
            return (x % nx) * ny + y

        # --- C2: inversion ---
        perm_c2 = np.arange(n_sites - 1, -1, -1)
        U_c2 = E[:, perm_c2]

        # --- Glide G (YC) ---
        perm_g = np.empty(n_sites, dtype=int)
        for x in range(nx):
            for y in range(ny):
                i = idx(x, y)
                if y % 2 == 0:  # even column
                    xp = (x + 1) % nx
                else:  # odd column
                    xp = x
                yp = ny - 1 - y
                perm_g[i] = idx(xp, yp)
        U_g = E[:, perm_g]

        # --- T^n: translation by n rows: (x, y) -> (x + n, y) ---
        def U_t_power(n):
            perm = np.empty(n_sites, dtype=int)
            for x in range(nx):
                for y in range(ny):
                    i = idx(x, y)
                    perm[i] = idx(x + n, y)
            return E[:, perm]

        # Base (point-group) ops without translations
        base_ops = [E, U_c2, U_g, U_c2 @ U_g]

        pg_ops = []
        pg_chars = []

        # n = 0: just the base ops
        for U in base_ops:
            if not any(np.array_equal(U, V) for V in pg_ops):
                pg_ops.append(U)
                pg_chars.append(1)

        # n = 1 .. nx-1: T^n and its products with base ops
        for n in range(1, nx):
            U_tn = U_t_power(n)
            for U0 in base_ops:
                U = U0 @ U_tn
                if not any(np.array_equal(U, V) for V in pg_ops):
                    pg_ops.append(U)
                    pg_chars.append(1)

        return pg_ops, pg_chars

    def build_pg_ops(self):
        if self.boundary in "xc":
            return self.build_pg_ops_xc()
        elif self.boundary == "yc":
            return self.build_pg_ops_yc()
        else:
            raise ValueError("build_pg_ops only implemented for 'xc' or 'yc'.")

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
