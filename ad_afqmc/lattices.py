from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence

import numpy as np
from jax import jit, lax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class


def make_phonon_basis(n_sites, max_phonons):
    basis = []
    coefficients = [1 for _ in range(n_sites)]
    for i in range(max_phonons + 1):
        basis += frobenius(n_sites, coefficients, i)
    return basis


def frobenius(n, coefficients, target):
    """
    Enumerates solutions of the Frobenius equation with n integers.

    Args:
    - n: int, number of integers in the solution
    - coefficients: list of ints, coefficients for the Frobenius equation
    - target: int, target value for the Frobenius equation

    Returns:
    - list of tuples, each tuple represents a solution of the Frobenius equation
    """
    dp = [0] + [-1] * target  # Initialize the dynamic programming array

    for i in range(1, target + 1):
        for j in range(n):
            if coefficients[j] <= i and dp[i - coefficients[j]] != -1:
                dp[i] = j
                break

    if dp[target] == -1:
        return []  # No solution exists

    solutions = []
    current_solution = [0] * n

    def get_solution(i, remaining):
        if i == -1:
            if remaining == 0:
                solutions.append(jnp.array(current_solution))
            return

        for j in range(remaining // coefficients[i], -1, -1):
            current_solution[i] = j
            get_solution(i - 1, remaining - j * coefficients[i])

    get_solution(n - 1, target)
    return solutions


def frobenius_0(n, coefficients, target):
    def backtrack(index, current_solution):
        nonlocal solutions

        # Base case: when we have n integers in the solution
        if index == n:
            # Check if the current solution satisfies the Frobenius equation
            if sum([coefficients[i] * current_solution[i] for i in range(n)]) == target:
                solutions.append(jnp.array(current_solution))
            return

        # Recursive case: for each possible value of the current integer
        for i in range(target + 1):
            current_solution[index] = i
            backtrack(index + 1, current_solution)

    solutions = []
    current_solution = [0] * n
    backtrack(0, current_solution)
    return solutions


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

    def make_polaron_basis(self, max_n_phonons):
        phonon_basis = make_phonon_basis(self.n_sites, max_n_phonons)
        polaron_basis = tuple(
            [(i,), phonon_state]
            for i in range(self.n_sites)
            for phonon_state in phonon_basis
        )
        return polaron_basis

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
        if (self.l_x == 2) & (self.l_y == 2):
            self.bonds = ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0))
        elif self.l_x == 2:
            self.bonds = tuple(
                [(0, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y)]
                + [(1, i, 0) for i in range(self.l_y)]
            )
        elif self.l_y == 2:
            self.bonds = tuple(
                [(0, 0, i) for i in self.l_x]
                + [(1, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y)]
            )
        else:
            self.bonds = tuple(
                [(0, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y)]
                + [(1, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y)]
            )

    def get_site_num(self, pos):
        return pos[1] + self.l_x * pos[0]

    def make_polaron_basis(self, max_n_phonons):
        phonon_basis = make_phonon_basis(self.l_x * self.l_y, max_n_phonons)
        assert self.sites is not None
        polaron_basis = tuple(
            [site, phonon_state.reshape((self.l_y, self.l_x))]
            for site in self.sites
            for phonon_state in phonon_basis
        )
        return polaron_basis

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
    boundary_condition: str = "pbc"

    def __post_init__(self):
        self.shape = (self.l_x, self.l_y)
        self.n_sites = self.l_x * self.l_y
        # Counting along the y-axis.
        self.sites = tuple(
            [(i // self.l_y, i % self.l_y) for i in range(self.n_sites)]
        )

    def get_site_num(self, pos):
        return pos[1] + self.l_y * pos[0]
    
    def get_site_coordinate_from_num(self, num):
        pos = np.array([num // self.l_y, num % self.l_y])
        return self.get_site_coordinate(pos)

    def get_site_coordinate(self, pos):
        """
        Returns the real space coordinate of the site specified by `pos`,
        assuming a primitive lattice vector of unit length.
        """
        theta = np.pi / 3.
        lattice_vecs = np.array([[1., np.cos(theta)],
                                 [0., np.sin(theta)]])
        if self.boundary_condition == "open_x":
            L1, L2 = lattice_vecs.T
            L3 = L2 - L1
            Ly = [L2, L3]
            coords = np.zeros(2)

            for i in range(1, pos[1]+1):
                coords += Ly[(i-1) % 2]

            coords += pos[0] * L1
            
        else: # PBC and OPBC.
            coords = pos @ lattice_vecs.T

        return coords
    
    def get_boundary_pairs(self):
        """
        Returns an array of lattice sites pairs connected by the periodic boundary
        condition.
        """
        l_x, l_y = self.l_x, self.l_y
        bottom_sites = np.arange(l_x) * l_x
        left_sites = np.arange(l_y)
        top_sites = bottom_sites + (l_y-1)
        right_sites = left_sites + (l_x-1) * l_y
        boundary_pairs_x = list(zip(bottom_sites, top_sites))
        boundary_pairs_y = list(zip(left_sites, right_sites))
        boundary_pairs = np.vstack([boundary_pairs_x, boundary_pairs_y])
        return boundary_pairs

    # @partial(jit, static_argnums=(0,))
    def get_nearest_neighbors(self, pos):
        """
        Assumes the lattice vectors
            L1 = [1,         0],
            L2 = [cos(pi/3), sin(pi/3)]
        """
        if self.boundary_condition == "opbc":
            n1 = (pos[0] + 1, pos[1])
            n2 = (pos[0], pos[1] + 1)
            n3 = (pos[0] - 1, pos[1])
            n4 = (pos[0], pos[1] - 1)
            n5 = (pos[0] + 1, pos[1] - 1)
            n6 = (pos[0] - 1, pos[1] + 1)

        else:
            if self.boundary_condition == "open_x":
                # Treat each staggered vertical stripe as parallel to the y-axis.
                # The x-axis is the usual.
                n1 = ((pos[0] + 1), pos[1])
                n3 = ((pos[0] - 1), pos[1])
                if pos[1] % 2 == 1:
                    n5 = ((pos[0] + 1), (pos[1] + 1) % self.l_y)
                    n6 = ((pos[0] + 1), (pos[1] - 1) % self.l_y)
                else:
                    n5 = ((pos[0] - 1), (pos[1] + 1) % self.l_y)
                    n6 = ((pos[0] - 1), (pos[1] - 1) % self.l_y)

            else: # PBC.
                n1 = ((pos[0] + 1) % self.l_x, pos[1])
                n3 = ((pos[0] - 1) % self.l_x, pos[1])
                n5 = ((pos[0] + 1) % self.l_x, (pos[1] - 1) % self.l_y)
                n6 = ((pos[0] - 1) % self.l_x, (pos[1] + 1) % self.l_y)

            n2 = (pos[0], (pos[1] + 1) % self.l_y)
            n4 = (pos[0], (pos[1] - 1) % self.l_y)

        return jnp.array([n1, n2, n3, n4, n5, n6])

    def create_adjacency_matrix(self):
        width, height = self.l_x, self.l_y
        size = width * height
        h = np.zeros((size, size), dtype=int)

        for q in range(width):
            for r in range(height):
                i = q * width + r
                neighbors = self.get_nearest_neighbors((q, r))
                for nq, nr in neighbors:
                    if 0 <= nq < width and 0 <= nr < height:  # Check bounds
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

    def make_polaron_basis(self, max_n_phonons):
        phonon_basis = make_phonon_basis(self.l_x * self.l_y * self.l_z, max_n_phonons)
        assert self.sites is not None
        polaron_basis = tuple(
            [site, phonon_state.reshape((self.l_x, self.l_y, self.l_z))]
            for site in self.sites
            for phonon_state in phonon_basis
        )
        return polaron_basis

    def make_polaron_basis_n(self, n_bands, max_n_phonons):
        phonon_basis = make_phonon_basis(self.l_x * self.l_y * self.l_z, max_n_phonons)
        assert self.sites is not None
        electronic_basis = tuple(
            [(n, site) for n in range(n_bands) for site in self.sites]
        )
        polaron_basis = tuple(
            [site, phonon_state.reshape((self.l_x, self.l_y, self.l_z))]
            for site in electronic_basis
            for phonon_state in phonon_basis
        )
        return polaron_basis

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
