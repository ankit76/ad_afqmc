"""
Extended Hubbard model on a triangular lattice. Currently only supports up to
4 nearest neighbours.

Units: atomic units (Ha, Bohr)
"""

import pprint
import scipy
import numpy

# Custom modules.
from constants import *

class ExtendedHubbard(object):
    def __init__(self, nelec, N_arr, t_arr, U_arr, direct_vecs, verbose=False):
        """
        Attributes
        ----------
        nelec : Tuple
            Tuple of (nup, ndown) particles.

        N_arr : Tuple
            Tuple of (Nx, Ny) lattice sites.

        t_arr : Tuple
            Tuple of 1,...,k-th nearest-neighbour hopping parameters.

        U_arr : Tuple
            Tuple of 0,...,k-th nearest-neighbour Coulomb parameters.

        direct_vecs : :class:`numpy.ndarray`
            Direct lattice vectors.    
        """
        self.nelec = nelec # (nup, ndown)
        self.N_arr = N_arr # (Nx, Ny)
        self.t_arr = t_arr
        self.U_arr = U_arr
        self.direct_vecs = direct_vecs
        self.verbose = verbose

        self.nsite = np.prod(self.N_arr)
        self.ne = np.sum(self.nelec)
        self.nt = len(self.t_arr)
        self.nU = len(self.U_arr)
        self.max_k = max(self.nt, self.nU) - 1
        self.L = scipy.linalg.norm(self.direct_vecs[:, 0])

        if self.verbose:
            print(f'\n# (nup, ndown) = {self.nelec}')
            print(f'# L = {self.L * BOHR_TO_ANGSTROM:.3f} A')
            print(f'# (Nx, Ny) = {self.N_arr}')
            print(f'# nsite = {self.nsite}')
            print(f'# furthest neighbour = {self.max_k}')
            
            print(f'\n# Hopping terms:')
            for it in range(self.nt):
                print(f'#\tt{it} = {self.t_arr[it] * HARTREE_TO_EV * 1000.:.3e} meV')
            
            print(f'\n# Coulomb terms:')
            for iU in range(self.nU):
                print(f'#\tU{iU} = {self.U_arr[iU] * HARTREE_TO_EV * 1000.:.3e} meV')
        
        # To compute.
        self.knn_shells = None

    def build_knn_shells(self, grid_index=True):
        """
        Builds the matrices storing vectors that connect a site to its kth
        nearest-neighbours. The vectors are stored as rows of the matrix.
        """
        # Hard-code the vectors defining the k-th nearest-neighbours for now.
        # Most likely we'll only need k < 5.
        theta = np.pi / 3.
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        L0 = np.zeros(2)
        L1 = self.direct_vecs[:, 0]
        L2 = np.sum(self.direct_vecs, axis=1)
        L3 = 2*L1
        L4_0 = np.sum(self.direct_vecs, axis=1) + self.direct_vecs[:, 0]
        L4_1 = np.sum(self.direct_vecs, axis=1) + self.direct_vecs[:, 1]
        L_arr = [[L0], [L1], [L2], [L3], [L4_0, L4_1]]

        max_k = self.max_k
        self.knn_shells = dict.fromkeys([k for k in range(1, max_k)])
        
        for k, L_subarr in enumerate(L_arr):
            if k > max_k: continue
            nsub = len(L_subarr)
            knn_shell = np.zeros((6*nsub, 2))
            
            for i, Lk in enumerate(L_subarr):
                knn_shell[6*i] = Lk  

                for j in range(1, 6):
                    idx = 6*i + j
                    knn_shell[idx] = R @ knn_shell[idx-1]

            self.knn_shells[k] = knn_shell

        # Grid-index representation.
        if grid_index:
            for k in self.knn_shells:
                self.knn_shells[k] = np.rint(
                        self.knn_shells[k] @ scipy.linalg.inv(self.direct_vecs.T)).astype(int)

        if self.verbose:
            print(f'\n# self.knn_shells:')
            pprint.pprint(self.knn_shells)
        
        return self.knn_shells

    def build_hopping_matrix(self, k, wrap_around=True):
        """
        Builds the k-th nearest-neighbour hopping matrix.
        """
        if k == 0:
            tk_mat = np.diag(np.ones(self.nsite))

            # Check.
            np.testing.assert_allclose(np.count_nonzero(tk_mat, axis=1), 
                                       np.ones(self.nsite))
            
            return tk_mat * self.t_arr[k]
        
        if self.knn_shells is None: self.build_knn_shells()
        knn_shell = self.knn_shells[k]
        tk_mat = np.zeros((self.nsite, self.nsite), dtype=float)

        for i in range(self.nsite):
            coord = self.map_index_to_coord(i)
            knns = coord + knn_shell

            for knn in knns:
                if wrap_around:
                    knn[0] %= self.N_arr[0]
                    knn[1] %= self.N_arr[1]

                j = self.map_coord_to_index(knn)
                tk_mat[i, j] = 1.
        
        # Check.
        #np.testing.assert_allclose(np.count_nonzero(tk_mat, axis=1), 
        #                           6*np.ones(self.nsite))

        return tk_mat * self.t_arr[k]

    def build_coulomb_matrix(self, k, wrap_around=True):
        """
        Builds the k-th nearest-neighbour coulomb matrix.
        """
        if k == 0:
            Uk_mat = np.diag(np.ones(self.nsite))
            
            # Check.
            np.testing.assert_allclose(np.count_nonzero(Uk_mat, axis=1), 
                                       np.ones(self.nsite))
            
            return Uk_mat * self.U_arr[k]
        
        if self.knn_shells is None: self.build_knn_shells()
        knn_shell = self.knn_shells[k]
        Uk_mat = np.zeros((self.nsite, self.nsite), dtype=float)

        for i in range(self.nsite):
            coord = self.map_index_to_coord(i)
            knns = coord + knn_shell

            for knn in knns:
                if wrap_around:
                    knn[0] %= self.N_arr[0]
                    knn[1] %= self.N_arr[1]

                j = self.map_coord_to_index(knn)
                Uk_mat[i, j] = 1.
        
        # Check.
        #np.testing.assert_allclose(np.count_nonzero(Uk_mat, axis=1), 
        #                           6*np.ones(self.nsite))

        return Uk_mat * self.U_arr[k]

    def build_coulomb_eri(self, coulomb_matrix=None):
        """
        Builds the coulomb ERI tensor from the coulomb matrix.
        """
        nsite = self.nsite
        eri = np.zeros((nsite, nsite, nsite, nsite))

        if coulomb_matrix is None:
            coulomb_matrix = np.zeros((nsite, nsite))

            for k in range(self.nU):
                coulomb_matrix += self.build_coulomb_matrix(k)

        for i in range(self.nsite):
            eri[i, i, i, i] = 0.5 * coulomb_matrix[i, i]

            for j in range(self.nsite):
                if i != j:
                    eri[i, i, j, j] = 0.5 * coulomb_matrix[i, j]

        return eri
            
    def map_coord_to_index(self, coord):
        """
        Maps a lattice coordinate (x, y) to a site index, e.g.

        (0,2) (1,2) (2,2)       6 7 8
        (0,1) (1,1) (2,1)  ->   3 4 5
        (0,0) (1,0) (2,0)       0 1 2

        The index is given by
            index = x + y*Nx
        """
        Nx = self.N_arr[0]
        return coord[0] + coord[1]*Nx

    def map_index_to_coord(self, index):
        """
        Maps a site index to a lattice coordinate (x, y), e.g.

        6 7 8       (0,2) (1,2) (2,2)
        3 4 5   ->  (0,1) (1,1) (2,1)
        0 1 2       (0,0) (1,0) (2,0)

        The coordinate is given by
            x = index % Nx
            y = index // Nx
        """
        Nx, Ny = self.N_arr
        return np.array([index % Nx, (index//Nx)])
