import os
import sys
import h5py
import numpy as np
import scipy as sp

module_path = os.path.abspath(os.path.join("/burg/home/su2254/libs/ad_afqmc"))
if module_path not in sys.path:
    sys.path.append(module_path)

from ad_afqmc import (
    driver,
    pyscf_interface,
    mpi_jax,
    linalg_utils,
    spin_utils,
    lattices,
    propagation,
    wavefunctions,
    hamiltonian,
)

from pyscf import gto, scf, ao2mo

np.set_printoptions(precision=5, suppress=True)

lattice = lattices.triangular_grid(6, 6, open_x=True)
n_sites = lattice.n_sites
u = 6.0
n_elec = (18, 18)

integrals = {}
integrals["h0"] = 0.0

h1 = -1.0 * lattice.create_adjacency_matrix()
integrals["h1"] = h1

h2 = np.zeros((n_sites, n_sites, n_sites, n_sites))
for i in range(n_sites):
    h2[i, i, i, i] = u
integrals["h2"] = ao2mo.restore(8, h2, n_sites)

# make dummy molecule
mol = gto.Mole()
mol.nelectron = sum(n_elec)
mol.incore_anyway = True
mol.spin = abs(n_elec[0] - n_elec[1])
mol.verbose = 5
mol.build()

umf = scf.UHF(mol)
umf.get_hcore = lambda *args: integrals["h1"]
umf.get_ovlp = lambda *args: np.eye(n_sites)
umf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
dm_init = 1.0 * umf.init_guess_by_1e()
dm_init += 1.0 * np.random.randn(*dm_init.shape)
umf.kernel(dm_init, max_cycle=-1)
dm_init = umf.make_rdm1()

# Get Choleskies.
chol_cut = 1e-10
eri = ao2mo.restore(4, integrals["h2"], n_sites)
chol0 = pyscf_interface.modified_cholesky(eri, max_error=chol_cut)
nchol = chol0.shape[0]
chol = np.zeros((nchol, n_sites, n_sites))
for i in range(nchol):
    for m in range(n_sites):
        for n in range(m + 1):
            triind = m * (m + 1) // 2 + n
            chol[i, m, n] = chol0[i, triind]
            chol[i, n, m] = chol0[i, triind]

chol = chol.reshape(nchol, n_sites**2)
_eri = (chol.T @ chol).reshape((n_sites,) * 4)
max_absdiff = np.amax(np.absolute(h2 - _eri))
print(f'max_absdiff = {max_absdiff}')

# Save integrals.
with h5py.File('hubbard_ints.h5', 'w') as fh5:
    fh5["header"] = np.array([n_elec[0], n_elec[1], n_sites, u, chol_cut])
    fh5["h1e"] = h1
    fh5["chol"] = chol
    fh5["ehf0"] = umf.e_tot
    fh5["dm0a"] = dm_init[0]
    fh5["dm0b"] = dm_init[1]

header = f'{n_elec[0]}, {n_elec[1]}, {n_sites}, {u}, {chol_cut}'
np.savetxt('hubbard_ints.h1.csv', h1, header=header, delimiter=',')
np.savetxt('hubbard_ints.chol.csv', chol, header=header, delimiter=',')
np.savetxt('hubbard_ints.dm0a.csv', dm_init[0], header=header, delimiter=',')
np.savetxt('hubbard_ints.dm0b.csv', dm_init[1], header=header, delimiter=',')
np.savetxt('hubbard_ints.ehf0.csv', [umf.e_tot], header=header, delimiter=',')
