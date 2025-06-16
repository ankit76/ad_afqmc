import os
import sys
import numpy as np
import scipy as sp
import jax.numpy as jnp

from pyscf import gto, scf, ao2mo

module_path = os.path.abspath(os.path.join("/projects/bcdd/shufay/ad_afqmc"))
if module_path not in sys.path:
    sys.path.append(module_path)

from ad_afqmc import (
    linalg_utils,
    spin_utils,
    lattices,
    hamiltonian,
    pyscf_interface,
    driver,
    mpi_jax,
    propagation,
    wavefunctions,
)

from ad_afqmc import (
    pyscf_interface,
    driver,
    mpi_jax,
    propagation,
    wavefunctions,
)

np.set_printoptions(precision=5, suppress=True)

# -----------------------------------------------------------------------------
U = 4.0
nup, ndown = 8, 8
n_elec = (nup, ndown)
nx, ny = 4, 4
nwalkers = 1
bc = 'open_x'
verbose = 1

# For saving.
tmpdir = f'./test_pyscf_interface'

# Lattice.
lattice = lattices.triangular_grid(nx, ny, open_x=True)
n_sites = lattice.n_sites
nocc = sum(n_elec)

# -----------------------------------------------------------------------------
# Integrals.
# UHF.
integrals = {}
integrals["h0"] = 0.0

h1 = -1.0 * lattice.create_adjacency_matrix()
integrals["h1"] = h1

h2 = np.zeros((n_sites, n_sites, n_sites, n_sites))
for i in range(n_sites): h2[i, i, i, i] = U
integrals["h2"] = ao2mo.restore(8, h2, n_sites)
integrals["u"] = U

# GHF.
integrals_g = {}
integrals_g["h0"] = 0.0
integrals_g["h1"] = sp.linalg.block_diag(h1, h1)
integrals_g["h2"] = integrals["h2"].copy()
integrals_g["u"] = U

# Diagonalize h1.
evals_h1, evecs_h1 = np.linalg.eigh(integrals["h1"])
evals_h1_g, evecs_h1_g = np.linalg.eigh(integrals_g["h1"])

# Check that they are identical.
mask = np.arange(n_sites) * 2
_evals_h1_g = evals_h1_g[mask]
np.testing.assert_allclose(evals_h1, _evals_h1_g)

# -----------------------------------------------------------------------------
# Make dummy molecule.
mol = gto.Mole()
mol.nelectron = nocc
mol.incore_anyway = True
mol.spin = abs(n_elec[0] - n_elec[1])
mol.verbose = verbose
mol.build()

# Prep AFQMC.
umf = scf.UHF(mol)
umf.get_hcore = lambda *args: integrals["h1"]
umf.get_ovlp = lambda *args: np.eye(n_sites)
umf._eri = ao2mo.restore(8, integrals["h2"] * 1.0, n_sites)
umf.mo_coeff = np.array([evecs_h1, evecs_h1])

gmf = scf.GHF(mol)
gmf.get_hcore = lambda *args: integrals_g["h1"]
gmf.get_ovlp = lambda *args: np.eye(2*n_sites)
gmf._eri = ao2mo.restore(8, integrals_g["h2"] * 1.0, n_sites)
gmf.mo_coeff = evecs_h1_g

# -----------------------------------------------------------------------------
def test_prep_afqmc():
    h1e, h1e_mod, chol = pyscf_interface.prep_afqmc(
        umf, basis_coeff=np.eye(n_sites), integrals=integrals, tmpdir=tmpdir
    )

    h1e_g, h1e_mod_g, chol_g = pyscf_interface.prep_afqmc_ghf(
        gmf, basis_coeff=np.eye(2*n_sites), integrals=integrals_g, tmpdir=tmpdir
    )
    
    # Check shapes.
    assert h1e.shape == (n_sites, n_sites)
    assert h1e_mod.shape == (n_sites, n_sites)
    assert chol.shape[-1] == n_sites**2
    
    assert h1e_g.shape == (2*n_sites, 2*n_sites)
    assert h1e_mod_g.shape == (2*n_sites, 2*n_sites)
    assert chol_g.shape[-1] == (2*n_sites)**2
    
    # Check elements.
    np.testing.assert_allclose(h1e, h1e_g[: n_sites, : n_sites])
    np.testing.assert_allclose(h1e, h1e_g[n_sites :, n_sites :])
    np.testing.assert_allclose(h1e_mod, h1e_mod_g[: n_sites, : n_sites])
    np.testing.assert_allclose(h1e_mod, h1e_mod_g[n_sites :, n_sites :])
    np.testing.assert_allclose(
        chol.reshape(-1, n_sites, n_sites), 
        chol_g.reshape(-1, 2*n_sites, 2*n_sites)[:, : n_sites, : n_sites]
    )
    np.testing.assert_allclose(
        chol.reshape(-1, n_sites, n_sites), 
        chol_g.reshape(-1, 2*n_sites, 2*n_sites)[:, n_sites :, n_sites :]
    )


if __name__ == '__main__':
    test_prep_afqmc()
