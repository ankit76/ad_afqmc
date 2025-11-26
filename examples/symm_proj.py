from ad_afqmc import lattices
from pyscf import gto, scf
import numpy as np
import scipy as sp
from ad_afqmc.vap import (
    cholesky_hamiltonian,
    hubbard_hamiltonian,
    sz_projector,
    singlet_projector,
    point_group_projector,
    optimize,
)
from jax import numpy as jnp

nx, ny = 6, 6
lattice = lattices.two_dimensional_grid(nx, ny, boundary="obc")
n_sites = lattice.n_sites
n_elec = (n_sites // 2, n_sites // 2)
u = 4

adj = lattice.create_adjacency_matrix()
h1 = -1.0 * adj
ene_h1, evec_h1 = np.linalg.eigh(h1)
ene_f = ene_h1[(n_sites) // 2 - 1]
print(f"Fermi energy at half-filling: {ene_f}")
for i in range(-4, 5):
    idx = (n_sites) // 2 - 1 + i
    print(f"State {idx}: Energy = {ene_h1[idx]}")
    if i == 0:
        print("  <-- Fermi level")

# make dummy molecule
mol = gto.Mole()
mol.nelectron = sum(n_elec)
mol.incore_anyway = True
mol.spin = abs(n_elec[0] - n_elec[1])
mol.nelec = n_elec
mol.nao = n_sites
mol.build()

gmf = scf.GHF(mol)
gmf.get_hcore = lambda *args: sp.linalg.block_diag(h1, h1)
gmf.get_ovlp = lambda *args: np.eye(2 * n_sites)
gmf._eri = None


def get_j(dm, U):
    N = n_sites
    V = np.zeros_like(dm, dtype=dm.dtype)
    for i in range(N):
        a = i
        b = i + N
        rho_i = np.array([[dm[a, a], dm[a, b]], [dm[b, a], dm[b, b]]], dtype=dm.dtype)
        Vi = U * (np.trace(rho_i) * np.eye(2, dtype=dm.dtype) - rho_i)
        V[a, a] += Vi[0, 0]
        V[a, b] += Vi[0, 1]
        V[b, a] += Vi[1, 0]
        V[b, b] += Vi[1, 1]
    return V


def get_jk(mol, dm, hermi=1, with_j=True, with_k=True, omega=None):
    V = get_j(dm, u) if (with_j or with_k) else np.zeros_like(dm)
    vj = V
    vk = np.zeros_like(V)
    return vj, vk


gmf.get_jk = get_jk

gmf.level_shift = 0.2
gmf.kernel()
mo1 = gmf.stability(external=True)
gmf = gmf.newton().run(mo1, gmf.mo_occ)
mo1 = gmf.stability(external=True)
gmf = gmf.newton().run(mo1, gmf.mo_occ)
mo1 = gmf.stability(external=True)
gmf = gmf.newton().run(mo1, gmf.mo_occ)

psi0 = jnp.array(gmf.mo_coeff[:, : sum(n_elec)])
ham = hubbard_hamiltonian(h1=jnp.array([h1, h1]), u=u)
projectors = (
    singlet_projector(n_alpha=5, n_beta=8, n_gamma=5),  # spin singlet
    # sz_projector(m=0, n_grid=10),
    # point_group_projector(pg_ops=pg_ops, pg_chars=pg_chars),
)
energy, psi_opt = optimize(
    psi=jnp.asarray(psi0),
    ham=ham,
    projectors=projectors,
    maxiter=200,
    step=0.01,
)

