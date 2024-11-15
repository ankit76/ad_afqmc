import sys
import numpy as np
import scipy as sp
import os

from jax import vmap, jit, numpy as jnp, random, lax, jvp, scipy as jsp

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
    hf_guess
)

from pyscf import fci, gto, scf, mp, ao2mo

np.set_printoptions(precision=5, suppress=True)

rank = mpi_jax.rank
comm = mpi_jax.comm
verbose = 4 if rank == 0 else 0

# -----------------------------------------------------------------------------
# Settings.
lattice = lattices.triangular_grid(6, 6, open_x=True)
n_sites = lattice.n_sites
u = 8.0
n_elec = (18, 18)
nocc = sum(n_elec)
run_cpmc = False

# -----------------------------------------------------------------------------
# Integrals.
integrals = {}
integrals["h0"] = 0.0

h1 = -1.0 * lattice.create_adjacency_matrix()
integrals["h1"] = h1

h2 = np.zeros((n_sites, n_sites, n_sites, n_sites))
for i in range(n_sites):
    h2[i, i, i, i] = u
integrals["h2"] = ao2mo.restore(8, h2, n_sites)

# -----------------------------------------------------------------------------
# make dummy molecule
mol = gto.Mole()
mol.nelectron = sum(n_elec)
mol.incore_anyway = True
mol.spin = abs(n_elec[0] - n_elec[1])
mol.verbose = verbose
mol.build()

# uhf
umf = scf.UHF(mol)
umf.get_hcore = lambda *args: integrals["h1"]
umf.get_ovlp = lambda *args: np.eye(n_sites)
umf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
dm_init = 1.0 * umf.init_guess_by_1e()
# for i in range(n_sites // 2):
#     dm_init[0, 2 * i, 2 * i] = 1.0
#     dm_init[1, 2 * i + 1, 2 * i + 1] = 1.0
dm_init += 1.0 * np.random.randn(*dm_init.shape)
umf.max_cycle = -1
umf.kernel(dm_init)

# -----------------------------------------------------------------------------
# ghf
gmf = scf.GHF(mol)
gmf.get_hcore = lambda *args: sp.linalg.block_diag(integrals["h1"], integrals["h1"])
gmf.get_ovlp = lambda *args: np.eye(2 * n_sites)
gmf._eri = ao2mo.restore(8, integrals["h2"], n_sites)

# costruct Neel guess.
occ = np.ones(2*n_sites)
occ[:nocc] = 1
psi0 = hf_guess.get_ghf_neel_guess(lattice)
dm_init = psi0 @ np.diag(occ) @ psi0.T.conj()
ao_ovlp = np.eye(n_sites)
epsilon0, mu, spin_axis = spin_utils.spin_collinearity_test(psi0[:, :nocc], ao_ovlp, verbose=verbose)

gmf.kernel(dm_init)
mo1 = gmf.stability(external=True)
gmf = gmf.newton().run(mo1, gmf.mo_occ)
mo1 = gmf.stability(external=True)
gmf = gmf.newton().run(mo1, gmf.mo_occ)
mo1 = gmf.stability(external=True)
gmf = gmf.newton().run(mo1, gmf.mo_occ)
mo1 = gmf.stability(external=True)

# Check spin collinearity.
gmf_coeff = gmf.mo_coeff
epsilon0, mu, spin_axis = spin_utils.spin_collinearity_test(gmf_coeff[:, :nocc], ao_ovlp, verbose=verbose)

rdm1 = gmf.make_rdm1()

# fci
if sum(n_elec) < 8:
    ci = fci.FCI(mol)
    e, ci_coeffs = ci.kernel(
        h1e=integrals["h1"], eri=integrals["h2"], norb=n_sites, nelec=n_elec
    )
    print(f"fci energy: {e}")

# -----------------------------------------------------------------------------
# ad afqmc
if rank == 0:
    pyscf_interface.prep_afqmc(umf, basis_coeff=np.eye(n_sites), integrals=integrals)

comm.Barrier()
options = {
    "dt": 0.005,
    "n_eql": 10,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 100,
    "n_prop_steps": 50,
    "n_walkers": 35,
    "seed": 98,
    "walker_type": "uhf",
    # "trial": "uhf",
    "save_walkers": False,
}

ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI = (
    mpi_jax._prep_afqmc(options)
)

# comment out this block for doing ph-afqmc
if run_cpmc:
    prop = propagation.propagator_cpmc(
        dt=options["dt"],
        n_walkers=options["n_walkers"],
    )

trial = wavefunctions.ghf_cpmc(n_sites, n_elec)
wave_data["mo_coeff"] = gmf_coeff[:, : n_elec[0] + n_elec[1]]
wave_data["rdm1"] = jnp.array([rdm1[:n_sites, :n_sites], rdm1[n_sites:, n_sites:]])
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)
ham_data["u"] = u

e_afqmc, err_afqmc = driver.afqmc(
    ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI
)
