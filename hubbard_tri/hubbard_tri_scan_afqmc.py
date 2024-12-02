import os
import sys
import h5py
import argparse
import numpy as np
import scipy as sp

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
from io_tools import read_real_array

np.set_printoptions(precision=5, suppress=True)

rank = mpi_jax.rank
comm = mpi_jax.comm

# -----------------------------------------------------------------------------
# Parser.
parser = argparse.ArgumentParser()

# Required.
parser.add_argument('--U', type=float, required=True)
parser.add_argument('--nup', type=int, required=True)
parser.add_argument('--ndown', type=int, required=True)
parser.add_argument('--nx', type=int, required=True)
parser.add_argument('--ny', type=int, required=True)
parser.add_argument('--nwalkers', type=int, required=True)

# Optional.
parser.add_argument('--open_x', type=int, required=False, nargs='?', default=0)
parser.add_argument('--run_cpmc', type=int, required=False, nargs='?', default=0)
parser.add_argument('--load_psi0', type=int, required=False, nargs='?', default=0)
parser.add_argument('--in_dir', type=str, required=False, nargs='?', default='./')
parser.add_argument('--out_dir', type=str, required=False, nargs='?', default='./')
parser.add_argument('--filetag', type=str, required=False, nargs='?', default='')
parser.add_argument('--verbose', type=int, required=False, nargs='?', default=0)

args = parser.parse_args()

U = args.U
nup = args.nup
ndown = args.ndown
nx = args.nx
ny = args.ny
nwalkers = args.nwalkers

open_x = args.open_x
load_psi0 = args.load_psi0
in_dir = args.in_dir
out_dir = args.out_dir
filetag = args.filetag
run_cpmc = args.run_cpmc
verbose = args.verbose if rank == 0 else 0

# -----------------------------------------------------------------------------
# Settings.
lattice = lattices.triangular_grid(nx, ny, open_x=open_x)
n_sites = lattice.n_sites
n_elec = (nup, ndown)
nocc = sum(n_elec)

# -----------------------------------------------------------------------------
# Integrals.
integrals = {}
integrals["h0"] = 0.0

h1 = -1.0 * lattice.create_adjacency_matrix()
integrals["h1"] = h1

h2 = np.zeros((n_sites, n_sites, n_sites, n_sites))
for i in range(n_sites):
    h2[i, i, i, i] = U
integrals["h2"] = ao2mo.restore(8, h2, n_sites)

# -----------------------------------------------------------------------------
# Make dummy molecule.
mol = gto.Mole()
mol.nelectron = nocc
mol.incore_anyway = True
mol.spin = abs(n_elec[0] - n_elec[1])
mol.verbose = verbose
mol.build()

# UHF.
umf = scf.UHF(mol)
umf.get_hcore = lambda *args: integrals["h1"]
umf.get_ovlp = lambda *args: np.eye(n_sites)
umf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
umf.max_cycle = -1
umf.kernel()

# -----------------------------------------------------------------------------
# GHF.
ghf_coeff = ghf_rdm1 = None

if load_psi0:
    fname = f'{in_dir}{filetag}.h5'
    h5 = h5py.File(fname)
    ghf_etot = read_real_array(h5, 'ghf_etot')
    ghf_coeff = read_real_array(h5, 'ghf_coeff')
    ghf_rdm1 = read_real_array(h5, 'ghf_rdm1')
    h5.close()
    
    if verbose: 
        print(f'\n# Loaded psi0 from {fname}')
        print(f'# E_GHF = {ghf_etot}')

else:
    gmf = scf.GHF(mol)
    gmf.get_hcore = lambda *args: sp.linalg.block_diag(integrals["h1"], integrals["h1"])
    gmf.get_ovlp = lambda *args: np.eye(2 * n_sites)
    gmf._eri = ao2mo.restore(8, integrals["h2"], n_sites)

    dm_init = 1.0 * umf.init_guess_by_1e()
    dm_init = sp.linalg.block_diag(dm_init[0], dm_init[1])
    noise = 0.5 * np.amax(np.absolute(dm_init))
    dm_init += noise * np.random.randn(*dm_init.shape)

    gmf.kernel(dm_init)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)
    mo1 = gmf.stability(external=True)

    ghf_coeff = gmf.mo_coeff
    ghf_rdm1 = gmf.make_rdm1()
    ao_ovlp = np.eye(n_sites)
    epsilon0, mu, spin_axis = spin_utils.spin_collinearity_test(
                                ghf_coeff[:, :nocc], ao_ovlp, verbose=verbose)

# -----------------------------------------------------------------------------
# FCI.
if sum(n_elec) < 8:
    ci = fci.FCI(mol)
    e, ci_coeffs = ci.kernel(
        h1e=integrals["h1"], eri=integrals["h2"], norb=n_sites, nelec=n_elec
    )
    print(f"fci energy: {e}")

# -----------------------------------------------------------------------------
# AFQMC.
if rank == 0:
    pyscf_interface.prep_afqmc(
            umf, basis_coeff=np.eye(n_sites), integrals=integrals, filetag=filetag)

comm.Barrier()
options = {
    "dt": 0.005,
    "n_eql": 10,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 100,
    "n_prop_steps": 50,
    "n_walkers": nwalkers,
    "seed": 98,
    "walker_type": "uhf",
    # "trial": "uhf",
    "save_walkers": False,
}

ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI = (
    mpi_jax._prep_afqmc(options, filetag=filetag)
)

if run_cpmc:
    if verbose: print(f'\n# Using CPMC propagator...')
    prop = propagation.propagator_cpmc(
        dt=options["dt"],
        n_walkers=options["n_walkers"],
    )

trial = wavefunctions.ghf_cpmc(n_sites, n_elec)
wave_data["mo_coeff"] = ghf_coeff[:, : n_elec[0] + n_elec[1]]
wave_data["rdm1"] = jnp.array([ghf_rdm1[:n_sites, :n_sites], ghf_rdm1[n_sites:, n_sites:]])
ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
ham_data = ham.build_propagation_intermediates(ham_data, prop, trial, wave_data)
ham_data["u"] = U

e_afqmc, err_afqmc = driver.afqmc(
    ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI
)