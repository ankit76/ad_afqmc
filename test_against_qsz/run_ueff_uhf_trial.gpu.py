import os
import sys
import h5py
import argparse
import numpy as np
import scipy as sp

import jax
jax.config.update("jax_debug_nans", True)

from jax import vmap, jit, numpy as jnp, random, lax, jvp, scipy as jsp
from pyscf import fci, gto, scf, mp, ao2mo
from io_tools import check_dir

module_path = os.path.abspath(os.path.join("/projects/bcdd/shufay/ad_afqmc"))
if module_path not in sys.path:
    sys.path.append(module_path)

from ad_afqmc import config
config.afqmc_config['use_gpu'] = True

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

import ad_afqmc
print(ad_afqmc.__file__)

np.set_printoptions(precision=5, suppress=True)

rank = mpi_jax.rank
comm = mpi_jax.comm

def save(fname, mf, dm):
    print(f'\n# Saving UHF trial to {fname}')

    with h5py.File(f'{fname}.h5', 'w') as h5:
        h5.create_dataset('dm', data=dm)
        h5.create_dataset('mo_coeff', data=mf.mo_coeff)
        h5.create_dataset('mo_energy', data=mf.mo_energy)
        h5.create_dataset('mo_occ', data=mf.mo_occ)
        h5.create_dataset('ehf', data=mf.e_tot)

# -----------------------------------------------------------------------------
# Parser.
parser = argparse.ArgumentParser()

# Required.
parser.add_argument('--U', type=float, required=True)
parser.add_argument('--Ueff', type=float, required=True)
parser.add_argument('--nup', type=int, required=True)
parser.add_argument('--ndown', type=int, required=True)
parser.add_argument('--nx', type=int, required=True)
parser.add_argument('--ny', type=int, required=True)
parser.add_argument('--nwalkers', type=int, required=True)

# Optional.
parser.add_argument('--bc', type=str, required=False, nargs='?', default='pbc')
parser.add_argument('--dt', type=float, required=False, nargs='?', default=0.005)
parser.add_argument('--n_eql', type=int, required=False, nargs='?', default=300)
parser.add_argument('--n_blocks', type=int, required=False, nargs='?', default=400)
parser.add_argument('--run_cpmc', type=int, required=False, nargs='?', default=0)
parser.add_argument('--set_e_estimate', type=int, required=False, nargs='?', default=0)
parser.add_argument('--pin_type', type=str, required=False, nargs='?', default='fm')
parser.add_argument('--v', type=float, required=False, nargs='?', default=0.25)
parser.add_argument('--verbose', type=int, required=False, nargs='?', default=0)

args = parser.parse_args()

U = args.U
Ueff = args.Ueff
nup = args.nup
ndown = args.ndown
nx = args.nx
ny = args.ny
nwalkers = args.nwalkers

bc = args.bc
dt = args.dt
n_eql = args.n_eql
n_blocks = args.n_blocks
run_cpmc = args.run_cpmc
set_e_estimate = args.set_e_estimate
pin_type = args.pin_type
v = args.v
verbose = args.verbose if rank == 0 else 0

if verbose: 
    print(f'\n# Ueff: {Ueff}')
    print(f'# Boundary condition: {bc}')

# For saving.
tmpdir = f'/projects/bcdd/shufay/hubbard_tri/bcs/test_against_qsz/uhf/{nx}x{ny}/{bc}/dt={dt}/U={U}/Ueff={Ueff}/pin={pin_type}/'
jobid = ''
try: jobid = '.' + os.environ["SLURM_JOB_ID"]
except: pass

# -----------------------------------------------------------------------------
# Settings.
open_x = False
if bc == 'open_x': open_x = True
lattice = lattices.two_dimensional_grid(nx, ny, open_x=open_x)
n_sites = lattice.n_sites
n_elec = (nup, ndown)
nocc = sum(n_elec)
filling = nocc / (2*n_sites)
density = nocc / n_sites

if verbose: 
    print(f'\n# Filling factor = {filling}')
    print(f'# Density = {density}')

# -----------------------------------------------------------------------------
# Integrals.
integrals = {}
integrals["h0"] = 0.0

# Nearest-neighbour hopping.
h1 = -1.0 * lattice.create_adjacency_matrix()

# Pinning field.
v1a = np.zeros(n_sites)
v1b = np.zeros(n_sites)

# Assuming the y-direction is the short one with PBC.
for iy in range(lattice.l_y):
    val = (-1)**iy * v
    site_num_left = iy * lattice.l_x + 0
    site_num_right = iy * lattice.l_x + (lattice.l_x - 1)
    v1a[site_num_left] = val
    v1b[site_num_left] = -val

    if pin_type == 'fm':
        v1a[site_num_right] = val
        v1b[site_num_right] = -val

    elif pin_type == 'afm':
        v1a[site_num_right] = -val
        v1b[site_num_right] = val

integrals["h1"] = np.array([h1 + np.diag(v1a), h1 + np.diag(v1b)])

h2 = np.zeros((n_sites, n_sites, n_sites, n_sites))
for i in range(n_sites): h2[i, i, i, i] = U
integrals["h2"] = ao2mo.restore(8, h2, n_sites)

# Integrals for UHF trial.
h2_hf = np.zeros((n_sites, n_sites, n_sites, n_sites))
for i in range(n_sites): h2_hf[i, i, i, i] = Ueff
h2_hf = ao2mo.restore(8, h2_hf, n_sites)

# -----------------------------------------------------------------------------
# Make dummy molecule.
mol = gto.Mole()
mol.nelectron = nocc
mol.incore_anyway = True
mol.spin = abs(n_elec[0] - n_elec[1])
mol.verbose = verbose
mol.build()

# UHF trial.
seed = 262
np.random.seed(seed)
if verbose: print(f'np.random.seed = {seed}')

umf = scf.UHF(mol)
umf.get_hcore = lambda *args: integrals["h1"]
umf.get_ovlp = lambda *args: np.eye(n_sites)
umf._eri = h2_hf

dm_init = 0.0 * umf.init_guess_by_1e()
for ix in range(lattice.l_x):
    for iy in range(lattice.l_y):
        site_num = iy * lattice.l_x + ix
        if (ix + iy) % 2 == 0: dm_init[0, site_num, site_num] = 1.0
        else: dm_init[1, site_num, site_num] = 1.0

umf.max_cycle = 1000
umf.level_shift = 0.5
umf.kernel(dm_init)
mo1 = umf.stability()[0]
umf = umf.newton().run(mo1, umf.mo_occ)
mo1 = umf.stability()[0]
umf = umf.newton().run(mo1, umf.mo_occ)
mo1 = umf.stability()[0]
umf = umf.newton().run(mo1, umf.mo_occ)
mo1 = umf.stability()

# -----------------------------------------------------------------------------
# Prep AFQMC.
if rank == 0:
    pyscf_interface.prep_afqmc(
            umf, basis_coeff=np.eye(n_sites), integrals=integrals, tmpdir=tmpdir)

comm.Barrier()
options = {
    "dt": dt,
    "n_eql": n_eql,
    "n_ene_blocks_eql": 1,
    "n_sr_blocks_eql": 5,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": n_blocks,
    "n_prop_steps": 50,
    "n_walkers": nwalkers,
    "seed": 98,
    "walker_type": "uhf",
    "trial": "uhf",
    "save_walkers": False,
    "ad_mode": "mixed",
}

ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI = (
    mpi_jax._prep_afqmc(options, tmpdir=tmpdir)
)

trial = wavefunctions.uhf_cpmc(n_sites, n_elec)
wave_data["mo_coeff"] = jnp.array([umf.mo_coeff[0][:, :n_elec[0]], 
                                   umf.mo_coeff[1][:, :n_elec[1]]])
wave_data["rdm1"] = jnp.array(umf.make_rdm1())
ham_data["u"] = U

if set_e_estimate:
    # GHF energy for `e_estimate`.
    dm_init = 0.0 * umf.init_guess_by_1e()

    for ix in range(lattice.l_x):
        for iy in range(lattice.l_y):
            site_num = iy * lattice.l_x + ix
            if (ix + iy) % 2 == 0: dm_init[0, site_num, site_num] = 1.0
            else: dm_init[1, site_num, site_num] = 1.0

    gmf = scf.GHF(mol)
    gmf.get_hcore = lambda *args: sp.linalg.block_diag(*integrals["h1"])
    gmf.get_ovlp = lambda *args: np.eye(2 * n_sites)
    gmf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
    dm_init = sp.linalg.block_diag(dm_init[0], dm_init[1])
    gmf.kernel(dm_init)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)
    mo1 = gmf.stability(external=True)

    ham_data["e_estimate"] = jnp.float64(gmf.e_tot)
    #ham_data["e_estimate"] = jnp.float64(-13.)
    if rank == 0: print(f'\n# Setting `e_estimate` = {ham_data["e_estimate"]}')

# -----------------------------------------------------------------------------
# Run AFQMC.
if run_cpmc:
    if verbose: print(f'\n# Using CPMC propagator...')
    #prop = propagation.propagator_cpmc_slow(
    prop = propagation.propagator_cpmc(
        dt=options["dt"],
        n_walkers=options["n_walkers"],
    )

e_afqmc, err_afqmc = driver.afqmc(
    ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI,
    tmpdir=tmpdir
)
