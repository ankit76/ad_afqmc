import os
import sys
import h5py
import argparse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from jax import vmap, jit, numpy as jnp, random, lax, jvp, scipy as jsp
from pyscf import fci, gto, scf, mcscf, ao2mo
from pyscf.fci import cistring
from io_tools import check_dir

module_path = os.path.abspath(os.path.join("/burg/ccce/users/su2254/ad_afqmc"))
if module_path not in sys.path:
    sys.path.append(module_path)

from ad_afqmc import config
config.afqmc_config['use_mpi'] = True

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
parser.add_argument('--bc', type=str, required=False, nargs='?', default='pbc')
parser.add_argument('--run_cpmc', type=int, required=False, nargs='?', default=0)
parser.add_argument('--verbose', type=int, required=False, nargs='?', default=0)

args = parser.parse_args()

U = args.U
nup = args.nup
ndown = args.ndown
nx = args.nx
ny = args.ny
nwalkers = args.nwalkers

bc = args.bc
run_cpmc = args.run_cpmc
verbose = args.verbose if rank == 0 else 0
if verbose: print(f'\n# Boundary condition: {bc}')

# For saving.
tmpdir = f'/burg/ccce/users/su2254/ad_afqmc/hubbard_tri/trs/{nx}x{ny}/{bc}/U={U}'
jobid = ''
try: jobid = '.' + os.environ["SLURM_JOB_ID"]
except: pass

# -----------------------------------------------------------------------------
# Settings.
lattice = lattices.triangular_grid(nx, ny, boundary_condition=bc)
n_sites = lattice.n_sites
n_elec = (nup, ndown)
nocc = sum(n_elec)
filling = nocc / (2*n_sites)
if verbose: print(f'\n# Filling factor = {filling}')

# -----------------------------------------------------------------------------
# Integrals.
integrals = {}
integrals["h0"] = 0.0

h1 = -1.0 * lattice.create_adjacency_matrix()

if bc == 'apbc': # Antiperiodic boundary conditions.
    boundary_pairs = lattice.get_boundary_pairs()

    for i, j in boundary_pairs:
        assert h1[i, j] == h1[j, i] == -1.
        h1[i, j] *= -1.
        h1[j, i] *= -1.

integrals["h1"] = h1

h2 = np.zeros((n_sites, n_sites, n_sites, n_sites))
for i in range(n_sites): h2[i, i, i, i] = U
integrals["h2"] = ao2mo.restore(8, h2, n_sites)

# Diagonalize h1.
evals_h1, evecs_h1 = np.linalg.eigh(integrals["h1"])

# -----------------------------------------------------------------------------
# make dummy molecule
mol = gto.Mole()
mol.nelectron = sum(n_elec)
mol.incore_anyway = True
mol.spin = abs(n_elec[0] - n_elec[1])
mol.nelec = n_elec
mol.nao = n_sites
mol.verbose = verbose
mol.build()

mf = scf.RHF(mol)
mf.get_hcore = lambda *args: integrals["h1"]
mf.get_ovlp = lambda *args: np.eye(n_sites)
mf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
mf.kernel()

np.random.seed(1)
umf = scf.UHF(mol)
umf.get_hcore = lambda *args: integrals["h1"]
umf.get_ovlp = lambda *args: np.eye(n_sites)
umf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
#dm_init = 1.0 * umf.init_guess_by_1e()
#dm_init += 1.0 * np.random.randn(*dm_init.shape)

dm_init = 0.0 * umf.init_guess_by_1e()
for i in range(lattice.l_x):
    for j in range(lattice.l_y):
        site_num = i * lattice.l_y + j
        if (i + j) % 2 == 0: dm_init[0, site_num, site_num] = 1.0
        else: dm_init[1, site_num, site_num] = 1.0

umf.kernel(dm_init)
mo1 = umf.stability()[0]
umf = umf.newton().run(mo1, umf.mo_occ)
mo1 = umf.stability()[0]
umf = umf.newton().run(mo1, umf.mo_occ)
mo1 = umf.stability()[0]
umf = umf.newton().run(mo1, umf.mo_occ)
umf.stability()

spin_rdm1 = umf.make_rdm1()
sz = (spin_rdm1[0] - spin_rdm1[1]).diagonal() / 2
charge = (spin_rdm1[0] + spin_rdm1[1]).diagonal()
sx = np.zeros_like(sz)

# -----------------------------------------------------------------------------
# Plots.
# Draw the triangular lattice with arrows showing the spin.
fig, ax = plt.subplots()
coords = np.array([lattice.get_site_coordinate(site) for site in lattice.sites])
arrow_size = 0.7
for i in range(n_sites):
    site = coords[i]
    x, y = site
    ax.plot(x, y, "ko", markersize=1)
    ax.arrow(
        x,
        y,
        arrow_size * sx.ravel()[i],
        arrow_size * sz.ravel()[i],
        head_width=0.1,
    )
    # Draw lines to nearest neighbors.
    nn = np.nonzero(lattice.create_adjacency_matrix()[i])[0]
    for n in nn:
        site_n = coords[n]
        x_n, y_n = site_n
        ax.plot(
            [x, x_n],
            [y, y_n],
            "k-",
            lw=0.1,
        )
ax.set_aspect("equal")
plt.savefig(f'{tmpdir}/uhf_soln.png')

# Orbital energies.
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
y = umf.mo_energy
x = np.arange(len(y[0]))
ax.scatter(x, y[0], marker='_', color='r', label='up')
ax.scatter(x, y[1], marker='_', color='k', label='dn')
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig(f'{tmpdir}/mo_energy.png')

# -----------------------------------------------------------------------------
# Prep AFQMC.
if rank == 0:
    pyscf_interface.prep_afqmc(
            umf, basis_coeff=np.eye(n_sites), integrals=integrals, tmpdir=tmpdir)

comm.Barrier()
options = {
    "dt": 0.005,
    "n_eql": 100,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 200,
    "n_prop_steps": 50,
    "n_walkers": nwalkers,
    "seed": 98,
    "walker_type": "uhf",
    "trial": "uhf",
    "save_walkers": False,
}

ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI = (
    mpi_jax._prep_afqmc(options, tmpdir)
)

# -----------------------------------------------------------------------------
# Prep trial.
trial_1 = wavefunctions.uhf_cpmc(n_sites, n_elec)
trial_2 = wavefunctions.uhf_cpmc(n_sites, n_elec)
trial = wavefunctions.sum_state(n_sites, n_elec, (trial_1, trial_2))

wave_data_0 = wave_data.copy()
wave_data_0["mo_coeff"] = [
    umf.mo_coeff[0][:, : n_elec[0]],
    umf.mo_coeff[1][:, : n_elec[1]],
]

wave_data_1 = wave_data.copy()
wave_data_1["mo_coeff"] = [
    umf.mo_coeff[1][:, : n_elec[0]],
    umf.mo_coeff[0][:, : n_elec[1]],
]

wave_data["coeffs"] = jnp.array([1 / 2**0.5, 1 / 2**0.5])
wave_data["0"] = wave_data_0
wave_data["1"] = wave_data_1

# -----------------------------------------------------------------------------
# Run AFQMC.
if run_cpmc:
    if verbose: print(f'\n# Using CPMC propagator...')
    prop = propagation.propagator_cpmc_slow(
        dt=options["dt"],
        n_walkers=options["n_walkers"],
    )

ham_data["u"] = U

e_afqmc, err_afqmc = driver.afqmc(
    ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI,
    tmpdir=tmpdir
)

