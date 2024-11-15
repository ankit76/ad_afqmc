import h5py
import argparse
import numpy as np
import scipy as sp

from pyscf import fci, gto, scf, mp, ao2mo

from ad_afqmc import (
    spin_utils,
    lattices,
    hf_guess
)

from io_tools import read_real_array

parser = argparse.ArgumentParser()

# Required.
parser.add_argument('--U', type=float, required=True)
parser.add_argument('--nup', type=int, required=True)
parser.add_argument('--ndown', type=int, required=True)
parser.add_argument('--nx', type=int, required=True)
parser.add_argument('--ny', type=int, required=True)

# Optional.
parser.add_argument('--open_x', type=int, required=False, nargs='?', default=0)
parser.add_argument('--load_dm0', type=int, required=False, nargs='?', default=0)
parser.add_argument('--in_dir', type=str, required=False, nargs='?', default='./')
parser.add_argument('--dm0_filetag', type=str, required=False, nargs='?', default='')
parser.add_argument('--verbose', type=int, required=False, nargs='?', default=0)

args = parser.parse_args()

U = args.U
nup = args.nup
ndown = args.ndown
nx = args.nx
ny = args.ny

open_x = args.open_x
load_dm0 = args.load_dm0
in_dir = args.in_dir
dm0_filetag = args.dm0_filetag
verbose = args.verbose

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
# make dummy molecule
mol = gto.Mole()
mol.nelectron = sum(n_elec)
mol.incore_anyway = True
mol.spin = abs(n_elec[0] - n_elec[1])
mol.verbose = verbose
mol.build()

# -----------------------------------------------------------------------------
# ghf
gmf = scf.GHF(mol)
gmf.get_hcore = lambda *args: sp.linalg.block_diag(integrals["h1"], integrals["h1"])
gmf.get_ovlp = lambda *args: np.eye(2 * n_sites)
gmf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
dm_init = None
ao_ovlp = np.eye(n_sites)

if load_dm0:
    dm0_fname = f'{in_dir}{dm0_filetag}.h5'
    if verbose: print(f'\n# Loading dm0 from {dm0_fname}...')
    h5 = h5py.File(dm0_fname)
    dm_init = read_real_array(h5, 'ghf_rdm1')
    h5.close()

else:
    # costruct Neel guess.
    occ = np.ones(2*n_sites)
    occ[:nocc] = 1
    psi0 = hf_guess.get_ghf_neel_guess(lattice)
    dm_init = psi0 @ np.diag(occ) @ psi0.T.conj()
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

# Save.
ehf = gmf.e_tot
rdm1 = gmf.make_rdm1()

hf_dir = './ghf/'
hf_subdir = ''
hf_fname = f'{hf_dir}{hf_subdir}lattice={nx}x{ny}_nelec={n_elec}_U={U}.h5'

with h5py.File(hf_fname, 'w') as h5:
    h5.create_dataset('ghf_coeff', data=gmf_coeff)
    h5.create_dataset('ghf_rdm1', data=rdm1)
    h5.create_dataset('ghf_etot', data=ehf)
    h5.create_dataset('epsilon0', data=epsilon0)
    h5.create_dataset('mu', data=mu)
    h5.create_dataset('spin_axis', data=spin_axis)

