import os
import sys
import h5py
import argparse
import numpy as np
import scipy as sp

from jax import vmap, jit, numpy as jnp, random, lax, jvp, scipy as jsp
from pyscf import fci, gto, scf, mcscf, ao2mo
from pyscf.fci import cistring
from io_tools import check_dir

module_path = os.path.abspath(os.path.join("/projects/bcdd/shufay/ad_afqmc"))
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
)

np.set_printoptions(precision=5, suppress=True)

rank = mpi_jax.rank
comm = mpi_jax.comm


def large_ci(ci, norb, nelec, tol=0.1, return_strs=True):
    '''
    Search for the largest CI coefficients
    '''
    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    assert ci.size == na * nb

    ci = ci.reshape(na, nb)
    addra, addrb = np.where(abs(ci) > tol)

    if addra.size == 0:
        # No large CI coefficient > tol, search for the largest coefficient
        addra, addrb = np.unravel_index(np.argmax(abs(ci)), ci.shape)
        addra = numpy.asarray([addra])
        addrb = numpy.asarray([addrb])

    strsa = cistring.addrs2str(norb, neleca, addra)
    strsb = cistring.addrs2str(norb, nelecb, addrb)

    if return_strs:
        strsa = [bin(x) for x in strsa]
        strsb = [bin(x) for x in strsb]
        return list(zip(ci[addra,addrb], strsa, strsb))

    else:
        occslsta = cistring._strs2occslst(strsa, norb)
        occslstb = cistring._strs2occslst(strsb, norb)
        return list(zip(ci[addra,addrb], occslsta, occslstb))

def get_fci_state(ci_coeffs, norb, nelec, ndets=None, tol=1e-4):
    if ndets is None: ndets = int(ci_coeffs.size)
    coeffs, occ_a, occ_b = zip(
        *large_ci(ci_coeffs, norb, nelec, tol=tol, return_strs=False)
    )
    coeffs, occ_a, occ_b = zip(
        *sorted(zip(coeffs, occ_a, occ_b), key=lambda x: -abs(x[0]))
    )
    state = {}
    for i in range(min(ndets, len(coeffs))):
        det = [[0 for _ in range(norb)], [0 for _ in range(norb)]]
        for j in range(nelec[0]):
            det[0][occ_a[i][j]] = 1
        for j in range(nelec[1]):
            det[1][occ_b[i][j]] = 1
        state[tuple(map(tuple, det))] = coeffs[i]
    return state

def create_msd_from_casci(casci, wave_data, copy=True, update=False, verbose=False):
    # Create MSD state.
    mo_coeff = casci.mo_coeff
    nbsf = mo_coeff.shape[0]
    ncore = casci.ncore
    ncas = casci.ncas
    nextern = nbsf - ncore - ncas
    nelecas = casci.nelecas
    ndet = len(fci_state)

    wave_data_arr = []
    coeffs = []
    if verbose: print(f'\n# Note that determinant coefficients are unnormalized!\n')

    for det in fci_state:
        nocc_a = np.array(ncore*(1,) + det[0] + nextern*(0,))
        nocc_b = np.array(ncore*(1,) + det[1] + nextern*(0,))
        if verbose: print(f'# {det} --> {nocc_a}, {nocc_b} : {fci_state[det]}')

        wave_data_i = wave_data.copy()
        wave_data_i['mo_coeff'] = [mo_coeff[:, nocc_a>0], mo_coeff[:, nocc_b>0]]
        del wave_data_i['rdm1']
        wave_data_arr.append(wave_data_i)
        coeffs.append(fci_state[det])

    if copy: wave_data = wave_data.copy()
    for i in range(ndet): wave_data[f'{i}'] = wave_data_arr[i]

    # Check if coeffs are normalized.
    coeffs = np.array(coeffs)
    norm = np.sqrt(np.sum(coeffs**2))
    try: np.testing.assert_allclose(norm, 1.)
    except: coeffs /= norm
    wave_data['coeffs'] = jnp.array(coeffs)

    if update:
        mo_coeff = wave_data['0']['mo_coeff'].copy()
        rdm1 = [mo_coeff[0] @ mo_coeff[0].T.conj(), mo_coeff[1] @ mo_coeff[1].T.conj()]
        wave_data['mo_coeff'] = mo_coeff
        wave_data['rdm1'] = rdm1

    return wave_data

# -----------------------------------------------------------------------------
# Parser.
parser = argparse.ArgumentParser()

# Required.
parser.add_argument('--U', type=float, required=True)
parser.add_argument('--nup', type=int, required=True)
parser.add_argument('--ndown', type=int, required=True)
parser.add_argument('--nup_cas', type=int, required=True)
parser.add_argument('--ndown_cas', type=int, required=True)
parser.add_argument('--nx', type=int, required=True)
parser.add_argument('--ny', type=int, required=True)
parser.add_argument('--ncas', type=int, required=True)
parser.add_argument('--nwalkers', type=int, required=True)

# Optional.
parser.add_argument('--bc', type=str, required=False, nargs='?', default='pbc')
parser.add_argument('--run_cpmc', type=int, required=False, nargs='?', default=0)
parser.add_argument('--det_tol', type=float, required=False, nargs='?', default=0.)
parser.add_argument('--verbose', type=int, required=False, nargs='?', default=0)

args = parser.parse_args()

U = args.U
nup = args.nup
ndown = args.ndown
nup_cas = args.nup_cas
ndown_cas = args.ndown_cas
nx = args.nx
ny = args.ny
ncas = args.ncas
nwalkers = args.nwalkers

bc = args.bc
run_cpmc = args.run_cpmc
det_tol = args.det_tol
verbose = args.verbose if rank == 0 else 0
if verbose: print(f'\n# Boundary condition: {bc}')

# For saving.
tmpdir = f'/projects/bcdd/shufay/ad_afqmc/hubbard_tri/cas/{nx}x{ny}/{bc}/cas_ndet=2/U={U}'
jobid = ''
try: jobid = '.' + os.environ["SLURM_JOB_ID"]
except: pass

# -----------------------------------------------------------------------------
# Settings.
lattice = lattices.triangular_grid(nx, ny, boundary_condition=bc)
open_x = False
if bc == 'open_x': open_x = True
#lattice = lattices.triangular_grid(nx, ny, open_x=open_x)
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
# Make dummy molecule.
mol = gto.Mole()
mol.nelectron = nocc
mol.incore_anyway = True
mol.spin = abs(n_elec[0] - n_elec[1])
mol.verbose = verbose
mol.build()

# Prep AFQMC.
mf = scf.RHF(mol)
mf.get_hcore = lambda *args: integrals["h1"]
mf.get_ovlp = lambda *args: np.eye(n_sites)
mf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
mf.mo_coeff = evecs_h1

umf = scf.UHF(mol)
umf.get_hcore = lambda *args: integrals["h1"]
umf.get_ovlp = lambda *args: np.eye(n_sites)
umf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
umf.mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])

if rank == 0:
    pyscf_interface.prep_afqmc(
            umf, basis_coeff=np.eye(n_sites), integrals=integrals, tmpdir=tmpdir)

comm.Barrier()
options = {
    "dt": 0.005,
    "n_eql": 500,
    "n_ene_blocks_eql": 1,
    "n_sr_blocks_eql": 1,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 300,
    "n_prop_steps": 50,
    "n_walkers": nwalkers,
    "seed": 98,
    "walker_type": "uhf",
    "trial": "uhf",
    "save_walkers": False,
}

ham_data, ham, prop, trial, _wave_data, sampler, observable, options, MPI = (
    mpi_jax._prep_afqmc(options, tmpdir)
)

# -----------------------------------------------------------------------------
# Prep CASCI trial.
mf = scf.RHF(mol)
mf.get_hcore = lambda *args: integrals["h1"]
mf.get_ovlp = lambda *args: np.eye(n_sites)
mf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
mf.mo_coeff = evecs_h1

nelec_cas = (nup_cas, ndown_cas)
casci = mcscf.CASCI(mf, ncas, nelec_cas)
casci.fix_spin_(ss=0)
etot, ecas, casci_coeffs, mo_coeff, mo_energy = casci.kernel()

if verbose:
    print(f'\n# n_sites: {n_sites}')
    print(f'# n_elec: {n_elec}')
    print(f'# casci.ncore: {casci.ncore}')
    print(f'# casci.ncas: {casci.ncas}')
    print(f'# casci.nelecas: {casci.nelecas}')

fci_state = get_fci_state(casci_coeffs, casci.ncas, casci.nelecas, tol=det_tol)

if verbose:
    print(f'\n# CASCI determinants:')

    for det in fci_state:
        print(f'# {det}: {fci_state[det]}')

wave_data = create_msd_from_casci(casci, _wave_data, update=True, verbose=verbose)

# -----------------------------------------------------------------------------
# Run AFQMC.
if run_cpmc:
    if verbose: print(f'\n# Using CPMC propagator...')
    prop = propagation.propagator_cpmc_slow(
        dt=options["dt"],
        n_walkers=options["n_walkers"],
    )

trial_arr = [wavefunctions.uhf_cpmc(n_sites, n_elec)] * len(fci_state)
trial = wavefunctions.sum_state(n_sites, n_elec, tuple(trial_arr))
ham_data["u"] = U

e_afqmc, err_afqmc = driver.afqmc(
    ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI,
    tmpdir=tmpdir
)
