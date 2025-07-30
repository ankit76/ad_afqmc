import sys
sys.path.append('../')

import numpy as np
import scipy as sp
import jax.numpy as jnp

from pyscf import gto, scf, ao2mo

from ad_afqmc import (
    lattices,
    driver,
    mpi_jax,
    propagation,
    wavefunctions,
    pyscf_interface,
    io_utils,
)

from ad_afqmc.legacy import driver as driver_legacy
from ad_afqmc.legacy import mpi_jax as mpi_jax_legacy
from ad_afqmc.legacy import propagation as propagation_legacy
from ad_afqmc.legacy import wavefunctions as wavefunctions_legacy
from ad_afqmc.legacy import pyscf_interface as pyscf_interface_legacy

np.set_printoptions(precision=5, suppress=True)

# -----------------------------------------------------------------------------
def run_uhf_cpmc_legacy(umf, options, integrals, tmpdir):
    n_elec = umf.mol.nelec
    n_sites = umf.mo_coeff[0].shape[0]
    n_walkers = options["n_walkers"]
    pyscf_interface_legacy.prep_afqmc(
        umf, basis_coeff=np.eye(n_sites), integrals=integrals, tmpdir=tmpdir
    )
    ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI = (
        mpi_jax_legacy._prep_afqmc(options, tmpdir=tmpdir)
    )
    trial = wavefunctions_legacy.uhf_cpmc(n_sites, n_elec)
    prop = propagation_legacy.propagator_cpmc(
        dt=options["dt"],
        n_walkers=options["n_walkers"],
    )

    init_walkers = [
        jnp.array([umf.mo_coeff[0][:, : n_elec[0]]] * n_walkers),
        jnp.array([umf.mo_coeff[1][:, : n_elec[1]]] * n_walkers),
    ]
    wave_data["mo_coeff"] = [init_walkers[0][0], init_walkers[1][0]]
    wave_data["rdm1"] = [
        wave_data["mo_coeff"][0] @ wave_data["mo_coeff"][0].T,
        wave_data["mo_coeff"][1] @ wave_data["mo_coeff"][1].T,
    ]
    ham_data["u"] = integrals["u"]

    e_afqmc, err_afqmc = driver_legacy.afqmc(
        ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI,
        tmpdir=tmpdir, init_walkers=init_walkers
    )
    return e_afqmc, err_afqmc

def run_ghf_cpmc(gmf, options, integrals, tmpdir):
    n_elec = gmf.mol.nelec
    nocc = sum(n_elec)
    n_sites = gmf.mo_coeff.shape[0] // 2
    n_walkers = options["n_walkers"]
    pyscf_interface.prep_afqmc_ghf(
        gmf, basis_coeff=np.eye(2*n_sites), integrals=integrals, tmpdir=tmpdir
    )
    ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI = (
        mpi_jax._prep_afqmc(options, tmpdir=tmpdir)
    )
    trial = wavefunctions.ghf_cpmc(n_sites, n_elec)
    prop = propagation.propagator_cpmc_generalized(
        dt=options["dt"],
        n_walkers=options["n_walkers"],
    )

    init_walkers = jnp.array([gmf.mo_coeff[:, : nocc]] * n_walkers)
    wave_data["mo_coeff"] = init_walkers[0]
    wave_data["rdm1"] = wave_data["mo_coeff"] @ wave_data["mo_coeff"].T
    ham_data["u"] = integrals["u"]

    e_afqmc, err_afqmc = driver.afqmc(
        ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI,
        tmpdir=tmpdir, init_walkers=init_walkers
    )
    return e_afqmc, err_afqmc


if __name__ == '__main__':
    U = 12.0
    nup, ndown = 8, 8
    n_elec = (nup, ndown)
    nx, ny = 4, 4
    nwalkers = 50
    bc = 'xc'

    # For saving.
    tmpdir = f'./test_ghf_cpmc'
    io_utils.check_dir(tmpdir)

    # QMC options.
    options = {
        "dt": 0.005,
        "n_eql": 50,
        "n_ene_blocks_eql": 1,
        "n_sr_blocks_eql": 1,
        "n_ene_blocks": 1,
        "n_sr_blocks": 5,
        "n_blocks": 40,
        "n_prop_steps": 50,
        "n_walkers": nwalkers,
        "seed": 98,
        "walker_type": "generalized",
        "trial": "ghf",
        "save_walkers": False,
    }

    # -------------------------------------------------------------------------
    # Lattice.
    lattice = lattices.triangular_grid(nx, ny, boundary=bc)
    n_sites = lattice.n_sites
    nocc = sum(n_elec)

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Make dummy molecule.
    mol = gto.Mole()
    mol.nelectron = nocc
    mol.incore_anyway = True
    mol.spin = abs(n_elec[0] - n_elec[1])
    mol.build()

    # Prep trial.
    umf = scf.UHF(mol)
    umf.get_hcore = lambda *args: integrals["h1"]
    umf.get_ovlp = lambda *args: np.eye(n_sites)
    umf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
    umf.mo_coeff = np.array([evecs_h1, evecs_h1])

    gmf = scf.GHF(mol)
    gmf.get_hcore = lambda *args: integrals_g["h1"]
    gmf.get_ovlp = lambda *args: np.eye(2*n_sites)
    gmf._eri = ao2mo.restore(8, integrals_g["h2"], n_sites)
    gmf.mo_coeff = np.zeros((2*n_sites, 2*n_sites))

    for i in range(n_sites):
        gmf.mo_coeff[: n_sites, 2*i] = evecs_h1[:, i]
        gmf.mo_coeff[n_sites :, 2*i+1] = evecs_h1[:, i]
    
    options["trial"] = "uhf"
    options["walker_type"] = "uhf"
    options["ad_mode"] = None
    e_qmc_ref, err_qmc_ref = run_uhf_cpmc_legacy(umf, options, integrals, tmpdir)
    
    options["trial"] = "ghf"
    options["walker_type"] = "generalized"
    options["ad_mode"] = "mixed"
    e_qmc, err_qmc = run_ghf_cpmc(gmf, options, integrals_g, tmpdir)
    
    np.testing.assert_allclose(e_qmc, e_qmc_ref)
    np.testing.assert_allclose(err_qmc, err_qmc_ref)
