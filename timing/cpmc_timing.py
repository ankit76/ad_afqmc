import os
import sys
import time
import argparse

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
import numpy as np
import scipy as sp
from jax import numpy as jnp

module_path = os.path.abspath(os.path.join("/projects/bcdd/shufay/ad_afqmc"))
if module_path not in sys.path:
    sys.path.append(module_path)

from ad_afqmc import (
    driver,
    pyscf_interface,
    mpi_jax,
    linalg_utils,
    lattices,
    propagation,
    wavefunctions,
    hamiltonian,
)

from pyscf import fci, gto, scf, mp, ao2mo

import itertools
from functools import partial

np.set_printoptions(precision=5, suppress=True)

def run_square(filling, nsitex, seed=0, jobid=None):
    if jobid is None: jobid = ""
    np.random.seed(seed)
    
    lattice = lattices.two_dimensional_grid(nsitex, nsitex)
    n_sites = lattice.n_sites
    u = 4.0
    _n_elec = int((filling * n_sites) // 2)
    n_elec = (_n_elec, _n_elec)

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
    mol.build()

    umf = scf.UHF(mol)
    umf.get_hcore = lambda *args: integrals["h1"]
    umf.get_ovlp = lambda *args: np.eye(n_sites)
    umf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
    dm_init = 1.0 * umf.init_guess_by_1e()
    # for i in range(n_sites // 2):
    #     dm_init[0, 2 * i, 2 * i] = 1.0
    #     dm_init[1, 2 * i + 1, 2 * i + 1] = 1.0
    dm_init += 1.0 * np.random.randn(*dm_init.shape)
    umf.kernel(dm_init)
    mo1 = umf.stability()[0]
    umf = umf.newton().run(mo1, umf.mo_occ)
    mo1 = umf.stability()[0]
    umf = umf.newton().run(mo1, umf.mo_occ)
    umf.stability()

    # ad afqmc
    fcidump_filename = "FCIDUMP_chol." + jobid
    pyscf_interface.prep_afqmc(umf, mo_coeff=np.eye(n_sites), integrals=integrals,
                               filename=fcidump_filename)
    options = {
        "dt": 0.005,
        "n_eql": 5,
        "n_ene_blocks": 1,
        "n_sr_blocks": 5,
        "n_blocks": 100,
        "n_prop_steps": 50,
        "n_walkers": 50,
        "seed": 98,
        "walker_type": "uhf",
        "trial": "uhf",
        "save_walkers": False,
        "fcidump_filename": fcidump_filename
    }

    ham_data, ham, prop, trial, wave_data, observable, options = mpi_jax._prep_afqmc(
        options
    )
    
    # comment out this block for doing ph-afqmc
    prop = propagation.propagator_cpmc(
        dt=options["dt"],
        n_walkers=options["n_walkers"],
        n_prop_steps=options["n_prop_steps"],
    )
    gamma = np.arccosh(np.exp(prop.dt * u / 2))
    const = jnp.exp(-prop.dt * u / 2)
    ham_data["hs_constant"] = const * jnp.array(
        [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
    )

    start = time.time()
    e_afqmc, err_afqmc = driver.afqmc(
        ham_data, ham, prop, trial, wave_data, observable, options
    )
    elapsed = time.time() - start
    print(f'\n# time elapsed [s]: {elapsed}')
    print(f'# time elapsed [m]: {elapsed/60}')
    print(f'# time elapsed [h]: {elapsed/3600}')

def run_tri(filling, nsitex, seed=0, jobid=None):
    if jobid is None: jobid = ""
    np.random.seed(seed)
    
    lattice = lattices.triangular_grid(nsitex, nsitex)
    n_sites = lattice.n_sites
    u = 4.0
    _n_elec = int((filling * n_sites) // 2)
    n_elec = (_n_elec, _n_elec)

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
    mol.build()

    umf = scf.UHF(mol)
    umf.get_hcore = lambda *args: integrals["h1"]
    umf.get_ovlp = lambda *args: np.eye(n_sites)
    umf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
    dm_init = 1.0 * umf.init_guess_by_1e()
    # for i in range(n_sites // 2):
    #     dm_init[0, 2 * i, 2 * i] = 1.0
    #     dm_init[1, 2 * i + 1, 2 * i + 1] = 1.0
    dm_init += 1.0 * np.random.randn(*dm_init.shape)
    umf.kernel(dm_init)
    mo1 = umf.stability()[0]
    umf = umf.newton().run(mo1, umf.mo_occ)
    mo1 = umf.stability()[0]
    umf = umf.newton().run(mo1, umf.mo_occ)
    umf.stability()

    # ghf
    gmf = scf.GHF(mol)
    gmf.get_hcore = lambda *args: sp.linalg.block_diag(integrals["h1"], integrals["h1"])
    gmf.get_ovlp = lambda *args: np.eye(2 * n_sites)
    gmf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
    dm_init = sp.linalg.block_diag(dm_init[0], dm_init[1])
    dm_init += 2.0 * np.random.randn(*dm_init.shape)
    gmf.kernel(dm_init)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)

    # fci
    if sum(n_elec) < 8:
        ci = fci.FCI(mol)
        e, ci_coeffs = ci.kernel(
            h1e=integrals["h1"], eri=integrals["h2"], norb=n_sites, nelec=n_elec
        )
        print(f"fci energy: {e}")

    # ad afqmc
    fcidump_filename = "FCIDUMP_chol." + jobid
    pyscf_interface.prep_afqmc(umf, mo_coeff=np.eye(n_sites), integrals=integrals, 
                               filename=fcidump_filename)
    options = {
        "dt": 0.005,
        "n_eql": 5,
        "n_ene_blocks": 1,
        "n_sr_blocks": 5,
        "n_blocks": 100,
        "n_prop_steps": 50,
        "n_walkers": 50,
        "seed": 98,
        "walker_type": "uhf",
        "trial": "uhf",
        "save_walkers": False,
        "fcidump_filename": fcidump_filename
    }

    ham_data, ham, prop, trial, wave_data, observable, options = mpi_jax._prep_afqmc(
        options
    )

    prop = propagation.propagator_cpmc(
        dt=options["dt"],
        n_walkers=options["n_walkers"],
        n_prop_steps=options["n_prop_steps"],
    )
    trial = wavefunctions.ghf(n_sites, n_elec)
    wave_data = gmf.mo_coeff[:, : n_elec[0] + n_elec[1]]
    ham = hamiltonian.hamiltonian_ghf(n_sites, n_elec, ham_data["chol"].shape[0])
    ham_data = ham.rot_ham(ham_data, wave_data)
    gamma = np.arccosh(np.exp(prop.dt * u / 2))
    const = jnp.exp(-prop.dt * u / 2)
    ham_data["hs_constant"] = const * jnp.array(
        [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
    )

    start = time.time()
    e_afqmc, err_afqmc = driver.afqmc(
        ham_data, ham, prop, trial, wave_data, observable, options
    )
    elapsed = time.time() - start
    print(f'\n# time elapsed [s]: {elapsed}')
    print(f'# time elapsed [m]: {elapsed/60}')
    print(f'# time elapsed [h]: {elapsed/3600}')

if __name__ == '__main__':
    print(f'numpy version: {np.__version__}')
    
    # Create the parser
    my_parser = argparse.ArgumentParser()

    # Add the arguments
    # Required.
    my_parser.add_argument('--filling', required=True, type=float)
    my_parser.add_argument('--nsitex', required=True, type=int)
    my_parser.add_argument('--lattice_type', required=True, type=str)

    # Execute the parse_args() method
    args = my_parser.parse_args()
    print(args)

    # Calculation params.
    filling = args.filling
    nsitex = args.nsitex
    lattice_type = args.lattice_type

    seed = 0
    np.random.seed(seed)
    print(f'\n# numpy seed = {seed}')

    jobid = os.environ["SLURM_JOB_ID"] 

    if lattice_type == 'sq':
        run_square(filling, nsitex, seed=seed, jobid=jobid)

    elif lattice_type == 'tri':
        run_tri(filling, nsitex, seed=seed, jobid=jobid)
