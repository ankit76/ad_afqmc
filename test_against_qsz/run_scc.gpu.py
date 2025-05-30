import os
import sys
import h5py
import argparse
import numpy as np

from io_tools import check_dir

module_path = os.path.abspath(os.path.join("/projects/bcdd/shufay/ad_afqmc"))
if module_path not in sys.path:
    sys.path.append(module_path)

from ad_afqmc import config
config.afqmc_config['use_gpu'] = True

from scc import *

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

# -----------------------------------------------------------------------------
rank = mpi_jax.rank
comm = mpi_jax.comm

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
parser.add_argument('--dt', type=float, required=False, nargs='?', default=0.005)
parser.add_argument('--n_eql', type=int, required=False, nargs='?', default=300)
parser.add_argument('--n_blocks', type=int, required=False, nargs='?', default=400)
parser.add_argument('--run_cpmc', type=int, required=False, nargs='?', default=0)
parser.add_argument('--set_e_estimate', type=int, required=False, nargs='?', default=0)
parser.add_argument('--init_trial', type=str, required=False, nargs='?', default='fe_single_occ')
parser.add_argument('--Ueff', type=float, required=False, nargs='?')
parser.add_argument('--pin_type', type=str, required=False, nargs='?', default='fm')
parser.add_argument('--v', type=float, required=False, nargs='?', default=0.25)
parser.add_argument('--proj_trs', type=int, required=False, nargs='?', default=0)
parser.add_argument('--approx_dm_pure', type=int, required=False, nargs='?', default=0)
parser.add_argument('--tol_delta_min', type=float, required=False, nargs='?', default=1e-3)
parser.add_argument('--verbose', type=int, required=False, nargs='?', default=0)

args = parser.parse_args()

U = args.U
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
init_trial = args.init_trial
Ueff = args.Ueff
pin_type = args.pin_type
v = args.v
tol_delta_min = args.tol_delta_min
verbose = args.verbose if rank == 0 else 0

proj_trs = args.proj_trs
trs_tag = ''
if proj_trs: trs_tag = 'trs'

approx_dm_pure = args.approx_dm_pure
dm_tag = 'dm_mixed'
if approx_dm_pure: dm_tag = 'approx_dm_pure'

if verbose: print(f'\n# Boundary condition: {bc}')

# For saving.
tmpdir = f'/projects/bcdd/shufay/hubbard_tri/self_consistent_constraint/test_against_qsz/{trs_tag}/{dm_tag}/{nx}x{ny}/{bc}/U={U}/pin={pin_type}/{init_trial}_trial/'
if rank == 0: check_dir(tmpdir)
jobid = ''
try: jobid = '.' + os.environ["SLURM_JOB_ID"]
except: pass
filename = f'scc{jobid}'

# -----------------------------------------------------------------------------
n_elec = (nup, ndown)

options = {
    "dt": dt,
    "n_eql": n_eql,
    "n_ene_blocks_eql": 1,
    "n_sr_blocks_eql": 1,
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

conv = run_scc(
        comm, U, nx, ny, n_elec, options, Ueff=Ueff, pin_type=pin_type, v=v, 
        bc=bc, run_cpmc=run_cpmc, proj_trs=proj_trs, 
        set_e_estimate=set_e_estimate, init_trial=init_trial, tmpdir=tmpdir, 
        filename=filename, approx_dm_pure=approx_dm_pure, tol_delta_min=tol_delta_min,
        do_plot_density=True, save_dm_hf=True, verbose=verbose)

