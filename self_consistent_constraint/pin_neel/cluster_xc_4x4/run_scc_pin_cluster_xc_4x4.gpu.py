import os
import sys
import h5py
import argparse
import numpy as np

sys.path.append('../')
module_path = os.path.abspath(os.path.join("/projects/bcdd/shufay/ad_afqmc"))
if module_path not in sys.path:
    sys.path.append(module_path)

from ad_afqmc import config
config.afqmc_config['use_gpu'] = True

from scc_pin import *
from io_tools import check_dir

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
parser.add_argument('--nwalkers', type=int, required=True)

# Optional.
parser.add_argument('--dt', type=float, required=False, nargs='?', default=0.005)
parser.add_argument('--n_eql', type=int, required=False, nargs='?', default=300)
parser.add_argument('--n_blocks', type=int, required=False, nargs='?', default=400)
parser.add_argument('--run_cpmc', type=int, required=False, nargs='?', default=0)
parser.add_argument('--set_e_estimate', type=int, required=False, nargs='?', default=0)
parser.add_argument('--init_trial', type=str, required=False, nargs='?', default='ghf')
parser.add_argument('--Ueff', type=float, required=False, nargs='?')
parser.add_argument('--v', type=float, required=False, nargs='?', default=0.5)
parser.add_argument('--approx_dm_pure', type=int, required=False, nargs='?', default=0)
parser.add_argument('--tol_delta_min', type=float, required=False, nargs='?', default=1e-3)
parser.add_argument('--verbose', type=int, required=False, nargs='?', default=0)

args = parser.parse_args()

U = args.U
nup = args.nup
ndown = args.ndown
nwalkers = args.nwalkers

dt = args.dt
n_eql = args.n_eql
n_blocks = args.n_blocks
run_cpmc = args.run_cpmc
set_e_estimate = args.set_e_estimate
init_trial = args.init_trial
Ueff = args.Ueff
v = args.v
tol_delta_min = args.tol_delta_min
verbose = args.verbose if rank == 0 else 0

approx_dm_pure = args.approx_dm_pure
dm_tag = 'dm_mixed'
if approx_dm_pure: dm_tag = 'approx_dm_pure'

# -----------------------------------------------------------------------------
n_elec = (nup, ndown)

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
    "walker_type": "generalized",
    "trial": "ghf",
    "save_walkers": False,
    "ad_mode": "mixed",
}

nx, ny = 4, 4
bc = 'xc'

# For saving.
tmpdir = f'/projects/bcdd/shufay/hubbard_tri/self_consistent_constraint/pin_neel/cluster_{bc}_{nx}x{ny}/{dm_tag}/U={U}/{init_trial}_trial/v={v}/'
if rank == 0: check_dir(tmpdir)
jobid = ''
try: jobid = '.' + os.environ["SLURM_JOB_ID"]
except: pass
filename = f'scc{jobid}'

sites_1 = [4, 7, 12, 15] # A sublattice
sites_2 = [0, 3, 8, 11] # B sublattice
sites_3 = [] # C sublattice
pin_sites = [sites_1, sites_2, sites_3]

idx_up = [2, 4, 7, 10, 12, 15]

conv = run_scc(
        comm, U, nx, ny, n_elec, pin_sites, idx_up, options, Ueff=Ueff, v=v, 
        bc=bc, run_cpmc=run_cpmc, set_e_estimate=set_e_estimate,
        init_trial=init_trial, tmpdir=tmpdir, filename=filename, 
        approx_dm_pure=approx_dm_pure, tol_delta_min=tol_delta_min,
        do_plot_density=True, save_dm_hf=True, verbose=verbose)

