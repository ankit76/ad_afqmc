import os
import numpy as np

from pyscf import gto, scf, cc
from ad_afqmc import config
config.afqmc_config["use_mpi"] = True # To run with MPI
#config.afqmc_config["use_gpu"] = True # To run on GPU
from ad_afqmc import afqmc

mol =  gto.M(atom ="""
    O        0.0000000000      0.0000000000      0.0000000000
    H        0.9562300000      0.0000000000      0.0000000000
    H       -0.2353791634      0.9268076728      0.0000000000
    """,
    basis = '6-31g',
    verbose = 3)

# RHF
mf = scf.RHF(mol)
mf.kernel()

# RCCSD 
mycc = cc.CCSD(mf)
mycc.kernel()

# afqmc @ RCCSD
af = afqmc.AFQMC(mycc)
af.nproc = 1 # Number of MPI processes
af.norb_frozen = 0 # Number of frozen orbitals
af.chol_cut = 1e-5 # Threshold for the Cholesky decomposition
af.dt = 0.005 # Time step
af.n_walkers = 50 # Number of walkers
af.n_blocks = 200 # Number of blocks
af.seed = np.random.randint(1, int(1e6)) # Seed for random numbers generation
af.n_eql = 20 # Number of equilibration steps at the beginning
af.walker_type = "restricted" # Walker type, i.e., resticted, unrestricted or generalized
af.tmpdir = "tmp" # Temporary directory
af.kernel()
