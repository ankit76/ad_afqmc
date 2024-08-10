import numpy as np
from pyscf import gto, scf

from ad_afqmc import pyscf_interface, run_afqmc


norb_frozen = 6
atomstring='''
C 0.000000 1.396792    0.000000
C 0.000000 -1.396792    0.000000
C 1.209657 0.698396    0.000000
C -1.209657 -0.698396    0.000000
C -1.209657 0.698396    0.000000
C 1.209657 -0.698396    0.000000
H 0.000000 2.484212    0.000000
H 2.151390 1.242106    0.000000
H -2.151390 -1.242106    0.000000
H -2.151390 1.242106    0.000000
H 2.151390 -1.242106    0.000000
H 0.000000 -2.484212    0.000000
'''
mol = gto.M(
    atom=atomstring,
    basis='ccpvdz',
    verbose=4,
    unit='angstrom')
mf = scf.RHF(mol)
mf.kernel()


options = {
    "dt": 0.01,
    "n_eql": 3,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 400,
    "n_walkers": 200,
    "seed": 98,
    "walker_type": "rhf",
    "trial": "rhf"
}
pyscf_interface.prep_afqmc(mf, norb_frozen=norb_frozen)
e_afqmc, err_afqmc = run_afqmc.run_afqmc(options=options, mpi_prefix='')

