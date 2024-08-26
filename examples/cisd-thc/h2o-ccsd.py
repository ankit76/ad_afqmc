from functools import partial

from pyscf import cc, gto, scf

from ad_afqmc import pyscf_interface, run_afqmc

print = partial(print, flush=True)

mol = gto.M(
    atom="""
        H    0.00000000    -1.44445108     1.00222970
        O    0.00000000    -0.00000000    -0.12629916
        H    0.00000000     1.44445108     1.00222970
    """,
    basis="cc-pvdz",
    verbose=3,
    symmetry=0,
    unit="B",
)
mf = scf.RHF(mol)
mf.kernel()

norb_frozen = 1
mycc = cc.CCSD(mf)
mycc.frozen = norb_frozen
mycc.run()
et = mycc.ccsd_t()
print(f"CCSD(T) energy: {mycc.e_tot + et}")

pyscf_interface.prep_afqmc(mycc)
options = {
    "dt": 0.005,
    "n_eql": 5,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 100,
    "n_prop_steps": 50,
    "n_walkers": 50,
    "seed": 8,
    "walker_type": "rhf",
    "trial": "cisd",
}

# pyscf cc initializes mpi so we need to finalize it before running afqmc
from mpi4py import MPI

MPI.Finalize()
run_afqmc.run_afqmc(options, nproc=4)
