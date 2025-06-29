from pyscf import gto, scf
from ad_afqmc import afqmc, config

mol =  gto.M(atom ="""
    O        0.0000000000      0.0000000000      0.0000000000
    H        0.9562300000      0.0000000000      0.0000000000
    H       -0.2353791634      0.9268076728      0.0000000000
    """,
    basis = 'cc-pvdz',
    verbose = 3)

# RHF
mf = scf.RHF(mol)
mf.kernel()

mycc = mf.CCSD(frozen=0).run()
et = mycc.ccsd_t()
print('CCSD(T) correlation energy', mf.e_tot+mycc.e_corr + et)

config.setup_jax()

af = afqmc.AFQMC(mycc, mycc)
af.free_projection = True
af.dt= 0.1
af.n_prop_steps= 10  # number of dt long steps in a propagation block
af.n_blocks= 8  # number of propagation and measurement blocks
af.n_ene_blocks= 10  # number of trajectories
af.n_walkers= 1000
#af.walker_type= "rhf"
af.ene0= mf.e_tot
af.seed = 5
af.kernel()
