from pyscf import gto, scf, fci
from ad_afqmc import afqmc

mol =  gto.M(atom ="""
    N        0.0000000000      0.0000000000      0.0000000000
    N        2.9562300000      0.0000000000      0.0000000000
    """,
    basis = 'cc-pvdz',
    # symmetry="True",
    verbose = 4)

# RHF
mf = scf.UHF(mol)
mf.kernel()
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)

mycc = mf.CCSD(frozen=0).run()
et = mycc.ccsd_t()
print('CCSD(T) correlation energy', mf.e_tot+mycc.e_corr + et)

# cisolver = fci.FCI(mf)
# cisolver.nroots = 3
# print('E(FCI)', cisolver.kernel()[0])


af = afqmc.AFQMC(mf)
af.free_projection = True
af.dt= 0.1
af.n_prop_steps= 1  # number of dt long steps in a propagation block
af.n_blocks= 40  # number of propagation and measurement blocks
af.n_ene_blocks= 10  # number of trajectories
af.n_walkers= 1000
af.walker_type= "uhf"
af.ene0= mf.e_tot
af.seed = 5
af.symmetry_projector="s2"
af.target_spin = 4.
#af.symmetry_projector="tr"
af.kernel()
