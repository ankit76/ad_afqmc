from pyscf import gto, scf, cc
from ad_afqmc import afqmc

mol =  gto.M(atom ="""
    N        0.0000000000      0.0000000000      0.0000000000
    H        1.0225900000      0.0000000000      0.0000000000
    H       -0.2281193615      0.9968208791      0.0000000000
    """,
    spin = 1,
    basis = '6-31g',
    verbose = 3)

# UHF
mf = scf.UHF(mol)
mf.kernel()

# UCCSD
mycc = cc.CCSD(mf)
mycc.kernel()

# afqmc @ UHF
af = afqmc.AFQMC(mycc)
af.n_walkers = 5 # !!! ONLY to make the example faster !!!
af.n_blocks = 10 # !!! ONLY to make the example faster !!!
af.kernel()
