from pyscf import gto, scf, cc
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
af.kernel()

# Frozen-core RCCSD 
mycc = cc.CCSD(mf)
mycc.frozen = 1
mycc.kernel()

# Frozen-core afqmc @ RCCSD
af = afqmc.AFQMC(mycc)
af.norb_frozen = 1
af.kernel()
