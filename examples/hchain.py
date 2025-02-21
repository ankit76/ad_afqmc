from functools import partial

import numpy as np
from pyscf import fci, gto, scf

from ad_afqmc import pyscf_interface, run_afqmc

print = partial(print, flush=True)

r = 2.0
nH = 6
atomstring = ""
for i in range(nH):
    atomstring += "H 0 0 %g\n" % (i * r)
mol = gto.M(atom=atomstring, basis="sto-6g", verbose=3, unit="bohr")
mf = scf.RHF(mol)
mf.kernel()

umf = scf.UHF(mol)
umf.kernel()
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)

h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)

# fci
cisolver = fci.FCI(mf)
fci_ene, fci_vec = cisolver.kernel()
print(f"fci_ene: {fci_ene}", flush=True)

# ad afqmc
pyscf_interface.prep_afqmc(umf)
options = {
    "n_eql": 2,
    "n_ene_blocks": 1,
    "n_sr_blocks": 50,
    "n_blocks": 10,
    "n_walkers": 50,
    "seed": 98,
    "trial": "uhf",
    "walker_type": "unrestricted",
}

# serial run
# run_afqmc.run_afqmc(options=options, mpi_prefix='')

run_afqmc.run_afqmc(options=options, nproc=4)

