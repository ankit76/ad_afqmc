from pyscf import gto, scf, cc, fci
import numpy as np
from functools import partial
import h5py
from ad_afqmc import driver, pyscf_interface

print = partial(print, flush=True)

r = 2.
nH = 6
atomstring = ""
for i in range(nH):
  atomstring += "H 0 0 %g\n"%(i*r)
mol = gto.M(atom = atomstring, basis = 'sto-6g', verbose = 3, unit = 'bohr')
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
print(f'fci_ene: {fci_ene}', flush=True)
dm1 = cisolver.make_rdm1(fci_vec, mol.nao, mol.nelec)
print(f'1e ene: {np.trace(np.dot(dm1, h1))}')

# ad afqmc
pyscf_interface.prep_afqmc(umf)
options = {'n_eql': 2,
           'n_ene_blocks': 1,
           'n_sr_blocks': 50,
           'n_blocks': 10,
           'n_walkers': 50,
           'seed': 98,
           'walker_type': 'uhf',
           'ad_mode': 'reverse'}
driver.run_afqmc(options=options, nproc=4)
