import os
import numpy
from pyscf import gto, scf, cc, fci
import numpy as np
from functools import partial
import h5py
from scipy.linalg import fractional_matrix_power
from ad_afqmc import driver, pyscf_interface

print = partial(print, flush=True)

r = 1.4
for nH in [ 4 ]:
  print(f'number of H: {nH}\n')
  atomstring = ""
  for i in range(nH):
    atomstring += "H 0 0 %g\n"%(i*r)
  mol = gto.M(atom = atomstring, basis = 'sto-6g', verbose = 3, unit = 'bohr')
  mf = scf.RHF(mol)
  mf.kernel()

  norb_frozen = 0
  overlap = mf.get_ovlp()
  lo = fractional_matrix_power(mf.get_ovlp(), -0.5).T
  #h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
  lo_mo = mf.mo_coeff.T.dot(overlap).dot(lo)
  h1 = np.einsum('i,j->ij', lo_mo[:, nH//2], lo_mo[:, nH//2])

  if nH < 15:
    cisolver = fci.FCI(mf)
    fci_ene, fci_vec = cisolver.kernel()
    print(f'fci_ene: {fci_ene}', flush=True)
    dm1 = cisolver.make_rdm1(fci_vec, mol.nao, mol.nelec)
    np.savetxt('rdm1_fci.txt', dm1)
    print(f'1e ene: {np.trace(np.dot(dm1, h1))}')

  # ccsd
  mycc = cc.CCSD(mf)
  mycc.frozen = norb_frozen
  mycc.kernel()
  dm1_cc = mycc.make_rdm1()

  et = mycc.ccsd_t()
  print('CCSD(T) energy', mycc.e_tot + et)

  with h5py.File('observable.h5', 'w') as fh5:
    fh5['constant'] = np.array([ 0. ])
    fh5['op'] = h1.flatten()

  pyscf_interface.prep_afqmc(mf)
  driver.run_afqmc(nproc=2)

  print('\nrelaxed finite difference h1e:')
  dE = 1.e-5

  E = numpy.array([ 0., -dE, 0. ])
  mf = scf.RHF(mol)
  h1e = mf.get_hcore() * (1 + E[1])
  mf.get_hcore = lambda *args: h1e
  mf.verbose = 1
  mf.kernel()
  emf_m = mf.e_tot
  mycc = cc.CCSD(mf)
  mycc.frozen = norb_frozen
  mycc.kernel()
  emp2_m = mycc.e_hf + mycc.emp2
  eccsd_m = mycc.e_tot
  et = mycc.ccsd_t()
  print('CCSD(T) energy', mycc.e_tot + et)
  eccsdpt_m = mycc.e_tot + et

  E = numpy.array([ 0., dE, 0. ])
  mf = scf.RHF(mol)
  h1e = mf.get_hcore() * (1 + E[1])
  mf.get_hcore = lambda *args: h1e
  mf.verbose = 1
  mf.kernel()
  emf_p = mf.e_tot
  mycc = cc.CCSD(mf)
  mycc.frozen = norb_frozen
  mycc.kernel()
  emp2_p = mycc.e_hf + mycc.emp2
  eccsd_p = mycc.e_tot
  et = mycc.ccsd_t()
  print('CCSD(T) energy', mycc.e_tot + et)
  eccsdpt_p = mycc.e_tot + et

  print(f'emf_m: {emf_m}, emf_p: {emf_p}, dip_mf: {(emf_p - emf_m) / 2 / dE}')
  print(f'emp2_m: {emp2_m}, emp2_p: {emp2_p}, dip_mp2: {(emp2_p - emp2_m) / 2 / dE}')
  print(f'eccsd_m: {eccsd_m}, eccsd_p: {eccsd_p}, dip_ccsd: {(eccsd_p - eccsd_m) / 2 / dE}')
  print(f'eccsdpt_m: {eccsdpt_m}, eccsd_p: {eccsdpt_p}, dip_ccsdpt: {(eccsdpt_p - eccsdpt_m) / 2 / dE}')

