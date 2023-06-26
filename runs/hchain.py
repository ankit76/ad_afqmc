import os
import numpy
from pyscf import gto, scf, cc, fci
import numpy as np
from functools import partial
import h5py
from scipy.linalg import fractional_matrix_power
from ad_afqmc import driver, pyscf_interface

print = partial(print, flush=True)

r = 2.4
for nH in [ 4 ]:
  print(f'number of H: {nH}\n')
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

  norb_frozen = 0
  overlap = mf.get_ovlp()
  h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)

  #lo = fractional_matrix_power(mf.get_ovlp(), -0.5).T
  #lo_mo = mf.mo_coeff.T.dot(overlap).dot(lo)
  #h1 = np.einsum('i,j->ij', lo_mo[:, nH//2], lo_mo[:, nH//2])

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

  pyscf_interface.prep_afqmc(umf)

  with h5py.File('FCIDUMP_chol', 'r') as fh5:
    h1 = np.array(fh5.get('hcore')).reshape(mol.nao, mol.nao)

  with h5py.File('observable.h5', 'w') as fh5:
    fh5['constant'] = np.array([ 0. ])
    fh5['op'] = h1.flatten()

  options = {'n_eql': 2,
             'n_ene_blocks': 25,
             'n_sr_blocks': 2,
             'n_blocks': 10,
             'n_walkers': 50,
             'seed': 98,
             'walker_type': 'uhf',
             'ad_mode': 'reverse'}
  driver.run_afqmc(options=options)

  print('\nrelaxed finite difference h1e:')
  pyscf_interface.finite_difference_properties(mol, mf.get_hcore())

  afqmc_rdm = np.load('rdm1_afqmc.npz')['rdm1']
  lo = fractional_matrix_power(mf.get_ovlp(), -0.5).T
  lo_mo = umf.mo_coeff[0].T.dot(overlap).dot(lo)
  afqmc_rdm = np.einsum('sij,ip,jq->spq', afqmc_rdm, lo_mo, lo_mo)

  lo_mo = mf.mo_coeff.T.dot(overlap).dot(lo)
  fci_rdm = np.einsum('ij,ip,jq->pq', dm1, lo_mo, lo_mo)
  print(f'\nafqmc_rdm:\n{afqmc_rdm}\n')
  print(f'\nfci_rdm:\n{fci_rdm / 2.}\n')

