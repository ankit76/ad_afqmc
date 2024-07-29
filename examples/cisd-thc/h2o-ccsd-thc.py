from functools import partial

import h5py
import numpy as np
from pyscf import gto, scf, cc

from ad_afqmc import pyscf_interface, run_afqmc, mpi_jax, wavefunctions, driver

print = partial(print, flush=True)

mol = gto.M(atom=
            '''     H                  0.00000000    -1.44445108     1.00222970;
                  O                  0.00000000    -0.00000000    -0.12629916;
                  H                  0.00000000     1.44445108     1.00222970''', basis="cc-pvdz", verbose=4, symmetry=0,unit='B')
mf = scf.RHF(mol)
mf.kernel()

nfrozen = 1
pyscf_interface.prep_afqmc(mf, norb_frozen = nfrozen)
options = {
    "dt": 0.005,
    "n_eql": 5,
    "n_ene_blocks": 5,
    "n_sr_blocks": 10,
    "n_blocks": 10,
    "n_prop_steps": 50,
    "n_walkers": 50,
    "seed": 8,
    "walker_type": "rhf",
}

ham_data, ham, prop, trial, wave_data, observable, options = mpi_jax._prep_afqmc(
    options
)


##
mycc = cc.CCSD(mf)
mycc.frozen = nfrozen
mycc.run()
et = mycc.ccsd_t()
print(mycc.e_corr + et)

ci2 = mycc.t2 + 0.5 * np.einsum('ia,jb->ijab', mycc.t1, mycc.t1)
ci2 = ci2.transpose(0,2,1,3)
ci1 = mycc.t1

##THC decompose the ci coefficients
ncore = mycc.frozen
nocc = mycc.t1.shape[0]
nvirt = mycc.t1.shape[1]
Xocc, Xvirt = pyscf_interface.getCollocationMatrices(mol, grid_level = 0, thc_eps = 1.e-4, 
                                                     mo1 = mf.mo_coeff[:,ncore:nocc+ncore], 
                                                     mo2 = mf.mo_coeff[:,nocc+ncore:], 
                                                     alpha=0.25)


VKL = pyscf_interface.solveLS_twoSided(ci2, Xocc, Xvirt)
trial = wavefunctions.CISD_THC(sum(ci1.shape), ci1.shape[0])

wave_data = {
    "ci1" : 1.*ci1,
    "Xocc" : 1.*Xocc,
    "Xvirt": 1.*Xvirt,
    "VKL": 1.*VKL
}

e_afqmc, err_afqmc = driver.afqmc(
    ham_data,
    ham,
    prop,
    trial,
    wave_data,
    observable,
    options,
)
