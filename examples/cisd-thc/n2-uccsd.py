from functools import partial

import h5py, pickle
import numpy as np
from pyscf import gto, scf, cc

from ad_afqmc import pyscf_interface, run_afqmc, mpi_jax, wavefunctions, driver

import jax.numpy as jnp

print = partial(print, flush=True)

r = 3.0
mol = gto.M(atom=
            f''' N 0 0 0 
            N 0 0 {r}''', 
                  basis="cc-pvdz", verbose=4, symmetry=0,unit='B')
mf = scf.UHF(mol)
mf.kernel()
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mf.stability()


#mf.mo_coeff[1] = 1*mf.mo_coeff[0]
##
nfrozen = 2
mycc = cc.UCCSD(mf)
mycc.frozen = nfrozen
mycc.run()
et = mycc.ccsd_t()
print(mycc.e_corr + et)

#mycc.t1 = (mycc.t1[0]*0., mycc.t1[1]*0.)

ci2AA = (mycc.t2[0] + 2.*np.einsum('ia,jb->ijab', mycc.t1[0], mycc.t1[0])).transpose(0,2,1,3)
ci2AB = (mycc.t2[1] +    np.einsum('ia,jb->ijab', mycc.t1[0], mycc.t1[1])).transpose(0,2,1,3)
ci2BB = (mycc.t2[2] + 2.*np.einsum('ia,jb->ijab', mycc.t1[1], mycc.t1[1])).transpose(0,2,1,3)

print(sum(mycc.t1[0].shape), (mycc.t1[0].shape[0], mycc.t1[1].shape[0]))
trial = wavefunctions.UCISD(sum(mycc.t1[0].shape), (mycc.t1[0].shape[0], mycc.t1[1].shape[0]))

print(np.max(abs(mf.mo_coeff - mycc.mo_coeff)))
pyscf_interface.prep_afqmc(mf, norb_frozen=nfrozen)

mo_coeff = jnp.array(np.load("mo_coeff.npz")["mo_coeff"])

print(np.max(abs(mf.mo_coeff[1][:,nfrozen:] - mf.mo_coeff[0][:,nfrozen:] @ mo_coeff[1])))
wave_data = {
    "mo_coeff": mo_coeff,
    "ci1A" : 1.*mycc.t1[0],
    "ci1B" : 1.*mycc.t1[1],
    "ci2AA": 1.*ci2AA,
    "ci2AB": 1.*ci2AB,
    "ci2BB": 1.*ci2BB
}
with open("trial.pkl", "wb") as f:
    pickle.dump([trial, wave_data], f)

options = {
    "dt": 0.005,
    "n_eql": 5,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 100,
    "n_prop_steps": 50,
    "n_walkers": 50,
    "seed": 8,
    "walker_type": "uhf",
}

from mpi4py import MPI

MPI.Finalize()
run_afqmc.run_afqmc(options, nproc=1)



