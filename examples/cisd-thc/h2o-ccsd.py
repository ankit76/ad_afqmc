from functools import partial

import numpy as np
from pyscf import cc, gto, scf

from ad_afqmc import driver, mpi_jax, pyscf_interface, wavefunctions

print = partial(print, flush=True)

mol = gto.M(
    atom="""     H                  0.00000000    -1.44445108     1.00222970;
                  O                  0.00000000    -0.00000000    -0.12629916;
                  H                  0.00000000     1.44445108     1.00222970""",
    basis="cc-pvdz",
    verbose=4,
    symmetry=0,
    unit="B",
)
mf = scf.RHF(mol)
mf.kernel()

nfrozen = 1
pyscf_interface.prep_afqmc(mf, norb_frozen=nfrozen)
options = {
    "dt": 0.005,
    "n_eql": 5,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 100,
    "n_prop_steps": 50,
    "n_walkers": 50,
    "seed": 8,
    "walker_type": "rhf",
}

ham_data, ham, prop, trial, wave_data, sampler, observable, options = (
    mpi_jax._prep_afqmc(options)
)


##
mycc = cc.CCSD(mf)
mycc.frozen = nfrozen
mycc.run()
et = mycc.ccsd_t()
print(mycc.e_corr + et)

ci2 = mycc.t2 + np.einsum("ia,jb->ijab", mycc.t1, mycc.t1)
ci2 = ci2.transpose(0, 2, 1, 3)
ci1 = mycc.t1

trial = wavefunctions.CISD(sum(ci1.shape), (ci1.shape[0], ci1.shape[0]))

wave_data["ci1"] = ci1
wave_data["ci2"] = ci2

e_afqmc, err_afqmc = driver.afqmc(
    ham_data,
    ham,
    prop,
    trial,
    wave_data,
    sampler,
    observable,
    options,
)
