from functools import partial

import h5py, pickle
import numpy as np
from pyscf import gto, scf, cc

from ad_afqmc import afqmc, pyscf_interface, run_afqmc, config
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


config.setup_jax()
af = afqmc.AFQMC(mycc,mf)
af.trial = "UCISD"
af.free_projection = True
af.dt= 0.1
af.n_prop_steps= 1  # number of dt long steps in a propagation block
af.n_blocks= 10  # number of propagation and measurement blocks
af.n_ene_blocks= 10  # number of trajectories
af.n_walkers= 1000
af.walker_type= "uhf"
af.ene0= mf.e_tot+mycc.e_corr + et
af.seed = 5
af.symmetry_projector = "s2"  # time-reversal symmetry projection
af.target_spin = 0
af.kernel()
