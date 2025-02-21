from functools import partial

import h5py
import numpy as np
from pyscf import gto, scf, cc

from ad_afqmc import pyscf_interface, run_afqmc

print = partial(print, flush=True)

r = 1.012
theta = 106.67 * np.pi / 180.0
rz = r * np.sqrt(np.cos(theta / 2) ** 2 - np.sin(theta / 2) ** 2 / 3)
dc = 2 * r * np.sin(theta / 2) / np.sqrt(3)
atomstring = f"""
                 N 0. 0. 0.
                 H 0. {dc} {rz}
                 H {r * np.sin(theta/2)} {-dc/2} {rz}
                 H {-r * np.sin(theta/2)} {-dc/2} {rz}
              """
mol = gto.M(atom=atomstring, basis="ccpvdz", verbose=3, symmetry=0)
mf = scf.RHF(mol)
mf.kernel()

norb_frozen = 1
cc = mf.CCSD()
cc.frozen = norb_frozen
cc.run()
et = cc.ccsd_t()
print("Total CCSD(T) energy: ", cc.e_corr+et+mf.e_tot)



# ad afqmc
pyscf_interface.prep_afqmc(mf, norb_frozen=norb_frozen)

options = {
    "n_eql": 2,
    "n_ene_blocks": 1,
    "n_sr_blocks": 1,
    "n_blocks": 500,
    "n_walkers": 50,
    "seed": 698,
    "phaseless_epsilon": 0.05,
#    "orbital_rotation": True,
#    "ad_mode": "forward",
    "trial": "rhf",
}
run_afqmc.run_afqmc(options=options, nproc=1)

