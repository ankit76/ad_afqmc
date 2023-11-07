from functools import partial

import h5py
import numpy as np
from pyscf import gto, scf

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
mol = gto.M(atom=atomstring, basis="sto-6g", verbose=3, symmetry=0)
mf = scf.RHF(mol)
mf.kernel()

norb_frozen = 1
# dipole integrals
nuc_dipmom = [0.0, 0.0, 0.0]
for i in range(mol.natm):
    for j in range(3):
        nuc_dipmom[j] += mol.atom_charge(i) * mol.atom_coord(i)[j]

# spatial orbitals
dip_ints_ao = -mol.intor_symmetric("int1e_r", comp=3)

# orbital relaxed hf, mp2, ccsd, ccsd(t) properties
#pyscf_interface.finite_difference_properties(
#    mol, dip_ints_ao[2], observable_constant=nuc_dipmom[2], norb_frozen=norb_frozen
#)

# ad afqmc
pyscf_interface.prep_afqmc(mf, norb_frozen=norb_frozen)
# frozen orbitals
dip_ints_mo = np.empty_like(dip_ints_ao)
dip_ints_mo_act = np.zeros(
    (dip_ints_ao.shape[0], mol.nao - norb_frozen, mol.nao - norb_frozen)
)
for i in range(dip_ints_ao.shape[0]):
    dip_ints_mo[i] = mf.mo_coeff.T.dot(dip_ints_ao[i]).dot(mf.mo_coeff)
    dip_ints_mo_act[i] = dip_ints_mo[i][norb_frozen:, norb_frozen:]
    nuc_dipmom[i] += 2.0 * np.trace(dip_ints_mo[i][:norb_frozen, :norb_frozen])
dip_ints_mo = dip_ints_mo_act
with h5py.File("observable.h5", "w") as fh5:
    fh5["constant"] = np.array([nuc_dipmom[2]])
    fh5["op"] = dip_ints_mo[2].flatten()

options = {
    "n_eql": 2,
    "n_ene_blocks": 1,
    "n_sr_blocks": 50,
    "n_blocks": 10,
    "n_walkers": 50,
    "seed": 698,
    "orbital_rotation": True,
    "ad_mode": "forward",
}
run_afqmc.run_afqmc(options=options, nproc=4)
