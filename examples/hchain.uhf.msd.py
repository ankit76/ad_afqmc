from pyscf import fci, gto, scf
from ad_afqmc import pyscf_interface, run_afqmc

r = 1.6  # 2.0
nH = 7
atomstring = ""
for i in range(nH):
    atomstring += "H 0 0 %g\n" % (i * r)
mol = gto.M(atom=atomstring, basis="sto-6g", verbose=3, unit="bohr", spin=1)
mf = scf.UHF(mol)
mf.kernel()

ci = fci.FCI(mf)
e_fci, _ = ci.kernel()
print("FCI energy: ", e_fci)

state_dict = pyscf_interface.get_fci_state(ci, ndets=10)

pyscf_interface.prep_afqmc_multislater(
    mf, state_dict, max_excitation=6, ndets=10)

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

run_afqmc.run_afqmc(options, nproc=2)
