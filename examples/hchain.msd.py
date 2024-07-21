from pyscf import fci, gto, scf

from ad_afqmc import (
    driver,
    hamiltonian,
    mpi_jax,
    propagation,
    pyscf_interface,
    wavefunctions,
)

r = 1.6  # 2.0
nH = 6
atomstring = ""
for i in range(nH):
    atomstring += "H 0 0 %g\n" % (i * r)
mol = gto.M(atom=atomstring, basis="sto-6g", verbose=3, unit="bohr")
mf = scf.RHF(mol)
mf.kernel()

ci = fci.FCI(mf)
e_fci, _ = ci.kernel()
print("FCI energy: ", e_fci)

pyscf_interface.prep_afqmc(mf)
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

ham_data, ham, prop, trial, wave_data, observable, options = mpi_jax._prep_afqmc(
    options
)

trial = wavefunctions.multislater_rhf(mol.nao, mol.nelec[0], max_excitation=6)
state_dict = pyscf_interface.get_fci_state(ci, ndets=10)
Acre, Ades, Bcre, Bdes, coeff = pyscf_interface.get_excitations(
    state=state_dict, max_excitation=6, ndets=10
)  # this function reads the Dice dets.bin file if state is not provided
wave_data = {
    "Acre": Acre,
    "Ades": Ades,
    "Bcre": Bcre,
    "Bdes": Bdes,
    "coeff": coeff,
}

e_afqmc, err_afqmc = driver.afqmc(
    ham_data,
    ham,
    prop,
    trial,
    wave_data,
    observable,
    options,
    # init_walkers=init_walkers,
)
