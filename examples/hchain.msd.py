import pickle

from pyscf import fci, gto, scf

from ad_afqmc import pyscf_interface, run_afqmc, wavefunctions

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

trial = wavefunctions.multislater(mol.nao, mol.nelec, max_excitation=6)
state_dict = pyscf_interface.get_fci_state(ci, ndets=10)
Acre, Ades, Bcre, Bdes, coeff, ref_det = pyscf_interface.get_excitations(
    state=state_dict, max_excitation=6, ndets=10
)  # this function reads the Dice dets.bin file if state is not provided
wave_data = {
    "Acre": Acre,
    "Ades": Ades,
    "Bcre": Bcre,
    "Bdes": Bdes,
    "coeff": coeff,
    "ref_det": ref_det,
}
# write wavefunction to disk
with open("trial.pkl", "wb") as f:
    pickle.dump([trial, wave_data], f)

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

run_afqmc.run_afqmc(options, nproc=2)

# from ad_afqmc import driver, mpi_jax
# ham_data, ham, prop, _, wave_data, sampler, observable, options = mpi_jax._prep_afqmc(
#     options
# )
#
# e_afqmc, err_afqmc = driver.afqmc(
#     ham_data,
#     ham,
#     prop,
#     trial,
#     wave_data,
#     sampler,
#     observable,
#     options,
#     # init_walkers=init_walkers,
# )
