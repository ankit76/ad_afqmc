import jax
import jax.numpy as jnp
from pyscf import gto, scf, fci, ci, cc
from ad_afqmc import afqmc, config, pyscf_interface, launch_script
import dill as pickle
import numpy as np

jax.config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=1000, precision=12)

mol =  gto.M(atom ="""
    N 0.0 0.0 0.0
    N 2.5 0.0 0.0
    """,
    basis = '6-31g',
    spin=0,
    verbose = 3)

# RHF
mf = scf.UHF(mol)
mf.kernel()
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)

mycc = cc.UCCSD(mf)
mycc.kernel()

#with open("mf.pkl", "wb") as f:
#    pickle.dump((mf, mycc), f)
#
#with open("mf.pkl", "rb") as f:
#    mf, mycc = pickle.load(f)

af = afqmc.AFQMC(mycc) #, mycc)
af.chol_cut = 1e-5
af.free_projection = True
af.dt= 0.1
af.n_prop_steps= 20  # number of dt long steps in a propagation block
af.n_blocks= 2 # number of propagation and measurement blocks
af.n_ene_blocks= 4  # number of trajectories
af.n_walkers= 5
af.walker_type= "uhf"
af.norb_frozen=0
af.ene0= mf.e_tot
af.seed = 42
af.ngrid = 3
af.symmetry_projector="s2"
af.target_spin = 0.
af.save_walkers = True
af.tmpdir = "."
af.kernel()

with open("options.bin", "rb") as f:
    options = pickle.load(f)

# Builds the block data from prop_data
def calc_energy(options, mycc, prop_data_bin):
    pyscf_interface.read_pyscf_ccsd(mycc, options["tmpdir"])

    (
        ham_data,
        ham,
        prop,
        trial_bra,
        wave_data_bra,
        trial_ket,
        wave_data_ket,
        sampler,
        observable,
        options,
    ) = launch_script.setup_afqmc_fp(options, options["tmpdir"])

    ham_data = ham.build_measurement_intermediates(ham_data, trial_bra, wave_data_bra)
    ham_data = ham.build_propagation_intermediates(
        ham_data, prop, trial_bra, wave_data_bra
    )

    l_prop_data = []
    with open(prop_data_bin, "rb") as f:
        while True:
            try:
                l_prop_data.append(pickle.load(f))
            except EOFError:
                break

    nmo = trial_bra.norb
    na, nb = trial_bra.nelec

    block_energy = []
    block_weight = []
    block_ovlp = []
    block_abs_ovlp = []
    for i, prop_data in enumerate(l_prop_data):
        prop_data["walkers"].data[0] = prop_data["walkers"].data[0].reshape((-1, nmo, na))
        prop_data["walkers"].data[1] = prop_data["walkers"].data[1].reshape((-1, nmo, nb))

        energy_samples = trial_bra.calc_energy(prop_data["walkers"], ham_data, wave_data_bra)

        prop_data["overlaps"] = np.atleast_2d(prop_data["overlaps"])
        prop_data["weights"] = np.atleast_2d(prop_data["weights"])
        energy_samples = energy_samples.reshape((-1, options["n_walkers"]))

        # Initial => No propagation
        if "init" in prop_data_bin:
            e = jnp.array(jnp.sum(energy_samples) / options["n_walkers"])
        else:
            energy_samples = jnp.where(
                jnp.abs(energy_samples - ham_data["ene0"]) > jnp.sqrt(2.0 / prop.dt),
                ham_data["ene0"],
                energy_samples,
            )
            e = jnp.sum(
                energy_samples * prop_data["overlaps"] * prop_data["weights"], axis=1
            ) / jnp.sum(prop_data["overlaps"] * prop_data["weights"], axis=1)

        # Initial => No propagation
        if "init" in prop_data_bin:
            w = jnp.sum(prop_data["weights"], axis=1)
        else:
            w = jnp.sum(prop_data["overlaps"] * prop_data["weights"], axis=1)

        o = jnp.sum(prop_data["overlaps"], axis=1)
        abs_o = jnp.sum(jnp.abs(prop_data["overlaps"]), axis=1)

        block_energy.append(e)
        block_weight.append(w)
        block_ovlp.append(o)
        block_abs_ovlp.append(abs_o)

    shape = (options["n_ene_blocks"], -1)
    block_energy = jnp.array(block_energy).reshape(shape)
    block_weight = jnp.array(block_weight).reshape(shape)
    block_ovlp = jnp.array(block_ovlp).reshape(shape)
    block_abs_ovlp = jnp.array(block_abs_ovlp).reshape(shape)

    return block_energy, block_weight, block_ovlp, block_abs_ovlp

# Concatenates the guess data with the ones obtained after propagation
def calc_energy_tot(options, mycc):
    e_i, w_i, o_i, ao_i = calc_energy(options, mycc, "prop_data_init_0.bin")
    e, w, o, ao = calc_energy(options, mycc, "prop_data_0.bin")

    total_energy = np.concatenate((e_i, e), axis=1)
    total_weight = np.concatenate((w_i, w), axis=1)
    total_ovlp = np.concatenate((o_i, o), axis=1)
    total_abs_ovlp = np.concatenate((ao_i, ao), axis=1)
    total_sign = total_ovlp / total_abs_ovlp

    return total_energy, total_weight, total_sign

# Loads the files written by the fp calculation
def load_energy_tot():
    with open("Raw_E.dat", "r") as f:
        total_energy = np.atleast_2d(np.loadtxt(f, dtype=complex).T)
    with open("Raw_W.dat", "r") as f:
        total_weight = np.atleast_2d(np.loadtxt(f, dtype=complex).T)
    with open("Raw_S.dat", "r") as f:
        total_sign = np.atleast_2d(np.loadtxt(f, dtype=complex).T)

    return total_energy, total_weight, total_sign

# Writes the summary
def write_samples(total_energy, total_weight, total_sign, options):
    for n in range(options["n_ene_blocks"]):
        mean_energies = np.sum(
            total_energy[: n + 1] * total_weight[: n + 1], axis=0
        ) / np.sum(total_weight[: n + 1], axis=0)

        if n == 0:
            errors = np.zeros_like(mean_energies)
        else:
            errors = np.std(total_energy[: n + 1], axis=0) / (n) ** 0.5

        mean_signs = np.sum(
            total_sign[: n + 1] * total_weight[: n + 1], axis=0
        ) / np.sum(total_weight[: n + 1], axis=0)

        times = options["dt"] * options["n_prop_steps"] * jnp.arange(options["n_blocks"] + 1)

        np.savetxt(
            "samples_test.dat",
            np.stack(
                (
                    times,
                    mean_energies.real,
                    errors.real,
                    mean_signs.real,
                )
            ).T,
        )

total_energy, total_weight, total_sign = calc_energy_tot(options, mycc)
total_energy2, total_weight2, total_sign2 = load_energy_tot()

#print(total_energy)
#print()
#print(total_energy2)
#print()
#print(total_energy - total_energy2)
err_e = np.linalg.norm(total_energy-total_energy2)
err_w = np.linalg.norm(total_weight-total_weight2)
err_s = np.linalg.norm(total_sign-total_sign2)

print()
print("Err(E):", err_e)
print("Err(W):", err_w)
print("Err(S):", err_s)

assert err_e < 1e-12
assert err_w < 1e-12
assert err_s < 1e-12

write_samples(total_energy, total_weight, total_sign, options)
ref = np.loadtxt("samples_raw.dat")
test = np.loadtxt("samples_test.dat")

err_samples = np.linalg.norm(ref-test)

print()
print("Err samples_raw:", err_samples)

assert err_samples < 1e-12
