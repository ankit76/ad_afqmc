import os

os.environ[
    "XLA_FLAGS"
] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import pickle
import time

import h5py
import numpy as np
from jax import numpy as jnp
from mpi4py import MPI

from ad_afqmc import driver, hamiltonian, propagation, wavefunctions

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def _prep_afqmc(options=None):
    if rank == 0:
        print(f"# Number of MPI ranks: {size}\n#")

    with h5py.File("FCIDUMP_chol", "r") as fh5:
        [nelec, nmo, ms, nchol] = fh5["header"]
        h0 = jnp.array(fh5.get("energy_core"))
        h1 = jnp.array(fh5.get("hcore")).reshape(nmo, nmo)
        chol = jnp.array(fh5.get("chol")).reshape(-1, nmo, nmo)

    nelec_sp = (nelec // 2 + abs(ms) // 2, nelec - nelec // 2 - abs(ms) // 2)

    norb = nmo

    if options is None:
        try:
            with open("options.bin", "rb") as f:
                options = pickle.load(f)
        except:
            options = {}

    options["dt"] = options.get("dt", 0.01)
    options["n_walkers"] = options.get("n_walkers", 50)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["n_ene_blocks"] = options.get("n_ene_blocks", 50)
    options["n_sr_blocks"] = options.get("n_sr_blocks", 1)
    options["n_blocks"] = options.get("n_blocks", 50)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_eql"] = options.get("n_eql", 1)
    options["ad_mode"] = options.get("ad_mode", None)
    assert options["ad_mode"] in [None, "forward", "reverse"]
    options["orbital_rotation"] = options.get("orbital_rotation", True)
    options["do_sr"] = options.get("do_sr", True)
    options["walker_type"] = options.get("walker_type", "rhf")
    options["symmetry"] = options.get("symmetry", False)
    options["save_walkers"] = options.get("save_walkers", False)
    options["trial"] = options.get("trial", "rhf")
    options["ene0"] = options.get("ene0", 0.0)
    options["free_projection"] = options.get("free_projection", False)

    try:
        with h5py.File("observable.h5", "r") as fh5:
            [observable_constant] = fh5["constant"]
            observable_op = np.array(fh5.get("op")).reshape(nmo, nmo)
            if options["walker_type"] == "uhf":
                observable_op = jnp.array([observable_op, observable_op])
            observable = [observable_op, observable_constant]
    except:
        observable = None

    ad_q = options["ad_mode"] != None
    ham_data = {}
    ham_data["h0"] = h0
    ham_data["chol"] = chol.reshape(nchol, -1)
    ham_data["ene0"] = options["ene0"]
    if options["walker_type"] == "rhf":
        ham_data["h1"] = h1
        if options["symmetry"]:
            ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
        else:
            ham_data["mask"] = jnp.ones(ham_data["h1"].shape)
        ham = hamiltonian.hamiltonian(nmo, nelec // 2, nchol)
        prop = propagation.propagator(
            options["dt"],
            options["n_prop_steps"],
            options["n_ene_blocks"],
            options["n_sr_blocks"],
            options["n_blocks"],
            ad_q,
            options["n_walkers"],
        )
        trial = wavefunctions.rhf(norb, nelec // 2)
        wave_data = jnp.eye(norb)
    elif options["walker_type"] == "uhf":
        ham_data["h1"] = jnp.array([h1, h1])
        if options["symmetry"]:
            ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
        else:
            ham_data["mask"] = jnp.ones(ham_data["h1"].shape)
        if options["trial"] == "noci":
            ham = hamiltonian.hamiltonian_noci(nmo, nelec_sp, nchol)
            if options["free_projection"]:
                prop = propagation.propagator_uhf(
                    options["dt"],
                    options["n_prop_steps"],
                    options["n_ene_blocks"],
                    options["n_sr_blocks"],
                    options["n_blocks"],
                    True,
                    options["n_walkers"],
                    10,
                )
            else:
                prop = propagation.propagator_uhf(
                    options["dt"],
                    options["n_prop_steps"],
                    options["n_ene_blocks"],
                    options["n_sr_blocks"],
                    options["n_blocks"],
                    ad_q,
                    options["n_walkers"],
                )
            with open("dets.pkl", "rb") as f:
                wave_data = pickle.load(f)
            wave_data = [
                jnp.array(wave_data[0]),
                [jnp.array(wave_data[1][0]), jnp.array(wave_data[1][1])],
            ]
            trial = wavefunctions.noci(norb, nelec_sp, wave_data[0].size)
        else:
            ham = hamiltonian.hamiltonian_uhf(nmo, nelec_sp, nchol)
            if options["free_projection"]:
                prop = propagation.propagator_uhf(
                    options["dt"],
                    options["n_prop_steps"],
                    options["n_ene_blocks"],
                    options["n_sr_blocks"],
                    options["n_blocks"],
                    True,
                    options["n_walkers"],
                    10,
                )
            else:
                prop = propagation.propagator_uhf(
                    options["dt"],
                    options["n_prop_steps"],
                    options["n_ene_blocks"],
                    options["n_sr_blocks"],
                    options["n_blocks"],
                    ad_q,
                    options["n_walkers"],
                )
            trial = wavefunctions.uhf(norb, nelec_sp)
            wave_data = jnp.array(np.load("uhf.npz")["mo_coeff"])

    if rank == 0:
        print(f"# norb: {norb}")
        print(f"# nelec: {nelec_sp}")
        print("#")
        for op in options:
            print(f"# {op}: {options[op]}")
        print("#")

    return ham_data, ham, prop, trial, wave_data, observable, options


if __name__ == "__main__":
    ham_data, ham, prop, trial, wave_data, observable, options = _prep_afqmc()
    init = time.time()
    comm.Barrier()
    e_afqmc, err_afqmc = 0.0, 0.0
    if options["free_projection"]:
        driver.fp_afqmc(ham_data, ham, prop, trial, wave_data, observable, options)
    else:
        e_afqmc, err_afqmc = driver.afqmc(
            ham_data, ham, prop, trial, wave_data, observable, options
        )
    comm.Barrier()
    end = time.time()
    if rank == 0:
        print(f"ph_afqmc walltime: {end - init}", flush=True)
        np.savetxt("ene_err.txt", np.array([e_afqmc, err_afqmc]))
    comm.Barrier()
