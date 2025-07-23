import time
import pickle
import argparse

import h5py
import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp
from functools import partial

from ad_afqmc import config

tmpdir = "."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tmpdir")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_mpi", action="store_true")
    args = parser.parse_args()

    if args.use_gpu:
        config.afqmc_config["use_gpu"] = True

    if args.use_mpi:
        assert config.afqmc_config["use_gpu"] is False, "Inter GPU MPI not supported."
        config.afqmc_config["use_mpi"] = True

    tmpdir = args.tmpdir

config.setup_jax()
MPI = config.setup_comm()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from ad_afqmc import driver, sampling, wavefunctions, propagation, hamiltonian

print = partial(print, flush=True)


def _prep_afqmc(options=None, tmpdir="."):
    if rank == 0:
        print(f"# Number of MPI ranks: {size}\n#")

    # -------------------------------------------------------------------------
    # Integrals.
    with h5py.File(tmpdir + "/FCIDUMP_chol", "r") as fh5:
        [nelec, nmo, ms, nchol] = fh5["header"]
        h0 = jnp.array(fh5.get("energy_core"))
        h1 = jnp.squeeze(jnp.array(fh5.get("hcore")).reshape(-1, nmo, nmo))
        chol = jnp.array(fh5.get("chol")).reshape(-1, nmo, nmo)

    assert type(ms) is np.int64
    assert type(nelec) is np.int64
    assert type(nmo) is np.int64
    assert type(nchol) is np.int64
    ms, nelec, nmo, nchol = int(ms), int(nelec), int(nmo), int(nchol)
    nelec_sp = ((nelec + abs(ms)) // 2, (nelec - abs(ms)) // 2)
    norb = nmo

    # -------------------------------------------------------------------------
    # Options.
    if options is None:
        try:
            with open(tmpdir + "/options.bin", "rb") as f:
                options = pickle.load(f)

        except:
            options = {}

    options["dt"] = options.get("dt", 0.01)
    options["n_walkers"] = options.get("n_walkers", 50)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["n_ene_blocks"] = options.get("n_ene_blocks", 50)
    options["n_sr_blocks"] = options.get("n_sr_blocks", 1)
    options["n_blocks"] = options.get("n_blocks", 50)
    options["n_ene_blocks_eql"] = options.get("n_ene_blocks_eql", 5)
    options["n_sr_blocks_eql"] = options.get("n_sr_blocks_eql", 10)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_eql"] = options.get("n_eql", 1)
    options["ad_mode"] = options.get("ad_mode", None)
    assert options["ad_mode"] in [None, "forward", "reverse", "2rdm", "mixed"]
    options["orbital_rotation"] = options.get("orbital_rotation", True)
    options["do_sr"] = options.get("do_sr", True)
    options["walker_type"] = options.get("walker_type", "restricted")
    options["symmetry"] = options.get("symmetry", False)
    options["save_walkers"] = options.get("save_walkers", False)
    options["trial"] = options.get("trial", None)

    if options["trial"] is None:
        if rank == 0:
            print(f"# No trial specified in options.")

    options["ene0"] = options.get("ene0", 0.0)
    options["free_projection"] = options.get("free_projection", False)
    options["n_batch"] = options.get("n_batch", 1)
    options["vhs_mixed_precision"] = options.get("vhs_mixed_precision", False)
    options["trial_mixed_precision"] = options.get(
        "trial_mixed_precision", False
    )  # only relevant for cisd for now
    options["memory_mode"] = options.get("memory_mode", "low")  # only relevant for cisd

    try:
        with h5py.File(tmpdir + "/observable.h5", "r") as fh5:
            [observable_constant] = fh5["constant"]
            observable_op = np.array(fh5.get("op")).reshape(nmo, nmo)
            
            if options["walker_type"] == "unrestricted":
                observable_op = jnp.array([observable_op, observable_op])
            
            if options["walker_type"] == "generalized":
                observable_op = jsp.linalg.block_diag([observable_op, observable_op])
            
            observable = [observable_op, observable_constant]

    except:
        observable = None
    
    # -------------------------------------------------------------------------
    # Hamiltonian.
    ham = hamiltonian.hamiltonian(nmo)
    ham_data = {}
    ham_data["h0"] = h0

    if (h1.ndim == 3) or (options["walker_type"] == "generalized"): 
        ham_data["h1"] = jnp.array(h1)

    else: 
        ham_data["h1"] = jnp.array([h1, h1])

    ham_data["chol"] = chol.reshape(nchol, -1)
    ham_data["ene0"] = options["ene0"]

    # -------------------------------------------------------------------------
    # Trial.
    wave_data = {}
    mo_coeff = jnp.array(np.load(tmpdir + "/mo_coeff.npz")["mo_coeff"])

    try:
        rdm1 = jnp.array(np.load(tmpdir + "/rdm1.npz")["rdm1"])
        assert rdm1.shape == (2, norb, norb)
        wave_data["rdm1"] = rdm1
        print(f"# Read RDM1 from disk")

    except:
        if options["walker_type"] == "unrestricted":
            wave_data["rdm1"] = jnp.array(
                [
                    mo_coeff[0][:, : nelec_sp[0]] @ mo_coeff[0][:, : nelec_sp[0]].T,
                    mo_coeff[1][:, : nelec_sp[1]] @ mo_coeff[1][:, : nelec_sp[1]].T,
                ]
            )

        elif options["walker_type"] == "generalized":
            wave_data["rdm1"] = (
                mo_coeff[0][:, : nelec_sp[0] + nelec_sp[1]] @ 
                mo_coeff[0][:, : nelec_sp[0] + nelec_sp[1]].T
            )

    if options["trial"] == "rhf":
        trial = wavefunctions.rhf(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]

    elif options["trial"] == "uhf":
        trial = wavefunctions.uhf(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : nelec_sp[0]],
            mo_coeff[1][:, : nelec_sp[1]],
        ]

    elif options["trial"] == "ghf":
        trial = wavefunctions.ghf(norb, nelec_sp, n_batch=options["n_batch"])
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0] + nelec_sp[1]]

    elif options["trial"] == "noci":
        with open(tmpdir + "/dets.pkl", "rb") as f:
            ci_coeffs_dets = pickle.load(f)

        ci_coeffs_dets = [
            jnp.array(ci_coeffs_dets[0]),
            [jnp.array(ci_coeffs_dets[1][0]), jnp.array(ci_coeffs_dets[1][1])],
        ]
        wave_data["ci_coeffs_dets"] = ci_coeffs_dets
        trial = wavefunctions.noci(
            norb, nelec_sp, ci_coeffs_dets[0].size, n_batch=options["n_batch"]
        )

    elif options["trial"] == "cisd":
        try:
            amplitudes = np.load(tmpdir + "/amplitudes.npz")
            ci1 = jnp.array(amplitudes["ci1"])
            ci2 = jnp.array(amplitudes["ci2"])
            trial_wave_data = {"ci1": ci1, "ci2": ci2}
            wave_data.update(trial_wave_data)
            if options["trial_mixed_precision"]:
                mixed_real_dtype = jnp.float32
                mixed_complex_dtype = jnp.complex64
            else:
                mixed_real_dtype = jnp.float64
                mixed_complex_dtype = jnp.complex128
            trial = wavefunctions.cisd(
                norb,
                nelec_sp,
                n_batch=options["n_batch"],
                mixed_real_dtype=mixed_real_dtype,
                mixed_complex_dtype=mixed_complex_dtype,
                memory_mode=options["memory_mode"],
            )
        except:
            raise ValueError("Trial specified as cisd, but amplitudes.npz not found.")

    elif options["trial"] == "ucisd":
        try:
            amplitudes = np.load(tmpdir + "/amplitudes.npz")
            ci1a = jnp.array(amplitudes["ci1a"])
            ci1b = jnp.array(amplitudes["ci1b"])
            ci2aa = jnp.array(amplitudes["ci2aa"])
            ci2ab = jnp.array(amplitudes["ci2ab"])
            ci2bb = jnp.array(amplitudes["ci2bb"])
            trial_wave_data = {
                "ci1A": ci1a,
                "ci1B": ci1b,
                "ci2AA": ci2aa,
                "ci2AB": ci2ab,
                "ci2BB": ci2bb,
                "mo_coeff": mo_coeff,
            }
            wave_data.update(trial_wave_data)
            if options["trial_mixed_precision"]:
                mixed_real_dtype = jnp.float32
                mixed_complex_dtype = jnp.complex64
            else:
                mixed_real_dtype = jnp.float64
                mixed_complex_dtype = jnp.complex128
            trial = wavefunctions.ucisd(
                norb,
                nelec_sp,
                n_batch=options["n_batch"],
                mixed_real_dtype=mixed_real_dtype,
                mixed_complex_dtype=mixed_complex_dtype,
            )
        except:
            raise ValueError("Trial specified as ucisd, but amplitudes.npz not found.")

    else:
        try:
            with open(tmpdir + "/trial.pkl", "rb") as f:
                [trial, trial_wave_data] = pickle.load(f)
            wave_data.update(trial_wave_data)
            if rank == 0:
                print(f"# Read trial of type {type(trial).__name__} from trial.pkl.")
        except:
            if rank == 0:
                print(
                    "# trial.pkl not found, make sure to construct the trial separately."
                )
            trial = None

    # -------------------------------------------------------------------------
    # Propagator.
    if options["walker_type"] == "restricted":
        if options["symmetry"]:
            ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
        else:
            ham_data["mask"] = jnp.ones(ham_data["h1"].shape)
        if options["vhs_mixed_precision"]:
            prop = propagation.propagator_restricted(
                options["dt"],
                options["n_walkers"],
                n_batch=options["n_batch"],
                vhs_real_dtype=jnp.float32,
                vhs_complex_dtype=jnp.complex64,
            )
        else:
            prop = propagation.propagator_restricted(
                options["dt"], options["n_walkers"], n_batch=options["n_batch"]
            )

    elif options["walker_type"] == "unrestricted":
        if options["symmetry"]:
            ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
        else:
            ham_data["mask"] = jnp.ones(ham_data["h1"].shape)
        if options["free_projection"]:
            prop = propagation.propagator_unrestricted(
                options["dt"],
                options["n_walkers"],
                10,
                n_batch=options["n_batch"],
            )
        else:
            if options["vhs_mixed_precision"]:
                prop = propagation.propagator_unrestricted(
                    options["dt"],
                    options["n_walkers"],
                    n_batch=options["n_batch"],
                    vhs_real_dtype=jnp.float32,
                    vhs_complex_dtype=jnp.complex64,
                )
            else:
                prop = propagation.propagator_unrestricted(
                    options["dt"],
                    options["n_walkers"],
                    n_batch=options["n_batch"],
                )

    elif options["walker_type"] == "generalized":
        prop = propagation.propagator_generalized(
            options["dt"],
            options["n_walkers"],
            n_batch=options["n_batch"],
        )
    
    # -------------------------------------------------------------------------
    # Sampler.
    if options["ad_mode"] == "mixed":
        sampler = sampling.sampler_mixed(
            options["n_prop_steps"],
            options["n_ene_blocks"],
            options["n_sr_blocks"],
            options["n_blocks"],
        )

    else:
        sampler = sampling.sampler(
            options["n_prop_steps"],
            options["n_ene_blocks"],
            options["n_sr_blocks"],
            options["n_blocks"],
        )

    if rank == 0:
        print(f"# norb: {norb}")
        print(f"# nelec: {nelec_sp}")
        print("#")
        for op in options:
            if options[op] is not None:
                print(f"# {op}: {options[op]}")
        print("#")

    return ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI


if __name__ == "__main__":
    ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ = (
        _prep_afqmc()
    )
    assert trial is not None
    init = time.time()
    comm.Barrier()
    e_afqmc, err_afqmc = 0.0, 0.0

    if options["free_projection"]:
        driver.fp_afqmc(
            ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI
        )

    else:
        e_afqmc, err_afqmc = driver.afqmc(
            ham_data,
            ham,
            prop,
            trial,
            wave_data,
            sampler,
            observable,
            options,
            MPI,
            tmpdir=tmpdir,
        )

    comm.Barrier()
    end = time.time()

    if rank == 0:
        print(f"ph_afqmc walltime: {end - init}", flush=True)
        np.savetxt(tmpdir + "/ene_err.txt", np.array([e_afqmc, err_afqmc]))
    
    comm.Barrier()
