import argparse
import pickle
import time

import h5py
import numpy as np

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

from functools import partial

from jax import numpy as jnp

from ad_afqmc import driver, hamiltonian, sampling

print = partial(print, flush=True)

from prep import read_fcidump, read_options, read_observable, read_wave_data, set_trial, set_prop

def _prep_afqmc(options=None):
    if rank == 0:
        print(f"# Number of MPI ranks: {size}\n#")

    h0, h1, chol, ms, nelec, nmo, nchol, nelec_sp = read_fcidump(tmpdir)
    norb = nmo

    options = read_options(options, rank, tmpdir)
    observable = read_observable(nmo, options, tmpdir)

    ham = hamiltonian.hamiltonian(nmo)
    ham_data = {}
    ham_data["h0"] = h0
    ham_data["h1"] = jnp.array([h1, h1])
    ham_data["chol"] = chol.reshape(nchol, -1)
    ham_data["ene0"] = options["ene0"]

    mo_coeff = jnp.array(np.load(tmpdir + "/mo_coeff.npz")["mo_coeff"])

    wave_data = read_wave_data(mo_coeff, norb, nelec_sp, tmpdir)
    trial = set_trial(options, mo_coeff, norb, nelec_sp, rank, wave_data, tmpdir)
    prop, ham_data = set_prop(options, ham_data)

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
