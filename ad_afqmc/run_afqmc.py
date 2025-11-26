import os
import pickle
import shlex
import subprocess
from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np

from ad_afqmc import config, driver, optimize_trial, utils, wavefunctions

print = partial(print, flush=True)


def run_afqmc(
    options: Optional[dict] = None,
    mpi_prefix: Optional[str] = None,
    nproc: Optional[int] = None,
    tmpdir: Optional[str] = None,
):
    """
    Run AFQMC calculation from pre-generated input files.

    Parameters:
        options : dict, optional
            Options for AFQMC.
        mpi_prefix : str, optional
            MPI prefix, used to launch MPI processes.
        nproc : int, optional
            Number of processes, if using MPI.
        tmpdir : str, optional
            Temporary directory where the input files are stored.
    """
    if tmpdir is None:
        try:
            with open("tmpdir.txt", "r") as f:
                tmpdir = f.read().strip()
            print(f"# tmpdir.txt file found: tmpdir is set to '{tmpdir}'\n#")
        except:
            tmpdir = "."
    assert os.path.isdir(tmpdir), f"tmpdir directory '{tmpdir}' does not exist."

    if options is not None:
        with open(tmpdir + "/options.bin", "wb") as f:
            pickle.dump(options, f)
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/launch_script.py"
    use_gpu = config.afqmc_config["use_gpu"]
    use_mpi = config.afqmc_config["use_mpi"]

    if not use_gpu and config.afqmc_config["use_mpi"] is not False:
        try:
            from mpi4py import MPI

            if not MPI.Is_finalized():
                MPI.Finalize()
            use_mpi = True
            print(f"# mpi4py found, using MPI.")
            if nproc is None:
                print(f"# Number of MPI ranks not specified, using 1 by default.")
                nproc = 1
        except ImportError:
            use_mpi = False
            if mpi_prefix is not None or nproc is not None:
                raise ValueError(
                    f"# MPI prefix or number of processes specified, but mpi4py not found. Please install mpi4py or remove the MPI options."
                )
            else:
                print(f"# Unable to import mpi4py, not using MPI.")

    gpu_flag = "--use_gpu" if use_gpu else ""
    mpi_flag = "--use_mpi" if use_mpi else ""
    if mpi_prefix is None:
        if use_mpi:
            mpi_prefix = "mpirun "
            if nproc is not None:
                mpi_prefix += f"-np {nproc} "

        else:
            mpi_prefix = ""
    elif nproc is not None:
        mpi_prefix += f"-np {nproc}"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    cmd = shlex.split(f"{mpi_prefix} python {script} {tmpdir} {gpu_flag} {mpi_flag}")
    # Launch process with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env,
        bufsize=1,
    )
    # Print output in real-time
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(output, end="")
    return_code = process.poll()
    if return_code != 0:
        if return_code is None:
            return_code = -1
        raise subprocess.CalledProcessError(return_code, cmd)

    try:
        ene_err = np.loadtxt(tmpdir + "/ene_err.txt")
    except:
        print("AFQMC did not execute correctly.")
        ene_err = 0.0, 0.0
    return ene_err[0], ene_err[1]


def run_afqmc_ph(
    pyscf_prep: Optional[dict] = None,
    options: Optional[dict] = None,
    mpi_prefix: Optional[str] = None,
    nproc: Optional[int] = None,
    tmpdir: Optional[str] = None,
):
    """
    Run AFQMC calculation from pre-generated input files.

    Parameters:
        options : dict, optional
            Options for AFQMC.
        mpi_prefix : str, optional
            MPI prefix, used to launch MPI processes.
        nproc : int, optional
            Number of processes, if using MPI.
        tmpdir : str, optional
            Temporary directory where the input files are stored.
    """
    config.setup_jax()
    comm = config.setup_comm()
    (
        ham_data,
        ham,
        prop,
        trial,
        wave_data,
        sampler,
        observable,
        options,
    ) = utils.setup_afqmc_ph(pyscf_prep, options, tmpdir)
    e_afqmc, err_afqmc = driver.afqmc(
        ham_data,
        ham,
        prop,
        trial,
        wave_data,
        sampler,
        observable,
        options,
        comm,
    )
    return e_afqmc, err_afqmc


def run_afqmc_fp(options=None, script=None, mpi_prefix=None, nproc=None, tmpdir=None):
    config.setup_jax()
    comm = config.setup_comm()
    (
        ham_data,
        ham,
        prop,
        trial,
        wave_data,
        trial_ket,
        wave_data_ket,
        sampler,
        observable,
        options,
    ) = utils.setup_afqmc_fp(options, options["tmpdir"])

    if (
        options["symmetry_projector"] == "s2"
        and isinstance(trial, wavefunctions.uhf)
        and options["optimize_trial"]
    ):
        orbitals = optimize_trial.optimize_trial(ham_data, trial, wave_data)
        wave_data["mo_coeff"] = orbitals
        nao = orbitals[0].shape[0]
        wave_data["rdm1"] = jnp.vstack(
            [orbitals[0] @ orbitals[0].T.conj(), orbitals[1] @ orbitals[1].T.conj()]
        ).reshape(-1, nao, nao)

    driver.fp_afqmc(
        ham_data,
        ham,
        prop,
        trial,
        wave_data,
        trial_ket,
        wave_data_ket,
        sampler,
        observable,
        options,
        comm,
    )
