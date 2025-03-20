import os
import pickle
from functools import partial

import numpy as np

from ad_afqmc import config

print = partial(print, flush=True)


def run_afqmc(options=None, mpi_prefix=None, nproc=None, tmpdir=None):
    if tmpdir is None:
        try:
            with open("tmpdir.txt", "r") as f:
                tmpdir = f.read().strip()
        except:
            tmpdir = "."
    if options is None:
        options = {}
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
        except ImportError:
            use_mpi = False
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
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script} {tmpdir} {gpu_flag} {mpi_flag}"
    )
    try:
        ene_err = np.loadtxt(tmpdir + "/ene_err.txt")
    except:
        print("AFQMC did not execute correctly.")
        ene_err = 0.0, 0.0
    return ene_err[0], ene_err[1]


def run_afqmc_fp(options=None, script=None, mpi_prefix=None, nproc=None):
    if options is None:
        options = {}
    with open("options.bin", "wb") as f:
        pickle.dump(options, f)
    if script is None:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        script = f"{dir_path}/launch_script.py"
    if mpi_prefix is None:
        mpi_prefix = "mpirun "
    if nproc is not None:
        mpi_prefix += f"-np {nproc} "
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script}"
    )
    # ene_err = np.loadtxt('ene_err.txt')
    # return ene_err[0], ene_err[1]
