import os
import subprocess
import time
from subprocess import PIPE

import numpy as np

os.environ[
    "XLA_FLAGS"
] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
import pickle
from functools import partial

# os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
from jax import dtypes, jvp, random, vjp

# from ad_afqmc import propagation, sampler, stat_utils

print = partial(print, flush=True)


def run_afqmc(options=None, script=None, mpi_prefix=None, nproc=None):
    if options is None:
        options = {}
    with open("options.bin", "wb") as f:
        pickle.dump(options, f)
    if script is None:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        script = f"{dir_path}/mpi_jax.py"
    if mpi_prefix is None:
        mpi_prefix = "mpirun "
        if nproc is not None:
            mpi_prefix += f"-np {nproc} "
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script}"
    )
    # os.system("mpirun hostname")
    # p = subprocess.run(["mpirun", "python", f"{script}"], stdout=PIPE, stderr=PIPE)
    # print(p.stderr)
    try:
        ene_err = np.loadtxt("ene_err.txt")
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
        script = f"{dir_path}/mpi_jax.py"
    if mpi_prefix is None:
        mpi_prefix = "mpirun "
        if nproc is not None:
            mpi_prefix += f"-np {nproc} "
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script}"
    )
    # ene_err = np.loadtxt('ene_err.txt')
    # return ene_err[0], ene_err[1]
