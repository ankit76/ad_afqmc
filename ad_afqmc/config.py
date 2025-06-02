import os
import sys
import platform
import socket
from dataclasses import dataclass

import numpy as np

from ad_afqmc.logger import Logger

afqmc_config = {"use_gpu": False, "use_mpi": None}
rank = 0


def is_jupyter_notebook():
    try:
        from IPython.core.getipython import get_ipython

        ipython = get_ipython()
        if ipython is not None and "IPKernelApp" in ipython.config:
            return True
        else:
            return False
    except ImportError:
        return False


if is_jupyter_notebook():
    afqmc_config["use_mpi"] = False


class not_a_comm:
    def __init__(self):
        self.size = 1
        self.rank = 0

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank

    def Barrier(self):
        pass

    def Reduce(self, sendbuf, recbuf, op=None, root=0):
        np.copyto(recbuf[0], sendbuf[0])

    def Bcast(self, buf, root=0):
        pass

    def bcast(self, buf, root=0):
        return buf

    def Gather(self, sendbuf, recbuf, root=0):
        recbuf[:] = sendbuf

    def Scatter(self, sendbuf, recbuf, root=0):
        recbuf[:] = sendbuf


@dataclass
class not_MPI:
    FLOAT = None
    INT = None
    SUM = None
    COMM_WORLD = not_a_comm()

    def Finalize(self):
        pass


def setup_jax(log: Logger):
    from jax import config

    config.update("jax_enable_x64", True)
    # breaking change in random number generation in jax v0.5
    config.update("jax_threefry_partitionable", False)

    if afqmc_config["use_gpu"]: # == True
        config.update("jax_platform_name", "gpu")
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        # TODO: add gpu performance xla flags
        hostname = socket.gethostname()
        system_type = platform.system()
        machine_type = platform.machine()
        processor = platform.processor()
        log.log(f"# Hostname: {hostname}")
        log.log(f"# System Type: {system_type}")
        log.log(f"# Machine Type: {machine_type}")
        log.log(f"# Processor: {processor}")
        uname_info = platform.uname()
        log.log("# Using GPU.")
        log.log(f"# System: {uname_info.system}")
        log.log(f"# Node Name: {uname_info.node}")
        log.log(f"# Release: {uname_info.release}")
        log.log(f"# Version: {uname_info.version}")
        log.log(f"# Machine: {uname_info.machine}")
        log.log(f"# Processor: {uname_info.processor}")
    else:
        #afqmc_config["use_gpu"] = False
        config.update("jax_platform_name", "cpu")
        os.environ["XLA_FLAGS"] = (
            "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        )


def setup_comm(log):
    MPI = setup_comm_no_print()
    if rank == 0 and afqmc_config["use_gpu"] == False:
        hostname = socket.gethostname()
        system_type = platform.system()
        machine_type = platform.machine()
        processor = platform.processor()
        log.log_0(f"# Hostname: {hostname}")
        log.log_0(f"# System Type: {system_type}")
        log.log_0(f"# Machine Type: {machine_type}")
        log.log_0(f"# Processor: {processor}")
    return MPI

def setup_comm_no_print():
    if afqmc_config["use_gpu"] == True:
        afqmc_config["use_mpi"] = False
    if afqmc_config["use_mpi"] == True:
        from mpi4py import MPI
    else:
        MPI = not_MPI()
    global rank
    rank = MPI.COMM_WORLD.Get_rank()

    return MPI

def mpi_print(message, flush=True):
    if rank == 0:
        print(message, flush=flush)
