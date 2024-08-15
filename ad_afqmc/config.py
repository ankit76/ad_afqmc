import os
from dataclasses import dataclass

afqmc_config = {"use_gpu": False}


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

    def Reduce(self, sendbuf, recbuf, op, root=0):
        recbuf[:] = sendbuf

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


def setup_jax():
    from jax.config import config

    config.update("jax_enable_x64", True)
    if afqmc_config["use_gpu"] == True:
        print("Using GPU.")
        # TODO: add gpu performance xla flags
    else:
        afqmc_config["use_gpu"] = False
        config.update("jax_platform_name", "cpu")
        os.environ["XLA_FLAGS"] = (
            "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        )


def setup_comm():
    if "use_mpi" not in afqmc_config:
        if afqmc_config["use_gpu"] == True:
            afqmc_config["use_mpi"] = False
        else:
            afqmc_config["use_mpi"] = True
    if afqmc_config["use_mpi"] == True:
        from mpi4py import MPI
    else:
        MPI = not_MPI()

    return MPI
