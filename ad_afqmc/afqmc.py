import os
import pickle
import shlex
import subprocess
from functools import partial
from typing import Optional, Union

import numpy as np
from pyscf import __config__, scf
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

from ad_afqmc import config, pyscf_interface

print = partial(print, flush=True)


class AFQMC:
    """
    AFQMC class.

    Attributes:
        mf_or_cc : Union[scf.uhf.UHF, scf.rhf.RHF, scf.rohf.ROHF, CCSD, UCCSD]
            The mean-field or CCSD object.
        basis_coeff : None
            Basis coefficients used for AFQMC propagation.
        norb_frozen : int
            Number of frozen orbitals.
        chol_cut : float
            Cholesky cut-off.
        integrals : None
            Dictionary of integrals in an orthonormal basis, {"h0": enuc, "h1": h1e, "h2": eri}.
        mpi_prefix : None
            MPI prefix, used to launch MPI processes.
        nproc : int
            Number of processes, if using MPI.
        dt : float
            AFQMC propagation time step.
        n_walkers : int
            Number of walkers.
        n_prop_steps : int
            Number of propagation steps.
        n_ene_blocks : int
            Number of energy blocks.
        n_sr_blocks : int
            Number of stochastic reconfiguration blocks.
        n_blocks : int
            Number of blocks.
        n_ene_blocks_eql : int
            Number of energy blocks for equilibration.
        n_sr_blocks_eql : int
            Number of stochastic reconfiguration blocks for equilibration.
        seed : int
            Random seed.
        n_eql : int
            Number of equilibration blocks.
        ad_mode : None
            AD mode.
        orbital_rotation : bool
            Orbital rotation option for AD.
        do_sr : bool
            Stochastic reconfiguration, relevant for AD.
        walker_type : str
            Walker type, "restricted" or "unrestricted".
        symmetry : bool
            Symmetry, relevant for AD.
        save_walkers : bool
            Save walkers.
        trial : str
            Trial.
        ene0 : float
            Initial energy used in free projection.
        n_batch : int
            Number of batches, relevant for GPU calculations.
        tmpdir : str
            Temporary directory.
    """

    def __init__(
        self, mf_or_cc: Union[scf.uhf.UHF, scf.rhf.RHF, scf.rohf.ROHF, CCSD, UCCSD]
    ):
        self.mf_or_cc = mf_or_cc
        self.basis_coeff = None
        self.norb_frozen = 0
        self.chol_cut = 1e-5
        self.integrals = None  # custom integrals
        self.mpi_prefix = None
        self.nproc = 1
        self.dt = 0.005
        self.n_walkers = 50
        self.n_prop_steps = 50
        self.n_ene_blocks = 1
        self.n_sr_blocks = 5
        self.n_blocks = 200
        self.n_ene_blocks_eql = 1
        self.n_sr_blocks_eql = 5
        self.seed = np.random.randint(1, int(1e6))
        self.n_eql = 20
        self.ad_mode = None
        self.orbital_rotation = True
        self.do_sr = True
        self.walker_type = "restricted"
        self.symmetry = False
        self.save_walkers = False
        if isinstance(mf_or_cc, scf.uhf.UHF) or isinstance(mf_or_cc, scf.rohf.ROHF):
            self.trial = "uhf"
        elif isinstance(mf_or_cc, scf.rhf.RHF):
            self.trial = "rhf"
        elif isinstance(mf_or_cc, UCCSD):
            self.trial = "ucisd"
        elif isinstance(mf_or_cc, CCSD):
            self.trial = "cisd"
        else:
            self.trial = None
        self.ene0 = 0.0
        self.n_batch = 1
        self.tmpdir = __config__.TMPDIR + f"/afqmc{np.random.randint(1, int(1e6))}/"

    def kernel(self, dry_run=False):
        """
        Run AFQMC.

        Args:
            dry_run : bool
                If True, writes input files like integrals to disk, useful for large calculations where one would like to run the trial generation and AFQMC calculations on different machines, on CPU and GPU, for example.
        """
        os.makedirs(self.tmpdir, exist_ok=True)
        pyscf_interface.prep_afqmc(
            self.mf_or_cc,
            basis_coeff=self.basis_coeff,
            norb_frozen=self.norb_frozen,
            chol_cut=self.chol_cut,
            integrals=self.integrals,
            tmpdir=self.tmpdir,
        )
        options = {}
        for attr in dir(self):
            if (
                attr
                not in [
                    "mf_or_cc",
                    "basis_coeff",
                    "norb_frozen",
                    "chol_cut",
                    "integrals",
                    "mpi_prefix",
                    "nproc",
                ]
                and not attr.startswith("__")
                and not callable(getattr(self, attr))
            ):
                options[attr] = getattr(self, attr)
        with open(self.tmpdir + "/options.bin", "wb") as f:
            pickle.dump(options, f)
        if dry_run:
            with open("tmpdir.txt", "w") as f:
                f.write(self.tmpdir)
            return self.tmpdir
        else:
            return run_afqmc(
                mpi_prefix=self.mpi_prefix, nproc=self.nproc, tmpdir=self.tmpdir
            )


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
        except:
            tmpdir = "."
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
        except ImportError:
            use_mpi = False
            print(f"# Unable to import mpi4py, not using MPI.")
        # use_mpi = False
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
