import os
import pickle
from functools import partial
from typing import Union, Optional

import numpy as np
from pyscf import __config__, scf
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

from ad_afqmc import pyscf_interface, run_afqmc, grad_utils

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
        vhs_mixed_precision : bool
            Use mixed precision for VHS.
        trial_mixed_precision : bool
            Use mixed precision for trial (CISD).
        memory_mode: str
            Memory mode, "high" or "low" (CISD).
        tmpdir : str
            Temporary directory.
    """

    def __init__(
        self, mf_or_cc: Union[scf.uhf.UHF, scf.rhf.RHF, scf.rohf.ROHF, CCSD, UCCSD],
        mf_or_cc_ket: Optional[Union[scf.uhf.UHF, scf.rhf.RHF, scf.rohf.ROHF, CCSD, UCCSD]] = None
    ):
        self.mf_or_cc = mf_or_cc
        self.mf_or_cc_ket = mf_or_cc_ket if mf_or_cc_ket is not None else mf_or_cc
        self.basis_coeff = None
        frozen = getattr(mf_or_cc, "frozen", 0)
        if isinstance(frozen, int):
            self.norb_frozen = frozen
        else:
            print("Warning: Frozen is not an integer, assuming 0 frozen orbitals.")
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
        self.dR = 1e-5  # displacement used in finite difference to calculate integral gradients for ad_mode = nuc_grad
        self.free_projection = False
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

        if isinstance(mf_or_cc_ket, scf.uhf.UHF) or isinstance(mf_or_cc_ket, scf.rohf.ROHF):
            self.trial_ket = "uhf"
        elif isinstance(mf_or_cc_ket, scf.rhf.RHF):
            self.trial_ket = "rhf"
        elif isinstance(mf_or_cc_ket, UCCSD):
            self.trial_ket = "uccsd"
        elif isinstance(mf_or_cc_ket, CCSD):
            self.trial_ket = "ccsd"
        else:
            self.trial_ket = self.trial

        self.ene0 = 0.0
        self.n_batch = 1
        self.vhs_mixed_precision = False
        self.trial_mixed_precision = False
        self.memory_mode = "low"
        self.tmpdir = __config__.TMPDIR + f"/afqmc{np.random.randint(1, int(1e6))}/"

    def kernel(self, dry_run=False):
        """
        Run AFQMC.

        Args:
            dry_run : bool
                If True, writes input files like integrals to disk, useful for large calculations where one would like to run the trial generation and AFQMC calculations on different machines, on CPU and GPU, for example.
        """
        os.makedirs(self.tmpdir, exist_ok=True)
        if self.ad_mode != "nuc_grad":
            pyscf_interface.prep_afqmc(
                self.mf_or_cc,
                basis_coeff=self.basis_coeff,
                norb_frozen=self.norb_frozen,
                chol_cut=self.chol_cut,
                integrals=self.integrals,
                tmpdir=self.tmpdir,
            )
        else:
            grad_utils.prep_afqmc_nuc_grad(self.mf_or_cc, self.dR, tmpdir=self.tmpdir)
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
        elif options["free_projection"]:
            if (self.mf_or_cc_ket != self.mf_or_cc):
                pyscf_interface.read_pyscf_ccsd(self.mf_or_cc_ket, options["tmpdir"])
            #options=None, script=None, mpi_prefix=None, nproc=None
            return run_afqmc.run_afqmc_fp(
                options = options, 
                mpi_prefix=self.mpi_prefix, 
                nproc=self.nproc,
                tmpdir=self.tmpdir
            )
        else:
            return run_afqmc.run_afqmc(
                mpi_prefix=self.mpi_prefix, nproc=self.nproc, tmpdir=self.tmpdir
            )
