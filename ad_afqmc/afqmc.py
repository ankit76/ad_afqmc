import os
import sys
import pickle
from functools import partial
from typing import Union

import numpy as np
from pyscf import __config__, scf
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

from ad_afqmc import pyscf_interface, run_afqmc
from ad_afqmc.options import Options

print = partial(print, flush=True)


class AFQMC():
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
        self, mf_or_cc: Union[scf.uhf.UHF, scf.rhf.RHF, scf.rohf.ROHF, CCSD, UCCSD]
    ):
        self.mf_or_cc = mf_or_cc
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
        self.tmpdir = __config__.TMPDIR + f"/afqmc{np.random.randint(1, int(1e6))}/"

        # Set default options
        for key, val in Options(mf_or_cc).to_dict().items():
            setattr(self, key, val)

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
            verbose=self.verbose,
        )

        options = {}
        for attr in dir(self):
            if attr in Options.get_keys():
                options[attr] = getattr(self, attr)
        with open(self.tmpdir + "/options.bin", "wb") as f:
            pickle.dump(options, f)

        if dry_run:
            with open("tmpdir.txt", "w") as f:
                f.write(self.tmpdir)
            return self.tmpdir
        else:
            return run_afqmc.run_afqmc(
                options=Options.from_dict(options), mpi_prefix=self.mpi_prefix, nproc=self.nproc, tmpdir=self.tmpdir
            )
