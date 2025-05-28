import unittest
import numpy as np
import os
from pyscf import scf, gto, cc
from ad_afqmc import pyscf_interface, run_afqmc

tmpdir = "tmp"
if not os.path.exists(tmpdir):
    os.makedirs(tmpdir)

default_options = {
    "dt": 0.005,
    "n_eql": 3,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 20,
    "n_prop_steps": 50,
    "n_walkers": 5,
    "seed": 8,
    "trial": "",
    "walker_type": "",
}


def check(testcase, obj, options, expected_e, atol, mpi):
    pyscf_interface.prep_afqmc(obj, tmpdir=tmpdir, chol_cut=1e-12)

    mpi_prefix = "mpirun " if mpi else None
    nproc = 2 if mpi else None

    ene, _ = run_afqmc.run_afqmc(
        options=options, mpi_prefix=mpi_prefix, nproc=nproc, tmpdir=tmpdir
    )
    testcase.assertAlmostEqual(ene, expected_e, delta=atol)
    return ene


def set_options(trial, walker_type):
    opts = default_options.copy()
    opts["trial"] = trial
    opts["walker_type"] = walker_type
    return opts


def use_mpi_tests():
    return os.environ.get("AFQMC_MPI") == "1" or os.environ.get("AFQMC_ALL") == "1"

def use_non_mpi_tests():
    return os.environ.get("AFQMC_ALL") == "1" or os.environ.get("AFQMC_MPI") != "1"


class TestAFQMC(unittest.TestCase):

    # Trick to avoid it running with pytest
    __test__ = False
    
    def test_h2o(self):
        mol = gto.M(
            atom="""
            O        0.0000000000      0.0000000000      0.0000000000
            H        0.9562300000      0.0000000000      0.0000000000
            H       -0.2353791634      0.9268076728      0.0000000000
            """,
            basis="6-31g",
            spin=0,
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.kernel()

        if use_non_mpi_tests():
            check(self, mf, set_options("rhf", "restricted"), -76.14187898749667, 1e-5, False)
            check(self, mf, set_options("rhf", "unrestricted"), -76.14187898749667, 1e-5, False)

        if use_mpi_tests():
            check(self, mf, set_options("rhf", "restricted"), -76.13233451013845, 1e-5, True)
            check(self, mf, set_options("rhf", "unrestricted"), -76.13233451013845, 1e-5, True)

        mycc = cc.RCCSD(mf)
        mycc.kernel()
        if use_non_mpi_tests():
            check(self, mycc, set_options("cisd", "restricted"), -76.12243596268871, 1e-5, False)
            mycc.frozen = 1
            mycc.kernel()
            check(self, mycc, set_options("cisd", "restricted"), -76.1215017439916, 1e-5, False)

        if use_mpi_tests():
            mycc.frozen = None
            mycc.kernel()
            check(self, mycc, set_options("cisd", "restricted"), -76.12240255883655, 1e-5, True)
            mycc.frozen = 1
            mycc.kernel()
            check(self, mycc, set_options("cisd", "restricted"), -76.12149845139608, 1e-5, True)

    def test_nh2(self):
        mol = gto.M(
            atom="""
            N        0.0000000000      0.0000000000      0.0000000000
            H        1.0225900000      0.0000000000      0.0000000000
            H       -0.2281193615      0.9968208791      0.0000000000
            """,
            basis="6-31g",
            spin=1,
            verbose=0,
        )
        mf = scf.UHF(mol)
        mf.kernel()

        if use_non_mpi_tests():
            check(self, mf, set_options("uhf", "restricted"), -55.65599052681822, 1e-5, False)
            check(self, mf, set_options("uhf", "unrestricted"), -55.655991946870884, 1e-5, False)

        if use_mpi_tests():
            check(self, mf, set_options("uhf", "restricted"), -55.648220092580814, 1e-5, True)
            check(self, mf, set_options("uhf", "unrestricted"), -55.648219500346904, 1e-5, True)

        mycc = cc.UCCSD(mf)
        mycc.kernel()
        if use_non_mpi_tests():
            check(self, mycc, set_options("ucisd", "restricted"), -55.636250653696585, 1e-5, False)
            check(self, mycc, set_options("ucisd", "unrestricted"), -55.636251418009444, 1e-5, False)
            mycc.frozen = 1
            mycc.kernel()
            check(self, mycc, set_options("ucisd", "restricted"), -55.63515276697131, 1e-5, False)

        if use_mpi_tests():
            mycc.frozen = None
            mycc.kernel()
            check(self, mycc, set_options("ucisd", "restricted"), -55.636109651463116, 1e-5, True)
            check(self, mycc, set_options("ucisd", "unrestricted"), -55.63613901998812, 1e-5, True)
            mycc.frozen = 1
            mycc.kernel()
            check(self, mycc, set_options("ucisd", "restricted"), -55.63511844686504, 1e-5, True)


if __name__ == "__main__":
    unittest.main()

