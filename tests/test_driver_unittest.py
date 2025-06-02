import os
import sys
import numpy as np
import unittest

from ad_afqmc import config
from ad_afqmc.logger import Logger

log = Logger(sys.stdout, 3)
config.setup_jax(log)
from ad_afqmc import run_afqmc

seed = 98

tmpdir = os.path.dirname(os.path.abspath(__file__))

class TestDriver(unittest.TestCase):

    # Trick to avoid it running with pytest
    __test__ = False

    def test_energy_mpi(self):
        options = {
            "n_eql": 1,
            "n_ene_blocks": 1,
            "n_sr_blocks": 10,
            "n_blocks": 10,
            "n_walkers": 50,
            "seed": seed,
            "trial": "uhf",
            "walker_type": "unrestricted",
        }
        ene, _ = run_afqmc.run_afqmc(
            options=options, mpi_prefix="mpirun ", nproc=2, tmpdir=tmpdir
        )
        self.assertTrue(np.isclose(ene, -3.239302058353345, atol=1e-5))

    def test_jvp_h1e(self):
        options = {
            "n_eql": 2,
            "n_ene_blocks": 1,
            "n_sr_blocks": 10,
            "n_blocks": 10,
            "n_walkers": 50,
            "seed": seed,
            "trial": "uhf",
            "walker_type": "unrestricted",
            "ad_mode": "forward",
        }
        ene, _ = run_afqmc.run_afqmc(options=options, nproc=2, tmpdir=tmpdir)
        self.assertTrue(np.isclose(ene, -3.2359788941631957, atol=1e-5))
        obs_err = np.loadtxt(f"{tmpdir}/obs_err.txt")
        self.assertTrue(np.isclose(obs_err[0], -11.9139997, atol=1e-5))
    
    
    def test_vjp_rdm(self):
        options = {
            "n_eql": 2,
            "n_ene_blocks": 1,
            "n_sr_blocks": 10,
            "n_blocks": 10,
            "n_walkers": 50,
            "seed": seed,
            "trial": "uhf",
            "walker_type": "unrestricted",
            "ad_mode": "reverse",
        }
        ene, _ = run_afqmc.run_afqmc(options=options, nproc=2, tmpdir=tmpdir)
        self.assertTrue(np.isclose(ene, -3.2359788941631957, atol=1e-5))
        rdm1 = np.load("rdm1_afqmc.npz")["rdm1"]
        self.assertTrue(np.isclose(np.trace(rdm1[0]), 3))
        self.assertTrue(np.isclose(np.trace(rdm1[1]), 3))
        self.assertTrue(np.isclose(np.sum(rdm1), 7.820594345322708, atol=1e-5))
    
    
    def test_energy(self):
        options = {
            "n_eql": 1,
            "n_ene_blocks": 1,
            "n_sr_blocks": 10,
            "n_blocks": 10,
            "n_walkers": 50,
            "seed": seed,
            "trial": "uhf",
            "walker_type": "unrestricted",
        }
        ene, _ = run_afqmc.run_afqmc(options=options, tmpdir=tmpdir)
        self.assertTrue(np.isclose(ene, -3.238196747261496, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
