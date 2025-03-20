import os

import numpy as np

from ad_afqmc import config

config.setup_jax()
import pytest

from ad_afqmc import run_afqmc

seed = 98

# __test__ = False

tmpdir = os.path.dirname(os.path.abspath(__file__))


def test_energy_mpi():
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
    assert np.isclose(ene, -3.239302058353345, atol=1e-5)


def test_jvp_h1e():
    options = {
        "n_eql": 2,
        "n_ene_blocks": 1,
        "n_sr_blocks": 10,
        "n_blocks": 10,
        "n_walkers": 50,
        "seed": seed,
        "trial": "uhf",
        "walker_type": "uhf",
        "ad_mode": "forward",
    }
    ene, _ = run_afqmc.run_afqmc(options=options, nproc=2, tmpdir=tmpdir)
    assert np.isclose(ene, -3.2359788941631957, atol=1e-5)
    obs_err = np.loadtxt(f"{tmpdir}/obs_err.txt")
    assert np.isclose(obs_err[0], -11.9139997, atol=1e-5)


def test_vjp_rdm():
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
    assert np.isclose(ene, -3.2359788941631957, atol=1e-5)
    rdm1 = np.load("rdm1_afqmc.npz")["rdm1"]
    assert np.isclose(np.trace(rdm1[0]), 3)
    assert np.isclose(np.trace(rdm1[1]), 3)
    assert np.isclose(np.sum(rdm1), 7.820594345322708, atol=1e-5)


def test_energy():
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
    ene, _ = run_afqmc.run_afqmc(options=options, mpi_prefix="", tmpdir=tmpdir)
    assert np.isclose(ene, -3.238196747261496, atol=1e-5)


if __name__ == "__main__":
    test_energy_mpi()
    test_jvp_h1e()
    test_vjp_rdm()
    test_energy()
