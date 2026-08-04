from trot import config

config.configure_once()

from typing import Any

import pytest
import jax.numpy as jnp
from pyscf import gto, scf
from trot.afqmc import Afqmc, AfqmcFp
from trot.prop.types import QmcParams, QmcParamsFp


def build_mf() -> Any:
    mol = gto.M(
        atom="""
        N        0.0000000000      0.0000000000      0.0000000000
        H        1.0225900000      0.0000000000      0.0000000000
        H       -0.2281193615      0.9968208791      0.0000000000
        """,
        basis="sto-6g",
        spin=1,
    )
    mf = scf.UHF(mol).newton()
    mf.kernel()
    return mf


mf_obj = build_mf()


@pytest.mark.parametrize(
    "mf, walker_kind, e_ref, err_ref",
    [
        (mf_obj, "unrestricted", -55.43066756011652, 0.00761980459817991),
    ],
)
def test_io(mf, tmp_path, params, walker_kind, e_ref, err_ref):
    h5_file = str(tmp_path / "nh2.h5")
    myafqmc = Afqmc(mf)
    myafqmc.chol_cut = 1e-6
    myafqmc.save_staged(h5_file)

    af = Afqmc.from_staged(h5_file)
    af.params = params
    af.mixed_precision = False
    af.walker_kind = walker_kind
    mean, err = af.kernel()
    assert jnp.isclose(mean, e_ref), (mean, e_ref, mean - e_ref)
    assert jnp.isclose(err, err_ref), (err, err_ref, err - err_ref)


@pytest.fixture(scope="module")
def params():
    return QmcParams(
        n_eql_blocks=4,
        n_blocks=20,
        seed=1234,
        n_walkers=5,
        error_method="blocking",
    )


@pytest.mark.parametrize(
    "mf, walker_kind, e_ref, err_ref",
    [
        (mf_obj, "unrestricted", -55.4067231213, 1.8318234e-02),
    ],
)
def test_io_fp(mf, tmp_path, params_fp, walker_kind, e_ref, err_ref):
    h5_file = str(tmp_path / "nh2.h5")
    myafqmc = AfqmcFp(mf)
    myafqmc.chol_cut = 1e-6
    myafqmc.save_staged(h5_file)

    af = AfqmcFp.from_staged(h5_file)
    af.params = params_fp
    af.mixed_precision = False
    af.walker_kind = walker_kind
    mean, err = af.kernel()
    assert jnp.isclose(mean[-1].real, e_ref), (mean[-1].real, e_ref, mean[-1].real - e_ref)
    assert jnp.isclose(err[-1].real, err_ref), (err[-1].real, err_ref, err[-1].real - err_ref)


@pytest.fixture(scope="module")
def params_fp():
    return QmcParamsFp(
        n_blocks=1,
        seed=1234,
        n_walkers=5,
        n_traj=4,
        ene0=mf_obj.e_tot,
    )


if __name__ == "__main__":
    pytest.main([__file__])
