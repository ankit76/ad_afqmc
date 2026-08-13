from trot.afqmc import Afqmc, AfqmcFp
from trot.prop.types import QmcParams, QmcParamsFp

from typing import Any

import pytest
from pyscf import gto, scf


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
    mf = scf.UHF(mol)
    mf.kernel()

    for i in range(2):
        mo1 = mf.stability()[0]
        mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
    mf.stability()
    return mf


mf_obj = build_mf()


@pytest.mark.parametrize(
    "mf, walker_kind",
    [
        (mf_obj, "unrestricted"),
    ],
)
def test_io(mf, tmp_path, params, walker_kind):
    h5_file = str(tmp_path / "nh2.h5")
    af = Afqmc(mf)
    af.params = params
    af.mixed_precision = False
    af.walker_kind = walker_kind
    af.save_staged(h5_file)
    e1, err1 = af.kernel()

    af = Afqmc.from_staged(h5_file)
    af.params = params
    af.mixed_precision = False
    af.walker_kind = walker_kind
    e2, err2 = af.kernel()
    assert abs(e1 - e2) < 1e-6, (e1, e2)
    assert abs(err1 - err2) < 1e-6, (err1, err2)


@pytest.fixture(scope="module")
def params():
    return QmcParams(
        n_eql_blocks=4,
        n_blocks=20,
        seed=1234,
        n_walkers=5,
    )


@pytest.mark.parametrize(
    "mf, walker_kind",
    [
        (mf_obj, "unrestricted"),
    ],
)
def test_io_fp(mf, tmp_path, params_fp, walker_kind):
    h5_file = str(tmp_path / "nh2.h5")
    af = AfqmcFp(mf)
    af.params = params_fp
    af.mixed_precision = False
    af.walker_kind = walker_kind
    af.save_staged(h5_file)
    e1, err1 = af.kernel()

    af = AfqmcFp.from_staged(h5_file)
    af.params = params_fp
    af.mixed_precision = False
    af.walker_kind = walker_kind
    e2, err2 = af.kernel()
    assert abs(e1[-1].real - e2[-1].real) < 1e-6, (e1[-1].real, e2[-1].real)
    assert abs(err1[-1].real - err2[-1].real) < 1e-6, (err1[-1].real, err2[-1].real)


@pytest.fixture(scope="module")
def params_fp():
    return QmcParamsFp(
        n_blocks=1,
        seed=1234,
        n_walkers=5,
        n_traj=10,
        ene0=mf_obj.e_tot,
    )


if __name__ == "__main__":
    pytest.main([__file__])
