from trot.afqmc import AfqmcFp
from trot.prop.types import QmcParamsFp

import pytest
from pyscf import gto, scf


@pytest.mark.parametrize(
    "walker_kind, e_ref, err_ref",
    [
        ("restricted", -75.7386645902, 1.0340682e-02),
    ],
)
def test_calc_rhf_hamiltonian(mf, params, walker_kind, e_ref, err_ref):
    af = AfqmcFp(mf)
    af.params = params
    af.walker_kind = walker_kind
    af.mixed_precision = False
    af.chol_cut = 1e-6
    e, err = af.kernel()

    assert abs(e[-1].real - e_ref) < 1e-6, (e[-1].real, e_ref)
    assert abs(err[-1].real - err_ref) < 1e-6, (err[-1].real, e_ref)


@pytest.fixture(scope="module")
def mf():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mf


@pytest.fixture(scope="module")
def params():
    return QmcParamsFp(
        n_blocks=1,
        n_prop_steps=100,
        seed=6,
        n_walkers=5,
        n_traj=10,
        dt=0.05,
        ene0=-75.67863248572299,
    )


if __name__ == "__main__":
    pytest.main([__file__])
