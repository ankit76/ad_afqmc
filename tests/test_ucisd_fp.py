from trot.afqmc import AfqmcFp
from trot.prop.types import QmcParamsFp

import pytest
from pyscf import gto, scf, cc


@pytest.mark.parametrize(
    "walker_kind, e_ref, err_ref",
    [
        ("unrestricted", -55.6037313899, 8.0228047e-04),
    ],
)
def test_calc_uhf_hamiltonian(mycc, params, walker_kind, e_ref, err_ref):
    af = AfqmcFp(mycc)
    af.params = params
    af.walker_kind = walker_kind
    af.mixed_precision = False
    af.chol_cut = 1e-6
    e, err = af.kernel()
    assert abs(e[-1].real - e_ref), (e[-1].real, e_ref)
    assert abs(err[-1].real - err_ref), (err[-1].real, err_ref)


@pytest.fixture(scope="module")
def mycc():
    mol = gto.M(
        atom="""
        N                 -1.67119571   -1.44021737    0.00000000
        H                 -2.12619571   -0.65213425    0.00000000
        H                 -0.76119571   -1.44021737    0.00000000
        """,
        basis="6-31G",
        spin=1,
    )
    mf = scf.UHF(mol)
    mf.kernel()
    mycc = cc.UCCSD(mf)
    mycc.kernel()
    return mycc


@pytest.fixture(scope="module")
def params():
    return QmcParamsFp(
        n_blocks=1,
        n_prop_steps=100,
        seed=6,
        n_walkers=5,
        n_traj=10,
        dt=0.05,
        ene0=-55.60298562645659,
    )


if __name__ == "__main__":
    pytest.main([__file__])
