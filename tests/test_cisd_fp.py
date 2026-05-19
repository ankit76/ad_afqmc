from trot.afqmc import AfqmcFp
from trot.prop.types import QmcParamsFp

import pytest
from pyscf import gto, scf, cc


@pytest.mark.parametrize(
    "walker_kind, e_ref, err_ref",
    [
        ("restricted", -76.1188891569, 1.0799288e-03),
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
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="6-31G",
    )
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = cc.CCSD(mf)
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
        ene0=-76.11915086149004,
    )


if __name__ == "__main__":
    pytest.main([__file__])
