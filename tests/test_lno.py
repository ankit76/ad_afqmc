import numpy as np

from trot import config

config.configure_once()
import jax.numpy as jnp
import pytest
from pyscf import gto, scf

from trot import testing
from trot.prop.types import QmcParams
from trot.trial.rhf import RhfTrial
from trot.afqmc import AfqmcLnoFrag, run_afqmc_lno_helper
from trot.staging import load as load_staged


def _make_random_rhf_trial(key, norb, nocc):
    return RhfTrial(mo_coeff=testing.rand_orthonormal_cols(key, norb, nocc))


def mf():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
    )
    mf = scf.RHF(mol).density_fit()
    mf.kernel()  # type: ignore
    return mf


mf = mf()  # type: ignore


@pytest.mark.parametrize(
    "mf, e_ref, err_ref, norb_frozen",
    [
        (mf, -0.06714345, 0.01170026, 0),
        (mf, -0.04946941, 0.00817172, 1),
    ],
)
def test_calc_rhf(mf, params, e_ref, err_ref, norb_frozen):
    mo_coeff = mf.mo_coeff
    orbs_frozen = np.array([i for i in range(norb_frozen)])
    norb_act = mf.mo_coeff.shape[1] - norb_frozen
    nactocc = mf.mol.nelectron // 2 - norb_frozen
    prjlo = np.array([[0] for i in range(nactocc - 1)] + [[1]])

    elcorr_afqmc, err_afqmc = run_afqmc_lno_helper(
        mf,
        mo_coeff=mo_coeff,
        norb_act=norb_act,
        nelec_act=nactocc * 2,
        frozen_orbitals=orbs_frozen,
        n_walkers=params.n_walkers,
        nblocks=params.n_blocks,
        seed=params.seed,
        chol_cut=1e-5,
        target_error=0,
        dt=0.01,
        prjlo=prjlo,
        n_eql=params.n_eql_blocks,
    )

    assert jnp.isclose(elcorr_afqmc, e_ref), (elcorr_afqmc, e_ref, elcorr_afqmc - e_ref)
    assert jnp.isclose(err_afqmc, err_ref), (err_afqmc, err_ref, err_afqmc - err_ref)


def test_lno_stage_roundtrips_array_frozen_orbitals(tmp_path):
    frozen_orbitals = np.array([0], dtype=np.int64)
    path = tmp_path / "lno_stage.h5"

    af = AfqmcLnoFrag(mf, frozen_orbitals=frozen_orbitals)
    af.save_staged(path)

    staged = load_staged(path)
    af_loaded = AfqmcLnoFrag.from_staged(path)

    assert np.array_equal(staged.ham.frozen, frozen_orbitals)
    assert np.array_equal(staged.trial.frozen, frozen_orbitals)
    assert np.array_equal(staged.meta["frozen"], frozen_orbitals)
    assert af_loaded.frozen_orbitals is not None
    assert np.array_equal(af_loaded.frozen_orbitals, frozen_orbitals)


@pytest.fixture(scope="module")
def params():
    return QmcParams(
        n_eql_blocks=4,
        n_blocks=20,
        seed=1234,
        n_walkers=5,
        error_method="blocking",
    )


if __name__ == "__main__":
    pytest.main([__file__])
