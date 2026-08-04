from trot import config

config.configure_once()

import jax
import jax.numpy as jnp
import pytest
from pyscf import gto, scf

from trot import driver
from trot.afqmc import Afqmc
from trot.prop.types import QmcParams
from trot import testing

from trot.core.ops import k_energy, k_force_bias
from trot.meas.rhf import (
    make_rhf_meas_ops,
)
from trot.trial.rhf import RhfTrial, make_rhf_trial_ops


def _make_random_rhf_trial(key, norb, nocc):
    return RhfTrial(mo_coeff=testing.rand_orthonormal_cols(key, norb, nocc))


@pytest.mark.parametrize("walker_kind", ["restricted", "unrestricted", "generalized"])
def test_auto_force_bias_matches_manual_rhf(walker_kind):
    norb = 5
    nocc = 2
    n_chol = 7

    key = jax.random.PRNGKey(0)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nocc, nocc),
        n_chol,
        make_trial_fn=_make_random_rhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nocc=nocc,
        ),
        make_trial_ops_fn=make_rhf_trial_ops,
        make_meas_ops_fn=make_rhf_meas_ops,
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(v_a, v_m, rtol=1e-7, atol=1e-8), (v_a, v_m)


@pytest.mark.parametrize("walker_kind", ["restricted", "unrestricted"])
def test_auto_energy_matches_manual_rhf(walker_kind):
    norb = 5
    nocc = 2
    n_chol = 7

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nocc, nocc),
        n_chol,
        make_trial_fn=_make_random_rhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nocc=nocc,
        ),
        make_trial_ops_fn=make_rhf_trial_ops,
        make_meas_ops_fn=make_rhf_meas_ops,
    )

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        emr = jnp.real(em)
        ear = jnp.real(ea)

        assert jnp.allclose(ear, emr, rtol=5e-3, atol=5e-4), (ear, emr)


def test_auto_force_bias_matches_manual_rhf_generalized():
    walker_kind = "generalized"
    norb = 5
    nocc = 2
    n_chol = 7

    key = jax.random.PRNGKey(2)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nocc, nocc),
        n_chol,
        make_trial_fn=_make_random_rhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nocc=nocc,
        ),
        make_trial_ops_fn=make_rhf_trial_ops,
        make_meas_ops_fn=make_rhf_meas_ops,
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)
        assert jnp.allclose(v_a, v_m, rtol=1e-7, atol=1e-8), (v_a, v_m)


def run_calc(sys, meas_ops, ham_data, trial_ops, trial_data, params, block_fn, prop_ops):
    mean, err, block_e_all, block_w_all = driver.run_qmc_energy(
        sys=sys,
        params=params,
        ham_data=ham_data,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
    )
    return mean, err, block_e_all, block_w_all


@pytest.mark.parametrize(
    "walker_kind, e_ref, err_ref",
    [
        ("restricted", -75.75594174131398, 0.01213379336719581),
        ("unrestricted", -75.75594174131398, 0.01213379336719581),
    ],
)
def test_calc_rhf_hamiltonian(mf, params, walker_kind, e_ref, err_ref):
    myafqmc = Afqmc(mf)
    myafqmc.params = params
    myafqmc.walker_kind = walker_kind
    myafqmc.mixed_precision = False
    myafqmc.chol_cut = 1e-6
    mean, err = myafqmc.kernel()
    assert jnp.isclose(mean, e_ref), (mean, e_ref, mean - e_ref)
    assert jnp.isclose(err, err_ref), (err, err_ref, err - err_ref)


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
    return QmcParams(
        n_eql_blocks=4,
        n_blocks=20,
        seed=1234,
        n_walkers=5,
        error_method="blocking",
    )


if __name__ == "__main__":
    pytest.main([__file__])
