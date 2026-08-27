from trot import config

config.configure_once()

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from pyscf import gto, scf

from trot import testing
from trot.afqmc import Afqmc
from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.ham.chol import HamChol, HamCholUhf
from trot.meas.uhf import (
    build_meas_ctx,
    energy_kernel_gw_rh,
    energy_kernel_rw_rh,
    energy_kernel_uw_rh,
    force_bias_kernel_gw_rh,
    force_bias_kernel_rw_rh,
    force_bias_kernel_uw_rh,
    make_uhf_meas_ops,
)
from trot.prop.types import QmcParams
from trot.trial.uhf import UhfTrial, make_uhf_trial_ops


def _make_uhf_trial(key, norb, nup, ndn, dtype=jnp.complex128) -> UhfTrial:
    ka, kb = jax.random.split(key)
    ca = testing.rand_orthonormal_cols(ka, norb, nup, dtype=dtype)
    cb = testing.rand_orthonormal_cols(kb, norb, ndn, dtype=dtype)
    return UhfTrial(mo_coeff_a=ca, mo_coeff_b=cb)


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 6, 2, 2, 8),
        ("unrestricted", 6, 2, 1, 8),
        ("generalized", 6, 2, 1, 8),
    ],
)
def test_auto_force_bias_matches_manual_uhf(walker_kind, norb, nup, ndn, n_chol):
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
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        make_meas_ops_fn=make_uhf_meas_ops,
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(v_a, v_m, rtol=5e-6, atol=5e-7), (v_a, v_m)


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 6, 2, 2, 8),
        ("unrestricted", 6, 2, 1, 8),
        ("generalized", 6, 2, 1, 8),
    ],
)
def test_auto_energy_matches_manual_uhf(walker_kind, norb, nup, ndn, n_chol):
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
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        make_meas_ops_fn=make_uhf_meas_ops,
    )

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(ea, em, rtol=5e-6, atol=5e-7), (ea, em)


def test_force_bias_equal_when_wu_eq_wr():
    norb = 6
    nup, ndn = 2, 2
    n_chol = 8
    walker_kind = "restricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wi = cast(jax.Array, wi)
        fbr = force_bias_kernel_rw_rh(wi, ham, ctx, trial)
        fbu = force_bias_kernel_uw_rh((wi, wi), ham, ctx, trial)

        assert jnp.allclose(fbr, fbu, atol=1e-12), (fbr, fbu)


def test_force_bias_equal_when_wg_eq_wu():
    norb = 6
    nup, ndn = 2, 2
    n_chol = 8
    walker_kind = "unrestricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wi = cast(tuple, wi)
        fbu = force_bias_kernel_uw_rh(wi, ham, ctx, trial)
        wa, wb = wi
        wi = jnp.zeros((2 * norb, nup + ndn), dtype=wa.dtype)
        wi = lax.dynamic_update_slice(wi, wa, (0, 0))
        wi = lax.dynamic_update_slice(wi, wb, (norb, nup))
        fbg = force_bias_kernel_gw_rh(wi, ham, ctx, trial)

        assert jnp.allclose(fbu, fbg, atol=1e-12), (fbu, fbg)


def test_energy_equal_when_wu_eq_wr():
    norb = 6
    nup, ndn = 2, 2
    n_chol = 8
    walker_kind = "restricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wi = cast(jax.Array, wi)
        er = energy_kernel_rw_rh(wi, ham, ctx, trial)
        eu = energy_kernel_uw_rh((wi, wi), ham, ctx, trial)

        assert jnp.allclose(er, eu, atol=1e-12), (er, eu)


def test_energy_equal_when_wg_eq_wu():
    norb = 6
    nup, ndn = 2, 1
    n_chol = 8
    walker_kind = "unrestricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_uhf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_uhf_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wi = cast(tuple, wi)
        eu = energy_kernel_uw_rh(wi, ham, ctx, trial)
        wa, wb = wi
        wi = jnp.zeros((2 * norb, nup + ndn), dtype=wa.dtype)
        wi = lax.dynamic_update_slice(wi, wa, (0, 0))
        wi = lax.dynamic_update_slice(wi, wb, (norb, nup))
        eg = energy_kernel_gw_rh(wi, ham, ctx, trial)

        assert jnp.allclose(eu, eg, atol=1e-12), (eu, eg)


def test_split_uhf_measurement_reduces_to_restricted_hamiltonian():
    norb, nup, ndn, n_chol = 6, 2, 1, 8
    key = jax.random.PRNGKey(101)
    k_ham, k_trial, k_walker = jax.random.split(key, 3)
    ham = testing.make_random_ham_chol(k_ham, norb, n_chol)
    split_ham = HamCholUhf(
        h0=ham.h0,
        h1_a=ham.h1,
        h1_b=ham.h1,
        chol_a=ham.chol,
        chol_b=ham.chol,
    )
    trial = _make_uhf_trial(k_trial, norb, nup, ndn)
    sys = System(norb=norb, nelec=(nup, ndn), walker_kind="unrestricted")
    walker = cast(tuple, testing.make_walkers(k_walker, sys))
    ctx = build_meas_ctx(ham, trial)
    split_ctx = build_meas_ctx(split_ham, trial)

    for regular, split in zip(ctx.tree_flatten()[0], split_ctx.tree_flatten()[0], strict=True):
        np.testing.assert_allclose(split, regular, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        force_bias_kernel_uw_rh(walker, split_ham, split_ctx, trial),
        force_bias_kernel_uw_rh(walker, ham, ctx, trial),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        energy_kernel_uw_rh(walker, split_ham, split_ctx, trial),
        energy_kernel_uw_rh(walker, ham, ctx, trial),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_split_uhf_measurement_uses_native_beta_tensors():
    norb, nup, ndn, n_chol = 5, 2, 1, 7
    key = jax.random.PRNGKey(111)
    k_a, k_b, k_trial, k_walker = jax.random.split(key, 4)
    ham_a = testing.make_random_ham_chol(k_a, norb, n_chol)
    ham_b = testing.make_random_ham_chol(k_b, norb, n_chol)
    split_ham = HamCholUhf(
        h0=ham_a.h0,
        h1_a=ham_a.h1,
        h1_b=ham_b.h1,
        chol_a=ham_a.chol,
        chol_b=ham_b.chol,
    )
    trial = _make_uhf_trial(k_trial, norb, nup, ndn)
    sys = System(norb=norb, nelec=(nup, ndn), walker_kind="unrestricted")
    walker = cast(tuple, testing.make_walkers(k_walker, sys))
    ctx = build_meas_ctx(split_ham, trial)

    ca_h = trial.mo_coeff_a.conj().T
    cb_h = trial.mo_coeff_b.conj().T
    assert jnp.allclose(ctx.rot_h1_a, ca_h @ split_ham.h1_a)
    assert jnp.allclose(ctx.rot_h1_b, cb_h @ split_ham.h1_b)
    assert jnp.allclose(
        ctx.rot_chol_a,
        jnp.einsum("pi,gij->gpj", ca_h, split_ham.chol_a),
    )
    assert jnp.allclose(
        ctx.rot_chol_b,
        jnp.einsum("pi,gij->gpj", cb_h, split_ham.chol_b),
    )

    wu, wd = walker
    mu = ca_h @ wu
    md = cb_h @ wd
    gu = jnp.linalg.solve(mu.T, wu.T)
    gd = jnp.linalg.solve(md.T, wd.T)
    expected_fb = jnp.einsum("gij,ij->g", ctx.rot_chol_a, gu)
    expected_fb += jnp.einsum("gij,ij->g", ctx.rot_chol_b, gd)
    assert jnp.allclose(
        force_bias_kernel_uw_rh(walker, split_ham, ctx, trial),
        expected_fb,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def mf():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
    )
    mf = scf.UHF(mol).newton()
    mf.kernel()
    return mf


def mf2():
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


mf = mf()  # type: ignore
mf2 = mf2()  # type: ignore


@pytest.mark.parametrize(
    "mf, walker_kind, e_ref, err_ref",
    [
        (mf, "restricted", -75.75594187783527, 0.01213383697785241),
        (mf2, "unrestricted", -55.43066756011652, 0.00761980459817991),
        (mf2, "generalized", -55.43066756011653, 0.007619804598170696),
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
