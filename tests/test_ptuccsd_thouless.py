from functools import partial

from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot.core.ops import MeasOps, k_energy
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.pt2ccsd import Pt2ccsdMeasCfg
from trot.meas.pt2ccsd import build_meas_ctx as build_pt2ccsd_meas_ctx
from trot.meas.pt2ccsd import energy_kernel_rw_rh as pt2ccsd_components
from trot.meas.pt2uccsd import make_pt2uccsd_estimator_ops, make_pt2uccsd_meas_ops
from trot.meas.ptccsd_thouless import (
    build_ptccsd_thouless_meas_ctx,
    energy_pt_rw_rh as ptccsd_thouless_energy,
    force_bias_pt_rw_rh as ptccsd_thouless_force_bias,
)
from trot.meas.ptuccsd_thouless import (
    PtuccsdThoulessMeasCfg,
    build_ptuccsd_thouless_meas_ctx,
    components_ptuccsd_thouless_rw_rh,
    components_ptuccsd_thouless_uw_rh,
    energy_kernel_rw_rh,
    energy_kernel_uw_rh,
    force_bias_kernel_rw_rh,
    force_bias_kernel_uw_rh,
)
from trot.prop.blocks import block_mixed_estimator
from trot.prop.types import PropOps, PropState, QmcParams
from trot.trial.pt2ccsd import Pt2ccsdTrial
from trot.trial.pt2ccsd import overlap_r as pt2ccsd_reference_overlap
from trot.trial.pt2uccsd import (
    Pt2uccsdTrial,
    make_pt2uccsd_trial_data,
    make_pt2uccsd_trial_ops,
)
from trot.trial.ptccsd_thouless import (
    PtccsdThoulessTrial,
    det_overlap_r as ptccsd_thouless_reference_overlap,
    overlap_ptccsd_thouless_r,
)
from trot.trial.ptuccsd_thouless import (
    PtuccsdThoulessTrial,
    overlap_r,
    overlap_u,
    reference_overlap_r,
    reference_overlap_u,
)
from trot.trial.ucisd_k_modes import UcisdKModeTrial, make_ucisd_k_mode_trial_ops


def _double_cfg() -> PtuccsdThoulessMeasCfg:
    return PtuccsdThoulessMeasCfg(
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )


def _same_spin_from_restricted(t2: np.ndarray) -> np.ndarray:
    return t2 - t2.transpose(0, 3, 2, 1)


def _same_spin_tensor(
    rng: np.random.Generator,
    nocc: int,
    nvir: int,
) -> np.ndarray:
    raw = rng.standard_normal((nocc, nvir, nocc, nvir))
    return 0.25 * (
        raw
        - raw.transpose(2, 1, 0, 3)
        - raw.transpose(0, 3, 2, 1)
        + raw.transpose(2, 3, 0, 1)
    )


def _random_ham(rng: np.random.Generator, norb: int, nchol: int) -> HamChol:
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    chol = rng.standard_normal((nchol, norb, norb))
    chol = 0.5 * (chol + chol.transpose(0, 2, 1))
    return HamChol(
        h0=jnp.asarray(0.37),
        h1=jnp.asarray(h1),
        chol=jnp.asarray(chol),
        basis="restricted",
    )


def test_ptuccsd_restricted_limit_matches_pt2ccsd_dense_kernels():
    rng = np.random.default_rng(2401)
    norb, nocc = 5, 2
    nvir = norb - nocc
    mo_t = np.vstack([np.eye(nocc), 0.08 * rng.standard_normal((nvir, nocc))])
    t2 = 0.03 * rng.standard_normal((nocc, nvir, nocc, nvir))
    t2 = 0.5 * (t2 + t2.transpose(2, 3, 0, 1))
    t2ss = _same_spin_from_restricted(t2)
    walker = mo_t + 0.12 * (
        rng.standard_normal((norb, nocc)) + 1.0j * rng.standard_normal((norb, nocc))
    )
    ham = _random_ham(rng, norb, nchol=6)

    restricted_trial = Pt2ccsdTrial(mo_t=jnp.asarray(mo_t), t2=jnp.asarray(t2))
    restricted_thouless_trial = PtccsdThoulessTrial(
        mo_t=jnp.asarray(mo_t), t2=jnp.asarray(t2)
    )
    unrestricted_trial = PtuccsdThoulessTrial(
        mo_t_a=jnp.asarray(mo_t),
        mo_t_b=jnp.asarray(mo_t),
        mo_coeff_b=jnp.eye(norb),
        t2aa=jnp.asarray(t2ss),
        t2ab=jnp.asarray(t2),
        t2bb=jnp.asarray(t2ss),
    )

    restricted_cfg = Pt2ccsdMeasCfg(
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    restricted_ctx = build_pt2ccsd_meas_ctx(ham, restricted_trial, restricted_cfg)
    restricted_thouless_ctx = build_ptccsd_thouless_meas_ctx(
        ham, restricted_thouless_trial
    )
    unrestricted_ctx = build_ptuccsd_thouless_meas_ctx(
        ham, unrestricted_trial, _double_cfg()
    )

    expected_components = pt2ccsd_components(
        jnp.asarray(walker), ham, restricted_ctx, restricted_trial
    )
    actual_components = jax.jit(components_ptuccsd_thouless_rw_rh)(
        jnp.asarray(walker), ham, unrestricted_ctx, unrestricted_trial
    )
    expected_energy = ptccsd_thouless_energy(
        jnp.asarray(walker), ham, restricted_thouless_ctx, restricted_thouless_trial
    )
    actual_energy = energy_kernel_rw_rh(
        jnp.asarray(walker), ham, unrestricted_ctx, unrestricted_trial
    )
    expected_force_bias = ptccsd_thouless_force_bias(
        jnp.asarray(walker), ham, restricted_thouless_ctx, restricted_thouless_trial
    )
    actual_force_bias = force_bias_kernel_rw_rh(
        jnp.asarray(walker), ham, unrestricted_ctx, unrestricted_trial
    )

    np.testing.assert_allclose(
        reference_overlap_r(jnp.asarray(walker), unrestricted_trial),
        pt2ccsd_reference_overlap(jnp.asarray(walker), restricted_trial),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(actual_components, expected_components, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(actual_energy, expected_energy, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(
        actual_force_bias, expected_force_bias, rtol=2.0e-11, atol=2.0e-11
    )
    np.testing.assert_allclose(
        overlap_r(jnp.asarray(walker), unrestricted_trial),
        overlap_ptccsd_thouless_r(jnp.asarray(walker), restricted_thouless_trial),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        ptccsd_thouless_reference_overlap(
            jnp.asarray(walker), restricted_thouless_trial
        ),
        reference_overlap_r(jnp.asarray(walker), unrestricted_trial),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_restricted_wrapper_preserves_spin_resolved_ucc_trial():
    rng = np.random.default_rng(2411)
    norb, noa, nob = 5, 2, 1
    nva, nvb = norb - noa, norb - nob
    beta_rotation, _ = np.linalg.qr(
        np.eye(norb) + 0.15 * rng.standard_normal((norb, norb))
    )
    mo_t_a = np.vstack([np.eye(noa), 0.08 * rng.standard_normal((nva, noa))])
    mo_t_b = np.vstack([np.eye(nob), 0.08 * rng.standard_normal((nvb, nob))])
    trial = PtuccsdThoulessTrial(
        mo_t_a=jnp.asarray(mo_t_a),
        mo_t_b=jnp.asarray(mo_t_b),
        mo_coeff_b=jnp.asarray(beta_rotation),
        t2aa=jnp.asarray(0.02 * _same_spin_tensor(rng, noa, nva)),
        t2ab=jnp.asarray(0.02 * rng.standard_normal((noa, nva, nob, nvb))),
        t2bb=jnp.asarray(0.02 * _same_spin_tensor(rng, nob, nvb)),
    )
    walker = mo_t_a + 0.1 * (
        rng.standard_normal((norb, noa)) + 1.0j * rng.standard_normal((norb, noa))
    )
    ham = _random_ham(rng, norb, nchol=5)
    ctx = build_ptuccsd_thouless_meas_ctx(ham, trial, _double_cfg())
    walker_r = jnp.asarray(walker)
    walker_u = (walker_r[:, :noa], walker_r[:, :nob])

    np.testing.assert_allclose(reference_overlap_r(walker_r, trial), reference_overlap_u(walker_u, trial))
    np.testing.assert_allclose(overlap_r(walker_r, trial), overlap_u(walker_u, trial))
    np.testing.assert_allclose(
        components_ptuccsd_thouless_rw_rh(walker_r, ham, ctx, trial),
        components_ptuccsd_thouless_uw_rh(walker_u, ham, ctx, trial),
    )
    np.testing.assert_allclose(
        force_bias_kernel_rw_rh(walker_r, ham, ctx, trial),
        force_bias_kernel_uw_rh(walker_u, ham, ctx, trial),
    )
    np.testing.assert_allclose(
        energy_kernel_rw_rh(walker_r, ham, ctx, trial),
        energy_kernel_uw_rh(walker_u, ham, ctx, trial),
    )


def test_open_shell_restricted_components_compile_with_mixed_precision():
    rng = np.random.default_rng(2417)
    norb, noa, nob = 5, 2, 1
    nva, nvb = norb - noa, norb - nob
    beta_rotation, _ = np.linalg.qr(
        np.eye(norb) + 0.15 * rng.standard_normal((norb, norb))
    )
    trial = PtuccsdThoulessTrial(
        mo_t_a=jnp.asarray(
            np.vstack([np.eye(noa), 0.08 * rng.standard_normal((nva, noa))])
        ),
        mo_t_b=jnp.asarray(
            np.vstack([np.eye(nob), 0.08 * rng.standard_normal((nvb, nob))])
        ),
        mo_coeff_b=jnp.asarray(beta_rotation),
        t2aa=jnp.asarray(0.02 * _same_spin_tensor(rng, noa, nva)),
        t2ab=jnp.asarray(0.02 * rng.standard_normal((noa, nva, nob, nvb))),
        t2bb=jnp.asarray(0.02 * _same_spin_tensor(rng, nob, nvb)),
    )
    walker = jnp.asarray(
        np.vstack([np.eye(noa), 0.1 * rng.standard_normal((nva, noa))])
        + 0.04j * rng.standard_normal((norb, noa))
    )
    ham = _random_ham(rng, norb, nchol=5)
    sys = System(norb=norb, nelec=(noa, nob), walker_kind="restricted")
    estimator_ops = make_pt2uccsd_estimator_ops(sys, mixed_precision=True)
    ctx = estimator_ops.build_estimator_ctx(ham, trial)

    components = jax.jit(estimator_ops.components)(walker, ham, ctx, trial)

    assert components.shape == (3,)
    assert np.all(np.isfinite(np.asarray(components)))


def test_trial_data_and_factories_support_restricted_walkers():
    rng = np.random.default_rng(2423)
    norb, nocc = 4, 2
    nvir = norb - nocc
    data = {
        "t1a": 0.02 * rng.standard_normal((nocc, nvir)),
        "t1b": 0.02 * rng.standard_normal((nocc, nvir)),
        "t2aa": 0.02 * rng.standard_normal((nocc, nocc, nvir, nvir)),
        "t2ab": 0.02 * rng.standard_normal((nocc, nocc, nvir, nvir)),
        "t2bb": 0.02 * rng.standard_normal((nocc, nocc, nvir, nvir)),
        "mo_coeff_b": np.eye(norb),
    }
    trial = make_pt2uccsd_trial_data(data)
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")
    trial_ops = make_pt2uccsd_trial_ops(sys)
    meas_ops = make_pt2uccsd_meas_ops(sys, mixed_precision=False, testing=True)
    estimator_ops = make_pt2uccsd_estimator_ops(sys, mixed_precision=False, testing=True)

    assert isinstance(trial, Pt2uccsdTrial)
    assert trial.t2aa.shape == (nocc, nvir, nocc, nvir)
    np.testing.assert_allclose(trial.t2ab, np.asarray(data["t2ab"]).transpose(0, 2, 1, 3))
    assert trial_ops.overlap is overlap_r
    assert meas_ops.overlap is overlap_r
    assert estimator_ops.reference_overlap is reference_overlap_r
    assert estimator_ops.component_names == ("theta", "electronic_0", "h_t")

    open_shell = System(norb=norb, nelec=(nocc, nocc - 1), walker_kind="restricted")
    assert make_pt2uccsd_trial_ops(open_shell).overlap is overlap_r
    assert make_pt2uccsd_meas_ops(open_shell).overlap is overlap_r
    assert make_pt2uccsd_estimator_ops(open_shell).reference_overlap is reference_overlap_r

    reversed_spin = System(norb=norb, nelec=(nocc - 1, nocc), walker_kind="restricted")
    with pytest.raises(ValueError, match="nup >= ndn"):
        make_pt2uccsd_trial_ops(reversed_spin)
    with pytest.raises(ValueError, match="nup >= ndn"):
        make_pt2uccsd_meas_ops(reversed_spin)
    with pytest.raises(ValueError, match="nup >= ndn"):
        make_pt2uccsd_estimator_ops(reversed_spin)


def _identity_step(state, **kwargs):
    del kwargs
    return state


def _identity_sr(walkers, weights, zeta, walker_kind):
    del zeta, walker_kind
    return walkers, weights


def _zero_energy(walker, ham_data, meas_ctx, trial_data):
    del walker, ham_data, meas_ctx, trial_data
    return jnp.asarray(0.0)


def test_ptuccsd_estimator_reweights_from_ucisd_mode_guide():
    rng = np.random.default_rng(2437)
    norb, noa, nob = 4, 2, 1
    nva, nvb = norb - noa, norb - nob
    beta_rotation, _ = np.linalg.qr(
        np.eye(norb) + 0.1 * rng.standard_normal((norb, norb))
    )
    pair_dim = noa * nva + nob * nvb
    guide = UcisdKModeTrial(
        mo_coeff_a=jnp.eye(norb),
        mo_coeff_b=jnp.asarray(beta_rotation),
        c1a=jnp.asarray(0.02 * rng.standard_normal((noa, nva))),
        c1b=jnp.asarray(0.02 * rng.standard_normal((nob, nvb))),
        eigenvalues=jnp.asarray([0.03, -0.02]),
        modes=jnp.asarray(0.1 * rng.standard_normal((2, pair_dim))),
    )
    estimator = PtuccsdThoulessTrial(
        mo_t_a=jnp.vstack(
            [jnp.eye(noa), jnp.asarray(0.03 * rng.standard_normal((nva, noa)))]
        ),
        mo_t_b=jnp.vstack(
            [jnp.eye(nob), jnp.asarray(0.03 * rng.standard_normal((nvb, nob)))]
        ),
        mo_coeff_b=jnp.asarray(beta_rotation),
        t2aa=jnp.asarray(0.02 * _same_spin_tensor(rng, noa, nva)),
        t2ab=jnp.asarray(0.02 * rng.standard_normal((noa, nva, nob, nvb))),
        t2bb=jnp.asarray(0.02 * _same_spin_tensor(rng, nob, nvb)),
    )
    sys = System(norb=norb, nelec=(noa, nob), walker_kind="restricted")
    ham = _random_ham(rng, norb, nchol=4)
    guide_ops = make_ucisd_k_mode_trial_ops(sys)
    guide_meas_ops = MeasOps(overlap=guide_ops.overlap, kernels={k_energy: _zero_energy})
    estimator_ops = make_pt2uccsd_estimator_ops(sys, mixed_precision=False, testing=True)
    estimator_ctx = estimator_ops.build_estimator_ctx(ham, estimator)
    walkers = jnp.asarray(
        np.stack(
            [
                np.vstack([np.eye(noa), 0.15 * rng.standard_normal((nva, noa))])
                + 0.04j * rng.standard_normal((norb, noa))
                for _ in range(3)
            ]
        )
    )
    weights = jnp.asarray([0.8, 1.1, 1.4])
    guide_overlaps = jax.vmap(guide_ops.overlap, in_axes=(0, None))(walkers, guide)
    state = PropState(
        walkers=walkers,
        weights=weights,
        overlaps=guide_overlaps,
        rng_key=jax.random.PRNGKey(2441),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )
    params = QmcParams(
        n_walkers=3,
        n_prop_steps=1,
        n_blocks=1,
        n_eql_blocks=1,
        n_chunks=1,
        seed=2447,
    )
    prop_ops = PropOps(
        init_prop_state=lambda **kwargs: state,
        build_prop_ctx=lambda ham_data, rdm1, params: None,
        step=_identity_step,
    )

    state_new, obs = jax.jit(
        partial(
            block_mixed_estimator,
            sys=sys,
            params=params,
            ham_data=ham,
            guide_data=guide,
            guide_ops=guide_ops,
            guide_meas_ops=guide_meas_ops,
            guide_meas_ctx=None,
            guide_prop_ops=prop_ops,
            guide_prop_ctx=None,
            estimator_data=estimator,
            estimator_ops=estimator_ops,
            estimator_ctx=estimator_ctx,
            sr_fn=_identity_sr,
        )
    )(state)

    guide_overlaps_new = jax.vmap(guide_ops.overlap, in_axes=(0, None))(
        state_new.walkers, guide
    )
    reference_overlaps = jax.vmap(estimator_ops.reference_overlap, in_axes=(0, None))(
        state_new.walkers, estimator
    )
    samples = jax.vmap(
        estimator_ops.components,
        in_axes=(0, None, None, None),
    )(state_new.walkers, ham, estimator_ctx, estimator)
    estimator_weights = weights * reference_overlaps / guide_overlaps_new
    expected_components = jnp.sum(estimator_weights[:, None] * samples, axis=0) / jnp.sum(
        estimator_weights
    )

    np.testing.assert_allclose(obs.scalars["estimator_components"], expected_components)
    np.testing.assert_allclose(obs.scalars["estimator_weight"], jnp.sum(estimator_weights))
