from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot import driver, testing
from trot.core.ops import BlockComponentEstimate, k_energy, k_force_bias
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.auto import make_auto_meas_ops
from trot.meas.ptuccsd_modes import (
    PtuccsdModeMeasCfg,
    PtuccsdModePairSamplingCfg,
    PtuccsdModePairTuningCfg,
    _energy_components_uw_rh_reference,
    _force_bias_kernel_rw_rh_full,
    _force_bias_kernel_uw_rh_full,
    _mode_quadratic_batched_realimag,
    _ptuccsd_mode_chol_index_terms,
    _ptuccsd_mode_chol_pair_terms,
    _ptuccsd_mode_chol_terms,
    _ptuccsd_mode_chol_terms_for_walkers,
    _ptuccsd_mode_energy_common_rw_rh,
    _ptuccsd_mode_energy_common_uw_rh,
    build_ptuccsd_mode_meas_ctx,
    components_ptuccsd_mode_rw_rh,
    components_ptuccsd_mode_uw_rh,
    energy_kernel_rw_rh as mode_energy_rw,
    energy_kernel_uw_rh as mode_energy_uw,
    force_bias_kernel_rw_rh as mode_force_bias_rw,
    force_bias_kernel_uw_rh as mode_force_bias_uw,
    get_ptuccsd_mode_meas_cfg,
    make_ptuccsd_mode_estimator_ops,
    make_ptuccsd_mode_force_bias_ops,
    make_ptuccsd_mode_meas_ops,
    pair_sampled_ptuccsd_block_components,
    select_ptuccsd_mode_pair_sampling,
    stream_ptuccsd_mode_population_statistics,
)
from trot.meas.ptuccsd_thouless import (
    PtuccsdThoulessMeasCfg,
    build_ptuccsd_thouless_meas_ctx,
    components_ptuccsd_thouless_rw_rh,
    components_ptuccsd_thouless_uw_rh,
    energy_kernel_rw_rh as dense_energy_rw,
    energy_kernel_uw_rh as dense_energy_uw,
    force_bias_kernel_rw_rh as dense_force_bias_rw,
    force_bias_kernel_uw_rh as dense_force_bias_uw,
)
from trot.prop.blocks import block
from trot.prop.afqmc import make_prop_ops
from trot.prop.types import QmcParams
from trot.trial.ptuccsd_modes import (
    PtuccsdThoulessModeTrial,
    factorize_t2_modes,
    factorize_ucisd_and_t2_modes_common_rank,
    get_rdm1 as mode_rdm1,
    greens_unrestricted as mode_greens_unrestricted,
    make_ptuccsd_thouless_mode_trial_data,
    make_ptuccsd_thouless_mode_trial_ops,
    mode_apply,
    mode_projections,
    mode_quadratic,
    overlap_r as mode_overlap_r,
    overlap_u as mode_overlap_u,
    reference_overlap_r as mode_reference_overlap_r,
    reference_overlap_u as mode_reference_overlap_u,
    theta_t2_u as mode_theta_t2_u,
)
from trot.trial.ptuccsd_thouless import (
    PtuccsdThoulessTrial,
    get_rdm1 as dense_rdm1,
    overlap_r as dense_overlap_r,
    overlap_u as dense_overlap_u,
    reference_overlap_r as dense_reference_overlap_r,
    reference_overlap_u as dense_reference_overlap_u,
    theta_t2_u as dense_theta_t2_u,
)
from trot.trial.ucisd_k_modes import factorize_ucisd_k_blocks


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


@dataclass(frozen=True)
class TrialCases:
    dense: PtuccsdThoulessTrial
    mode: PtuccsdThoulessModeTrial
    kernel: np.ndarray
    walker_r: jax.Array
    ham: HamChol


@pytest.fixture(scope="module")
def trial_cases() -> TrialCases:
    rng = np.random.default_rng(2501)
    norb, noa, nob = 6, 3, 2
    nva, nvb = norb - noa, norb - nob
    mo_t_a = np.vstack([np.eye(noa), 0.06 * rng.standard_normal((nva, noa))])
    mo_t_b = np.vstack([np.eye(nob), 0.06 * rng.standard_normal((nvb, nob))])
    beta_rotation, _ = np.linalg.qr(
        np.eye(norb) + 0.12 * rng.standard_normal((norb, norb))
    )
    t2aa = 0.03 * _same_spin_tensor(rng, noa, nva)
    t2ab = 0.03 * rng.standard_normal((noa, nva, nob, nvb))
    t2bb = 0.03 * _same_spin_tensor(rng, nob, nvb)
    factorization = factorize_t2_modes(
        t2aa,
        t2ab,
        t2bb,
        mode_threshold=0.0,
        solver="dense",
    )
    dense = PtuccsdThoulessTrial(
        mo_t_a=jnp.asarray(mo_t_a),
        mo_t_b=jnp.asarray(mo_t_b),
        mo_coeff_b=jnp.asarray(beta_rotation),
        t2aa=jnp.asarray(t2aa),
        t2ab=jnp.asarray(t2ab),
        t2bb=jnp.asarray(t2bb),
    )
    mode = PtuccsdThoulessModeTrial(
        mo_t_a=dense.mo_t_a,
        mo_t_b=dense.mo_t_b,
        mo_coeff_b=dense.mo_coeff_b,
        eigenvalues=jnp.asarray(factorization.eigenvalues),
        modes=jnp.asarray(factorization.modes),
    )
    da = noa * nva
    db = nob * nvb
    kernel = np.empty((da + db, da + db))
    kernel[:da, :da] = t2aa.reshape(da, da)
    kernel[:da, da:] = t2ab.reshape(da, db)
    kernel[da:, :da] = t2ab.reshape(da, db).T
    kernel[da:, da:] = t2bb.reshape(db, db)
    walker_r = jnp.asarray(
        np.eye(norb, noa)
        + 0.11 * rng.standard_normal((norb, noa))
        + 0.05j * rng.standard_normal((norb, noa))
    )
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    chol = rng.standard_normal((7, norb, norb))
    chol = 0.5 * (chol + chol.transpose(0, 2, 1))
    ham = HamChol(
        h0=jnp.asarray(0.31),
        h1=jnp.asarray(h1),
        chol=jnp.asarray(chol),
        basis="restricted",
    )
    return TrialCases(
        dense=dense,
        mode=mode,
        kernel=kernel,
        walker_r=walker_r,
        ham=ham,
    )


def _dense_from_modes(trial: PtuccsdThoulessModeTrial) -> PtuccsdThoulessTrial:
    modes = np.asarray(trial.modes)
    kernel = (modes.T * np.asarray(trial.eigenvalues)) @ modes
    da, db = trial.pair_dim
    noa, nob = trial.nocc
    nva, nvb = trial.nvir
    return PtuccsdThoulessTrial(
        mo_t_a=trial.mo_t_a,
        mo_t_b=trial.mo_t_b,
        mo_coeff_b=trial.mo_coeff_b,
        t2aa=jnp.asarray(kernel[:da, :da].reshape(noa, nva, noa, nva)),
        t2ab=jnp.asarray(kernel[:da, da:].reshape(noa, nva, nob, nvb)),
        t2bb=jnp.asarray(kernel[da:, da:].reshape(nob, nvb, nob, nvb)),
    )


def _restricted_walker_population(case: TrialCases) -> jax.Array:
    return jnp.stack(
        (
            case.walker_r,
            case.walker_r + 0.01j * jnp.roll(case.walker_r, 1, axis=0),
            case.walker_r - 0.015 * jnp.roll(case.walker_r, 2, axis=0),
        )
    )


def test_ucc_mode_batch_supports_an_empty_beta_pair_space():
    trial = PtuccsdThoulessModeTrial(
        mo_t_a=jnp.asarray([[1.0], [0.0]], dtype=jnp.float64),
        mo_t_b=jnp.zeros((2, 0), dtype=jnp.float64),
        mo_coeff_b=jnp.eye(2, dtype=jnp.float64),
        eigenvalues=jnp.asarray([0.4], dtype=jnp.float64),
        modes=jnp.asarray([[1.0]], dtype=jnp.float64),
    )
    matrices_a = jnp.asarray(
        [[[0.2 + 0.1j]], [[-0.3 + 0.05j]], [[0.1 - 0.2j]]],
        dtype=jnp.complex128,
    )
    matrices_b = jnp.zeros((3, 0, 2), dtype=jnp.complex128)
    cfg = PtuccsdModeMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )

    quadratic = jax.jit(
        _mode_quadratic_batched_realimag,
        static_argnums=(3, 4),
    )(trial, matrices_a, matrices_b, cfg, 1)

    np.testing.assert_allclose(
        quadratic,
        0.5 * trial.eigenvalues[0] * matrices_a[:, 0, 0] ** 2,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_ucc_one_electron_pair_sampled_estimator_is_finite_and_exact_with_full_head():
    trial = PtuccsdThoulessModeTrial(
        mo_t_a=jnp.asarray([[1.0], [0.04]], dtype=jnp.float64),
        mo_t_b=jnp.zeros((2, 0), dtype=jnp.float64),
        mo_coeff_b=jnp.eye(2, dtype=jnp.float64),
        eigenvalues=jnp.asarray([0.0], dtype=jnp.float64),
        modes=jnp.asarray([[1.0]], dtype=jnp.float64),
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(2502),
        norb=2,
        n_chol=3,
        basis="restricted",
    )
    walkers = jnp.asarray(
        [
            [[1.0 + 0.02j], [0.1 - 0.03j]],
            [[0.98 - 0.01j], [0.08 + 0.02j]],
        ],
        dtype=jnp.complex128,
    )
    weights = jnp.asarray([1.0 + 0.1j, 0.8 - 0.05j], dtype=jnp.complex128)
    sampling = PtuccsdModePairSamplingCfg(
        chol_head_size=ham.chol.shape[0],
        pair_sample_size=4,
        head_chol_batch_size=1,
        track_half_sample_diagnostic=True,
    )
    ctx = build_ptuccsd_mode_meas_ctx(
        ham,
        trial,
        PtuccsdModeMeasCfg(
            memory_mode="high",
            mixed_real_dtype=jnp.float64,
            mixed_complex_dtype=jnp.complex128,
            mixed_real_dtype_testing=jnp.float64,
            mixed_complex_dtype_testing=jnp.complex128,
        ),
        component_sampling=sampling,
    )
    exact = jax.vmap(
        components_ptuccsd_mode_rw_rh,
        in_axes=(0, None, None, None),
    )(walkers, ham, ctx, trial)
    sampled = jax.jit(pair_sampled_ptuccsd_block_components, static_argnums=3)(
        walkers,
        weights,
        jax.random.PRNGKey(2504),
        1,
        ham,
        ctx,
        trial,
    )

    np.testing.assert_allclose(sampled.weight, jnp.sum(weights), rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        sampled.numerator,
        jnp.sum(weights[:, None] * exact, axis=0),
        rtol=4.0e-11,
        atol=4.0e-11,
    )
    assert bool(jnp.all(jnp.isfinite(sampled.numerator)))


@pytest.mark.parametrize("rank", [None, 5, 0])
def test_mode_primitives_match_reconstructed_combined_kernel(
    trial_cases: TrialCases,
    rank: int | None,
):
    case = trial_cases
    if rank is None:
        rank = case.mode.mode_rank
    trial = PtuccsdThoulessModeTrial(
        mo_t_a=case.mode.mo_t_a,
        mo_t_b=case.mode.mo_t_b,
        mo_coeff_b=case.mode.mo_coeff_b,
        eigenvalues=case.mode.eigenvalues[:rank],
        modes=case.mode.modes[:rank],
    )
    rng = np.random.default_rng(2503 + rank)
    matrix_a = jnp.asarray(
        rng.standard_normal((trial.nocc[0], trial.nvir[0]))
        + 1.0j * rng.standard_normal((trial.nocc[0], trial.nvir[0]))
    )
    matrix_b = jnp.asarray(
        rng.standard_normal((trial.nocc[1], trial.nvir[1]))
        + 1.0j * rng.standard_normal((trial.nocc[1], trial.nvir[1]))
    )
    projections = mode_projections(trial, matrix_a, matrix_b)
    applied_a, applied_b = mode_apply(trial, matrix_a, matrix_b, projections)
    quadratic = mode_quadratic(trial, matrix_a, matrix_b, projections)

    modes = np.asarray(trial.modes)
    reconstructed = (modes.T * np.asarray(trial.eigenvalues)) @ modes
    vector = np.concatenate((np.asarray(matrix_a).reshape(-1), np.asarray(matrix_b).reshape(-1)))
    expected_applied = reconstructed @ vector
    expected_quadratic = 0.5 * vector @ reconstructed @ vector
    da, _ = trial.pair_dim
    np.testing.assert_allclose(
        applied_a,
        expected_applied[:da].reshape(matrix_a.shape),
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        applied_b,
        expected_applied[da:].reshape(matrix_b.shape),
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(quadratic, expected_quadratic, rtol=3.0e-12, atol=3.0e-12)


def test_full_rank_mode_trial_matches_dense_exponential_overlap(trial_cases: TrialCases):
    case = trial_cases
    noa, nob = case.mode.nocc
    walker_u = (case.walker_r[:, :noa], case.walker_r[:, :nob])

    np.testing.assert_allclose(
        (np.asarray(case.mode.modes).T * np.asarray(case.mode.eigenvalues))
        @ np.asarray(case.mode.modes),
        case.kernel,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        mode_reference_overlap_u(walker_u, case.mode),
        dense_reference_overlap_u(walker_u, case.dense),
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        mode_theta_t2_u(walker_u, case.mode),
        dense_theta_t2_u(walker_u, case.dense),
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        mode_overlap_u(walker_u, case.mode),
        dense_overlap_u(walker_u, case.dense),
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        mode_reference_overlap_r(case.walker_r, case.mode),
        dense_reference_overlap_r(case.walker_r, case.dense),
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        jax.jit(mode_overlap_r)(case.walker_r, case.mode),
        dense_overlap_r(case.walker_r, case.dense),
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(mode_rdm1(case.mode), dense_rdm1(case.dense))


@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_full_rank_mode_force_bias_matches_dense_open_shell(
    trial_cases: TrialCases,
    memory_mode: str,
):
    case = trial_cases
    cfg = PtuccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    dense_ctx = build_ptuccsd_thouless_meas_ctx(case.ham, case.dense, cfg)
    mode_ctx = build_ptuccsd_mode_meas_ctx(case.ham, case.mode, cfg)
    noa, nob = case.mode.nocc
    walker_u = (case.walker_r[:, :noa], case.walker_r[:, :nob])

    expected_u = dense_force_bias_uw(walker_u, case.ham, dense_ctx, case.dense)
    actual_u = jax.jit(mode_force_bias_uw)(walker_u, case.ham, mode_ctx, case.mode)
    expected_r = dense_force_bias_rw(case.walker_r, case.ham, dense_ctx, case.dense)
    actual_r = jax.jit(mode_force_bias_rw)(case.walker_r, case.ham, mode_ctx, case.mode)

    np.testing.assert_allclose(actual_u, expected_u, rtol=3.0e-11, atol=3.0e-11)
    np.testing.assert_allclose(actual_r, expected_r, rtol=3.0e-11, atol=3.0e-11)
    np.testing.assert_allclose(actual_r, actual_u, rtol=3.0e-12, atol=3.0e-12)


def test_half_green_overlap_and_force_bias_match_full_green_in_complex_gauge(
    trial_cases: TrialCases,
):
    case = trial_cases
    gauge_a = jnp.asarray(
        [
            [1.05 + 0.10j, -0.04 + 0.02j, 0.03 - 0.01j],
            [0.02 - 0.03j, 0.93 - 0.08j, -0.05 + 0.04j],
            [-0.01 + 0.02j, 0.04 + 0.01j, 1.08 + 0.06j],
        ]
    )
    gauge_b = jnp.asarray(
        [[1.07 + 0.09j, -0.03 + 0.02j], [0.04 - 0.01j, 0.91 - 0.07j]]
    )
    trial = PtuccsdThoulessModeTrial(
        mo_t_a=case.mode.mo_t_a @ gauge_a,
        mo_t_b=case.mode.mo_t_b @ gauge_b,
        mo_coeff_b=case.mode.mo_coeff_b,
        eigenvalues=case.mode.eigenvalues[:5],
        modes=case.mode.modes[:5],
    )
    cfg = PtuccsdModeMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    ctx = build_ptuccsd_mode_meas_ctx(case.ham, trial, cfg)
    noa, nob = trial.nocc
    walker_u = (case.walker_r[:, :noa], case.walker_r[:, :nob])

    green_a, green_b = mode_greens_unrestricted(walker_u, trial)
    full_overlap_u = mode_reference_overlap_u(walker_u, trial) * jnp.exp(
        mode_quadratic(
            trial,
            green_a[:noa, noa:],
            green_b[:nob, nob:],
        )
    )
    np.testing.assert_allclose(
        mode_overlap_u(walker_u, trial),
        full_overlap_u,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        mode_overlap_r(case.walker_r, trial),
        full_overlap_u,
        rtol=3.0e-12,
        atol=3.0e-12,
    )

    full_u = _force_bias_kernel_uw_rh_full(walker_u, case.ham, ctx, trial)
    half_u = mode_force_bias_uw(walker_u, case.ham, ctx, trial)
    full_r = _force_bias_kernel_rw_rh_full(case.walker_r, case.ham, ctx, trial)
    half_r = mode_force_bias_rw(case.walker_r, case.ham, ctx, trial)
    np.testing.assert_allclose(half_u, full_u, rtol=3.0e-11, atol=3.0e-11)
    np.testing.assert_allclose(half_r, full_r, rtol=3.0e-11, atol=3.0e-11)


@pytest.mark.parametrize("memory_mode", ["high", "low"])
@pytest.mark.parametrize("n_mode_chunks", [1, 3, 7])
def test_full_rank_mode_components_and_energy_match_dense_open_shell(
    trial_cases: TrialCases,
    memory_mode: str,
    n_mode_chunks: int,
):
    case = trial_cases
    cfg = PtuccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    dense_ctx = build_ptuccsd_thouless_meas_ctx(case.ham, case.dense, cfg)
    mode_ctx = build_ptuccsd_mode_meas_ctx(
        case.ham,
        case.mode,
        cfg,
        n_mode_chunks=n_mode_chunks,
    )
    noa, nob = case.mode.nocc
    walker_u = (case.walker_r[:, :noa], case.walker_r[:, :nob])

    expected_components_u = components_ptuccsd_thouless_uw_rh(
        walker_u,
        case.ham,
        dense_ctx,
        case.dense,
    )
    actual_components_u = jax.jit(components_ptuccsd_mode_uw_rh)(
        walker_u,
        case.ham,
        mode_ctx,
        case.mode,
    )
    expected_components_r = components_ptuccsd_thouless_rw_rh(
        case.walker_r,
        case.ham,
        dense_ctx,
        case.dense,
    )
    actual_components_r = jax.jit(components_ptuccsd_mode_rw_rh)(
        case.walker_r,
        case.ham,
        mode_ctx,
        case.mode,
    )
    expected_energy_u = dense_energy_uw(walker_u, case.ham, dense_ctx, case.dense)
    actual_energy_u = mode_energy_uw(walker_u, case.ham, mode_ctx, case.mode)
    expected_energy_r = dense_energy_rw(case.walker_r, case.ham, dense_ctx, case.dense)
    actual_energy_r = mode_energy_rw(case.walker_r, case.ham, mode_ctx, case.mode)

    np.testing.assert_allclose(
        actual_components_u,
        expected_components_u,
        rtol=4.0e-11,
        atol=4.0e-11,
    )
    np.testing.assert_allclose(
        actual_components_r,
        expected_components_r,
        rtol=4.0e-11,
        atol=4.0e-11,
    )
    np.testing.assert_allclose(actual_components_r, actual_components_u, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(actual_energy_u, expected_energy_u, rtol=4.0e-11, atol=4.0e-11)
    np.testing.assert_allclose(actual_energy_r, expected_energy_r, rtol=4.0e-11, atol=4.0e-11)


@pytest.mark.parametrize("memory_mode", ["high", "low"])
@pytest.mark.parametrize("n_mode_chunks", [1, 3])
def test_ucc_cholesky_residual_sum_matches_previous_deterministic_kernel(
    trial_cases: TrialCases,
    memory_mode: str,
    n_mode_chunks: int,
):
    case = trial_cases
    cfg = PtuccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    ctx = build_ptuccsd_mode_meas_ctx(
        case.ham,
        case.mode,
        cfg,
        n_mode_chunks=n_mode_chunks,
    )
    noa, nob = case.mode.nocc
    walker_u = (case.walker_r[:, :noa], case.walker_r[:, :nob])

    common = jax.jit(_ptuccsd_mode_energy_common_uw_rh)(
        walker_u,
        case.ham,
        ctx,
        case.mode,
    )
    chol_components = jax.jit(_ptuccsd_mode_chol_terms)(
        common,
        case.ham.chol,
        ctx.rot_chol_a,
        ctx.chol_b,
        ctx.rot_chol_b,
        ctx,
        case.mode,
    )
    chol_sum = jnp.sum(chol_components, axis=0)
    rebuilt = jnp.stack(
        (
            common.theta,
            common.electronic_0_base + chol_sum[0],
            common.h_t_base + chol_sum[1],
        )
    )
    previous = jnp.stack(
        _energy_components_uw_rh_reference(
            walker_u,
            case.ham,
            ctx,
            case.mode,
        )
    )
    production = components_ptuccsd_mode_rw_rh(
        case.walker_r,
        case.ham,
        ctx,
        case.mode,
    )

    assert chol_components.shape == (case.ham.chol.shape[0], 2)
    np.testing.assert_allclose(rebuilt, previous, rtol=4.0e-11, atol=4.0e-11)
    np.testing.assert_allclose(production, previous, rtol=4.0e-11, atol=4.0e-11)


@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_ucc_restricted_walker_batched_and_indexed_residuals_match_full_table(
    trial_cases: TrialCases,
    memory_mode: str,
):
    case = trial_cases
    cfg = PtuccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    ctx = build_ptuccsd_mode_meas_ctx(
        case.ham,
        case.mode,
        cfg,
        n_mode_chunks=3,
    )
    walkers = _restricted_walker_population(case)
    common = jax.vmap(
        _ptuccsd_mode_energy_common_rw_rh,
        in_axes=(0, None, None, None),
    )(walkers, case.ham, ctx, case.mode)
    chol_components = jax.jit(
        lambda common_i: _ptuccsd_mode_chol_terms_for_walkers(
            common_i,
            case.ham.chol,
            ctx.rot_chol_a,
            ctx.chol_b,
            ctx.rot_chol_b,
            ctx,
            case.mode,
            n_chunks=2,
        )
    )(common)
    expected = jax.vmap(
        lambda common_i: _ptuccsd_mode_chol_terms(
            common_i,
            case.ham.chol,
            ctx.rot_chol_a,
            ctx.chol_b,
            ctx.rot_chol_b,
            ctx,
            case.mode,
        )
    )(common)
    chol_sum = jnp.sum(chol_components, axis=1)
    rebuilt = jnp.stack(
        (
            common.theta,
            common.electronic_0_base + chol_sum[:, 0],
            common.h_t_base + chol_sum[:, 1],
        ),
        axis=1,
    )
    production = jax.vmap(
        components_ptuccsd_mode_rw_rh,
        in_axes=(0, None, None, None),
    )(walkers, case.ham, ctx, case.mode)
    indices = jnp.asarray([6, 0, 3, 2], dtype=jnp.int32)
    indexed = _ptuccsd_mode_chol_index_terms(
        jax.tree_util.tree_map(lambda value: value[0], common),
        indices,
        case.ham,
        ctx,
        case.mode,
        n_chunks=2,
    )

    np.testing.assert_allclose(chol_components, expected, rtol=3.0e-12, atol=3.0e-12)
    np.testing.assert_allclose(rebuilt, production, rtol=4.0e-11, atol=4.0e-11)
    np.testing.assert_allclose(
        indexed,
        chol_components[0, indices],
        rtol=3.0e-12,
        atol=3.0e-12,
    )


def test_ucc_component_sampling_configuration_and_factory(trial_cases: TrialCases):
    case = trial_cases
    sys = System(case.mode.norb, case.mode.nocc, walker_kind="restricted")
    sampling = PtuccsdModePairSamplingCfg(
        chol_head_size=2,
        pair_sample_size=16,
        head_chol_batch_size=1,
        track_half_sample_diagnostic=True,
    )
    defaults = PtuccsdModePairSamplingCfg(chol_head_size=2, pair_sample_size=16)
    deterministic_ops = make_ptuccsd_mode_estimator_ops(sys, mixed_precision=False)
    sampled_ops = make_ptuccsd_mode_estimator_ops(
        sys,
        mixed_precision=False,
        component_sampling=sampling,
    )
    tuning = PtuccsdModePairTuningCfg(
        target_tail_std_ha=1.0e6,
        candidate_sample_sizes=(8,),
        tuning_population_count=1,
        settling_blocks=0,
    )
    tuned_ops = make_ptuccsd_mode_estimator_ops(
        sys,
        mixed_precision=False,
        component_sampling=sampling,
        component_tuning=tuning,
    )
    ctx = sampled_ops.build_estimator_ctx(case.ham, case.mode)

    assert defaults.rank_head_by_guide is False
    assert defaults.head_chol_batch_size == 0
    assert defaults.tail_probability_uniform_mix == 0.0
    assert deterministic_ops.block_components is None
    assert deterministic_ops.use_for_population_control is False
    assert sampled_ops.block_components is pair_sampled_ptuccsd_block_components
    assert sampled_ops.use_for_population_control is True
    assert sampled_ops.retune_block_components is None
    assert tuned_ops.retune_block_components is not None
    assert ctx.component_sampling == sampling
    assert ctx.reference_chol_scores.shape == (case.ham.chol.shape[0],)
    assert ctx.chol_head_indices.shape == (2,)
    assert ctx.chol_tail_indices.shape == (case.ham.chol.shape[0] - 2,)
    np.testing.assert_allclose(jnp.sum(ctx.chol_tail_prob), 1.0, atol=1.0e-14)
    assert bool(jnp.all(ctx.chol_tail_prob > 0.0))

    with pytest.raises(ValueError, match="must not exceed"):
        build_ptuccsd_mode_meas_ctx(
            case.ham,
            case.mode,
            component_sampling=PtuccsdModePairSamplingCfg(
                chol_head_size=case.ham.chol.shape[0] + 1,
                pair_sample_size=8,
            ),
        )
    with pytest.raises(ValueError, match="at least two"):
        PtuccsdModePairSamplingCfg(
            chol_head_size=0,
            pair_sample_size=1,
            track_half_sample_diagnostic=True,
        )
    with pytest.raises(ValueError, match="requires an equilibration"):
        make_ptuccsd_mode_estimator_ops(
            sys,
            mixed_precision=False,
            component_tuning=tuning,
        )


@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_ucc_full_head_block_components_match_exact_complex_numerator(
    trial_cases: TrialCases,
    memory_mode: str,
):
    case = trial_cases
    sys = System(case.mode.norb, case.mode.nocc, walker_kind="restricted")
    walkers = _restricted_walker_population(case)
    sampling = PtuccsdModePairSamplingCfg(
        chol_head_size=case.ham.chol.shape[0],
        pair_sample_size=8,
        head_chol_batch_size=2,
        track_half_sample_diagnostic=True,
    )
    ops = make_ptuccsd_mode_estimator_ops(
        sys,
        n_mode_chunks=3,
        memory_mode=memory_mode,
        mixed_precision=False,
        testing=True,
        component_sampling=sampling,
    )
    ctx = ops.build_estimator_ctx(case.ham, case.mode)
    candidate_weights = jnp.asarray(
        [1.0 + 0.2j, 0.7 - 0.1j, 1.3 + 0.4j],
        dtype=jnp.complex128,
    )
    exact_components = jax.vmap(
        components_ptuccsd_mode_rw_rh,
        in_axes=(0, None, None, None),
    )(walkers, case.ham, ctx, case.mode)
    expected_weight = jnp.sum(candidate_weights)
    expected_numerator = jnp.sum(
        candidate_weights[:, None] * exact_components,
        axis=0,
    )

    evaluate = jax.jit(pair_sampled_ptuccsd_block_components, static_argnums=3)
    result = evaluate(
        walkers,
        candidate_weights,
        jax.random.PRNGKey(2603),
        2,
        case.ham,
        ctx,
        case.mode,
    )
    result_other_key = evaluate(
        walkers,
        candidate_weights,
        jax.random.PRNGKey(2609),
        2,
        case.ham,
        ctx,
        case.mode,
    )

    assert isinstance(result, BlockComponentEstimate)
    assert ctx.chol_tail_indices.shape == (0,)
    np.testing.assert_allclose(result.weight, expected_weight, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        result.numerator,
        expected_numerator,
        rtol=4.0e-11,
        atol=4.0e-11,
    )
    np.testing.assert_allclose(result_other_key.numerator, result.numerator)
    np.testing.assert_allclose(
        result.diagnostics["pt_component_sampling_noise_real"], 0.0
    )
    np.testing.assert_allclose(
        result.diagnostics["pt_component_sampling_noise_imag"], 0.0
    )


def test_ucc_sampled_tail_uses_real_projected_walker_proposal(
    trial_cases: TrialCases,
):
    case = trial_cases
    sys = System(case.mode.norb, case.mode.nocc, walker_kind="restricted")
    walkers = _restricted_walker_population(case)
    sampling = PtuccsdModePairSamplingCfg(
        chol_head_size=2,
        pair_sample_size=16384,
        head_chol_batch_size=1,
        tail_probability_uniform_mix=0.05,
        track_half_sample_diagnostic=True,
        walker_guide_policy="head_rms",
        walker_guide_weight_mix=0.2,
    )
    ops = make_ptuccsd_mode_estimator_ops(
        sys,
        n_mode_chunks=3,
        mixed_precision=False,
        testing=True,
        component_sampling=sampling,
    )
    ctx = ops.build_estimator_ctx(case.ham, case.mode)
    candidate_weights = jnp.asarray(
        [1.0 + 0.5j, -0.35 + 0.8j, 0.9 - 0.4j],
        dtype=jnp.complex128,
    )
    common = jax.vmap(
        _ptuccsd_mode_energy_common_rw_rh,
        in_axes=(0, None, None, None),
    )(walkers, case.ham, ctx, case.mode)
    all_terms = _ptuccsd_mode_chol_terms_for_walkers(
        common,
        case.ham.chol,
        ctx.rot_chol_a,
        ctx.chol_b,
        ctx.rot_chol_b,
        ctx,
        case.mode,
        n_chunks=2,
    )
    sample_walker = jnp.asarray([2, 0, 1, 2], dtype=jnp.int32)
    sample_chol = jnp.asarray([6, 0, 3, 1], dtype=jnp.int32)
    gathered = _ptuccsd_mode_chol_pair_terms(
        common,
        sample_walker,
        sample_chol,
        case.ham,
        ctx,
        case.mode,
        n_chunks=2,
    )
    exact_components = jax.vmap(
        components_ptuccsd_mode_rw_rh,
        in_axes=(0, None, None, None),
    )(walkers, case.ham, ctx, case.mode)
    exact_numerator = jnp.sum(candidate_weights[:, None] * exact_components, axis=0)

    normalized_weights = candidate_weights / jnp.sum(candidate_weights)
    theta_reference = jnp.sum(normalized_weights * common.theta)
    effective_head = (
        all_terms[:, ctx.chol_head_indices, 1]
        + (1.0 - theta_reference) * all_terms[:, ctx.chol_head_indices, 0]
    )
    projected_head = jnp.real(
        normalized_weights[:, None] * effective_head
    )
    head_scores = jnp.sqrt(jnp.sum(projected_head**2, axis=1))
    abs_prob = jnp.abs(candidate_weights) / jnp.sum(jnp.abs(candidate_weights))
    guided_prob = head_scores / jnp.sum(head_scores)
    walker_prob = 0.2 * abs_prob + 0.8 * guided_prob
    expected_ess = 1.0 / jnp.sum(walker_prob**2)

    tail_terms = all_terms[:, ctx.chol_tail_indices]
    importance_values = (
        candidate_weights[:, None, None]
        * tail_terms
        / (walker_prob[:, None, None] * ctx.chol_tail_prob[None, :, None])
    )
    joint_prob = walker_prob[:, None] * ctx.chol_tail_prob[None, :]
    tail_mean = jnp.sum(joint_prob[:, :, None] * importance_values, axis=(0, 1))
    exact_tail_numerator = jnp.sum(
        candidate_weights[:, None, None] * tail_terms,
        axis=(0, 1),
    )
    real_variance = jnp.sum(
        joint_prob[:, :, None]
        * (jnp.real(importance_values) - jnp.real(tail_mean)) ** 2,
        axis=(0, 1),
    )
    imag_variance = jnp.sum(
        joint_prob[:, :, None]
        * (jnp.imag(importance_values) - jnp.imag(tail_mean)) ** 2,
        axis=(0, 1),
    )
    real_tolerance = 8.0 * jnp.sqrt(real_variance / sampling.pair_sample_size) + 1.0e-10
    imag_tolerance = 8.0 * jnp.sqrt(imag_variance / sampling.pair_sample_size) + 1.0e-10

    rng_key = jax.random.PRNGKey(2617)
    result = jax.jit(pair_sampled_ptuccsd_block_components, static_argnums=3)(
        walkers,
        candidate_weights,
        rng_key,
        2,
        case.ham,
        ctx,
        case.mode,
    )

    np.testing.assert_allclose(gathered, all_terms[sample_walker, sample_chol])
    np.testing.assert_allclose(result.weight, jnp.sum(candidate_weights), atol=2.0e-12)
    np.testing.assert_allclose(result.numerator[0], exact_numerator[0], atol=4.0e-11)
    np.testing.assert_allclose(tail_mean, exact_tail_numerator, atol=3.0e-12)
    assert bool(
        jnp.all(jnp.abs(jnp.real(result.numerator[1:] - exact_numerator[1:])) < real_tolerance)
    )
    assert bool(
        jnp.all(jnp.abs(jnp.imag(result.numerator[1:] - exact_numerator[1:])) < imag_tolerance)
    )
    np.testing.assert_allclose(
        result.diagnostics["pt_walker_proposal_ess"],
        expected_ess,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    key_walker, key_chol = jax.random.split(rng_key)
    sample_walker = jax.random.choice(
        key_walker,
        walkers.shape[0],
        shape=(sampling.pair_sample_size,),
        replace=True,
        p=walker_prob,
    )
    sample_chol_rel = jax.random.choice(
        key_chol,
        ctx.chol_tail_indices.shape[0],
        shape=(sampling.pair_sample_size,),
        replace=True,
        p=ctx.chol_tail_prob,
    )
    sampled_importance = importance_values[sample_walker, sample_chol_rel]
    first_size = sampling.pair_sample_size // 2
    second_size = sampling.pair_sample_size - first_size
    scale = jnp.sqrt(first_size * second_size) / sampling.pair_sample_size
    half_difference = scale * (
        jnp.mean(sampled_importance[:first_size], axis=0)
        - jnp.mean(sampled_importance[first_size:], axis=0)
    )
    normalized_difference = half_difference / result.weight
    theta_mean = result.numerator[0] / result.weight
    expected_noise = normalized_difference[1] + (
        1.0 - theta_mean
    ) * normalized_difference[0]
    np.testing.assert_allclose(
        result.diagnostics["pt_component_sampling_noise_real"],
        jnp.real(expected_noise),
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        result.diagnostics["pt_component_sampling_noise_imag"],
        jnp.imag(expected_noise),
        rtol=3.0e-12,
        atol=3.0e-12,
    )


@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_ucc_streamed_tuning_statistics_are_real_projected(
    trial_cases: TrialCases,
    memory_mode: str,
):
    case = trial_cases
    walkers = _restricted_walker_population(case)
    sampling = PtuccsdModePairSamplingCfg(chol_head_size=1, pair_sample_size=8)
    cfg = PtuccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    ctx = build_ptuccsd_mode_meas_ctx(
        case.ham,
        case.mode,
        cfg,
        n_mode_chunks=2,
        component_sampling=sampling,
    )
    candidate_weights = jnp.asarray(
        [1.0 + 0.4j, -0.2 + 0.6j, 1.1 - 0.5j],
        dtype=jnp.complex128,
    )
    stats = stream_ptuccsd_mode_population_statistics(
        walkers,
        candidate_weights,
        case.ham,
        ctx,
        case.mode,
        n_walker_chunks=2,
        chol_batch_size=2,
    )

    common = jax.vmap(
        _ptuccsd_mode_energy_common_rw_rh,
        in_axes=(0, None, None, None),
    )(walkers, case.ham, ctx, case.mode)
    all_terms = _ptuccsd_mode_chol_terms_for_walkers(
        common,
        case.ham.chol,
        ctx.rot_chol_a,
        ctx.chol_b,
        ctx.rot_chol_b,
        ctx,
        case.mode,
        n_chunks=2,
    )
    normalized_weights = candidate_weights / jnp.sum(candidate_weights)
    walker_prob = jnp.abs(candidate_weights) / jnp.sum(jnp.abs(candidate_weights))
    theta_reference = jnp.sum(normalized_weights * common.theta)
    effective = all_terms[..., 1] + (1.0 - theta_reference) * all_terms[..., 0]
    projected = jnp.real(normalized_weights[:, None] * effective)
    expected_means = jnp.sum(projected, axis=0)
    expected_seconds = jnp.sum(projected**2 / walker_prob[:, None], axis=0)
    exact_components = jax.vmap(
        components_ptuccsd_mode_rw_rh,
        in_axes=(0, None, None, None),
    )(walkers, case.ham, ctx, case.mode)
    mean_components = jnp.sum(
        candidate_weights[:, None] * exact_components,
        axis=0,
    ) / jnp.sum(candidate_weights)
    expected_energy = jnp.real(
        case.ham.h0
        + mean_components[1]
        + mean_components[2]
        - mean_components[0] * mean_components[1]
    )

    np.testing.assert_allclose(stats.term_means, expected_means, atol=4.0e-11)
    np.testing.assert_allclose(
        stats.term_second_moments,
        expected_seconds,
        atol=4.0e-11,
    )
    np.testing.assert_allclose(stats.exact_block_energy_ha, expected_energy, atol=4.0e-11)

    tuning = PtuccsdModePairTuningCfg(
        target_tail_std_ha=1.0e6,
        candidate_sample_sizes=(8,),
        tuning_population_count=1,
        minimum_head_fraction=0.0,
        maximum_head_fraction=1.0,
        walker_guide_policy="head_rms",
        settling_blocks=0,
    )
    selected = select_ptuccsd_mode_pair_sampling(
        stats,
        tuning,
        n_walkers=walkers.shape[0],
        reference_guide_scores=np.asarray(ctx.reference_chol_scores),
        calibration_std_ha=1.0,
        calibration_source="test",
        final_error_target_ha=None,
        n_blocks=10,
    )
    assert selected.sampling.chol_head_size == 0
    assert selected.sampling.pair_sample_size == 8
    assert selected.sampling.walker_guide_policy == "head_rms"


def test_ucc_component_tuning_installs_selected_context(trial_cases: TrialCases):
    case = trial_cases
    sys = System(case.mode.norb, case.mode.nocc, walker_kind="restricted")
    equilibration_sampling = PtuccsdModePairSamplingCfg(
        chol_head_size=1,
        pair_sample_size=8,
        track_half_sample_diagnostic=False,
    )
    tuning = PtuccsdModePairTuningCfg(
        target_tail_std_ha=1.0e6,
        candidate_sample_sizes=(8,),
        minimum_head_fraction=0.0,
        maximum_head_fraction=0.0,
        tuning_n_chunks=2,
        tuning_chol_batch_size=2,
        tuning_population_count=1,
        track_half_sample_diagnostic=False,
        settling_blocks=0,
    )
    estimator_ops = make_ptuccsd_mode_estimator_ops(
        sys,
        mixed_precision=False,
        component_sampling=equilibration_sampling,
        component_tuning=tuning,
    )
    estimator_ctx = estimator_ops.build_estimator_ctx(case.ham, case.mode)
    guide_ops = make_ptuccsd_mode_meas_ops(sys, mixed_precision=False)
    state = SimpleNamespace(
        walkers=_restricted_walker_population(case),
        weights=jnp.asarray([1.0, 0.7, 1.2]),
    )

    def unused_advance(state_i, *, n_blocks):
        del state_i, n_blocks
        raise AssertionError("one-population tuning must not advance propagation")

    assert estimator_ops.retune_block_components is not None
    result = estimator_ops.retune_block_components(
        state,
        jnp.zeros((0, 3), dtype=jnp.complex128),
        jnp.zeros((0,), dtype=jnp.complex128),
        SimpleNamespace(n_blocks=20),
        case.ham,
        estimator_ctx,
        case.mode,
        guide_data=case.mode,
        guide_meas_ops=guide_ops,
        guide_meas_ctx=None,
        advance_blocks=unused_advance,
        target_error=None,
    )
    assert result.state is state
    assert result.estimator_ctx.component_sampling is not None
    assert result.estimator_ctx.component_sampling.chol_head_size == 0
    assert result.estimator_ctx.component_sampling.pair_sample_size == 8


@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_truncated_mode_components_match_reconstructed_dense_kernel(
    trial_cases: TrialCases,
    memory_mode: str,
):
    case = trial_cases
    trial = PtuccsdThoulessModeTrial(
        mo_t_a=case.mode.mo_t_a,
        mo_t_b=case.mode.mo_t_b,
        mo_coeff_b=case.mode.mo_coeff_b,
        eigenvalues=case.mode.eigenvalues[:5],
        modes=case.mode.modes[:5],
    )
    dense = _dense_from_modes(trial)
    cfg = PtuccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    dense_ctx = build_ptuccsd_thouless_meas_ctx(case.ham, dense, cfg)
    mode_ctx = build_ptuccsd_mode_meas_ctx(
        case.ham,
        trial,
        cfg,
        n_mode_chunks=3,
    )
    expected = components_ptuccsd_thouless_rw_rh(
        case.walker_r,
        case.ham,
        dense_ctx,
        dense,
    )
    actual = components_ptuccsd_mode_rw_rh(case.walker_r, case.ham, mode_ctx, trial)
    np.testing.assert_allclose(actual, expected, rtol=4.0e-11, atol=4.0e-11)


def test_mode_force_bias_mixed_precision_matches_dense_accuracy_policy(
    trial_cases: TrialCases,
):
    case = trial_cases
    sys = System(case.mode.norb, case.mode.nocc, walker_kind="restricted")
    mixed_trial = PtuccsdThoulessModeTrial(
        mo_t_a=case.mode.mo_t_a,
        mo_t_b=case.mode.mo_t_b,
        mo_coeff_b=case.mode.mo_coeff_b,
        eigenvalues=case.mode.eigenvalues,
        modes=case.mode.modes.astype(jnp.float32),
    )
    double_cfg = PtuccsdThoulessMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    double_ctx = build_ptuccsd_thouless_meas_ctx(case.ham, case.dense, double_cfg)
    expected = dense_force_bias_rw(case.walker_r, case.ham, double_ctx, case.dense)

    ops = make_ptuccsd_mode_force_bias_ops(sys, mixed_precision=True)
    mixed_ctx = ops.build_meas_ctx(case.ham, mixed_trial)
    actual = jax.jit(ops.require_kernel(k_force_bias))(
        case.walker_r,
        case.ham,
        mixed_ctx,
        mixed_trial,
    )
    relative_error = float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))
    cfg = get_ptuccsd_mode_meas_cfg(ops)

    assert cfg is not None
    assert cfg.mixed_real_dtype == jnp.float32
    assert cfg.mixed_complex_dtype == jnp.complex64
    assert ops.has_kernel(k_force_bias)
    assert not ops.has_kernel(k_energy)
    assert relative_error < 2.0e-5


@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_mode_energy_mixed_precision_matches_dense_accuracy_policy(
    trial_cases: TrialCases,
    memory_mode: str,
):
    case = trial_cases
    sys = System(case.mode.norb, case.mode.nocc, walker_kind="restricted")
    mixed_trial = PtuccsdThoulessModeTrial(
        mo_t_a=case.mode.mo_t_a,
        mo_t_b=case.mode.mo_t_b,
        mo_coeff_b=case.mode.mo_coeff_b,
        eigenvalues=case.mode.eigenvalues,
        modes=case.mode.modes.astype(jnp.float32),
    )
    double_cfg = PtuccsdThoulessMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    dense_ctx = build_ptuccsd_thouless_meas_ctx(case.ham, case.dense, double_cfg)
    expected_components = components_ptuccsd_thouless_rw_rh(
        case.walker_r,
        case.ham,
        dense_ctx,
        case.dense,
    )
    expected_energy = dense_energy_rw(case.walker_r, case.ham, dense_ctx, case.dense)

    ops = make_ptuccsd_mode_meas_ops(
        sys,
        n_mode_chunks=3,
        memory_mode=memory_mode,
        mixed_precision=True,
    )
    mode_ctx = ops.build_meas_ctx(case.ham, mixed_trial)
    actual_components = jax.jit(components_ptuccsd_mode_rw_rh)(
        case.walker_r,
        case.ham,
        mode_ctx,
        mixed_trial,
    )
    actual_energy = jax.jit(ops.require_kernel(k_energy))(
        case.walker_r,
        case.ham,
        mode_ctx,
        mixed_trial,
    )
    component_error = float(jnp.max(jnp.abs(actual_components - expected_components)))
    energy_error = float(jnp.abs(actual_energy - expected_energy))

    assert ops.has_kernel(k_force_bias)
    assert ops.has_kernel(k_energy)
    assert mode_ctx.n_mode_chunks == 3
    assert component_error < 2.0e-4
    assert energy_error < 2.0e-4


def test_mode_estimator_factory_exposes_pt_components(trial_cases: TrialCases):
    case = trial_cases
    sys = System(case.mode.norb, case.mode.nocc, walker_kind="restricted")
    estimator_ops = make_ptuccsd_mode_estimator_ops(
        sys,
        n_mode_chunks=3,
        mixed_precision=False,
        testing=True,
    )
    ctx = estimator_ops.build_estimator_ctx(case.ham, case.mode)
    components = estimator_ops.components(case.walker_r, case.ham, ctx, case.mode)
    energy = estimator_ops.combine_energy(case.ham.h0, components)

    assert estimator_ops.reference_overlap is mode_reference_overlap_r
    assert estimator_ops.component_names == ("theta", "electronic_0", "h_t")
    np.testing.assert_allclose(
        energy,
        mode_energy_rw(case.walker_r, case.ham, ctx, case.mode),
    )


def test_mode_trial_runs_a_restricted_open_shell_afqmc_smoke_calculation():
    """Tiny smoke test only; this is not a meaningful energy comparison."""

    rng = np.random.default_rng(2549)
    norb, noa, nob = 4, 2, 1
    nva, nvb = norb - noa, norb - nob
    factorization = factorize_t2_modes(
        0.01 * _same_spin_tensor(rng, noa, nva),
        0.01 * rng.standard_normal((noa, nva, nob, nvb)),
        0.01 * _same_spin_tensor(rng, nob, nvb),
        mode_threshold=0.0,
        solver="dense",
    )
    trial = PtuccsdThoulessModeTrial(
        mo_t_a=jnp.asarray(
            np.vstack([np.eye(noa), 0.03 * rng.standard_normal((nva, noa))])
        ),
        mo_t_b=jnp.asarray(
            np.vstack([np.eye(nob), 0.03 * rng.standard_normal((nvb, nob))])
        ),
        mo_coeff_b=jnp.eye(norb),
        eigenvalues=jnp.asarray(factorization.eigenvalues),
        modes=jnp.asarray(factorization.modes, dtype=jnp.float32),
    )
    sys = System(norb, (noa, nob), walker_kind="restricted")
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(2551),
        norb=norb,
        n_chol=5,
        basis="restricted",
    )
    params = QmcParams(
        dt=0.005,
        n_walkers=4,
        n_prop_steps=1,
        n_eql_blocks=1,
        n_blocks=2,
        n_chunks=1,
        seed=2557,
    )
    result = driver.run_qmc(
        sys=sys,
        params=params,
        ham_data=ham,
        trial_data=trial,
        trial_ops=make_ptuccsd_thouless_mode_trial_ops(sys),
        meas_ops=make_ptuccsd_mode_meas_ops(
            sys,
            n_mode_chunks=2,
            mixed_precision=True,
        ),
        prop_ops=make_prop_ops(ham.basis, sys.walker_kind, mixed_precision=True),
        block_fn=block,
    )

    assert result.block_energies.shape[0] > 0
    assert jnp.all(jnp.isfinite(result.block_energies))


def test_truncated_mode_force_bias_matches_overlap_derivative(trial_cases: TrialCases):
    case = trial_cases
    trial = PtuccsdThoulessModeTrial(
        mo_t_a=case.mode.mo_t_a,
        mo_t_b=case.mode.mo_t_b,
        mo_coeff_b=case.mode.mo_coeff_b,
        eigenvalues=case.mode.eigenvalues[:5],
        modes=case.mode.modes[:5],
    )
    sys = System(trial.norb, trial.nocc, walker_kind="restricted")
    cfg = PtuccsdModeMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    mode_ctx = build_ptuccsd_mode_meas_ctx(case.ham, trial, cfg)
    trial_ops = make_ptuccsd_thouless_mode_trial_ops(sys)
    auto_ops = make_auto_meas_ops(sys, trial_ops)
    auto_ctx = auto_ops.build_meas_ctx(case.ham, trial)

    expected = auto_ops.require_kernel(k_force_bias)(
        case.walker_r,
        case.ham,
        auto_ctx,
        trial,
    )
    actual = mode_force_bias_rw(case.walker_r, case.ham, mode_ctx, trial)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-11, atol=3.0e-11)


def test_positive_threshold_retains_expected_combined_modes(trial_cases: TrialCases):
    case = trial_cases
    magnitudes = np.sort(np.abs(np.linalg.eigvalsh(case.kernel)))[::-1]
    threshold = float(0.5 * (magnitudes[3] + magnitudes[4]))
    factorization = factorize_t2_modes(
        np.asarray(case.dense.t2aa),
        np.asarray(case.dense.t2ab),
        np.asarray(case.dense.t2bb),
        mode_threshold=threshold,
        solver="dense",
    )
    expected_values, expected_vectors = np.linalg.eigh(case.kernel)
    order = np.argsort(np.abs(expected_values))[::-1][:4]
    expected_kernel = (
        expected_vectors[:, order] * expected_values[order]
    ) @ expected_vectors[:, order].T
    actual_kernel = (factorization.modes.T * factorization.eigenvalues) @ factorization.modes

    assert factorization.rank == 4
    assert factorization.solver == "dense"
    assert factorization.discarded_norm_fraction > 0.0
    np.testing.assert_allclose(actual_kernel, expected_kernel, rtol=3.0e-12, atol=3.0e-12)


def test_discarded_norm_target_retains_minimum_required_combined_modes(
    trial_cases: TrialCases,
):
    case = trial_cases
    eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(case.kernel)))[::-1]
    full_norm_sq = float(np.vdot(eigenvalues, eigenvalues).real)
    discarded_at_three = np.sqrt(np.sum(eigenvalues[3:] ** 2) / full_norm_sq)
    discarded_at_four = np.sqrt(np.sum(eigenvalues[4:] ** 2) / full_norm_sq)
    target = float(0.5 * (discarded_at_three + discarded_at_four))

    natural = factorize_t2_modes(
        np.asarray(case.dense.t2aa),
        np.asarray(case.dense.t2ab),
        np.asarray(case.dense.t2bb),
        mode_threshold=None,
        discarded_norm_target=target,
        solver="dense",
    )
    extended = factorize_t2_modes(
        np.asarray(case.dense.t2aa),
        np.asarray(case.dense.t2ab),
        np.asarray(case.dense.t2bb),
        mode_threshold=None,
        discarded_norm_target=target,
        minimum_rank=6,
        solver="dense",
    )

    assert natural.rank == 4
    assert natural.natural_rank == 4
    assert natural.discarded_norm_fraction <= target
    assert extended.rank == 6
    assert extended.natural_rank == natural.natural_rank
    assert extended.discarded_norm_fraction < natural.discarded_norm_fraction


def test_ucisd_guide_and_t2_estimator_use_common_norm_controlled_rank(
    trial_cases: TrialCases,
):
    case = trial_cases
    noa, nob = case.dense.nocc
    t1a = np.asarray(case.dense.mo_t_a[noa:, :]).T
    t1b = np.asarray(case.dense.mo_t_b[nob:, :]).T
    t2aa = np.asarray(case.dense.t2aa)
    t2ab = np.asarray(case.dense.t2ab)
    t2bb = np.asarray(case.dense.t2bb)
    ci2aa = (
        t2aa
        + np.einsum("ia,jb->iajb", t1a, t1a)
        - np.einsum("ib,ja->iajb", t1a, t1a)
    )
    ci2ab = t2ab + np.einsum("ia,jb->iajb", t1a, t1b)
    ci2bb = (
        t2bb
        + np.einsum("ia,jb->iajb", t1b, t1b)
        - np.einsum("ib,ja->iajb", t1b, t1b)
    )
    target = 0.35

    guide_natural = factorize_ucisd_k_blocks(
        ci2aa,
        ci2ab,
        ci2bb,
        threshold=None,
        discarded_norm_target=target,
        solver="dense",
    )
    estimator_natural = factorize_t2_modes(
        t2aa,
        t2ab,
        t2bb,
        mode_threshold=None,
        discarded_norm_target=target,
        solver="dense",
    )
    ci2aa_work = ci2aa.copy()
    ci2ab_work = ci2ab.copy()
    ci2bb_work = ci2bb.copy()
    guide, estimator = factorize_ucisd_and_t2_modes_common_rank(
        ci2aa_work,
        ci2ab_work,
        ci2bb_work,
        t1a,
        t1b,
        mode_threshold=None,
        discarded_norm_target=target,
        solver="dense",
        overwrite_ci2=True,
    )

    expected_rank = max(guide_natural.rank, estimator_natural.rank)
    assert guide.rank == estimator.rank == expected_rank
    assert guide.natural_rank == guide_natural.rank
    assert estimator.natural_rank == estimator_natural.rank
    assert guide.discarded_norm_fraction <= target
    assert estimator.discarded_norm_fraction <= target
    np.testing.assert_allclose(ci2aa_work, t2aa, rtol=0.0, atol=2.0e-17)
    np.testing.assert_allclose(ci2ab_work, t2ab, rtol=0.0, atol=2.0e-17)
    np.testing.assert_allclose(ci2bb_work, t2bb, rtol=0.0, atol=2.0e-17)


def test_loader_factories_and_pytree_support_restricted_open_shell(trial_cases: TrialCases):
    case = trial_cases
    noa, nob = case.dense.nocc
    sys = System(norb=case.dense.norb, nelec=(noa, nob), walker_kind="restricted")
    data = {
        "mo_t_a": case.dense.mo_t_a,
        "mo_t_b": case.dense.mo_t_b,
        "mo_coeff_b": case.dense.mo_coeff_b,
        "t2aa": np.asarray(case.dense.t2aa).transpose(0, 2, 1, 3),
        "t2ab": np.asarray(case.dense.t2ab).transpose(0, 2, 1, 3),
        "t2bb": np.asarray(case.dense.t2bb).transpose(0, 2, 1, 3),
    }
    loaded = make_ptuccsd_thouless_mode_trial_data(
        data,
        sys,
        mixed_precision=False,
        mode_solver="dense",
    )
    precomputed = make_ptuccsd_thouless_mode_trial_data(
        {
            "mo_t_a": case.dense.mo_t_a,
            "mo_t_b": case.dense.mo_t_b,
            "mo_coeff_b": case.dense.mo_coeff_b,
            "eigenvalues": loaded.eigenvalues,
            "eigenvectors": loaded.modes.T,
        },
        sys,
        mixed_precision=True,
    )
    restored = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(precomputed),
        jax.tree_util.tree_leaves(precomputed),
    )

    assert loaded.modes.dtype == jnp.float64
    assert precomputed.modes.dtype == jnp.float32
    assert restored.mode_rank == precomputed.mode_rank
    assert make_ptuccsd_thouless_mode_trial_ops(sys).overlap is mode_overlap_r
    np.testing.assert_allclose(
        mode_overlap_r(case.walker_r, loaded),
        dense_overlap_r(case.walker_r, case.dense),
        rtol=3.0e-12,
        atol=3.0e-12,
    )

    unrestricted = System(
        norb=case.dense.norb,
        nelec=(noa, nob),
        walker_kind="unrestricted",
    )
    assert make_ptuccsd_thouless_mode_trial_ops(unrestricted).overlap is mode_overlap_u

    with pytest.raises(ValueError, match="nup >= ndn"):
        make_ptuccsd_thouless_mode_trial_ops(
            System(case.dense.norb, (nob, noa), walker_kind="restricted")
        )
    with pytest.raises(ValueError, match="orbital mismatch"):
        make_ptuccsd_thouless_mode_trial_data(
            data,
            System(case.dense.norb + 1, (noa, nob), walker_kind="restricted"),
            mixed_precision=False,
            mode_solver="dense",
        )
