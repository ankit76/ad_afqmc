from __future__ import annotations

from dataclasses import dataclass

from trot import config

config.configure_once()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot.core.ops import BlockComponentEstimate, k_energy, k_force_bias
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.pt2ccsd import build_meas_ctx as build_pt2_dense_ctx
from trot.meas.pt2ccsd import energy_kernel_rw_rh as pt2_dense_components
from trot.meas.ptccsd import build_ptccsd_meas_ctx
from trot.meas.ptccsd import energy_components_pt_rw_rh as pt_dense_inverse_components
from trot.meas.ptccsd import energy_pt_rw_rh as pt_dense_energy
from trot.meas.ptccsd import force_bias_pt_rw_rh as pt_dense_force_bias
from trot.meas.ptccsd_modes import (
    _components_pt_thouless_rw_rh_full,
    _force_bias_pt_thouless_rw_rh_full,
    _ptccsd_thouless_mode_chol_index_sum_for_walkers,
    _ptccsd_thouless_mode_chol_index_terms,
    _ptccsd_thouless_mode_chol_pair_terms,
    _ptccsd_thouless_mode_chol_terms,
    _ptccsd_thouless_mode_energy_common,
    PtccsdModePairSamplingCfg,
    build_ptccsd_mode_meas_ctx,
    build_ptccsd_thouless_mode_meas_ctx,
    components_pt_rw_rh,
    components_pt_thouless_rw_rh,
    energy_pt_rw_rh as pt_mode_energy,
    energy_pt_thouless_rw_rh as pt_thouless_mode_energy,
    force_bias_pt_rw_rh as pt_mode_force_bias,
    force_bias_pt_thouless_rw_rh as pt_thouless_mode_force_bias,
    inverse_guide_components_pt_rw_rh as pt_mode_inverse_components,
    make_ptccsd_mode_meas_ops,
    make_ptccsd_thouless_mode_estimator_ops,
    make_ptccsd_thouless_mode_meas_ops,
    pair_sampled_ptccsd_block_components,
)
from trot.meas.ptccsd_thouless import build_ptccsd_thouless_meas_ctx
from trot.meas.ptccsd_thouless import energy_pt_rw_rh as pt_thouless_dense_energy
from trot.meas.ptccsd_thouless import force_bias_pt_rw_rh as pt_thouless_dense_force_bias
from trot.trial.pt2ccsd import Pt2ccsdTrial
from trot.trial.ptccsd import PtccsdTrial, overlap_pt_r as pt_dense_overlap
from trot.trial.ptccsd_modes import (
    PtccsdModeTrial,
    PtccsdThoulessModeTrial,
    decompose_t2_modes,
    overlap_pt_r as pt_mode_overlap,
    overlap_ptccsd_thouless_r as pt_thouless_mode_overlap,
)
from trot.trial.ptccsd_thouless import (
    PtccsdThoulessTrial,
    overlap_ptccsd_thouless_r as pt_thouless_dense_overlap,
)


@dataclass(frozen=True)
class PtCases:
    sys: System
    ham: HamChol
    walkers: jax.Array
    dense: PtccsdTrial
    mode: PtccsdModeTrial
    dense_thouless: PtccsdThoulessTrial
    mode_thouless: PtccsdThoulessModeTrial


@pytest.fixture(scope="module")
def pt_cases() -> PtCases:
    rng = np.random.default_rng(8127)
    nocc, nvir, nchol = 2, 3, 5
    norb = nocc + nvir

    t1 = 0.08 * rng.normal(size=(nocc, nvir))
    raw = 0.04 * rng.normal(size=(nocc, nvir, nocc, nvir))
    t2 = 0.5 * (raw + raw.transpose(2, 3, 0, 1))
    eigenvalues, modes = decompose_t2_modes(t2)

    h1_raw = rng.normal(size=(norb, norb))
    h1 = 0.5 * (h1_raw + h1_raw.T)
    chol_raw = rng.normal(size=(nchol, norb, norb))
    chol = 0.5 * (chol_raw + chol_raw.transpose(0, 2, 1))
    ham = HamChol(
        h0=jnp.asarray(0.37),
        h1=jnp.asarray(h1),
        chol=jnp.asarray(chol),
        basis="restricted",
    )

    walkers = []
    for _ in range(4):
        walker = np.eye(norb, nocc) + 0.12 * rng.normal(size=(norb, nocc))
        walker = walker + 0.07j * rng.normal(size=(norb, nocc))
        walkers.append(walker)
    walkers_array = jnp.asarray(np.stack(walkers))

    mo_t = jnp.vstack([jnp.eye(nocc), jnp.asarray(t1).T])
    dense = PtccsdTrial(t1=jnp.asarray(t1), t2=jnp.asarray(t2))
    mode = PtccsdModeTrial(
        t1=jnp.asarray(t1),
        eigenvalues=jnp.asarray(eigenvalues),
        modes=jnp.asarray(modes),
    )
    dense_thouless = PtccsdThoulessTrial(mo_t=mo_t, t2=jnp.asarray(t2))
    mode_thouless = PtccsdThoulessModeTrial(
        mo_t=mo_t,
        eigenvalues=jnp.asarray(eigenvalues),
        modes=jnp.asarray(modes),
    )
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")
    return PtCases(
        sys=sys,
        ham=ham,
        walkers=walkers_array,
        dense=dense,
        mode=mode,
        dense_thouless=dense_thouless,
        mode_thouless=mode_thouless,
    )


@pytest.mark.parametrize("n_mode_chunks", [1, 2, 5])
def test_dense_and_full_rank_pt_modes_match(pt_cases: PtCases, n_mode_chunks: int):
    case = pt_cases
    dense_ctx = build_ptccsd_meas_ctx(case.ham, case.dense)
    mode_ctx = build_ptccsd_mode_meas_ctx(
        case.ham,
        case.mode,
        n_mode_chunks=n_mode_chunks,
    )

    dense_overlap = jax.vmap(pt_dense_overlap, in_axes=(0, None))(case.walkers, case.dense)
    mode_overlap = jax.vmap(pt_mode_overlap, in_axes=(0, None))(case.walkers, case.mode)
    np.testing.assert_allclose(mode_overlap, dense_overlap, rtol=2.0e-11, atol=2.0e-11)

    dense_fb = jax.vmap(pt_dense_force_bias, in_axes=(0, None, None, None))(
        case.walkers, case.ham, dense_ctx, case.dense
    )
    mode_fb = jax.vmap(pt_mode_force_bias, in_axes=(0, None, None, None))(
        case.walkers, case.ham, mode_ctx, case.mode
    )
    np.testing.assert_allclose(mode_fb, dense_fb, rtol=2.0e-10, atol=2.0e-10)

    dense_energy = jax.vmap(pt_dense_energy, in_axes=(0, None, None, None))(
        case.walkers, case.ham, dense_ctx, case.dense
    )
    mode_energy = jax.vmap(pt_mode_energy, in_axes=(0, None, None, None))(
        case.walkers, case.ham, mode_ctx, case.mode
    )
    np.testing.assert_allclose(mode_energy, dense_energy, rtol=3.0e-10, atol=3.0e-10)

    dense_components = jax.vmap(
        pt_dense_inverse_components,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, dense_ctx, case.dense)
    mode_components = jax.vmap(
        pt_mode_inverse_components,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, mode_ctx, case.mode)
    np.testing.assert_allclose(mode_components, dense_components, rtol=3.0e-10, atol=3.0e-10)


@pytest.mark.parametrize("n_mode_chunks", [1, 2, 5])
def test_dense_and_full_rank_thouless_modes_match(pt_cases: PtCases, n_mode_chunks: int):
    case = pt_cases
    dense_ctx = build_ptccsd_thouless_meas_ctx(case.ham, case.dense_thouless)
    mode_ctx = build_ptccsd_thouless_mode_meas_ctx(
        case.ham,
        case.mode_thouless,
        n_mode_chunks=n_mode_chunks,
    )

    dense_overlap = jax.vmap(pt_thouless_dense_overlap, in_axes=(0, None))(
        case.walkers, case.dense_thouless
    )
    mode_overlap = jax.vmap(pt_thouless_mode_overlap, in_axes=(0, None))(
        case.walkers, case.mode_thouless
    )
    np.testing.assert_allclose(mode_overlap, dense_overlap, rtol=2.0e-11, atol=2.0e-11)

    dense_fb = jax.vmap(pt_thouless_dense_force_bias, in_axes=(0, None, None, None))(
        case.walkers, case.ham, dense_ctx, case.dense_thouless
    )
    mode_fb = jax.vmap(
        pt_thouless_mode_force_bias,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, mode_ctx, case.mode_thouless)
    np.testing.assert_allclose(mode_fb, dense_fb, rtol=3.0e-10, atol=3.0e-10)

    dense_energy = jax.vmap(pt_thouless_dense_energy, in_axes=(0, None, None, None))(
        case.walkers, case.ham, dense_ctx, case.dense_thouless
    )
    mode_energy = jax.vmap(
        pt_thouless_mode_energy,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, mode_ctx, case.mode_thouless)
    np.testing.assert_allclose(mode_energy, dense_energy, rtol=3.0e-10, atol=3.0e-10)


def test_dense_pt2_components_and_full_rank_modes_match(pt_cases: PtCases):
    case = pt_cases
    dense_pt2 = Pt2ccsdTrial(
        mo_t=case.dense_thouless.mo_t,
        t2=case.dense_thouless.t2,
    )
    dense_ctx = build_pt2_dense_ctx(case.ham, dense_pt2)
    mode_ctx = build_ptccsd_thouless_mode_meas_ctx(case.ham, case.mode_thouless)

    dense_components = jax.vmap(pt2_dense_components, in_axes=(0, None, None, None))(
        case.walkers, case.ham, dense_ctx, dense_pt2
    )
    mode_components = jax.vmap(
        components_pt_thouless_rw_rh,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, mode_ctx, case.mode_thouless)
    np.testing.assert_allclose(mode_components, dense_components, rtol=3.0e-10, atol=3.0e-10)

    weights = jnp.asarray([0.7, 1.2, 0.9, 1.4])
    dense_avg = jnp.sum(weights[:, None] * dense_components, axis=0) / jnp.sum(weights)
    mode_avg = jnp.sum(weights[:, None] * mode_components, axis=0) / jnp.sum(weights)

    def combine(components):
        theta, electronic_0, h_t = components
        return case.ham.h0 + electronic_0 + h_t - theta * electronic_0

    np.testing.assert_allclose(combine(mode_avg), combine(dense_avg), rtol=3.0e-10, atol=3.0e-10)


@pytest.mark.parametrize("memory_mode", ["high", "low"])
@pytest.mark.parametrize("n_mode_chunks", [1, 2, 5])
def test_half_green_thouless_kernels_match_full_green_oracle(
    pt_cases: PtCases,
    n_mode_chunks: int,
    memory_mode: str,
):
    case = pt_cases
    ctx = build_ptccsd_thouless_mode_meas_ctx(
        case.ham,
        case.mode_thouless,
        n_mode_chunks=n_mode_chunks,
        memory_mode=memory_mode,
    )

    full_force_bias = jax.vmap(
        _force_bias_pt_thouless_rw_rh_full,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    half_force_bias = jax.vmap(
        pt_thouless_mode_force_bias,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    np.testing.assert_allclose(half_force_bias, full_force_bias, rtol=3.0e-10, atol=3.0e-10)

    full_components = jax.vmap(
        _components_pt_thouless_rw_rh_full,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    half_components = jax.vmap(
        components_pt_thouless_rw_rh,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    np.testing.assert_allclose(half_components, full_components, rtol=3.0e-10, atol=3.0e-10)

    half_energy = jax.vmap(
        pt_thouless_mode_energy,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    full_energy = (
        case.ham.h0
        + full_components[:, 1]
        + full_components[:, 2]
        - full_components[:, 0] * full_components[:, 1]
    )
    np.testing.assert_allclose(half_energy, full_energy, rtol=3.0e-10, atol=3.0e-10)


@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_half_green_thouless_kernels_match_full_oracle_in_complex_gauge(
    pt_cases: PtCases,
    memory_mode: str,
):
    """Exercise the identities without assuming the occupied block of C is I."""

    case = pt_cases
    occupied_gauge = jnp.asarray(
        [[1.1 + 0.2j, -0.1 + 0.05j], [0.08 - 0.04j, 0.9 - 0.15j]]
    )
    trial = PtccsdThoulessModeTrial(
        mo_t=case.mode_thouless.mo_t @ occupied_gauge,
        eigenvalues=case.mode_thouless.eigenvalues,
        modes=case.mode_thouless.modes,
    )
    ctx = build_ptccsd_thouless_mode_meas_ctx(
        case.ham,
        trial,
        n_mode_chunks=2,
        memory_mode=memory_mode,
    )

    full_force_bias = jax.vmap(
        _force_bias_pt_thouless_rw_rh_full,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, trial)
    half_force_bias = jax.vmap(
        pt_thouless_mode_force_bias,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, trial)
    np.testing.assert_allclose(half_force_bias, full_force_bias, rtol=3.0e-10, atol=3.0e-10)

    full_components = jax.vmap(
        _components_pt_thouless_rw_rh_full,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, trial)
    half_components = jax.vmap(
        components_pt_thouless_rw_rh,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, trial)
    np.testing.assert_allclose(half_components, full_components, rtol=3.0e-10, atol=3.0e-10)


@pytest.mark.parametrize("memory_mode", ["high", "low"])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_thouless_connected_residual_decomposition_matches_components(
    pt_cases: PtCases,
    memory_mode: str,
    mixed_precision: bool,
):
    case = pt_cases
    ops = make_ptccsd_thouless_mode_meas_ops(
        case.sys,
        n_mode_chunks=2,
        memory_mode=memory_mode,
        mixed_precision=mixed_precision,
    )
    ctx = ops.build_meas_ctx(case.ham, case.mode_thouless)
    full_precision_ctx = build_ptccsd_thouless_mode_meas_ctx(
        case.ham,
        case.mode_thouless,
        n_mode_chunks=2,
        memory_mode=memory_mode,
    )

    def reconstruct(walker):
        common = _ptccsd_thouless_mode_energy_common(
            walker,
            case.ham,
            ctx,
            case.mode_thouless,
        )
        residual = _ptccsd_thouless_mode_chol_terms(
            common,
            case.ham.chol,
            ctx.rot_chol,
            ctx,
            case.mode_thouless,
        )
        return jnp.stack(
            [
                common.theta,
                common.electronic_0,
                common.h_t_base + jnp.sum(residual),
            ]
        )

    reconstructed = jax.jit(jax.vmap(reconstruct))(case.walkers)
    components = jax.vmap(
        components_pt_thouless_rw_rh,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    full_precision = jax.vmap(
        components_pt_thouless_rw_rh,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, full_precision_ctx, case.mode_thouless)
    full_green_oracle = jax.vmap(
        _components_pt_thouless_rw_rh_full,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)

    algebra_tolerance = 1.0e-7 if mixed_precision else 3.0e-12
    np.testing.assert_allclose(
        reconstructed,
        components,
        rtol=algebra_tolerance,
        atol=algebra_tolerance,
    )
    tolerance = 2.0e-4 if mixed_precision else 3.0e-10
    np.testing.assert_allclose(
        reconstructed,
        full_precision,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        reconstructed,
        full_green_oracle,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_thouless_residual_index_head_and_pair_kernels_are_exact(
    pt_cases: PtCases,
    memory_mode: str,
):
    case = pt_cases
    ctx = build_ptccsd_thouless_mode_meas_ctx(
        case.ham,
        case.mode_thouless,
        n_mode_chunks=2,
        memory_mode=memory_mode,
    )
    common = jax.vmap(
        _ptccsd_thouless_mode_energy_common,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    all_terms = jax.vmap(
        _ptccsd_thouless_mode_chol_terms,
        in_axes=(0, None, None, None, None),
    )(common, case.ham.chol, ctx.rot_chol, ctx, case.mode_thouless)

    head_indices = jnp.asarray([4, 1, 3], dtype=jnp.int32)
    indexed_terms = jax.jit(
        lambda common_i, indices: _ptccsd_thouless_mode_chol_index_terms(
            common_i,
            indices,
            case.ham,
            ctx,
            case.mode_thouless,
            n_chunks=2,
        )
    )(jax.tree_util.tree_map(lambda value: value[2], common), head_indices)
    np.testing.assert_allclose(
        indexed_terms,
        all_terms[2, head_indices],
        rtol=3.0e-12,
        atol=3.0e-12,
    )

    head_sum = jax.jit(
        lambda common_i: _ptccsd_thouless_mode_chol_index_sum_for_walkers(
            common_i,
            head_indices,
            case.ham,
            ctx,
            case.mode_thouless,
            n_walker_chunks=2,
            chol_batch_size=2,
        )
    )(common)
    np.testing.assert_allclose(
        head_sum,
        jnp.sum(all_terms[:, head_indices], axis=1),
        rtol=3.0e-12,
        atol=3.0e-12,
    )

    empty_head = _ptccsd_thouless_mode_chol_index_sum_for_walkers(
        common,
        jnp.asarray([], dtype=jnp.int32),
        case.ham,
        ctx,
        case.mode_thouless,
        n_walker_chunks=2,
        chol_batch_size=2,
    )
    np.testing.assert_array_equal(empty_head, jnp.zeros_like(common.h_t_base))

    sample_walker = jnp.asarray([3, 0, 2, 1, 3], dtype=jnp.int32)
    sample_chol = jnp.asarray([4, 0, 2, 1, 3], dtype=jnp.int32)
    pair_terms = jax.jit(
        lambda common_i, walker_i, chol_i: _ptccsd_thouless_mode_chol_pair_terms(
            common_i,
            walker_i,
            chol_i,
            case.ham,
            ctx,
            case.mode_thouless,
            n_chunks=2,
        )
    )(common, sample_walker, sample_chol)
    np.testing.assert_allclose(
        pair_terms,
        all_terms[sample_walker, sample_chol],
        rtol=3.0e-12,
        atol=3.0e-12,
    )


def test_ptccsd_component_sampling_configuration_and_factory(pt_cases: PtCases):
    case = pt_cases
    sampling = PtccsdModePairSamplingCfg(
        chol_head_size=2,
        pair_sample_size=16,
        head_chol_batch_size=1,
        pair_sample_batch_size=4,
        track_half_sample_diagnostic=True,
    )
    deterministic_ops = make_ptccsd_thouless_mode_estimator_ops(
        case.sys,
        mixed_precision=False,
    )
    sampled_ops = make_ptccsd_thouless_mode_estimator_ops(
        case.sys,
        mixed_precision=False,
        component_sampling=sampling,
    )
    assert deterministic_ops.block_components is None
    assert sampled_ops.block_components is pair_sampled_ptccsd_block_components

    ctx = sampled_ops.build_estimator_ctx(case.ham, case.mode_thouless)
    assert ctx.component_sampling == sampling
    assert ctx.reference_chol_scores.shape == (case.ham.chol.shape[0],)
    assert ctx.chol_head_indices.shape == (2,)
    assert ctx.chol_tail_indices.shape == (3,)
    np.testing.assert_allclose(jnp.sum(ctx.chol_tail_prob), 1.0, atol=1.0e-14)
    assert bool(jnp.all(ctx.chol_tail_prob > 0.0))

    with pytest.raises(ValueError, match="must not exceed"):
        build_ptccsd_thouless_mode_meas_ctx(
            case.ham,
            case.mode_thouless,
            component_sampling=PtccsdModePairSamplingCfg(
                chol_head_size=case.ham.chol.shape[0] + 1,
                pair_sample_size=8,
            ),
        )
    with pytest.raises(ValueError, match="at least two"):
        PtccsdModePairSamplingCfg(
            chol_head_size=0,
            pair_sample_size=1,
            track_half_sample_diagnostic=True,
        )


def test_ptccsd_full_head_block_components_match_exact_complex_numerator(
    pt_cases: PtCases,
):
    case = pt_cases
    sampling = PtccsdModePairSamplingCfg(
        chol_head_size=case.ham.chol.shape[0],
        pair_sample_size=8,
        track_half_sample_diagnostic=True,
    )
    ops = make_ptccsd_thouless_mode_estimator_ops(
        case.sys,
        n_mode_chunks=2,
        mixed_precision=False,
        component_sampling=sampling,
    )
    ctx = ops.build_estimator_ctx(case.ham, case.mode_thouless)
    candidate_weights = jnp.asarray(
        [1.0 + 0.2j, 0.7 - 0.1j, 1.3 + 0.4j, 0.9 - 0.3j],
        dtype=jnp.complex128,
    )
    exact_components = jax.vmap(
        components_pt_thouless_rw_rh,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    expected_weight = jnp.sum(candidate_weights)
    expected_numerator = jnp.sum(
        candidate_weights[:, None] * exact_components,
        axis=0,
    )

    evaluate = jax.jit(pair_sampled_ptccsd_block_components, static_argnums=3)
    result = evaluate(
        case.walkers,
        candidate_weights,
        jax.random.PRNGKey(1201),
        2,
        case.ham,
        ctx,
        case.mode_thouless,
    )
    result_other_key = evaluate(
        case.walkers,
        candidate_weights,
        jax.random.PRNGKey(1203),
        2,
        case.ham,
        ctx,
        case.mode_thouless,
    )

    assert isinstance(result, BlockComponentEstimate)
    assert ctx.chol_tail_indices.shape == (0,)
    np.testing.assert_allclose(result.weight, expected_weight, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        result.numerator,
        expected_numerator,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        result_other_key.numerator,
        result.numerator,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.diagnostics["pt_component_sampling_noise_real"], 0.0
    )
    np.testing.assert_allclose(
        result.diagnostics["pt_component_sampling_noise_imag"], 0.0
    )


def test_ptccsd_sampled_tail_is_unbiased_for_complex_block_numerator(
    pt_cases: PtCases,
):
    case = pt_cases
    sampling = PtccsdModePairSamplingCfg(
        chol_head_size=1,
        pair_sample_size=16384,
        head_chol_batch_size=1,
        pair_sample_batch_size=128,
        tail_probability_uniform_mix=0.05,
        track_half_sample_diagnostic=True,
    )
    ops = make_ptccsd_thouless_mode_estimator_ops(
        case.sys,
        n_mode_chunks=2,
        mixed_precision=False,
        component_sampling=sampling,
    )
    ctx = ops.build_estimator_ctx(case.ham, case.mode_thouless)
    candidate_weights = jnp.asarray(
        [1.0 + 0.5j, -0.35 + 0.8j, 0.9 - 0.4j, 0.6 + 0.2j],
        dtype=jnp.complex128,
    )
    common = jax.vmap(
        _ptccsd_thouless_mode_energy_common,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    all_terms = jax.vmap(
        _ptccsd_thouless_mode_chol_terms,
        in_axes=(0, None, None, None, None),
    )(common, case.ham.chol, ctx.rot_chol, ctx, case.mode_thouless)
    exact_components = jax.vmap(
        components_pt_thouless_rw_rh,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    exact_numerator = jnp.sum(candidate_weights[:, None] * exact_components, axis=0)

    result = jax.jit(pair_sampled_ptccsd_block_components, static_argnums=3)(
        case.walkers,
        candidate_weights,
        jax.random.PRNGKey(1213),
        2,
        case.ham,
        ctx,
        case.mode_thouless,
    )

    walker_prob = jnp.abs(candidate_weights) / jnp.sum(jnp.abs(candidate_weights))
    tail_terms = all_terms[:, ctx.chol_tail_indices]
    importance_values = (
        candidate_weights[:, None]
        * tail_terms
        / (walker_prob[:, None] * ctx.chol_tail_prob[None, :])
    )
    joint_prob = walker_prob[:, None] * ctx.chol_tail_prob[None, :]
    tail_mean = jnp.sum(joint_prob * importance_values)
    exact_tail_numerator = jnp.sum(candidate_weights[:, None] * tail_terms)
    real_variance = jnp.sum(
        joint_prob * (jnp.real(importance_values) - jnp.real(tail_mean)) ** 2
    )
    imag_variance = jnp.sum(
        joint_prob * (jnp.imag(importance_values) - jnp.imag(tail_mean)) ** 2
    )
    real_tolerance = 8.0 * jnp.sqrt(real_variance / sampling.pair_sample_size) + 1.0e-10
    imag_tolerance = 8.0 * jnp.sqrt(imag_variance / sampling.pair_sample_size) + 1.0e-10

    np.testing.assert_allclose(result.weight, jnp.sum(candidate_weights), atol=2.0e-12)
    np.testing.assert_allclose(result.numerator[:2], exact_numerator[:2], atol=3.0e-12)
    np.testing.assert_allclose(tail_mean, exact_tail_numerator, atol=3.0e-12)
    assert abs(float(jnp.real(result.numerator[2] - exact_numerator[2]))) < float(
        real_tolerance
    )
    assert abs(float(jnp.imag(result.numerator[2] - exact_numerator[2]))) < float(
        imag_tolerance
    )
    assert np.isfinite(result.diagnostics["pt_component_sampling_noise_real"])
    assert np.isfinite(result.diagnostics["pt_component_sampling_noise_imag"])


def test_mode_meas_factories_expose_guide_kernels(pt_cases: PtCases):
    pt_ops = make_ptccsd_mode_meas_ops(pt_cases.sys)
    thouless_ops = make_ptccsd_thouless_mode_meas_ops(pt_cases.sys)
    assert pt_ops.has_kernel(k_force_bias) and pt_ops.has_kernel(k_energy)
    assert thouless_ops.has_kernel(k_force_bias) and thouless_ops.has_kernel(k_energy)
    assert thouless_ops.build_meas_ctx(pt_cases.ham, pt_cases.mode_thouless).memory_mode == "high"


def test_ptccsd_mode_mixed_precision_matches_cisd_accuracy_policy(pt_cases: PtCases):
    case = pt_cases
    mixed_trial = PtccsdModeTrial(
        t1=case.mode.t1,
        eigenvalues=case.mode.eigenvalues,
        modes=case.mode.modes.astype(jnp.float32),
    )
    full_ctx = build_ptccsd_mode_meas_ctx(case.ham, case.mode, n_mode_chunks=2)
    mixed_ops = make_ptccsd_mode_meas_ops(case.sys, n_mode_chunks=2)
    mixed_ctx = mixed_ops.build_meas_ctx(case.ham, mixed_trial)

    assert mixed_ctx.cfg.mixed_real_dtype == jnp.float32
    assert mixed_ctx.cfg.mixed_complex_dtype == jnp.complex64

    full_fb = jax.vmap(pt_mode_force_bias, in_axes=(0, None, None, None))(
        case.walkers, case.ham, full_ctx, case.mode
    )
    mixed_fb = jax.vmap(pt_mode_force_bias, in_axes=(0, None, None, None))(
        case.walkers, case.ham, mixed_ctx, mixed_trial
    )
    full_energy = jax.vmap(pt_mode_energy, in_axes=(0, None, None, None))(
        case.walkers, case.ham, full_ctx, case.mode
    )
    mixed_energy = jax.vmap(pt_mode_energy, in_axes=(0, None, None, None))(
        case.walkers, case.ham, mixed_ctx, mixed_trial
    )
    full_overlap = jax.vmap(pt_mode_overlap, in_axes=(0, None))(case.walkers, case.mode)
    mixed_overlap = jax.vmap(pt_mode_overlap, in_axes=(0, None))(
        case.walkers, mixed_trial
    )

    overlap_error = float(
        jnp.linalg.norm(mixed_overlap - full_overlap) / jnp.linalg.norm(full_overlap)
    )
    fb_error = float(jnp.linalg.norm(mixed_fb - full_fb) / jnp.linalg.norm(full_fb))
    energy_error = float(jnp.max(jnp.abs(mixed_energy - full_energy)))
    assert overlap_error < 1.0e-5
    assert fb_error < 2.0e-5
    assert energy_error < 2.0e-4


@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_ptccsd_thouless_mixed_precision_matches_cisd_accuracy_policy(
    pt_cases: PtCases,
    memory_mode: str,
):
    case = pt_cases
    mixed_trial = PtccsdThoulessModeTrial(
        mo_t=case.mode_thouless.mo_t,
        eigenvalues=case.mode_thouless.eigenvalues,
        modes=case.mode_thouless.modes.astype(jnp.float32),
    )
    full_ctx = build_ptccsd_thouless_mode_meas_ctx(
        case.ham,
        case.mode_thouless,
        n_mode_chunks=2,
        memory_mode=memory_mode,
    )
    mixed_ops = make_ptccsd_thouless_mode_meas_ops(
        case.sys,
        n_mode_chunks=2,
        memory_mode=memory_mode,
    )
    mixed_ctx = mixed_ops.build_meas_ctx(case.ham, mixed_trial)

    assert mixed_ctx.cfg.mixed_real_dtype == jnp.float32
    assert mixed_ctx.cfg.mixed_complex_dtype == jnp.complex64

    full_fb = jax.vmap(
        pt_thouless_mode_force_bias,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, full_ctx, case.mode_thouless)
    mixed_fb = jax.vmap(
        pt_thouless_mode_force_bias,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, mixed_ctx, mixed_trial)
    full_components = jax.vmap(
        components_pt_thouless_rw_rh,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, full_ctx, case.mode_thouless)
    mixed_components = jax.vmap(
        components_pt_thouless_rw_rh,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, mixed_ctx, mixed_trial)
    full_energy = jax.vmap(
        pt_thouless_mode_energy,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, full_ctx, case.mode_thouless)
    mixed_energy = jax.vmap(
        pt_thouless_mode_energy,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, mixed_ctx, mixed_trial)

    fb_error = float(jnp.linalg.norm(mixed_fb - full_fb) / jnp.linalg.norm(full_fb))
    component_error = float(jnp.max(jnp.abs(mixed_components - full_components)))
    energy_error = float(jnp.max(jnp.abs(mixed_energy - full_energy)))
    assert fb_error < 2.0e-5
    assert component_error < 2.0e-4
    assert energy_error < 2.0e-4
