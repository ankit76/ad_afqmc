from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot import testing
from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.meas.cisd import (
    CisdMeasCfg,
    build_meas_ctx as build_dense_meas_ctx,
    energy_kernel_rw_rh as dense_energy_kernel,
    force_bias_kernel_rw_rh_high as dense_force_bias_kernel,
)
from trot.meas.cisd_modes import (
    build_meas_ctx as build_mode_meas_ctx,
    energy_kernel_rw_rh as mode_energy_kernel,
    force_bias_kernel_rw_rh as mode_force_bias_kernel,
    get_cisd_mode_meas_cfg,
    make_cisd_mode_meas_ops,
)
from trot.trial.cisd import CisdTrial, overlap_r as dense_overlap_r
from trot.trial.cisd_modes import (
    CisdModeTrial,
    make_cisd_mode_trial_data,
    make_cisd_mode_trial_ops,
    mode_apply,
    mode_projections,
    mode_quadratic,
    overlap_r as mode_overlap_r,
)


def _make_dense_and_mode_trials(
    *,
    seed: int = 719,
    nocc: int = 3,
    nvir: int = 4,
    nocc_t_core: int = 0,
    nvir_t_outer: int = 0,
    mode_dtype=jnp.float64,
) -> tuple[CisdTrial, CisdModeTrial, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ci1 = 0.05 * rng.standard_normal((nocc, nvir))

    pair_dim = nocc * nvir
    ci2_pair = 0.02 * rng.standard_normal((pair_dim, pair_dim))
    ci2_pair = 0.5 * (ci2_pair + ci2_pair.T)
    ci2 = ci2_pair.reshape(nocc, nvir, nocc, nvir)

    direct = ci2.reshape(pair_dim, pair_dim)
    exchange = np.transpose(ci2, (0, 3, 2, 1)).reshape(pair_dim, pair_dim)
    kernel = 2.0 * direct - exchange
    np.testing.assert_allclose(kernel, kernel.T, rtol=0.0, atol=1.0e-14)

    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    modes = eigenvectors.T.reshape(pair_dim, nocc, nvir)
    dense_trial = CisdTrial(
        ci1=jnp.asarray(ci1, dtype=jnp.float64),
        ci2=jnp.asarray(ci2, dtype=jnp.float64),
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    mode_trial = CisdModeTrial(
        ci1=jnp.asarray(ci1, dtype=jnp.float64),
        eigenvalues=jnp.asarray(eigenvalues, dtype=jnp.float64),
        modes=jnp.asarray(modes, dtype=mode_dtype),
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    return dense_trial, mode_trial, kernel, eigenvectors


@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
def test_full_rank_double_mode_overlap_matches_dense_cisd(nocc_t_core, nvir_t_outer):
    dense_trial, mode_trial, _, _ = _make_dense_and_mode_trials(
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(811),
        mode_trial.norb,
        mode_trial.nocc_full,
        mix=0.25,
    )

    dense_overlap = dense_overlap_r(walker, dense_trial)
    mode_overlap = mode_overlap_r(walker, mode_trial)
    mode_overlap_jit = jax.jit(mode_overlap_r)(walker, mode_trial)
    np.testing.assert_allclose(mode_overlap, dense_overlap, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(mode_overlap_jit, dense_overlap, rtol=1.0e-12, atol=1.0e-12)


def test_mode_helpers_match_explicit_k_contractions():
    _, trial, kernel, _ = _make_dense_and_mode_trials()
    key_r, key_i = jax.random.split(jax.random.PRNGKey(827))
    matrix = jax.random.normal(key_r, (trial.nocc, trial.nvir), dtype=jnp.float64)
    matrix = matrix + 1.0j * jax.random.normal(key_i, (trial.nocc, trial.nvir), dtype=jnp.float64)

    projections = mode_projections(trial, matrix)
    projections_from_apply, applied = mode_apply(trial, matrix)
    quadratic = mode_quadratic(trial, matrix, projections)

    matrix_np = np.asarray(matrix).reshape(-1)
    expected_applied = (kernel @ matrix_np).reshape(trial.nocc, trial.nvir)
    expected_quadratic = matrix_np @ kernel @ matrix_np
    np.testing.assert_allclose(projections_from_apply, projections, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(applied, expected_applied, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(quadratic, expected_quadratic, rtol=1.0e-12, atol=1.0e-12)


def test_mixed_trial_data_uses_lambda64_vectors32_and_dp_reductions():
    dense_trial, mode_dp, _, eigenvectors = _make_dense_and_mode_trials()
    sys = System(
        norb=dense_trial.norb,
        nelec=(dense_trial.nocc_full, dense_trial.nocc_full),
        walker_kind="restricted",
    )
    data = {
        "ci1": np.asarray(dense_trial.ci1),
        "eigenvalues": np.asarray(mode_dp.eigenvalues),
        "eigenvectors": eigenvectors,
    }
    mode_mixed = make_cisd_mode_trial_data(data, sys, mixed_precision=True)
    mode_double = make_cisd_mode_trial_data(data, sys, mixed_precision=False)

    assert mode_mixed.ci1.dtype == jnp.float64
    assert mode_mixed.eigenvalues.dtype == jnp.float64
    assert mode_mixed.modes.dtype == jnp.float32
    assert mode_double.modes.dtype == jnp.float64

    matrix = jnp.asarray(
        np.random.default_rng(839).standard_normal((mode_mixed.nocc, mode_mixed.nvir)),
        dtype=jnp.complex128,
    )
    assert mode_projections(mode_mixed, matrix).dtype == jnp.complex128
    assert mode_quadratic(mode_mixed, matrix).dtype == jnp.complex128
    assert mode_apply(mode_mixed, matrix)[1].dtype == jnp.complex128

    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(853), mode_mixed.norb, mode_mixed.nocc_full, mix=0.25
    )
    dense_overlap = dense_overlap_r(walker, dense_trial)
    mixed_overlap = mode_overlap_r(walker, mode_mixed)
    relative_error = float(jnp.abs(mixed_overlap - dense_overlap) / jnp.abs(dense_overlap))
    assert relative_error < 1.0e-5


def test_mode_trial_is_a_pytree_and_rejects_truncated_storage():
    _, trial, _, _ = _make_dense_and_mode_trials(nocc_t_core=1, nvir_t_outer=2)
    leaves, treedef = jax.tree_util.tree_flatten(trial)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.nocc_t_core == 1
    assert restored.nvir_t_outer == 2
    np.testing.assert_array_equal(restored.eigenvalues, trial.eigenvalues)

    with pytest.raises(ValueError, match="full-rank modes"):
        CisdModeTrial(
            ci1=trial.ci1,
            eigenvalues=trial.eigenvalues[:-1],
            modes=trial.modes[:-1],
        )


def test_mode_trial_ops_require_restricted_closed_shell_walkers():
    restricted = System(norb=7, nelec=(2, 2), walker_kind="restricted")
    ops = make_cisd_mode_trial_ops(restricted)
    assert ops.overlap is mode_overlap_r

    with pytest.raises(ValueError, match="nup == ndn"):
        make_cisd_mode_trial_ops(System(norb=7, nelec=(3, 2), walker_kind="restricted"))
    with pytest.raises(ValueError, match="restricted walkers"):
        make_cisd_mode_trial_ops(System(norb=7, nelec=(2, 2), walker_kind="unrestricted"))


@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
def test_full_rank_double_mode_force_bias_and_energy_match_dense(
    nocc_t_core,
    nvir_t_outer,
):
    dense_trial, mode_trial, _, _ = _make_dense_and_mode_trials(
        nocc=2,
        nvir=3,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(877),
        norb=mode_trial.norb,
        n_chol=7,
        basis="restricted",
    )
    dp_cfg = CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    dense_ctx = build_dense_meas_ctx(ham, dense_trial, cfg=dp_cfg)
    mode_ctx = build_mode_meas_ctx(
        ham,
        mode_trial,
        cfg=dp_cfg,
        mode_chunk_size=4,
    )
    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(881),
        mode_trial.norb,
        mode_trial.nocc_full,
        mix=0.25,
    )

    dense_fb = dense_force_bias_kernel(walker, ham, dense_ctx, dense_trial)
    mode_fb = jax.jit(mode_force_bias_kernel)(walker, ham, mode_ctx, mode_trial)
    dense_energy = dense_energy_kernel(walker, ham, dense_ctx, dense_trial)
    mode_energy = jax.jit(mode_energy_kernel)(walker, ham, mode_ctx, mode_trial)

    np.testing.assert_allclose(mode_fb, dense_fb, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(mode_energy, dense_energy, rtol=2.0e-12, atol=2.0e-12)


def test_mode_measurement_mixed_precision_policy_and_accuracy():
    dense_trial, mode_double, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    mode_mixed = CisdModeTrial(
        ci1=mode_double.ci1,
        eigenvalues=mode_double.eigenvalues,
        modes=mode_double.modes.astype(jnp.float32),
    )
    sys = System(
        norb=mode_mixed.norb,
        nelec=(mode_mixed.nocc_full, mode_mixed.nocc_full),
        walker_kind="restricted",
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(883),
        norb=mode_mixed.norb,
        n_chol=7,
        basis="restricted",
    )
    dp_cfg = CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    dense_ctx = build_dense_meas_ctx(ham, dense_trial, cfg=dp_cfg)
    mode_ops = make_cisd_mode_meas_ops(sys, mixed_precision=True, mode_chunk_size=4)
    mode_ctx = mode_ops.build_meas_ctx(ham, mode_mixed)
    cfg = get_cisd_mode_meas_cfg(mode_ops)
    assert cfg is not None
    assert cfg.mixed_real_dtype == jnp.float32
    assert cfg.mixed_complex_dtype == jnp.complex64

    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(887), mode_mixed.norb, mode_mixed.nocc_full, mix=0.25
    )
    dense_fb = dense_force_bias_kernel(walker, ham, dense_ctx, dense_trial)
    mixed_fb = mode_ops.require_kernel(k_force_bias)(walker, ham, mode_ctx, mode_mixed)
    dense_energy = dense_energy_kernel(walker, ham, dense_ctx, dense_trial)
    mixed_energy = mode_ops.require_kernel(k_energy)(walker, ham, mode_ctx, mode_mixed)

    fb_relative_error = float(jnp.linalg.norm(mixed_fb - dense_fb) / jnp.linalg.norm(dense_fb))
    energy_absolute_error = float(jnp.abs(mixed_energy - dense_energy))
    assert fb_relative_error < 1.0e-5
    assert energy_absolute_error < 1.0e-4


def test_mode_energy_is_independent_of_chunk_partition_in_double_precision():
    dense_trial, mode_trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(907),
        norb=mode_trial.norb,
        n_chol=5,
        basis="restricted",
    )
    cfg = CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    dense_ctx = build_dense_meas_ctx(ham, dense_trial, cfg=cfg)
    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(911), mode_trial.norb, mode_trial.nocc_full, mix=0.25
    )
    reference = dense_energy_kernel(walker, ham, dense_ctx, dense_trial)

    for chunk_size in (1, 4, 32):
        mode_ctx = build_mode_meas_ctx(
            ham,
            mode_trial,
            cfg=cfg,
            mode_chunk_size=chunk_size,
        )
        candidate = jax.jit(mode_energy_kernel)(walker, ham, mode_ctx, mode_trial)
        np.testing.assert_allclose(candidate, reference, rtol=2.0e-12, atol=2.0e-12)
