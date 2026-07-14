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
    CisdModePairSamplingCfg,
    _cisd_mode_chol_pair_terms,
    _cisd_mode_chol_terms_for_walkers,
    _cisd_mode_energy_common,
    build_meas_ctx as build_mode_meas_ctx,
    energy_kernel_rw_rh as mode_energy_kernel,
    force_bias_kernel_rw_rh as mode_force_bias_kernel,
    get_cisd_mode_meas_cfg,
    make_cisd_mode_meas_ops,
    pair_sampled_block_energy,
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
    mode_overlap_batch = jax.vmap(mode_overlap_r, in_axes=(0, None))(
        jnp.stack((walker, walker)), mode_trial
    )
    np.testing.assert_allclose(mode_overlap, dense_overlap, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(mode_overlap_jit, dense_overlap, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        mode_overlap_batch,
        jnp.stack((dense_overlap, dense_overlap)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


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


def test_mode_trial_is_a_pytree_and_accepts_consistent_truncated_storage():
    dense_trial, trial, kernel, eigenvectors = _make_dense_and_mode_trials(
        nocc_t_core=1, nvir_t_outer=2
    )
    leaves, treedef = jax.tree_util.tree_flatten(trial)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.nocc_t_core == 1
    assert restored.nvir_t_outer == 2
    np.testing.assert_array_equal(restored.eigenvalues, trial.eigenvalues)

    mode_rank = trial.mode_rank - 2
    truncated = CisdModeTrial(
        ci1=trial.ci1,
        eigenvalues=trial.eigenvalues[:mode_rank],
        modes=trial.modes[:mode_rank],
        nocc_t_core=trial.nocc_t_core,
        nvir_t_outer=trial.nvir_t_outer,
    )
    assert truncated.mode_rank == mode_rank

    sys = System(
        norb=dense_trial.norb,
        nelec=(dense_trial.nocc_full, dense_trial.nocc_full),
        walker_kind="restricted",
    )
    loaded = make_cisd_mode_trial_data(
        {
            "ci1": np.asarray(trial.ci1),
            "eigenvalues": np.asarray(trial.eigenvalues[:mode_rank]),
            "eigenvectors": eigenvectors[:, :mode_rank],
            "nocc_t_core": trial.nocc_t_core,
            "nvir_t_outer": trial.nvir_t_outer,
        },
        sys,
        mixed_precision=False,
    )
    assert loaded.modes.shape == (mode_rank, trial.nocc, trial.nvir)
    np.testing.assert_allclose(loaded.eigenvalues, truncated.eigenvalues)
    np.testing.assert_allclose(loaded.modes, truncated.modes)

    matrix = np.random.default_rng(863).standard_normal((trial.nocc, trial.nvir))
    matrix = jnp.asarray(matrix, dtype=jnp.complex128)
    truncated_modes = np.asarray(truncated.modes).reshape(mode_rank, -1)
    truncated_kernel = (
        truncated_modes.T @ np.diag(np.asarray(truncated.eigenvalues)) @ truncated_modes
    )
    expected_applied = (truncated_kernel @ np.asarray(matrix).reshape(-1)).reshape(matrix.shape)
    _, applied = mode_apply(truncated, matrix)
    expected_quadratic = (
        np.asarray(matrix).reshape(-1) @ truncated_kernel @ np.asarray(matrix).reshape(-1)
    )
    np.testing.assert_allclose(applied, expected_applied, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        mode_quadratic(truncated, matrix),
        expected_quadratic,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert kernel.shape == (trial.mode_rank, trial.mode_rank)

    with pytest.raises(ValueError, match="eigenvalues must have shape"):
        CisdModeTrial(
            ci1=trial.ci1,
            eigenvalues=trial.eigenvalues[: mode_rank - 1],
            modes=trial.modes[:mode_rank],
        )


def test_mode_trial_ops_require_restricted_closed_shell_walkers():
    restricted = System(norb=7, nelec=(2, 2), walker_kind="restricted")
    ops = make_cisd_mode_trial_ops(restricted)
    assert ops.overlap is mode_overlap_r

    with pytest.raises(ValueError, match="nup == ndn"):
        make_cisd_mode_trial_ops(System(norb=7, nelec=(3, 2), walker_kind="restricted"))
    with pytest.raises(ValueError, match="restricted walkers"):
        make_cisd_mode_trial_ops(System(norb=7, nelec=(2, 2), walker_kind="unrestricted"))


def test_truncated_overlap_force_bias_and_energy_match_zero_padded_modes():
    _, full_trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    mode_rank = 3
    truncated_trial = CisdModeTrial(
        ci1=full_trial.ci1,
        eigenvalues=full_trial.eigenvalues[:mode_rank],
        modes=full_trial.modes[:mode_rank],
    )
    zero_padded_trial = CisdModeTrial(
        ci1=full_trial.ci1,
        eigenvalues=full_trial.eigenvalues.at[mode_rank:].set(0.0),
        modes=full_trial.modes,
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(869),
        norb=full_trial.norb,
        n_chol=7,
        basis="restricted",
    )
    cfg = CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    truncated_ctx = build_mode_meas_ctx(ham, truncated_trial, cfg=cfg, n_mode_chunks=2)
    padded_ctx = build_mode_meas_ctx(ham, zero_padded_trial, cfg=cfg, n_mode_chunks=2)
    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(871),
        full_trial.norb,
        full_trial.nocc_full,
        mix=0.25,
    )

    truncated_overlap = jax.jit(mode_overlap_r)(walker, truncated_trial)
    padded_overlap = jax.jit(mode_overlap_r)(walker, zero_padded_trial)
    truncated_fb = jax.jit(mode_force_bias_kernel)(walker, ham, truncated_ctx, truncated_trial)
    padded_fb = jax.jit(mode_force_bias_kernel)(walker, ham, padded_ctx, zero_padded_trial)
    truncated_energy = jax.jit(mode_energy_kernel)(walker, ham, truncated_ctx, truncated_trial)
    padded_energy = jax.jit(mode_energy_kernel)(walker, ham, padded_ctx, zero_padded_trial)

    np.testing.assert_allclose(truncated_overlap, padded_overlap, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(truncated_fb, padded_fb, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(truncated_energy, padded_energy, rtol=2.0e-12, atol=2.0e-12)


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
        n_mode_chunks=1,
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
    mode_ops = make_cisd_mode_meas_ops(sys, mixed_precision=True, n_mode_chunks=2)
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


def test_n_mode_chunks_validation_and_rank_cap():
    _, mode_trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(899),
        norb=mode_trial.norb,
        n_chol=5,
        basis="restricted",
    )

    with pytest.raises(ValueError, match="n_mode_chunks must be positive"):
        build_mode_meas_ctx(ham, mode_trial, n_mode_chunks=0)

    mode_ctx = build_mode_meas_ctx(ham, mode_trial, n_mode_chunks=100)
    assert mode_ctx.n_mode_chunks == mode_trial.mode_rank


def test_mode_energy_is_independent_of_n_mode_chunks_in_double_precision():
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

    for n_mode_chunks in (1, 2, 4, 32):
        mode_ctx = build_mode_meas_ctx(
            ham,
            mode_trial,
            cfg=cfg,
            n_mode_chunks=n_mode_chunks,
        )
        candidate = jax.jit(mode_energy_kernel)(walker, ham, mode_ctx, mode_trial)
        np.testing.assert_allclose(candidate, reference, rtol=2.0e-12, atol=2.0e-12)


def test_pair_sampling_config_validation_and_factory_opt_in():
    with pytest.raises(ValueError, match="chol_head_size must be nonnegative"):
        CisdModePairSamplingCfg(chol_head_size=-1, pair_sample_size=8)
    with pytest.raises(ValueError, match="pair_sample_size must be positive"):
        CisdModePairSamplingCfg(chol_head_size=0, pair_sample_size=0)

    _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    sys = System(
        norb=trial.norb,
        nelec=(trial.nocc_full, trial.nocc_full),
        walker_kind="restricted",
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(919),
        norb=trial.norb,
        n_chol=5,
        basis="restricted",
    )
    sampling = CisdModePairSamplingCfg(chol_head_size=2, pair_sample_size=16)
    deterministic_ops = make_cisd_mode_meas_ops(sys, mixed_precision=False)
    sampled_ops = make_cisd_mode_meas_ops(
        sys,
        mixed_precision=False,
        energy_sampling=sampling,
    )
    assert deterministic_ops.block_energy is None
    assert sampled_ops.block_energy is pair_sampled_block_energy

    sampled_ctx = sampled_ops.build_meas_ctx(ham, trial)
    assert sampled_ctx.energy_sampling == sampling
    assert sampled_ctx.chol_tail_prob.shape == (3,)
    np.testing.assert_allclose(jnp.sum(sampled_ctx.chol_tail_prob), 1.0, atol=1.0e-14)
    assert bool(jnp.all(sampled_ctx.chol_tail_prob > 0.0))

    invalid_sampling = CisdModePairSamplingCfg(chol_head_size=6, pair_sample_size=16)
    with pytest.raises(ValueError, match="must not exceed"):
        build_mode_meas_ctx(ham, trial, energy_sampling=invalid_sampling)


def test_sampled_pair_terms_gather_inside_chunks_matches_full_pair_matrix():
    _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(923),
        norb=trial.norb,
        n_chol=5,
        basis="restricted",
    )
    ctx = build_mode_meas_ctx(
        ham,
        trial,
        cfg=CisdMeasCfg(memory_mode="high"),
        n_mode_chunks=2,
    )
    walkers = jnp.stack(
        [
            testing.make_restricted_walker_near_ref(
                jax.random.PRNGKey(seed),
                trial.norb,
                trial.nocc_full,
                mix=0.25,
            )
            for seed in (927, 929, 937)
        ]
    )
    common = jax.vmap(
        _cisd_mode_energy_common,
        in_axes=(0, None, None, None),
    )(walkers, ham, ctx, trial)
    all_terms = _cisd_mode_chol_terms_for_walkers(
        common,
        ham.chol,
        ctx.rot_chol,
        ctx.lci1,
        ctx,
        trial,
    )
    sample_walker = jnp.asarray([2, 0, 1, 2, 1, 0, 2], dtype=jnp.int32)
    sample_chol = jnp.asarray([4, 1, 3, 0, 2, 4, 2], dtype=jnp.int32)
    expected = all_terms[sample_walker, sample_chol]

    candidate = jax.jit(
        lambda walker_indices, chol_indices: _cisd_mode_chol_pair_terms(
            common,
            walker_indices,
            chol_indices,
            ham,
            ctx,
            trial,
            n_chunks=3,
        )
    )(sample_walker, sample_chol)

    np.testing.assert_allclose(candidate, expected, rtol=2.0e-12, atol=2.0e-12)


def test_pair_sampled_block_energy_full_head_matches_weighted_deterministic_energy():
    _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(929),
        norb=trial.norb,
        n_chol=5,
        basis="restricted",
    )
    sampling = CisdModePairSamplingCfg(chol_head_size=5, pair_sample_size=8)
    ctx = build_mode_meas_ctx(
        ham,
        trial,
        cfg=CisdMeasCfg(memory_mode="high"),
        n_mode_chunks=2,
        energy_sampling=sampling,
    )
    walkers = jnp.stack(
        [
            testing.make_restricted_walker_near_ref(
                jax.random.PRNGKey(seed),
                trial.norb,
                trial.nocc_full,
                mix=0.25,
            )
            for seed in (937, 941, 947)
        ]
    )
    weights = jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float64)
    exact = jax.vmap(mode_energy_kernel, in_axes=(0, None, None, None))(
        walkers,
        ham,
        ctx,
        trial,
    )
    expected = jnp.sum(weights * jnp.real(exact)) / jnp.sum(weights)
    candidate = jax.jit(pair_sampled_block_energy, static_argnums=4)(
        walkers,
        weights,
        jnp.ones_like(weights, dtype=jnp.complex128),
        jax.random.PRNGKey(953),
        2,
        ham,
        ctx,
        trial,
    )
    assert ctx.chol_tail_prob.shape == (0,)
    np.testing.assert_allclose(candidate, expected, rtol=2.0e-12, atol=2.0e-12)


def test_pair_sampled_tail_matches_exact_energy_within_analytic_sampling_error():
    _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(967),
        norb=trial.norb,
        n_chol=5,
        basis="restricted",
    )
    pair_sample_size = 8192
    sampling = CisdModePairSamplingCfg(
        chol_head_size=2,
        pair_sample_size=pair_sample_size,
    )
    ctx = build_mode_meas_ctx(
        ham,
        trial,
        cfg=CisdMeasCfg(memory_mode="high"),
        n_mode_chunks=2,
        energy_sampling=sampling,
    )
    walkers = jnp.stack(
        [
            testing.make_restricted_walker_near_ref(
                jax.random.PRNGKey(seed),
                trial.norb,
                trial.nocc_full,
                mix=0.25,
            )
            for seed in (971, 977, 983)
        ]
    )
    weights = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    norm_weights = weights / jnp.sum(weights)
    common = jax.vmap(
        _cisd_mode_energy_common,
        in_axes=(0, None, None, None),
    )(walkers, ham, ctx, trial)
    all_terms = _cisd_mode_chol_terms_for_walkers(
        common,
        ham.chol,
        ctx.rot_chol,
        ctx.lci1,
        ctx,
        trial,
    )
    exact_per_walker = jnp.real(common.base + jnp.sum(all_terms, axis=1))
    exact_block = jnp.sum(norm_weights * exact_per_walker)

    tail_terms = jnp.real(all_terms[:, sampling.chol_head_size :])
    importance_values = tail_terms / ctx.chol_tail_prob[None, :]
    joint_prob = norm_weights[:, None] * ctx.chol_tail_prob[None, :]
    tail_mean = jnp.sum(joint_prob * importance_values)
    tail_variance = jnp.sum(joint_prob * (importance_values - tail_mean) ** 2)
    standard_error = jnp.sqrt(tail_variance / pair_sample_size)

    candidate = jax.jit(pair_sampled_block_energy, static_argnums=4)(
        walkers,
        weights,
        jnp.ones_like(weights, dtype=jnp.complex128),
        jax.random.PRNGKey(991),
        2,
        ham,
        ctx,
        trial,
    )
    np.testing.assert_allclose(
        candidate,
        exact_block,
        rtol=0.0,
        atol=float(6.0 * standard_error + 1.0e-12),
    )
