from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot import driver, testing
from trot.core.ops import BlockEnergyEstimate, d_energy_sampling_noise, k_energy, k_force_bias
from trot.core.system import System
from trot.meas.ucisd import UcisdMeasCfg
from trot.meas.ucisd_k import (
    _ucisd_k_energy_common,
    build_meas_ctx as build_k_meas_ctx,
    energy_kernel_rw_rh as k_energy_kernel,
    force_bias_kernel_rw_rh as k_force_bias_kernel,
)
from trot.meas.ucisd_k_modes import (
    UcisdKModePairSamplingCfg,
    UcisdKModePairTuningCfg,
    UcisdKModePopulationStats,
    _k_mode_apply_realimag,
    _k_mode_quadratic_batched_realimag,
    _ucisd_k_mode_chol_pair_terms,
    _ucisd_k_mode_chol_terms_for_walkers,
    average_ucisd_k_mode_population_statistics,
    build_meas_ctx as build_k_mode_meas_ctx,
    energy_kernel_rw_rh as k_mode_energy_kernel,
    force_bias_kernel_rw_rh as k_mode_force_bias_kernel,
    get_ucisd_k_mode_meas_cfg,
    make_ucisd_k_mode_meas_ops,
    pair_sampled_block_energy,
    retune_ucisd_k_mode_pair_sampling,
    select_ucisd_k_mode_pair_sampling,
    stream_ucisd_k_mode_population_statistics,
)
from trot.prop.afqmc import make_prop_ops
from trot.prop.blocks import block
from trot.prop.types import PropState, QmcParams
from trot.trial.ucisd_k import UcisdKTrial, overlap_r as k_overlap_r
from trot.trial import ucisd_k_modes as ucisd_k_modes_module
from trot.trial.ucisd_k_modes import (
    UcisdKModeTrial,
    factorize_ucisd_k_blocks,
    k_mode_apply,
    k_mode_quadratic,
    make_ucisd_k_mode_trial_data,
    make_ucisd_k_mode_trial_ops,
    mode_projections,
    overlap_r as k_mode_overlap_r,
)


def _same_spin_tensor(
    rng: np.random.Generator,
    nocc: int,
    nvir: int,
) -> np.ndarray:
    raw = rng.standard_normal((nocc, nvir, nocc, nvir))
    return 0.25 * (
        raw - raw.transpose(2, 1, 0, 3) - raw.transpose(0, 3, 2, 1) + raw.transpose(2, 3, 0, 1)
    )


def _make_trials(
    *,
    seed: int = 1801,
    norb: int = 6,
    noa: int = 3,
    nob: int = 2,
    rank: int | None = None,
    mode_dtype=jnp.float64,
) -> tuple[UcisdKTrial, UcisdKModeTrial, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    nva = norb - noa
    nvb = norb - nob
    rotation, _ = np.linalg.qr(np.eye(norb) + 0.15 * rng.standard_normal((norb, norb)))
    c1a = 0.03 * rng.standard_normal((noa, nva))
    c1b = 0.03 * rng.standard_normal((nob, nvb))
    c2aa = 0.02 * _same_spin_tensor(rng, noa, nva)
    c2ab = 0.02 * rng.standard_normal((noa, nva, nob, nvb))
    c2bb = 0.02 * _same_spin_tensor(rng, nob, nvb)
    aa = c2aa.reshape(noa * nva, noa * nva)
    ab = c2ab.reshape(noa * nva, nob * nvb)
    bb = c2bb.reshape(nob * nvb, nob * nvb)
    kernel = np.block([[aa, ab], [ab.T, bb]])
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if rank is None:
        rank = int(kernel.shape[0])
    eigenvalues = eigenvalues[:rank]
    modes = eigenvectors[:, :rank].T
    reconstructed = (modes.T * eigenvalues) @ modes

    exact = UcisdKTrial(
        mo_coeff_a=jnp.eye(norb, dtype=jnp.float64),
        mo_coeff_b=jnp.asarray(rotation, dtype=jnp.float64),
        c1a=jnp.asarray(c1a, dtype=jnp.float64),
        c1b=jnp.asarray(c1b, dtype=jnp.float64),
        k=jnp.asarray(reconstructed, dtype=jnp.float64),
    )
    mode = UcisdKModeTrial(
        mo_coeff_a=exact.mo_coeff_a,
        mo_coeff_b=exact.mo_coeff_b,
        c1a=exact.c1a,
        c1b=exact.c1b,
        eigenvalues=jnp.asarray(eigenvalues, dtype=jnp.float64),
        modes=jnp.asarray(modes, dtype=mode_dtype),
    )
    return exact, mode, (c2aa, c2ab, c2bb)


def _double_cfg() -> UcisdMeasCfg:
    return UcisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )


def _walker(trial: UcisdKModeTrial, seed: int = 1811) -> jax.Array:
    return testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(seed),
        trial.norb,
        max(trial.nocc),
        mix=0.15,
    )


@pytest.mark.parametrize("rank", [None, 5, 0])
def test_k_mode_helpers_match_explicit_reconstructed_kernel(rank):
    exact, mode, _ = _make_trials(rank=rank)
    rng = np.random.default_rng(1817)
    shape_a = (mode.nocc[0], mode.nvir[0])
    shape_b = (mode.nocc[1], mode.nvir[1])
    matrix_a_np = rng.standard_normal(shape_a) + 1.0j * rng.standard_normal(shape_a)
    matrix_b_np = rng.standard_normal(shape_b) + 1.0j * rng.standard_normal(shape_b)
    matrix_a = jnp.asarray(matrix_a_np)
    matrix_b = jnp.asarray(matrix_b_np)

    projections = mode_projections(mode, matrix_a, matrix_b)
    applied_a, applied_b = k_mode_apply(
        mode,
        matrix_a,
        matrix_b,
        projections,
    )
    quadratic = k_mode_quadratic(
        mode,
        matrix_a,
        matrix_b,
        projections,
    )
    vector = np.concatenate((matrix_a_np.reshape(-1), matrix_b_np.reshape(-1)))
    expected_applied = np.asarray(exact.k) @ vector
    expected_quadratic = 0.5 * vector @ np.asarray(exact.k) @ vector
    da, _ = mode.pair_dim

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
    np.testing.assert_allclose(
        quadratic,
        expected_quadratic,
        rtol=3.0e-12,
        atol=3.0e-12,
    )


@pytest.mark.parametrize("rank", [None, 5, 0])
def test_k_modes_match_reconstructed_k_overlap_force_bias_and_energy(rank):
    exact, mode, _ = _make_trials(seed=1823, rank=rank)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1829),
        mode.norb,
        n_chol=7,
        basis="restricted",
    )
    exact_ctx = build_k_meas_ctx(ham, exact, cfg=_double_cfg())
    mode_ctx = build_k_mode_meas_ctx(
        ham,
        mode,
        cfg=_double_cfg(),
        n_mode_chunks=3,
    )
    walker = _walker(mode, 1831)

    exact_overlap = k_overlap_r(walker, exact)
    mode_overlap = jax.jit(k_mode_overlap_r)(walker, mode)
    exact_fb = k_force_bias_kernel(walker, ham, exact_ctx, exact)
    mode_fb = jax.jit(k_mode_force_bias_kernel)(walker, ham, mode_ctx, mode)
    exact_energy = k_energy_kernel(walker, ham, exact_ctx, exact)
    mode_energy = jax.jit(k_mode_energy_kernel)(walker, ham, mode_ctx, mode)

    np.testing.assert_allclose(mode_overlap, exact_overlap, rtol=4.0e-12, atol=4.0e-12)
    np.testing.assert_allclose(mode_fb, exact_fb, rtol=4.0e-12, atol=4.0e-12)
    np.testing.assert_allclose(mode_energy, exact_energy, rtol=4.0e-12, atol=4.0e-12)


def test_k_mode_batched_quadratic_and_chunking_match_exact_k():
    exact, mode, _ = _make_trials(seed=1847, rank=7)
    rng = np.random.default_rng(1861)
    shape_a = (5, mode.nocc[0], mode.nvir[0])
    shape_b = (5, mode.nocc[1], mode.nvir[1])
    matrices_a = jnp.asarray(rng.standard_normal(shape_a) + 1.0j * rng.standard_normal(shape_a))
    matrices_b = jnp.asarray(rng.standard_normal(shape_b) + 1.0j * rng.standard_normal(shape_b))
    cfg = _double_cfg()

    one_chunk = _k_mode_quadratic_batched_realimag(
        mode,
        matrices_a,
        matrices_b,
        cfg,
        1,
    )
    many_chunks = _k_mode_quadratic_batched_realimag(
        mode,
        matrices_a,
        matrices_b,
        cfg,
        4,
    )
    vectors = np.concatenate(
        (
            np.asarray(matrices_a).reshape(5, -1),
            np.asarray(matrices_b).reshape(5, -1),
        ),
        axis=1,
    )
    expected = 0.5 * np.einsum("sp,pq,sq->s", vectors, np.asarray(exact.k), vectors)

    np.testing.assert_allclose(one_chunk, expected, rtol=4.0e-12, atol=4.0e-12)
    np.testing.assert_allclose(many_chunks, expected, rtol=4.0e-12, atol=4.0e-12)


def test_dense_and_lanczos_factorizations_reconstruct_same_truncated_kernel():
    _, _, blocks = _make_trials(seed=1867)
    c2aa, c2ab, c2bb = blocks
    aa = c2aa.reshape(c2aa.shape[0] * c2aa.shape[1], -1)
    ab = c2ab.reshape(c2ab.shape[0] * c2ab.shape[1], -1)
    bb = c2bb.reshape(c2bb.shape[0] * c2bb.shape[1], -1)
    kernel = np.block([[aa, ab], [ab.T, bb]])
    magnitudes = np.sort(np.abs(np.linalg.eigvalsh(kernel)))[::-1]
    threshold = float(0.5 * (magnitudes[3] + magnitudes[4]))

    dense = factorize_ucisd_k_blocks(
        c2aa,
        c2ab,
        c2bb,
        threshold=threshold,
        solver="dense",
    )
    lanczos = factorize_ucisd_k_blocks(
        c2aa,
        c2ab,
        c2bb,
        threshold=threshold,
        solver="lanczos",
        lanczos_initial_rank=5,
        lanczos_tol=1.0e-12,
    )
    dense_kernel = (dense.modes.T * dense.eigenvalues) @ dense.modes
    lanczos_kernel = (lanczos.modes.T * lanczos.eigenvalues) @ lanczos.modes

    assert dense.rank == 4
    assert lanczos.rank == dense.rank
    assert lanczos.solver == "lanczos"
    np.testing.assert_allclose(lanczos_kernel, dense_kernel, rtol=2.0e-10, atol=2.0e-12)
    np.testing.assert_allclose(
        lanczos.discarded_norm_fraction,
        dense.discarded_norm_fraction,
        rtol=2.0e-10,
        atol=2.0e-12,
    )


def test_discarded_norm_target_selects_same_dense_and_lanczos_rank():
    _, _, blocks = _make_trials(seed=1869)
    c2aa, c2ab, c2bb = blocks
    aa = c2aa.reshape(c2aa.shape[0] * c2aa.shape[1], -1)
    ab = c2ab.reshape(c2ab.shape[0] * c2ab.shape[1], -1)
    bb = c2bb.reshape(c2bb.shape[0] * c2bb.shape[1], -1)
    kernel = np.block([[aa, ab], [ab.T, bb]])
    eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(kernel)))[::-1]
    full_norm_sq = float(np.vdot(eigenvalues, eigenvalues).real)
    discarded_at_three = np.sqrt(np.sum(eigenvalues[3:] ** 2) / full_norm_sq)
    discarded_at_four = np.sqrt(np.sum(eigenvalues[4:] ** 2) / full_norm_sq)
    target = float(0.5 * (discarded_at_three + discarded_at_four))

    dense = factorize_ucisd_k_blocks(
        c2aa,
        c2ab,
        c2bb,
        threshold=None,
        discarded_norm_target=target,
        solver="dense",
    )
    lanczos = factorize_ucisd_k_blocks(
        c2aa,
        c2ab,
        c2bb,
        threshold=None,
        discarded_norm_target=target,
        solver="lanczos",
        lanczos_initial_rank=5,
        lanczos_tol=1.0e-12,
    )

    assert dense.rank == 4
    assert dense.natural_rank == 4
    assert dense.discarded_norm_fraction <= target
    assert lanczos.rank == dense.rank
    assert lanczos.natural_rank == dense.natural_rank
    np.testing.assert_allclose(
        lanczos.discarded_norm_fraction,
        dense.discarded_norm_fraction,
        rtol=2.0e-10,
        atol=2.0e-12,
    )


def test_auto_solver_respects_available_host_memory(monkeypatch):
    _, _, blocks = _make_trials(seed=1868)
    c2aa, c2ab, c2bb = blocks

    monkeypatch.setattr(
        ucisd_k_modes_module,
        "format_dense_memory_selection",
        lambda dimension: (False, f"mock insufficient memory for {dimension}"),
    )
    lanczos = factorize_ucisd_k_blocks(
        c2aa,
        c2ab,
        c2bb,
        threshold=None,
        discarded_norm_target=0.5,
        solver="auto",
        dense_max_dim=1,
        lanczos_initial_rank=5,
        lanczos_tol=1.0e-12,
    )

    monkeypatch.setattr(
        ucisd_k_modes_module,
        "format_dense_memory_selection",
        lambda dimension: (True, f"mock sufficient memory for {dimension}"),
    )
    dense = factorize_ucisd_k_blocks(
        c2aa,
        c2ab,
        c2bb,
        threshold=None,
        discarded_norm_target=0.5,
        solver="auto",
        dense_max_dim=1,
    )

    assert lanczos.solver == "lanczos"
    assert dense.solver == "dense"


def test_auto_solver_retries_lanczos_after_dense_memory_error(monkeypatch):
    _, _, blocks = _make_trials(seed=1869)
    c2aa, c2ab, c2bb = blocks
    dense_attempts = 0

    monkeypatch.setattr(
        ucisd_k_modes_module,
        "format_dense_memory_selection",
        lambda dimension: (True, f"mock sufficient memory for {dimension}"),
    )

    def fail_dense(*args, **kwargs):
        nonlocal dense_attempts
        dense_attempts += 1
        raise MemoryError("mock dense allocation failure")

    monkeypatch.setattr(ucisd_k_modes_module.np.linalg, "eigh", fail_dense)
    factorization = factorize_ucisd_k_blocks(
        c2aa,
        c2ab,
        c2bb,
        threshold=None,
        discarded_norm_target=0.5,
        solver="auto",
        lanczos_initial_rank=5,
        lanczos_tol=1.0e-12,
    )

    assert dense_attempts == 1
    assert factorization.solver == "lanczos"


def test_minimum_rank_retains_extra_modes_without_changing_natural_rank():
    _, _, blocks = _make_trials(seed=1870)
    c2aa, c2ab, c2bb = blocks
    natural = factorize_ucisd_k_blocks(
        c2aa,
        c2ab,
        c2bb,
        threshold=None,
        discarded_norm_target=0.5,
        solver="dense",
    )
    requested_rank = natural.rank + 2
    extended = factorize_ucisd_k_blocks(
        c2aa,
        c2ab,
        c2bb,
        threshold=None,
        discarded_norm_target=0.5,
        minimum_rank=requested_rank,
        solver="dense",
    )

    assert extended.natural_rank == natural.rank
    assert extended.rank == requested_rank
    assert extended.discarded_norm_fraction < natural.discarded_norm_fraction


def test_k_mode_loader_mixed_precision_and_measurements():
    exact, mode_double, _ = _make_trials(seed=1871)
    sys = System(mode_double.norb, mode_double.nocc, walker_kind="restricted")
    mixed = make_ucisd_k_mode_trial_data(
        {
            "mo_coeff_a": mode_double.mo_coeff_a,
            "mo_coeff_b": mode_double.mo_coeff_b,
            "c1a": mode_double.c1a,
            "c1b": mode_double.c1b,
            "eigenvalues": mode_double.eigenvalues,
            "eigenvectors": mode_double.modes.T,
        },
        sys,
        mixed_precision=True,
    )
    assert mixed.eigenvalues.dtype == jnp.float64
    assert mixed.modes.dtype == jnp.float32
    leaves, treedef = jax.tree_util.tree_flatten(mixed)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.mode_rank == mixed.mode_rank

    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1873),
        mixed.norb,
        n_chol=7,
        basis="restricted",
    )
    exact_ctx = build_k_meas_ctx(ham, exact, cfg=_double_cfg())
    mixed_ops = make_ucisd_k_mode_meas_ops(
        sys,
        mixed_precision=True,
        n_mode_chunks=3,
    )
    mixed_ctx = mixed_ops.build_meas_ctx(ham, mixed)
    cfg = get_ucisd_k_mode_meas_cfg(mixed_ops)
    assert cfg is not None
    assert cfg.mixed_real_dtype == jnp.float32
    assert cfg.mixed_complex_dtype == jnp.complex64

    walker = _walker(mixed, 1877)
    exact_fb = k_force_bias_kernel(walker, ham, exact_ctx, exact)
    mixed_fb = mixed_ops.require_kernel(k_force_bias)(walker, ham, mixed_ctx, mixed)
    exact_energy = k_energy_kernel(walker, ham, exact_ctx, exact)
    mixed_energy = mixed_ops.require_kernel(k_energy)(walker, ham, mixed_ctx, mixed)
    fb_error = float(jnp.linalg.norm(mixed_fb - exact_fb) / jnp.linalg.norm(exact_fb))
    energy_error = float(jnp.abs(mixed_energy - exact_energy))
    assert fb_error < 2.0e-5
    assert energy_error < 2.0e-4


def test_k_mode_apply_mixed_precision_matches_reconstructed_kernel():
    exact, mode, _ = _make_trials(seed=1889, rank=5, mode_dtype=jnp.float32)
    rng = np.random.default_rng(1901)
    matrix_a = jnp.asarray(
        rng.standard_normal((mode.nocc[0], mode.nvir[0]))
        + 1.0j * rng.standard_normal((mode.nocc[0], mode.nvir[0]))
    )
    matrix_b = jnp.asarray(
        rng.standard_normal((mode.nocc[1], mode.nvir[1]))
        + 1.0j * rng.standard_normal((mode.nocc[1], mode.nvir[1]))
    )
    cfg = UcisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float32,
        mixed_complex_dtype=jnp.complex64,
        mixed_real_dtype_testing=jnp.float32,
        mixed_complex_dtype_testing=jnp.complex64,
    )
    applied_a, applied_b = _k_mode_apply_realimag(
        mode,
        matrix_a,
        matrix_b,
        cfg,
    )
    vector = np.concatenate((np.asarray(matrix_a).reshape(-1), np.asarray(matrix_b).reshape(-1)))
    expected = np.asarray(exact.k, dtype=np.float32) @ vector.astype(np.complex64)
    da, _ = mode.pair_dim
    np.testing.assert_allclose(
        applied_a,
        expected[:da].reshape(matrix_a.shape),
        rtol=3.0e-6,
        atol=3.0e-6,
    )
    np.testing.assert_allclose(
        applied_b,
        expected[da:].reshape(matrix_b.shape),
        rtol=3.0e-6,
        atol=3.0e-6,
    )


def test_k_mode_sampled_pair_terms_match_full_walker_cholesky_table():
    _, trial, _ = _make_trials(seed=1903, rank=5)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1905),
        trial.norb,
        n_chol=7,
        basis="restricted",
    )
    ctx = build_k_mode_meas_ctx(
        ham,
        trial,
        cfg=_double_cfg(),
        n_mode_chunks=3,
    )
    walkers = jnp.stack([_walker(trial, seed) for seed in (1909, 1911, 1915)])
    common = jax.vmap(
        lambda walker: _ucisd_k_energy_common(
            walker,
            ham,
            ctx,
            trial,
            _k_mode_apply_realimag,
        )
    )(walkers)
    all_terms = _ucisd_k_mode_chol_terms_for_walkers(
        common,
        ham,
        ctx,
        trial,
        n_chunks=2,
    )
    sample_walker = jnp.asarray([2, 0, 1, 2, 1, 0, 2], dtype=jnp.int32)
    sample_chol = jnp.asarray([6, 1, 4, 0, 3, 5, 2], dtype=jnp.int32)
    expected = all_terms[sample_walker, sample_chol]

    candidate = jax.jit(
        lambda walker_indices, chol_indices: _ucisd_k_mode_chol_pair_terms(
            common,
            walker_indices,
            chol_indices,
            ham,
            ctx,
            trial,
            n_chunks=3,
        )
    )(sample_walker, sample_chol)

    np.testing.assert_allclose(candidate, expected, rtol=3.0e-12, atol=3.0e-12)


def test_k_mode_pair_sampled_full_head_matches_deterministic_block_energy():
    _, trial, _ = _make_trials(seed=1917, rank=5)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1919),
        trial.norb,
        n_chol=7,
        basis="restricted",
    )
    sampling = UcisdKModePairSamplingCfg(
        chol_head_size=7,
        pair_sample_size=8,
        head_chol_batch_size=2,
        track_half_sample_diagnostic=True,
    )
    ops = make_ucisd_k_mode_meas_ops(
        System(trial.norb, trial.nocc, walker_kind="restricted"),
        mixed_precision=False,
        n_mode_chunks=3,
        energy_sampling=sampling,
    )
    ctx = ops.build_meas_ctx(ham, trial)
    walkers = jnp.stack([_walker(trial, seed) for seed in (1921, 1923, 1927)])
    weights = jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float64)
    exact = jax.vmap(k_mode_energy_kernel, in_axes=(0, None, None, None))(
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
        jax.random.PRNGKey(1931),
        2,
        ham,
        ctx,
        trial,
        jnp.asarray(0.0),
        jnp.asarray(20.0),
    )

    assert ops.block_energy is pair_sampled_block_energy
    assert ctx.energy_sampling == sampling
    assert ctx.reference_chol_scores.shape == (7,)
    assert ctx.chol_tail_prob.shape == (0,)
    assert isinstance(candidate, BlockEnergyEstimate)
    np.testing.assert_allclose(candidate.energy, expected, rtol=3.0e-12, atol=3.0e-12)
    np.testing.assert_allclose(
        candidate.diagnostics[d_energy_sampling_noise],
        0.0,
        rtol=0.0,
        atol=0.0,
    )


def test_k_mode_one_electron_pair_sampled_energy_matches_deterministic_energy():
    trial = UcisdKModeTrial(
        mo_coeff_a=jnp.eye(2, dtype=jnp.float64),
        mo_coeff_b=jnp.eye(2, dtype=jnp.float64),
        c1a=jnp.asarray([[0.04]], dtype=jnp.float64),
        c1b=jnp.zeros((0, 2), dtype=jnp.float64),
        eigenvalues=jnp.asarray([0.0], dtype=jnp.float64),
        modes=jnp.asarray([[1.0]], dtype=jnp.float64),
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1932),
        norb=2,
        n_chol=3,
        basis="restricted",
    )
    sampling = UcisdKModePairSamplingCfg(
        chol_head_size=3,
        pair_sample_size=4,
        head_chol_batch_size=1,
        track_half_sample_diagnostic=True,
    )
    ops = make_ucisd_k_mode_meas_ops(
        System(trial.norb, trial.nocc, walker_kind="restricted"),
        mixed_precision=False,
        energy_sampling=sampling,
    )
    ctx = ops.build_meas_ctx(ham, trial)
    walkers = jnp.asarray(
        [
            [[1.0 + 0.02j], [0.1 - 0.03j]],
            [[0.98 - 0.01j], [0.08 + 0.02j]],
        ],
        dtype=jnp.complex128,
    )
    weights = jnp.asarray([1.0, 0.8], dtype=jnp.float64)
    exact = jax.vmap(k_mode_energy_kernel, in_axes=(0, None, None, None))(
        walkers,
        ham,
        ctx,
        trial,
    )
    sampled = jax.jit(pair_sampled_block_energy, static_argnums=4)(
        walkers,
        weights,
        jnp.ones_like(weights, dtype=jnp.complex128),
        jax.random.PRNGKey(1934),
        1,
        ham,
        ctx,
        trial,
        jnp.asarray(0.0),
        jnp.asarray(20.0),
    )

    np.testing.assert_allclose(
        sampled.energy,
        jnp.sum(weights * jnp.real(exact)) / jnp.sum(weights),
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    assert bool(jnp.isfinite(sampled.energy))


def test_k_mode_pair_sampled_tail_matches_the_drawn_importance_estimator():
    _, trial, _ = _make_trials(seed=1933, rank=5)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1935),
        trial.norb,
        n_chol=7,
        basis="restricted",
    )
    pair_sample_size = 128
    sampling = UcisdKModePairSamplingCfg(
        chol_head_size=2,
        pair_sample_size=pair_sample_size,
        rank_head_by_guide=True,
        tail_probability_uniform_mix=0.1,
        track_half_sample_diagnostic=True,
    )
    ctx = build_k_mode_meas_ctx(
        ham,
        trial,
        cfg=_double_cfg(),
        n_mode_chunks=3,
        energy_sampling=sampling,
    )
    walkers = jnp.stack([_walker(trial, seed) for seed in (1937, 1941, 1943)])
    weights = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    norm_weights = weights / jnp.sum(weights)
    common = jax.vmap(
        lambda walker: _ucisd_k_energy_common(
            walker,
            ham,
            ctx,
            trial,
            _k_mode_apply_realimag,
        )
    )(walkers)
    all_terms = jnp.real(
        _ucisd_k_mode_chol_terms_for_walkers(
            common,
            ham,
            ctx,
            trial,
            n_chunks=2,
        )
    )
    head_energy = jnp.real(common.base) + jnp.sum(
        all_terms[:, ctx.chol_head_indices],
        axis=1,
    )

    rng_key = jax.random.PRNGKey(1945)
    key_walker, key_chol = jax.random.split(rng_key)
    sample_walker = jax.random.choice(
        key_walker,
        weights.shape[0],
        shape=(pair_sample_size,),
        replace=True,
        p=norm_weights,
    )
    sample_chol_rel = jax.random.choice(
        key_chol,
        ctx.chol_tail_prob.shape[0],
        shape=(pair_sample_size,),
        replace=True,
        p=ctx.chol_tail_prob,
    )
    sample_values = (
        all_terms[sample_walker, ctx.chol_tail_indices[sample_chol_rel]]
        / ctx.chol_tail_prob[sample_chol_rel]
    )
    expected_energy = jnp.sum(norm_weights * head_energy) + jnp.mean(sample_values)
    half_size = pair_sample_size // 2
    expected_diagnostic = 0.5 * (
        jnp.mean(sample_values[:half_size]) - jnp.mean(sample_values[half_size:])
    )

    candidate = jax.jit(pair_sampled_block_energy, static_argnums=4)(
        walkers,
        weights,
        jnp.ones_like(weights, dtype=jnp.complex128),
        rng_key,
        2,
        ham,
        ctx,
        trial,
        jnp.asarray(0.0),
        jnp.asarray(20.0),
    )

    assert isinstance(candidate, BlockEnergyEstimate)
    np.testing.assert_allclose(candidate.energy, expected_energy, rtol=3.0e-12, atol=3.0e-12)
    np.testing.assert_allclose(
        candidate.diagnostics[d_energy_sampling_noise],
        expected_diagnostic,
        rtol=3.0e-12,
        atol=3.0e-12,
    )


def test_k_mode_streaming_population_statistics_match_full_term_table():
    _, trial, _ = _make_trials(seed=1947, rank=5)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1949),
        trial.norb,
        n_chol=7,
        basis="restricted",
    )
    ctx = build_k_mode_meas_ctx(
        ham,
        trial,
        cfg=_double_cfg(),
        n_mode_chunks=3,
    )
    walkers = jnp.stack([_walker(trial, seed) for seed in (1951, 1955, 1957)])
    weights = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    norm_weights = np.asarray(weights / jnp.sum(weights))
    common = jax.vmap(
        lambda walker: _ucisd_k_energy_common(
            walker,
            ham,
            ctx,
            trial,
            _k_mode_apply_realimag,
        )
    )(walkers)
    terms = np.real(
        np.asarray(
            _ucisd_k_mode_chol_terms_for_walkers(
                common,
                ham,
                ctx,
                trial,
            )
        )
    )
    base = np.real(np.asarray(common.base))

    stats = stream_ucisd_k_mode_population_statistics(
        walkers,
        weights,
        ham,
        ctx,
        trial,
        n_walker_chunks=2,
        chol_batch_size=3,
    )
    expected_means = np.sum(norm_weights[:, None] * terms, axis=0)
    expected_seconds = np.sum(norm_weights[:, None] * terms**2, axis=0)
    expected_local = base + np.sum(terms, axis=1)
    np.testing.assert_allclose(stats.term_means, expected_means, rtol=3.0e-12, atol=3.0e-12)
    np.testing.assert_allclose(
        stats.term_second_moments,
        expected_seconds,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(stats.rms_scores, np.sqrt(expected_seconds), rtol=3.0e-12)
    np.testing.assert_allclose(stats.local_energies, expected_local, rtol=3.0e-12, atol=3.0e-12)
    np.testing.assert_allclose(
        stats.exact_block_energy_ha,
        np.sum(norm_weights * expected_local),
        rtol=3.0e-12,
        atol=3.0e-12,
    )


def test_k_mode_tuner_uses_shared_policy_and_returns_ucisd_sampling_config():
    stats = UcisdKModePopulationStats(
        term_means=np.zeros(4, dtype=np.float64),
        term_second_moments=np.asarray([16.0, 4.0, 1.0, 0.25]),
        rms_scores=np.asarray([4.0, 2.0, 1.0, 0.5]),
        local_energies=np.zeros(2, dtype=np.float64),
        exact_block_energy_ha=0.0,
        independent_population_std_ha=0.16,
        wall_seconds=0.0,
    )
    averaged = average_ucisd_k_mode_population_statistics([stats, stats])
    selected = select_ucisd_k_mode_pair_sampling(
        averaged,
        UcisdKModePairTuningCfg(
            target_tail_std_fraction=0.5,
            safety_factor=1.0,
            candidate_sample_sizes=(64, 256),
            maximum_head_fraction=0.75,
            production_head_chol_batch_size=2,
            settling_blocks=0,
        ),
        n_walkers=10,
        calibration_std_ha=0.16,
        calibration_source="late equilibration block standard deviation",
    )

    assert isinstance(selected.sampling, UcisdKModePairSamplingCfg)
    assert selected.sampling.chol_head_size == 3
    assert selected.sampling.pair_sample_size == 64
    assert selected.sampling.head_chol_batch_size == 2
    assert selected.estimated_pair_evaluations == 94
    np.testing.assert_allclose(selected.estimated_tail_std_ha, 0.0625)
    np.testing.assert_allclose(selected.target_tail_std_ha, 0.08)


def test_k_mode_retune_collects_temporally_separated_populations():
    _, trial, _ = _make_trials(seed=1959, norb=5, noa=2, nob=2, rank=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1961),
        trial.norb,
        n_chol=4,
        basis="restricted",
    )
    equil_sampling = UcisdKModePairSamplingCfg(
        chol_head_size=2,
        pair_sample_size=16,
        rank_head_by_guide=True,
    )
    ctx = build_k_mode_meas_ctx(
        ham,
        trial,
        cfg=_double_cfg(),
        n_mode_chunks=2,
        energy_sampling=equil_sampling,
    )
    walkers = jnp.stack([_walker(trial, seed) for seed in (1963, 1965)])
    state = PropState(
        walkers=walkers,
        weights=jnp.ones(2, dtype=jnp.float64),
        overlaps=jnp.ones(2, dtype=jnp.complex128),
        rng_key=jax.random.PRNGKey(1967),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )
    advance_calls = []

    def advance_blocks(state, *, n_blocks: int):
        advance_calls.append(n_blocks)
        return (
            state,
            {
                "energy": jnp.zeros(n_blocks, dtype=jnp.float64),
                "weight": jnp.ones(n_blocks, dtype=jnp.float64),
            },
            (),
        )

    retuned = retune_ucisd_k_mode_pair_sampling(
        state,
        jnp.zeros(4, dtype=jnp.float64),
        jnp.ones(4, dtype=jnp.float64),
        None,
        ham,
        ctx,
        trial,
        advance_blocks=advance_blocks,
        tuning_cfg=UcisdKModePairTuningCfg(
            guide_policy="population_rms",
            target_tail_std_ha=1.0e6,
            safety_factor=1.0,
            candidate_sample_sizes=(16,),
            tuning_n_chunks=1,
            tuning_chol_batch_size=2,
            tuning_population_count=3,
            tuning_population_spacing_blocks=2,
            tail_probability_uniform_mix=0.0,
            track_half_sample_diagnostic=False,
            settling_blocks=0,
        ),
    )

    assert advance_calls == [2, 2]
    assert isinstance(retuned.meas_ctx.energy_sampling, UcisdKModePairSamplingCfg)
    assert retuned.meas_ctx.energy_sampling.chol_head_size == 4
    assert retuned.meas_ctx.energy_sampling.pair_sample_size == 16
    assert retuned.settling_blocks == 0


def test_k_mode_ops_require_restricted_walkers_and_validate_chunks():
    restricted = System(norb=6, nelec=(3, 2), walker_kind="restricted")
    assert make_ucisd_k_mode_trial_ops(restricted).overlap is k_mode_overlap_r
    make_ucisd_k_mode_meas_ops(restricted)

    unrestricted = System(norb=6, nelec=(3, 2), walker_kind="unrestricted")
    with pytest.raises(ValueError, match="only restricted walkers"):
        make_ucisd_k_mode_trial_ops(unrestricted)
    with pytest.raises(ValueError, match="only restricted walkers"):
        make_ucisd_k_mode_meas_ops(unrestricted)
    with pytest.raises(ValueError, match="n_mode_chunks must be positive"):
        make_ucisd_k_mode_meas_ops(restricted, n_mode_chunks=0)

    with pytest.raises(ValueError, match="chol_head_size must be nonnegative"):
        UcisdKModePairSamplingCfg(chol_head_size=-1, pair_sample_size=8)
    with pytest.raises(ValueError, match="pair_sample_size must be positive"):
        UcisdKModePairSamplingCfg(chol_head_size=0, pair_sample_size=0)

    with pytest.raises(ValueError, match="requires an equilibration"):
        make_ucisd_k_mode_meas_ops(
            restricted,
            energy_tuning=UcisdKModePairTuningCfg(),
        )


def test_k_mode_trial_runs_a_restricted_afqmc_smoke_calculation():
    """Tiny smoke test only; this is not a meaningful energy comparison."""
    _, trial, _ = _make_trials(norb=4, noa=2, nob=2, rank=3)
    sys = System(trial.norb, trial.nocc, walker_kind="restricted")
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1907),
        norb=trial.norb,
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
        seed=1913,
    )
    result = driver.run_qmc(
        sys=sys,
        params=params,
        ham_data=ham,
        trial_data=trial,
        trial_ops=make_ucisd_k_mode_trial_ops(sys),
        meas_ops=make_ucisd_k_mode_meas_ops(
            sys,
            mixed_precision=True,
            n_mode_chunks=2,
            energy_sampling=UcisdKModePairSamplingCfg(
                chol_head_size=2,
                pair_sample_size=8,
                rank_head_by_guide=True,
                tail_probability_uniform_mix=0.1,
                track_half_sample_diagnostic=True,
            ),
            energy_tuning=UcisdKModePairTuningCfg(
                target_tail_std_ha=1.0e6,
                candidate_sample_sizes=(8,),
                tuning_n_chunks=1,
                tuning_chol_batch_size=2,
                tuning_population_count=1,
                production_initial_n_chunks=1,
                tail_probability_uniform_mix=0.1,
                track_half_sample_diagnostic=True,
                settling_blocks=0,
            ),
        ),
        prop_ops=make_prop_ops(ham.basis, sys.walker_kind, mixed_precision=True),
        block_fn=block,
    )

    assert result.block_energies.shape[0] > 0
    assert jnp.all(jnp.isfinite(result.block_energies))
