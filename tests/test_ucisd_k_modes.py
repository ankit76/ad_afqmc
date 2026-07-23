from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot import driver, testing
from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.meas.ucisd import UcisdMeasCfg
from trot.meas.ucisd_k import (
    build_meas_ctx as build_k_meas_ctx,
    energy_kernel_rw_rh as k_energy_kernel,
    force_bias_kernel_rw_rh as k_force_bias_kernel,
)
from trot.meas.ucisd_k_modes import (
    _k_mode_apply_realimag,
    _k_mode_quadratic_batched_realimag,
    build_meas_ctx as build_k_mode_meas_ctx,
    energy_kernel_rw_rh as k_mode_energy_kernel,
    force_bias_kernel_rw_rh as k_mode_force_bias_kernel,
    get_ucisd_k_mode_meas_cfg,
    make_ucisd_k_mode_meas_ops,
)
from trot.prop.afqmc import make_prop_ops
from trot.prop.blocks import block
from trot.prop.types import QmcParams
from trot.trial.ucisd_k import UcisdKTrial, overlap_r as k_overlap_r
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
        ),
        prop_ops=make_prop_ops(ham.basis, sys.walker_kind, mixed_precision=True),
        block_fn=block,
    )

    assert result.block_energies.shape[0] > 0
    assert jnp.all(jnp.isfinite(result.block_energies))
