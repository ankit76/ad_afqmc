from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot import driver, testing
from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.meas.ucisd import (
    UcisdMeasCfg,
    build_meas_ctx as build_dense_meas_ctx,
    energy_kernel_rw_rh as dense_energy_kernel,
    force_bias_kernel_rw_rh as dense_force_bias_kernel,
)
from trot.meas.ucisd_k import (
    _combined_pair_batch,
    _k_quadratic_batched_realimag,
    build_meas_ctx as build_k_meas_ctx,
    energy_kernel_rw_rh as k_energy_kernel,
    force_bias_kernel_rw_rh as k_force_bias_kernel,
    get_ucisd_k_meas_cfg,
    make_ucisd_k_meas_ops,
)
from trot.prop.afqmc import make_prop_ops
from trot.prop.blocks import block
from trot.prop.types import QmcParams
from trot.trial.ucisd import UcisdTrial, overlap_r as dense_overlap_r
from trot.trial.ucisd_k import (
    UcisdKTrial,
    build_ucisd_kernel,
    k_apply,
    k_quadratic,
    make_ucisd_k_trial_data,
    make_ucisd_k_trial_ops,
    overlap_r as k_overlap_r,
)


def _same_spin_tensor(rng: np.random.Generator, nocc: int, nvir: int) -> np.ndarray:
    raw = rng.standard_normal((nocc, nvir, nocc, nvir))
    return 0.25 * (
        raw
        - raw.transpose(2, 1, 0, 3)
        - raw.transpose(0, 3, 2, 1)
        + raw.transpose(2, 3, 0, 1)
    )


def _make_dense_and_k_trials(
    *,
    seed: int = 1501,
    norb: int = 6,
    noa: int = 3,
    nob: int = 2,
) -> tuple[UcisdTrial, UcisdKTrial, np.ndarray]:
    rng = np.random.default_rng(seed)
    nva = norb - noa
    nvb = norb - nob
    rotation, _ = np.linalg.qr(np.eye(norb) + 0.15 * rng.standard_normal((norb, norb)))
    c1a = 0.03 * rng.standard_normal((noa, nva))
    c1b = 0.03 * rng.standard_normal((nob, nvb))
    c2aa = 0.02 * _same_spin_tensor(rng, noa, nva)
    c2ab = 0.02 * rng.standard_normal((noa, nva, nob, nvb))
    c2bb = 0.02 * _same_spin_tensor(rng, nob, nvb)
    kernel = np.block(
        [
            [c2aa.reshape(noa * nva, noa * nva), c2ab.reshape(noa * nva, nob * nvb)],
            [c2ab.reshape(noa * nva, nob * nvb).T, c2bb.reshape(nob * nvb, nob * nvb)],
        ]
    )
    dense = UcisdTrial(
        mo_coeff_a=jnp.eye(norb, dtype=jnp.float64),
        mo_coeff_b=jnp.asarray(rotation, dtype=jnp.float64),
        c1a=jnp.asarray(c1a, dtype=jnp.float64),
        c1b=jnp.asarray(c1b, dtype=jnp.float64),
        c2aa=jnp.asarray(c2aa, dtype=jnp.float64),
        c2ab=jnp.asarray(c2ab, dtype=jnp.float64),
        c2bb=jnp.asarray(c2bb, dtype=jnp.float64),
    )
    k_trial = UcisdKTrial(
        mo_coeff_a=dense.mo_coeff_a,
        mo_coeff_b=dense.mo_coeff_b,
        c1a=dense.c1a,
        c1b=dense.c1b,
        k=jnp.asarray(kernel, dtype=jnp.float64),
    )
    return dense, k_trial, kernel


def _double_cfg() -> UcisdMeasCfg:
    return UcisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )


def _walker(trial: UcisdKTrial, seed: int = 1511) -> jax.Array:
    return testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(seed),
        trial.norb,
        max(trial.nocc),
        mix=0.18,
    )


def test_combined_k_batch_supports_an_empty_beta_pair_space():
    trial = UcisdKTrial(
        mo_coeff_a=jnp.eye(2, dtype=jnp.float64),
        mo_coeff_b=jnp.eye(2, dtype=jnp.float64),
        c1a=jnp.zeros((1, 1), dtype=jnp.float64),
        c1b=jnp.zeros((0, 2), dtype=jnp.float64),
        k=jnp.asarray([[0.4]], dtype=jnp.float64),
    )
    matrices_a = jnp.asarray(
        [[[0.2 + 0.1j]], [[-0.3 + 0.05j]], [[0.1 - 0.2j]]],
        dtype=jnp.complex128,
    )
    matrices_b = jnp.zeros((3, 0, 2), dtype=jnp.complex128)

    vectors, leading_shape = _combined_pair_batch(trial, matrices_a, matrices_b)
    quadratic = jax.jit(_k_quadratic_batched_realimag, static_argnums=(3, 4))(
        trial,
        matrices_a,
        matrices_b,
        _double_cfg(),
        1,
    )

    assert leading_shape == (3,)
    assert vectors.shape == (3, 1)
    np.testing.assert_allclose(vectors[:, 0], matrices_a[:, 0, 0])
    np.testing.assert_allclose(
        quadratic,
        0.5 * trial.k[0, 0] * matrices_a[:, 0, 0] ** 2,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_combined_k_one_electron_measurement_matches_dense():
    c1a = jnp.asarray([[0.04]], dtype=jnp.float64)
    c1b = jnp.zeros((0, 2), dtype=jnp.float64)
    dense = UcisdTrial(
        mo_coeff_a=jnp.eye(2, dtype=jnp.float64),
        mo_coeff_b=jnp.eye(2, dtype=jnp.float64),
        c1a=c1a,
        c1b=c1b,
        c2aa=jnp.zeros((1, 1, 1, 1), dtype=jnp.float64),
        c2ab=jnp.zeros((1, 1, 0, 2), dtype=jnp.float64),
        c2bb=jnp.zeros((0, 2, 0, 2), dtype=jnp.float64),
    )
    trial = UcisdKTrial(
        mo_coeff_a=dense.mo_coeff_a,
        mo_coeff_b=dense.mo_coeff_b,
        c1a=c1a,
        c1b=c1b,
        k=jnp.zeros((1, 1), dtype=jnp.float64),
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1503),
        norb=2,
        n_chol=3,
        basis="restricted",
    )
    walker = jnp.asarray([[1.0 + 0.02j], [0.1 - 0.03j]], dtype=jnp.complex128)
    dense_ctx = build_dense_meas_ctx(ham, dense, cfg=_double_cfg())
    k_ctx = build_k_meas_ctx(ham, trial, cfg=_double_cfg())

    dense_fb = dense_force_bias_kernel(walker, ham, dense_ctx, dense)
    candidate_fb = jax.jit(k_force_bias_kernel)(walker, ham, k_ctx, trial)
    dense_energy = dense_energy_kernel(walker, ham, dense_ctx, dense)
    candidate_energy = jax.jit(k_energy_kernel)(walker, ham, k_ctx, trial)

    np.testing.assert_allclose(candidate_fb, dense_fb, rtol=3.0e-12, atol=3.0e-12)
    np.testing.assert_allclose(candidate_energy, dense_energy, rtol=3.0e-12, atol=3.0e-12)


def test_ucisd_kernel_and_trial_data_conversion_discard_dense_blocks():
    dense, trial, expected = _make_dense_and_k_trials()
    sys = System(dense.norb, dense.nocc, walker_kind="restricted")
    converted = make_ucisd_k_trial_data(
        {
            "mo_coeff_a": dense.mo_coeff_a,
            "mo_coeff_b": dense.mo_coeff_b,
            "c1a": dense.c1a,
            "c1b": dense.c1b,
            "ci2aa": dense.c2aa,
            "ci2ab": dense.c2ab,
            "ci2bb": dense.c2bb,
        },
        sys,
    )
    converted_from_k = make_ucisd_k_trial_data(
        {
            "mo_coeff_a": dense.mo_coeff_a,
            "mo_coeff_b": dense.mo_coeff_b,
            "c1a": dense.c1a,
            "c1b": dense.c1b,
            "k": expected,
        },
        sys,
    )

    np.testing.assert_allclose(
        build_ucisd_kernel(dense.c2aa, dense.c2ab, dense.c2bb),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(converted.k, trial.k, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(converted_from_k.k, trial.k, rtol=0.0, atol=0.0)
    assert converted.k.dtype == jnp.float64
    assert not hasattr(converted, "c2aa")
    assert not hasattr(converted, "c2ab")
    assert not hasattr(converted, "c2bb")


def test_k_helpers_match_explicit_spin_block_contractions():
    dense, trial, kernel = _make_dense_and_k_trials()
    key_ar, key_ai, key_br, key_bi = jax.random.split(jax.random.PRNGKey(1517), 4)
    shape_a = (trial.nocc[0], trial.nvir[0])
    shape_b = (trial.nocc[1], trial.nvir[1])
    matrix_a = jax.random.normal(key_ar, shape_a, dtype=jnp.float64)
    matrix_a += 1.0j * jax.random.normal(key_ai, shape_a, dtype=jnp.float64)
    matrix_b = jax.random.normal(key_br, shape_b, dtype=jnp.float64)
    matrix_b += 1.0j * jax.random.normal(key_bi, shape_b, dtype=jnp.float64)

    applied_a, applied_b = k_apply(trial, matrix_a, matrix_b)
    quadratic = k_quadratic(trial, matrix_a, matrix_b, (applied_a, applied_b))
    vector = np.concatenate((np.asarray(matrix_a).reshape(-1), np.asarray(matrix_b).reshape(-1)))
    expected_applied = kernel @ vector
    da, _ = trial.pair_dim
    expected_quadratic = 0.5 * vector @ kernel @ vector

    np.testing.assert_allclose(
        applied_a,
        expected_applied[:da].reshape(shape_a),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        applied_b,
        expected_applied[da:].reshape(shape_b),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(quadratic, expected_quadratic, rtol=2.0e-12, atol=2.0e-12)

    expected_block = (
        0.5 * jnp.einsum("iajb,ia,jb->", dense.c2aa, matrix_a, matrix_a)
        + jnp.einsum("iajb,ia,jb->", dense.c2ab, matrix_a, matrix_b)
        + 0.5 * jnp.einsum("iajb,ia,jb->", dense.c2bb, matrix_b, matrix_b)
    )
    np.testing.assert_allclose(quadratic, expected_block, rtol=2.0e-12, atol=2.0e-12)


def test_k_overlap_matches_dense_under_jit_and_vmap():
    dense, trial, _ = _make_dense_and_k_trials()
    walker = _walker(trial, 1523)
    reference = dense_overlap_r(walker, dense)
    candidate = k_overlap_r(walker, trial)
    candidate_jit = jax.jit(k_overlap_r)(walker, trial)
    candidate_batch = jax.vmap(k_overlap_r, in_axes=(0, None))(jnp.stack((walker, walker)), trial)

    np.testing.assert_allclose(candidate, reference, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(candidate_jit, reference, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        candidate_batch,
        jnp.stack((reference, reference)),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_k_force_bias_and_energy_match_dense_in_double_precision():
    dense, trial, _ = _make_dense_and_k_trials(seed=1531)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1543),
        norb=trial.norb,
        n_chol=7,
        basis="restricted",
    )
    dense_ctx = build_dense_meas_ctx(ham, dense, cfg=_double_cfg())
    k_ctx = build_k_meas_ctx(ham, trial, cfg=_double_cfg())
    walker = _walker(trial, 1549)

    dense_fb = dense_force_bias_kernel(walker, ham, dense_ctx, dense)
    candidate_fb = jax.jit(k_force_bias_kernel)(walker, ham, k_ctx, trial)
    dense_energy = dense_energy_kernel(walker, ham, dense_ctx, dense)
    candidate_energy = jax.jit(k_energy_kernel)(walker, ham, k_ctx, trial)

    np.testing.assert_allclose(candidate_fb, dense_fb, rtol=3.0e-12, atol=3.0e-12)
    np.testing.assert_allclose(candidate_energy, dense_energy, rtol=3.0e-12, atol=3.0e-12)


def test_k_mixed_precision_policy_is_accurate():
    dense, trial, _ = _make_dense_and_k_trials(seed=1553)
    sys = System(trial.norb, trial.nocc, walker_kind="restricted")
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1559),
        norb=trial.norb,
        n_chol=7,
        basis="restricted",
    )
    dense_ctx = build_dense_meas_ctx(ham, dense, cfg=_double_cfg())
    mixed_ops = make_ucisd_k_meas_ops(sys, mixed_precision=True)
    mixed_ctx = mixed_ops.build_meas_ctx(ham, trial)
    cfg = get_ucisd_k_meas_cfg(mixed_ops)
    assert cfg is not None
    assert cfg.mixed_real_dtype == jnp.float32
    assert cfg.mixed_complex_dtype == jnp.complex64
    assert trial.k.dtype == jnp.float64

    walker = _walker(trial, 1567)
    dense_fb = dense_force_bias_kernel(walker, ham, dense_ctx, dense)
    dense_energy = dense_energy_kernel(walker, ham, dense_ctx, dense)
    mixed_fb = mixed_ops.require_kernel(k_force_bias)(walker, ham, mixed_ctx, trial)
    mixed_energy = mixed_ops.require_kernel(k_energy)(walker, ham, mixed_ctx, trial)
    fb_error = float(jnp.linalg.norm(mixed_fb - dense_fb) / jnp.linalg.norm(dense_fb))
    energy_error = float(jnp.abs(mixed_energy - dense_energy))
    assert fb_error < 2.0e-5
    assert energy_error < 2.0e-4


def test_k_trial_and_measurement_validation():
    _, trial, _ = _make_dense_and_k_trials()
    leaves, treedef = jax.tree_util.tree_flatten(trial)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(restored.k, trial.k)

    with pytest.raises(ValueError, match="k must have shape"):
        UcisdKTrial(
            mo_coeff_a=trial.mo_coeff_a,
            mo_coeff_b=trial.mo_coeff_b,
            c1a=trial.c1a,
            c1b=trial.c1b,
            k=trial.k[:-1],
        )
    with pytest.raises(ValueError, match="real K matrix"):
        UcisdKTrial(
            mo_coeff_a=trial.mo_coeff_a,
            mo_coeff_b=trial.mo_coeff_b,
            c1a=trial.c1a,
            c1b=trial.c1b,
            k=trial.k.astype(jnp.complex128),
        )

    restricted = System(trial.norb, trial.nocc, walker_kind="restricted")
    assert make_ucisd_k_trial_ops(restricted).overlap is k_overlap_r
    make_ucisd_k_meas_ops(restricted)
    unrestricted = System(trial.norb, trial.nocc, walker_kind="unrestricted")
    with pytest.raises(ValueError, match="only restricted walkers"):
        make_ucisd_k_trial_ops(unrestricted)
    with pytest.raises(ValueError, match="only restricted walkers"):
        make_ucisd_k_meas_ops(unrestricted)
    with pytest.raises(ValueError, match="memory_mode='high'"):
        make_ucisd_k_meas_ops(restricted, memory_mode="low")


def test_k_trial_runs_a_restricted_afqmc_smoke_calculation():
    """Tiny smoke test only; this is not a meaningful energy comparison."""
    _, trial, _ = _make_dense_and_k_trials(norb=4, noa=2, nob=2)
    sys = System(trial.norb, trial.nocc, walker_kind="restricted")
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1571),
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
        seed=1579,
    )
    result = driver.run_qmc(
        sys=sys,
        params=params,
        ham_data=ham,
        trial_data=trial,
        trial_ops=make_ucisd_k_trial_ops(sys),
        meas_ops=make_ucisd_k_meas_ops(sys, mixed_precision=True),
        prop_ops=make_prop_ops(ham.basis, sys.walker_kind, mixed_precision=True),
        block_fn=block,
    )

    assert result.block_energies.shape[0] > 0
    assert jnp.all(jnp.isfinite(result.block_energies))
