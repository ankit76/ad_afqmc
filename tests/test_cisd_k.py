from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot import driver, testing
from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.meas.cisd import (
    CisdMeasCfg,
    build_meas_ctx as build_dense_meas_ctx,
    energy_kernel_rw_rh as dense_energy_kernel,
    force_bias_kernel_rw_rh_high as dense_force_bias_kernel,
    make_cisd_meas_ops,
)
from trot.meas.cisd_k import (
    build_meas_ctx as build_k_meas_ctx,
    energy_kernel_rw_rh as k_energy_kernel,
    force_bias_kernel_rw_rh as k_force_bias_kernel,
    get_cisd_k_meas_cfg,
    make_cisd_k_meas_ops,
)
from trot.prop.afqmc import make_prop_ops
from trot.prop.blocks import block
from trot.prop.types import QmcParams
from trot.trial.cisd import CisdTrial, overlap_r as dense_overlap_r
from trot.trial.cisd_k import (
    CisdKTrial,
    build_spin_adapted_kernel,
    k_apply,
    k_quadratic,
    make_cisd_k_trial_data,
    make_cisd_k_trial_ops,
    overlap_r as k_overlap_r,
)


def _make_dense_and_k_trials(
    *,
    seed: int = 1201,
    nocc: int = 3,
    nvir: int = 4,
    nocc_t_core: int = 0,
    nvir_t_outer: int = 0,
) -> tuple[CisdTrial, CisdKTrial, np.ndarray]:
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

    dense_trial = CisdTrial(
        ci1=jnp.asarray(ci1, dtype=jnp.float64),
        ci2=jnp.asarray(ci2, dtype=jnp.float64),
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    k_trial = CisdKTrial(
        ci1=jnp.asarray(ci1, dtype=jnp.float64),
        k=jnp.asarray(kernel, dtype=jnp.float64),
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    return dense_trial, k_trial, kernel


def _double_cfg() -> CisdMeasCfg:
    return CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )


def test_spin_adapted_kernel_and_trial_data_conversion_discard_ci2():
    dense_trial, k_trial, expected_kernel = _make_dense_and_k_trials(nocc=2, nvir=3)
    sys = System(
        norb=dense_trial.norb,
        nelec=(dense_trial.nocc_full, dense_trial.nocc_full),
        walker_kind="restricted",
    )
    data = {
        "ci1": np.asarray(dense_trial.ci1, dtype=np.float32),
        "ci2": np.asarray(dense_trial.ci2, dtype=np.float32),
    }
    converted = make_cisd_k_trial_data(data, sys)
    converted_from_k = make_cisd_k_trial_data({"ci1": data["ci1"], "k": expected_kernel}, sys)

    np.testing.assert_allclose(
        build_spin_adapted_kernel(dense_trial.ci2),
        expected_kernel,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(converted.k, k_trial.k, rtol=2.0e-7, atol=2.0e-9)
    np.testing.assert_allclose(converted_from_k.k, k_trial.k, rtol=0.0, atol=0.0)
    assert converted.ci1.dtype == jnp.float64
    assert converted.k.dtype == jnp.float64
    assert not hasattr(converted, "ci2")


def test_k_helpers_match_explicit_bilinear_contractions():
    _, trial, kernel = _make_dense_and_k_trials()
    key_r, key_i = jax.random.split(jax.random.PRNGKey(1213))
    matrix = jax.random.normal(key_r, (trial.nocc, trial.nvir), dtype=jnp.float64)
    matrix = matrix + 1.0j * jax.random.normal(key_i, (trial.nocc, trial.nvir), dtype=jnp.float64)

    applied = k_apply(trial, matrix)
    quadratic = k_quadratic(trial, matrix, applied)
    matrix_flat = np.asarray(matrix).reshape(-1)
    expected_applied = (kernel @ matrix_flat).reshape(matrix.shape)
    expected_quadratic = matrix_flat @ kernel @ matrix_flat
    np.testing.assert_allclose(applied, expected_applied, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(quadratic, expected_quadratic, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
def test_k_overlap_matches_dense_under_jit_and_vmap(nocc_t_core, nvir_t_outer):
    dense_trial, k_trial, _ = _make_dense_and_k_trials(
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(1223),
        k_trial.norb,
        k_trial.nocc_full,
        mix=0.25,
    )
    reference = dense_overlap_r(walker, dense_trial)
    candidate = k_overlap_r(walker, k_trial)
    candidate_jit = jax.jit(k_overlap_r)(walker, k_trial)
    candidate_batch = jax.vmap(k_overlap_r, in_axes=(0, None))(jnp.stack((walker, walker)), k_trial)

    np.testing.assert_allclose(candidate, reference, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(candidate_jit, reference, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        candidate_batch,
        jnp.stack((reference, reference)),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
def test_k_force_bias_and_energy_match_dense_in_double_precision(
    nocc_t_core,
    nvir_t_outer,
):
    dense_trial, k_trial, _ = _make_dense_and_k_trials(
        nocc=2,
        nvir=3,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1231),
        norb=k_trial.norb,
        n_chol=7,
        basis="restricted",
    )
    cfg = _double_cfg()
    dense_ctx = build_dense_meas_ctx(ham, dense_trial, cfg=cfg)
    k_ctx = build_k_meas_ctx(ham, k_trial, cfg=cfg)
    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(1237), k_trial.norb, k_trial.nocc_full, mix=0.25
    )

    dense_fb = dense_force_bias_kernel(walker, ham, dense_ctx, dense_trial)
    candidate_fb = jax.jit(k_force_bias_kernel)(walker, ham, k_ctx, k_trial)
    dense_energy = dense_energy_kernel(walker, ham, dense_ctx, dense_trial)
    candidate_energy = jax.jit(k_energy_kernel)(walker, ham, k_ctx, k_trial)

    np.testing.assert_allclose(candidate_fb, dense_fb, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(candidate_energy, dense_energy, rtol=2.0e-12, atol=2.0e-12)


def test_k_mixed_precision_policy_matches_dense_and_remains_accurate():
    dense_trial, k_trial, _ = _make_dense_and_k_trials(nocc=2, nvir=3)
    sys = System(
        norb=k_trial.norb,
        nelec=(k_trial.nocc_full, k_trial.nocc_full),
        walker_kind="restricted",
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1249),
        norb=k_trial.norb,
        n_chol=7,
        basis="restricted",
    )
    dense_dp_ctx = build_dense_meas_ctx(ham, dense_trial, cfg=_double_cfg())
    dense_mixed_ops = make_cisd_meas_ops(
        sys, memory_mode="high", mixed_precision=True, testing=False
    )
    k_mixed_ops = make_cisd_k_meas_ops(sys, memory_mode="high", mixed_precision=True, testing=False)
    dense_mixed_ctx = dense_mixed_ops.build_meas_ctx(ham, dense_trial)
    k_mixed_ctx = k_mixed_ops.build_meas_ctx(ham, k_trial)
    cfg = get_cisd_k_meas_cfg(k_mixed_ops)
    assert cfg is not None
    assert cfg.mixed_real_dtype == jnp.float32
    assert cfg.mixed_complex_dtype == jnp.complex64
    assert cfg.mixed_real_dtype_testing == jnp.float32
    assert cfg.mixed_complex_dtype_testing == jnp.complex64
    assert k_trial.k.dtype == jnp.float64

    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(1259), k_trial.norb, k_trial.nocc_full, mix=0.25
    )
    dense_dp_fb = dense_force_bias_kernel(walker, ham, dense_dp_ctx, dense_trial)
    dense_dp_energy = dense_energy_kernel(walker, ham, dense_dp_ctx, dense_trial)
    dense_mixed_fb = dense_mixed_ops.require_kernel(k_force_bias)(
        walker, ham, dense_mixed_ctx, dense_trial
    )
    dense_mixed_energy = dense_mixed_ops.require_kernel(k_energy)(
        walker, ham, dense_mixed_ctx, dense_trial
    )
    k_mixed_fb = k_mixed_ops.require_kernel(k_force_bias)(walker, ham, k_mixed_ctx, k_trial)
    k_mixed_energy = k_mixed_ops.require_kernel(k_energy)(walker, ham, k_mixed_ctx, k_trial)

    dense_fb_error = float(
        jnp.linalg.norm(dense_mixed_fb - dense_dp_fb) / jnp.linalg.norm(dense_dp_fb)
    )
    k_fb_error = float(jnp.linalg.norm(k_mixed_fb - dense_dp_fb) / jnp.linalg.norm(dense_dp_fb))
    dense_energy_error = float(jnp.abs(dense_mixed_energy - dense_dp_energy))
    k_energy_error = float(jnp.abs(k_mixed_energy - dense_dp_energy))
    assert dense_fb_error < 1.0e-5
    assert k_fb_error < 1.0e-5
    assert dense_energy_error < 1.0e-4
    assert k_energy_error < 1.0e-4


def test_k_trial_and_measurement_validation():
    _, trial, _ = _make_dense_and_k_trials(nocc_t_core=1, nvir_t_outer=2)
    leaves, treedef = jax.tree_util.tree_flatten(trial)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.nocc_t_core == 1
    assert restored.nvir_t_outer == 2
    np.testing.assert_array_equal(restored.k, trial.k)

    with pytest.raises(ValueError, match="k must have shape"):
        CisdKTrial(ci1=trial.ci1, k=trial.k[:-1])
    with pytest.raises(ValueError, match="real K matrix"):
        CisdKTrial(ci1=trial.ci1, k=trial.k.astype(jnp.complex128))

    restricted = System(norb=trial.norb, nelec=(3, 3), walker_kind="restricted")
    assert make_cisd_k_trial_ops(restricted).overlap is k_overlap_r
    with pytest.raises(ValueError, match="nup == ndn"):
        make_cisd_k_trial_ops(System(norb=trial.norb, nelec=(3, 2), walker_kind="restricted"))
    with pytest.raises(ValueError, match="restricted walkers"):
        make_cisd_k_trial_ops(System(norb=trial.norb, nelec=(3, 3), walker_kind="unrestricted"))
    with pytest.raises(ValueError, match="memory_mode='high'"):
        make_cisd_k_meas_ops(restricted, memory_mode="low")


def test_k_trial_runs_a_restricted_afqmc_smoke_calculation():
    """Tiny smoke test only; this is not a meaningful energy comparison."""
    _, trial, _ = _make_dense_and_k_trials(nocc=2, nvir=2)
    sys = System(
        norb=trial.norb,
        nelec=(trial.nocc_full, trial.nocc_full),
        walker_kind="restricted",
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1277),
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
        seed=1283,
    )
    result = driver.run_qmc(
        sys=sys,
        params=params,
        ham_data=ham,
        trial_data=trial,
        trial_ops=make_cisd_k_trial_ops(sys),
        meas_ops=make_cisd_k_meas_ops(sys, mixed_precision=True),
        prop_ops=make_prop_ops(ham.basis, sys.walker_kind, mixed_precision=True),
        block_fn=block,
    )

    assert result.block_energies.shape[0] > 0
    assert jnp.all(jnp.isfinite(result.block_energies))
