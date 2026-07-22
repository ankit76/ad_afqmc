from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot import testing
from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.ucisd import (
    UcisdMeasCfg,
    build_meas_ctx as build_dense_meas_ctx,
    energy_kernel_rw_rh as dense_energy_kernel,
    force_bias_kernel_rw_rh as dense_force_bias_kernel,
)
from trot.meas.ucisd_modes import (
    _chol_contract,
    _energy_gl_batched_realimag,
    _ucisd_mode_chol_terms,
    _ucisd_mode_energy_common,
    build_meas_ctx as build_mode_meas_ctx,
    energy_kernel_rw_rh as mode_energy_kernel,
    force_bias_kernel_rw_rh as mode_force_bias_kernel,
    get_ucisd_mode_meas_cfg,
    make_ucisd_mode_meas_ops,
)
from trot.trial.ucisd import UcisdTrial, overlap_r as dense_overlap_r
from trot.trial.ucisd_modes import (
    UcisdModeTrial,
    doubles_apply,
    doubles_projections,
    doubles_quadratic,
    make_ucisd_mode_trial_data,
    make_ucisd_mode_trial_ops,
    overlap_r as mode_overlap_r,
)


def _same_spin_tensor(rng: np.random.Generator, nocc: int, nvir: int) -> np.ndarray:
    raw = rng.standard_normal((nocc, nvir, nocc, nvir))
    return 0.25 * (
        raw
        - raw.transpose(2, 1, 0, 3)
        - raw.transpose(0, 3, 2, 1)
        + raw.transpose(2, 3, 0, 1)
    )


def _factorize_blockwise(
    c2aa: np.ndarray,
    c2ab: np.ndarray,
    c2bb: np.ndarray,
) -> tuple[np.ndarray, ...]:
    noa, nva, _, _ = c2aa.shape
    nob, nvb, _, _ = c2bb.shape
    da = noa * nva
    db = nob * nvb
    aa = c2aa.reshape(da, da)
    ab = c2ab.reshape(da, db)
    bb = c2bb.reshape(db, db)
    np.testing.assert_allclose(aa, aa.T, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(bb, bb.T, rtol=0.0, atol=1.0e-14)

    values_aa, vectors_aa = np.linalg.eigh(aa)
    values_bb, vectors_bb = np.linalg.eigh(bb)
    left_ab, values_ab, right_ab_t = np.linalg.svd(ab, full_matrices=False)
    order_aa = np.argsort(np.abs(values_aa))[::-1]
    order_ab = np.argsort(values_ab)[::-1]
    order_bb = np.argsort(np.abs(values_bb))[::-1]
    return (
        values_aa[order_aa],
        vectors_aa[:, order_aa].T.reshape(da, noa, nva),
        values_ab[order_ab],
        left_ab[:, order_ab].T.reshape(len(values_ab), noa, nva),
        right_ab_t[order_ab].reshape(len(values_ab), nob, nvb),
        values_bb[order_bb],
        vectors_bb[:, order_bb].T.reshape(db, nob, nvb),
    )


def _make_trials(
    *,
    seed: int = 1217,
    norb: int = 6,
    noa: int = 3,
    nob: int = 2,
    ranks: tuple[int, int, int] | None = None,
    mode_dtype=jnp.float64,
) -> tuple[UcisdTrial, UcisdModeTrial]:
    rng = np.random.default_rng(seed)
    nva = norb - noa
    nvb = norb - nob
    rotation, _ = np.linalg.qr(np.eye(norb) + 0.15 * rng.standard_normal((norb, norb)))
    c1a = 0.03 * rng.standard_normal((noa, nva))
    c1b = 0.03 * rng.standard_normal((nob, nvb))
    c2aa = 0.02 * _same_spin_tensor(rng, noa, nva)
    c2ab = 0.02 * rng.standard_normal((noa, nva, nob, nvb))
    c2bb = 0.02 * _same_spin_tensor(rng, nob, nvb)
    factors = _factorize_blockwise(c2aa, c2ab, c2bb)
    if ranks is None:
        ranks = (len(factors[0]), len(factors[2]), len(factors[5]))
    raa, rab, rbb = ranks
    dense = UcisdTrial(
        mo_coeff_a=jnp.eye(norb, dtype=jnp.float64),
        mo_coeff_b=jnp.asarray(rotation, dtype=jnp.float64),
        c1a=jnp.asarray(c1a, dtype=jnp.float64),
        c1b=jnp.asarray(c1b, dtype=jnp.float64),
        c2aa=jnp.asarray(c2aa, dtype=jnp.float64),
        c2ab=jnp.asarray(c2ab, dtype=jnp.float64),
        c2bb=jnp.asarray(c2bb, dtype=jnp.float64),
    )
    modes = UcisdModeTrial(
        mo_coeff_a=dense.mo_coeff_a,
        mo_coeff_b=dense.mo_coeff_b,
        c1a=dense.c1a,
        c1b=dense.c1b,
        eigenvalues_aa=jnp.asarray(factors[0][:raa], dtype=jnp.float64),
        modes_aa=jnp.asarray(factors[1][:raa], dtype=mode_dtype),
        singular_values_ab=jnp.asarray(factors[2][:rab], dtype=jnp.float64),
        left_modes_ab=jnp.asarray(factors[3][:rab], dtype=mode_dtype),
        right_modes_ab=jnp.asarray(factors[4][:rab], dtype=mode_dtype),
        eigenvalues_bb=jnp.asarray(factors[5][:rbb], dtype=jnp.float64),
        modes_bb=jnp.asarray(factors[6][:rbb], dtype=mode_dtype),
    )
    return dense, modes


def _reconstruct_dense(trial: UcisdModeTrial) -> UcisdTrial:
    da, db = trial.pair_dim
    raa, rab, rbb = trial.mode_rank
    uaa = np.asarray(trial.modes_aa).reshape(raa, da).T
    uab = np.asarray(trial.left_modes_ab).reshape(rab, da).T
    vab = np.asarray(trial.right_modes_ab).reshape(rab, db).T
    ubb = np.asarray(trial.modes_bb).reshape(rbb, db).T
    c2aa = (uaa * np.asarray(trial.eigenvalues_aa)) @ uaa.T
    c2ab = (uab * np.asarray(trial.singular_values_ab)) @ vab.T
    c2bb = (ubb * np.asarray(trial.eigenvalues_bb)) @ ubb.T
    noa, nob = trial.nocc
    nva, nvb = trial.nvir
    return UcisdTrial(
        mo_coeff_a=trial.mo_coeff_a,
        mo_coeff_b=trial.mo_coeff_b,
        c1a=trial.c1a,
        c1b=trial.c1b,
        c2aa=jnp.asarray(c2aa.reshape(noa, nva, noa, nva)),
        c2ab=jnp.asarray(c2ab.reshape(noa, nva, nob, nvb)),
        c2bb=jnp.asarray(c2bb.reshape(nob, nvb, nob, nvb)),
    )


def _double_cfg() -> UcisdMeasCfg:
    return UcisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )


def _walker(trial: UcisdModeTrial, seed: int = 1231) -> jax.Array:
    return testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(seed),
        trial.norb,
        max(trial.nocc),
        mix=0.12,
    )


def _assert_mode_matches_dense(
    dense: UcisdTrial,
    modes: UcisdModeTrial,
    *,
    seed: int,
    n_mode_chunks: int = 1,
) -> None:
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(seed), modes.norb, n_chol=7, basis="restricted"
    )
    dense_ctx = build_dense_meas_ctx(ham, dense, _double_cfg())
    mode_ctx = build_mode_meas_ctx(
        ham, modes, cfg=_double_cfg(), n_mode_chunks=n_mode_chunks
    )
    walker = _walker(modes, seed + 1)

    dense_overlap = dense_overlap_r(walker, dense)
    mode_overlap = jax.jit(mode_overlap_r)(walker, modes)
    dense_fb = dense_force_bias_kernel(walker, ham, dense_ctx, dense)
    mode_fb = jax.jit(mode_force_bias_kernel)(walker, ham, mode_ctx, modes)
    dense_energy = dense_energy_kernel(walker, ham, dense_ctx, dense)
    mode_energy = jax.jit(mode_energy_kernel)(walker, ham, mode_ctx, modes)
    np.testing.assert_allclose(mode_overlap, dense_overlap, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(mode_fb, dense_fb, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(mode_energy, dense_energy, rtol=2.0e-12, atol=2.0e-12)


def test_full_rank_overlap_force_bias_and_energy_match_dense_ucisd():
    dense, modes = _make_trials()
    _assert_mode_matches_dense(dense, modes, seed=1249, n_mode_chunks=3)

    walkers = jnp.stack((_walker(modes, 1259), _walker(modes, 1277)))
    expected = jax.vmap(dense_overlap_r, in_axes=(0, None))(walkers, dense)
    actual = jax.vmap(mode_overlap_r, in_axes=(0, None))(walkers, modes)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)


def test_full_rank_individual_cholesky_contributions_match_dense_differences():
    dense, modes = _make_trials(seed=1283)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1289), modes.norb, n_chol=5, basis="restricted"
    )
    walker = _walker(modes, 1291)
    mode_ctx = build_mode_meas_ctx(ham, modes, cfg=_double_cfg(), n_mode_chunks=2)
    common = _ucisd_mode_energy_common(walker, ham, mode_ctx, modes)
    base = mode_ctx.base
    mode_terms = _ucisd_mode_chol_terms(
        common,
        ham.chol,
        base.rot_chol_a,
        base.lci1_a,
        base.chol_b,
        base.rot_chol_b,
        base.lci1_b,
        mode_ctx,
        modes,
    )

    zero_ham = HamChol(
        h0=ham.h0,
        h1=ham.h1,
        chol=jnp.zeros((1, modes.norb, modes.norb), dtype=ham.chol.dtype),
        basis="restricted",
    )
    zero_ctx = build_dense_meas_ctx(zero_ham, dense, _double_cfg())
    zero_energy = dense_energy_kernel(walker, zero_ham, zero_ctx, dense)
    dense_terms = []
    for chol_index in range(int(ham.chol.shape[0])):
        one_ham = HamChol(
            h0=ham.h0,
            h1=ham.h1,
            chol=ham.chol[chol_index : chol_index + 1],
            basis="restricted",
        )
        one_ctx = build_dense_meas_ctx(one_ham, dense, _double_cfg())
        dense_terms.append(dense_energy_kernel(walker, one_ham, one_ctx, dense) - zero_energy)
    np.testing.assert_allclose(
        mode_terms,
        jnp.asarray(dense_terms),
        rtol=3.0e-12,
        atol=3.0e-12,
    )


@pytest.mark.parametrize("ranks", [(4, 3, 2), (0, 5, 0), (3, 0, 2), (0, 0, 0)])
def test_arbitrary_truncated_modes_match_reconstructed_dense_oracle(ranks):
    _, modes = _make_trials(seed=1301, ranks=ranks)
    reconstructed = _reconstruct_dense(modes)
    _assert_mode_matches_dense(reconstructed, modes, seed=1303, n_mode_chunks=2)


def test_mode_helpers_match_explicit_block_contractions():
    _, trial = _make_trials(seed=1321)
    rng = np.random.default_rng(1327)
    matrix_a = rng.standard_normal((trial.nocc[0], trial.nvir[0]))
    matrix_a = jnp.asarray(matrix_a + 1.0j * rng.standard_normal(matrix_a.shape))
    matrix_b = rng.standard_normal((trial.nocc[1], trial.nvir[1]))
    matrix_b = jnp.asarray(matrix_b + 1.0j * rng.standard_normal(matrix_b.shape))
    dense = _reconstruct_dense(trial)

    projections = doubles_projections(trial, matrix_a, matrix_b)
    applied_a, applied_b = doubles_apply(
        trial,
        matrix_a,
        matrix_b,
        projections=projections,
    )
    expected_a = jnp.einsum("ptqu,pt->qu", dense.c2aa, matrix_a, optimize="optimal")
    expected_a += jnp.einsum("ptqu,qu->pt", dense.c2ab, matrix_b, optimize="optimal")
    expected_b = jnp.einsum("ptqu,pt->qu", dense.c2bb, matrix_b, optimize="optimal")
    expected_b += jnp.einsum("ptqu,pt->qu", dense.c2ab, matrix_a, optimize="optimal")
    expected_quadratic = 0.5 * jnp.einsum(
        "pt,qu,ptqu->", matrix_a, matrix_a, dense.c2aa, optimize="optimal"
    )
    expected_quadratic += jnp.einsum(
        "pt,qu,ptqu->", matrix_a, matrix_b, dense.c2ab, optimize="optimal"
    )
    expected_quadratic += 0.5 * jnp.einsum(
        "pt,qu,ptqu->", matrix_b, matrix_b, dense.c2bb, optimize="optimal"
    )
    np.testing.assert_allclose(applied_a, expected_a, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(applied_b, expected_b, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        doubles_quadratic(
            trial,
            matrix_a,
            matrix_b,
            projections=projections,
        ),
        expected_quadratic,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_mixed_realimag_cholesky_helpers_match_complex_contractions():
    rng = np.random.default_rng(1351)
    cfg = UcisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float32,
        mixed_complex_dtype=jnp.complex64,
        mixed_real_dtype_testing=jnp.float32,
        mixed_complex_dtype_testing=jnp.complex64,
    )
    chol = jnp.asarray(rng.standard_normal((5, 6, 6)), dtype=jnp.float64)
    matrix_np = rng.standard_normal((6, 6)) + 1.0j * rng.standard_normal((6, 6))
    matrix = jnp.asarray(matrix_np, dtype=jnp.complex128)
    green_np = rng.standard_normal((3, 6)) + 1.0j * rng.standard_normal((3, 6))
    green = jnp.asarray(green_np, dtype=jnp.complex128)

    contraction = _chol_contract(chol, matrix, cfg)
    expected_contraction = jnp.einsum(
        "gij,ij->g",
        chol.astype(jnp.float32),
        matrix.astype(jnp.complex64),
        optimize="optimal",
    )
    gl = _energy_gl_batched_realimag(green, chol, cfg)
    expected_gl = jnp.einsum(
        "pj,gji->gpi",
        green.astype(jnp.complex64),
        chol.astype(jnp.float32),
        optimize="optimal",
    )
    assert contraction.dtype == jnp.complex64
    assert gl.dtype == jnp.complex64
    np.testing.assert_allclose(contraction, expected_contraction, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(gl, expected_gl, rtol=2.0e-6, atol=2.0e-6)


def test_trial_loader_mixed_precision_and_pytree_behavior():
    dense, trial = _make_trials(seed=1361)
    sys = System(norb=trial.norb, nelec=trial.nocc, walker_kind="restricted")
    data = {
        "mo_coeff_a": np.asarray(trial.mo_coeff_a),
        "mo_coeff_b": np.asarray(trial.mo_coeff_b),
        "c1a": np.asarray(trial.c1a),
        "c1b": np.asarray(trial.c1b),
        "eigenvalues_aa": np.asarray(trial.eigenvalues_aa),
        "eigenvectors_aa": np.asarray(trial.modes_aa).reshape(trial.mode_rank[0], -1).T,
        "singular_values_ab": np.asarray(trial.singular_values_ab),
        "left_singular_vectors_ab": np.asarray(trial.left_modes_ab)
        .reshape(trial.mode_rank[1], -1)
        .T,
        "right_singular_vectors_ab": np.asarray(trial.right_modes_ab)
        .reshape(trial.mode_rank[1], -1)
        .T,
        "eigenvalues_bb": np.asarray(trial.eigenvalues_bb),
        "eigenvectors_bb": np.asarray(trial.modes_bb).reshape(trial.mode_rank[2], -1).T,
    }
    mixed = make_ucisd_mode_trial_data(data, sys, mixed_precision=True)
    assert mixed.eigenvalues_aa.dtype == jnp.float64
    assert mixed.singular_values_ab.dtype == jnp.float64
    assert mixed.eigenvalues_bb.dtype == jnp.float64
    assert mixed.modes_aa.dtype == jnp.float32
    assert mixed.left_modes_ab.dtype == jnp.float32
    assert mixed.right_modes_ab.dtype == jnp.float32
    assert mixed.modes_bb.dtype == jnp.float32

    leaves, treedef = jax.tree_util.tree_flatten(mixed)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.mode_rank == mixed.mode_rank
    np.testing.assert_array_equal(restored.singular_values_ab, mixed.singular_values_ab)

    walker = _walker(mixed, 1367)
    dense_overlap = dense_overlap_r(walker, dense)
    mixed_overlap = mode_overlap_r(walker, mixed)
    assert float(jnp.abs(mixed_overlap - dense_overlap) / jnp.abs(dense_overlap)) < 1.0e-5


def test_mixed_measurements_are_accurate_and_report_policy():
    dense, modes = _make_trials(seed=1381, mode_dtype=jnp.float32)
    sys = System(norb=modes.norb, nelec=modes.nocc, walker_kind="restricted")
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1399), modes.norb, n_chol=7, basis="restricted"
    )
    dense_ctx = build_dense_meas_ctx(ham, dense, _double_cfg())
    mode_ops = make_ucisd_mode_meas_ops(sys, mixed_precision=True, n_mode_chunks=3)
    mode_ctx = mode_ops.build_meas_ctx(ham, modes)
    cfg = get_ucisd_mode_meas_cfg(mode_ops)
    assert cfg is not None
    assert cfg.mixed_real_dtype == jnp.float32
    assert cfg.mixed_complex_dtype == jnp.complex64

    walker = _walker(modes, 1409)
    dense_fb = dense_force_bias_kernel(walker, ham, dense_ctx, dense)
    mode_fb = mode_ops.require_kernel(k_force_bias)(walker, ham, mode_ctx, modes)
    dense_energy = dense_energy_kernel(walker, ham, dense_ctx, dense)
    mode_energy = mode_ops.require_kernel(k_energy)(walker, ham, mode_ctx, modes)
    fb_error = float(jnp.linalg.norm(mode_fb - dense_fb) / jnp.linalg.norm(dense_fb))
    energy_error = float(jnp.abs(mode_energy - dense_energy))
    assert fb_error < 2.0e-5
    assert energy_error < 2.0e-4


def test_mode_chunks_do_not_change_double_precision_energy():
    _, modes = _make_trials(seed=1423)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1427), modes.norb, n_chol=6, basis="restricted"
    )
    walker = _walker(modes, 1429)
    ctx_one = build_mode_meas_ctx(ham, modes, cfg=_double_cfg(), n_mode_chunks=1)
    ctx_many = build_mode_meas_ctx(ham, modes, cfg=_double_cfg(), n_mode_chunks=100)
    energy_one = mode_energy_kernel(walker, ham, ctx_one, modes)
    energy_many = mode_energy_kernel(walker, ham, ctx_many, modes)
    np.testing.assert_allclose(energy_many, energy_one, rtol=2.0e-12, atol=2.0e-12)
    assert ctx_many.n_mode_chunks == max(modes.mode_rank)


def test_ops_require_restricted_walkers_but_allow_spin_imbalanced_trials():
    restricted = System(norb=6, nelec=(3, 2), walker_kind="restricted")
    assert make_ucisd_mode_trial_ops(restricted).overlap is mode_overlap_r
    make_ucisd_mode_meas_ops(restricted)

    unrestricted = System(norb=6, nelec=(3, 2), walker_kind="unrestricted")
    with pytest.raises(ValueError, match="only restricted walkers"):
        make_ucisd_mode_trial_ops(unrestricted)
    with pytest.raises(ValueError, match="only restricted walkers"):
        make_ucisd_mode_meas_ops(unrestricted)


def test_shape_validation_and_mode_chunk_validation():
    _, trial = _make_trials(seed=1433)
    with pytest.raises(ValueError, match="alpha-beta rank"):
        UcisdModeTrial(
            **{
                **trial.__dict__,
                "singular_values_ab": jnp.ones(min(trial.pair_dim) + 1),
                "left_modes_ab": jnp.ones((min(trial.pair_dim) + 1,) + trial.modes_aa.shape[1:]),
                "right_modes_ab": jnp.ones(
                    (min(trial.pair_dim) + 1,) + trial.modes_bb.shape[1:]
                ),
            }
        )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(1439), trial.norb, n_chol=3, basis="restricted"
    )
    with pytest.raises(ValueError, match="n_mode_chunks must be positive"):
        build_mode_meas_ctx(ham, trial, n_mode_chunks=0)
