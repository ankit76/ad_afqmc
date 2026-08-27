from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from trot.core.system import System
from trot.ham.chol import HamCholUhf
from trot.meas.lno_ptuccsd_modes import make_lno_ptuccsd_mode_estimator_ops
from trot.meas.lno_ptuccsd_thouless import make_lno_estimator_ops
from trot.meas.ptuccsd_modes import PtuccsdModePairSamplingCfg
from trot.trial.ptuccsd_modes import PtuccsdThoulessModeTrial, factorize_t2_modes
from trot.trial.ptuccsd_thouless import PtuccsdThoulessTrial, thouless_mo_from_t1


def _symmetric(rng, shape, scale=1.0):
    value = rng.normal(scale=scale, size=shape)
    return 0.5 * (value + value.swapaxes(-1, -2))


def _same_spin_t2(rng, nocc, nvir):
    value = rng.normal(scale=0.04, size=(nocc, nocc, nvir, nvir))
    value -= value.swapaxes(0, 1)
    value -= value.swapaxes(2, 3)
    value += value.transpose(1, 0, 3, 2)
    return 0.25 * value.transpose(0, 2, 1, 3)


def _case(seed=510, *, split=False):
    rng = np.random.default_rng(seed)
    norb, noa, nob, nchol = 6, 2, 3, 4
    nva, nvb = norb - noa, norb - nob
    t1a = rng.normal(scale=0.07, size=(noa, nva))
    t1b = rng.normal(scale=0.07, size=(nob, nvb))
    t2aa = _same_spin_t2(rng, noa, nva)
    t2ab = rng.normal(scale=0.04, size=(noa, nva, nob, nvb))
    t2bb = _same_spin_t2(rng, nob, nvb)
    if split:
        t1b[:, -1] = 0.0
        t2ab[:, :, :, -1] = 0.0
        t2bb[:, -1, :, :] = 0.0
        t2bb[:, :, :, -1] = 0.0
    dense = PtuccsdThoulessTrial(
        mo_t_a=thouless_mo_from_t1(jnp.asarray(t1a)),
        mo_t_b=thouless_mo_from_t1(jnp.asarray(t1b)),
        mo_coeff_b=jnp.eye(norb),
        t2aa=jnp.asarray(t2aa),
        t2ab=jnp.asarray(t2ab),
        t2bb=jnp.asarray(t2bb),
    )
    factorization = factorize_t2_modes(
        t2aa, t2ab, t2bb, mode_threshold=0.0, solver="dense"
    )
    mode = PtuccsdThoulessModeTrial(
        mo_t_a=dense.mo_t_a,
        mo_t_b=dense.mo_t_b,
        mo_coeff_b=dense.mo_coeff_b,
        eigenvalues=jnp.asarray(factorization.eigenvalues),
        modes=jnp.asarray(factorization.modes),
    )
    qa, _ = np.linalg.qr(rng.normal(size=(noa, noa)))
    qb, _ = np.linalg.qr(rng.normal(size=(nob, nob)))
    weight_a = jnp.asarray(qa @ np.diag([0.2, 0.8]) @ qa.T)
    weight_b = jnp.asarray(qb @ np.diag([0.15, 0.45, 0.85]) @ qb.T)
    h1_b = _symmetric(rng, (norb, norb))
    chol_b = _symmetric(rng, (nchol, norb, norb), 0.2)
    if split:
        h1_b[-1, :] = h1_b[:, -1] = 0.0
        chol_b[:, -1, :] = chol_b[:, :, -1] = 0.0
    ham = HamCholUhf(
        h0=jnp.asarray(0.4),
        h1_a=jnp.asarray(_symmetric(rng, (norb, norb))),
        h1_b=jnp.asarray(h1_b),
        chol_a=jnp.asarray(_symmetric(rng, (nchol, norb, norb), 0.2)),
        chol_b=jnp.asarray(chol_b),
        norb_spin=(norb, norb - int(split)),
    )
    walker_a = rng.normal(size=(norb, noa)) + 1.0j * rng.normal(size=(norb, noa))
    walker_b = rng.normal(size=(norb, nob)) + 1.0j * rng.normal(size=(norb, nob))
    walker_a[:noa] += np.eye(noa)
    walker_b[:nob] += np.eye(nob)
    return {
        "sys": System(norb, (noa, nob), walker_kind="unrestricted"),
        "ham": ham,
        "dense": dense,
        "mode": mode,
        "weights": (weight_a, weight_b),
        "walker": (jnp.asarray(walker_a), jnp.asarray(walker_b)),
    }


def _truncate(mode, rank=5):
    return PtuccsdThoulessModeTrial(
        mo_t_a=mode.mo_t_a,
        mo_t_b=mode.mo_t_b,
        mo_coeff_b=mode.mo_coeff_b,
        eigenvalues=mode.eigenvalues[:rank],
        modes=mode.modes[:rank],
    )


@pytest.mark.parametrize("split", [False, True])
def test_full_rank_mode_components_match_dense_historical_lno(split):
    case = _case(split=split)
    weight_a, weight_b = case["weights"]
    dense_ops = make_lno_estimator_ops(case["sys"], weight_a, weight_b)
    dense_ctx = dense_ops.build_estimator_ctx(case["ham"], case["dense"])
    expected = dense_ops.components(
        case["walker"], case["ham"], dense_ctx, case["dense"]
    )

    mode_ops = make_lno_ptuccsd_mode_estimator_ops(
        case["sys"],
        weight_a,
        weight_b,
        mixed_precision=False,
        testing=True,
    )
    mode_ctx = mode_ops.build_estimator_ctx(case["ham"], case["mode"])
    actual = mode_ops.components(
        case["walker"], case["ham"], mode_ctx, case["mode"]
    )
    np.testing.assert_allclose(actual, expected, rtol=5.0e-11, atol=5.0e-11)


def test_truncated_modes_match_dense_reconstruction_under_lno_weighting():
    case = _case(split=True)
    base = case["mode"]
    truncated = _truncate(base)
    modes = np.asarray(truncated.modes)
    kernel = (modes.T * np.asarray(truncated.eigenvalues)) @ modes
    da, db = truncated.pair_dim
    noa, nob = truncated.nocc
    nva, nvb = truncated.nvir
    dense = PtuccsdThoulessTrial(
        mo_t_a=truncated.mo_t_a,
        mo_t_b=truncated.mo_t_b,
        mo_coeff_b=truncated.mo_coeff_b,
        t2aa=jnp.asarray(kernel[:da, :da].reshape(noa, nva, noa, nva)),
        t2ab=jnp.asarray(kernel[:da, da:].reshape(noa, nva, nob, nvb)),
        t2bb=jnp.asarray(kernel[da:, da:].reshape(nob, nvb, nob, nvb)),
    )
    # Deliberately nonsymmetric weights catch an accidental W rather than W.T
    # in V[k,a] = sum_i U[i,a] W[i,k].
    weight_a = jnp.asarray([[0.2, 0.3], [-0.1, 0.7]])
    weight_b = jnp.asarray(
        [[0.5, -0.2, 0.1], [0.1, 0.4, 0.2], [-0.1, 0.05, 0.6]]
    )
    dense_ops = make_lno_estimator_ops(case["sys"], weight_a, weight_b)
    dense_ctx = dense_ops.build_estimator_ctx(case["ham"], dense)
    expected = dense_ops.components(case["walker"], case["ham"], dense_ctx, dense)

    mode_ops = make_lno_ptuccsd_mode_estimator_ops(
        case["sys"], weight_a, weight_b, mixed_precision=False
    )
    mode_ctx = mode_ops.build_estimator_ctx(case["ham"], truncated)
    actual = mode_ops.components(case["walker"], case["ham"], mode_ctx, truncated)
    np.testing.assert_allclose(actual, expected, rtol=5.0e-11, atol=5.0e-11)


def test_full_head_pair_sampling_matches_deterministic_population_numerator():
    case = _case()
    weight_a, weight_b = case["weights"]
    trial = _truncate(case["mode"])
    sampling = PtuccsdModePairSamplingCfg(
        chol_head_size=case["ham"].chol_a.shape[0],
        pair_sample_size=4,
        rank_head_by_guide=True,
        walker_guide_policy="head_rms",
    )
    ops = make_lno_ptuccsd_mode_estimator_ops(
        case["sys"],
        weight_a,
        weight_b,
        mixed_precision=False,
        testing=True,
        component_sampling=sampling,
    )
    ctx = ops.build_estimator_ctx(case["ham"], trial)
    walkers = (
        jnp.stack((case["walker"][0], case["walker"][0] + 0.01j)),
        jnp.stack((case["walker"][1], case["walker"][1] - 0.015j)),
    )
    weights = jnp.asarray([0.8 + 0.1j, 1.1 - 0.04j])
    components = jnp.stack(
        [
            ops.components(
                (walkers[0][i], walkers[1][i]), case["ham"], ctx, trial
            )
            for i in range(2)
        ]
    )
    sampled = ops.block_components(
        walkers,
        weights,
        jax.random.PRNGKey(7),
        1,
        case["ham"],
        ctx,
        trial,
    )
    np.testing.assert_allclose(sampled.weight, jnp.sum(weights), atol=1.0e-12)
    np.testing.assert_allclose(
        sampled.numerator,
        jnp.sum(weights[:, None] * components, axis=0),
        rtol=5.0e-11,
        atol=5.0e-11,
    )


def test_single_cholesky_tail_is_exact_for_one_walker():
    case = _case()
    weight_a, weight_b = case["weights"]
    trial = _truncate(case["mode"])
    sampling = PtuccsdModePairSamplingCfg(
        chol_head_size=case["ham"].chol_a.shape[0] - 1,
        pair_sample_size=2,
        rank_head_by_guide=True,
        track_half_sample_diagnostic=True,
    )
    ops = make_lno_ptuccsd_mode_estimator_ops(
        case["sys"],
        weight_a,
        weight_b,
        mixed_precision=False,
        testing=True,
        component_sampling=sampling,
    )
    ctx = ops.build_estimator_ctx(case["ham"], trial)
    walkers = (case["walker"][0][None], case["walker"][1][None])
    weight = jnp.asarray([0.9 + 0.08j])
    expected = weight[0] * ops.components(
        case["walker"], case["ham"], ctx, trial
    )
    sampled = jax.jit(ops.block_components, static_argnums=3)(
        walkers,
        weight,
        jax.random.PRNGKey(19),
        1,
        case["ham"],
        ctx,
        trial,
    )
    np.testing.assert_allclose(sampled.numerator, expected, rtol=5.0e-11, atol=5.0e-11)
