from trot import config

config.configure_once(use_gpu=False)

import jax.numpy as jnp
import numpy as np

from trot.core.system import System
from trot.ham.chol import HamCholUhf
from trot.lno_uhf import _make_pt2uccsd_trial, joint_df_pair_cholesky
from trot.meas.lno_ptuccsd_thouless import make_lno_estimator_ops
from trot.trial.pt2uccsd import make_pt2uccsd_trial_data


def _symmetric(rng, shape, scale=1.0):
    array = rng.normal(scale=scale, size=shape)
    return 0.5 * (array + array.swapaxes(-1, -2))


def _same_spin_t2(rng, nocc, nvir):
    t2 = rng.normal(scale=0.04, size=(nocc, nocc, nvir, nvir))
    t2 -= t2.swapaxes(0, 1)
    t2 -= t2.swapaxes(2, 3)
    t2 += t2.transpose(1, 0, 3, 2)
    return 0.25 * t2


def test_joint_cholesky_reconstructs_all_spin_blocks():
    rng = np.random.default_rng(7)
    df_a = rng.normal(size=(8, 6))
    df_b = rng.normal(size=(8, 3))
    chol = joint_df_pair_cholesky(df_a, df_b, chol_cut=1.0e-12)
    la = chol.chol_a[:, np.tril_indices(3)[0], np.tril_indices(3)[1]]
    lb = chol.chol_b[:, np.tril_indices(2)[0], np.tril_indices(2)[1]]

    np.testing.assert_allclose(la.T @ la, df_a.T @ df_a, atol=1.0e-11)
    np.testing.assert_allclose(la.T @ lb, df_a.T @ df_b, atol=1.0e-11)
    np.testing.assert_allclose(lb.T @ lb, df_b.T @ df_b, atol=1.0e-11)


def test_exact_lno_estimator_matches_historical_result():
    rng = np.random.default_rng(91)
    nora, norb, noa, nob, nchol = 5, 4, 2, 2, 4
    h1a = _symmetric(rng, (nora, nora))
    h1b = _symmetric(rng, (norb, norb))
    chola = _symmetric(rng, (nchol, nora, nora), 0.2)
    cholb = _symmetric(rng, (nchol, norb, norb), 0.2)
    t1a = rng.normal(scale=0.08, size=(noa, nora - noa))
    t1b = rng.normal(scale=0.08, size=(nob, norb - nob))
    t2aa = _same_spin_t2(rng, noa, nora - noa)
    t2ab = rng.normal(
        scale=0.04, size=(noa, nora - noa, nob, norb - nob)
    ).transpose(0, 2, 1, 3)
    t2bb = _same_spin_t2(rng, nob, norb - nob)
    qa, _ = np.linalg.qr(rng.normal(size=(noa, noa)))
    qb, _ = np.linalg.qr(rng.normal(size=(nob, nob)))
    uocc = (
        qa @ np.diag(np.sqrt([0.2, 0.8])),
        qb @ np.diag(np.sqrt([0.3, 0.7])),
    )
    weight_a, weight_b = (u @ u.T for u in uocc)

    walker_a = rng.normal(size=(nora, noa)) + 1j * rng.normal(size=(nora, noa))
    walker_b = rng.normal(size=(norb, nob)) + 1j * rng.normal(size=(norb, nob))
    walker_a[:noa] += np.eye(noa)
    walker_b[:nob] += np.eye(nob)
    walker = (
        jnp.asarray(walker_a),
        jnp.asarray(np.pad(walker_b, ((0, nora - norb), (0, 0)))),
    )

    trial_input = _make_pt2uccsd_trial(
        (t1a, t1b), (t2aa, t2ab, t2bb), (nora, norb), uocc
    )
    np.testing.assert_array_equal(trial_input.data["t1b"][:, -1], 0.0)
    np.testing.assert_array_equal(trial_input.data["t2ab"][:, :, :, -1], 0.0)
    sys = System(norb=nora, nelec=(noa, nob), walker_kind="unrestricted")
    trial = make_pt2uccsd_trial_data(trial_input.data, sys)
    ham = HamCholUhf(
        h0=jnp.asarray(0.7),
        h1_a=jnp.asarray(h1a),
        h1_b=jnp.asarray(np.pad(h1b, ((0, nora - norb),) * 2)),
        chol_a=jnp.asarray(chola),
        chol_b=jnp.asarray(
            np.pad(cholb, ((0, 0), (0, nora - norb), (0, nora - norb)))
        ),
        norb_spin=(nora, norb),
    )
    ops = make_lno_estimator_ops(sys, weight_a, weight_b)
    context = ops.build_estimator_ctx(ham, trial)

    expected = np.asarray(
        [
            0.04146924344309247 + 0.19937182373717310j,
            -2.1181641754870104 - 0.9226406218735894j,
            0.21281045736794620 - 0.02840671675250106j,
            -1.6174052916504527 - 1.4365559538119710j,
        ]
    )
    np.testing.assert_allclose(
        ops.components(walker, ham, context, trial), expected, rtol=2e-11, atol=2e-11
    )
    np.testing.assert_allclose(
        ops.reference_overlap(walker, trial),
        -1.4851284278662578 + 1.4146050370220455j,
        rtol=2e-12,
        atol=2e-12,
    )


def test_nonlinear_energy_uses_population_means():
    sys = System(norb=2, nelec=(1, 1), walker_kind="unrestricted")
    ops = make_lno_estimator_ops(sys, jnp.eye(1), jnp.eye(1))
    blocks = jnp.asarray([[0.0, 1.0, 4.0, 10.0], [2.0, 3.0, 6.0, 20.0]])

    np.testing.assert_allclose(ops.combine_energy(0.0, blocks.mean(axis=0)), -8.0)
    np.testing.assert_allclose(ops.combine_energy(0.0, blocks).mean(), -13.0)
