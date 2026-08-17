from __future__ import annotations

from functools import partial

from trot import config

config.configure_once()

import jax
import jax.numpy as jnp
import numpy as np

from trot.core.ops import (
    BlockEnergyRetuneResult,
    EstimatorOps,
    MeasOps,
    TrialOps,
    k_energy,
)
from trot.core.system import System
from trot.driver import run_mixed_estimator_qmc
from trot.ham.chol import HamChol
from trot.prop.blocks import BlockObs, block_mixed_estimator
from trot.prop.types import PropOps, PropState, QmcParams
from trot.stat_utils import blocking_analysis_components, component_estimator_outlier_mask


def _guide_overlap(walker, trial_data):
    del trial_data
    return 1.0 + walker[1, 0]


def _reference_overlap(walker, estimator_data):
    del estimator_data
    guide_overlap = 1.0 + walker[1, 0]
    return guide_overlap * (2.0 + walker[1, 0])


def _guide_energy(walker, ham_data, meas_ctx, guide_data):
    del ham_data, meas_ctx, guide_data
    return 5.0 + walker[1, 0]


def _components(walker, ham_data, estimator_ctx, estimator_data):
    del ham_data, estimator_ctx, estimator_data
    marker = walker[1, 0]
    return jnp.stack([10.0 + 2.0 * marker, 4.0 - marker])


def _combine(h0, components):
    return h0 + components[..., 0] * components[..., 1]


def _get_rdm1(trial_data):
    del trial_data
    return jnp.stack([jnp.diag(jnp.asarray([1.0, 0.0]))] * 2)


def _step(state, **kwargs):
    del kwargs
    return state


def _identity_sr(walkers, weights, zeta, walker_kind):
    del zeta, walker_kind
    return walkers, weights


def _outlier_mixed_block(state, **kwargs):
    del kwargs
    block_index = state.node_encounters
    components = jnp.where(
        block_index == 12,
        jnp.asarray([1000.0, 2.0]),
        jnp.asarray([1.0, 2.0]),
    )
    state = state._replace(node_encounters=block_index + 1)
    return state, BlockObs(
        scalars={
            "guide_energy": jnp.asarray(5.75),
            "guide_weight": jnp.asarray(4.0),
            "estimator_weight": jnp.asarray(1.0 + 0.0j),
            "estimator_components": components,
        },
        observables={},
    )


def _make_case(*, n_blocks: int = 25, n_eql_blocks: int = 2):
    sys = System(norb=2, nelec=(1, 1), walker_kind="restricted")
    params = QmcParams(
        dt=0.005,
        n_walkers=2,
        n_prop_steps=1,
        n_eql_blocks=n_eql_blocks,
        n_blocks=n_blocks,
        n_chunks=1,
        shift_ema=0.25,
        error_method="blocking",
        seed=7,
    )
    ham_data = HamChol(
        h0=jnp.asarray(0.5),
        h1=jnp.zeros((2, 2)),
        chol=jnp.zeros((1, 2, 2)),
        basis="restricted",
    )
    walkers = jnp.asarray(
        [
            [[1.0], [0.0]],
            [[0.0], [1.0]],
        ],
        dtype=jnp.complex128,
    )
    weights = jnp.asarray([1.0, 3.0])
    state = PropState(
        walkers=walkers,
        weights=weights,
        overlaps=jnp.asarray([1.0, 2.0], dtype=jnp.complex128),
        rng_key=jax.random.PRNGKey(17),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(5.5),
        node_encounters=jnp.asarray(0),
    )
    guide_ops = TrialOps(overlap=_guide_overlap, get_rdm1=_get_rdm1)
    guide_meas_ops = MeasOps(
        overlap=_guide_overlap,
        kernels={k_energy: _guide_energy},
    )
    guide_prop_ops = PropOps(
        init_prop_state=lambda **kwargs: state,
        build_prop_ctx=lambda ham_data, rdm1, params: None,
        step=_step,
    )
    estimator_ops = EstimatorOps(
        reference_overlap=_reference_overlap,
        components=_components,
        combine_energy=_combine,
        component_names=("left", "right"),
    )
    return (
        sys,
        params,
        ham_data,
        state,
        guide_ops,
        guide_meas_ops,
        guide_prop_ops,
        estimator_ops,
    )


def test_mixed_estimator_block_reweights_reference_independently_of_guide():
    (
        sys,
        params,
        ham_data,
        state,
        guide_ops,
        guide_meas_ops,
        guide_prop_ops,
        estimator_ops,
    ) = _make_case()

    state_new, obs = jax.jit(
        lambda state_i: block_mixed_estimator(
            state_i,
            sys=sys,
            params=params,
            ham_data=ham_data,
            guide_data=jnp.asarray(0.0),
            guide_ops=guide_ops,
            guide_meas_ops=guide_meas_ops,
            guide_meas_ctx=None,
            guide_prop_ops=guide_prop_ops,
            guide_prop_ctx=None,
            estimator_data=jnp.asarray(0.0),
            estimator_ops=estimator_ops,
            estimator_ctx=None,
            sr_fn=_identity_sr,
        )
    )(state)

    estimator_weights = np.asarray([2.0, 9.0])
    component_samples = np.asarray([[10.0, 4.0], [12.0, 3.0]])
    expected_components = np.sum(
        estimator_weights[:, None] * component_samples,
        axis=0,
    ) / np.sum(estimator_weights)

    np.testing.assert_allclose(obs.scalars["guide_energy"], 5.75)
    np.testing.assert_allclose(obs.scalars["guide_weight"], 4.0)
    np.testing.assert_allclose(obs.scalars["estimator_weight"], 11.0)
    np.testing.assert_allclose(obs.scalars["estimator_components"], expected_components)
    np.testing.assert_allclose(state_new.weights, state.weights)


def test_component_blocking_preserves_nonlinear_component_covariance():
    weights = np.linspace(0.7, 1.3, 40)
    x = np.linspace(-0.2, 0.2, 40)
    components = np.column_stack((1.5 + x, 2.0 - 0.5 * x))
    result = blocking_analysis_components(
        0.25,
        weights,
        components,
        _combine,
        print_q=False,
    )

    expected_components = np.sum(weights[:, None] * components, axis=0) / np.sum(weights)
    expected_energy = _combine(0.25, expected_components)
    np.testing.assert_allclose(result["mean_components"], expected_components)
    np.testing.assert_allclose(result["mu"], expected_energy)
    assert result["se_star"] is not None
    assert np.isfinite(result["se_star"])


def test_component_estimator_outlier_mask_matches_historical_zeta_rule():
    weights = np.ones(25)
    components = np.tile(np.asarray([1.0, 2.0]), (25, 1))
    components[12, 0] = 1000.0

    proxy_energies, keep = component_estimator_outlier_mask(
        0.5,
        weights,
        components,
        _combine,
        zeta=20.0,
    )

    np.testing.assert_allclose(proxy_energies[:12], 2.5)
    np.testing.assert_allclose(proxy_energies[12], 2000.5)
    assert np.count_nonzero(keep) == 24
    assert not keep[12]


def test_generic_mixed_estimator_driver_returns_named_components(capsys):
    (
        sys,
        params,
        ham_data,
        state,
        guide_ops,
        guide_meas_ops,
        guide_prop_ops,
        estimator_ops,
    ) = _make_case(n_eql_blocks=10)
    result = run_mixed_estimator_qmc(
        sys=sys,
        params=params,
        ham_data=ham_data,
        guide_data=jnp.asarray(0.0),
        guide_ops=guide_ops,
        guide_prop_ops=guide_prop_ops,
        guide_meas_ops=guide_meas_ops,
        estimator_data=jnp.asarray(0.0),
        estimator_ops=estimator_ops,
        mixed_block_fn=partial(block_mixed_estimator, sr_fn=_identity_sr),
        state=state,
        guide_meas_ctx=jnp.asarray(0.0),
        guide_prop_ctx=jnp.asarray(0.0),
        estimator_ctx=jnp.asarray(0.0),
    )

    expected_components = np.asarray([128.0 / 11.0, 35.0 / 11.0])
    expected_energy = _combine(ham_data.h0, expected_components)
    assert result.estimator_component_names == ("left", "right")
    assert result.guide_block_energies.shape == (params.n_blocks,)
    assert result.estimator_block_components.shape == (params.n_blocks, 2)
    assert result.estimator_block_proxy_energies.shape == (params.n_blocks,)
    assert np.all(result.estimator_block_keep_mask)
    assert result.estimator_analysis["n_rejected_blocks"] == 0
    np.testing.assert_allclose(result.guide_mean_energy, 5.75)
    np.testing.assert_allclose(result.estimator_mean_components, expected_components)
    np.testing.assert_allclose(result.estimator_mean_energy, expected_energy)
    output = capsys.readouterr().out
    assert "Mixed-estimator equilibration:" in output
    assert "Guide_E_blk" in output
    assert "Estimator_E_blk" in output
    assert "[eql    2/10]" in output
    assert "[eql   10/10]" in output
    assert "Mixed-estimator sampling:" in output
    assert "Guide_E_avg" in output
    assert "Estimator_E_avg" in output
    assert "[blk    2/25]" in output
    assert "[blk   25/25]" in output


def test_generic_mixed_estimator_driver_cleans_outlier_and_preserves_raw_blocks(capsys):
    (
        sys,
        params,
        ham_data,
        state,
        guide_ops,
        guide_meas_ops,
        guide_prop_ops,
        estimator_ops,
    ) = _make_case(n_blocks=25, n_eql_blocks=0)
    result = run_mixed_estimator_qmc(
        sys=sys,
        params=params,
        ham_data=ham_data,
        guide_data=jnp.asarray(0.0),
        guide_ops=guide_ops,
        guide_prop_ops=guide_prop_ops,
        guide_meas_ops=guide_meas_ops,
        estimator_data=jnp.asarray(0.0),
        estimator_ops=estimator_ops,
        mixed_block_fn=_outlier_mixed_block,
        state=state,
        guide_meas_ctx=jnp.asarray(0.0),
        guide_prop_ctx=jnp.asarray(0.0),
        estimator_ctx=jnp.asarray(0.0),
    )

    assert result.estimator_block_components.shape == (25, 2)
    np.testing.assert_allclose(result.estimator_block_components[12], [1000.0, 2.0])
    np.testing.assert_allclose(result.estimator_block_proxy_energies[12], 2000.5)
    assert np.count_nonzero(result.estimator_block_keep_mask) == 24
    assert not result.estimator_block_keep_mask[12]
    np.testing.assert_allclose(result.estimator_mean_components, [1.0, 2.0])
    np.testing.assert_allclose(result.estimator_mean_energy, 2.5)
    assert result.estimator_analysis["n_raw_blocks"] == 25
    assert result.estimator_analysis["n_retained_blocks"] == 24
    assert result.estimator_analysis["n_rejected_blocks"] == 1
    assert "rejected 1/25 projected-estimator blocks" in capsys.readouterr().out


def test_mixed_estimator_retuning_advance_uses_guide_scalar_contract(capsys):
    (
        sys,
        params,
        ham_data,
        state,
        guide_ops,
        guide_meas_ops,
        guide_prop_ops,
        estimator_ops,
    ) = _make_case()
    retune_calls = []

    def retune(
        state_i,
        equilibration_energies,
        equilibration_weights,
        params_i,
        ham_data_i,
        meas_ctx_i,
        guide_data_i,
        *,
        advance_blocks,
        target_error=None,
    ):
        del params_i, ham_data_i, guide_data_i, target_error
        state_n, scalars, observables = advance_blocks(state_i, n_blocks=1)
        assert set(scalars) == {"energy", "weight"}
        np.testing.assert_allclose(scalars["energy"], 5.75)
        np.testing.assert_allclose(scalars["weight"], 4.0)
        assert observables == ()
        retune_calls.append(
            (
                np.asarray(equilibration_energies),
                np.asarray(equilibration_weights),
            )
        )
        return BlockEnergyRetuneResult(
            state=state_n,
            meas_ctx=meas_ctx_i,
            initial_n_chunks=1,
            settling_blocks=2,
        )

    guide_meas_ops = MeasOps(
        overlap=guide_meas_ops.overlap,
        kernels=guide_meas_ops.kernels,
        retune_block_energy=retune,
    )
    run_mixed_estimator_qmc(
        sys=sys,
        params=params,
        ham_data=ham_data,
        guide_data=jnp.asarray(0.0),
        guide_ops=guide_ops,
        guide_prop_ops=guide_prop_ops,
        guide_meas_ops=guide_meas_ops,
        estimator_data=jnp.asarray(0.0),
        estimator_ops=estimator_ops,
        mixed_block_fn=partial(block_mixed_estimator, sr_fn=_identity_sr),
        state=state,
        guide_meas_ctx=jnp.asarray(0.0),
        guide_prop_ctx=jnp.asarray(0.0),
        estimator_ctx=jnp.asarray(0.0),
    )

    assert len(retune_calls) == 1
    np.testing.assert_allclose(retune_calls[0][0], 5.75)
    np.testing.assert_allclose(retune_calls[0][1], 4.0)
    output = capsys.readouterr().out
    assert "Post-tuning settling: 2 blocks" in output
    assert "[settle    1/2]" in output
    assert "[settle    2/2]" in output
