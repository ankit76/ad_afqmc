from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from trot import driver
from trot.core.ops import (
    BlockEnergyEstimate,
    BlockEnergyRetuneResult,
    MeasOps,
    TrialOps,
    d_energy_sampling_noise,
    k_energy,
)
from trot.core.system import System
from trot.prop.blocks import block
from trot.prop.types import PropOps, PropState, QmcParams


def _overlap(walker, trial_data):
    del walker, trial_data
    return jnp.asarray(1.0, dtype=jnp.complex128)


def _get_rdm1(trial_data):
    del trial_data
    return jnp.eye(2, dtype=jnp.float64)


def _step(state, **kwargs):
    del kwargs
    return state


def _build_prop_ctx(ham_data, rdm1, params):
    del ham_data, rdm1, params
    return None


def _init_prop_state(**kwargs):
    raise AssertionError(f"unused init_prop_state called with {kwargs}")


def _make_inputs(meas_ops: MeasOps):
    sys = System(norb=2, nelec=(1, 1), walker_kind="restricted")
    params = QmcParams(
        dt=0.005,
        n_walkers=2,
        n_prop_steps=1,
        n_chunks=1,
        shift_ema=0.25,
        seed=0,
    )
    walkers = jnp.asarray(
        [
            [[1.0], [0.0]],
            [[1.0], [0.0]],
        ],
        dtype=jnp.complex128,
    )
    rng_key = jax.random.PRNGKey(17)
    state = PropState(
        walkers=walkers,
        weights=jnp.asarray([1.0, 3.0], dtype=jnp.float64),
        overlaps=jnp.ones(2, dtype=jnp.complex128),
        rng_key=rng_key,
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )
    trial_ops = TrialOps(overlap=_overlap, get_rdm1=_get_rdm1)
    prop_ops = PropOps(
        init_prop_state=_init_prop_state,
        build_prop_ctx=_build_prop_ctx,
        step=_step,
    )
    return sys, params, state, trial_ops, prop_ops, meas_ops


def _run_block(meas_ops: MeasOps, sr_fn):
    sys, params, state, trial_ops, prop_ops, meas_ops = _make_inputs(meas_ops)
    state_new, obs = jax.jit(
        lambda state_i: block(
            state_i,
            sys=sys,
            params=params,
            ham_data=jnp.asarray(2.0),
            trial_data=jnp.asarray(4.0),
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            meas_ctx=jnp.asarray(3.0),
            prop_ops=prop_ops,
            prop_ctx=None,
            sr_fn=sr_fn,
        )
    )(state)
    return params, state, state_new, obs


def test_standard_energy_path_preserves_existing_rng_split():
    def energy(walker, ham_data, meas_ctx, trial_data):
        del walker, ham_data, meas_ctx, trial_data
        return jnp.asarray(5.0)

    def sr_fn(walkers, weights, zeta, walker_kind):
        del walker_kind
        return walkers, weights + zeta

    meas_ops = MeasOps(
        overlap=_overlap,
        kernels={k_energy: energy},
    )
    params, state, state_new, obs = _run_block(meas_ops, sr_fn)

    expected_key, key_sr = jax.random.split(state.rng_key)
    expected_zeta = jax.random.uniform(key_sr)
    np.testing.assert_array_equal(state_new.rng_key, expected_key)
    np.testing.assert_allclose(state_new.weights, state.weights + expected_zeta)
    np.testing.assert_allclose(obs.scalars["energy"], 5.0)
    np.testing.assert_allclose(obs.scalars["weight"], jnp.sum(state.weights))
    np.testing.assert_allclose(state_new.e_estimate, params.shift_ema * 5.0)


def test_block_energy_hook_gets_dedicated_key_and_needs_no_energy_kernel():
    def block_energy(
        walkers,
        weights,
        overlaps,
        rng_key,
        n_chunks,
        ham_data,
        meas_ctx,
        trial_data,
        e_ref,
        energy_clip_threshold,
    ):
        assert n_chunks == 1
        walker_mean = jnp.sum(weights * jnp.real(walkers[:, 0, 0])) / jnp.sum(weights)
        overlap_check = jnp.sum(jnp.real(overlaps)) - overlaps.shape[0]
        return (
            walker_mean
            + overlap_check
            + ham_data
            + meas_ctx
            + trial_data
            + 0.0 * e_ref
            + 0.0 * energy_clip_threshold
            + jax.random.uniform(rng_key)
        )

    def sr_fn(walkers, weights, zeta, walker_kind):
        del walker_kind
        return walkers, weights + zeta

    meas_ops = MeasOps(
        overlap=_overlap,
        block_energy=block_energy,
    )
    params, state, state_new, obs = _run_block(meas_ops, sr_fn)

    expected_key, key_energy, key_sr = jax.random.split(state.rng_key, 3)
    expected_energy = 1.0 + 2.0 + 3.0 + 4.0 + jax.random.uniform(key_energy)
    expected_zeta = jax.random.uniform(key_sr)
    np.testing.assert_array_equal(state_new.rng_key, expected_key)
    np.testing.assert_allclose(state_new.weights, state.weights + expected_zeta)
    np.testing.assert_allclose(obs.scalars["energy"], expected_energy)
    np.testing.assert_allclose(obs.scalars["weight"], jnp.sum(state.weights))
    np.testing.assert_allclose(
        state_new.e_estimate,
        params.shift_ema * expected_energy,
    )


def test_block_energy_diagnostics_are_added_to_block_scalars():
    def block_energy(*args, **kwargs):
        del args, kwargs
        return BlockEnergyEstimate(
            energy=jnp.asarray(3.0),
            diagnostics={d_energy_sampling_noise: jnp.asarray(0.125)},
        )

    meas_ops = MeasOps(overlap=_overlap, block_energy=block_energy)
    _, _, _, obs = _run_block(
        meas_ops,
        lambda walkers, weights, zeta, walker_kind: (
            walkers,
            weights,
        ),
    )
    np.testing.assert_allclose(obs.scalars["energy"], 3.0)
    np.testing.assert_allclose(obs.scalars[d_energy_sampling_noise], 0.125)


def test_driver_rebuilds_blocks_after_post_equilibration_retune(monkeypatch):
    selector_calls = []

    def fake_make_run_blocks_with_auto_chunks(**kwargs):
        params_i = kwargs["params"]
        meas_ctx_i = kwargs["meas_ctx"]
        selector_calls.append((params_i.n_chunks, float(meas_ctx_i)))

        def run_blocks(state, *, n_blocks, **run_kwargs):
            del run_kwargs
            scalars = {
                "energy": jnp.full((n_blocks,), meas_ctx_i, dtype=jnp.float64),
                "weight": jnp.ones((n_blocks,), dtype=jnp.float64),
            }
            if float(meas_ctx_i) == 2.0:
                scalars[d_energy_sampling_noise] = jnp.full(
                    (n_blocks,),
                    0.002,
                    dtype=jnp.float64,
                )
            return state, scalars, ()

        return params_i, run_blocks

    monkeypatch.setattr(
        driver,
        "_make_run_blocks_with_auto_chunks",
        fake_make_run_blocks_with_auto_chunks,
    )

    retune_calls = []

    def retune(
        state,
        equilibration_energies,
        equilibration_weights,
        params,
        ham_data,
        meas_ctx,
        trial_data,
        *,
        advance_blocks,
        target_error=None,
    ):
        del params, ham_data, trial_data
        assert callable(advance_blocks)
        retune_calls.append(
            (
                np.asarray(equilibration_energies),
                np.asarray(equilibration_weights),
                float(meas_ctx),
                target_error,
            )
        )
        return BlockEnergyRetuneResult(
            state=state._replace(e_estimate=jnp.asarray(2.0)),
            meas_ctx=jnp.asarray(2.0),
            initial_n_chunks=3,
            settling_blocks=2,
        )

    meas_ops = MeasOps(
        overlap=_overlap,
        block_energy=lambda *args, **kwargs: jnp.asarray(0.0),
        retune_block_energy=retune,
    )
    sys, _, state, trial_ops, prop_ops, _ = _make_inputs(meas_ops)
    params = QmcParams(
        dt=0.005,
        n_walkers=2,
        n_prop_steps=1,
        n_eql_blocks=2,
        n_blocks=20,
        n_chunks=1,
        seed=0,
    )
    result = driver.run_qmc(
        sys=sys,
        params=params,
        ham_data=jnp.asarray(0.0),
        trial_data=jnp.asarray(0.0),
        meas_ops=meas_ops,
        trial_ops=trial_ops,
        prop_ops=prop_ops,
        block_fn=block,
        state=state,
        meas_ctx=jnp.asarray(1.0),
        prop_ctx=None,
        target_error=0.0,
    )

    assert selector_calls == [(1, 1.0), (3, 2.0)]
    assert len(retune_calls) == 1
    np.testing.assert_allclose(retune_calls[0][0], np.ones(2))
    np.testing.assert_allclose(retune_calls[0][1], np.ones(2))
    assert retune_calls[0][3] == 0.0
    np.testing.assert_allclose(result.block_energies[:5], np.asarray([0.0, 1.0, 1.0, 2.0, 2.0]))
    np.testing.assert_allclose(result.block_energies[5:], 2.0)
    np.testing.assert_allclose(result.block_diagnostics[d_energy_sampling_noise], 0.002)
    assert result.error_method == "gamma"
    assert result.stderr_energy == result.stderr_gamma
    assert result.stderr_blocking is not None
    assert result.blocking_selection_reason in {
        "plateau",
        "near_maximum_fallback",
        "insufficient_blocks",
        "nonfinite_se_curve",
    }
