"""Chunking must retain data parallelism, including with a model axis."""

import os
from dataclasses import replace
from functools import partial

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from trot import walkers as wk
from trot.meas.rhf import RhfMeasCfg, RhfMeasCtx, _two_body_energy_restricted

jax.config.update("jax_enable_x64", True)


def _mesh(n_data):
    if jax.local_device_count() != 4:
        pytest.skip("requires four logical CPU devices")
    return Mesh(
        np.asarray(jax.devices()).reshape(n_data, 4 // n_data),
        ("data", "model"),
        axis_types=(AxisType.Auto, AxisType.Auto),
    )


@pytest.mark.parametrize("n_data", [4, 2, 1])
@pytest.mark.parametrize("n_chunks", [2, 4, 14])
def test_rhf_energy_chunks_preserve_sharding(n_data, n_chunks):
    mesh = _mesh(n_data)
    rng = np.random.default_rng(42)
    # Same global walker count as C60, but only a few MiB of workspace.
    g = rng.normal(size=(400, 4, 8)) + 1j * rng.normal(size=(400, 4, 8))
    chol = rng.normal(size=(32, 4, 8))
    ctx = RhfMeasCtx(
        jnp.zeros((4, 8)), jnp.asarray(chol), jnp.asarray(chol.reshape(32, -1)), RhfMeasCfg()
    )
    expected = jax.jit(jax.vmap(_two_body_energy_restricted, in_axes=(0, None)))(g, ctx)
    g_s = jax.device_put(g, NamedSharding(mesh, P("data")))
    ctx_s = RhfMeasCtx(
        jax.device_put(ctx.rot_h1, NamedSharding(mesh, P())),
        jax.device_put(ctx.rot_chol, NamedSharding(mesh, P("model"))),
        jax.device_put(ctx.rot_chol_flat, NamedSharding(mesh, P("model"))),
        ctx.cfg,
    )
    fn = jax.jit(wk.vmap_chunked(_two_body_energy_restricted, n_chunks, in_axes=(0, None)))
    compiled = fn.lower(g_s, ctx_s).compile()
    actual = compiled(g_s, ctx_s)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-10)
    assert actual.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), 1)
    # No walker or full-Hamiltonian all-gathers belong in this kernel.
    assert " all-gather(" not in compiled.as_text()
    if n_data == 4 and n_chunks == 4:
        unchunked = (
            jax.jit(wk.vmap_chunked(_two_body_energy_restricted, 1, in_axes=(0, None)))
            .lower(g_s, ctx_s)
            .compile()
        )
        assert compiled.memory_analysis().temp_size_in_bytes < (
            0.5 * unchunked.memory_analysis().temp_size_in_bytes
        )


@pytest.mark.parametrize("n_data", [4, 2])
@pytest.mark.parametrize("n_chunks", [3, 40])
def test_chunks_support_tuple_walkers_fields_and_static_options(n_data, n_chunks):
    mesh = _mesh(n_data)
    rng = np.random.default_rng(24)
    up, down = rng.normal(size=(40, 8, 3)), rng.normal(size=(40, 8, 2))
    fields = rng.normal(size=(40, 32))
    chol = rng.normal(size=(32, 8, 8))

    def kernel(w, field, chol, n_terms, *, scale):
        v = jnp.einsum("g,gij->ij", field, chol)
        u, d = w
        for _ in range(n_terms):
            u, d = u + scale * (v @ u), d + scale * (v @ d)
        return {"walkers": (u, d), "norm": jnp.sum(u * u) + jnp.sum(d * d)}

    args = ((up, down), fields, chol, 2)
    expected = jax.vmap(lambda w, f: kernel(w, f, chol, 2, scale=0.01))(*args[:2])
    args_s = (
        tuple(jax.device_put(x, NamedSharding(mesh, P("data"))) for x in (up, down)),
        jax.device_put(fields, NamedSharding(mesh, P("data", "model"))),
        jax.device_put(chol, NamedSharding(mesh, P("model"))),
    )
    mapped = wk.vmap_chunked(kernel, n_chunks, in_axes=(0, 0, None, None))
    fn = jax.jit(lambda w, f, c: mapped(w, f, c, 2, scale=0.01))
    compiled = fn.lower(*args_s).compile()
    actual = compiled(*args_s)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)
        assert a.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), a.ndim)
    assert " all-gather(" not in compiled.as_text()


def test_shared_indices_keep_generic_chunking():
    mesh = _mesh(4)
    indices = jax.device_put(np.arange(32), NamedSharding(mesh, P()))
    fn = jax.jit(wk.vmap_chunked(lambda i: i * i, 4, shard_walkers=False))
    actual = fn(indices)
    np.testing.assert_array_equal(actual, np.arange(32) ** 2)
    assert actual.is_fully_replicated


def test_plain_data_mesh_supports_eager_chunking():
    mesh = Mesh(_mesh(4).devices.reshape(4), ("data",), axis_types=(AxisType.Auto,))
    x = jax.device_put(np.arange(24).reshape(12, 2), NamedSharding(mesh, P("data")))
    actual = wk.vmap_chunked(lambda row: row * row, 2)(x)
    np.testing.assert_array_equal(actual, np.asarray(x) ** 2)
    assert actual.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), 2)


@pytest.mark.parametrize("n_data", [4, 2])
@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_chunked_rhf_blocks_match_single_device(n_data, mixed_precision, memory_mode):
    """Functional regression, including global population control and RNG."""
    from trot.driver import make_run_blocks
    from trot.core.system import System
    from trot.meas.rhf import make_rhf_meas_ops
    from trot.prop.blocks import block
    from trot.prop.types import QmcParams
    from trot.setup import setup
    from trot.staging import HamInput, StagedInputs, TrialInput

    mesh = _mesh(n_data)
    rng = np.random.default_rng(89)
    chol = 0.01 * rng.normal(size=(32, 8, 8))
    chol += chol.transpose(0, 2, 1)
    staged = StagedInputs(
        ham=HamInput(
            h0=0.2,
            h1=np.diag(np.linspace(-1, 1, 8)),
            chol=chol,
            nelec=(4, 4),
            norb=8,
            chol_cut=1e-5,
            frozen=0,
            source_kind="mf",
            basis="restricted",
        ),
        trial=TrialInput(kind="rhf", data={"mo": np.eye(8)}, frozen=0, source_kind="mf"),
        meta={},
    )
    params = QmcParams(n_walkers=40, n_chunks=3, n_prop_steps=2, seed=34)
    results = []
    for devices in (None, mesh):
        job = setup(
            staged,
            mesh=devices,
            walker_kind="restricted",
            mixed_precision=mixed_precision,
            params=replace(params, n_chunks=1) if devices is None else params,
            meas_ops=make_rhf_meas_ops(
                System(norb=8, nelec=(4, 4), walker_kind="restricted"),
                memory_mode="high" if devices is None else memory_mode,
                chol_batch_size=3,
            ),
        )
        state, meas_ctx, prop_ctx = job._prepare_runtime()
        assert meas_ctx.cfg.chol_batch_size == 3
        assert meas_ctx.cfg.memory_mode == ("high" if devices is None else memory_mode)
        assert wk.n_local_walkers(state.walkers) == (40 if devices is None else 40 // n_data)
        sr = (
            wk.stochastic_reconfiguration
            if devices is None
            else partial(
                wk.stochastic_reconfiguration, data_sharding=NamedSharding(mesh, P("data"))
            )
        )
        run = make_run_blocks(
            block_fn=partial(block, sr_fn=sr),
            sys=job.sys,
            params=job.params,
            trial_ops=job.trial_ops,
            meas_ops=job.meas_ops,
            prop_ops=job.prop_ops,
            observable_names=("rdm1",),
        )
        result = run(
            state,
            ham_data=job.ham_data,
            trial_data=job.trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=3,
        )
        jax.block_until_ready(result)
        results.append(result)
    tolerance = 3e-7 if mixed_precision else 3e-11
    for a, b in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1])):
        np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)
    final_state = results[1][0]
    for a in (final_state.walkers, final_state.weights, final_state.overlaps):
        assert a.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), a.ndim)
