"""Exact batched RHF energy, including tails and model/data sharding."""

import os
from dataclasses import replace

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from trot import walkers as wk
from trot.meas.rhf import (
    RhfMeasCfg,
    RhfMeasCtx,
    _two_body_energy_restricted,
    _two_body_energy_unrestricted,
)

jax.config.update("jax_enable_x64", True)


def _ctx(chol, batch_size):
    return RhfMeasCtx(
        jnp.zeros(chol.shape[1:]),
        jnp.asarray(chol),
        jnp.asarray(chol.reshape(len(chol), np.prod(chol.shape[1:]))),
        RhfMeasCfg(memory_mode="low", chol_batch_size=batch_size),
    )


@pytest.mark.parametrize("n_chol,batch_size", [(0, 4), (3, 4), (8, 4), (9, 4), (9, 1), (518, 256)])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_batched_energy_matches_full_cholesky_sum(n_chol, batch_size, unrestricted):
    rng = np.random.default_rng(42)
    # Complex half-rotations and nontrivial Green functions exercise both spins.
    chol = rng.normal(size=(n_chol, 3, 6)) + 1j * rng.normal(size=(n_chol, 3, 6))
    gu, gd = (rng.normal(size=(3, 6)) + 1j * rng.normal(size=(3, 6)) for _ in range(2))
    ctx = _ctx(chol, batch_size)
    high = replace(ctx, cfg=RhfMeasCfg())
    if unrestricted:
        fn = jax.jit(_two_body_energy_unrestricted)
        actual, expected = fn(gu, gd, ctx), fn(gu, gd, high)
    else:
        fn = jax.jit(_two_body_energy_restricted)
        actual, expected = fn(gu, ctx), fn(gu, high)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-10)


@pytest.mark.parametrize("n_data", [4, 2, 1])
@pytest.mark.parametrize("n_chunks", [1, 3])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_batched_energy_keeps_cholesky_and_walkers_distributed(n_data, n_chunks, unrestricted):
    if jax.local_device_count() != 4:
        pytest.skip("requires four logical CPU devices")
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(n_data, 4 // n_data),
        ("data", "model"),
        axis_types=(AxisType.Auto, AxisType.Auto),
    )
    rng = np.random.default_rng(42)
    chol = rng.normal(size=(518, 4, 8))
    greens = tuple(rng.normal(size=(40, 4, 8)) + 1j * rng.normal(size=(40, 4, 8)) for _ in range(2))
    ctx = _ctx(chol, 256)
    kernel = (
        (lambda g, c: _two_body_energy_unrestricted(*g, c))
        if unrestricted
        else (lambda g, c: _two_body_energy_restricted(g[0], c))
    )
    expected = jax.jit(jax.vmap(kernel, in_axes=(0, None)))(greens, replace(ctx, cfg=RhfMeasCfg()))
    n_model = 4 // n_data
    padded = np.pad(chol, ((0, (-len(chol)) % n_model), (0, 0), (0, 0)))
    ctx_s = _ctx(padded, 256)
    ctx_s = replace(
        ctx_s,
        rot_h1=jax.device_put(ctx_s.rot_h1, NamedSharding(mesh, P())),
        rot_chol=jax.device_put(ctx_s.rot_chol, NamedSharding(mesh, P("model"))),
        rot_chol_flat=jax.device_put(ctx_s.rot_chol_flat, NamedSharding(mesh, P("model"))),
    )
    greens_s = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P("data"))), greens)
    fn = jax.jit(wk.vmap_chunked(kernel, n_chunks, in_axes=(0, None)))
    compiled = fn.lower(greens_s, ctx_s).compile()
    actual = compiled(greens_s, ctx_s)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-10)
    assert actual.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), 1)
    assert " all-gather(" not in compiled.as_text()


@pytest.mark.parametrize("size", [0, -1, 1.5])
def test_rejects_invalid_cholesky_batch_size(size):
    with pytest.raises(ValueError, match="positive integer"):
        RhfMeasCfg(memory_mode="low", chol_batch_size=size)


def test_batched_workspace_does_not_materialize_all_cholesky_vectors():
    if jax.local_device_count() != 4:
        pytest.skip("requires four logical CPU devices")
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(4, 1),
        ("data", "model"),
        axis_types=(AxisType.Auto, AxisType.Auto),
    )
    rng = np.random.default_rng(123)
    # C60 Cholesky count, but only 8 orbitals/4 occupied and 10 local walkers.
    ctx = _ctx(rng.normal(size=(5126, 4, 8)), 256)
    ctx = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P())), ctx)
    g = rng.normal(size=(40, 4, 8)) + 1j * rng.normal(size=(40, 4, 8))
    g = jax.device_put(g, NamedSharding(mesh, P("data")))
    fn = jax.jit(wk.vmap_chunked(_two_body_energy_restricted, 1, in_axes=(0, None)))
    low = fn.lower(g, ctx).compile()
    high = fn.lower(g, replace(ctx, cfg=RhfMeasCfg())).compile()
    assert low.memory_analysis().temp_size_in_bytes < (
        0.5 * high.memory_analysis().temp_size_in_bytes
    )
    assert " all-gather(" not in low.as_text()
