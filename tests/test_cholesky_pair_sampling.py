"""Host layouts and synthetic-table estimator checks; no CPU AFQMC kernels."""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from dataclasses import dataclass
from functools import partial
import re
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, PartitionSpec as P

from trot.ham.chol import HamChol
from trot.meas.cisd_modes import CisdModePairSamplingCfg
from trot.meas.pair_sampling import (
    local_cholesky_head, local_cholesky_tail, with_local_cholesky_sampling,
)
from trot.sharding import plan_cholesky_layout, replicate, shard_cholesky_layout

jax.config.update("jax_enable_x64", True)


def model_mesh(n=4):
    if jax.local_device_count() < n:
        pytest.skip(f"Requires {n} devices")
    return Mesh(np.asarray(jax.local_devices()[:n]).reshape(1, n), ("data", "model"),
                axis_types=(AxisType.Auto, AxisType.Auto))


@pytest.mark.parametrize("n,d,h", [(17, 4, 5), (16, 4, 0), (9, 4, 9), (2, 4, 1), (7, 1, 3)])
def test_layout_membership_and_placement(n, d, h):
    head = np.arange(n)[::-1][:h]
    tail = np.setdiff1d(np.arange(n), head)
    p = np.arange(1, tail.size + 1, dtype=float)
    p = p / p.sum() if p.size else p
    layout = plan_cholesky_layout(n, d, head, tail, p)
    perm = layout.permutation
    np.testing.assert_array_equal(perm[layout.inverse_permutation], np.arange(n))
    np.testing.assert_array_equal(perm[layout.head_indices], head)
    np.testing.assert_array_equal(perm[layout.tail_indices], tail)
    np.testing.assert_allclose(layout.local_tail_prob.ravel()[layout.tail_indices], p)
    assert np.all(layout.local_tail_prob.ravel()[perm < 0] == 0)
    assert np.ptp(layout.local_head_valid.sum(axis=1)) <= 1
    x = np.arange(n * 6).reshape(n, 2, 3).astype(float)
    before = x.copy()
    placed = shard_cholesky_layout(x, model_mesh(d), layout, dtype=np.float32)
    expected = np.zeros((perm.size, 2, 3))
    expected[perm >= 0] = x[perm[perm >= 0]]
    np.testing.assert_array_equal(placed, expected)
    np.testing.assert_array_equal(x, before)
    assert placed.dtype == jnp.float32


def test_balanced_skewed_proposal_is_deterministic():
    p = np.exp(-np.linspace(0, 12, 2538)); p /= p.sum()
    args = (2688, 4, np.arange(150), np.arange(150, 2688), p)
    a, b = plan_cholesky_layout(*args), plan_cholesky_layout(*args)
    np.testing.assert_array_equal(a.permutation, b.permutation)
    np.testing.assert_allclose(a.local_tail_prob.sum(axis=1), .25, atol=1e-5)
    np.testing.assert_array_equal(a.local_head_valid.sum(axis=1), [38, 38, 37, 37])


@pytest.mark.parametrize("head,tail,p", [([0], [0, 2], [.5, .5]),
    ([0], [1], [1.]), ([0], [1, 2], [1., 0.]), ([0], [1, 2], [.2, .3]),
    ([0.5], [1, 2], [.5, .5]), ([0], [1, 2], [np.nan, .5])])
def test_invalid_layout_rejected(head, tail, p):
    with pytest.raises(ValueError):
        plan_cholesky_layout(3, 4, head, tail, p)


@partial(jax.tree_util.register_dataclass,
         data_fields=["chol_head_indices", "chol_tail_indices", "chol_tail_prob", "model_sampling"],
         meta_fields=["energy_sampling"])
@dataclass(frozen=True)
class TableContext:
    chol_head_indices: object
    chol_tail_indices: object
    chol_tail_prob: object
    energy_sampling: object
    model_sampling: object = None


class Common(NamedTuple):
    base: object


def context_specs(ctx):
    return jax.tree.map(lambda _: P(), ctx)


def head_terms(common, indices, h, ctx, trial, *, n_chunks):
    return h.chol[indices, :, 0].T


def pair_terms(common, iw, ig, h, ctx, trial, *, n_chunks):
    return h.chol[ig, iw, 0]


def table_inputs(n, head, *, budget=17, diagnostics=True):
    mesh = model_mesh()
    table = np.sin(np.arange(n * 8).reshape(n, 8)) + .3j
    tail = np.setdiff1d(np.arange(n), head)
    p = np.arange(1, len(tail) + 1, dtype=float)
    p = p / p.sum() if p.size else p
    layout = plan_cholesky_layout(n, 4, head, tail, p)
    cfg = CisdModePairSamplingCfg(len(head), budget, head_chol_batch_size=2,
                                  track_half_sample_diagnostic=diagnostics)
    ctx = with_local_cholesky_sampling(TableContext(None, None, None, cfg), layout, mesh)
    zero = replicate(np.asarray(0.), mesh)
    h = HamChol(zero, replicate(np.zeros((8, 8)), mesh),
                shard_cholesky_layout(table[:, :, None], mesh, layout))
    common = Common(replicate(np.zeros(8, dtype=complex), mesh))
    return table, layout, common, h, ctx, zero


@pytest.mark.parametrize("n,head", [(17, [0, 3, 4, 5, 10]), (9, list(range(9))), (7, []), (2, [1])])
def test_local_head_complex_sum_squares_and_communication(n, head):
    table, layout, common, h, ctx, t = table_inputs(n, head)
    f = jax.jit(lambda common, h, c, t: local_cholesky_head(
        common, h, c, t, terms_fn=head_terms, context_specs_fn=context_specs, n_chunks=3))
    compiled = f.lower(common, h, ctx, t).compile()
    actual, squared = compiled(common, h, ctx, t)
    np.testing.assert_allclose(actual, table[head].sum(axis=0), atol=1e-14)
    np.testing.assert_allclose(squared, (table[head].real**2).sum(axis=0), atol=1e-14)
    assert_small_collectives(compiled, 16)


def assert_small_collectives(compiled, limit=8):
    for line in compiled.as_text().splitlines():
        if re.search(r"\b(all-gather|all-to-all|collective-permute)(-start|-done)?\(", line):
            pytest.fail(f"Unexpected data movement: {line}")
        if re.search(r"\b(all-reduce|reduce-scatter)(-start|-done)?\(", line):
            shapes = re.findall(r"(?:f|s|u|c)\d+\[([\d,]*)\]", line.split("metadata=")[0])
            assert all(np.prod([int(d) for d in s.split(',') if d]) <= limit for s in shapes), line


@pytest.mark.parametrize("n,head,zero_weight", [(17, [0, 3, 4, 5, 10], False),
    (3, [1], False), (9, list(range(9)), False), (7, [0], True)])
def test_tail_mean_variance_noise_and_empty_strata(n, head, zero_weight):
    table, layout, common, h, ctx, t = table_inputs(n, head)
    pi = np.arange(1, 9, dtype=float); pi /= pi.sum()
    pi[1::3] = 0  # Guarded walkers keep zero weight, with no renormalization.
    q = pi * np.array([1, 4, 2, 1, 5, 2, 3, 1]); q /= q.sum()
    if zero_weight:
        pi[:] = 0
    args = (common, replicate(pi, ctx.model_sampling.mesh), replicate(q, ctx.model_sampling.mesh),
            jax.random.PRNGKey(42), h, ctx, t)
    f = jax.jit(lambda common, pi, q, key, h, c, t: jnp.stack(local_cholesky_tail(
        common, pi, q, key, h, c, t, pair_fn=pair_terms, context_specs_fn=context_specs, n_chunks=3)))
    compiled = f.lower(*args).compile()
    assert_small_collectives(compiled)
    # Stream keys through scan so the collective buffers stay scalar.
    def sample_many(args, keys):
        def step(_, key):
            return None, f(*args[:3], key, *args[4:])
        return jax.lax.scan(step, None, keys)[1]
    samples = np.asarray(jax.jit(sample_many)(args, jax.random.split(jax.random.PRNGKey(93), 8192)))
    exact, variance = 0., 0.
    original = layout.permutation.reshape(4, -1)
    for d, pd in enumerate(layout.local_tail_prob):
        if pd.sum() == 0 or pi.sum() == 0:
            continue
        ids = original[d, pd > 0]
        probability = (pd[pd > 0] / pd.sum())[:, None] * q[None, q > 0]
        values = table[ids][:, q > 0].real * pi[None, q > 0] / probability
        mean = np.sum(probability * values)
        exact += mean
        variance += (np.sum(probability * values**2) - mean**2) / (5 if d == 0 else 4)
    if variance == 0:
        np.testing.assert_array_equal(samples, 0)
    else:
        assert abs(samples[:, 0].mean() - exact) < 6 * np.sqrt(variance / len(samples))
        assert abs(samples[:, 1].mean()) < 6 * np.sqrt(variance / len(samples))
        np.testing.assert_allclose(samples.var(axis=0, ddof=1), variance, rtol=.07)


def test_insufficient_pair_budget_rejected():
    with pytest.raises(ValueError, match="at least 2 pairs"):
        table_inputs(7, [0], budget=7)
