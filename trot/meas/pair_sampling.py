"""Sampling helpers for replicated Hamiltonians and distributed walkers."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from .. import walkers as wk


def _sample_indices(key, probabilities, size):
    """Inverse-CDF draws with a local binary search, including zero-probability bins.

    JAX 0.10's searchsorted-based choice has incomplete support for Manual
    mesh values. Expressing the search with ordinary primitives also avoids
    materializing a (sample count, category count) categorical-noise array.
    """
    cdf = jnp.cumsum(probabilities, dtype=jnp.float64)
    targets = jax.random.uniform(key, (size,), dtype=jnp.float64) * cdf[-1]
    targets = jnp.minimum(targets, jnp.nextafter(cdf[-1], jnp.asarray(0., dtype=cdf.dtype)))

    def search(_, bounds):
        lo, hi = bounds
        mid = (lo + hi) // 2
        right = cdf[jnp.minimum(mid, cdf.shape[0] - 1)] <= targets
        return jnp.where(right, mid + 1, lo), jnp.where(right, hi, mid)

    lo = jnp.zeros_like(targets, dtype=jnp.int32)
    hi = lo + cdf.shape[0]
    lo, _ = lax.fori_loop(0, int(cdf.shape[0]).bit_length(), search, (lo, hi))
    return jnp.minimum(lo, cdf.shape[0] - 1)


def local_pair_mesh(walkers, enabled):
    """Return the data mesh for the opt-in local sampler, or the serial path.

    This first implementation requires a replicated Hamiltonian. It must not
    silently replicate model-sharded inputs just to make a local gather work.
    """
    if not enabled:
        return None
    mesh = wk._walker_data_mesh(walkers)
    if mesh is not None and mesh.shape.get("model", 1) != 1:
        raise ValueError("Local walker pair sampling requires a replicated Hamiltonian (model size 1).")
    return mesh


def local_common_and_head(
    mesh, walkers, ham_data, meas_ctx, trial_data, *, common_fn, moments_fn, n_chunks,
):
    """Keep walker contractions local, passing all array inputs into the map.

    Explicit arguments avoid capturing Auto-mesh arrays in the manual data
    map, including in nested walker, Cholesky, and mode chunk loops.
    """
    def evaluate(w, h, c, t):
        common = wk.vmap_chunked(common_fn, n_chunks, in_axes=(0, None, None, None))(w, h, c, t)
        sampling = c.energy_sampling
        head, squared = moments_fn(
            common, c.chol_head_indices, h, c, t,
            n_walker_chunks=n_chunks, chol_batch_size=sampling.head_chol_batch_size,
            compute_squared_norm=sampling.walker_guide_policy == "head_rms",
        )
        return common, head, squared

    return jax.shard_map(
        evaluate, mesh=mesh, axis_names={"data"},
        in_specs=(P("data"), P(), P(), P()), out_specs=(P("data"), P("data"), P("data")),
    )(walkers, ham_data, meas_ctx, trial_data)


def local_pair_tail(
    mesh, common, accepted_weights, walker_probabilities, rng_key,
    ham_data, meas_ctx, trial_data, *, pair_fn, n_chunks,
):
    """Sum unbiased stratum estimates, communicating only energy/noise scalars.

    ``accepted_weights`` are global normalized weights, zero for guarded
    walkers. Condition the existing global proposal q on each shard d, then
    average pi[w] * term[w,g] / (q_d[w] * p[g]) over its S_d samples and sum
    over shards. S_d differ by at most one and sum to the original pair budget.
    The half-sample noise is formed within each shard before the scalar sum.
    """
    sampling = meas_ctx.energy_sampling
    n_data = mesh.shape["data"]
    minimum = 2 if sampling.track_half_sample_diagnostic else 1
    if sampling.pair_sample_size < minimum * n_data:
        raise ValueError(
            f"Local walker sampling requires at least {minimum} pairs per data shard "
            f"({minimum * n_data} total) for this diagnostic setting."
        )
    base_count, remainder = divmod(sampling.pair_sample_size, n_data)
    capacity = base_count + bool(remainder)

    def local(common, pi, q, key, h, c, t):
        shard_index = lax.axis_index("data")
        count = base_count + (shard_index < remainder).astype(jnp.int32)
        key_walker, key_chol = jax.random.split(jax.random.fold_in(key, shard_index))

        def evaluate(_):
            proposal = q / jnp.sum(q, dtype=jnp.float64)
            iw = _sample_indices(key_walker, proposal, capacity)
            ig = _sample_indices(key_chol, c.chol_tail_prob, capacity)
            walker_batch = math.ceil(pi.shape[0] / n_chunks)
            terms = pair_fn(common, iw, c.chol_tail_indices[ig], h, c, t,
                            n_chunks=math.ceil(capacity / walker_batch))
            values = pi[iw] * jnp.real(terms) / (proposal[iw] * c.chol_tail_prob[ig])
            indices = jnp.arange(capacity)
            valid = indices < count
            estimate = jnp.sum(jnp.where(valid, values, 0.0), dtype=jnp.float64) / count
            noise = jnp.asarray(0.0, dtype=jnp.float64)
            if sampling.track_half_sample_diagnostic:
                first = count // 2
                second = count - first
                mean_first = jnp.sum(jnp.where(indices < first, values, 0.0), dtype=jnp.float64) / first
                mean_second = jnp.sum(jnp.where(valid & (indices >= first), values, 0.0), dtype=jnp.float64) / second
                noise = jnp.sqrt(first.astype(jnp.float64) * second) / count * (mean_first - mean_second)
            return jnp.stack((estimate, noise))

        # No accepted weight means no tail contribution. In particular, do not
        # evaluate singular/guarded walkers and subsequently multiply NaN by 0.
        result = lax.cond(jnp.sum(pi, dtype=jnp.float64) > 0.0, evaluate,
                          lambda _: jnp.zeros_like(pi, shape=(2,), dtype=jnp.float64), operand=None)
        return lax.psum(result, "data")

    result = jax.shard_map(
        local, mesh=mesh, axis_names={"data"},
        in_specs=(P("data"), P("data"), P("data"), P(), P(), P(), P()), out_specs=P(),
    )(common, accepted_weights, walker_probabilities, rng_key, ham_data, meas_ctx, trial_data)
    return result[0], result[1]
