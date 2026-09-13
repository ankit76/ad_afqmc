"""Sampling helpers for local walker or local Cholesky strata."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .. import walkers as wk
from ..ham.chol import HamChol
from ..sharding import CholeskyLayout, replicate


@partial(jax.tree_util.register_dataclass,
         data_fields=["head_indices", "head_valid", "tail_prob"], meta_fields=["mesh"])
@dataclass(frozen=True)
class ModelPairSamplingData:
    """Small, explicitly placed sampler metadata for a frozen Cholesky layout."""

    head_indices: jax.Array  # (n_model, max_local_head), local indices
    head_valid: jax.Array
    tail_prob: jax.Array  # (n_model, local_nchol), global probabilities, zeros elsewhere
    mesh: Mesh


def with_local_cholesky_sampling(meas_ctx, layout: CholeskyLayout, mesh: Mesh):
    """Attach a frozen sampler to a context built from the reordered Hamiltonian.

    This does not reorder the Hamiltonian or any derived context. The caller
    must first use the layout when placing the Hamiltonian and rebuild all
    contexts from it. Global indices remain available for diagnostic probes.
    Automatic retuning is unsupported while this metadata is attached.
    """
    if ("model" not in mesh.axis_names or mesh.shape["model"] != layout.n_model
            or mesh.shape.get("data", 1) != 1):
        raise ValueError("Local Cholesky sampling requires matching model shards and replicated walkers.")
    sampling = (meas_ctx.component_sampling if hasattr(meas_ctx, "component_sampling")
                else meas_ctx.energy_sampling)
    if sampling is None or getattr(sampling, "sample_local_walkers", False):
        raise ValueError("Use pair sampling with sample_local_walkers=False for a Cholesky layout.")
    if sampling.chol_head_size != layout.head_indices.size:
        raise ValueError("Sampler head size does not match the frozen Cholesky layout.")
    if meas_ctx.model_sampling is not None:
        raise ValueError("Rebuild the context before replacing a frozen Cholesky layout.")
    minimum = 2 if sampling.track_half_sample_diagnostic else 1
    if layout.tail_indices.size and sampling.pair_sample_size < minimum * layout.n_model:
        raise ValueError(f"Local Cholesky sampling requires at least {minimum} pairs per model shard.")
    sh = NamedSharding(mesh, P("model"))
    data = ModelPairSamplingData(
        jax.device_put(layout.local_head_indices, sh),
        jax.device_put(layout.local_head_valid, sh),
        jax.device_put(layout.local_tail_prob, sh), mesh,
    )
    return replace(meas_ctx, model_sampling=data,
        chol_head_indices=replicate(layout.head_indices, mesh),
        chol_tail_indices=replicate(layout.tail_indices, mesh),
        chol_tail_prob=replicate(layout.tail_prob, mesh))


def _model_context_specs(meas_ctx, context_specs_fn):
    specs = context_specs_fn(meas_ctx)
    return replace(specs, model_sampling=replace(specs.model_sampling,
        head_indices=P("model"), head_valid=P("model"), tail_prob=P("model")))


def local_cholesky_head(common, ham_data, meas_ctx, trial_data, *, terms_fn, context_specs_fn, n_chunks):
    """Sum local head terms and squared real terms before global guard/proposal construction."""
    data = meas_ctx.model_sampling
    if data is None:
        raise ValueError("Missing local Cholesky sampler metadata.")
    if data.head_indices.shape[1] == 0:
        return (jnp.zeros_like(common.base, dtype=jnp.complex128),
                jnp.zeros_like(common.base.real, dtype=jnp.float64))
    limit = meas_ctx.energy_sampling.head_chol_batch_size
    width = data.head_indices.shape[1]
    batch = min(limit, width) if limit > 0 else width

    def local(common, arrays, ctx, trial):
        # HamChol's static nchol must describe the local block, so reconstruct
        # it here instead of unflattening a global HamChol inside shard_map.
        h = HamChol(*arrays, basis=ham_data.basis)
        indices, valid = ctx.model_sampling.head_indices[0], ctx.model_sampling.head_valid[0]
        pad = (-width) % batch
        indices = jnp.pad(indices, (0, pad)).reshape(-1, batch)
        valid = jnp.pad(valid, (0, pad)).reshape(-1, batch)
        # Carry model-axis variance through nested scans.
        zero = jnp.zeros_like(indices, shape=common.base.shape, dtype=jnp.complex128)

        def step(carry, xs):
            idx, keep = xs
            terms = terms_fn(common, idx, h, ctx, trial, n_chunks=n_chunks)
            terms = jnp.where(keep[None, :], terms, 0.0)
            total, squared = carry
            return (total + jnp.sum(terms, axis=1, dtype=jnp.complex128),
                    squared + jnp.sum(terms.real.astype(jnp.float64)**2, axis=1)), None

        (total, squared), _ = lax.scan(step, (zero, zero.real), (indices, valid))
        return lax.psum(total, "model"), lax.psum(squared, "model")

    return jax.shard_map(
        local, mesh=data.mesh, axis_names={"model"},
        in_specs=(P(), (P(), P(), P("model")), _model_context_specs(meas_ctx, context_specs_fn), P()),
        out_specs=(P(), P()),
    )(common, (ham_data.h0, ham_data.h1, ham_data.chol), meas_ctx, trial_data)


def local_cholesky_component_head(common, theta_reference, ham_data, meas_ctx, trial_data,
                                  *, terms_fn, project_fn, context_specs_fn, n_chunks):
    """Reduce exact PT head components and their complex projection moments.

    Project each Cholesky term using the global theta reference before summing
    squared real/imaginary parts and the cross moment. These determine the
    common walker proposal even when estimator weights have complex phases.
    """
    data = meas_ctx.model_sampling
    if data is None:
        raise ValueError("Missing local Cholesky sampler metadata.")
    shape = (common.theta.shape[0], 2)
    zero = jnp.zeros(shape, dtype=jnp.complex128)
    if data.head_indices.shape[1] == 0:
        return zero, zero[:, 0].real, zero[:, 0].real, zero[:, 0].real
    sampling = meas_ctx.component_sampling
    width = data.head_indices.shape[1]
    limit = sampling.head_chol_batch_size
    batch = min(limit, width) if limit > 0 else width

    def local(common, theta, arrays, ctx, trial):
        h = HamChol(*arrays, basis=ham_data.basis)
        indices, valid = ctx.model_sampling.head_indices[0], ctx.model_sampling.head_valid[0]
        pad = (-width) % batch
        indices = jnp.pad(indices, (0, pad)).reshape(-1, batch)
        valid = jnp.pad(valid, (0, pad)).reshape(-1, batch)
        # The scan carry must retain the manual model-axis variation.
        total = jnp.zeros_like(indices, shape=shape, dtype=jnp.complex128)
        moment = total[:, 0].real

        def step(carry, xs):
            idx, keep = xs
            terms = terms_fn(common, idx, h, ctx, trial, n_chunks=n_chunks)
            terms = jnp.where(keep[None, :, None], terms, 0.0)
            total, rr, ii, ri = carry
            total = total + jnp.sum(terms, axis=1, dtype=jnp.complex128)
            if sampling.walker_guide_policy == "head_rms":
                projected = project_fn(theta, terms)
                real, imag = projected.real.astype(jnp.float64), projected.imag.astype(jnp.float64)
                rr = rr + jnp.sum(real**2, axis=1)
                ii = ii + jnp.sum(imag**2, axis=1)
                ri = ri + jnp.sum(real * imag, axis=1)
            return (total, rr, ii, ri), None

        sums, _ = lax.scan(step, (total, moment, moment, moment), (indices, valid))
        return jax.tree.map(lambda a: lax.psum(a, "model"), sums)

    return jax.shard_map(
        local, mesh=data.mesh, axis_names={"model"},
        in_specs=(P(), P(), (P(), P(), P("model")), _model_context_specs(meas_ctx, context_specs_fn), P()),
        out_specs=(P(), P(), P(), P()),
    )(common, theta_reference, (ham_data.h0, ham_data.h1, ham_data.chol), meas_ctx, trial_data)


def local_cholesky_tail(common, pi, q, rng_key, ham_data, meas_ctx, trial_data,
                        *, pair_fn, context_specs_fn, n_chunks, components=False):
    """Unbiased local-tail strata with replicated walkers and a fixed total budget.

    Draw w~q globally and gamma~p/P_d locally. Average pi[w]*Re(term)/(q[w]*p_d)
    within each device, then sum device estimates (not their average).
    pi retains its original normalized population weight and is zero for guards.

    With components=True, pi contains unnormalized complex PT weights and
    terms has two complex components. Sample them jointly, retain their phases,
    and return component sums and half-sample differences. The caller keeps
    the exact denominator and projects the diagnostic globally.
    """
    data = meas_ctx.model_sampling
    if data is None:
        raise ValueError("Missing local Cholesky sampler metadata.")
    sampling = meas_ctx.component_sampling if components else meas_ctx.energy_sampling
    n_model = data.mesh.shape["model"]
    minimum = 2 if sampling.track_half_sample_diagnostic else 1
    if sampling.pair_sample_size < minimum * n_model:
        raise ValueError(f"Local Cholesky sampling requires at least {minimum} pairs per model shard.")
    base_count, rem = divmod(sampling.pair_sample_size, n_model)
    capacity = base_count + bool(rem)
    dtype = jnp.complex128 if components else jnp.float64
    result_shape = (2, 2) if components else (2,)

    def local(common, pi, q, key, arrays, ctx, trial):
        d = lax.axis_index("model")
        count = base_count + (d < rem).astype(jnp.int32)
        p = ctx.model_sampling.tail_prob[0]
        mass = jnp.sum(p, dtype=jnp.float64)
        h = HamChol(*arrays, basis=ham_data.basis)

        def evaluate(_):
            pd = p / mass
            kw, kg = jax.random.split(jax.random.fold_in(key, d))
            iw = _sample_indices(kw, q, capacity)
            ig = _sample_indices(kg, pd, capacity)
            walker_batch = math.ceil(pi.size / n_chunks)
            terms = pair_fn(common, iw, ig, h, ctx, trial,
                            n_chunks=math.ceil(capacity / walker_batch))
            values = (pi[iw, None] * terms / (q[iw, None] * pd[ig, None]) if components
                      else pi[iw] * terms.real / (q[iw] * pd[ig]))
            indices = jnp.arange(capacity)
            if components:
                indices = indices[:, None]
            valid = indices < count
            estimate = jnp.sum(jnp.where(valid, values, 0.0), axis=0, dtype=dtype) / count
            noise = jnp.zeros_like(estimate)
            if sampling.track_half_sample_diagnostic:
                first, second = count // 2, count - count // 2
                a = jnp.sum(jnp.where(indices < first, values, 0.0), axis=0, dtype=dtype) / first
                b = jnp.sum(jnp.where(valid & (indices >= first), values, 0.0), axis=0, dtype=dtype) / second
                noise = jnp.sqrt(first.astype(jnp.float64) * second) / count * (a - b)
            return jnp.stack((estimate, noise))

        # Complex weights may cancel exactly while the numerator is nonzero.
        active_weight = jnp.sum(jnp.abs(pi)) if components else jnp.sum(pi)
        result = lax.cond((mass > 0) & (active_weight > 0), evaluate,
            lambda _: jnp.zeros_like(p, shape=result_shape, dtype=dtype), operand=None)
        return lax.psum(result, "model")

    result = jax.shard_map(
        local, mesh=data.mesh, axis_names={"model"},
        in_specs=(P(), P(), P(), P(), (P(), P(), P("model")),
                  _model_context_specs(meas_ctx, context_specs_fn), P()), out_specs=P(),
    )(common, pi, q, rng_key, (ham_data.h0, ham_data.h1, ham_data.chol), meas_ctx, trial_data)
    return result[0], result[1]


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
