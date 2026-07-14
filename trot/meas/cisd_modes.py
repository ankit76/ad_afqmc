from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from .. import walkers as wk
from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.cisd_modes import CisdModeTrial, mode_apply, mode_quadratic
from ..trial.cisd_modes import overlap_r as cisd_mode_overlap_r
from .cisd import CisdMeasCfg, _energy_gl_batched_realimag, _force_bias_chol_contract_high_realimag

_CISD_MODE_MEAS_CFG_ATTR = "_cisd_mode_meas_cfg"


def _greens_restricted(walker: jax.Array, nocc: int) -> jax.Array:
    wocc = walker[:nocc, :]
    return jnp.linalg.solve(wocc.T, walker.T)


def _active_green_blocks(
    green: jax.Array,
    trial_data: CisdModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    green_act = green[trial_data.occ_act_slice, :]
    green_occ = green[trial_data.occ_act_slice, trial_data.vir_act_slice]

    greenp = jnp.zeros((trial_data.norb, trial_data.nvir), dtype=green.dtype)
    greenp = greenp.at[: trial_data.nocc_full, :].set(green[:, trial_data.vir_act_slice])
    greenp = greenp.at[trial_data.vir_act_slice, :].set(
        -jnp.eye(trial_data.nvir, dtype=green.dtype)
    )
    return green_act, green_occ, greenp


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CisdModeMeasCtx:
    rot_chol: jax.Array
    lci1: jax.Array
    chol_tail_prob: jax.Array
    cfg: CisdMeasCfg
    n_mode_chunks: int
    energy_sampling: CisdModePairSamplingCfg | None

    def tree_flatten(self):
        children = (self.rot_chol, self.lci1, self.chol_tail_prob)
        aux = (self.cfg, self.n_mode_chunks, self.energy_sampling)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, n_mode_chunks, energy_sampling = aux
        rot_chol, lci1, chol_tail_prob = children
        return cls(
            rot_chol=rot_chol,
            lci1=lci1,
            chol_tail_prob=chol_tail_prob,
            cfg=cfg,
            n_mode_chunks=n_mode_chunks,
            energy_sampling=energy_sampling,
        )


@dataclass(frozen=True)
class CisdModePairSamplingCfg:
    """Walker--Cholesky pair sampling for the block energy.

    The first ``chol_head_size`` Cholesky contributions are evaluated exactly
    for every walker. ``pair_sample_size`` weighted walker--Cholesky pairs are
    drawn from the remaining tail. Every retained K mode is evaluated
    deterministically in both the head and tail.
    """

    chol_head_size: int
    pair_sample_size: int

    def __post_init__(self) -> None:
        if self.chol_head_size < 0:
            raise ValueError("chol_head_size must be nonnegative.")
        if self.pair_sample_size <= 0:
            raise ValueError("pair_sample_size must be positive.")


class CisdModeEnergyCommon(NamedTuple):
    """Per-walker intermediates shared by all Cholesky contributions."""

    green: jax.Array
    greenp: jax.Array
    overlap: jax.Array
    ci1g1: jax.Array
    ci2_green: jax.Array
    base: jax.Array


def get_cisd_mode_meas_cfg(meas_ops: MeasOps) -> CisdMeasCfg | None:
    cfg = getattr(meas_ops, _CISD_MODE_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, CisdMeasCfg) else None


def build_meas_ctx(
    ham_data: HamChol,
    trial_data: CisdModeTrial,
    *,
    cfg: CisdMeasCfg = CisdMeasCfg(memory_mode="high"),
    n_mode_chunks: int = 1,
    energy_sampling: CisdModePairSamplingCfg | None = None,
) -> CisdModeMeasCtx:
    """Build full-Cholesky measurement intermediates.

    ``n_mode_chunks=1`` evaluates the complete mode axis in one batch. Larger
    values reduce mode-dependent temporary memory by scanning over that many
    partitions; the requested count is capped at the retained mode rank.
    """
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    if cfg.memory_mode != "high":
        raise ValueError("Mode-native CISD measurements currently require memory_mode='high'.")

    if ham_data.basis != "restricted":
        raise ValueError("CISD mode MeasOps requires HamChol.basis == 'restricted'.")

    chol = ham_data.chol
    n_chol = int(chol.shape[0])
    if energy_sampling is not None and energy_sampling.chol_head_size > n_chol:
        raise ValueError(
            f"chol_head_size must not exceed the number of Cholesky vectors ({n_chol})."
        )
    rot_chol = chol[:, : trial_data.nocc_full, :]
    lci1 = jnp.einsum(
        "git,pt->gip",
        chol[:, :, trial_data.vir_act_slice],
        trial_data.ci1,
        optimize="optimal",
    )
    meas_ctx = CisdModeMeasCtx(
        rot_chol=rot_chol,
        lci1=lci1,
        chol_tail_prob=jnp.empty((0,), dtype=jnp.float64),
        cfg=cfg,
        n_mode_chunks=min(int(n_mode_chunks), trial_data.mode_rank),
        energy_sampling=energy_sampling,
    )
    if energy_sampling is not None:
        chol_tail_prob = _build_chol_tail_prob(ham_data, meas_ctx, trial_data)
        meas_ctx = replace(meas_ctx, chol_tail_prob=chol_tail_prob)
    return meas_ctx


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
) -> jax.Array:
    green = _greens_restricted(walker, trial_data.nocc_full)
    green_act, green_occ, greenp = _active_green_blocks(green, trial_data)

    lg = jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")
    ci1g = jnp.einsum("pt,pt->", trial_data.ci1, green_occ, optimize="optimal")
    ci1gp = jnp.einsum("pt,it->pi", trial_data.ci1, greenp, optimize="optimal")
    gci1gp = jnp.einsum("pj,pi->ij", green_act, ci1gp, optimize="optimal")

    projections, kg = mode_apply(trial_data, green_occ)
    gkg = mode_quadratic(trial_data, green_occ, projections)
    overlap = 1.0 + 2.0 * ci1g + gkg

    cisd_green = -2.0 * (greenp @ kg.T) @ green_act
    correction = _force_bias_chol_contract_high_realimag(
        ham_data.chol,
        cisd_green - 2.0 * gci1gp,
        meas_ctx.cfg,
    )
    return (2.0 * lg + 4.0 * ci1g * lg + 2.0 * lg * gkg + correction) / overlap


def _mode_quadratic_matrices(
    trial_data: CisdModeTrial,
    meas_ctx: CisdModeMeasCtx,
    matrices: jax.Array,
) -> jax.Array:
    """Evaluate ``x.T @ K @ x`` for a batch of pair-space matrices.

    Every retained mode is included. ``n_mode_chunks`` only bounds the
    temporary projection matrix and introduces no stochastic approximation.
    """
    pair_shape = (trial_data.nocc, trial_data.nvir)
    if matrices.shape[-2:] != pair_shape:
        raise ValueError(
            f"matrices must end in pair-space shape {pair_shape}, got {matrices.shape}."
        )

    leading_shape = matrices.shape[:-2]
    matrices_flat = matrices.reshape((-1,) + pair_shape)
    rank = trial_data.mode_rank

    def evaluate_chunk(eigenvalues, modes):
        matrices_r = jnp.real(matrices_flat).astype(modes.dtype)
        projections_r = jnp.einsum("spt,rpt->sr", matrices_r, modes, optimize="optimal")
        if jnp.issubdtype(matrices.dtype, jnp.complexfloating):
            matrices_i = jnp.imag(matrices_flat).astype(modes.dtype)
            projections_i = jnp.einsum("spt,rpt->sr", matrices_i, modes, optimize="optimal")
            projections = projections_r.astype(jnp.complex128) + 1.0j * projections_i.astype(
                jnp.complex128
            )
            reduction_dtype = jnp.complex128
        else:
            projections = projections_r.astype(jnp.float64)
            reduction_dtype = jnp.float64
        return jnp.sum(
            eigenvalues.astype(jnp.float64)[None, :] * projections * projections,
            axis=1,
            dtype=reduction_dtype,
        )

    n_mode_chunks = min(meas_ctx.n_mode_chunks, rank)
    if n_mode_chunks == 1:
        values = evaluate_chunk(trial_data.eigenvalues, trial_data.modes)
        return values.reshape(leading_shape)

    base_chunk_size = rank // n_mode_chunks
    n_larger_chunks = rank % n_mode_chunks
    chunk_size = base_chunk_size + int(n_larger_chunks > 0)
    chunk_offsets = jnp.arange(chunk_size, dtype=jnp.int32)
    result_dtype = (
        jnp.complex128 if jnp.issubdtype(matrices.dtype, jnp.complexfloating) else jnp.float64
    )
    zero = jnp.zeros((matrices_flat.shape[0],), dtype=result_dtype)

    def scan_body(total, chunk_index):
        is_larger = chunk_index < n_larger_chunks
        chunk_length = base_chunk_size + is_larger.astype(jnp.int32)
        start = chunk_index * base_chunk_size + jnp.minimum(chunk_index, n_larger_chunks)
        indices = start + chunk_offsets
        valid = chunk_offsets < chunk_length
        indices = jnp.minimum(indices, rank - 1)
        eigenvalues = jnp.where(valid, trial_data.eigenvalues[indices], 0.0)
        modes = trial_data.modes[indices]
        return total + evaluate_chunk(eigenvalues, modes), None

    values, _ = lax.scan(
        scan_body,
        zero,
        jnp.arange(n_mode_chunks, dtype=jnp.int32),
    )
    return values.reshape(leading_shape)


def _cisd_mode_energy_common(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
) -> CisdModeEnergyCommon:
    """Build the exact per-walker base and reusable Cholesky intermediates."""
    green = _greens_restricted(walker, trial_data.nocc_full)
    green_act, green_occ, greenp = _active_green_blocks(green, trial_data)

    h1 = ham_data.h1
    chol = ham_data.chol
    hg = jnp.einsum("pj,pj->", h1[: trial_data.nocc_full, :], green, optimize="optimal")
    e1_0 = 2.0 * hg

    ci1g = jnp.einsum("pt,pt->", trial_data.ci1, green_occ, optimize="optimal")
    ci1_green = (greenp @ trial_data.ci1.T) @ green_act
    e1_1 = 4.0 * ci1g * hg - 2.0 * jnp.einsum("ij,ij->", h1, ci1_green, optimize="optimal")

    projections, kg = mode_apply(trial_data, green_occ)
    doubles = mode_quadratic(trial_data, green_occ, projections)
    ci2_green = (greenp @ kg.T) @ green_act
    e1_2 = 2.0 * hg * doubles - 2.0 * jnp.einsum("ij,ij->", h1, ci2_green, optimize="optimal")

    lg = jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")
    e2_0_direct = 2.0 * (lg @ lg)
    lci1g = _force_bias_chol_contract_high_realimag(chol, ci1_green, meas_ctx.cfg)
    e2_1_2 = -2.0 * (lci1g @ lg)
    lci2g = _force_bias_chol_contract_high_realimag(chol, ci2_green, meas_ctx.cfg)
    e2_2_2_1 = -(lci2g @ lg)

    overlap = 1.0 + 2.0 * ci1g + doubles
    ci1g1 = trial_data.ci1 @ green[:, trial_data.vir_act_slice].T
    base = (
        ham_data.h0 + e2_0_direct + (e1_0 + e1_1 + e1_2 + 2.0 * e2_1_2 + 4.0 * e2_2_2_1) / overlap
    )
    return CisdModeEnergyCommon(
        green=green,
        greenp=greenp,
        overlap=overlap,
        ci1g1=ci1g1,
        ci2_green=ci2_green,
        base=base,
    )


def _cisd_mode_chol_terms(
    common: CisdModeEnergyCommon,
    chol: jax.Array,
    rot_chol: jax.Array,
    lci1: jax.Array,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
) -> jax.Array:
    """Return one walker's residual energy contribution for each Cholesky vector."""
    lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, common.green, optimize="optimal")
    e20 = -jnp.sum(lg1 * jnp.swapaxes(lg1, -1, -2), axis=(-1, -2))
    e2131 = jnp.einsum(
        "gpq,gqa,ap->g",
        lg1,
        lg1[:, :, trial_data.occ_act_slice],
        common.ci1g1,
        optimize="optimal",
    )
    lci1g_mat = jnp.einsum("gia,qi->gaq", lci1, common.green, optimize="optimal")
    e2132 = -jnp.einsum(
        "gaq,gqa->g",
        lci1g_mat,
        lg1[:, :, trial_data.occ_act_slice],
        optimize="optimal",
    )

    gl = _energy_gl_batched_realimag(common.green, chol, meas_ctx.cfg)
    lci2_green = jnp.einsum("gpk,ik->gpi", rot_chol, common.ci2_green, optimize="optimal")
    e2222 = 0.5 * jnp.einsum("gpi,gpi->g", gl, lci2_green, optimize="optimal")
    glgp = jnp.einsum("gpi,it->gpt", gl, common.greenp, optimize="optimal")
    glgp = glgp[:, trial_data.occ_act_slice, :]
    e223 = _mode_quadratic_matrices(trial_data, meas_ctx, glgp)
    return e20 + (2.0 * (e2131 + e2132) + 4.0 * e2222 + e223) / common.overlap


def _cisd_mode_chol_terms_for_walkers(
    common: CisdModeEnergyCommon,
    chol: jax.Array,
    rot_chol: jax.Array,
    lci1: jax.Array,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
    n_chunks: int = 1,
) -> jax.Array:
    """Return residual terms for every walker--Cholesky combination."""
    return wk.vmap_chunked(
        lambda common_i: _cisd_mode_chol_terms(
            common_i,
            chol,
            rot_chol,
            lci1,
            meas_ctx,
            trial_data,
        ),
        n_chunks=n_chunks,
    )(common)


def _cisd_mode_chol_pair_terms(
    common: CisdModeEnergyCommon,
    chol: jax.Array,
    rot_chol: jax.Array,
    lci1: jax.Array,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
    n_chunks: int = 1,
) -> jax.Array:
    """Return residual terms for aligned walker--Cholesky pairs."""
    return wk.vmap_chunked(
        lambda common_i, chol_i, rot_chol_i, lci1_i: _cisd_mode_chol_terms(
            common_i,
            chol_i[None, ...],
            rot_chol_i[None, ...],
            lci1_i[None, ...],
            meas_ctx,
            trial_data,
        )[0],
        n_chunks=n_chunks,
        in_axes=(0, 0, 0, 0),
    )(common, chol, rot_chol, lci1)


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
) -> jax.Array:
    """Complete deterministic local energy from every stored K mode."""
    common = _cisd_mode_energy_common(walker, ham_data, meas_ctx, trial_data)
    chol_terms = _cisd_mode_chol_terms(
        common,
        ham_data.chol,
        meas_ctx.rot_chol,
        meas_ctx.lci1,
        meas_ctx,
        trial_data,
    )
    return common.base + jnp.sum(chol_terms, dtype=jnp.complex128)


def _build_chol_tail_prob(
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
) -> jax.Array:
    """Build a static importance guide from the reference determinant."""
    sampling = meas_ctx.energy_sampling
    if sampling is None:
        return jnp.empty((0,), dtype=jnp.float64)

    n_chol = int(ham_data.chol.shape[0])
    if sampling.chol_head_size == n_chol:
        return jnp.empty((0,), dtype=jnp.float64)

    reference_walker = jnp.eye(
        trial_data.norb,
        trial_data.nocc_full,
        dtype=jnp.complex128,
    )
    common = _cisd_mode_energy_common(
        reference_walker,
        ham_data,
        meas_ctx,
        trial_data,
    )
    terms = _cisd_mode_chol_terms(
        common,
        ham_data.chol[sampling.chol_head_size :],
        meas_ctx.rot_chol[sampling.chol_head_size :],
        meas_ctx.lci1[sampling.chol_head_size :],
        meas_ctx,
        trial_data,
    )
    scores = jnp.maximum(jnp.abs(terms).astype(jnp.float64), 1.0e-300)
    return scores / jnp.sum(scores, dtype=jnp.float64)


def pair_sampled_block_energy(
    walkers: jax.Array,
    weights: jax.Array,
    overlaps: jax.Array,
    rng_key: jax.Array,
    n_chunks: int,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
) -> jax.Array:
    """Exact Cholesky head plus sampled walker--Cholesky tail energy.

    Walkers are sampled according to their normalized phaseless weights and
    tail Cholesky vectors according to the reference-determinant importance
    guide stored in ``meas_ctx``. All retained K modes are summed exactly for
    every evaluated pair.
    """
    del overlaps
    sampling = meas_ctx.energy_sampling
    if sampling is None:
        raise ValueError("pair_sampled_block_energy requires an energy sampling config.")

    common = wk.vmap_chunked(
        _cisd_mode_energy_common,
        n_chunks=n_chunks,
        in_axes=(0, None, None, None),
    )(walkers, ham_data, meas_ctx, trial_data)

    weights_real = jnp.real(weights).astype(jnp.float64)
    weight_sum = jnp.sum(weights_real, dtype=jnp.float64)
    weight_sum_safe = jnp.where(weight_sum == 0.0, 1.0, weight_sum)
    norm_weights = weights_real / weight_sum_safe
    uniform_weights = jnp.ones_like(weights_real) / weights_real.shape[0]
    sample_weights = jnp.where(weight_sum == 0.0, uniform_weights, norm_weights)

    if sampling.chol_head_size > 0:
        head_terms = _cisd_mode_chol_terms_for_walkers(
            common,
            ham_data.chol[: sampling.chol_head_size],
            meas_ctx.rot_chol[: sampling.chol_head_size],
            meas_ctx.lci1[: sampling.chol_head_size],
            meas_ctx,
            trial_data,
            n_chunks=n_chunks,
        )
        head_energy = jnp.real(common.base + jnp.sum(head_terms, axis=1))
    else:
        head_energy = jnp.real(common.base)
    block_head = jnp.sum(norm_weights * head_energy, dtype=jnp.float64)

    tail_size = int(meas_ctx.chol_tail_prob.shape[0])
    if tail_size == 0:
        return block_head

    key_walker, key_chol = jax.random.split(rng_key)
    sample_walker = jax.random.choice(
        key_walker,
        weights_real.shape[0],
        shape=(sampling.pair_sample_size,),
        replace=True,
        p=sample_weights,
    )
    sample_chol_rel = jax.random.choice(
        key_chol,
        tail_size,
        shape=(sampling.pair_sample_size,),
        replace=True,
        p=meas_ctx.chol_tail_prob,
    )
    sample_chol = sample_chol_rel + sampling.chol_head_size
    sample_common = tree_util.tree_map(lambda value: value[sample_walker], common)
    walker_batch_size = (int(weights_real.shape[0]) + n_chunks - 1) // n_chunks
    pair_n_chunks = (sampling.pair_sample_size + walker_batch_size - 1) // walker_batch_size
    sample_terms = _cisd_mode_chol_pair_terms(
        sample_common,
        ham_data.chol[sample_chol],
        meas_ctx.rot_chol[sample_chol],
        meas_ctx.lci1[sample_chol],
        meas_ctx,
        trial_data,
        n_chunks=pair_n_chunks,
    )
    tail_estimate = jnp.mean(
        jnp.real(sample_terms) / meas_ctx.chol_tail_prob[sample_chol_rel],
        dtype=jnp.float64,
    )
    return block_head + tail_estimate


def make_cisd_mode_meas_ops(
    sys: System,
    *,
    mixed_precision: bool = True,
    n_mode_chunks: int = 1,
    energy_sampling: CisdModePairSamplingCfg | None = None,
) -> MeasOps:
    """Build retained-mode CISD measurements.

    The deterministic default batches every Cholesky vector. Passing
    ``energy_sampling`` instead evaluates its Cholesky head exactly and uses
    unbiased weighted walker--Cholesky sampling for the tail. All retained
    modes remain deterministic in either case. The result is exact when all
    pair-space modes are retained and is the consistent truncated-K
    approximation otherwise.
    """
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"CISD mode MeasOps currently supports only restricted walkers, got: {sys.walker_kind}"
        )
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")

    cfg = CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype_testing=jnp.complex64 if mixed_precision else jnp.complex128,
    )
    meas_ops = MeasOps(
        overlap=cisd_mode_overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(
            ham_data,
            trial_data,
            cfg=cfg,
            n_mode_chunks=n_mode_chunks,
            energy_sampling=energy_sampling,
        ),
        kernels={k_force_bias: force_bias_kernel_rw_rh, k_energy: energy_kernel_rw_rh},
        observables={},
        block_energy=pair_sampled_block_energy if energy_sampling is not None else None,
    )
    object.__setattr__(meas_ops, _CISD_MODE_MEAS_CFG_ATTR, cfg)
    return meas_ops
