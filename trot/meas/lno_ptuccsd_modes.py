from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from .. import walkers as wk
from ..core.ops import (
    BlockComponentEstimate,
    EstimatorOps,
    d_pt_component_sampling_noise_imag,
    d_pt_component_sampling_noise_real,
    d_pt_estimator_phase_coherence,
    d_pt_walker_proposal_ess,
)
from ..core.system import System
from ..ham.chol import HamCholUhf
from ..trial.ptuccsd_modes import PtuccsdThoulessModeTrial, reference_overlap_u
from .lno_ptuccsd_thouless import (
    _fragment_t1_reference_energy,
    _thouless_transform,
    _ufock,
)
from .ptuccsd_modes import PtuccsdModeMeasCfg, PtuccsdModePairSamplingCfg


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LnoPtuccsdModeMeasCtx:
    exp_t1_a: jax.Array
    exp_t1_b: jax.Array
    h1bar_a: jax.Array
    h1bar_b: jax.Array
    cholbar_a: jax.Array
    cholbar_b: jax.Array
    fockbar_a: jax.Array
    fockbar_b: jax.Array
    projected_modes_a: jax.Array
    projected_modes_b: jax.Array
    weight_a: jax.Array
    weight_b: jax.Array
    e0t1_f: jax.Array
    reference_chol_scores: jax.Array
    chol_head_indices: jax.Array
    chol_tail_indices: jax.Array
    chol_tail_prob: jax.Array
    cfg: PtuccsdModeMeasCfg
    component_sampling: PtuccsdModePairSamplingCfg | None
    mode_batch_size: int

    def tree_flatten(self):
        children = (
            self.exp_t1_a,
            self.exp_t1_b,
            self.h1bar_a,
            self.h1bar_b,
            self.cholbar_a,
            self.cholbar_b,
            self.fockbar_a,
            self.fockbar_b,
            self.projected_modes_a,
            self.projected_modes_b,
            self.weight_a,
            self.weight_b,
            self.e0t1_f,
            self.reference_chol_scores,
            self.chol_head_indices,
            self.chol_tail_indices,
            self.chol_tail_prob,
        )
        return children, (self.cfg, self.component_sampling, self.mode_batch_size)

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, component_sampling, mode_batch_size = aux
        return cls(
            *children,
            cfg=cfg,
            component_sampling=component_sampling,
            mode_batch_size=mode_batch_size,
        )


class LnoPtuccsdModeCommon(NamedTuple):
    green_a: jax.Array
    green_b: jax.Array
    green_occ_a: jax.Array
    green_occ_b: jax.Array
    greenp_a: jax.Array
    greenp_b: jax.Array
    t2_green_a_tot: jax.Array
    t2_green_b_tot: jax.Array
    theta_f: jax.Array
    e0_f_base: jax.Array
    h_t_f_base: jax.Array
    e0_base: jax.Array


def _half_green(transformed_walker: jax.Array, nocc: int) -> jax.Array:
    return jnp.linalg.solve(transformed_walker[:nocc].T, transformed_walker.T)


def _same_spin_exchange(
    projected_modes: jax.Array,
    modes: jax.Array,
    green_occ: jax.Array,
    eigenvalues: jax.Array,
    mode_batch_size: int,
) -> jax.Array:
    """Contract ``sum_r lambda_r V_r @ G.T @ U_r`` without dense T2."""

    rank = int(eigenvalues.shape[0])
    zero = jnp.zeros(
        (projected_modes.shape[1], modes.shape[2]),
        dtype=green_occ.dtype,
    )
    if rank == 0:
        return zero

    batch_size = min(mode_batch_size, rank)
    n_batches = math.ceil(rank / batch_size)
    padded_size = n_batches * batch_size
    indices = jnp.arange(padded_size, dtype=jnp.int32).reshape(n_batches, batch_size)
    valid = indices < rank
    indices = jnp.minimum(indices, rank - 1)

    def scan_batch(total, xs):
        indices_i, valid_i = xs
        projected_i = projected_modes[indices_i]
        modes_i = modes[indices_i]
        values_i = jnp.where(valid_i, eigenvalues[indices_i], 0.0)
        intermediate = jnp.einsum(
            "ria,ja->rij", projected_i, green_occ, optimize="optimal"
        )
        contribution = jnp.einsum(
            "r,rij,rjb->ib", values_i, intermediate, modes_i, optimize="optimal"
        )
        return total + contribution, None

    result, _ = lax.scan(scan_batch, zero, (indices, valid))
    return result


def _mode_h2_t2(
    modes_a: jax.Array,
    modes_b: jax.Array,
    projected_a: jax.Array,
    projected_b: jax.Array,
    eigenvalues: jax.Array,
    glgp_a: jax.Array,
    glgp_b: jax.Array,
    mode_batch_size: int,
) -> jax.Array:
    """Evaluate the retained-mode Cholesky bilinear in bounded mode batches."""

    rank = int(eigenvalues.shape[0])
    zero = jnp.zeros(glgp_a.shape[0], dtype=glgp_a.dtype)
    if rank == 0:
        return zero

    batch_size = min(mode_batch_size, rank)
    n_batches = math.ceil(rank / batch_size)
    padded_size = n_batches * batch_size
    indices = jnp.arange(padded_size, dtype=jnp.int32).reshape(n_batches, batch_size)
    valid = indices < rank
    indices = jnp.minimum(indices, rank - 1)

    def scan_batch(total, xs):
        indices_i, valid_i = xs
        modes_a_i = modes_a[indices_i]
        modes_b_i = modes_b[indices_i]
        projected_a_i = projected_a[indices_i]
        projected_b_i = projected_b[indices_i]
        values_i = jnp.where(valid_i, eigenvalues[indices_i], 0.0)
        pua = jnp.einsum("ria,gia->gr", modes_a_i, glgp_a, optimize="optimal")
        pub = jnp.einsum("ria,gia->gr", modes_b_i, glgp_b, optimize="optimal")
        pva = jnp.einsum(
            "ria,gia->gr", projected_a_i, glgp_a, optimize="optimal"
        )
        pvb = jnp.einsum(
            "ria,gia->gr", projected_b_i, glgp_b, optimize="optimal"
        )
        contribution = 0.5 * jnp.einsum(
            "r,gr,gr->g", values_i, pva + pvb, pua + pub, optimize="optimal"
        )
        return total + contribution, None

    result, _ = lax.scan(scan_batch, zero, (indices, valid))
    return result


def _energy_common(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholUhf,
    meas_ctx: LnoPtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> LnoPtuccsdModeCommon:
    del ham_data
    noa, nob = trial_data.nocc
    nva, nvb = trial_data.nvir
    walker_a, walker_b = walker
    walker_bar_a = meas_ctx.exp_t1_a @ walker_a
    walker_bar_b = meas_ctx.exp_t1_b @ walker_b
    green_a = _half_green(walker_bar_a, noa)
    green_b = _half_green(walker_bar_b, nob)
    green_occ_a = green_a[:, noa:]
    green_occ_b = green_b[:, nob:]
    greenp_a = jnp.vstack((green_occ_a, -jnp.eye(nva, dtype=green_a.dtype)))
    greenp_b = jnp.vstack((green_occ_b, -jnp.eye(nvb, dtype=green_b.dtype)))

    rank = trial_data.mode_rank
    da, _ = trial_data.pair_dim
    cfg = meas_ctx.cfg
    modes = trial_data.modes.astype(cfg.mixed_real_dtype)
    modes_a = modes[:, :da].reshape(rank, noa, nva)
    modes_b = modes[:, da:].reshape(rank, nob, nvb)
    projected_a = meas_ctx.projected_modes_a.astype(cfg.mixed_real_dtype)
    projected_b = meas_ctx.projected_modes_b.astype(cfg.mixed_real_dtype)
    ga = green_occ_a.astype(cfg.mixed_complex_dtype)
    gb = green_occ_b.astype(cfg.mixed_complex_dtype)
    eigenvalues = trial_data.eigenvalues.astype(cfg.mixed_real_dtype)

    pua = jnp.einsum("ria,ia->r", modes_a, ga, optimize="optimal")
    pub = jnp.einsum("ria,ia->r", modes_b, gb, optimize="optimal")
    pva = jnp.einsum("ria,ia->r", projected_a, ga, optimize="optimal")
    pvb = jnp.einsum("ria,ia->r", projected_b, gb, optimize="optimal")
    theta_f = 0.5 * jnp.einsum(
        "r,r,r->", eigenvalues, pva + pvb, pua + pub, optimize="optimal"
    )

    t2g_aa_c = 0.25 * jnp.einsum(
        "r,r,rjb->jb", eigenvalues, pva, modes_a, optimize="optimal"
    )
    t2g_aa_e = 0.25 * _same_spin_exchange(
        projected_a, modes_a, ga, eigenvalues, meas_ctx.mode_batch_size
    )
    t2g_bb_c = 0.25 * jnp.einsum(
        "r,r,rjb->jb", eigenvalues, pvb, modes_b, optimize="optimal"
    )
    t2g_bb_e = 0.25 * _same_spin_exchange(
        projected_b, modes_b, gb, eigenvalues, meas_ctx.mode_batch_size
    )
    t2g_ab_a = 0.5 * jnp.einsum(
        "r,r,rjb->jb", eigenvalues, pva, modes_b, optimize="optimal"
    )
    t2g_ab_b = 0.5 * jnp.einsum(
        "r,r,ria->ia", eigenvalues, pub, projected_a, optimize="optimal"
    )
    t2g_ba_a = 0.5 * jnp.einsum(
        "r,r,ria->ia", eigenvalues, pua, projected_b, optimize="optimal"
    )
    t2g_ba_b = 0.5 * jnp.einsum(
        "r,r,rjb->jb", eigenvalues, pvb, modes_a, optimize="optimal"
    )

    t2_green_aaa_c = jnp.einsum(
        "pb,jb,jq->pq", greenp_a, t2g_aa_c, green_a, optimize="optimal"
    )
    t2_green_aaa_e = jnp.einsum(
        "pb,ib,iq->pq", greenp_a, t2g_aa_e, green_a, optimize="optimal"
    )
    t2_green_bbb_c = jnp.einsum(
        "pb,jb,jq->pq", greenp_b, t2g_bb_c, green_b, optimize="optimal"
    )
    t2_green_bbb_e = jnp.einsum(
        "pb,ib,iq->pq", greenp_b, t2g_bb_e, green_b, optimize="optimal"
    )
    t2_green_aba = jnp.einsum(
        "pa,ia,iq->pq", greenp_a, t2g_ab_b, green_a, optimize="optimal"
    )
    t2_green_baa = jnp.einsum(
        "pb,jb,jq->pq", greenp_a, t2g_ba_b, green_a, optimize="optimal"
    )
    t2_green_bab = jnp.einsum(
        "pa,ia,iq->pq", greenp_b, t2g_ba_a, green_b, optimize="optimal"
    )
    t2_green_abb = jnp.einsum(
        "pb,jb,jq->pq", greenp_b, t2g_ab_a, green_b, optimize="optimal"
    )
    t2_green_aaa = 2.0 * (t2_green_aaa_c - t2_green_aaa_e)
    t2_green_bbb = 2.0 * (t2_green_bbb_c - t2_green_bbb_e)
    t2_green_a = t2_green_aaa + t2_green_aba + t2_green_baa
    t2_green_b = t2_green_bbb + t2_green_bab + t2_green_abb
    t2_green_a_tot = 2.0 * t2_green_aaa + 2.0 * (
        t2_green_aba + t2_green_baa
    )
    t2_green_b_tot = 2.0 * t2_green_bbb + 2.0 * (
        t2_green_bab + t2_green_abb
    )

    e1_0 = jnp.einsum(
        "pj,pj->", meas_ctx.h1bar_a[:noa], green_a, optimize="optimal"
    )
    e1_0 += jnp.einsum(
        "pj,pj->", meas_ctx.h1bar_b[:nob], green_b, optimize="optimal"
    )
    e1_2 = theta_f * e1_0
    e1_2 -= jnp.einsum(
        "pq,pq->", t2_green_a, meas_ctx.h1bar_a, optimize="optimal"
    )
    e1_2 -= jnp.einsum(
        "pq,pq->", t2_green_b, meas_ctx.h1bar_b, optimize="optimal"
    )
    e1_f = jnp.einsum(
        "ia,ik,ka->",
        green_occ_a,
        meas_ctx.weight_a,
        meas_ctx.fockbar_a[:noa, noa:],
        optimize="optimal",
    )
    e1_f += jnp.einsum(
        "ia,ik,ka->",
        green_occ_b,
        meas_ctx.weight_b,
        meas_ctx.fockbar_b[:nob, nob:],
        optimize="optimal",
    )
    return LnoPtuccsdModeCommon(
        green_a=green_a,
        green_b=green_b,
        green_occ_a=green_occ_a,
        green_occ_b=green_occ_b,
        greenp_a=greenp_a,
        greenp_b=greenp_b,
        t2_green_a_tot=t2_green_a_tot,
        t2_green_b_tot=t2_green_b_tot,
        theta_f=theta_f,
        e0_f_base=meas_ctx.e0t1_f + e1_f,
        h_t_f_base=e1_2,
        e0_base=e1_0,
    )


def _chol_terms(
    common: LnoPtuccsdModeCommon,
    chol_a: jax.Array,
    chol_b: jax.Array,
    meas_ctx: LnoPtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    rank = trial_data.mode_rank
    da, _ = trial_data.pair_dim
    cfg = meas_ctx.cfg
    modes = trial_data.modes.astype(cfg.mixed_real_dtype_testing)
    modes_a = modes[:, :da].reshape(rank, noa, trial_data.nvir[0])
    modes_b = modes[:, da:].reshape(rank, nob, trial_data.nvir[1])
    projected_a = meas_ctx.projected_modes_a.astype(cfg.mixed_real_dtype_testing)
    projected_b = meas_ctx.projected_modes_b.astype(cfg.mixed_real_dtype_testing)
    eigenvalues = trial_data.eigenvalues.astype(cfg.mixed_real_dtype_testing)

    gl_a = jnp.einsum("ir,gpr->gip", common.green_a, chol_a, optimize="optimal")
    gl_b = jnp.einsum("ir,gpr->gip", common.green_b, chol_b, optimize="optimal")
    tr_a = jnp.trace(gl_a[:, :, :noa], axis1=1, axis2=2)
    tr_b = jnp.trace(gl_b[:, :, :nob], axis1=1, axis2=2)
    tr_sum = tr_a + tr_b
    e2_0 = 0.5 * tr_sum * tr_sum
    e2_0 -= 0.5 * jnp.einsum(
        "gij,gji->g", gl_a[:, :, :noa], gl_a[:, :, :noa], optimize="optimal"
    )
    e2_0 -= 0.5 * jnp.einsum(
        "gij,gji->g", gl_b[:, :, :nob], gl_b[:, :, :nob], optimize="optimal"
    )

    lt2g_a = jnp.einsum(
        "gpq,pq->g", chol_a, common.t2_green_a_tot, optimize="optimal"
    )
    lt2g_b = jnp.einsum(
        "gpq,pq->g", chol_b, common.t2_green_b_tot, optimize="optimal"
    )
    h2_direct = -0.5 * (lt2g_a + lt2g_b) * tr_sum
    lt2_green_a = jnp.einsum(
        "gpi,ji->gpj", chol_a[:, :noa], common.t2_green_a_tot, optimize="optimal"
    )
    lt2_green_b = jnp.einsum(
        "gpi,ji->gpj", chol_b[:, :nob], common.t2_green_b_tot, optimize="optimal"
    )
    h2_exchange = 0.5 * (
        jnp.einsum("gip,gip->g", gl_a, lt2_green_a, optimize="optimal")
        + jnp.einsum("gip,gip->g", gl_b, lt2_green_b, optimize="optimal")
    )

    glgp_a = jnp.einsum(
        "gip,pa->gia", gl_a, common.greenp_a, optimize="optimal"
    ).astype(cfg.mixed_complex_dtype_testing)
    glgp_b = jnp.einsum(
        "gip,pa->gia", gl_b, common.greenp_b, optimize="optimal"
    ).astype(cfg.mixed_complex_dtype_testing)
    h2_t2 = _mode_h2_t2(
        modes_a,
        modes_b,
        projected_a,
        projected_b,
        eigenvalues,
        glgp_a,
        glgp_b,
        meas_ctx.mode_batch_size,
    )

    d_a = jnp.einsum(
        "gia,ka->gik", chol_a[:, :noa, noa:], common.green_occ_a, optimize="optimal"
    )
    d_b = jnp.einsum(
        "gia,ka->gik", chol_b[:, :nob, nob:], common.green_occ_b, optimize="optimal"
    )
    trace_d = jnp.trace(d_a, axis1=1, axis2=2) + jnp.trace(
        d_b, axis1=1, axis2=2
    )
    weighted_trace = jnp.einsum(
        "ik,gik->g", meas_ctx.weight_a, d_a, optimize="optimal"
    )
    weighted_trace += jnp.einsum(
        "ik,gik->g", meas_ctx.weight_b, d_b, optimize="optimal"
    )
    e2_f = 0.5 * trace_d * weighted_trace
    e2_f -= 0.5 * jnp.einsum(
        "gij,gjk,ik->g", d_a, d_a, meas_ctx.weight_a, optimize="optimal"
    )
    e2_f -= 0.5 * jnp.einsum(
        "gij,gjk,ik->g", d_b, d_b, meas_ctx.weight_b, optimize="optimal"
    )
    h_t_f = common.theta_f * e2_0 + h2_direct + h2_exchange + h2_t2
    return jnp.stack((e2_f, h_t_f, e2_0), axis=-1)


def _components(walker, ham_data, meas_ctx, trial_data):
    common = _energy_common(walker, ham_data, meas_ctx, trial_data)
    sampling = meas_ctx.component_sampling
    if sampling is None:
        terms = _chol_terms(
            common, meas_ctx.cholbar_a, meas_ctx.cholbar_b, meas_ctx, trial_data
        )
    else:
        terms = _chol_index_terms(
            common,
            jnp.arange(meas_ctx.cholbar_a.shape[0], dtype=jnp.int32),
            meas_ctx,
            trial_data,
            chol_batch_size=sampling.guide_chol_batch_size,
        )
    total = jnp.sum(terms, axis=0)
    return jnp.stack(
        (
            common.theta_f,
            common.e0_f_base + total[0],
            common.h_t_f_base + total[1],
            common.e0_base + total[2],
        )
    )


def _combine(h0, components):
    del h0
    theta_f, e0_f, h_t_f, e0 = jnp.moveaxis(components, -1, 0)
    return e0_f + h_t_f - theta_f * e0


def _project_energy_terms(theta_f, terms):
    return terms[..., 0] + terms[..., 1] - theta_f * terms[..., 2]


def _chol_terms_for_walkers(
    common,
    chol_indices,
    meas_ctx,
    trial_data,
    *,
    n_chunks,
):
    return wk.vmap_chunked(
        lambda common_i: _chol_terms(
            common_i,
            meas_ctx.cholbar_a[chol_indices],
            meas_ctx.cholbar_b[chol_indices],
            meas_ctx,
            trial_data,
        ),
        n_chunks=n_chunks,
    )(common)


def _head_chol_moments_for_walkers(
    common,
    chol_indices,
    meas_ctx,
    trial_data,
    *,
    n_walker_chunks,
    chol_batch_size,
    theta_reference,
    compute_projected_moment,
):
    """Stream the exact head sum and its per-walker projected squared norm."""

    n_walkers = int(common.theta_f.shape[0])
    component_dtype = jnp.result_type(
        common.theta_f,
        common.e0_f_base,
        common.h_t_f_base,
        common.e0_base,
    )
    zero_total = jnp.zeros((n_walkers, 3), dtype=component_dtype)
    zero_moment = jnp.zeros((n_walkers,), dtype=jnp.float64)
    head_size = int(chol_indices.shape[0])
    if head_size == 0:
        return zero_total, zero_moment

    batch_size = min(chol_batch_size, head_size)
    n_batches = math.ceil(head_size / batch_size)
    padded_size = n_batches * batch_size
    padded_indices = jnp.pad(
        chol_indices, (0, padded_size - head_size)
    ).reshape(n_batches, batch_size)
    valid = (jnp.arange(padded_size) < head_size).reshape(n_batches, batch_size)

    def scan_batch(carry, xs):
        total, projected_moment = carry
        indices_i, valid_i = xs
        terms_i = _chol_terms_for_walkers(
            common,
            indices_i,
            meas_ctx,
            trial_data,
            n_chunks=n_walker_chunks,
        )
        terms_i = jnp.where(valid_i[None, :, None], terms_i, 0.0)
        total = total + jnp.sum(terms_i, axis=1)
        if compute_projected_moment:
            projected_i = _project_energy_terms(theta_reference, terms_i)
            projected_moment = projected_moment + jnp.sum(
                jnp.abs(projected_i) ** 2,
                axis=1,
                dtype=jnp.float64,
            )
        return (total, projected_moment), None

    result, _ = lax.scan(
        scan_batch,
        (zero_total, zero_moment),
        (padded_indices, valid),
    )
    return result


def _chol_pair_terms(
    common,
    sample_walker,
    sample_chol,
    meas_ctx,
    trial_data,
    *,
    n_chunks,
):
    return wk.vmap_chunked(
        lambda walker_i, chol_i: _chol_terms(
            tree_util.tree_map(lambda value: value[walker_i], common),
            meas_ctx.cholbar_a[chol_i][None],
            meas_ctx.cholbar_b[chol_i][None],
            meas_ctx,
            trial_data,
        )[0],
        n_chunks=n_chunks,
        in_axes=(0, 0),
    )(sample_walker, sample_chol)


def _chol_index_terms(
    common,
    chol_indices,
    meas_ctx,
    trial_data,
    *,
    chol_batch_size,
):
    """Evaluate one walker's indexed Cholesky terms in bounded batches."""

    n_chol = int(chol_indices.shape[0])
    n_chunks = max(1, math.ceil(n_chol / chol_batch_size))
    return wk.vmap_chunked(
        lambda chol_i: _chol_terms(
            common,
            meas_ctx.cholbar_a[chol_i][None],
            meas_ctx.cholbar_b[chol_i][None],
            meas_ctx,
            trial_data,
        )[0],
        n_chunks=n_chunks,
    )(chol_indices)


def _configure_sampling(meas_ctx, sampling, scores):
    n_chol = int(scores.shape[0])
    if sampling.chol_head_size > n_chol:
        raise ValueError("chol_head_size exceeds the number of Cholesky vectors.")
    order = jnp.argsort(-scores) if sampling.rank_head_by_guide else jnp.arange(n_chol)
    head = jnp.sort(order[: sampling.chol_head_size]).astype(jnp.int32)
    tail = jnp.sort(order[sampling.chol_head_size :]).astype(jnp.int32)
    if int(tail.shape[0]) == 0:
        probability = jnp.empty((0,), dtype=jnp.float64)
    else:
        probability = jnp.maximum(scores[tail], 1.0e-300)
        probability /= jnp.sum(probability, dtype=jnp.float64)
        mix = sampling.tail_probability_uniform_mix
        if mix:
            probability = (1.0 - mix) * probability + mix / tail.shape[0]
    return replace(
        meas_ctx,
        reference_chol_scores=scores,
        chol_head_indices=head,
        chol_tail_indices=tail,
        chol_tail_prob=probability,
        component_sampling=sampling,
    )


def _reference_scores(
    ham_data,
    meas_ctx,
    trial_data,
    *,
    chol_batch_size,
):
    walker = (trial_data.mo_t_a, trial_data.mo_coeff_b @ trial_data.mo_t_b)
    common = _energy_common(walker, ham_data, meas_ctx, trial_data)
    terms = _chol_index_terms(
        common,
        jnp.arange(meas_ctx.cholbar_a.shape[0], dtype=jnp.int32),
        meas_ctx,
        trial_data,
        chol_batch_size=chol_batch_size,
    )
    projected = _project_energy_terms(common.theta_f, terms)
    return jnp.maximum(jnp.abs(jnp.real(projected)).astype(jnp.float64), 1.0e-300)


def build_lno_ptuccsd_mode_meas_ctx(
    ham_data,
    trial_data,
    *,
    weight_a,
    weight_b,
    cfg,
    component_sampling=None,
    mode_batch_size=64,
):
    if not isinstance(ham_data, HamCholUhf):
        raise ValueError("split LNO PT-UCCSD mode estimators require HamCholUhf.")
    if mode_batch_size <= 0:
        raise ValueError("mode_batch_size must be positive.")
    noa, nob = trial_data.nocc
    nva, nvb = trial_data.nvir
    weight_a, weight_b = jnp.asarray(weight_a), jnp.asarray(weight_b)
    if weight_a.shape != (noa, noa) or weight_b.shape != (nob, nob):
        raise ValueError("LNO weights must match the alpha and beta occupied spaces.")
    if jnp.issubdtype(weight_a.dtype, jnp.complexfloating) or jnp.issubdtype(
        weight_b.dtype, jnp.complexfloating
    ):
        raise ValueError("LNO retained-mode contractions currently require real weights.")
    exp_t1_a, exp_mt1_a = _thouless_transform(trial_data.mo_t_a, noa)
    exp_t1_b, exp_mt1_b = _thouless_transform(trial_data.mo_t_b, nob)
    h1bar_a = exp_t1_a @ ham_data.h1_a @ exp_mt1_a
    h1bar_b = exp_t1_b @ ham_data.h1_b @ exp_mt1_b
    cholbar_a = jnp.einsum(
        "pr,grs,sq->gpq", exp_t1_a, ham_data.chol_a, exp_mt1_a, optimize="optimal"
    )
    cholbar_b = jnp.einsum(
        "pr,grs,sq->gpq", exp_t1_b, ham_data.chol_b, exp_mt1_b, optimize="optimal"
    )
    fockbar_a, fockbar_b = _ufock(
        h1bar_a, h1bar_b, cholbar_a, cholbar_b, noa, nob
    )
    rank = trial_data.mode_rank
    da, _ = trial_data.pair_dim
    modes_a = trial_data.modes[:, :da].reshape(rank, noa, nva)
    modes_b = trial_data.modes[:, da:].reshape(rank, nob, nvb)
    projected_a = jnp.einsum(
        "ria,ik->rka", modes_a, weight_a.astype(modes_a.dtype), optimize="optimal"
    )
    projected_b = jnp.einsum(
        "ria,ik->rka", modes_b, weight_b.astype(modes_b.dtype), optimize="optimal"
    )
    e0t1_f = _fragment_t1_reference_energy(
        exp_t1_a[:noa, noa:],
        exp_t1_b[:nob, nob:],
        ham_data.chol_a,
        ham_data.chol_b,
        weight_a,
        weight_b,
        noa,
        nob,
    )
    result = LnoPtuccsdModeMeasCtx(
        exp_t1_a=exp_t1_a,
        exp_t1_b=exp_t1_b,
        h1bar_a=h1bar_a,
        h1bar_b=h1bar_b,
        cholbar_a=cholbar_a,
        cholbar_b=cholbar_b,
        fockbar_a=fockbar_a,
        fockbar_b=fockbar_b,
        projected_modes_a=projected_a,
        projected_modes_b=projected_b,
        weight_a=weight_a,
        weight_b=weight_b,
        e0t1_f=e0t1_f,
        reference_chol_scores=jnp.empty((0,), dtype=jnp.float64),
        chol_head_indices=jnp.empty((0,), dtype=jnp.int32),
        chol_tail_indices=jnp.empty((0,), dtype=jnp.int32),
        chol_tail_prob=jnp.empty((0,), dtype=jnp.float64),
        cfg=cfg,
        component_sampling=None,
        mode_batch_size=int(mode_batch_size),
    )
    if component_sampling is None:
        return result
    return _configure_sampling(
        result,
        component_sampling,
        _reference_scores(
            ham_data,
            result,
            trial_data,
            chol_batch_size=component_sampling.guide_chol_batch_size,
        ),
    )


def pair_sampled_lno_ptuccsd_mode_block_components(
    walkers,
    candidate_weights,
    rng_key,
    n_chunks,
    ham_data,
    meas_ctx,
    trial_data,
):
    sampling = meas_ctx.component_sampling
    if sampling is None:
        raise ValueError("pair-sampled LNO components require a sampling config.")
    n_walkers = int(walkers[0].shape[0])
    if walkers[1].shape[0] != n_walkers or candidate_weights.shape != (n_walkers,):
        raise ValueError("walker population and candidate weights are inconsistent.")
    common = wk.vmap_chunked(
        _energy_common, n_chunks=n_chunks, in_axes=(0, None, None, None)
    )(walkers, ham_data, meas_ctx, trial_data)
    finite = jnp.isfinite(candidate_weights)
    for value in tree_util.tree_leaves(common):
        finite &= jnp.all(jnp.isfinite(value.reshape(n_walkers, -1)), axis=1)
    preliminary_weights = jnp.where(finite, candidate_weights, 0.0)
    preliminary_weight = jnp.sum(preliminary_weights)
    preliminary_safe = jnp.where(preliminary_weight == 0.0, 1.0, preliminary_weight)
    theta_reference = jnp.sum(preliminary_weights * common.theta_f) / preliminary_safe
    theta_reference = jnp.where(preliminary_weight == 0.0, 0.0, theta_reference)
    head_batch_size = sampling.head_chol_batch_size
    if head_batch_size <= 0:
        head_batch_size = sampling.guide_chol_batch_size
    head_sum, head_projected_moment = _head_chol_moments_for_walkers(
        common,
        meas_ctx.chol_head_indices,
        meas_ctx,
        trial_data,
        n_walker_chunks=n_chunks,
        chol_batch_size=head_batch_size,
        theta_reference=theta_reference,
        compute_projected_moment=sampling.walker_guide_policy == "head_rms",
    )
    exact_components = jnp.stack(
        (
            common.theta_f,
            common.e0_f_base + head_sum[:, 0],
            common.h_t_f_base + head_sum[:, 1],
            common.e0_base + head_sum[:, 2],
        ),
        axis=1,
    )
    valid = finite & jnp.all(jnp.isfinite(exact_components), axis=1)
    weights = jnp.where(valid, candidate_weights, 0.0)
    numerator = jnp.sum(
        weights[:, None] * jnp.where(valid[:, None], exact_components, 0.0), axis=0
    )
    weight = jnp.sum(weights)
    abs_weights = jnp.abs(weights).astype(jnp.float64)
    abs_sum = jnp.sum(abs_weights, dtype=jnp.float64)
    abs_safe = jnp.where(abs_sum == 0.0, 1.0, abs_sum)
    abs_probability = abs_weights / abs_safe
    abs_probability = jnp.where(
        abs_sum == 0.0,
        jnp.full_like(abs_probability, 1.0 / n_walkers),
        abs_probability,
    )
    if sampling.walker_guide_policy == "head_rms":
        weight_safe = jnp.where(weight == 0.0, 1.0, weight)
        head_scores = jnp.abs(weights / weight_safe) * jnp.sqrt(
            jnp.maximum(head_projected_moment, 0.0)
        )
        head_scores = jnp.where(valid & jnp.isfinite(head_scores), head_scores, 0.0)
        score_sum = jnp.sum(head_scores, dtype=jnp.float64)
        guided = head_scores / jnp.where(score_sum == 0.0, 1.0, score_sum)
        guided = jnp.where(score_sum == 0.0, abs_probability, guided)
        mix = sampling.walker_guide_weight_mix
        walker_probability = mix * abs_probability + (1.0 - mix) * guided
    else:
        walker_probability = abs_probability
    diagnostics = {
        d_pt_estimator_phase_coherence: jnp.where(
            abs_sum == 0.0, 0.0, jnp.abs(weight) / abs_safe
        ),
        d_pt_walker_proposal_ess: jnp.where(
            abs_sum == 0.0,
            0.0,
            1.0 / jnp.sum(walker_probability**2, dtype=jnp.float64),
        ),
    }
    tail_size = int(meas_ctx.chol_tail_indices.shape[0])
    if tail_size == 0:
        if sampling.track_half_sample_diagnostic:
            diagnostics[d_pt_component_sampling_noise_real] = jnp.asarray(0.0)
            diagnostics[d_pt_component_sampling_noise_imag] = jnp.asarray(0.0)
        return BlockComponentEstimate(weight, numerator, diagnostics)

    def sample_tail(key):
        key_walker, key_chol = jax.random.split(key)
        sample_walker = jax.random.choice(
            key_walker,
            n_walkers,
            shape=(sampling.pair_sample_size,),
            replace=True,
            p=walker_probability,
        )
        sample_chol_rel = jax.random.choice(
            key_chol,
            tail_size,
            shape=(sampling.pair_sample_size,),
            replace=True,
            p=meas_ctx.chol_tail_prob,
        )
        sample_chol = meas_ctx.chol_tail_indices[sample_chol_rel]
        walker_batch = (n_walkers + n_chunks - 1) // n_chunks
        pair_chunks = (sampling.pair_sample_size + walker_batch - 1) // walker_batch
        terms = _chol_pair_terms(
            common,
            sample_walker,
            sample_chol,
            meas_ctx,
            trial_data,
            n_chunks=pair_chunks,
        )
        samples = weights[sample_walker, None] * terms
        samples /= walker_probability[sample_walker, None]
        samples /= meas_ctx.chol_tail_prob[sample_chol_rel, None]
        tail_numerator = jnp.mean(samples, axis=0)
        if not sampling.track_half_sample_diagnostic:
            return tail_numerator, jnp.zeros_like(tail_numerator)
        first_size = sampling.pair_sample_size // 2
        second_size = sampling.pair_sample_size - first_size
        difference = jnp.mean(samples[:first_size], axis=0)
        difference -= jnp.mean(samples[first_size:], axis=0)
        scale = math.sqrt(first_size * second_size) / sampling.pair_sample_size
        return tail_numerator, scale * difference

    zero_tail = jnp.zeros((3,), dtype=numerator.dtype)
    tail_numerator, half_difference = lax.cond(
        abs_sum > 0.0,
        sample_tail,
        lambda key: (zero_tail, zero_tail),
        rng_key,
    )
    numerator = numerator.at[1:].add(tail_numerator)
    if sampling.track_half_sample_diagnostic:
        weight_safe = jnp.where(weight == 0.0, 1.0, weight)
        energy_difference = _project_energy_terms(
            numerator[0] / weight_safe, half_difference / weight_safe
        )
        diagnostics[d_pt_component_sampling_noise_real] = jnp.real(energy_difference)
        diagnostics[d_pt_component_sampling_noise_imag] = jnp.imag(energy_difference)
    return BlockComponentEstimate(weight, numerator, diagnostics)


def make_lno_ptuccsd_mode_estimator_ops(
    sys: System,
    weight_a: jax.Array,
    weight_b: jax.Array,
    *,
    memory_mode: str = "high",
    mixed_precision: bool = True,
    testing: bool = False,
    component_sampling: PtuccsdModePairSamplingCfg | None = None,
    mode_batch_size: int = 64,
) -> EstimatorOps:
    if sys.walker_kind.lower() != "unrestricted":
        raise ValueError("split LNO PT-UCCSD mode estimators require UHF walkers.")
    exact_testing = testing or not mixed_precision
    cfg = PtuccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if exact_testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if exact_testing else jnp.complex64,
    )
    return EstimatorOps(
        reference_overlap=reference_overlap_u,
        components=_components,
        combine_energy=_combine,
        component_names=("theta_F", "e0_F", "hT_F", "e0"),
        build_estimator_ctx=lambda ham_data, trial_data: (
            build_lno_ptuccsd_mode_meas_ctx(
                ham_data,
                trial_data,
                weight_a=weight_a,
                weight_b=weight_b,
                cfg=cfg,
                component_sampling=component_sampling,
                mode_batch_size=mode_batch_size,
            )
        ),
        block_components=(
            pair_sampled_lno_ptuccsd_mode_block_components
            if component_sampling is not None
            else None
        ),
        use_for_population_control=False,
    )


__all__ = [
    "LnoPtuccsdModeMeasCtx",
    "build_lno_ptuccsd_mode_meas_ctx",
    "make_lno_ptuccsd_mode_estimator_ops",
    "pair_sampled_lno_ptuccsd_mode_block_components",
]
