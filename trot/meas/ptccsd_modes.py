from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import tree_util

from .. import walkers as wk
from ..core.ops import BlockComponentEstimate, EstimatorOps, MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.cisd_modes import mode_apply, mode_quadratic
from ..trial.ptccsd_modes import (
    PtccsdModeTrial,
    PtccsdThoulessModeTrial,
    det_overlap_thouless_r,
    greenp_thouless,
    greens_pt_r,
    half_green_thouless_r,
    hf_overlap_r,
    overlap_pt_r,
    overlap_ptccsd_thouless_r,
)
from .cisd_modes import mode_quadratic_matrices
from .ptccsd import o_pt_components
from .pt2ccsd import combine_first_order_energy

d_pt_component_sampling_noise_real = "pt_component_sampling_noise_real"
d_pt_component_sampling_noise_imag = "pt_component_sampling_noise_imag"
d_pt_estimator_phase_coherence = "pt_estimator_phase_coherence"
d_pt_walker_proposal_ess = "pt_walker_proposal_ess"


@dataclass(frozen=True)
class PtccsdModeMeasCfg:
    """Precision and memory policy for mode-native PT-CCSD contractions."""

    memory_mode: Literal["low", "high"] = "high"
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64


@dataclass(frozen=True)
class PtccsdModePairSamplingCfg:
    """Fixed walker--Cholesky sampling policy for the RCC PT component numerator."""

    chol_head_size: int
    pair_sample_size: int
    rank_head_by_guide: bool = True
    guide_chol_batch_size: int = 16
    head_chol_batch_size: int = 16
    pair_sample_batch_size: int = 256
    tail_probability_uniform_mix: float = 0.05
    track_half_sample_diagnostic: bool = False

    def __post_init__(self) -> None:
        if self.chol_head_size < 0:
            raise ValueError("chol_head_size must be nonnegative.")
        if self.pair_sample_size <= 0:
            raise ValueError("pair_sample_size must be positive.")
        if self.guide_chol_batch_size <= 0:
            raise ValueError("guide_chol_batch_size must be positive.")
        if self.head_chol_batch_size <= 0:
            raise ValueError("head_chol_batch_size must be positive.")
        if self.pair_sample_batch_size <= 0:
            raise ValueError("pair_sample_batch_size must be positive.")
        if not 0.0 <= self.tail_probability_uniform_mix <= 1.0:
            raise ValueError("tail_probability_uniform_mix must lie in [0, 1].")
        if self.track_half_sample_diagnostic and self.pair_sample_size < 2:
            raise ValueError(
                "track_half_sample_diagnostic requires pair_sample_size to be at least two."
            )


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtccsdModeMeasCtx:
    rot_chol: jax.Array
    l_t1: jax.Array
    n_mode_chunks: int
    cfg: PtccsdModeMeasCfg

    def tree_flatten(self):
        return (self.rot_chol, self.l_t1), (self.n_mode_chunks, self.cfg)

    @classmethod
    def tree_unflatten(cls, aux, children):
        n_mode_chunks, cfg = aux
        rot_chol, l_t1 = children
        return cls(
            rot_chol=rot_chol,
            l_t1=l_t1,
            n_mode_chunks=n_mode_chunks,
            cfg=cfg,
        )


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtccsdThoulessModeMeasCtx:
    rot_chol: jax.Array
    reference_chol_scores: jax.Array
    chol_head_indices: jax.Array
    chol_tail_indices: jax.Array
    chol_tail_prob: jax.Array
    n_mode_chunks: int
    cfg: PtccsdModeMeasCfg
    component_sampling: PtccsdModePairSamplingCfg | None

    @property
    def memory_mode(self) -> Literal["low", "high"]:
        return self.cfg.memory_mode

    def tree_flatten(self):
        children = (
            self.rot_chol,
            self.reference_chol_scores,
            self.chol_head_indices,
            self.chol_tail_indices,
            self.chol_tail_prob,
        )
        return children, (self.n_mode_chunks, self.cfg, self.component_sampling)

    @classmethod
    def tree_unflatten(cls, aux, children):
        n_mode_chunks, cfg, component_sampling = aux
        (
            rot_chol,
            reference_chol_scores,
            chol_head_indices,
            chol_tail_indices,
            chol_tail_prob,
        ) = children
        return cls(
            rot_chol=rot_chol,
            reference_chol_scores=reference_chol_scores,
            chol_head_indices=chol_head_indices,
            chol_tail_indices=chol_tail_indices,
            chol_tail_prob=chol_tail_prob,
            n_mode_chunks=n_mode_chunks,
            cfg=cfg,
            component_sampling=component_sampling,
        )


class PtccsdThoulessModeEnergyCommon(NamedTuple):
    """Per-walker PT sufficient-statistic data shared by Cholesky terms.

    ``h_t_base`` excludes the connected, Cholesky-resolved T2 residual. This
    decomposition is an algebraic building block for a future population-level
    estimator; it is not a per-walker local-energy definition.
    """

    half_green: jax.Array
    greenp: jax.Array
    t2_green: jax.Array
    theta: jax.Array
    electronic_0: jax.Array
    h_t_base: jax.Array


def build_ptccsd_mode_meas_ctx(
    ham_data: HamChol,
    trial_data: PtccsdModeTrial,
    *,
    n_mode_chunks: int = 1,
    cfg: PtccsdModeMeasCfg = PtccsdModeMeasCfg(),
) -> PtccsdModeMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("PT-CCSD mode kernels require a restricted Hamiltonian.")
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    rot_chol = ham_data.chol[:, : trial_data.nocc, :]
    l_t1 = jnp.einsum(
        "git,pt->gip",
        ham_data.chol[:, :, trial_data.nocc :],
        trial_data.t1,
        optimize="optimal",
    )
    return PtccsdModeMeasCtx(
        rot_chol=rot_chol,
        l_t1=l_t1,
        n_mode_chunks=min(int(n_mode_chunks), trial_data.mode_rank),
        cfg=cfg,
    )


def build_ptccsd_thouless_mode_meas_ctx(
    ham_data: HamChol,
    trial_data: PtccsdThoulessModeTrial,
    *,
    n_mode_chunks: int = 1,
    memory_mode: Literal["low", "high"] = "high",
    cfg: PtccsdModeMeasCfg | None = None,
    component_sampling: PtccsdModePairSamplingCfg | None = None,
) -> PtccsdThoulessModeMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("PT-CCSD Thouless mode kernels require a restricted Hamiltonian.")
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    if memory_mode not in {"low", "high"}:
        raise ValueError("memory_mode must be 'low' or 'high'.")
    if cfg is None:
        cfg = PtccsdModeMeasCfg(memory_mode=memory_mode)
    elif cfg.memory_mode != memory_mode:
        raise ValueError("cfg.memory_mode and memory_mode must agree.")
    n_chol = int(ham_data.chol.shape[0])
    if component_sampling is not None and component_sampling.chol_head_size > n_chol:
        raise ValueError(
            f"chol_head_size must not exceed the number of Cholesky vectors ({n_chol})."
        )
    meas_ctx = PtccsdThoulessModeMeasCtx(
        rot_chol=jnp.einsum(
            "pi,gpq->giq",
            trial_data.mo_t.conj(),
            ham_data.chol,
            optimize="optimal",
        ),
        reference_chol_scores=jnp.empty((0,), dtype=jnp.float64),
        chol_head_indices=jnp.empty((0,), dtype=jnp.int32),
        chol_tail_indices=jnp.empty((0,), dtype=jnp.int32),
        chol_tail_prob=jnp.empty((0,), dtype=jnp.float64),
        n_mode_chunks=min(int(n_mode_chunks), trial_data.mode_rank),
        cfg=cfg,
        component_sampling=None,
    )
    if component_sampling is None:
        return meas_ctx
    reference_scores = _build_ptccsd_thouless_reference_chol_scores(
        ham_data,
        meas_ctx,
        trial_data,
        chol_batch_size=component_sampling.guide_chol_batch_size,
    )
    return configure_ptccsd_mode_pair_sampling(
        meas_ctx,
        component_sampling,
        reference_scores,
    )


def _chol_contract(
    chol: jax.Array,
    matrix: jax.Array,
    cfg: PtccsdModeMeasCfg,
) -> jax.Array:
    """Contract real Cholesky vectors with a complex matrix in mixed precision."""

    chol_r = chol.astype(cfg.mixed_real_dtype)
    matrix_r = jnp.real(matrix).astype(cfg.mixed_real_dtype)
    matrix_i = jnp.imag(matrix).astype(cfg.mixed_real_dtype)
    real_part = jnp.einsum("gij,ij->g", chol_r, matrix_r, optimize="optimal")
    imag_part = jnp.einsum("gij,ij->g", chol_r, matrix_i, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    return real_part.astype(cfg.mixed_complex_dtype) + imag_unit * imag_part.astype(
        cfg.mixed_complex_dtype
    )


def _energy_gl_batched(
    green: jax.Array,
    chol: jax.Array,
    cfg: PtccsdModeMeasCfg,
) -> jax.Array:
    green_r = jnp.real(green).astype(cfg.mixed_real_dtype)
    green_i = jnp.imag(green).astype(cfg.mixed_real_dtype)
    chol_r = chol.astype(cfg.mixed_real_dtype)
    real_part = jnp.einsum("pr,gqr->gpq", green_r, chol_r, optimize="optimal")
    imag_part = jnp.einsum("pr,gqr->gpq", green_i, chol_r, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    return real_part.astype(cfg.mixed_complex_dtype) + imag_unit * imag_part.astype(
        cfg.mixed_complex_dtype
    )


def _energy_gl_scalar(
    green: jax.Array,
    chol_i: jax.Array,
    cfg: PtccsdModeMeasCfg,
) -> jax.Array:
    green_r = jnp.real(green).astype(cfg.mixed_real_dtype)
    green_i = jnp.imag(green).astype(cfg.mixed_real_dtype)
    chol_i_r = chol_i.astype(cfg.mixed_real_dtype)
    real_part = jnp.einsum("ir,qr->iq", green_r, chol_i_r, optimize="optimal")
    imag_part = jnp.einsum("ir,qr->iq", green_i, chol_i_r, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    return real_part.astype(cfg.mixed_complex_dtype) + imag_unit * imag_part.astype(
        cfg.mixed_complex_dtype
    )


def _pt_green_blocks(
    walker: jax.Array,
    trial_data: PtccsdModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    green = greens_pt_r(walker, trial_data)
    green_occ = green[:, trial_data.nocc :]
    greenp = jnp.vstack(
        [
            green_occ,
            -jnp.eye(trial_data.nvir, dtype=green.dtype),
        ]
    )
    return green, green_occ, greenp


def _thouless_green_blocks(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    _, green, _, green_occ, greenp = _thouless_half_green_blocks(walker, trial_data)
    return green, green_occ, greenp


def _thouless_half_green_blocks(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return half- and full-Green blocks for a Thouless reference.

    If ``C`` is ``trial_data.mo_t`` and ``R`` is the half Green function, the
    full transition Green function factors as ``G = C.conj() @ R``.  Keeping
    ``R`` makes it possible for the energy kernel to avoid intermediates with
    shape ``(n_chol, norb, norb)``.
    """

    half_green = half_green_thouless_r(walker, trial_data)
    green = trial_data.mo_t.conj() @ half_green
    green_rows = green[: trial_data.nocc, :]
    green_occ = green[: trial_data.nocc, trial_data.nocc :]
    greenp = greenp_thouless(green, trial_data)
    return half_green, green, green_rows, green_occ, greenp


def _mode_t2_green(
    trial_data,
    green: jax.Array,
    green_occ: jax.Array,
    greenp: jax.Array,
    *,
    green_rows: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    del green
    _, t2_green, theta2 = _mode_t2_green_factors(
        trial_data,
        green_occ,
        greenp,
        green_rows=green_rows,
    )
    return t2_green, theta2


def _mode_t2_green_factors(
    trial_data,
    green_occ: jax.Array,
    greenp: jax.Array,
    *,
    green_rows: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the left factor, full correction, and scalar T2 contraction."""

    projections, kg = mode_apply(trial_data, green_occ)
    theta2 = mode_quadratic(trial_data, green_occ, projections)
    t2_left = greenp @ kg.T
    t2_green = t2_left @ green_rows
    return t2_left, t2_green, theta2


def force_bias_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdModeMeasCtx,
    trial_data: PtccsdModeTrial,
) -> jax.Array:
    green, green_occ, greenp = _pt_green_blocks(walker, trial_data)
    f0 = 2.0 * jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")

    t1gp = jnp.einsum("pt,it->pi", trial_data.t1, greenp, optimize="optimal")
    gt1gp = jnp.einsum("pj,pi->ij", green, t1gp, optimize="optimal")
    t2_green, _ = _mode_t2_green(
        trial_data,
        green,
        green_occ,
        greenp,
        green_rows=green,
    )
    connected = _chol_contract(
        ham_data.chol,
        -2.0 * t2_green - 2.0 * gt1gp,
        meas_ctx.cfg,
    )
    return f0 + connected


def components_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdModeMeasCtx,
    trial_data: PtccsdModeTrial,
) -> jax.Array:
    """Return unweighted ``[theta, electronic_0, h_t]`` PT sufficient statistics."""

    green, green_occ, greenp = _pt_green_blocks(walker, trial_data)
    h1 = ham_data.h1
    chol = ham_data.chol
    nocc = trial_data.nocc

    hg = jnp.einsum("pj,pj->", h1[:nocc, :], green, optimize="optimal")
    e1_0 = 2.0 * hg
    lg = jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")
    lg1 = jnp.einsum("gpj,qj->gpq", meas_ctx.rot_chol, green, optimize="optimal")
    e2_0 = 2.0 * (lg @ lg) - jnp.sum(lg1 * jnp.swapaxes(lg1, -1, -2))
    electronic_0 = e1_0 + e2_0

    t1g = jnp.einsum("pt,pt->", trial_data.t1, green_occ, optimize="optimal")
    theta1 = 2.0 * t1g
    t1_green = (greenp @ trial_data.t1.T) @ green
    e1_1 = 4.0 * t1g * hg - 2.0 * jnp.einsum(
        "ij,ij->", h1, t1_green, optimize="optimal"
    )

    t2_green, theta2 = _mode_t2_green(
        trial_data,
        green,
        green_occ,
        greenp,
        green_rows=green,
    )
    e1_2 = 2.0 * hg * theta2 - 2.0 * jnp.einsum(
        "ij,ij->", h1, t2_green, optimize="optimal"
    )

    lt1g = _chol_contract(chol, t1_green, meas_ctx.cfg)
    t1g1 = trial_data.t1 @ green[:, nocc:].T
    l_t1g = jnp.einsum("gia,qi->gaq", meas_ctx.l_t1, green, optimize="optimal")
    e2_1_3_1 = jnp.einsum("gpq,gqa,ap->", lg1, lg1, t1g1, optimize="optimal")
    e2_1_3_2 = -jnp.einsum("gaq,gqa->", l_t1g, lg1, optimize="optimal")
    e2_1 = e2_0 * theta1 + 2.0 * (-2.0 * (lt1g @ lg) + e2_1_3_1 + e2_1_3_2)

    lt2g = _chol_contract(chol, t2_green, meas_ctx.cfg)
    lt2_green = jnp.einsum(
        "gpi,ji->gpj",
        meas_ctx.rot_chol.astype(meas_ctx.cfg.mixed_complex_dtype),
        t2_green.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    gl = _energy_gl_batched(green, chol, meas_ctx.cfg)
    e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lt2_green, optimize="optimal")
    glgp = jnp.einsum(
        "gpi,it->gpt",
        gl,
        greenp.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_2_3 = jnp.sum(
        mode_quadratic_matrices(
            trial_data,
            glgp,
            n_mode_chunks=meas_ctx.n_mode_chunks,
        )
    )
    e2_2 = e2_0 * theta2 + 4.0 * (-(lt2g @ lg) + e2_2_2_2) + e2_2_3

    theta = theta1 + theta2
    h_t = e1_1 + e1_2 + e2_1 + e2_2
    return jnp.stack([theta, electronic_0, h_t])


def energy_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdModeMeasCtx,
    trial_data: PtccsdModeTrial,
) -> jax.Array:
    theta, electronic_0, h_t = components_pt_rw_rh(walker, ham_data, meas_ctx, trial_data)
    return ham_data.h0 + electronic_0 + h_t - theta * electronic_0


def inverse_guide_components_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdModeMeasCtx,
    trial_data: PtccsdModeTrial,
) -> jax.Array:
    components = components_pt_rw_rh(walker, ham_data, meas_ctx, trial_data)
    reweight = jnp.exp(-components[0])
    return jnp.concatenate([reweight[None], reweight * components])


def _force_bias_pt_thouless_rw_rh_full(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    """Reference full-Green implementation retained as a test oracle."""

    green, green_occ, greenp = _thouless_green_blocks(walker, trial_data)
    f0 = 2.0 * jnp.einsum("gpq,pq->g", ham_data.chol, green, optimize="optimal")
    t2_green, _ = _mode_t2_green(
        trial_data,
        green,
        green_occ,
        greenp,
        green_rows=green[: trial_data.nocc, :],
    )
    return f0 + _chol_contract(ham_data.chol, -2.0 * t2_green, meas_ctx.cfg)


def force_bias_pt_thouless_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    """Restricted exponential-guide force bias using a half Green function."""

    half_green, _, green_rows, green_occ, greenp = _thouless_half_green_blocks(
        walker,
        trial_data,
    )
    f0 = 2.0 * jnp.einsum(
        "giq,iq->g",
        meas_ctx.rot_chol,
        half_green,
        optimize="optimal",
    )
    _, t2_green, _ = _mode_t2_green_factors(
        trial_data,
        green_occ,
        greenp,
        green_rows=green_rows,
    )
    return f0 + _chol_contract(ham_data.chol, -2.0 * t2_green, meas_ctx.cfg)


def _components_pt_thouless_rw_rh_full(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    """Reference full-Green components retained as a test oracle."""

    green, green_occ, greenp = _thouless_green_blocks(walker, trial_data)
    h1 = ham_data.h1
    chol = ham_data.chol
    nocc = trial_data.nocc

    hg = jnp.einsum("pq,pq->", h1, green, optimize="optimal")
    e1_0 = 2.0 * hg
    t2_green, theta2 = _mode_t2_green(
        trial_data,
        green,
        green_occ,
        greenp,
        green_rows=green[:nocc, :],
    )
    e1_2 = 2.0 * hg * theta2 - 2.0 * jnp.einsum(
        "pq,pq->", h1, t2_green, optimize="optimal"
    )

    lg = jnp.einsum("gpq,pq->g", chol, green, optimize="optimal")
    gl = _energy_gl_batched(green, chol, meas_ctx.cfg)
    e2_0 = 2.0 * (lg @ lg) - jnp.sum(gl * jnp.swapaxes(gl, -1, -2))

    lt2g = _chol_contract(chol, t2_green, meas_ctx.cfg)
    lt2_green = jnp.einsum(
        "gpr,qr->gpq",
        chol.astype(meas_ctx.cfg.mixed_real_dtype),
        t2_green.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_2_2_2 = 0.5 * jnp.einsum("gpq,gpq->", gl, lt2_green, optimize="optimal")
    glgp = jnp.einsum(
        "gpi,it->gpt",
        gl[:, :nocc, :],
        greenp.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_2_3 = jnp.sum(
        mode_quadratic_matrices(
            trial_data,
            glgp,
            n_mode_chunks=meas_ctx.n_mode_chunks,
        )
    )
    e2_2 = e2_0 * theta2 + 4.0 * (-(lt2g @ lg) + e2_2_2_2) + e2_2_3

    electronic_0 = e1_0 + e2_0
    h_t = e1_2 + e2_2
    return jnp.stack([theta2, electronic_0, h_t])


def _ptccsd_thouless_mode_energy_common(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> PtccsdThoulessModeEnergyCommon:
    """Build exact per-walker data outside the connected T2 residual sum."""

    half_green, green, green_rows, green_occ, greenp = _thouless_half_green_blocks(
        walker,
        trial_data,
    )
    h1 = ham_data.h1
    hg = jnp.einsum("pq,pq->", h1, green, optimize="optimal")
    e1_0 = 2.0 * hg
    _, t2_green, theta2 = _mode_t2_green_factors(
        trial_data,
        green_occ,
        greenp,
        green_rows=green_rows,
    )
    e1_2 = 2.0 * hg * theta2 - 2.0 * jnp.einsum(
        "pq,pq->", h1, t2_green, optimize="optimal"
    )

    # Keep the determinant-reference two-body energy exact. Only the more
    # expensive connected T2 residual is split into Cholesky contributions.
    lg = jnp.einsum(
        "giq,iq->g",
        meas_ctx.rot_chol,
        half_green,
        optimize="optimal",
    )
    lg1 = jnp.einsum(
        "gip,jp->gij",
        meas_ctx.rot_chol,
        half_green,
        optimize="optimal",
    )
    e2_0 = 2.0 * (lg @ lg) - jnp.sum(lg1 * jnp.swapaxes(lg1, -1, -2))

    return PtccsdThoulessModeEnergyCommon(
        half_green=half_green,
        greenp=greenp,
        t2_green=t2_green,
        theta=theta2,
        electronic_0=e1_0 + e2_0,
        h_t_base=e1_2 + theta2 * e2_0,
    )


def _ptccsd_thouless_mode_chol_terms(
    common: PtccsdThoulessModeEnergyCommon,
    chol: jax.Array,
    rot_chol: jax.Array,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    """Return one walker's connected T2 residual for supplied Cholesky vectors."""

    reference_occ = trial_data.mo_t.conj()[: trial_data.nocc, :].astype(
        meas_ctx.cfg.mixed_complex_dtype
    )
    greenp_mixed = common.greenp.astype(meas_ctx.cfg.mixed_complex_dtype)
    t2_green_mixed = common.t2_green.astype(meas_ctx.cfg.mixed_complex_dtype)

    def scalar_term(chol_i: jax.Array, rot_chol_i: jax.Array) -> jax.Array:
        lg_i = jnp.einsum(
            "iq,iq->",
            rot_chol_i,
            common.half_green,
            optimize="optimal",
        )
        lt2g_i = _chol_contract(
            chol_i[None, ...], common.t2_green, meas_ctx.cfg
        )[0]
        gl_half_i = _energy_gl_scalar(common.half_green, chol_i, meas_ctx.cfg)
        lt2_half_i = jnp.einsum(
            "ir,qr->iq",
            rot_chol_i.astype(meas_ctx.cfg.mixed_complex_dtype),
            t2_green_mixed,
            optimize="optimal",
        )
        e222_i = 0.5 * jnp.einsum(
            "iq,iq->", gl_half_i, lt2_half_i, optimize="optimal"
        )
        gl_occ_i = reference_occ @ gl_half_i
        glgp_i = jnp.einsum(
            "pi,it->pt", gl_occ_i, greenp_mixed, optimize="optimal"
        )
        e223_i = mode_quadratic_matrices(
            trial_data,
            glgp_i[None, ...],
            n_mode_chunks=meas_ctx.n_mode_chunks,
        )[0]
        return 4.0 * (-lt2g_i * lg_i + e222_i) + e223_i

    if meas_ctx.memory_mode == "low":
        zero = jnp.zeros((), dtype=jnp.result_type(common.half_green, chol))

        def scan_term(carry, xs):
            chol_i, rot_chol_i = xs
            return carry, scalar_term(chol_i, rot_chol_i)

        _, terms = jax.lax.scan(scan_term, zero, (chol, rot_chol))
        return terms

    lg = jnp.einsum(
        "giq,iq->g",
        rot_chol,
        common.half_green,
        optimize="optimal",
    )
    lt2g = _chol_contract(chol, common.t2_green, meas_ctx.cfg)
    gl_half = _energy_gl_batched(common.half_green, chol, meas_ctx.cfg)
    lt2_half = jnp.einsum(
        "gir,qr->giq",
        rot_chol.astype(meas_ctx.cfg.mixed_complex_dtype),
        t2_green_mixed,
        optimize="optimal",
    )
    e222 = 0.5 * jnp.einsum(
        "giq,giq->g", gl_half, lt2_half, optimize="optimal"
    )
    gl_occ = jnp.einsum(
        "ji,gip->gjp",
        reference_occ,
        gl_half,
        optimize="optimal",
    )
    glgp = jnp.einsum(
        "gpi,it->gpt", gl_occ, greenp_mixed, optimize="optimal"
    )
    e223 = mode_quadratic_matrices(
        trial_data,
        glgp,
        n_mode_chunks=meas_ctx.n_mode_chunks,
    )
    return 4.0 * (-lt2g * lg + e222) + e223


def _ptccsd_thouless_mode_chol_terms_for_walkers(
    common: PtccsdThoulessModeEnergyCommon,
    chol: jax.Array,
    rot_chol: jax.Array,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
    *,
    n_chunks: int = 1,
) -> jax.Array:
    """Return a bounded walker-by-supplied-Cholesky residual batch."""

    return wk.vmap_chunked(
        lambda common_i: _ptccsd_thouless_mode_chol_terms(
            common_i,
            chol,
            rot_chol,
            meas_ctx,
            trial_data,
        ),
        n_chunks=n_chunks,
    )(common)


def _ptccsd_thouless_mode_chol_index_terms(
    common: PtccsdThoulessModeEnergyCommon,
    chol_indices: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
    *,
    n_chunks: int,
) -> jax.Array:
    """Return one walker's residual terms at arbitrary Cholesky indices."""

    return wk.vmap_chunked(
        lambda chol_i: _ptccsd_thouless_mode_chol_terms(
            common,
            ham_data.chol[chol_i][None, ...],
            meas_ctx.rot_chol[chol_i][None, ...],
            meas_ctx,
            trial_data,
        )[0],
        n_chunks=n_chunks,
    )(chol_indices)


def _ptccsd_thouless_mode_chol_index_sum_for_walkers(
    common: PtccsdThoulessModeEnergyCommon,
    chol_indices: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
    *,
    n_walker_chunks: int,
    chol_batch_size: int,
) -> jax.Array:
    """Stream an exact residual head without materializing every pair."""

    head_size = int(chol_indices.shape[0])
    if head_size == 0:
        return jnp.zeros_like(common.h_t_base)

    batch_size = head_size if chol_batch_size <= 0 else min(chol_batch_size, head_size)
    n_batches = math.ceil(head_size / batch_size)
    padded_size = n_batches * batch_size
    padded_indices = jnp.pad(chol_indices, (0, padded_size - head_size)).reshape(
        n_batches, batch_size
    )
    valid = (jnp.arange(padded_size) < head_size).reshape(n_batches, batch_size)

    def scan_batch(total, xs):
        indices_i, valid_i = xs
        terms_i = _ptccsd_thouless_mode_chol_terms_for_walkers(
            common,
            ham_data.chol[indices_i],
            meas_ctx.rot_chol[indices_i],
            meas_ctx,
            trial_data,
            n_chunks=n_walker_chunks,
        )
        terms_i = jnp.where(valid_i[None, :], terms_i, 0.0)
        return total + jnp.sum(terms_i, axis=1), None

    total, _ = jax.lax.scan(
        scan_batch,
        jnp.zeros_like(common.h_t_base),
        (padded_indices, valid),
    )
    return total


def _ptccsd_thouless_mode_chol_pair_terms(
    common: PtccsdThoulessModeEnergyCommon,
    sample_walker: jax.Array,
    sample_chol: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
    *,
    n_chunks: int = 1,
) -> jax.Array:
    """Evaluate gathered walker--Cholesky residual pairs in microbatches."""

    return wk.vmap_chunked(
        lambda walker_i, chol_i: _ptccsd_thouless_mode_chol_terms(
            tree_util.tree_map(lambda value: value[walker_i], common),
            ham_data.chol[chol_i][None, ...],
            meas_ctx.rot_chol[chol_i][None, ...],
            meas_ctx,
            trial_data,
        )[0],
        n_chunks=n_chunks,
        in_axes=(0, 0),
    )(sample_walker, sample_chol)


def _build_ptccsd_thouless_reference_chol_scores(
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
    *,
    chol_batch_size: int,
) -> jax.Array:
    """Build bounded reference-determinant scores for the connected residual."""

    n_chol = int(ham_data.chol.shape[0])
    common = _ptccsd_thouless_mode_energy_common(
        trial_data.mo_t,
        ham_data,
        meas_ctx,
        trial_data,
    )
    indices = jnp.arange(n_chol, dtype=jnp.int32)
    n_chunks = max(1, math.ceil(n_chol / chol_batch_size))
    terms = _ptccsd_thouless_mode_chol_index_terms(
        common,
        indices,
        ham_data,
        meas_ctx,
        trial_data,
        n_chunks=n_chunks,
    )
    return jnp.maximum(jnp.abs(terms).astype(jnp.float64), 1.0e-300)


def configure_ptccsd_mode_pair_sampling(
    meas_ctx: PtccsdThoulessModeMeasCtx,
    sampling: PtccsdModePairSamplingCfg,
    guide_scores: jax.Array,
) -> PtccsdThoulessModeMeasCtx:
    """Attach an exact head and strictly positive fixed tail proposal."""

    scores = jnp.asarray(guide_scores, dtype=jnp.float64)
    n_chol = int(meas_ctx.rot_chol.shape[0])
    if scores.shape != (n_chol,):
        raise ValueError(f"guide_scores must have shape {(n_chol,)}, got {scores.shape}.")
    if sampling.chol_head_size > n_chol:
        raise ValueError(
            f"chol_head_size must not exceed the number of Cholesky vectors ({n_chol})."
        )

    if sampling.rank_head_by_guide:
        order = jnp.argsort(-scores)
    else:
        order = jnp.arange(n_chol, dtype=jnp.int32)
    head_indices = jnp.sort(order[: sampling.chol_head_size]).astype(jnp.int32)
    tail_indices = jnp.sort(order[sampling.chol_head_size :]).astype(jnp.int32)
    if int(tail_indices.shape[0]) == 0:
        tail_prob = jnp.empty((0,), dtype=jnp.float64)
    else:
        tail_scores = jnp.maximum(scores[tail_indices], 1.0e-300)
        tail_prob = tail_scores / jnp.sum(tail_scores, dtype=jnp.float64)
        uniform_mix = sampling.tail_probability_uniform_mix
        if uniform_mix > 0.0:
            uniform_prob = jnp.full_like(tail_prob, 1.0 / tail_prob.shape[0])
            tail_prob = (1.0 - uniform_mix) * tail_prob + uniform_mix * uniform_prob
    return replace(
        meas_ctx,
        reference_chol_scores=scores,
        chol_head_indices=head_indices,
        chol_tail_indices=tail_indices,
        chol_tail_prob=tail_prob,
        component_sampling=sampling,
    )


def pair_sampled_ptccsd_block_components(
    walkers: jax.Array,
    candidate_weights: jax.Array,
    rng_key: jax.Array,
    n_chunks: int,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> BlockComponentEstimate:
    """Estimate the block PT component numerator with an exact denominator.

    ``theta`` and ``electronic_0`` remain exact. The exact head and sampled
    tail contribute only to the unnormalized ``h_t`` numerator; no per-walker
    scalar local energy is formed.
    """

    sampling = meas_ctx.component_sampling
    if sampling is None:
        raise ValueError(
            "pair_sampled_ptccsd_block_components requires a component sampling config."
        )
    n_walkers = int(walkers.shape[0])
    if candidate_weights.shape != (n_walkers,):
        raise ValueError(
            f"candidate_weights must have shape {(n_walkers,)}, got "
            f"{candidate_weights.shape}."
        )

    common = wk.vmap_chunked(
        _ptccsd_thouless_mode_energy_common,
        n_chunks=n_chunks,
        in_axes=(0, None, None, None),
    )(walkers, ham_data, meas_ctx, trial_data)
    head_sum = _ptccsd_thouless_mode_chol_index_sum_for_walkers(
        common,
        meas_ctx.chol_head_indices,
        ham_data,
        meas_ctx,
        trial_data,
        n_walker_chunks=n_chunks,
        chol_batch_size=sampling.head_chol_batch_size,
    )
    exact_components = jnp.stack(
        (
            common.theta,
            common.electronic_0,
            common.h_t_base + head_sum,
        ),
        axis=1,
    )

    finite_common = jnp.ones((n_walkers,), dtype=jnp.bool_)
    for value in tree_util.tree_leaves(common):
        finite_common = finite_common & jnp.all(
            jnp.isfinite(value.reshape(n_walkers, -1)), axis=1
        )
    finite_components = jnp.all(jnp.isfinite(exact_components), axis=1)
    finite_weights = jnp.isfinite(candidate_weights)
    valid = finite_common & finite_components & finite_weights
    estimator_weights = jnp.where(valid, candidate_weights, 0.0)
    safe_components = jnp.where(valid[:, None], exact_components, 0.0)
    estimator_weight = jnp.sum(estimator_weights)
    numerator = jnp.sum(estimator_weights[:, None] * safe_components, axis=0)

    abs_weights = jnp.abs(estimator_weights).astype(jnp.float64)
    abs_weight_sum = jnp.sum(abs_weights, dtype=jnp.float64)
    abs_weight_sum_safe = jnp.where(abs_weight_sum == 0.0, 1.0, abs_weight_sum)
    walker_prob = abs_weights / abs_weight_sum_safe
    walker_prob = jnp.where(
        abs_weight_sum == 0.0,
        jnp.full_like(walker_prob, 1.0 / n_walkers),
        walker_prob,
    )
    diagnostics: dict[str, jax.Array] = {
        d_pt_estimator_phase_coherence: jnp.where(
            abs_weight_sum == 0.0,
            0.0,
            jnp.abs(estimator_weight) / abs_weight_sum_safe,
        ),
        d_pt_walker_proposal_ess: jnp.where(
            abs_weight_sum == 0.0,
            0.0,
            1.0 / jnp.sum(walker_prob**2, dtype=jnp.float64),
        ),
    }

    tail_size = int(meas_ctx.chol_tail_indices.shape[0])
    if tail_size == 0:
        if sampling.track_half_sample_diagnostic:
            diagnostics[d_pt_component_sampling_noise_real] = jnp.asarray(
                0.0, dtype=jnp.float64
            )
            diagnostics[d_pt_component_sampling_noise_imag] = jnp.asarray(
                0.0, dtype=jnp.float64
            )
        return BlockComponentEstimate(
            weight=estimator_weight,
            numerator=numerator,
            diagnostics=diagnostics,
        )

    def sample_tail(key):
        key_walker, key_chol = jax.random.split(key)
        sample_walker = jax.random.choice(
            key_walker,
            n_walkers,
            shape=(sampling.pair_sample_size,),
            replace=True,
            p=walker_prob,
        )
        sample_chol_rel = jax.random.choice(
            key_chol,
            tail_size,
            shape=(sampling.pair_sample_size,),
            replace=True,
            p=meas_ctx.chol_tail_prob,
        )
        sample_chol = meas_ctx.chol_tail_indices[sample_chol_rel]
        pair_n_chunks = max(
            1,
            math.ceil(sampling.pair_sample_size / sampling.pair_sample_batch_size),
        )
        residual = _ptccsd_thouless_mode_chol_pair_terms(
            common,
            sample_walker,
            sample_chol,
            ham_data,
            meas_ctx,
            trial_data,
            n_chunks=pair_n_chunks,
        )
        importance_samples = (
            estimator_weights[sample_walker]
            * residual
            / (
                walker_prob[sample_walker]
                * meas_ctx.chol_tail_prob[sample_chol_rel]
            )
        )
        tail_numerator = jnp.mean(importance_samples)
        if not sampling.track_half_sample_diagnostic:
            return tail_numerator, jnp.zeros((), dtype=tail_numerator.dtype)

        first_size = sampling.pair_sample_size // 2
        second_size = sampling.pair_sample_size - first_size
        first_mean = jnp.mean(importance_samples[:first_size])
        second_mean = jnp.mean(importance_samples[first_size:])
        scale = math.sqrt(first_size * second_size) / sampling.pair_sample_size
        return tail_numerator, scale * (first_mean - second_mean)

    zero_tail = jnp.zeros((), dtype=numerator.dtype)
    tail_numerator, half_difference = jax.lax.cond(
        abs_weight_sum > 0.0,
        sample_tail,
        lambda key: (zero_tail, zero_tail),
        rng_key,
    )
    numerator = numerator.at[2].add(tail_numerator)
    if sampling.track_half_sample_diagnostic:
        estimator_weight_safe = jnp.where(
            estimator_weight == 0.0, 1.0, estimator_weight
        )
        normalized_difference = half_difference / estimator_weight_safe
        diagnostics[d_pt_component_sampling_noise_real] = jnp.real(
            normalized_difference
        )
        diagnostics[d_pt_component_sampling_noise_imag] = jnp.imag(
            normalized_difference
        )
    return BlockComponentEstimate(
        weight=estimator_weight,
        numerator=numerator,
        diagnostics=diagnostics,
    )


def components_pt_thouless_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    """Return PT2/Thouless components using half-Green Cholesky contractions.

    With ``G = C.conj() @ R``, the full-Green implementation forms Cholesky
    intermediates of shape ``(n_chol, norb, norb)``.  This implementation
    factors those contractions through ``R`` and keeps their largest shape at
    ``(n_chol, nocc, norb)`` without changing the estimator algebra.
    """

    common = _ptccsd_thouless_mode_energy_common(
        walker,
        ham_data,
        meas_ctx,
        trial_data,
    )
    residual = _ptccsd_thouless_mode_chol_terms(
        common,
        ham_data.chol,
        meas_ctx.rot_chol,
        meas_ctx,
        trial_data,
    )
    h_t = common.h_t_base + jnp.sum(residual)
    return jnp.stack([common.theta, common.electronic_0, h_t])


def energy_pt_thouless_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    theta2, electronic_0, h_t = components_pt_thouless_rw_rh(
        walker, ham_data, meas_ctx, trial_data
    )
    return ham_data.h0 + electronic_0 + h_t - theta2 * electronic_0


def inverse_guide_components_pt_thouless_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    components = components_pt_thouless_rw_rh(walker, ham_data, meas_ctx, trial_data)
    reweight = jnp.exp(-components[0])
    return jnp.concatenate([reweight[None], reweight * components])


def make_ptccsd_mode_meas_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError("PT-CCSD mode measurements require a closed-shell restricted walker.")
    cfg = PtccsdModeMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    return MeasOps(
        overlap=overlap_pt_r,
        build_meas_ctx=lambda ham_data, trial_data: build_ptccsd_mode_meas_ctx(
            ham_data,
            trial_data,
            n_mode_chunks=n_mode_chunks,
            cfg=cfg,
        ),
        kernels={k_force_bias: force_bias_pt_rw_rh, k_energy: energy_pt_rw_rh},
        observables={o_pt_components: inverse_guide_components_pt_rw_rh},
    )


def make_ptccsd_mode_estimator_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
    mixed_precision: bool = True,
    testing: bool = False,
) -> EstimatorOps:
    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError("PT-CCSD mode estimators require a closed-shell restricted walker.")
    cfg = PtccsdModeMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    return EstimatorOps(
        reference_overlap=hf_overlap_r,
        components=components_pt_rw_rh,
        combine_energy=combine_first_order_energy,
        component_names=("theta", "electronic_0", "h_t"),
        build_estimator_ctx=lambda ham_data, trial_data: build_ptccsd_mode_meas_ctx(
            ham_data,
            trial_data,
            n_mode_chunks=n_mode_chunks,
            cfg=cfg,
        ),
    )


def make_ptccsd_thouless_mode_meas_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
    exponentiate_t2: bool = True,
    memory_mode: Literal["low", "high"] = "high",
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "PT-CCSD Thouless mode measurements require a closed-shell restricted walker."
        )
    overlap = overlap_ptccsd_thouless_r if exponentiate_t2 else det_overlap_thouless_r
    observables = (
        {o_pt_components: inverse_guide_components_pt_thouless_rw_rh}
        if exponentiate_t2
        else {}
    )
    cfg = PtccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    return MeasOps(
        overlap=overlap,
        build_meas_ctx=lambda ham_data, trial_data: build_ptccsd_thouless_mode_meas_ctx(
            ham_data,
            trial_data,
            n_mode_chunks=n_mode_chunks,
            memory_mode=memory_mode,
            cfg=cfg,
        ),
        kernels={
            k_force_bias: force_bias_pt_thouless_rw_rh,
            k_energy: energy_pt_thouless_rw_rh,
        },
        observables=observables,
    )


def make_ptccsd_thouless_mode_estimator_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
    memory_mode: Literal["low", "high"] = "high",
    mixed_precision: bool = True,
    testing: bool = False,
    component_sampling: PtccsdModePairSamplingCfg | None = None,
) -> EstimatorOps:
    """Build the common mode-native PT2/Thouless projected estimator."""

    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "PT-CCSD Thouless mode estimators require a closed-shell restricted walker."
        )
    cfg = PtccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    return EstimatorOps(
        reference_overlap=det_overlap_thouless_r,
        components=components_pt_thouless_rw_rh,
        combine_energy=combine_first_order_energy,
        component_names=("theta", "electronic_0", "h_t"),
        build_estimator_ctx=lambda ham_data, trial_data: build_ptccsd_thouless_mode_meas_ctx(
            ham_data,
            trial_data,
            n_mode_chunks=n_mode_chunks,
            memory_mode=memory_mode,
            cfg=cfg,
            component_sampling=component_sampling,
        ),
        block_components=(
            pair_sampled_ptccsd_block_components
            if component_sampling is not None
            else None
        ),
    )
