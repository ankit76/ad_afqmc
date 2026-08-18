from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import EstimatorOps, MeasOps, k_energy, k_force_bias
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


@dataclass(frozen=True)
class PtccsdModeMeasCfg:
    """Precision and memory policy for mode-native PT-CCSD contractions."""

    memory_mode: Literal["low", "high"] = "high"
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64


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
    n_mode_chunks: int
    cfg: PtccsdModeMeasCfg

    @property
    def memory_mode(self) -> Literal["low", "high"]:
        return self.cfg.memory_mode

    def tree_flatten(self):
        return (self.rot_chol,), (self.n_mode_chunks, self.cfg)

    @classmethod
    def tree_unflatten(cls, aux, children):
        n_mode_chunks, cfg = aux
        (rot_chol,) = children
        return cls(
            rot_chol=rot_chol,
            n_mode_chunks=n_mode_chunks,
            cfg=cfg,
        )


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
    return PtccsdThoulessModeMeasCtx(
        rot_chol=jnp.einsum(
            "pi,gpq->giq",
            trial_data.mo_t.conj(),
            ham_data.chol,
            optimize="optimal",
        ),
        n_mode_chunks=min(int(n_mode_chunks), trial_data.mode_rank),
        cfg=cfg,
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

    half_green, green, green_rows, green_occ, greenp = _thouless_half_green_blocks(
        walker,
        trial_data,
    )
    h1 = ham_data.h1
    chol = ham_data.chol

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

    # The determinant-reference terms have the same half-rotated form as the
    # corresponding CISD contractions.
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

    lt2g = _chol_contract(chol, t2_green, meas_ctx.cfg)
    e2_2_2_1 = -(lt2g @ lg)

    reference_occ = trial_data.mo_t.conj()[: trial_data.nocc, :].astype(
        meas_ctx.cfg.mixed_complex_dtype
    )
    greenp_mixed = greenp.astype(meas_ctx.cfg.mixed_complex_dtype)
    t2_green_mixed = t2_green.astype(meas_ctx.cfg.mixed_complex_dtype)
    if meas_ctx.memory_mode == "low":
        zero = jnp.zeros((), dtype=jnp.result_type(half_green, chol, t2_green))

        def scan_doubles(carry, xs):
            e222_acc, e223_acc = carry
            chol_i, rot_chol_i = xs
            gl_half_i = _energy_gl_scalar(half_green, chol_i, meas_ctx.cfg)
            lt2_half_i = jnp.einsum(
                "ir,qr->iq",
                rot_chol_i.astype(meas_ctx.cfg.mixed_complex_dtype),
                t2_green_mixed,
                optimize="optimal",
            )
            e222_acc = e222_acc + 0.5 * jnp.einsum(
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
            return (e222_acc, e223_acc + e223_i), None

        (e2_2_2_2, e2_2_3), _ = jax.lax.scan(
            scan_doubles,
            (zero, zero),
            (chol, meas_ctx.rot_chol),
        )
    else:
        gl_half = _energy_gl_batched(half_green, chol, meas_ctx.cfg)
        lt2_half = jnp.einsum(
            "gir,qr->giq",
            meas_ctx.rot_chol.astype(meas_ctx.cfg.mixed_complex_dtype),
            t2_green_mixed,
            optimize="optimal",
        )
        e2_2_2_2 = 0.5 * jnp.einsum(
            "giq,giq->", gl_half, lt2_half, optimize="optimal"
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
        e2_2_3 = jnp.sum(
            mode_quadratic_matrices(
                trial_data,
                glgp,
                n_mode_chunks=meas_ctx.n_mode_chunks,
            )
        )
    e2_2 = e2_0 * theta2 + 4.0 * (e2_2_2_1 + e2_2_2_2) + e2_2_3

    electronic_0 = e1_0 + e2_0
    h_t = e1_2 + e2_2
    return jnp.stack([theta2, electronic_0, h_t])


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
        ),
    )
