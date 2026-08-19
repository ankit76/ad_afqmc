from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import EstimatorOps, MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.ptuccsd_modes import (
    PtuccsdThoulessModeTrial,
    overlap_r,
    overlap_u,
    reference_overlap_r,
)
from ..trial.ptuccsd_thouless import PtuccsdThoulessTrial, greenp_from_green
from .ptuccsd_thouless import (
    PtuccsdThoulessMeasCfg,
    PtuccsdThoulessMeasCtx,
    _chol_contract,
    _energy_gl_batched,
    _energy_gl_scalar,
    build_ptuccsd_thouless_meas_ctx,
    o_pt_components,
)
from .pt2ccsd import combine_first_order_energy

PtuccsdModeMeasCfg = PtuccsdThoulessMeasCfg


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtuccsdModeMeasCtx:
    """Spin-rotated Hamiltonian intermediates and mode batching policy."""

    base: PtuccsdThoulessMeasCtx
    n_mode_chunks: int

    @property
    def h1_b(self) -> jax.Array:
        return self.base.h1_b

    @property
    def chol_b(self) -> jax.Array:
        return self.base.chol_b

    @property
    def rot_chol_a(self) -> jax.Array:
        return self.base.rot_chol_a

    @property
    def rot_chol_b(self) -> jax.Array:
        return self.base.rot_chol_b

    @property
    def cfg(self) -> PtuccsdModeMeasCfg:
        return self.base.cfg

    def tree_flatten(self):
        return (self.base,), (self.n_mode_chunks,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (n_mode_chunks,) = aux
        (base,) = children
        return cls(base=base, n_mode_chunks=n_mode_chunks)

_PTUCCSD_MODE_MEAS_CFG_ATTR = "_ptuccsd_mode_meas_cfg"


def get_ptuccsd_mode_meas_cfg(meas_ops: MeasOps) -> PtuccsdModeMeasCfg | None:
    cfg = getattr(meas_ops, _PTUCCSD_MODE_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, PtuccsdModeMeasCfg) else None


def build_ptuccsd_mode_meas_ctx(
    ham_data: HamChol,
    trial_data: PtuccsdThoulessModeTrial,
    cfg: PtuccsdModeMeasCfg = PtuccsdModeMeasCfg(),
    *,
    n_mode_chunks: int = 1,
) -> PtuccsdModeMeasCtx:
    """Build the spin-rotated Hamiltonian intermediates used by the mode guide."""

    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    # Context construction accesses only the two Thouless references and the
    # beta orbital rotation, which the dense and mode trials share exactly.
    base = build_ptuccsd_thouless_meas_ctx(
        ham_data, cast(PtuccsdThoulessTrial, trial_data), cfg
    )
    chunks = min(int(n_mode_chunks), trial_data.mode_rank) if trial_data.mode_rank else 1
    return PtuccsdModeMeasCtx(
        base=base,
        n_mode_chunks=chunks,
    )


def _mode_apply_realimag(
    trial_data: PtuccsdThoulessModeTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
    cfg: PtuccsdModeMeasCfg,
) -> tuple[jax.Array, jax.Array]:
    """Apply the retained combined modes using the configured precision."""

    expected_a = (trial_data.nocc[0], trial_data.nvir[0])
    expected_b = (trial_data.nocc[1], trial_data.nvir[1])
    if matrix_a.shape != expected_a or matrix_b.shape != expected_b:
        raise ValueError(
            f"matrix pair must have shapes {expected_a} and {expected_b}, got "
            f"{matrix_a.shape} and {matrix_b.shape}."
        )

    da, _ = trial_data.pair_dim
    modes = trial_data.modes.astype(cfg.mixed_real_dtype)
    modes_a = modes[:, :da]
    modes_b = modes[:, da:]
    values = trial_data.eigenvalues.astype(cfg.mixed_real_dtype)

    vector_a_r = jnp.real(matrix_a).reshape(-1).astype(cfg.mixed_real_dtype)
    vector_b_r = jnp.real(matrix_b).reshape(-1).astype(cfg.mixed_real_dtype)
    projection_a_r = jnp.einsum("rp,p->r", modes_a, vector_a_r, optimize="optimal")
    projection_b_r = jnp.einsum("rp,p->r", modes_b, vector_b_r, optimize="optimal")
    projection_r = (
        projection_a_r.astype(jnp.float64) + projection_b_r.astype(jnp.float64)
    ).astype(cfg.mixed_real_dtype)

    vector_a_i = jnp.imag(matrix_a).reshape(-1).astype(cfg.mixed_real_dtype)
    vector_b_i = jnp.imag(matrix_b).reshape(-1).astype(cfg.mixed_real_dtype)
    projection_a_i = jnp.einsum("rp,p->r", modes_a, vector_a_i, optimize="optimal")
    projection_b_i = jnp.einsum("rp,p->r", modes_b, vector_b_i, optimize="optimal")
    projection_i = (
        projection_a_i.astype(jnp.float64) + projection_b_i.astype(jnp.float64)
    ).astype(cfg.mixed_real_dtype)

    applied_r = jnp.einsum("r,rp->p", values * projection_r, modes, optimize="optimal")
    applied_i = jnp.einsum("r,rp->p", values * projection_i, modes, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    applied = applied_r.astype(cfg.mixed_complex_dtype)
    applied += imag_unit * applied_i.astype(cfg.mixed_complex_dtype)

    return applied[:da].reshape(expected_a), applied[da:].reshape(expected_b)


def _mode_quadratic_batched_realimag(
    trial_data: PtuccsdThoulessModeTrial,
    matrices_a: jax.Array,
    matrices_b: jax.Array,
    cfg: PtuccsdModeMeasCfg,
    n_mode_chunks: int = 1,
) -> jax.Array:
    """Evaluate ``0.5 * z.T @ K_modes @ z`` for a batch of matrix pairs."""

    if matrices_a.shape[:-2] != matrices_b.shape[:-2]:
        raise ValueError("alpha and beta matrices must have identical leading shapes.")
    expected_a = (trial_data.nocc[0], trial_data.nvir[0])
    expected_b = (trial_data.nocc[1], trial_data.nvir[1])
    if matrices_a.shape[-2:] != expected_a or matrices_b.shape[-2:] != expected_b:
        raise ValueError(
            f"matrix pair must end in shapes {expected_a} and {expected_b}, got "
            f"{matrices_a.shape} and {matrices_b.shape}."
        )

    leading_shape = matrices_a.shape[:-2]
    vectors_a = matrices_a.reshape((-1, trial_data.pair_dim[0]))
    vectors_b = matrices_b.reshape((-1, trial_data.pair_dim[1]))
    rank = trial_data.mode_rank
    result_dtype = (
        jnp.complex128
        if jnp.issubdtype(vectors_a.dtype, jnp.complexfloating)
        or jnp.issubdtype(vectors_b.dtype, jnp.complexfloating)
        else jnp.float64
    )
    if rank == 0:
        return jnp.zeros(leading_shape, dtype=result_dtype)

    da, _ = trial_data.pair_dim
    vectors_a_r = jnp.real(vectors_a).astype(cfg.mixed_real_dtype_testing)
    vectors_b_r = jnp.real(vectors_b).astype(cfg.mixed_real_dtype_testing)
    vectors_a_i = jnp.imag(vectors_a).astype(cfg.mixed_real_dtype_testing)
    vectors_b_i = jnp.imag(vectors_b).astype(cfg.mixed_real_dtype_testing)

    def evaluate_chunk(values_i: jax.Array, modes_i: jax.Array) -> jax.Array:
        modes_i = modes_i.astype(cfg.mixed_real_dtype_testing)
        modes_a_i = modes_i[:, :da]
        modes_b_i = modes_i[:, da:]
        projection_a_r = jnp.einsum(
            "sp,rp->sr", vectors_a_r, modes_a_i, optimize="optimal"
        )
        projection_b_r = jnp.einsum(
            "sp,rp->sr", vectors_b_r, modes_b_i, optimize="optimal"
        )
        values_t = values_i.astype(jnp.float64)[None, :]
        if result_dtype == jnp.complex128:
            projection_a_i = jnp.einsum(
                "sp,rp->sr", vectors_a_i, modes_a_i, optimize="optimal"
            )
            projection_b_i = jnp.einsum(
                "sp,rp->sr", vectors_b_i, modes_b_i, optimize="optimal"
            )
            projection_a = projection_a_r.astype(jnp.complex128)
            projection_a += 1.0j * projection_a_i.astype(jnp.complex128)
            projection_b = projection_b_r.astype(jnp.complex128)
            projection_b += 1.0j * projection_b_i.astype(jnp.complex128)
            contribution = 0.5 * values_t * (projection_a + projection_b) ** 2
            return jnp.sum(contribution, axis=1, dtype=jnp.complex128)

        projection = projection_a_r.astype(jnp.float64)
        projection += projection_b_r.astype(jnp.float64)
        contribution = 0.5 * values_t * projection**2
        return jnp.sum(contribution, axis=1, dtype=jnp.float64)

    chunks = min(int(n_mode_chunks), rank)
    if chunks == 1:
        result = evaluate_chunk(trial_data.eigenvalues, trial_data.modes)
        return result.reshape(leading_shape)

    base_chunk_size = rank // chunks
    n_larger_chunks = rank % chunks
    chunk_size = base_chunk_size + int(n_larger_chunks > 0)
    chunk_offsets = jnp.arange(chunk_size, dtype=jnp.int32)
    zero = jnp.zeros((vectors_a.shape[0],), dtype=result_dtype)

    def scan_body(total, chunk_index):
        is_larger = chunk_index < n_larger_chunks
        chunk_length = base_chunk_size + is_larger.astype(jnp.int32)
        start = chunk_index * base_chunk_size + jnp.minimum(chunk_index, n_larger_chunks)
        indices = start + chunk_offsets
        valid = chunk_offsets < chunk_length
        indices = jnp.minimum(indices, rank - 1)
        values_i = jnp.where(valid, trial_data.eigenvalues[indices], 0.0)
        contribution = evaluate_chunk(values_i, trial_data.modes[indices])
        return total + contribution, None

    result, _ = lax.scan(scan_body, zero, jnp.arange(chunks, dtype=jnp.int32))
    return result.reshape(leading_shape)


def _green_blocks(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    _, _, green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b = (
        _half_green_blocks(walker, trial_data)
    )
    return green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b


def _half_green_blocks(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessModeTrial,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Return spin-resolved half- and full-Green-function blocks."""

    walker_a, walker_b = walker
    walker_b_beta = trial_data.mo_coeff_b.conj().T @ walker_b
    overlap_a = trial_data.mo_t_a.conj().T @ walker_a
    overlap_b = trial_data.mo_t_b.conj().T @ walker_b_beta
    half_green_a = jnp.linalg.solve(overlap_a.T, walker_a.T)
    half_green_b = jnp.linalg.solve(overlap_b.T, walker_b_beta.T)
    green_a = trial_data.mo_t_a.conj() @ half_green_a
    green_b = trial_data.mo_t_b.conj() @ half_green_b
    noa, nob = trial_data.nocc
    green_occ_a = green_a[:noa, noa:]
    green_occ_b = green_b[:nob, nob:]
    greenp_a = greenp_from_green(green_a, noa)
    greenp_b = greenp_from_green(green_b, nob)
    return (
        half_green_a,
        half_green_b,
        green_a,
        green_b,
        green_occ_a,
        green_occ_b,
        greenp_a,
        greenp_b,
    )


def force_bias_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    """Mode-native PT-UCCSD exponential-guide force bias."""

    green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b = _green_blocks(
        walker,
        trial_data,
    )
    noa, nob = trial_data.nocc

    # As in dense PT-UCCSD and UCISD, keep the determinant-reference force
    # bias in full precision and evaluate only the correlation correction in
    # the configured mixed precision.
    f0_a = jnp.einsum("gij,ij->g", ham_data.chol, green_a, optimize="optimal")
    f0_b = jnp.einsum("gij,ij->g", meas_ctx.chol_b, green_b, optimize="optimal")
    applied_a, applied_b = _mode_apply_realimag(
        trial_data,
        green_occ_a,
        green_occ_b,
        meas_ctx.cfg,
    )
    t2_green_a = (greenp_a @ applied_a.T) @ green_a[:noa, :]
    t2_green_b = (greenp_b @ applied_b.T) @ green_b[:nob, :]
    correction = -_chol_contract(ham_data.chol, t2_green_a, meas_ctx.cfg)
    correction -= _chol_contract(meas_ctx.chol_b, t2_green_b, meas_ctx.cfg)
    return f0_a + f0_b + correction


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    """Mode-native force bias for a restricted open-shell walker."""

    noa, nob = trial_data.nocc
    return force_bias_kernel_uw_rh(
        (walker[:, :noa], walker[:, :nob]),
        ham_data,
        meas_ctx,
        trial_data,
    )


def _energy_components_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return mode-native ``(theta2, electronic_0, h_t)`` components."""

    (
        half_green_a,
        half_green_b,
        green_a,
        green_b,
        green_occ_a,
        green_occ_b,
        greenp_a,
        greenp_b,
    ) = _half_green_blocks(walker, trial_data)
    noa, nob = trial_data.nocc
    cfg = meas_ctx.cfg
    h1_a = 0.5 * (ham_data.h1 + ham_data.h1.T.conj())
    h1_b = meas_ctx.h1_b
    chol_a = ham_data.chol
    chol_b = meas_ctx.chol_b
    rot_chol_a = meas_ctx.rot_chol_a
    rot_chol_b = meas_ctx.rot_chol_b

    e1_0 = jnp.einsum("ij,ij->", h1_a, green_a, optimize="optimal")
    e1_0 += jnp.einsum("ij,ij->", h1_b, green_b, optimize="optimal")

    applied_a, applied_b = _mode_apply_realimag(
        trial_data,
        green_occ_a,
        green_occ_b,
        cfg,
    )
    theta2 = 0.5 * jnp.einsum(
        "pt,pt->", green_occ_a, applied_a, optimize="optimal"
    )
    theta2 += 0.5 * jnp.einsum(
        "pt,pt->", green_occ_b, applied_b, optimize="optimal"
    )
    combo_a = (greenp_a @ applied_a.T) @ green_a[:noa, :]
    combo_b = (greenp_b @ applied_b.T) @ green_b[:nob, :]
    e1_2 = e1_0 * theta2
    e1_2 -= jnp.einsum("ij,ij->", h1_a, combo_a, optimize="optimal")
    e1_2 -= jnp.einsum("ij,ij->", h1_b, combo_b, optimize="optimal")

    lg_a = jnp.einsum("giq,iq->g", rot_chol_a, half_green_a, optimize="optimal")
    lg_b = jnp.einsum("giq,iq->g", rot_chol_b, half_green_b, optimize="optimal")
    lg = lg_a + lg_b
    lg1_a = jnp.einsum(
        "gip,jp->gij", rot_chol_a, half_green_a, optimize="optimal"
    )
    lg1_b = jnp.einsum(
        "gip,jp->gij", rot_chol_b, half_green_b, optimize="optimal"
    )
    e2_0 = 0.5 * (lg @ lg)
    e2_0 -= 0.5 * (
        jnp.sum(lg1_a * jnp.swapaxes(lg1_a, -1, -2))
        + jnp.sum(lg1_b * jnp.swapaxes(lg1_b, -1, -2))
    )
    electronic_0 = e1_0 + e2_0

    combo2_a = 2.0 * combo_a
    combo2_b = 2.0 * combo_b
    lt2g_a = _chol_contract(chol_a, combo2_a, cfg)
    lt2g_b = _chol_contract(chol_b, combo2_b, cfg)
    e2_2_2_1 = -0.5 * ((lt2g_a + lt2g_b) @ lg)

    # Cast before constructing walker--Cholesky batches so the selected mixed
    # precision controls their memory footprint, as in dense PT-UCCSD.
    reference_occ_a = trial_data.mo_t_a.conj()[:noa, :].astype(cfg.mixed_complex_dtype)
    reference_occ_b = trial_data.mo_t_b.conj()[:nob, :].astype(cfg.mixed_complex_dtype)
    greenp_a_mixed = greenp_a.astype(cfg.mixed_complex_dtype)
    greenp_b_mixed = greenp_b.astype(cfg.mixed_complex_dtype)
    combo2_a_mixed = combo2_a.astype(cfg.mixed_complex_dtype)
    combo2_b_mixed = combo2_b.astype(cfg.mixed_complex_dtype)

    if cfg.memory_mode == "low":
        zero_e222 = jnp.zeros_like(e2_2_2_1)
        zero_e23 = jnp.asarray(0.0, dtype=jnp.complex128)

        def scan_doubles(carry, xs):
            e222_acc, e23_acc = carry
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = xs
            gl_half_a_i = _energy_gl_scalar(half_green_a, chol_a_i, cfg)
            gl_half_b_i = _energy_gl_scalar(half_green_b, chol_b_i, cfg)
            lcombo_a_i = jnp.einsum(
                "pi,ji->pj",
                rot_chol_a_i.astype(cfg.mixed_complex_dtype),
                combo2_a_mixed,
                optimize="optimal",
            )
            lcombo_b_i = jnp.einsum(
                "pi,ji->pj",
                rot_chol_b_i.astype(cfg.mixed_complex_dtype),
                combo2_b_mixed,
                optimize="optimal",
            )
            e222_acc += 0.5 * (
                jnp.einsum("pi,pi->", gl_half_a_i, lcombo_a_i, optimize="optimal")
                + jnp.einsum("pi,pi->", gl_half_b_i, lcombo_b_i, optimize="optimal")
            )

            gl_occ_a_i = reference_occ_a @ gl_half_a_i
            gl_occ_b_i = reference_occ_b @ gl_half_b_i
            glgp_a_i = jnp.einsum(
                "pi,it->pt", gl_occ_a_i, greenp_a_mixed, optimize="optimal"
            ).astype(cfg.mixed_complex_dtype_testing)
            glgp_b_i = jnp.einsum(
                "pi,it->pt", gl_occ_b_i, greenp_b_mixed, optimize="optimal"
            ).astype(cfg.mixed_complex_dtype_testing)
            e23_i = _mode_quadratic_batched_realimag(
                trial_data,
                glgp_a_i[None, ...],
                glgp_b_i[None, ...],
                cfg,
                meas_ctx.n_mode_chunks,
            )[0]
            return (e222_acc, e23_acc + e23_i), None

        (e2_2_2_2, e2_2_3), _ = lax.scan(
            scan_doubles,
            (zero_e222, zero_e23),
            (chol_a, rot_chol_a, chol_b, rot_chol_b),
        )
    else:
        gl_half_a = _energy_gl_batched(half_green_a, chol_a, cfg)
        gl_half_b = _energy_gl_batched(half_green_b, chol_b, cfg)
        lcombo_a = jnp.einsum(
            "gpi,ji->gpj",
            rot_chol_a.astype(cfg.mixed_complex_dtype),
            combo2_a_mixed,
            optimize="optimal",
        )
        lcombo_b = jnp.einsum(
            "gpi,ji->gpj",
            rot_chol_b.astype(cfg.mixed_complex_dtype),
            combo2_b_mixed,
            optimize="optimal",
        )
        e2_2_2_2 = 0.5 * (
            jnp.einsum("gpi,gpi->", gl_half_a, lcombo_a, optimize="optimal")
            + jnp.einsum("gpi,gpi->", gl_half_b, lcombo_b, optimize="optimal")
        )

        gl_occ_a = jnp.einsum(
            "ij,gjq->giq", reference_occ_a, gl_half_a, optimize="optimal"
        )
        gl_occ_b = jnp.einsum(
            "ij,gjq->giq", reference_occ_b, gl_half_b, optimize="optimal"
        )
        glgp_a = jnp.einsum(
            "gpi,it->gpt", gl_occ_a, greenp_a_mixed, optimize="optimal"
        ).astype(cfg.mixed_complex_dtype_testing)
        glgp_b = jnp.einsum(
            "gpi,it->gpt", gl_occ_b, greenp_b_mixed, optimize="optimal"
        ).astype(cfg.mixed_complex_dtype_testing)
        e2_2_3 = jnp.sum(
            _mode_quadratic_batched_realimag(
                trial_data,
                glgp_a,
                glgp_b,
                cfg,
                meas_ctx.n_mode_chunks,
            ),
            dtype=jnp.complex128,
        )

    e2_2 = e2_0 * theta2 + e2_2_2_1 + e2_2_2_2 + e2_2_3
    h_t = e1_2 + e2_2
    return theta2, electronic_0, h_t


def components_ptuccsd_mode_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    """Return guide-independent ``[theta, electronic_0, h_t]`` components."""

    return jnp.stack(_energy_components_uw_rh(walker, ham_data, meas_ctx, trial_data))


def components_ptuccsd_mode_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    return components_ptuccsd_mode_uw_rh(
        (walker[:, :noa], walker[:, :nob]),
        ham_data,
        meas_ctx,
        trial_data,
    )


def energy_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    components = components_ptuccsd_mode_uw_rh(walker, ham_data, meas_ctx, trial_data)
    return combine_first_order_energy(ham_data.h0, components)


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    components = components_ptuccsd_mode_rw_rh(walker, ham_data, meas_ctx, trial_data)
    return combine_first_order_energy(ham_data.h0, components)


def make_ptuccsd_mode_force_bias_ops(
    sys: System,
    *,
    memory_mode: Literal["low", "high"] = "high",
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    """Build propagation-only mode operations containing the force bias.

    This deliberately omits ``k_energy`` for callers that need only the
    propagation kernel.  Use :func:`make_ptuccsd_mode_meas_ops` for a complete
    guide measurement bundle.
    """

    cfg = PtuccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    walker_kind = sys.walker_kind.lower()
    if walker_kind == "restricted":
        if sys.nup < sys.ndn:
            raise ValueError("Restricted PT-UCCSD mode force bias requires nup >= ndn.")
        overlap_fn = overlap_r
        force_bias_fn = force_bias_kernel_rw_rh
    elif walker_kind == "unrestricted":
        overlap_fn = overlap_u
        force_bias_fn = force_bias_kernel_uw_rh
    else:
        raise ValueError(
            "PT-UCCSD mode force bias supports restricted/unrestricted walkers, "
            f"got: {sys.walker_kind}"
        )

    meas_ops = MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=lambda ham_data, trial_data: build_ptuccsd_mode_meas_ctx(
            ham_data,
            trial_data,
            cfg,
        ),
        kernels={k_force_bias: force_bias_fn},
    )
    object.__setattr__(meas_ops, _PTUCCSD_MODE_MEAS_CFG_ATTR, cfg)
    return meas_ops


def make_ptuccsd_mode_meas_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
    memory_mode: Literal["low", "high"] = "high",
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    """Build complete mode-native PT-UCCSD guide measurements."""

    cfg = PtuccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    walker_kind = sys.walker_kind.lower()
    if walker_kind == "restricted":
        if sys.nup < sys.ndn:
            raise ValueError("Restricted PT-UCCSD mode measurements require nup >= ndn.")
        overlap_fn = overlap_r
        force_bias_fn = force_bias_kernel_rw_rh
        energy_fn = energy_kernel_rw_rh
        components_fn = components_ptuccsd_mode_rw_rh
    elif walker_kind == "unrestricted":
        overlap_fn = overlap_u
        force_bias_fn = force_bias_kernel_uw_rh
        energy_fn = energy_kernel_uw_rh
        components_fn = components_ptuccsd_mode_uw_rh
    else:
        raise ValueError(
            "PT-UCCSD mode measurements support restricted/unrestricted walkers, "
            f"got: {sys.walker_kind}"
        )

    meas_ops = MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=lambda ham_data, trial_data: build_ptuccsd_mode_meas_ctx(
            ham_data,
            trial_data,
            cfg,
            n_mode_chunks=n_mode_chunks,
        ),
        kernels={k_force_bias: force_bias_fn, k_energy: energy_fn},
        observables={o_pt_components: components_fn},
    )
    object.__setattr__(meas_ops, _PTUCCSD_MODE_MEAS_CFG_ATTR, cfg)
    return meas_ops


def make_ptuccsd_mode_estimator_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
    memory_mode: Literal["low", "high"] = "high",
    mixed_precision: bool = True,
    testing: bool = False,
) -> EstimatorOps:
    """Build a mode-native PT2-UCCSD estimator for a separate guide."""

    if sys.walker_kind.lower() != "restricted" or sys.nup < sys.ndn:
        raise ValueError(
            "PT-UCCSD mode estimators require a restricted walker with nup >= ndn."
        )
    cfg = PtuccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    return EstimatorOps(
        reference_overlap=reference_overlap_r,
        components=components_ptuccsd_mode_rw_rh,
        combine_energy=combine_first_order_energy,
        component_names=("theta", "electronic_0", "h_t"),
        build_estimator_ctx=lambda ham_data, trial_data: build_ptuccsd_mode_meas_ctx(
            ham_data,
            trial_data,
            cfg,
            n_mode_chunks=n_mode_chunks,
        ),
    )


__all__ = [
    "PtuccsdModeMeasCfg",
    "PtuccsdModeMeasCtx",
    "build_ptuccsd_mode_meas_ctx",
    "components_ptuccsd_mode_rw_rh",
    "components_ptuccsd_mode_uw_rh",
    "energy_kernel_rw_rh",
    "energy_kernel_uw_rh",
    "force_bias_kernel_rw_rh",
    "force_bias_kernel_uw_rh",
    "get_ptuccsd_mode_meas_cfg",
    "make_ptuccsd_mode_estimator_ops",
    "make_ptuccsd_mode_force_bias_ops",
    "make_ptuccsd_mode_meas_ops",
]
