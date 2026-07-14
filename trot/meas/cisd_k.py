from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.cisd_k import CisdKTrial
from ..trial.cisd_k import overlap_r as cisd_k_overlap_r
from .cisd import (
    CisdMeasCfg,
    _energy_gl_batched_realimag,
    _force_bias_chol_contract_high_realimag,
)

_CISD_K_MEAS_CFG_ATTR = "_cisd_k_meas_cfg"


def _greens_restricted(walker: jax.Array, nocc: int) -> jax.Array:
    wocc = walker[:nocc, :]
    return jnp.linalg.solve(wocc.T, walker.T)


def _active_green_blocks(
    green: jax.Array,
    trial_data: CisdKTrial,
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
class CisdKMeasCtx:
    rot_chol: jax.Array
    lci1: jax.Array
    cfg: CisdMeasCfg

    def tree_flatten(self):
        return (self.rot_chol, self.lci1), (self.cfg,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cfg,) = aux
        rot_chol, lci1 = children
        return cls(rot_chol=rot_chol, lci1=lci1, cfg=cfg)


def get_cisd_k_meas_cfg(meas_ops: MeasOps) -> CisdMeasCfg | None:
    cfg = getattr(meas_ops, _CISD_K_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, CisdMeasCfg) else None


def build_meas_ctx(
    ham_data: HamChol,
    trial_data: CisdKTrial,
    cfg: CisdMeasCfg = CisdMeasCfg(memory_mode="high"),
) -> CisdKMeasCtx:
    """Build full-Cholesky intermediates for the K-native estimators."""
    if cfg.memory_mode != "high":
        raise ValueError("K-native CISD measurements currently require memory_mode='high'.")
    if ham_data.basis != "restricted":
        raise ValueError("CISD K MeasOps requires HamChol.basis == 'restricted'.")

    chol = ham_data.chol
    rot_chol = chol[:, : trial_data.nocc_full, :]
    lci1 = jnp.einsum(
        "git,pt->gip",
        chol[:, :, trial_data.vir_act_slice],
        trial_data.ci1,
        optimize="optimal",
    )
    return CisdKMeasCtx(rot_chol=rot_chol, lci1=lci1, cfg=cfg)


def _k_apply_realimag(
    kernel: jax.Array,
    matrix: jax.Array,
    *,
    real_dtype: jnp.dtype,
    complex_dtype: jnp.dtype,
) -> jax.Array:
    """Apply a real pair-space K matrix at an explicit precision."""
    output_shape = matrix.shape
    kernel_t = kernel.astype(real_dtype)
    matrix_r = jnp.real(matrix).reshape(-1).astype(real_dtype)
    matrix_i = jnp.imag(matrix).reshape(-1).astype(real_dtype)
    applied_r = jnp.einsum("pq,q->p", kernel_t, matrix_r, optimize="optimal")
    applied_i = jnp.einsum("pq,q->p", kernel_t, matrix_i, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=complex_dtype)
    applied = applied_r.astype(complex_dtype) + imag_unit * applied_i.astype(complex_dtype)
    return applied.reshape(output_shape)


def _k_quadratic_batched_realimag(
    kernel: jax.Array,
    matrices: jax.Array,
    *,
    real_dtype: jnp.dtype,
    complex_dtype: jnp.dtype,
) -> jax.Array:
    """Sum ``x_g.T @ K @ x_g`` over a batch of pair-space matrices."""
    kernel_t = kernel.astype(real_dtype)
    matrices_r = jnp.real(matrices).reshape(matrices.shape[0], -1).astype(real_dtype)
    matrices_i = jnp.imag(matrices).reshape(matrices.shape[0], -1).astype(real_dtype)
    applied_r = jnp.einsum("pq,gq->gp", kernel_t, matrices_r, optimize="optimal")
    applied_i = jnp.einsum("pq,gq->gp", kernel_t, matrices_i, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=complex_dtype)
    applied = applied_r.astype(complex_dtype) + imag_unit * applied_i.astype(complex_dtype)
    matrices_t = matrices.astype(complex_dtype).reshape(matrices.shape[0], -1)
    return jnp.sum(matrices_t * applied, dtype=complex_dtype)


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdKMeasCtx,
    trial_data: CisdKTrial,
) -> jax.Array:
    """Exact full-K restricted force bias."""
    green = _greens_restricted(walker, trial_data.nocc_full)
    green_act, green_occ, greenp = _active_green_blocks(green, trial_data)

    lg = jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")
    ci1g = jnp.einsum("pt,pt->", trial_data.ci1, green_occ, optimize="optimal")
    ci1gp = jnp.einsum("pt,it->pi", trial_data.ci1, greenp, optimize="optimal")
    gci1gp = jnp.einsum("pj,pi->ij", green_act, ci1gp, optimize="optimal")

    kg = _k_apply_realimag(
        trial_data.k,
        green_occ,
        real_dtype=meas_ctx.cfg.mixed_real_dtype,
        complex_dtype=meas_ctx.cfg.mixed_complex_dtype,
    )
    gkg = jnp.einsum("qu,qu->", kg, green_occ, optimize="optimal")
    overlap = 1.0 + 2.0 * ci1g + gkg

    cisd_green = -2.0 * (greenp @ kg.T) @ green_act
    correction = _force_bias_chol_contract_high_realimag(
        ham_data.chol,
        cisd_green - 2.0 * gci1gp,
        meas_ctx.cfg,
    )
    return (2.0 * lg + 4.0 * ci1g * lg + 2.0 * lg * gkg + correction) / overlap


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdKMeasCtx,
    trial_data: CisdKTrial,
) -> jax.Array:
    """Exact full-K deterministic local energy for a restricted walker."""
    ci1 = trial_data.ci1
    green = _greens_restricted(walker, trial_data.nocc_full)
    green_act, green_occ, greenp = _active_green_blocks(green, trial_data)

    h1 = ham_data.h1
    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol

    hg = jnp.einsum("pj,pj->", h1[: trial_data.nocc_full, :], green, optimize="optimal")
    e1_0 = 2.0 * hg

    ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
    ci1_green = (greenp @ ci1.T) @ green_act
    e1_1 = 4.0 * ci1g * hg - 2.0 * jnp.einsum("ij,ij->", h1, ci1_green, optimize="optimal")

    kg = _k_apply_realimag(
        trial_data.k,
        green_occ,
        real_dtype=meas_ctx.cfg.mixed_real_dtype,
        complex_dtype=meas_ctx.cfg.mixed_complex_dtype,
    )
    ci2_green = (greenp @ kg.T) @ green_act
    gkg = jnp.einsum("qu,qu->", kg, green_occ, optimize="optimal")
    e1_2 = 2.0 * hg * gkg - 2.0 * jnp.einsum("ij,ij->", h1, ci2_green, optimize="optimal")
    e1 = e1_0 + e1_1 + e1_2

    lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
    e2_0_1 = 2.0 * (lg @ lg)
    lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
    e2_0_2 = -jnp.sum(lg1 * jnp.swapaxes(lg1, -1, -2))
    e2_0 = e2_0_1 + e2_0_2

    e2_1_1 = 2.0 * e2_0 * ci1g
    lci1g = _force_bias_chol_contract_high_realimag(chol, ci1_green, meas_ctx.cfg)
    e2_1_2 = -2.0 * (lci1g @ lg)
    ci1g1 = ci1 @ green[:, trial_data.vir_act_slice].T
    e2_1_3_1 = jnp.einsum(
        "gpq,gqa,ap->",
        lg1,
        lg1[:, :, trial_data.occ_act_slice],
        ci1g1,
        optimize="optimal",
    )
    lci1g_mat = jnp.einsum("gia,qi->gaq", meas_ctx.lci1, green, optimize="optimal")
    e2_1_3_2 = -jnp.einsum(
        "gaq,gqa->",
        lci1g_mat,
        lg1[:, :, trial_data.occ_act_slice],
        optimize="optimal",
    )
    e2_1 = e2_1_1 + 2.0 * (e2_1_2 + e2_1_3_1 + e2_1_3_2)

    e2_2_1 = e2_0 * gkg
    lci2g = _force_bias_chol_contract_high_realimag(chol, ci2_green, meas_ctx.cfg)
    e2_2_2_1 = -(lci2g @ lg)
    lci2_green = jnp.einsum("gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal")
    gl = _energy_gl_batched_realimag(green, chol, meas_ctx.cfg)
    e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")

    glgp = jnp.einsum("gpi,it->gpt", gl, greenp, optimize="optimal").astype(
        meas_ctx.cfg.mixed_complex_dtype_testing
    )
    glgp = glgp[:, trial_data.occ_act_slice, :]
    e2_2_3 = _k_quadratic_batched_realimag(
        trial_data.k,
        glgp,
        real_dtype=meas_ctx.cfg.mixed_real_dtype_testing,
        complex_dtype=meas_ctx.cfg.mixed_complex_dtype_testing,
    )

    e2_2_2 = 4.0 * (e2_2_2_1 + e2_2_2_2)
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3
    e2 = e2_0 + e2_1 + e2_2

    overlap = 1.0 + 2.0 * ci1g + gkg
    return (e1 + e2) / overlap + ham_data.h0


def make_cisd_k_meas_ops(
    sys: System,
    memory_mode: str = "high",
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    """Build opt-in exact K-native restricted CISD measurements.

    The precision configuration mirrors ``make_cisd_meas_ops``. K remains
    stored in float64; mixed force-bias and energy contractions cast it to the
    configured float32 path without retaining a second K array in the trial.
    """
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"CISD K MeasOps currently supports only restricted walkers, got: {sys.walker_kind}"
        )
    if memory_mode != "high":
        raise ValueError("K-native CISD measurements currently require memory_mode='high'.")

    cfg = CisdMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    meas_ops = MeasOps(
        overlap=cisd_k_overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels={k_force_bias: force_bias_kernel_rw_rh, k_energy: energy_kernel_rw_rh},
        observables={},
    )
    object.__setattr__(meas_ops, _CISD_K_MEAS_CFG_ATTR, cfg)
    return meas_ops
