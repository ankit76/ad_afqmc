from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.ucisd import UcisdTrial
from ..trial.ucisd_k import UcisdKTrial
from ..trial.ucisd_k import overlap_r as ucisd_k_overlap_r
from .ucisd import UcisdMeasCfg, UcisdMeasCtx
from .ucisd import build_meas_ctx as build_dense_meas_ctx
from .ucisd_modes import (
    _energy_gl_batched_realimag,
    _spin_sum_chol_contract,
)

_UCISD_K_MEAS_CFG_ATTR = "_ucisd_k_meas_cfg"


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdKMeasCtx:
    """Dense-independent UCISD measurement intermediates."""

    base: UcisdMeasCtx

    def tree_flatten(self):
        return (self.base,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        (base,) = children
        return cls(base=base)


class UcisdKEnergyCommon(NamedTuple):
    """Per-walker intermediates shared by all Cholesky contributions."""

    green_a: jax.Array
    green_b: jax.Array
    greenp_a: jax.Array
    greenp_b: jax.Array
    overlap: jax.Array
    m1_a: jax.Array
    m1_b: jax.Array
    m2_a: jax.Array
    m2_b: jax.Array
    z1_a: jax.Array
    z1_b: jax.Array
    base: jax.Array


def get_ucisd_k_meas_cfg(meas_ops: MeasOps) -> UcisdMeasCfg | None:
    cfg = getattr(meas_ops, _UCISD_K_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, UcisdMeasCfg) else None


def build_meas_ctx(
    ham_data: HamChol,
    trial_data: UcisdKTrial,
    *,
    cfg: UcisdMeasCfg = UcisdMeasCfg(memory_mode="high"),
) -> UcisdKMeasCtx:
    """Build full-Cholesky intermediates for combined-K UCISD estimators."""
    if cfg.memory_mode != "high":
        raise ValueError("K-native UCISD measurements currently require memory_mode='high'.")
    if ham_data.basis != "restricted":
        raise ValueError("UCISD K MeasOps requires HamChol.basis == 'restricted'.")

    # Dense UCISD context construction does not access a doubles tensor.
    base = build_dense_meas_ctx(ham_data, cast(UcisdTrial, trial_data), cfg)
    return UcisdKMeasCtx(base=base)


def _greens_restricted(
    walker: jax.Array,
    trial_data: UcisdKTrial,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build alpha/beta Green functions from one shared restricted walker."""
    noa, nob = trial_data.nocc
    nva, nvb = trial_data.nvir
    wa = walker[:, :noa]
    wb = trial_data.mo_coeff_b.T @ walker[:, :nob]
    green_a = jnp.linalg.solve(wa[:noa].T, wa.T)
    green_b = jnp.linalg.solve(wb[:nob].T, wb.T)
    green_occ_a = green_a[:, noa:]
    green_occ_b = green_b[:, nob:]
    greenp_a = jnp.vstack((green_occ_a, -jnp.eye(nva, dtype=green_a.dtype)))
    greenp_b = jnp.vstack((green_occ_b, -jnp.eye(nvb, dtype=green_b.dtype)))
    return green_a, green_b, greenp_a, greenp_b


def _combined_pair_batch(
    trial_data: UcisdKTrial,
    matrices_a: jax.Array,
    matrices_b: jax.Array,
) -> tuple[jax.Array, tuple[int, ...]]:
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
    a = matrices_a.reshape((-1, trial_data.pair_dim[0]))
    b = matrices_b.reshape((-1, trial_data.pair_dim[1]))
    return jnp.concatenate((a, b), axis=1), leading_shape


def _k_apply_realimag(
    trial_data: UcisdKTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
    cfg: UcisdMeasCfg,
) -> tuple[jax.Array, jax.Array]:
    """Apply one real combined K through explicit real mixed-precision GEMMs."""
    vectors, _ = _combined_pair_batch(
        trial_data,
        matrix_a[None, ...],
        matrix_b[None, ...],
    )
    kernel = trial_data.k.astype(cfg.mixed_real_dtype)
    vector_r = jnp.real(vectors[0]).astype(cfg.mixed_real_dtype)
    vector_i = jnp.imag(vectors[0]).astype(cfg.mixed_real_dtype)
    applied_r = jnp.einsum("pq,q->p", kernel, vector_r, optimize="optimal")
    applied_i = jnp.einsum("pq,q->p", kernel, vector_i, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    applied = applied_r.astype(cfg.mixed_complex_dtype)
    applied += imag_unit * applied_i.astype(cfg.mixed_complex_dtype)

    da, _ = trial_data.pair_dim
    shape_a = (trial_data.nocc[0], trial_data.nvir[0])
    shape_b = (trial_data.nocc[1], trial_data.nvir[1])
    return applied[:da].reshape(shape_a), applied[da:].reshape(shape_b)


def _k_quadratic_batched_realimag(
    trial_data: UcisdKTrial,
    matrices_a: jax.Array,
    matrices_b: jax.Array,
    cfg: UcisdMeasCfg,
) -> jax.Array:
    """Return ``0.5 * z.T @ K @ z`` independently for every leading index."""
    vectors, leading_shape = _combined_pair_batch(trial_data, matrices_a, matrices_b)
    kernel = trial_data.k.astype(cfg.mixed_real_dtype_testing)
    vectors_r = jnp.real(vectors).astype(cfg.mixed_real_dtype_testing)
    vectors_i = jnp.imag(vectors).astype(cfg.mixed_real_dtype_testing)
    applied_r = jnp.einsum("pq,sq->sp", kernel, vectors_r, optimize="optimal")
    applied_i = jnp.einsum("pq,sq->sp", kernel, vectors_i, optimize="optimal")

    if jnp.issubdtype(vectors.dtype, jnp.complexfloating):
        applied = applied_r.astype(jnp.complex128)
        applied += 1.0j * applied_i.astype(jnp.complex128)
        vectors_t = vectors.astype(jnp.complex128)
        result = 0.5 * jnp.sum(vectors_t * applied, axis=1, dtype=jnp.complex128)
    else:
        result = 0.5 * jnp.sum(
            vectors_r.astype(jnp.float64) * applied_r.astype(jnp.float64),
            axis=1,
            dtype=jnp.float64,
        )
    return result.reshape(leading_shape)


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdKMeasCtx,
    trial_data: UcisdKTrial,
) -> jax.Array:
    """Exact combined-K UCISD force bias for a restricted walker."""
    base = meas_ctx.base
    green_a, green_b, greenp_a, greenp_b = _greens_restricted(walker, trial_data)
    noa, nob = trial_data.nocc
    green_occ_a = green_a[:, noa:]
    green_occ_b = green_b[:, nob:]

    lg_a = jnp.einsum("gpj,pj->g", base.rot_chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gpj,pj->g", base.rot_chol_b, green_b, optimize="optimal")
    lg = lg_a + lg_b
    singles = jnp.einsum("pt,pt->", trial_data.c1a, green_occ_a, optimize="optimal")
    singles += jnp.einsum("pt,pt->", trial_data.c1b, green_occ_b, optimize="optimal")
    m1_a = (greenp_a @ trial_data.c1a.T) @ green_a
    m1_b = (greenp_b @ trial_data.c1b.T) @ green_b

    ya, yb = _k_apply_realimag(trial_data, green_occ_a, green_occ_b, base.cfg)
    doubles = 0.5 * jnp.einsum("pt,pt->", green_occ_a, ya, optimize="optimal")
    doubles += 0.5 * jnp.einsum("pt,pt->", green_occ_b, yb, optimize="optimal")
    m2_a = (greenp_a @ ya.T) @ green_a
    m2_b = (greenp_b @ yb.T) @ green_b
    overlap = 1.0 + singles + doubles

    correction = _spin_sum_chol_contract(
        ham_data.chol,
        m1_a + m2_a,
        m1_b + m2_b,
        trial_data.mo_coeff_b,
        base.cfg,
    )
    return (lg * overlap - correction) / overlap


def _ucisd_k_energy_common(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdKMeasCtx,
    trial_data: UcisdKTrial,
) -> UcisdKEnergyCommon:
    """Build the walker-only energy base and reusable K intermediates."""
    base = meas_ctx.base
    green_a, green_b, greenp_a, greenp_b = _greens_restricted(walker, trial_data)
    noa, nob = trial_data.nocc
    green_occ_a = green_a[:, noa:]
    green_occ_b = green_b[:, nob:]
    h1_a = 0.5 * (ham_data.h1 + ham_data.h1.T)
    h1_b = base.h1_b

    hg = jnp.einsum("pj,pj->", h1_a[:noa], green_a, optimize="optimal")
    hg += jnp.einsum("pj,pj->", h1_b[:nob], green_b, optimize="optimal")
    singles = jnp.einsum("pt,pt->", trial_data.c1a, green_occ_a, optimize="optimal")
    singles += jnp.einsum("pt,pt->", trial_data.c1b, green_occ_b, optimize="optimal")
    m1_a = (greenp_a @ trial_data.c1a.T) @ green_a
    m1_b = (greenp_b @ trial_data.c1b.T) @ green_b

    ya, yb = _k_apply_realimag(trial_data, green_occ_a, green_occ_b, base.cfg)
    doubles = 0.5 * jnp.einsum("pt,pt->", green_occ_a, ya, optimize="optimal")
    doubles += 0.5 * jnp.einsum("pt,pt->", green_occ_b, yb, optimize="optimal")
    m2_a = (greenp_a @ ya.T) @ green_a
    m2_b = (greenp_b @ yb.T) @ green_b
    overlap = 1.0 + singles + doubles

    one_body = hg * overlap
    one_body -= jnp.einsum("ij,ij->", h1_a, m1_a + m2_a, optimize="optimal")
    one_body -= jnp.einsum("ij,ij->", h1_b, m1_b + m2_b, optimize="optimal")
    return UcisdKEnergyCommon(
        green_a=green_a,
        green_b=green_b,
        greenp_a=greenp_a,
        greenp_b=greenp_b,
        overlap=overlap,
        m1_a=m1_a,
        m1_b=m1_b,
        m2_a=m2_a,
        m2_b=m2_b,
        z1_a=trial_data.c1a @ green_occ_a.T,
        z1_b=trial_data.c1b @ green_occ_b.T,
        base=ham_data.h0 + one_body / overlap,
    )


def _ucisd_k_chol_terms(
    common: UcisdKEnergyCommon,
    ham_data: HamChol,
    meas_ctx: UcisdKMeasCtx,
    trial_data: UcisdKTrial,
) -> jax.Array:
    """Return one walker's normalized contribution for each Cholesky vector."""
    base = meas_ctx.base
    cfg = base.cfg
    green_a = common.green_a
    green_b = common.green_b
    chol_a = ham_data.chol
    chol_b = base.chol_b
    rot_chol_a = base.rot_chol_a
    rot_chol_b = base.rot_chol_b

    lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b, optimize="optimal")
    lg = lg_a + lg_b
    q_a = jnp.einsum("gpj,qj->gpq", rot_chol_a, green_a, optimize="optimal")
    q_b = jnp.einsum("gpj,qj->gpq", rot_chol_b, green_b, optimize="optimal")
    e20 = 0.5 * lg * lg
    e20 -= 0.5 * jnp.sum(q_a * jnp.swapaxes(q_a, -1, -2), axis=(-1, -2))
    e20 -= 0.5 * jnp.sum(q_b * jnp.swapaxes(q_b, -1, -2), axis=(-1, -2))

    r1 = jnp.einsum("gpq,gqr,rp->g", q_a, q_a, common.z1_a, optimize="optimal")
    r1 += jnp.einsum("gpq,gqr,rp->g", q_b, q_b, common.z1_b, optimize="optimal")
    lci1g_a = jnp.einsum("gip,qi->gpq", base.lci1_a, green_a, optimize="optimal")
    lci1g_b = jnp.einsum("gip,qi->gpq", base.lci1_b, green_b, optimize="optimal")
    r1 -= jnp.einsum("gpq,gqp->g", lci1g_a, q_a, optimize="optimal")
    r1 -= jnp.einsum("gpq,gqp->g", lci1g_b, q_b, optimize="optimal")

    lm12 = _spin_sum_chol_contract(
        chol_a,
        common.m1_a + common.m2_a,
        common.m1_b + common.m2_b,
        trial_data.mo_coeff_b,
        cfg,
    )
    gl_a = _energy_gl_batched_realimag(green_a, chol_a, cfg)
    gl_b = _energy_gl_batched_realimag(green_b, chol_b, cfg)
    rot_m2_a = jnp.einsum("gpi,ji->gpj", rot_chol_a, common.m2_a, optimize="optimal")
    rot_m2_b = jnp.einsum("gpi,ji->gpj", rot_chol_b, common.m2_b, optimize="optimal")
    r2 = jnp.einsum("gpi,gpi->g", gl_a, rot_m2_a, optimize="optimal")
    r2 += jnp.einsum("gpi,gpi->g", gl_b, rot_m2_b, optimize="optimal")

    x_a = jnp.einsum(
        "gpi,it->gpt",
        gl_a,
        common.greenp_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    x_b = jnp.einsum(
        "gpi,it->gpt",
        gl_b,
        common.greenp_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    r3 = _k_quadratic_batched_realimag(trial_data, x_a, x_b, cfg)
    numerator = common.overlap * e20 - lm12 * lg + r1 + r2 + r3
    return numerator / common.overlap


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdKMeasCtx,
    trial_data: UcisdKTrial,
) -> jax.Array:
    """Exact deterministic local energy from the full combined UCISD K."""
    common = _ucisd_k_energy_common(walker, ham_data, meas_ctx, trial_data)
    chol_terms = _ucisd_k_chol_terms(common, ham_data, meas_ctx, trial_data)
    return common.base + jnp.sum(chol_terms, dtype=jnp.complex128)


def make_ucisd_k_meas_ops(
    sys: System,
    *,
    memory_mode: str = "high",
    mixed_precision: bool = True,
) -> MeasOps:
    """Build exact combined-K UCISD measurements for restricted walkers."""
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "UCISD K MeasOps currently supports only restricted walkers, got: "
            f"{sys.walker_kind}"
        )
    if memory_mode != "high":
        raise ValueError("K-native UCISD measurements currently require memory_mode='high'.")
    cfg = UcisdMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype_testing=jnp.complex64 if mixed_precision else jnp.complex128,
    )
    meas_ops = MeasOps(
        overlap=ucisd_k_overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(
            ham_data,
            trial_data,
            cfg=cfg,
        ),
        kernels={k_force_bias: force_bias_kernel_rw_rh, k_energy: energy_kernel_rw_rh},
        observables={},
    )
    object.__setattr__(meas_ops, _UCISD_K_MEAS_CFG_ATTR, cfg)
    return meas_ops
