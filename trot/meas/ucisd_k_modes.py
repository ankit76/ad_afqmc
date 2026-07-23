from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.ucisd import UcisdTrial
from ..trial.ucisd_k_modes import UcisdKModeTrial
from ..trial.ucisd_k_modes import overlap_r as ucisd_k_mode_overlap_r
from .ucisd import UcisdMeasCfg, UcisdMeasCtx
from .ucisd import build_meas_ctx as build_dense_meas_ctx
from .ucisd_k import (
    UcisdKEnergyCommon,
    _combined_pair_batch,
    _force_bias_kernel_rw_rh_with_apply,
    _ucisd_k_chol_terms,
    _ucisd_k_energy_common,
)

_UCISD_K_MODE_MEAS_CFG_ATTR = "_ucisd_k_mode_meas_cfg"


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdKModeMeasCtx:
    """UCISD measurement intermediates and static combined-mode chunking."""

    base: UcisdMeasCtx
    n_mode_chunks: int

    def tree_flatten(self):
        return (self.base,), (self.n_mode_chunks,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (n_mode_chunks,) = aux
        (base,) = children
        return cls(base=base, n_mode_chunks=n_mode_chunks)


def get_ucisd_k_mode_meas_cfg(meas_ops: MeasOps) -> UcisdMeasCfg | None:
    cfg = getattr(meas_ops, _UCISD_K_MODE_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, UcisdMeasCfg) else None


def build_meas_ctx(
    ham_data: HamChol,
    trial_data: UcisdKModeTrial,
    *,
    cfg: UcisdMeasCfg = UcisdMeasCfg(memory_mode="high"),
    n_mode_chunks: int = 1,
) -> UcisdKModeMeasCtx:
    """Build full-Cholesky intermediates for retained combined-K modes."""
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    if cfg.memory_mode != "high":
        raise ValueError("UCISD K-mode measurements currently require memory_mode='high'.")
    if ham_data.basis != "restricted":
        raise ValueError("UCISD K-mode MeasOps requires HamChol.basis == 'restricted'.")

    base = build_dense_meas_ctx(ham_data, cast(UcisdTrial, trial_data), cfg)
    chunks = min(int(n_mode_chunks), trial_data.mode_rank) if trial_data.mode_rank else 1
    return UcisdKModeMeasCtx(base=base, n_mode_chunks=chunks)


def _k_mode_apply_realimag(
    trial_data: UcisdKModeTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
    cfg: UcisdMeasCfg,
) -> tuple[jax.Array, jax.Array]:
    """Apply retained combined modes through explicit real mixed-precision GEMMs."""
    vectors, _ = _combined_pair_batch(
        trial_data,
        matrix_a[None, ...],
        matrix_b[None, ...],
    )
    modes = trial_data.modes.astype(cfg.mixed_real_dtype)
    values = trial_data.eigenvalues.astype(cfg.mixed_real_dtype)
    vector_r = jnp.real(vectors[0]).astype(cfg.mixed_real_dtype)
    vector_i = jnp.imag(vectors[0]).astype(cfg.mixed_real_dtype)
    projection_r = jnp.einsum("rp,p->r", modes, vector_r, optimize="optimal")
    projection_i = jnp.einsum("rp,p->r", modes, vector_i, optimize="optimal")
    applied_r = jnp.einsum("r,rp->p", values * projection_r, modes, optimize="optimal")
    applied_i = jnp.einsum("r,rp->p", values * projection_i, modes, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    applied = applied_r.astype(cfg.mixed_complex_dtype)
    applied += imag_unit * applied_i.astype(cfg.mixed_complex_dtype)

    da, _ = trial_data.pair_dim
    shape_a = (trial_data.nocc[0], trial_data.nvir[0])
    shape_b = (trial_data.nocc[1], trial_data.nvir[1])
    return applied[:da].reshape(shape_a), applied[da:].reshape(shape_b)


def _k_mode_quadratic_batched_realimag(
    trial_data: UcisdKModeTrial,
    matrices_a: jax.Array,
    matrices_b: jax.Array,
    cfg: UcisdMeasCfg,
    n_mode_chunks: int = 1,
) -> jax.Array:
    """Return ``0.5 * z.T @ K_R @ z`` for every leading matrix index."""
    vectors, leading_shape = _combined_pair_batch(trial_data, matrices_a, matrices_b)
    rank = trial_data.mode_rank
    result_dtype = (
        jnp.complex128 if jnp.issubdtype(vectors.dtype, jnp.complexfloating) else jnp.float64
    )
    if rank == 0:
        return jnp.zeros(leading_shape, dtype=result_dtype)

    vectors_r = jnp.real(vectors).astype(cfg.mixed_real_dtype_testing)
    vectors_i = jnp.imag(vectors).astype(cfg.mixed_real_dtype_testing)

    def evaluate_chunk(values_i: jax.Array, modes_i: jax.Array) -> jax.Array:
        modes_i = modes_i.astype(cfg.mixed_real_dtype_testing)
        projection_r = jnp.einsum("sp,rp->sr", vectors_r, modes_i, optimize="optimal")
        if result_dtype == jnp.complex128:
            projection_i = jnp.einsum("sp,rp->sr", vectors_i, modes_i, optimize="optimal")
            projection = projection_r.astype(jnp.complex128)
            projection += 1.0j * projection_i.astype(jnp.complex128)
            return 0.5 * jnp.sum(
                values_i.astype(jnp.float64)[None, :] * projection * projection,
                axis=1,
                dtype=jnp.complex128,
            )
        return 0.5 * jnp.sum(
            values_i.astype(jnp.float64)[None, :]
            * projection_r.astype(jnp.float64)
            * projection_r.astype(jnp.float64),
            axis=1,
            dtype=jnp.float64,
        )

    chunks = min(int(n_mode_chunks), rank)
    if chunks == 1:
        result = evaluate_chunk(trial_data.eigenvalues, trial_data.modes)
        return result.reshape(leading_shape)

    base_chunk_size = rank // chunks
    n_larger_chunks = rank % chunks
    chunk_size = base_chunk_size + int(n_larger_chunks > 0)
    chunk_offsets = jnp.arange(chunk_size, dtype=jnp.int32)
    zero = jnp.zeros((vectors.shape[0],), dtype=result_dtype)

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


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdKModeMeasCtx,
    trial_data: UcisdKModeTrial,
) -> jax.Array:
    """Deterministic combined-K-mode force bias for a restricted walker."""
    return _force_bias_kernel_rw_rh_with_apply(
        walker,
        ham_data,
        meas_ctx,
        trial_data,
        _k_mode_apply_realimag,
    )


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdKModeMeasCtx,
    trial_data: UcisdKModeTrial,
) -> jax.Array:
    """Deterministic local energy from retained combined UCISD K modes."""
    common: UcisdKEnergyCommon = _ucisd_k_energy_common(
        walker,
        ham_data,
        meas_ctx,
        trial_data,
        _k_mode_apply_realimag,
    )
    chol_terms = _ucisd_k_chol_terms(
        common,
        ham_data,
        meas_ctx,
        trial_data,
        _k_mode_quadratic_batched_realimag,
    )
    return common.base + jnp.sum(chol_terms, dtype=jnp.complex128)


def make_ucisd_k_mode_meas_ops(
    sys: System,
    *,
    memory_mode: str = "high",
    mixed_precision: bool = True,
    n_mode_chunks: int = 1,
) -> MeasOps:
    """Build retained combined-K UCISD measurements for restricted walkers."""
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "UCISD K-mode MeasOps currently supports only restricted walkers, got: "
            f"{sys.walker_kind}"
        )
    if memory_mode != "high":
        raise ValueError("UCISD K-mode measurements currently require memory_mode='high'.")
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    cfg = UcisdMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype_testing=jnp.complex64 if mixed_precision else jnp.complex128,
    )
    meas_ops = MeasOps(
        overlap=ucisd_k_mode_overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(
            ham_data,
            trial_data,
            cfg=cfg,
            n_mode_chunks=n_mode_chunks,
        ),
        kernels={k_force_bias: force_bias_kernel_rw_rh, k_energy: energy_kernel_rw_rh},
        observables={},
    )
    object.__setattr__(meas_ops, _UCISD_K_MODE_MEAS_CFG_ATTR, cfg)
    return meas_ops
